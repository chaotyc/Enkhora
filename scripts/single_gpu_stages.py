"""Stage bodies for scripts/single_gpu.py. Each stage runs in its own process:

    python scripts/single_gpu_stages.py <stage> --ctx <ctx.json> --stage-dir <dir>

and writes <dir>/result.json incrementally (so a crash or timeout still leaves the
checks done so far). Helper workers used by the stages:

    python scripts/single_gpu_stages.py worker-host --device cuda:0 --n 4096 --out rates.txt
    python scripts/single_gpu_stages.py worker-hog  --device cuda:0 --bytes N
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

GiB = 1 << 30
MiB = 1 << 20


def say(msg: str = ""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------- results
class Result:
    """Checks (pass/fail/skip/info), metrics and a one-line headline for one stage."""

    def __init__(self, stage_dir: str, stage: str):
        self.path = os.path.join(stage_dir, "result.json")
        self.d = {"stage": stage, "status": "running", "headline": "", "checks": [], "metrics": {}, "notes": []}
        self.save()

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.d, f, indent=1, default=_jsonable)
        os.replace(tmp, self.path)

    def check(self, name: str, ok: bool, detail: str = "", status: str | None = None) -> bool:
        st = status or ("pass" if ok else "fail")
        self.d["checks"].append({"name": name, "status": st, "detail": detail})
        say(f"  [{st.upper():4s}] {name}: {detail}")
        self.save()
        return ok

    def skip(self, name: str, reason: str):
        self.check(name, True, reason, status="skip")

    def info(self, name: str, detail: str):
        self.check(name, True, detail, status="info")

    def metric(self, **kw):
        self.d["metrics"].update(kw)
        self.save()

    def headline(self, s: str):
        self.d["headline"] = s
        self.save()

    def note(self, s: str):
        self.d["notes"].append(s)
        say(f"  note: {s}")
        self.save()

    def finish(self):
        sts = [c["status"] for c in self.d["checks"]]
        if "fail" in sts:
            self.d["status"] = "fail"
        elif sts and all(s == "skip" for s in sts):
            self.d["status"] = "skip"
        else:
            self.d["status"] = "pass"
        self.save()


def _jsonable(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    try:
        return float(x)
    except Exception:
        return str(x)


def run_logged(cmd: list[str], log_path: str, timeout: float | None = None, cwd: str = REPO, echo: bool = True):
    """Run a command, tee its output to a log file (and our stdout). Returns (rc, text)."""
    say(f"  $ {' '.join(cmd)}")
    lines = []
    with open(log_path, "w") as log:
        # Same process group as the stage: the driver's cleanup reaches anything left behind.
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             bufsize=1, env=os.environ.copy())
        timer = threading.Timer(timeout, p.kill) if timeout else None
        if timer:
            timer.start()
        try:
            for line in p.stdout:
                log.write(line)
                log.flush()
                lines.append(line)
                if echo:
                    print("    " + line.rstrip(), flush=True)
            rc = p.wait()
        finally:
            if timer:
                timer.cancel()
    return rc, "".join(lines)


def nvidia_smi(*args: str, timeout: float = 20) -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return None


def compute_apps() -> list[dict] | None:
    out = nvidia_smi("--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits")
    if out is None:
        return None
    apps = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            apps.append({"pid": int(parts[0]), "name": parts[1], "used_mib": parts[2]})
    return apps


def pctl(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))] if xs else None


def tps(n_tokens: int, secs: float) -> float:
    return round(n_tokens / secs, 2) if secs > 0 else float("nan")


def sync(device: str):
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def free_gpu(device: str) -> tuple[int, int]:
    import torch

    if not device.startswith("cuda"):
        return 0, 0
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info(device)


def compare_runs(ref, out, steps: int | None = None) -> dict:
    """Like enkhora.anchor.anchor.compare, optionally on the first ``steps`` steps,
    plus the first step whose logits differ (for debugging a mismatch)."""
    import torch

    (ta, la), (tb, lb) = ref, out
    if steps is not None:
        ta, la = ta[:, :steps], la[:steps]
    eq_steps = [bool(torch.equal(la[i], lb[i])) for i in range(min(len(la), len(lb)))]
    first_diff = next((i for i, e in enumerate(eq_steps) if not e), None)
    return {"tokens_equal": bool(torch.equal(ta, tb)),
            "logits_bitwise": bool(la.shape == lb.shape and torch.equal(la, lb)),
            "max_abs_logit_diff": float((la.float() - lb.float()).abs().max()) if la.shape == lb.shape else None,
            "first_differing_step": first_diff}


def fmt_cmp(r: dict) -> str:
    s = f"tokens_equal={r['tokens_equal']} logits_bitwise={r['logits_bitwise']} max|dlogit|={r['max_abs_logit_diff']}"
    if r.get("first_differing_step") is not None:
        s += f" first differing step={r['first_differing_step']}"
    return s


def reset_dispatcher(d):
    d.stats.clear()
    d.rungs.clear()
    d.layer_ms.clear()
    d.epochs_seen.clear()


def donor_log_has(log_dir: str, donor: str, needle: str) -> bool:
    try:
        with open(os.path.join(log_dir, f"{donor}.log")) as f:
            return needle in f.read()
    except OSError:
        return False


def settled(min_copies: int):
    key = {1: "ge1", 2: "ge2"}.get(min(min_copies, 2), "ge1")

    def pred(s):
        c = s["coverage"]
        return c[key] == c["bundles"] and all(not d["loading"] for d in s["donors"].values())

    return pred


class ChaosHook:
    """Fire events at fixed dispatcher layer-call counts during one generation."""

    def __init__(self, dispatcher, events: dict[int, tuple[str, callable]]):
        self.d, self.events, self.fired = dispatcher, events, []

    def __call__(self, n):
        ev = self.events.get(n)
        if ev and ev[0] not in self.fired:
            self.fired.append(ev[0])
            say(f"    layer call {n:5d}: {ev[0]}")
            ev[1]()

    def __enter__(self):
        reset_dispatcher(self.d)
        self.d.on_layer = self
        return self

    def __exit__(self, *exc):
        self.d.on_layer = None


class Hog:
    """A real separate process holding GPU memory: the 'host job grows' event with NVML."""

    def __init__(self, device: str, nbytes: int, log_path: str):
        self.log = open(log_path, "w")
        self.p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker-hog", "--device", device,
                                   "--bytes", str(int(nbytes))], stdout=subprocess.PIPE, stderr=self.log,
                                  stdin=subprocess.PIPE, text=True)
        line = self.p.stdout.readline()
        self.ok = line.startswith("ready")
        self.msg = line.strip()

    def release(self):
        if self.p.poll() is None:
            self.p.kill()
            self.p.wait(10)
        self.log.close()


# ---------------------------------------------------------------------------- stage 0
def stage_env(ctx, res: Result):
    import importlib.metadata as md

    m = {}
    m["mode"] = "dry-run (CPU)" if ctx.dry_run else "gpu"
    m["python"] = sys.version.split()[0]
    m["platform"] = platform.platform()
    for pkg in ["torch", "transformers", "safetensors", "numpy", "tokenizers", "huggingface-hub", "nvidia-ml-py",
                "pytest"]:
        try:
            m[f"ver_{pkg}"] = md.version(pkg)
        except md.PackageNotFoundError:
            m[f"ver_{pkg}"] = None
    import torch

    m["torch_cuda_build"] = torch.version.cuda
    m["cuda_available"] = bool(torch.cuda.is_available())
    m["cpu_count"] = os.cpu_count()
    m["cpu_model"] = _cpu_model()
    ram = _ram()
    m.update(ram)
    du = shutil.disk_usage(ctx.out_dir)
    m["disk_free_gib_out"] = round(du.free / GiB, 1)
    os.makedirs(ctx.work_dir, exist_ok=True)
    m["disk_free_gib_work"] = round(shutil.disk_usage(ctx.work_dir).free / GiB, 1)
    m["hf_home"] = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    m["hf_token_set"] = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    m["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        m["git_sha"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        m["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,
                                             text=True).stdout.strip())
        m["git_branch"] = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, capture_output=True,
                                         text=True).stdout.strip()
    except Exception:
        m["git_sha"] = None
    res.metric(**m)
    res.info("software", f"python {m['python']}, torch {m['ver_torch']} (CUDA build {m['torch_cuda_build']}), "
                         f"transformers {m['ver_transformers']}, nvidia-ml-py {m['ver_nvidia-ml-py']}")
    res.info("host", f"{m['cpu_count']} CPUs ({m['cpu_model']}), RAM {m.get('ram_total_gib')} GiB total / "
                     f"{m.get('ram_avail_gib')} GiB available, disk free {m['disk_free_gib_work']} GiB (work dir)")
    res.info("git", f"{m.get('git_branch')} @ {m.get('git_sha')}{' (dirty)' if m.get('git_dirty') else ''}")

    smi = nvidia_smi()
    if smi is not None:
        with open(os.path.join(ctx.stage_dir, "nvidia-smi.txt"), "w") as f:
            f.write(smi)
        q = nvidia_smi("--query-gpu=index,name,memory.total,memory.used,driver_version,compute_mode,mig.mode.current,"
                       "power.limit,clocks.max.sm,pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader")
        m["nvidia_smi_query"] = q.strip() if q else None
        res.metric(nvidia_smi_query=m["nvidia_smi_query"])

    if not m["cuda_available"]:
        if ctx.dry_run:
            res.skip("cuda", "no CUDA device; dry-run mode runs every stage on CPU")
            res.headline(f"dry run on {platform.machine()} CPU ({m['cpu_count']} cores); torch {m['ver_torch']}")
        else:
            res.check("cuda", False, "torch.cuda.is_available() is False. Check the NVIDIA driver (nvidia-smi), "
                                     "and that torch is a CUDA build (torch.version.cuda="
                                     f"{m['torch_cuda_build']}). Use --dry-run to test the harness on CPU.")
            res.headline("NO CUDA DEVICE")
        return

    dev = torch.device(ctx.device)
    props = torch.cuda.get_device_properties(dev)
    free, total = torch.cuda.mem_get_info(dev)
    g = {"gpu_name": props.name, "gpu_total_gib": round(total / GiB, 2), "gpu_free_gib": round(free / GiB, 2),
         "gpu_cc": f"{props.major}.{props.minor}", "gpu_sms": props.multi_processor_count,
         "gpu_count": torch.cuda.device_count(), "bf16_supported": torch.cuda.is_bf16_supported(),
         "cudnn": torch.backends.cudnn.version()}
    res.metric(**g)
    res.check("cuda", True, f"{props.name}, {g['gpu_total_gib']} GiB ({g['gpu_free_gib']} GiB free), "
                            f"cc {g['gpu_cc']}, {g['gpu_sms']} SMs, {g['gpu_count']} visible GPU(s)")
    res.check("bf16", g["bf16_supported"], "bf16 kernels available" if g["bf16_supported"] else
              "no bf16 on this GPU: the real-model stage will run in float16/float32 and may not fit")

    # NVML: driver, compute mode, process utilization (the AIMD signal)
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(ctx.nvml_index)
        drv = pynvml.nvmlSystemGetDriverVersion()
        drv = drv.decode() if isinstance(drv, bytes) else drv
        cuda_drv = pynvml.nvmlSystemGetCudaDriverVersion()
        mode = pynvml.nvmlDeviceGetComputeMode(h)
        modes = {0: "Default", 1: "Exclusive_Thread", 2: "Prohibited", 3: "Exclusive_Process"}
        try:
            mig = pynvml.nvmlDeviceGetMigMode(h)[0]
        except pynvml.NVMLError:
            mig = None
        try:
            pynvml.nvmlDeviceGetProcessUtilization(h, 0)
            putil = "supported"
        except pynvml.NVMLError as e:
            putil = "supported (no samples yet)" if "Not Found" in str(e) else f"error: {e}"
        res.metric(nvml_driver=drv, nvml_cuda_driver=f"{cuda_drv // 1000}.{cuda_drv % 1000 // 10}",
                   compute_mode=modes.get(mode, mode), mig_mode=mig, nvml_process_util=putil)
        res.check("nvml", True, f"driver {drv}, CUDA driver API {cuda_drv // 1000}.{cuda_drv % 1000 // 10}, "
                                f"NVML index {ctx.nvml_index}")
        res.check("compute mode", mode == 0,
                  f"{modes.get(mode, mode)}" + ("" if mode == 0 else
                                               ": donors need several CUDA contexts on one GPU; set "
                                               "`nvidia-smi -c DEFAULT` (stages 2-6 will fail otherwise)"))
        if mig:
            res.check("MIG", False, "MIG is enabled: per-process memory and utilization differ; disable MIG")
        res.info("NVML per-process utilization", putil + " (the AIMD governor's host-activity signal)")
        name = pynvml.nvmlDeviceGetName(h)
        res.metric(nvml_name=name.decode() if isinstance(name, bytes) else name)
    except Exception as e:
        res.check("nvml", False, f"NVML unavailable ({e!r}); install nvidia-ml-py. The NVML watchdog, the "
                                 "spike test and the AIMD stage need it")
    apps = compute_apps()
    others = [a for a in (apps or []) if a["pid"] != os.getpid()]
    if others:
        res.check("GPU idle", False, f"other processes are using the GPU: {others}. Results (memory, "
                                     "co-location numbers) will be skewed", status="info")
    else:
        res.check("GPU idle", True, "no other compute processes on the GPU")
    res.headline(f"{props.name} {g['gpu_total_gib']:.0f} GiB · driver {res.d['metrics'].get('nvml_driver', '?')} · "
                 f"torch {m['ver_torch']} (CUDA {m['torch_cuda_build']}) · transformers {m['ver_transformers']}")


def _cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True).stdout.strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "?"


def _ram() -> dict:
    try:
        if sys.platform == "darwin":
            tot = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
            return {"ram_total_gib": round(tot / GiB, 1), "ram_avail_gib": None}
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
        return {"ram_total_gib": round(info["MemTotal"] / GiB, 1),
                "ram_avail_gib": round(info.get("MemAvailable", 0) / GiB, 1)}
    except Exception:
        return {"ram_total_gib": None, "ram_avail_gib": None}


# ---------------------------------------------------------------------------- stage 1
def stage_pytest(ctx, res: Result):
    rc, out = run_logged([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                         os.path.join(ctx.stage_dir, "pytest.log"), timeout=ctx.timeout - 30)
    summary = next((ln.strip(" =") for ln in reversed(out.splitlines()) if re.search(r"\d+ (passed|failed)", ln)), "")
    n_pass = int(m.group(1)) if (m := re.search(r"(\d+) passed", summary)) else 0
    n_fail = int(m.group(1)) if (m := re.search(r"(\d+) failed", summary)) else 0
    res.metric(passed=n_pass, failed=n_fail)
    res.check("pytest", rc == 0, summary or f"exit code {rc}")
    res.headline(summary or f"pytest exit code {rc}")


# ---------------------------------------------------------------------------- stage 2
def stage_tiny(ctx, res: Result):
    import torch

    from enkhora.anchor.dispatcher import DispatchConfig
    from enkhora.cluster import DonorSpec, LocalCluster
    from enkhora.control.planner import PlannerConfig
    from enkhora.model.hf import generate, tiny_qwen3_moe

    gpu = not ctx.dry_run
    dev = ctx.device if gpu else "cpu"
    log_dir = os.path.join(ctx.stage_dir, "donor-logs")
    work = os.path.join(ctx.work_dir, "tiny")
    shutil.rmtree(work, ignore_errors=True)
    hog_headroom_mib = 6144
    if gpu:
        free, _ = free_gpu(dev)
        g3 = max(2048.0, free / MiB - hog_headroom_mib)
        common = ["--device", dev, "--nvml", str(ctx.nvml_index)]
        extra = {"d1": common + ["--guard-mib", "1024", "--slack-mib", "256"],
                 "d2": common + ["--guard-mib", "1024", "--slack-mib", "256"],
                 "d3": common + ["--guard-mib", f"{g3:.0f}", "--slack-mib", "256"]}
        res.note(f"donors on {dev} with the NVML watchdog (index {ctx.nvml_index}); d3's guard is {g3 / 1024:.1f} "
                 f"GiB so a real host allocation can push it under its watermark without touching d1/d2")
    else:
        extra = {d: ["--guard-mib", "0.25", "--slack-mib", "0.25"] for d in ("d1", "d2", "d3")}
    donors = [DonorSpec("d1", "job-idle-a", kappa=1.0, mem_mib=4, life_s=1800, extra=extra["d1"]),
              DonorSpec("d2", "job-quiet-b", kappa=0.5, mem_mib=4, life_s=36000, extra=extra["d2"]),
              DonorSpec("d3", "job-busy-c", kappa=0.0, mem_mib=4, life_s=72000, extra=extra["d3"])]
    planner = PlannerConfig(expert_demand=0.5, alpha=1.0, copy_rate_Bps=1e12, max_replicas=3)
    dispatch = DispatchConfig(deadline_init_s=0.25, deadline_min_s=0.03)
    rng = random.Random(0)
    prompts = [[1] + [rng.randrange(2, 256) for _ in range(rng.randint(3, 12))] for _ in range(6)]
    max_new = 48 if ctx.quick else 96
    n_tok = len(prompts) * max_new

    c = LocalCluster(work, donors, planner=planner, dispatch=dispatch, device=dev, log_dir=log_dir)
    hog = None
    try:
        t0 = time.monotonic()
        c.start(timeout=180)
        st = c.wait(settled(2), timeout=120)
        res.check("placement", True, f"{st['coverage']} in {time.monotonic() - t0:.1f}s; tier "
                                     f"{st['tier_bytes'] / MiB:.2f} MiB; epoch {st['epoch']}")
        a = c.anchor
        if gpu:
            apps = compute_apps()
            if apps:
                mine = {c.procs[d].pid for d in c.procs} | {os.getpid()}
                seen = mine & {x["pid"] for x in apps}
                res.check("separate CUDA contexts", len(apps) >= 4,
                          f"{len(apps)} compute processes on the GPU (anchor + 3 donors expected), {len(seen)} with "
                          "our PIDs: " + ", ".join(f"{x['pid']}:{x['used_mib']}MiB" for x in apps),
                          status=None if len(apps) >= 4 else "info")
            else:
                res.info("separate CUDA contexts", "nvidia-smi lists no compute processes (container PID namespace?)")

        # A. distributed == reference, bitwise
        t = time.monotonic()
        ref = a.generate_local(prompts, max_new)
        sync(dev)
        t_ref = time.monotonic() - t
        a.generate(prompts[:2], 4)                       # warm donor kernels and RTT estimates
        reset_dispatcher(a.dispatcher)
        t = time.monotonic()
        out = a.generate(prompts, max_new)
        t_dist = time.monotonic() - t
        r = compare_runs(ref, out)
        rep = a.dispatcher.report()
        res.check("A: distributed == all-local reference", r["logits_bitwise"] and r["tokens_equal"], fmt_cmp(r))
        served = rep["served_by"]
        on_anchor = sum(v for k, v in served.items() if k.startswith("anchor"))
        share = on_anchor / max(1, sum(served.values()))
        res.check("A: served by donors", served.get("donor", 0) > 0 and share <= 0.05,
                  f"served_by={served} ({100 * share:.1f}% on the anchor; a hedge after a scheduling hiccup is "
                  "allowed, more than 5% is not)")
        res.metric(tiny_ref_tok_s=tps(n_tok, t_ref), tiny_dist_tok_s=tps(n_tok, t_dist), tiny_layer_ms=rep["layer_ms"],
                   tiny_stats_clean=rep["stats"])
        res.info("A: per-layer round trip", f"p50 {rep['layer_ms']['p50']} ms, p95 {rep['layer_ms']['p95']} ms, "
                                            f"p99 {rep['layer_ms']['p99']} ms over {rep['stats'].get('layers')} layer calls "
                                            f"(anchor -> donor process -> anchor over loopback TCP)")
        hf_model = tiny_qwen3_moe(seed=0).to(dev)
        hf = generate(hf_model, prompts, max_new)
        del hf_model
        d = float((hf[1] - out[1]).abs().max())
        res.check("A: matches unpatched HF model", bool(torch.equal(hf[0], out[0])) and d < 1e-3,
                  f"tokens_equal={bool(torch.equal(hf[0], out[0]))}, max|dlogit|={d:.2e} (tolerance 1e-3: HF sums "
                  "experts in a different order)")

        # B. kappa shrink is a routing-table edit
        before = c.status()["counters"]["bytes_issued"]
        c.admin("set_kappa", donor="d1", kappa=0.0)
        st = c.wait(lambda s: not s["donors"]["d1"]["primary_for"] and s["donors"]["d1"]["kappa"] == 0.0)
        moved = st["counters"]["bytes_issued"] - before
        res.check("B: kappa shrink moves no bytes", moved == 0,
                  f"d1 kappa 1 -> 0: primaries d2={len(st['donors']['d2']['primary_for'])}, bytes moved {moved}")
        c.admin("set_kappa", donor="d1", kappa=1.0)
        c.wait(lambda s: s["donors"]["d1"]["primary_for"])

        # C. chaos during one generation
        def host_spike():
            nonlocal hog
            if gpu:
                free_now = torch.cuda.mem_get_info(dev)[0]
                nbytes = free_now - int(float(extra["d3"][extra["d3"].index("--guard-mib") + 1]) * MiB) + GiB
                hog = Hog(dev, nbytes, os.path.join(ctx.stage_dir, "hog.log"))
                say(f"      hog process: {hog.msg}")
            else:
                c.admin("host_alloc", donor="d3", bytes=int(3.5 * MiB))

        spike_label = ("a separate process allocates GPU memory to push free memory below d3's guard (NVML)"
                       if gpu else "simulated host on d3 grows by 3.5 MiB")
        total_calls = len(a.master.layers) * max_new      # the demo's schedule (30/80/120/200/260 of 384)
        at = lambda f: max(1, int(f * total_calls))  # noqa: E731
        events = {at(0.08): ("SIGKILL d1 (primary)", lambda: c.kill("d1")),
                  at(0.21): ("SIGSTOP d2 (donor hangs, connection stays open)", lambda: c.kill("d2", signal.SIGSTOP)),
                  at(0.31): (spike_label, host_spike),
                  at(0.52): ("SIGCONT d2", lambda: c.kill("d2", signal.SIGCONT)),
                  at(0.68): ("control plane dies", c.stop_control)}
        say("  C: chaos during one generation")
        with ChaosHook(a.dispatcher, events) as hook:
            t = time.monotonic()
            out = a.generate(prompts, max_new)
            t_chaos = time.monotonic() - t
        r = compare_runs(ref, out)
        rep = a.dispatcher.report()
        res.check("C: chaos output == reference", r["logits_bitwise"] and len(hook.fired) == len(events),
                  f"{len(hook.fired)}/{len(events)} events fired in {t_chaos:.2f}s; {fmt_cmp(r)}")
        res.metric(tiny_chaos_report=rep)
        res.info("C: dispatcher", f"stats={rep['stats']} served_by={rep['served_by']}")
        time.sleep(0.5)
        evicted = donor_log_has(log_dir, "d3", "watchdog evicted")
        if gpu and hog is not None:
            res.check("C: host spike via real allocation", hog.ok, hog.msg)
        res.check("C: watchdog evicted on d3", evicted,
                  ("NVML probe saw free memory under d3's guard and freed its slabs" if gpu else
                   "simulated probe") + ("" if evicted else "; no 'watchdog evicted' line in d3's log"))
        if hog is not None:
            hog.release()
            hog = None

        # D. every donor gone: the anchor floor
        for did in ("d2", "d3"):
            c.kill(did)
        reset_dispatcher(a.dispatcher)
        t = time.monotonic()
        out = a.generate(prompts, max_new)
        t_floor = time.monotonic() - t
        r = compare_runs(ref, out)
        res.check("D: all donors killed -> anchor floor == reference", r["logits_bitwise"],
                  f"{fmt_cmp(r)}; served_by={a.dispatcher.report()['served_by']}")
        res.metric(tiny_floor_tok_s=tps(n_tok, t_floor))
        res.headline(f"bitwise through clean/chaos/floor runs; per-layer p50 {res.d['metrics']['tiny_layer_ms']['p50']} ms"
                     f" ({'GPU donors, NVML watchdog' if gpu else 'CPU, simulated probe'})")
        if not gpu:
            res.skip("NVML host spike", "needs CUDA + NVML; the simulated probe's host_alloc was used instead")
    finally:
        if hog is not None:
            hog.release()
        c.stop()


# ---------------------------------------------------------------------------- stage 3
PROMPTS = [
    "The history of the printing press begins",
    "Photosynthesis is the process by which plants",
    "Here is a Python function that returns the n-th Fibonacci number:\n\ndef fib(n):",
    "The three most important ideas in thermodynamics are",
    "Once upon a time, in a small village by the sea, there lived",
    "To make a good cup of coffee, you should first",
    "The capital of France is Paris. The capital of Japan is",
    "A mixture-of-experts language model works by",
    "The main causes of the First World War were",
    "Water boils at a lower temperature at high altitude because",
    "In mathematics, a prime number is",
    "The best way to learn a new programming language is",
]

PPL_TEXT = ("The quick brown fox jumps over the lazy dog. This sentence is famous because it contains every letter "
            "of the English alphabet. It has been used for many years to test typewriters and computer keyboards, "
            "and to display examples of fonts. Teachers also use it to help students practice handwriting.")


def tiny_random_hf(kind: str, path: str):
    """Small random-weight OLMoE / Qwen3-MoE in the real HF architecture, saved with save_pretrained
    (v5 writes the legacy per-expert keys, so loading exercises the checkpoint fusion path)."""
    import torch

    if kind == "olmoe":
        from transformers import OlmoeConfig as Cfg
        from transformers import OlmoeForCausalLM as M

        kw = dict(vocab_size=512, hidden_size=64, intermediate_size=32, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=4, num_experts=16, num_experts_per_tok=4, norm_topk_prob=False,
                  max_position_embeddings=512, pad_token_id=1, bos_token_id=None, eos_token_id=2)
    else:
        from transformers import Qwen3MoeConfig as Cfg
        from transformers import Qwen3MoeForCausalLM as M

        kw = dict(vocab_size=512, hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=32, num_experts_per_tok=4,
                  norm_topk_prob=True, max_position_embeddings=512, pad_token_id=None, bos_token_id=1, eos_token_id=2)
    torch.manual_seed(0)
    model = M(Cfg(initializer_range=0.08, tie_word_embeddings=False, **kw)).eval()
    g = torch.Generator().manual_seed(1)
    for layer in model.model.layers:            # decisive routing, as in enkhora.model.hf.tiny_qwen3_moe
        w = layer.mlp.gate.weight
        w.data = torch.randn(w.shape, generator=g) / kw["hidden_size"] ** 0.5 * 4
    model.save_pretrained(path)
    return path


def stage_real(ctx, res: Result):
    if ctx.dry_run:
        res.note("dry run: no download. Two small random-weight models in the real OLMoE and Qwen3-MoE HF "
                 "architectures go through the same load -> export -> patch -> distributed path")
        for kind, bundle in (("olmoe", 4), ("qwen3moe", 8)):
            path = tiny_random_hf(kind, os.path.join(ctx.work_dir, f"tiny-random-{kind}"))
            say(f"\n  === tiny-random-{kind} ===")
            run_real(ctx, res, label=f"tiny-random-{kind}", src=path, tokenizer=None, dtype="float32",
                     bundle_size=bundle, prefix=f"{kind}: ")
        return
    import torch

    m = ctx.model
    big = "30B" in m or "235B" in m
    if big and (ctx.gpu_total_gib or 0) < 79:
        res.check("model fits", False, f"{m} needs an 80 GB GPU (this one has {ctx.gpu_total_gib} GiB). "
                                       "Use the default allenai/OLMoE-1B-7B-0924")
        return
    size_gib = _hub_size_gib(m)
    if size_gib:
        res.metric(model_checkpoint_gib=round(size_gib, 2))
        ram = _ram().get("ram_avail_gib")
        if ram is not None and ram < size_gib * 1.15:
            res.check("host RAM", False, f"loading materializes the whole model in host RAM first: needs "
                                         f"~{size_gib * 1.15:.0f} GiB, {ram} GiB available")
            return
        res.info("host RAM", f"checkpoint {size_gib:.1f} GiB; loading materializes it in host RAM once "
                             f"({ram} GiB available)")
    if not torch.cuda.is_bf16_supported():
        res.note("no bf16 on this GPU; using float16")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    say(f"  downloading / verifying {m} in the HF cache ({f'{size_gib:.1f}' if size_gib else '?'} GiB; can take minutes)")
    t = time.monotonic()
    snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "tokenizer*"])
    res.info("download", f"{m} present in the HF cache after {time.monotonic() - t:.0f}s")
    tok = AutoTokenizer.from_pretrained(m)
    run_real(ctx, res, label=m, src=m, tokenizer=tok,
             dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
             bundle_size=ctx.bundle_size or (16 if big else 8), prefix="")


def _hub_size_gib(model_id: str) -> float | None:
    try:
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(model_id, "model.safetensors.index.json")) as f:
            return json.load(f)["metadata"]["total_size"] / GiB
    except Exception:
        return None


def run_real(ctx, res: Result, label: str, src: str, tokenizer, dtype: str, bundle_size: int, prefix: str):
    import torch
    from transformers import AutoModelForCausalLM

    from enkhora.anchor.dispatcher import DispatchConfig
    from enkhora.anchor.fallback import LocalExecutor
    from enkhora.cluster import DonorSpec, LocalCluster
    from enkhora.control.planner import PlannerConfig
    from enkhora.model.hf import export_master, find_moe_blocks, generate, patch_model, spec_from_model
    from enkhora.store import MasterStore

    gpu = not ctx.dry_run
    dev = ctx.device if gpu else "cpu"
    P = prefix
    slug = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-")
    sdir = ctx.stage_dir
    mtr = {}

    # ---- load
    t = time.monotonic()
    model, info = AutoModelForCausalLM.from_pretrained(src, dtype=getattr(torch, dtype), output_loading_info=True)
    model.eval()
    t_load = time.monotonic() - t
    bad = {k: v for k, v in info.items() if v and k in ("missing_keys", "mismatched_keys")}
    res.check(P + "load", not bad, f"{type(model).__name__} {dtype} in {t_load:.0f}s"
              + (f"; PROBLEM {({k: list(v)[:5] for k, v in bad.items()})}" if bad else "; no missing/mismatched keys")
              + (f"; {len(info.get('unexpected_keys') or [])} unexpected keys" if info.get("unexpected_keys") else ""))
    blocks = find_moe_blocks(model)
    n_layers = model.config.num_hidden_layers
    if not res.check(P + "MoE blocks found", bool(blocks),
                     f"{len(blocks)}/{n_layers} layers have a fused-expert block "
                     + (f"({type(blocks[0][1]).__name__}: router {type(blocks[0][1].gate).__name__}, experts "
                        f"{type(blocks[0][1].experts).__name__})" if blocks else "")):
        return
    spec = spec_from_model(model)
    E = blocks[0][1].experts.gate_up_proj.shape[0]
    gu_shape = tuple(blocks[0][1].experts.gate_up_proj.shape)
    dn_shape = tuple(blocks[0][1].experts.down_proj.shape)
    layout_ok = gu_shape == (E, 2 * spec.inter, spec.hidden) and dn_shape == (E, spec.hidden, spec.inter)
    res.check(P + "expert layout", layout_ok, f"gate_up {gu_shape}, down {dn_shape}; spec hidden={spec.hidden} "
                                              f"inter={spec.inter} act={spec.act} dtype={spec.dtype}; top-k="
                                              f"{model.config.num_experts_per_tok}, norm_topk_prob="
                                              f"{getattr(model.config, 'norm_topk_prob', None)}")
    expert_bytes = sum(b.experts.gate_up_proj.nbytes + b.experts.down_proj.nbytes for _, b in blocks)
    total_bytes = sum(p.nbytes for p in model.parameters())
    trunk_bytes = total_bytes - expert_bytes

    # ---- export (cached across runs in the work dir)
    mdir = os.path.join(ctx.work_dir, f"master-{slug}-{dtype}-b{bundle_size}")
    reuse = False
    if os.path.exists(os.path.join(mdir, "manifest.json")):
        try:
            ms = MasterStore(mdir)
            reuse = ms.model.get("name") == label and ms.spec == spec and len(ms.bundles) == len(blocks) * math.ceil(
                E / bundle_size) and not ctx.dry_run
        except Exception:
            reuse = False
    t = time.monotonic()
    if not reuse:
        shutil.rmtree(mdir, ignore_errors=True)
        export_master(model, mdir, bundle_size, {"name": label, "dtype": dtype})
    master = MasterStore(mdir)
    t_export = time.monotonic() - t
    tier = master.tier_bytes()
    # spot-check: stored bundles are bit-identical to the model's expert slices
    ok = True
    for bid in (min(master.bundles), max(master.bundles)):
        w = master.load(bid)
        blk = dict(blocks)[w.layer]
        idx = torch.tensor(w.experts)
        ok &= torch.equal(w.gate_up, blk.experts.gate_up_proj.data[idx].cpu()) and \
            torch.equal(w.down, blk.experts.down_proj.data[idx].cpu())
    res.check(P + "export master store", ok and tier == expert_bytes,
              f"{len(master.bundles)} bundles of {bundle_size} experts, {tier / GiB:.2f} GiB "
              f"({'reused cached store' if reuse else f'written in {t_export:.0f}s'}); trunk {trunk_bytes / GiB:.2f} GiB; "
              f"spot-checked bundles bit-identical: {ok}")
    mtr.update(tier_gib=round(tier / GiB, 3), trunk_gib=round(trunk_bytes / GiB, 3), bundles=len(master.bundles),
               bundle_size=bundle_size, load_s=round(t_load, 1), export_s=round(t_export, 1))

    # ---- prompts
    batches = ctx.batches or ([8] if ctx.quick else [1, 8, 32])
    big = tier > 30 * GiB
    max_new = ctx.max_new or ((16 if big else 24) if ctx.quick else (32 if big else 64))
    if ctx.dry_run:
        batches = ctx.batches or [1, 4, 8]
        max_new = ctx.max_new or 16
    if tokenizer is not None:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
        base = [tokenizer(p).input_ids for p in PROMPTS]
    else:
        pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
        rng = random.Random(1)
        base = [[rng.randrange(3, model.config.vocab_size) for _ in range(rng.randint(4, 14))] for _ in range(12)]
    prompts_for = {b: [base[i % len(base)] for i in range(b)] for b in batches}
    main_b = 8 if 8 in batches else batches[len(batches) // 2]
    mtr.update(batches=batches, max_new=max_new, main_batch=main_b)

    # ---- unpatched HF model (native baseline + correctness of the patch), if it fits
    hf_out = None
    if gpu:
        free, _ = free_gpu(dev)
        fits = total_bytes + 4 * GiB < free
    else:
        fits = True
    if fits:
        model.to(dev)
        generate(model, prompts_for[main_b][:2], 2, pad_id=pad_id)       # warm-up
        sync(dev)
        t = time.monotonic()
        hf_out = generate(model, prompts_for[main_b], max_new, pad_id=pad_id)
        sync(dev)
        t_hf = time.monotonic() - t
        mtr["hf_native_tok_s"] = tps(main_b * max_new, t_hf)
        res.info(P + "HF native (unpatched) generate", f"batch {main_b}: {mtr['hf_native_tok_s']} tok/s "
                                                       "(HF's own expert loop, whole model on one device)")
    else:
        res.skip(P + "HF native (unpatched) generate", f"full model ({total_bytes / GiB:.1f} GiB) does not fit "
                                                       "next to the run on this GPU")
    patch_model(model)                       # drops the experts; only the trunk stays
    del blocks
    model.to(dev)
    free_gpu(dev)

    # ---- donors: sized from the free memory left once the trunk is resident
    bundle_bytes = max(b.nbytes for b in master.bundles.values())
    if gpu:
        free, _ = free_gpu(dev)
        reserve = 16 * bundle_bytes + 4 * GiB + 2 * GiB     # anchor expert cache + activations/KV + safety
        per_donor = tier * 1.03 + 0.8 * GiB                 # one tier copy + CUDA context
        n_rep = int(max(0, free - reserve) // per_donor)
    else:
        n_rep = 3
    n_donors = max(1, min(3, n_rep))
    mem_mib = (tier * 1.03 / MiB + 16) if n_rep >= 1 else max(64.0, (free - reserve - 0.8 * GiB) / MiB)
    dextra = ["--device", dev] if gpu else []
    all_specs = [DonorSpec("d1", "job-idle-a", kappa=1.0, mem_mib=mem_mib, life_s=1800, extra=dextra),
                 DonorSpec("d2", "job-quiet-b", kappa=0.5, mem_mib=mem_mib, life_s=36000, extra=dextra),
                 DonorSpec("d3", "job-busy-c", kappa=0.0, mem_mib=mem_mib, life_s=72000, extra=dextra)]
    specs = all_specs[:n_donors]
    if n_rep == 0:
        res.note(f"the expert tier ({tier / GiB:.1f} GiB) does not fit once next to the anchor; one donor holds "
                 f"{mem_mib / 1024:.1f} GiB of it and the anchor fallback serves the rest (slow, still exact)")
    res.info(P + "donor plan", f"{n_donors} donor process(es) on {dev}, each with a {mem_mib / 1024:.2f} GiB budget "
                               f"(explicit budgets: several donors share one GPU, so NVML free memory cannot be "
                               f"each donor's headroom); roles " + ", ".join(f"{s.id}(kappa {s.kappa})" for s in specs))
    planner = PlannerConfig(expert_demand=0.5, alpha=1.0, copy_rate_Bps=1e12, max_replicas=max(1, min(3, n_rep)))
    dispatch = DispatchConfig()
    c = LocalCluster(os.path.join(ctx.work_dir, f"cluster-{slug}"), specs, planner=planner, dispatch=dispatch,
                     device=dev, log_dir=os.path.join(sdir, f"donor-logs-{slug}"),
                     model_loader=lambda: (model, master))
    gens = {}
    try:
        c.start(spawn_donors=False)
        a = c.anchor
        ram = _ram().get("ram_avail_gib")
        if gpu and ram:                       # cap the anchor's pinned-RAM fallback cache
            a.fallback.ram_bundles = max(16, int(min(32.0, 0.4 * ram) * GiB // bundle_bytes))

        # ---- reference: every bundle on the anchor, same kernel contract
        if gpu:
            free, _ = free_gpu(dev)
            n_fit = int(max(0, free - 6 * GiB) // bundle_bytes)
        else:
            n_fit = len(master.bundles)
        all_fit = n_fit >= len(master.bundles)
        ref_ex = LocalExecutor(master, dev, dispatch.chunk_rows, cache_bundles=len(master.bundles) if all_fit else max(16, n_fit))
        if all_fit:
            ref_ex.store.ram_bundles = 0      # straight to the device, no host RAM copy
            t = time.monotonic()
            for b in sorted(master.bundles):
                ref_ex.store.get(b)
            sync(dev)
            res.info(P + "reference executor", f"all {len(master.bundles)} bundles resident on {dev} "
                                               f"(loaded in {time.monotonic() - t:.1f}s)")
        else:
            res.info(P + "reference executor", f"only {n_fit}/{len(master.bundles)} bundles fit on the device: "
                                               "the reference streams the rest from host RAM (slow)")
        refs, ref_tps = {}, {}
        for b in batches:
            t = time.monotonic()
            refs[b] = a.generate_local(prompts_for[b], max_new, pad_id=pad_id, executor=ref_ex)
            sync(dev)
            ref_tps[b] = tps(b * max_new, time.monotonic() - t)
            say(f"    reference batch {b}: {ref_tps[b]} tok/s")
        mtr["local_tok_s"] = ref_tps

        # perplexity of a fixed English paragraph through our expert path (sanity of the weights)
        if tokenizer is not None:
            from enkhora.model.hf import set_executor

            ids = torch.tensor([tokenizer(PPL_TEXT).input_ids], device=dev)
            set_executor(a.model, ref_ex)
            with torch.no_grad():
                loss = float(a.model(input_ids=ids, labels=ids).loss)
            set_executor(a.model, a.dispatcher)
            ppl = math.exp(loss)
            mtr["ppl"] = round(ppl, 2)
            res.check(P + "perplexity sanity", ppl < 60, f"ppl {ppl:.2f} on a fixed English paragraph through the "
                                                         "patched model (random weights would give ~vocab size)")
        else:
            res.skip(P + "perplexity sanity", "random weights, no tokenizer")
        del ref_ex
        free_gpu(dev)

        if hf_out is not None:
            ours = refs[main_b]
            tok_agree = float((hf_out[0] == ours[0]).float().mean())
            step0 = bool(torch.equal(hf_out[1][0].float().argmax(-1), ours[1][0].float().argmax(-1)))
            d0 = float((hf_out[1][0].float() - ours[1][0].float()).abs().max())
            mtr.update(hf_token_agreement=round(tok_agree, 4), hf_step0_max_abs_diff=d0)
            tol_ok = (tok_agree == 1.0 and d0 < 1e-3) if dtype == "float32" else (step0 and tok_agree >= 0.5)
            res.check(P + "patched model == HF model", tol_ok,
                      f"step-0 argmax equal {step0}, step-0 max|dlogit| {d0:.3g}, generated-token agreement "
                      f"{100 * tok_agree:.1f}% (not bitwise: HF sums experts in another order"
                      + (", and bf16 greedy decodes drift after a near-tie)" if dtype != "float32" else ")"))

        # ---- donors
        t = time.monotonic()
        c.start_donors(wait_replicas=1, timeout=ctx.timeout * 0.4)
        want = min(2, n_donors) if n_rep >= 1 else 1
        st = c.wait(settled(want) if n_rep >= 1 else (lambda s: all(not d["loading"] for d in s["donors"].values())),
                    timeout=ctx.timeout * 0.3, poll=0.5)
        t_place = time.monotonic() - t
        loaded = sum(len(d["ready"]) for d in st["donors"].values())
        res.check(P + "placement", True, f"{st['coverage']} after {t_place:.0f}s; {loaded} bundle replicas "
                                         f"({loaded * bundle_bytes / GiB:.1f} GiB) loaded over TCP from the master store; "
                  + ", ".join(f"{k}: {len(v['ready'])} ready / {len(v['primary_for'])} primary"
                              for k, v in st["donors"].items()))
        mtr.update(placement_s=round(t_place, 1), replicas_loaded=loaded, n_donors=n_donors)

        # ---- distributed decode, batch sweep
        a.generate(prompts_for[main_b][:2], 4, pad_id=pad_id)       # warm donor kernels and RTT estimates
        dist_tps, reports, all_ok = {}, {}, True
        for b in batches:
            reset_dispatcher(a.dispatcher)
            t = time.monotonic()
            out = a.generate(prompts_for[b], max_new, pad_id=pad_id)
            sync(dev)
            dist_tps[b] = tps(b * max_new, time.monotonic() - t)
            r = compare_runs(refs[b], out)
            rep = a.dispatcher.report()
            reports[b] = {"compare": r, **rep}
            all_ok &= r["logits_bitwise"] and r["tokens_equal"]
            res.check(P + f"batch {b}: distributed == reference", r["logits_bitwise"] and r["tokens_equal"],
                      f"{fmt_cmp(r)}; {dist_tps[b]} tok/s distributed vs {ref_tps[b]} local; layer ms "
                      f"p50/p95/p99 {rep['layer_ms']['p50']}/{rep['layer_ms']['p95']}/{rep['layer_ms']['p99']}; "
                      f"served_by {rep['served_by']}")
            if b == main_b:
                gens["distributed"] = out[0]
        mtr.update(dist_tok_s=dist_tps, dist_reports=reports)
        gens["local"] = refs[main_b][0]
        if hf_out is not None:
            gens["hf_native"] = hf_out[0]
        _write_generations(os.path.join(sdir, f"generations-{slug}.md"), label, tokenizer,
                           prompts_for[main_b], gens, max_new)

        # ---- chaos with the real model
        ref = refs[main_b]
        total_calls = len(master.layers) * max_new
        at = lambda f: max(1, int(f * total_calls))  # noqa: E731
        if n_donors >= 2:
            events = {at(0.15): ("SIGKILL d1 (primary)", lambda: c.kill("d1")),
                      at(0.35): ("SIGSTOP d2 (hung donor)", lambda: c.kill("d2", signal.SIGSTOP)),
                      at(0.60): ("SIGCONT d2", lambda: c.kill("d2", signal.SIGCONT)),
                      at(0.80): ("control plane dies", c.stop_control)}
        else:
            events = {at(0.25): ("SIGSTOP d1 (hung donor)", lambda: c.kill("d1", signal.SIGSTOP)),
                      at(0.40): ("SIGCONT d1", lambda: c.kill("d1", signal.SIGCONT)),
                      at(0.70): ("control plane dies", c.stop_control)}
        say(f"  {P}chaos during one generation (batch {main_b}, {total_calls} layer calls)")
        with ChaosHook(a.dispatcher, events) as hook:
            t = time.monotonic()
            out = a.generate(prompts_for[main_b], max_new, pad_id=pad_id)
            t_chaos = time.monotonic() - t
        r = compare_runs(ref, out)
        rep = a.dispatcher.report()
        res.check(P + "chaos generation == reference", r["logits_bitwise"] and len(hook.fired) == len(events),
                  f"{len(hook.fired)}/{len(events)} events in {t_chaos:.1f}s; {fmt_cmp(r)}; stats {rep['stats']}; "
                  f"served_by {rep['served_by']}")
        mtr.update(chaos_s=round(t_chaos, 1), chaos_report=rep)

        # ---- every donor gone: the anchor floor (bounded: it streams the tier over PCIe each step)
        if tier > 30 * GiB:
            res.skip(P + "anchor floor", f"tier {tier / GiB:.0f} GiB: the floor streams it from host RAM every step")
        else:
            for s in specs:
                if c.procs[s.id].poll() is None:
                    c.kill(s.id)
            n_floor = min(max_new, 8 if ctx.quick else 16)
            reset_dispatcher(a.dispatcher)
            t = time.monotonic()
            out = a.generate(prompts_for[main_b], n_floor, pad_id=pad_id)
            t_floor = time.monotonic() - t
            r = compare_runs(ref, out, steps=n_floor)
            mtr["floor_tok_s"] = tps(main_b * n_floor, t_floor)
            res.check(P + "all donors killed -> anchor floor", r["logits_bitwise"] and r["tokens_equal"],
                      f"first {n_floor} steps vs reference: {fmt_cmp(r)}; {mtr['floor_tok_s']} tok/s; "
                      f"served_by {a.dispatcher.report()['served_by']}")
    finally:
        c.stop()
    res.metric(**{f"{slug}": mtr})
    head = (f"{label}: bitwise {'OK' if all_ok else 'MISMATCH'}; batch {main_b} {dist_tps.get(main_b)} tok/s "
            f"distributed vs {ref_tps.get(main_b)} local; layer p50 {reports[main_b]['layer_ms']['p50']} ms")
    res.headline((res.d["headline"] + " | " if res.d["headline"] else "") + head)
    if not ctx.keep_work and not ctx.dry_run:
        shutil.rmtree(mdir, ignore_errors=True)
        res.note(f"deleted the {tier / GiB:.1f} GiB master store (pass --keep-work to keep and reuse it)")


def _write_generations(path, label, tok, prompts, gens: dict, max_new):
    with open(path, "w") as f:
        f.write(f"# Generations: {label}\n\nGreedy, {max_new} new tokens. `local` = all experts on the anchor; "
                "`distributed` = experts on donor processes (bitwise equal to local when the check passed); "
                "`hf_native` = the unpatched HF model.\n\n")
        for i, p in enumerate(prompts):
            ptxt = tok.decode(p, skip_special_tokens=True) if tok else str(p)
            f.write(f"## Prompt {i}\n\n```\n{ptxt}\n```\n\n")
            for k, toks in gens.items():
                txt = tok.decode(toks[i], skip_special_tokens=True) if tok else str(toks[i].tolist())
                f.write(f"**{k}**\n\n```\n{txt}\n```\n\n")


# ---------------------------------------------------------------------------- stage 4
def stage_colocate(ctx, res: Result):
    csv_path = os.path.join(ctx.stage_dir, "colocate.csv")
    cmd = [sys.executable, os.path.join(REPO, "bench", "colocate.py"), "--out", csv_path]
    if ctx.dry_run:
        cmd += ["--device", "cpu", "--allow-cpu", "--hidden", "256", "--inter", "128", "--experts", "8",
                "--tokens", "16", "--host-matmul", "256", "--kappas", "0.05,0.25", "--windows", "2", "--window-s", "0.4"]
        res.note("dry run: CPU contention is not GPU time-slicing; these numbers mean nothing")
    else:
        cmd += ["--device", ctx.device]
        if ctx.quick:
            cmd += ["--kappas", "0.01,0.02,0.05,0.1", "--windows", "3", "--window-s", "1.5"]
        else:
            cmd += ["--kappas", "0.005,0.01,0.02,0.05,0.1,0.25", "--windows", "6", "--window-s", "2.0"]
    if ctx.host_cmd:
        cmd += ["--host-cmd", ctx.host_cmd]
        res.note(f"host code: {ctx.host_cmd}")
    else:
        res.note("host: the synthetic back-to-back fp32 matmul loop (the most contention-sensitive kind of host); "
                 "pass --host-cmd for real host codes")
    rc, out = run_logged(cmd, os.path.join(ctx.stage_dir, "colocate.log"), timeout=ctx.timeout - 30)
    rows = []
    for line in out.splitlines():
        mm = re.match(r"kappa\s+([\d.]+)\s+slowdown\s+(-?[\d.]+)% ± ([\d.]+)\s+duty\s+([\d.]+)%\s+call p50 ([\d.]+) ms"
                      r"\s+p99 ([\d.]+) ms", line.strip())
        if mm:
            rows.append({"kappa": float(mm.group(1)), "slowdown_pct": float(mm.group(2)),
                         "sd_pct": float(mm.group(3)), "duty_pct": float(mm.group(4)),
                         "call_ms_p50": float(mm.group(5)), "call_ms_p99": float(mm.group(6))})
    idle = re.search(r"idle second context: slowdown (-?[\d.]+)%", out)
    first = re.search(r"first kappa over the 5% cap: (\S+(?: in grid)?)", out)
    kill = "KILL CRITERION HIT" in out
    res.metric(rows=rows, idle_ctx_slowdown_pct=float(idle.group(1)) if idle else None,
               first_kappa_over_5pct=first.group(1) if first else None, kill_criterion_hit=kill)
    if not res.check("harness ran", rc == 0 and bool(rows), f"exit code {rc}, {len(rows)} kappa points; CSV {csv_path}"):
        return
    for r in rows:
        res.info(f"kappa {r['kappa']}", f"host slowdown {r['slowdown_pct']:.2f}% ± {r['sd_pct']:.2f}, our duty "
                                        f"{r['duty_pct']:.2f}%, call p50 {r['call_ms_p50']} ms p99 {r['call_ms_p99']} ms")
    fk = first.group(1) if first else "?"
    verdict = ("KILL CRITERION HIT: 5% slowdown below kappa 2% -> busy hosts become store-only" if kill else
               "kill criterion not hit (no 5% slowdown at or below kappa 2%)")
    res.check("kill criterion (design section 11, exp. 3)", True, verdict, status="info")
    res.headline(("[CPU smoke, meaningless] " if ctx.dry_run else "") +
                 f"first kappa over 5%: {fk}; idle 2nd context: {idle.group(1) if idle else '?'}%; {verdict}")


# ---------------------------------------------------------------------------- stage 5
def stage_spike(ctx, res: Result):
    cmd = [sys.executable, os.path.join(REPO, "bench", "spike.py")]
    if ctx.dry_run:
        trials = ctx.spike_trials or (200 if ctx.quick else 1000)
        cmd += ["--simulate", "--trials", str(trials)]
        res.note("dry run: simulated probe (no NVML, no real allocations)")
    else:
        tot = ctx.gpu_total_gib or 80.0
        s = tot / 80.0
        trials = ctx.spike_trials or (10 if ctx.quick else 50)
        p = {"gpu-gib": tot, "guard-gib": 12 * s, "slack-gib": 2 * s, "slab-gib": 1.41 * s, "max-step-gib": 16 * s,
             "host-base-gib": 10 * s}
        cmd += ["--device", ctx.device, "--nvml", str(ctx.nvml_index), "--trials", str(trials), "--poll-ms", "20",
                "--mean-gap-ms", "50"] + [x for k, v in p.items() for x in (f"--{k}", f"{v:.2f}")]
        res.note("sizes scaled to this GPU from the 80 GB defaults: " + ", ".join(f"{k} {v:.2f}" for k, v in p.items())
                 + "; poll 20 ms; host steps every ~50 ms in a separate process")
        res.metric(params=p)
    res.metric(trials=trials)
    rc, out = run_logged(cmd, os.path.join(ctx.stage_dir, "spike.log"), timeout=ctx.timeout - 30)
    mm = re.search(r"trials (\d+): (\w+) <= guard: (\d+) failures / (\d+)\s+\w+ > guard: (\d+) failures / (\d+)", out)
    ff = re.search(r"time to first free \(ms\): p50 ([\d.]+)\s+p99 ([\d.]+)", out)
    rx = re.search(r"watchdog reaction \(ms[^)]*\): p50 ([\d.]+)\s+p99 ([\d.]+)\s+max ([\d.]+)", out)
    if rx:
        res.metric(reaction_ms_p50=float(rx.group(1)), reaction_ms_p99=float(rx.group(2)),
                   reaction_ms_max=float(rx.group(3)))
    verdict = "PASS" if re.search(r"^PASS$", out, re.M) else ("FAIL" if "FAIL" in out else None)
    if mm:
        res.metric(inside_fail=int(mm.group(3)), inside_n=int(mm.group(4)), outside_fail=int(mm.group(5)),
                   outside_n=int(mm.group(6)))
    if ff:
        res.metric(first_free_ms_p50=float(ff.group(1)), first_free_ms_p99=float(ff.group(2)))
    if rc != 0 or verdict is None or not mm:
        res.check("spike test ran", False, f"exit code {rc}; see spike.log")
        return
    res.check("spike test", verdict == "PASS",
              f"{verdict}: {mm.group(3)}/{mm.group(4)} host failures with the {mm.group(2)} inside the guard; "
              f"{mm.group(5)}/{mm.group(6)} outside (residual risk without ring 3)")
    if rx:
        res.info("watchdog reaction", f"host allocation -> first slab freed: p50 {rx.group(1)} ms, p99 {rx.group(2)} ms, "
                                      f"max {rx.group(3)} ms (poll interval 20 ms)")
    if ff:
        res.info("time to first free", f"p50 {ff.group(1)} ms, p99 {ff.group(2)} ms, measured from the host process "
                                       "launch (includes its interpreter + CUDA start-up)")
    elif not ctx.dry_run:
        res.info("time to first free", "the watchdog never had to free (no step reached the guard)")
    res.headline(("[simulated] " if ctx.dry_run else "") + f"{verdict}: inside-guard failures {mm.group(3)}/{mm.group(4)}, "
                 f"outside {mm.group(5)}/{mm.group(6)}" + (f"; watchdog reaction p50 {rx.group(1)} ms" if rx else ""))


# ---------------------------------------------------------------------------- stage 6
def stage_aimd(ctx, res: Result):
    import torch

    from enkhora.anchor.dispatcher import DispatchConfig
    from enkhora.cluster import DonorSpec, LocalCluster
    from enkhora.control.planner import PlannerConfig
    from enkhora.experts import build_route
    from enkhora.model.hf import export_master, patch_model, tiny_qwen3_moe
    from enkhora.transport import LoopThread, RpcClient

    gpu = not ctx.dry_run
    dev = ctx.device if gpu else "cpu"
    secs = ctx.aimd_seconds or ((45 if ctx.quick else 180) if gpu else (20 if ctx.quick else 40))
    res.note("exploratory: the AIMD governor (on/off windows, NVML per-process SM utilization as the host signal) "
             "has never run against real counters")
    if gpu:
        kw = dict(vocab_size=256, hidden_size=2048, intermediate_size=4096, moe_intermediate_size=1408,
                  num_hidden_layers=2, num_attention_heads=16, num_key_value_heads=4, head_dim=128, num_experts=64,
                  num_experts_per_tok=8)
        dtype, T = torch.bfloat16, 256
    else:
        kw = dict(vocab_size=256, hidden_size=128, intermediate_size=256, moe_intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_experts=16,
                  num_experts_per_tok=4)
        dtype, T = torch.float32, 64
    model = tiny_qwen3_moe(seed=0, **kw).to(dtype)
    mdir = os.path.join(ctx.work_dir, "aimd-master")
    shutil.rmtree(mdir, ignore_errors=True)
    master = export_master(model, mdir, 16, {"name": "aimd-random"})
    patch_model(model)
    res.info("expert tier", f"{len(master.bundles)} bundles, {master.tier_bytes() / GiB:.2f} GiB, "
                            f"D={kw['hidden_size']} I={kw['moe_intermediate_size']} E={kw['num_experts']} {dtype}")

    host = None
    rates_path = os.path.join(ctx.stage_dir, "host_rates.txt")
    if gpu:
        spec = DonorSpec("d1", "job-host", kappa=1.0, extra=["--device", dev, "--nvml", str(ctx.nvml_index), "--aimd",
                                                             "--guard-mib", "2048", "--slack-mib", "512"])
        host_log = open(os.path.join(ctx.stage_dir, "host.log"), "w")
        host = subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker-host", "--device", dev, "--n",
                                 "4096", "--out", rates_path], stdout=host_log, stderr=subprocess.STDOUT)
        res.note("host: a separate process running back-to-back 4096^2 fp32 matmuls on the same GPU")
    else:
        spec = DonorSpec("d1", "job-host", kappa=1.0, mem_mib=64, aimd=True, sensitivity=2.0)
        res.note("dry run: simulated probe; host work rate = 1 - 2.0 x our duty, so AIMD should settle near "
                 "kappa 0.025 (5% tolerance / sensitivity 2)")
    c = LocalCluster(os.path.join(ctx.work_dir, "aimd-cluster"), [spec],
                     planner=PlannerConfig(expert_demand=0.5, alpha=1.0, copy_rate_Bps=1e12, max_replicas=1),
                     dispatch=DispatchConfig(), device="cpu", log_dir=os.path.join(ctx.stage_dir, "donor-logs"),
                     model_loader=lambda: (model, master))
    lt = None
    stop = threading.Event()
    counts = {"ok": 0, "refused": 0, "err": 0}
    try:
        c.start(wait_replicas=0, timeout=120)
        st = c.wait(lambda s: "d1" in s["donors"] and len(s["donors"]["d1"]["ready"]) == len(master.bundles)
                    and not s["donors"]["d1"]["loading"], timeout=300, poll=0.5)
        res.check("donor loaded", True, f"d1 holds {len(st['donors']['d1']['ready'])} bundles")
        if gpu:
            apps = compute_apps() or []
            if apps and c.procs["d1"].pid not in {x["pid"] for x in apps}:
                res.check("NVML sees our PIDs", False, "NVML process PIDs differ from this container's PIDs: "
                          "NvmlProbe cannot exclude the donor's own utilization, so the slowdown signal is "
                          "contaminated. Run with the host PID namespace (e.g. docker --pid=host)", status="info")
            elif apps:
                res.check("NVML sees our PIDs", True, f"donor pid {c.procs['d1'].pid} visible to NVML")
        addr = c.anchor.dispatcher.table.donors.get("d1")
        t_wait = time.monotonic()
        while addr is None and time.monotonic() - t_wait < 30:
            time.sleep(0.2)
            addr = c.anchor.dispatcher.table.donors.get("d1")
        lt = LoopThread("loadgen")
        client = lt.call(RpcClient(*addr).connect(5.0))
        reqs = [_exec_request(master, layer, T, kw["num_experts_per_tok"], dtype, seed=layer, build_route=build_route)
                for layer in master.layers]

        def loadgen():
            i = 0
            while not stop.is_set():
                h, tt = reqs[i % len(reqs)]
                i += 1
                try:
                    rh, _ = lt.call(client.request(h, tt), timeout=30)
                    if rh.get("ok"):
                        counts["ok"] += 1
                    else:
                        counts["refused"] += 1
                        time.sleep(0.01)
                except Exception:
                    counts["err"] += 1
                    time.sleep(0.05)

        def sample(t0):
            s = c.status()["donors"].get("d1", {})
            return {"t": round(time.monotonic() - t0, 2), "kappa": s.get("kappa"),
                    "exec_ms": (s.get("stats") or {}).get("exec_ms", 0.0), "ok": counts["ok"],
                    "refused": counts["refused"]}

        baseline_s = 10 if ctx.quick else 20
        t0 = time.monotonic()
        say(f"  idle-donor baseline for {baseline_s}s (governor running, no expert calls offered)")
        trace = []
        while time.monotonic() - t0 < baseline_s:
            time.sleep(1.0)
            trace.append({**sample(t0), "phase": "baseline"})
        t_load = time.monotonic()
        th = threading.Thread(target=loadgen, daemon=True)
        th.start()
        say(f"  offering continuous expert calls for {secs}s; kappa every 1s:")
        while time.monotonic() - t_load < secs:
            time.sleep(1.0)
            trace.append({**sample(t0), "phase": "load"})
            if len(trace) % 10 == 0:
                say(f"    t={trace[-1]['t']:6.1f}s kappa={trace[-1]['kappa']} calls ok={counts['ok']} "
                    f"refused={counts['refused']}")
        stop.set()
        th.join(10)
        t_end = time.monotonic()
    finally:
        stop.set()
        if lt is not None:
            lt.stop()
        c.stop()
        if host is not None:
            host.terminate()
            host.wait(10)

    # host rates: lines "t_monotonic rate"
    rates = []
    if gpu and os.path.exists(rates_path):
        with open(rates_path) as f:
            for line in f:
                try:
                    tt, r = line.split()
                    rates.append((float(tt), float(r)))
                except ValueError:
                    pass
    for row in trace:                         # host rate over the second before each sample
        ts = t0 + row["t"]
        xs = [r for tt, r in rates if ts - 1.0 <= tt < ts]
        row["host_rate"] = round(statistics.mean(xs), 2) if xs else None
    load = [r for r in trace if r["phase"] == "load"]
    for prev, row in zip(trace, trace[1:]):
        row["duty"] = round((row["exec_ms"] - prev["exec_ms"]) / ((row["t"] - prev["t"]) * 1e3), 4)
    with open(os.path.join(ctx.stage_dir, "kappa_trace.csv"), "w") as f:
        keys = ["t", "phase", "kappa", "duty", "host_rate", "ok", "refused", "exec_ms"]
        f.write(",".join(keys) + "\n")
        for row in trace:
            f.write(",".join("" if row.get(k) is None else str(row.get(k)) for k in keys) + "\n")
    ks = [r["kappa"] for r in load if r["kappa"] is not None]
    changes = sum(1 for a_, b_ in zip(ks, ks[1:]) if a_ != b_)
    cuts = sum(1 for a_, b_ in zip(ks, ks[1:]) if b_ < a_)
    third = load[len(load) * 2 // 3:]
    k_last = [r["kappa"] for r in third if r["kappa"] is not None]
    duty_last = [r["duty"] for r in third if r.get("duty") is not None]
    base_rates = [r for tt, r in rates if t0 <= tt < t_load]
    last_rates = [r for tt, r in rates if t0 + third[0]["t"] - 1.0 <= tt < t_end] if third else []
    m = {"seconds": secs, "kappa_updates": changes, "kappa_cuts": cuts, "kappa_max": max(ks) if ks else None,
         "kappa_last_third_mean": round(statistics.mean(k_last), 4) if k_last else None,
         "duty_last_third_mean": round(statistics.mean(duty_last), 4) if duty_last else None,
         "calls": dict(counts)}
    if base_rates and last_rates:
        m["host_rate_baseline"] = round(statistics.mean(base_rates), 2)
        m["host_rate_last_third"] = round(statistics.mean(last_rates), 2)
        m["host_slowdown_last_third_pct"] = round(100 * (1 - m["host_rate_last_third"] / m["host_rate_baseline"]), 2)
    res.metric(**m)
    res.check("governor ran", changes >= 3 and counts["ok"] > 0,
              f"{changes} kappa updates ({cuts} multiplicative cuts) over {secs}s; {counts['ok']} calls served, "
              f"{counts['refused']} refused, {counts['err']} errors; trace kappa_trace.csv")
    if cuts >= 2:
        conv = f"sawtooth: cut {cuts} times; last-third kappa mean {m['kappa_last_third_mean']}"
    elif ks and max(ks) >= 0.999:
        conv = "no cut: kappa climbed to 1.0 (the signal never showed >5% slowdown)"
    else:
        conv = f"not converged within {secs}s: kappa max {m['kappa_max']}, {cuts} cut(s)"
    res.info("convergence", conv)
    if "host_slowdown_last_third_pct" in m:
        res.info("realized host slowdown", f"{m['host_slowdown_last_third_pct']}% in the last third vs the idle-donor "
                                           f"baseline (target <= 5%); our duty {m['duty_last_third_mean']}")
    elif gpu:
        res.info("realized host slowdown", "no host rate samples")
    res.headline(f"[exploratory{', simulated probe' if not gpu else ''}] {conv}"
                 + (f"; host slowdown {m['host_slowdown_last_third_pct']}%" if "host_slowdown_last_third_pct" in m else ""))


def _exec_request(master, layer, T, k, dtype, seed, build_route):
    """One exec message for every bundle of ``layer``, packed exactly as Dispatcher._send does."""
    import torch

    g = torch.Generator().manual_seed(seed)
    E, D = master.num_experts, master.spec.hidden
    topk = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])
    w = torch.rand(T, k, generator=g).to(dtype)
    hidden = torch.randn(T, D, generator=g).to(dtype)
    route = build_route(topk, w, master.expert_to_bundle(layer))
    bundles = route.bundles
    U = torch.unique(torch.cat([route.urows[b] for b in bundles]), sorted=True)
    rows, exps, gates, offs = [], [], [], [0]
    for b in bundles:
        r, e, gg = route.pairs[b]
        rows.append(torch.searchsorted(U, r).to(torch.int32))
        exps.append(e.to(torch.int32))
        gates.append(gg)
        offs.append(offs[-1] + len(r))
    header = {"op": "exec", "epoch": 0, "layer": layer, "bundles": bundles, "act": master.spec.act,
              "chunk_rows": 4096}
    tensors = {"hidden": hidden.index_select(0, U), "rows": torch.cat(rows), "experts": torch.cat(exps),
               "gates": torch.cat(gates), "offsets": torch.tensor(offs, dtype=torch.int64)}
    return header, tensors


# ---------------------------------------------------------------------------- stage 7
def stage_sim(ctx, res: Result):
    out_json = os.path.join(ctx.stage_dir, "sim.json")
    days = "7" if ctx.quick else "14"
    cmd = [sys.executable, "-m", "enkhora", "sim", "--days", days, "--out", out_json]
    rc, out = run_logged(cmd, os.path.join(ctx.stage_dir, "sim.log"), timeout=ctx.timeout - 30)
    if rc != 0 and "unrecognized arguments" in out:
        rc, out = run_logged(cmd[:-2], os.path.join(ctx.stage_dir, "sim.log"), timeout=ctx.timeout - 30)
    last = [ln.strip() for ln in out.splitlines() if ln.strip()]
    verdict = last[-1] if last else ""
    res.check("simulator", rc == 0, f"--days {days}: {verdict}" if rc == 0 else f"exit code {rc}")
    res.headline(f"{days}-day synthetic trace: {verdict}")


# ---------------------------------------------------------------------------- workers
def worker_host(a):
    """Back-to-back matmuls on the GPU; appends '<monotonic> <iters/s>' every 0.25 s."""
    import torch

    torch.cuda.set_device(a.device)
    torch.manual_seed(0)
    x = torch.randn(a.n, a.n, device=a.device)
    y = torch.randn(a.n, a.n, device=a.device)
    it, last = 0, time.monotonic()
    with open(a.out, "a", buffering=1) as f:
        while True:
            _ = x @ y
            torch.cuda.synchronize()
            it += 1
            now = time.monotonic()
            if now - last >= 0.25:
                f.write(f"{now:.4f} {it / (now - last):.3f}\n")
                it, last = 0, now


def worker_hog(a):
    """Hold ``bytes`` of GPU memory until killed."""
    import torch

    try:
        x = torch.empty(int(a.bytes), dtype=torch.uint8, device=a.device)
        x.fill_(1)
        torch.cuda.synchronize()
        print(f"ready: holding {a.bytes / GiB:.2f} GiB on {a.device}", flush=True)
    except Exception as e:
        print(f"failed: {e!r}", flush=True)
        return
    try:
        sys.stdin.read()
    except Exception:
        pass
    time.sleep(1e9)


STAGES = {"env": stage_env, "pytest": stage_pytest, "tiny": stage_tiny, "real": stage_real,
          "colocate": stage_colocate, "spike": stage_spike, "aimd": stage_aimd, "sim": stage_sim}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=[*STAGES, "worker-host", "worker-hog"])
    ap.add_argument("--ctx")
    ap.add_argument("--stage-dir")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--out")
    ap.add_argument("--bytes", type=float, default=0)
    a = ap.parse_args()
    if a.what == "worker-host":
        return worker_host(a)
    if a.what == "worker-hog":
        return worker_hog(a)
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")     # \r-bars make unreadable logs
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:
        pass
    with open(a.ctx) as f:
        ctx = SimpleNamespace(**json.load(f))
    ctx.stage_dir = a.stage_dir
    os.makedirs(ctx.stage_dir, exist_ok=True)
    res = Result(ctx.stage_dir, a.what)
    try:
        STAGES[a.what](ctx, res)
    except Exception as e:
        traceback.print_exc()
        res.check("exception", False, f"{type(e).__name__}: {e}"[:500])
    res.finish()
    sys.stdout.flush()
    os._exit(0)          # do not wait on lingering non-daemon threads / CUDA teardown


if __name__ == "__main__":
    main()
