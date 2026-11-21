"""Single-GPU validation: everything in this repo that one GPU can test, in one run.

    bash scripts/test_single_gpu.sh                     # full run (~30-45 min on an 80 GB GPU)
    bash scripts/test_single_gpu.sh --quick             # ~10 min once the model is downloaded
    bash scripts/test_single_gpu.sh --dry-run           # no GPU: every stage's orchestration on CPU

Stages (each in its own process, with a timeout; a failure is recorded and the run goes on):

    0 env       GPU / driver / CUDA / torch / host capture; stops the run if there is no CUDA
    1 pytest    the unit + e2e suite (CPU)
    2 tiny      tiny model, anchor + 3 donor processes on one GPU (separate CUDA contexts, NVML
                watchdog): distributed == local bitwise, then SIGKILL / SIGSTOP / real host
                allocation / control-plane death / all donors killed, still bitwise
    3 real      real MoE (default allenai/OLMoE-1B-7B-0924): export, patch, distributed decode vs
                local reference on real prompts, tokens/s, per-layer latency, batch sweep, chaos
    4 colocate  experiment 3: host slowdown vs kappa (bench/colocate.py)
    5 spike     experiment 4: reclaim spike test with NVML (bench/spike.py)
    6 aimd      AIMD governor against real NVML counters (exploratory)
    7 sim       trace simulator (CPU)

Output: results/single-gpu-<timestamp>/{summary.md, summary.json, <NN-stage>/...} and a .tar.gz.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES_PY = os.path.join(REPO, "scripts", "single_gpu_stages.py")

# key, title, timeout full (s), timeout quick (s)
STAGES = [
    ("env", "Environment capture", 180, 180),
    ("pytest", "Unit + e2e tests (CPU)", 900, 900),
    ("tiny", "Tiny model on GPU: split decode + chaos", 900, 600),
    ("real", "Real model end to end", 3600, 2400),
    ("colocate", "Exp. 3: host slowdown vs kappa", 1200, 600),
    ("spike", "Exp. 4: reclaim spike test", 1800, 600),
    ("aimd", "AIMD governor vs NVML (exploratory)", 900, 400),
    ("sim", "Trace simulator (CPU)", 900, 900),
]
KEYS = [s[0] for s in STAGES]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924",
                    help="HF id for stage 3. Qwen/Qwen3-30B-A3B needs an 80 GB GPU and ~70 GB free host RAM")
    ap.add_argument("--quick", action="store_true", help="fewer tokens/batches/trials/windows")
    ap.add_argument("--dry-run", "--cpu", dest="dry_run", action="store_true",
                    help="no GPU: run every stage's orchestration on CPU with tiny/random models")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip", action="append", default=[], help="stage key or number; repeatable or comma-separated")
    ap.add_argument("--only", action="append", default=[], help="run only these stages (env always runs)")
    ap.add_argument("--out", help="results directory (default results/single-gpu-<timestamp>)")
    ap.add_argument("--work-dir", help="scratch for master stores etc. (default <repo>/.single-gpu-work)")
    ap.add_argument("--keep-work", action="store_true", help="keep the exported master store for the next run")
    ap.add_argument("--host-cmd", help="stage 4: real host code printing one throughput number per stdout line")
    ap.add_argument("--batches", help="stage 3 batch sizes, e.g. 1,8,32")
    ap.add_argument("--max-new", type=int, help="stage 3 new tokens per generation")
    ap.add_argument("--bundle-size", type=int, help="stage 3 experts per bundle (default 8; 16 for 30B+)")
    ap.add_argument("--spike-trials", type=int)
    ap.add_argument("--aimd-seconds", type=int)
    ap.add_argument("--timeout-scale", type=float, default=1.0, help="multiply every stage timeout")
    ap.add_argument("--no-tar", action="store_true")
    return ap.parse_args(argv)


def _expand(xs: list[str]) -> set[str]:
    out = set()
    for x in xs:
        for y in x.split(","):
            y = y.strip()
            if not y:
                continue
            if y.isdigit() and int(y) < len(KEYS):
                y = KEYS[int(y)]
            if y not in KEYS:
                sys.exit(f"unknown stage {y!r}; stages: {', '.join(f'{i}={k}' for i, k in enumerate(KEYS))}")
            out.add(y)
    return out


def nvml_index(device: str) -> int:
    """NVML enumerates physical GPUs; CUDA honours CUDA_VISIBLE_DEVICES."""
    i = int(device.split(":")[1]) if ":" in device else 0
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        parts = [p.strip() for p in cvd.split(",") if p.strip()]
        if i < len(parts) and parts[i].isdigit():
            return int(parts[i])
    return i


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(msg + "\n")


def run_stage(num: int, key: str, title: str, timeout: float, ctx: dict, out_dir: str, say) -> dict:
    sdir = os.path.join(out_dir, f"{num}-{key}")
    os.makedirs(sdir, exist_ok=True)
    ctx = dict(ctx, timeout=timeout)
    ctx_path = os.path.join(sdir, "ctx.json")
    with open(ctx_path, "w") as f:
        json.dump(ctx, f, indent=1)
    log_path = os.path.join(sdir, "stage.log")
    cmd = [sys.executable, "-u", STAGES_PY, key, "--ctx", ctx_path, "--stage-dir", sdir]
    say(f"\n{'=' * 78}\n[{num}/{len(STAGES) - 1}] {title}  ({key}; timeout {timeout / 60:.0f} min; log {log_path})\n{'=' * 78}")
    t0 = time.monotonic()
    timed_out = False
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             env=os.environ.copy(), start_new_session=True)

        def pump():
            for line in p.stdout:
                log.write(line)
                log.flush()
                print(f"  {key:>8} | {line.rstrip()}", flush=True)

        th = threading.Thread(target=pump, daemon=True)
        th.start()
        try:
            p.wait(timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            say(f"  {key}: TIMEOUT after {timeout:.0f}s, killing the stage's process group")
        finally:
            try:
                os.killpg(p.pid, signal.SIGKILL)       # stage + any donors / workers it left behind
            except (ProcessLookupError, PermissionError):
                pass
            p.wait()
            th.join(5)
    dt = time.monotonic() - t0
    try:
        with open(os.path.join(sdir, "result.json")) as f:
            r = json.load(f)
    except (OSError, ValueError):
        r = {"stage": key, "status": "fail", "headline": "", "checks": [], "metrics": {}, "notes": []}
    if timed_out:
        r["status"] = "fail"
        r["checks"].append({"name": "timeout", "status": "fail", "detail": f"killed after {timeout:.0f}s"})
        r["headline"] = f"TIMEOUT after {timeout:.0f}s. " + r.get("headline", "")
    elif r.get("status") == "running":
        r["status"] = "fail"
        r["checks"].append({"name": "crashed", "status": "fail", "detail": f"exit code {p.returncode}; see stage.log"})
    if r["status"] == "fail" and not r.get("headline"):
        fails = [c for c in r["checks"] if c["status"] == "fail"]
        r["headline"] = (fails[0]["name"] + ": " + fails[0]["detail"])[:200] if fails else "failed"
    r.update(num=num, key=key, title=title, seconds=round(dt, 1), dir=os.path.relpath(sdir, out_dir))
    with open(os.path.join(sdir, "result.json"), "w") as f:
        json.dump(r, f, indent=1)
    say(f"  -> {r['status'].upper()} in {dt:.0f}s: {r.get('headline', '')}")
    return r


def skipped(num, key, title, reason, out_dir) -> dict:
    return {"num": num, "key": key, "title": title, "status": "skip", "seconds": 0, "headline": reason,
            "checks": [], "metrics": {}, "notes": [], "dir": None}


def _md_escape(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def write_summary(out_dir: str, results: list[dict], args, started: str, total_s: float) -> str:
    env = next((r for r in results if r["key"] == "env"), {}).get("metrics", {})
    mode = "DRY RUN (CPU only; GPU-specific numbers are not real)" if args.dry_run else "GPU"
    lines = [f"# Single-GPU validation: {started}", "",
             f"- Mode: **{mode}**{' (quick)' if args.quick else ''}; device `{'cpu' if args.dry_run else args.device}`; "
             f"model `{'tiny random OLMoE + Qwen3-MoE' if args.dry_run else args.model}`",
             f"- GPU: {env.get('gpu_name', 'none')} ({env.get('gpu_total_gib', '-')} GiB), driver "
             f"{env.get('nvml_driver', '-')}, compute mode {env.get('compute_mode', '-')}",
             f"- Software: torch {env.get('ver_torch')} (CUDA {env.get('torch_cuda_build')}), transformers "
             f"{env.get('ver_transformers')}, python {env.get('python')}",
             f"- Host: {env.get('cpu_count')} CPUs ({env.get('cpu_model')}), RAM {env.get('ram_total_gib')} GiB",
             f"- Code: `{env.get('git_branch')}` @ `{(env.get('git_sha') or '')[:12]}`"
             f"{' (dirty)' if env.get('git_dirty') else ''}",
             f"- Wall time: {total_s / 60:.1f} min", "",
             "| # | Stage | Status | Time | Headline |", "|---|---|---|---|---|"]
    icon = {"pass": "PASS", "fail": "**FAIL**", "skip": "skip"}
    for r in results:
        lines.append(f"| {r['num']} | {r['title']} | {icon.get(r['status'], r['status'])} | "
                     f"{r['seconds']:.0f}s | {_md_escape(r.get('headline', ''))} |")
    n = {s: sum(1 for r in results if r["status"] == s) for s in ("pass", "fail", "skip")}
    lines += ["", f"**{n['pass']} passed, {n['fail']} failed, {n['skip']} skipped.**", ""]
    for r in results:
        lines += [f"## {r['num']}. {r['title']}: {r['status'].upper()}", ""]
        if r.get("dir"):
            lines.append(f"Logs: `{r['dir']}/`")
            lines.append("")
        if r.get("headline"):
            lines += [r["headline"], ""]
        for note in r.get("notes", []):
            lines.append(f"> {_md_escape(note)}")
        if r.get("notes"):
            lines.append("")
        if r.get("checks"):
            lines += ["| Check | Result | Detail |", "|---|---|---|"]
            for c in r["checks"]:
                lines.append(f"| {_md_escape(c['name'])} | {c['status']} | {_md_escape(c['detail'])} |")
            lines.append("")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"started": started, "mode": "dry-run" if args.dry_run else "gpu", "quick": args.quick,
                   "args": vars(args), "wall_s": round(total_s, 1), "counts": n, "stages": results}, f, indent=1)
    return md


def main(argv=None):
    args = parse_args(argv)
    skip, only = _expand(args.skip), _expand(args.only)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.abspath(args.out or os.path.join(REPO, "results", f"single-gpu-{stamp}"
                                                       + ("-dryrun" if args.dry_run else "")))
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.abspath(args.work_dir or os.path.join(REPO, ".single-gpu-work"))
    os.makedirs(work_dir, exist_ok=True)
    say = Tee(os.path.join(out_dir, "driver.log"))
    os.environ["PYTHONPATH"] = REPO + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    say(f"single-GPU validation -> {out_dir}")
    say(f"python {sys.executable}; mode {'dry-run (CPU)' if args.dry_run else 'GPU ' + args.device}"
        f"{'; quick' if args.quick else ''}; model {args.model}")
    ctx = {"out_dir": out_dir, "work_dir": work_dir, "repo": REPO, "device": args.device,
           "nvml_index": nvml_index(args.device), "dry_run": args.dry_run, "quick": args.quick,
           "model": args.model, "host_cmd": args.host_cmd,
           "batches": [int(x) for x in args.batches.split(",")] if args.batches else None,
           "max_new": args.max_new, "bundle_size": args.bundle_size, "keep_work": args.keep_work,
           "spike_trials": args.spike_trials, "aimd_seconds": args.aimd_seconds, "gpu_total_gib": None}
    started = _dt.datetime.now().isoformat(timespec="seconds")
    t0 = time.monotonic()
    results, abort = [], None
    for num, (key, title, t_full, t_quick) in enumerate(STAGES):
        if abort:
            results.append(skipped(num, key, title, abort, out_dir))
            continue
        if key in skip or (only and key not in only and key != "env"):
            results.append(skipped(num, key, title, "skipped by --skip/--only", out_dir))
            say(f"\n[{num}/{len(STAGES) - 1}] {title}: skipped")
            continue
        timeout = (t_quick if args.quick else t_full) * args.timeout_scale
        if key == "real" and ("30B" in args.model or "235B" in args.model):
            timeout *= 1.5
        r = run_stage(num, key, title, timeout, ctx, out_dir, say)
        results.append(r)
        if key == "env":
            m = r.get("metrics", {})
            ctx["gpu_total_gib"] = m.get("gpu_total_gib")
            if not args.dry_run and not m.get("cuda_available"):
                abort = "not run: no CUDA device (see stage 0); use --dry-run to test the harness on CPU"
                say("\nERROR: no usable CUDA device. " + next((c["detail"] for c in r.get("checks", [])
                                                               if c["name"] == "cuda"), ""))
    total = time.monotonic() - t0
    md = write_summary(out_dir, results, args, started, total)
    if not args.keep_work:
        for d in ("tiny", "aimd-master", "aimd-cluster"):
            shutil.rmtree(os.path.join(work_dir, d), ignore_errors=True)
    say("\n" + "=" * 78)
    say(md)
    if not args.no_tar:
        tgz = out_dir.rstrip("/") + ".tar.gz"
        with tarfile.open(tgz, "w:gz") as tf:
            tf.add(out_dir, arcname=os.path.basename(out_dir))
        say(f"results: {out_dir}\nshareable archive: {tgz}")
    else:
        say(f"results: {out_dir}")
    return 1 if any(r["status"] == "fail" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
