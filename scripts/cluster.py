#!/usr/bin/env python3
"""Multi-node validation of the enkhora prototype. Entry point: scripts/test_cluster.sh (-h for usage).

Subcommands:
  run        the whole test (stages 0-5), or ``--print-plan`` to show the exact commands only
  cleanup    kill everything a run launched, on every node (the bash trap calls this)
  node-info  stage 0 inventory of the node it runs on (prints one ``@@NODEINFO`` line)
  agent      per-node signal agent: the anchor driver asks it to SIGKILL/SIGSTOP/SIGCONT roles
  role       wrapper that pins a GPU, writes a pid file, arms a TTL and execs a role

Every process a run starts carries ``-X enk_run=<tag>`` on its command line and leads its own
process group, so cleanup can find and kill it on any node with pkill and kill -- -PGID.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import datetime
import glob
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import dataclass, field

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFE = re.compile(r"^[A-Za-z0-9_.@-]+$")
STAGES = {0: "inventory", 1: "fabric", 2: "decode", 3: "chaos", 4: "colocate", 5: "collect"}
SIGNALS = {"KILL", "STOP", "CONT", "TERM", "INT"}
PASS_ENV = ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_ENDPOINT",
            "TORCH_HOME", "OMP_NUM_THREADS")
DEFAULT_HF_MODEL = "allenai/OLMoE-1B-7B-0924"


def q(s) -> str:
    return shlex.quote(str(s))


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def say(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pack(obj) -> str:
    return base64.urlsafe_b64encode(zlib.compress(json.dumps(obj).encode())).decode()


def unpack(s: str):
    return json.loads(zlib.decompress(base64.urlsafe_b64decode(s.encode())))


def run_capture(argv: list[str], timeout: float) -> tuple[int, str]:
    """Run a local command with a hard timeout (kills its whole process group on expiry)."""
    try:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             text=True, errors="replace", start_new_session=True)
    except OSError as e:
        return 127, f"{argv[0]}: {e}"
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        _killpg(p.pid)
        try:
            out, _ = p.communicate(timeout=5)
        except Exception:
            out = ""
        return 124, (out or "") + f"\n[timed out after {timeout:.0f}s]"


def _killpg(pid: int, sig=signal.SIGKILL):
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


# ===================================================================== hostlists and nodes
def _split_top(expr: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in expr:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def expand_hostlist(expr: str) -> list[str]:
    """Pure-Python fallback for ``scontrol show hostnames`` (e.g. ``gpu[01-03,07],login1``)."""
    out = []
    for part in _split_top(expr):
        m = re.match(r"^([^\[]*)\[([^\]]+)\](.*)$", part)
        if not m:
            out.append(part)
            continue
        pre, body, post = m.groups()
        for r in body.split(","):
            if "-" in r:
                a, b = r.split("-", 1)
                names = [f"{i:0{len(a)}d}" for i in range(int(a), int(b) + 1)]
            else:
                names = [r]
            for n in names:
                out.extend(expand_hostlist(pre + n + post))
    return out


@dataclass
class Node:
    idx: int
    name: str                       # hostname: srun -w target, display name, pid-dir name
    target: str                     # what the launcher addresses (ssh user@host, or the name)
    slots_hint: int | None = None   # hostfile slots= / fake GPU count
    addr_hint: str | None = None    # hostfile addr=
    job_hint: str | None = None     # hostfile job=
    gpus: int = 0
    gpu_names: list = field(default_factory=list)
    gpu_mem_mib: list = field(default_factory=list)
    addr: str = ""
    job: str = ""
    short: str = ""
    info: dict = field(default_factory=dict)


def parse_hostfile(path: str) -> list[Node]:
    nodes = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            toks = line.split()
            target, kv = toks[0], dict(t.split("=", 1) for t in toks[1:] if "=" in t)
            name = target.split("@", 1)[-1]
            nodes.append(Node(len(nodes), name, target,
                              slots_hint=int(kv["slots"]) if "slots" in kv else None,
                              addr_hint=kv.get("addr"), job_hint=kv.get("job")))
    if not nodes:
        raise SystemExit(f"hostfile {path} lists no hosts")
    return nodes


# ===================================================================== launchers
class Launcher:
    """Turns 'run this shell command on node N' into a local argv (bash, ssh or srun)."""

    kind = "base"

    def __init__(self, o):
        self.o = o

    def discover(self) -> list[Node]:
        raise NotImplementedError

    def argv(self, node: Node, cmd: str, name: str) -> list[str]:
        raise NotImplementedError

    def run(self, node: Node, cmd: str, name: str, timeout: float) -> tuple[int, str]:
        return run_capture(self.argv(node, cmd, name), timeout)

    def start(self, node: Node, cmd: str, name: str, log_path: str) -> subprocess.Popen:
        f = open(log_path, "ab")
        try:
            return subprocess.Popen(self.argv(node, cmd, name), stdout=f, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        finally:
            f.close()

    def extra_cleanup(self, tag: str) -> dict:
        return {}

    def registry(self) -> dict:
        return {}


class LocalLauncher(Launcher):
    """Every 'node' is this machine; roles are subprocesses. Same shell commands as ssh/srun."""

    kind = "local"

    def discover(self):
        return [Node(i, f"local{i}", "127.0.0.1", slots_hint=self.o.local_gpus, addr_hint="127.0.0.1",
                     job_hint=None) for i in range(self.o.local_nodes)]

    def argv(self, node, cmd, name):
        return ["bash", "-c", cmd]


class SshLauncher(Launcher):
    kind = "ssh"
    BASE_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
                 "-o", "ServerAliveCountMax=4", "-o", "StrictHostKeyChecking=accept-new"]

    def __init__(self, o):
        super().__init__(o)
        self.opts = self.BASE_OPTS + shlex.split(o.ssh_opts or "")

    def discover(self):
        if not self.o.hosts:
            raise SystemExit("--launcher ssh needs --hosts HOSTFILE")
        return parse_hostfile(self.o.hosts)

    def argv(self, node, cmd, name):
        return ["ssh", *self.opts, node.target, "bash -c " + q(cmd)]

    def rsync_argv(self, node, remote_repo):
        ssh = " ".join(["ssh", *map(q, self.opts)])
        return ["rsync", "-az", "--delete", "--exclude", ".git", "--exclude", ".venv", "--exclude", "results",
                "--exclude", ".cache", "--exclude", "__pycache__", "--exclude", "*.egg-info",
                "--exclude", ".pytest_cache", "--exclude", "master", "-e", ssh,
                REPO + "/", f"{node.target}:{remote_repo}/"]

    def registry(self):
        return {"ssh_opts": self.opts}


class SlurmLauncher(Launcher):
    kind = "slurm"

    def __init__(self, o):
        super().__init__(o)
        self.jobid = o.jobid or os.environ.get("SLURM_JOB_ID")

    def discover(self):
        expr = self.o.nodelist or os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
        if not expr:
            raise SystemExit("slurm launcher: no SLURM_JOB_NODELIST (run inside salloc/sbatch, or pass --nodelist)")
        names = None
        if shutil.which("scontrol") and not self.o.print_plan:
            rc, out = run_capture(["scontrol", "show", "hostnames", expr], 30)
            if rc == 0:
                names = [x.strip() for x in out.split() if x.strip()]
        names = names or expand_hostlist(expr)
        return [Node(i, n, n, slots_hint=self.o.fake_gpus if self.o.print_plan else None) for i, n in enumerate(names)]

    def step_name(self, name):
        return f"enk-{self.o.tag}-{name}"[:120]

    def argv(self, node, cmd, name):
        a = ["srun", "--nodes=1", "--ntasks=1", f"--nodelist={node.name}", "--overlap",
             f"--job-name={self.step_name(name)}"]
        if self.o.jobid and not os.environ.get("SLURM_JOB_ID"):
            a.append(f"--jobid={self.o.jobid}")
        cpt = self.o.srun_cpus or os.environ.get("SLURM_CPUS_PER_TASK")
        if cpt:
            a.append(f"--cpus-per-task={cpt}")
        a += shlex.split(self.o.srun_args or "")
        return a + ["bash", "-c", cmd]

    def extra_cleanup(self, tag):
        """Cancel any of our steps that are still registered (e.g. a role ignored the kill)."""
        if not self.jobid or not shutil.which("squeue"):
            return {}
        rc, out = run_capture(["squeue", "--steps", f"--jobs={self.jobid}", "--noheader", "--format=%i %j"], 30)
        steps = [ln.split()[0] for ln in out.splitlines()
                 if rc == 0 and len(ln.split()) >= 2 and ln.split()[1].startswith(f"enk-{tag}-")]
        if steps:
            run_capture(["scancel", "--signal=KILL", *steps], 30)
            run_capture(["scancel", *steps], 30)
        return {"scancelled_steps": steps}

    def registry(self):
        return {"jobid": self.jobid}


def make_launcher(o) -> Launcher:
    kind = o.launcher
    if kind == "auto":
        kind = "slurm" if (os.environ.get("SLURM_JOB_ID") or o.nodelist) else ("ssh" if o.hosts else "local")
    o.launcher = kind
    return {"local": LocalLauncher, "ssh": SshLauncher, "slurm": SlurmLauncher}[kind](o)


# ===================================================================== remote command construction
class Commands:
    """Shell command strings executed on a node (identical for local, ssh and srun)."""

    def __init__(self, o, repo: str, py: str, workdir: str):
        self.o, self.repo, self.py, self.workdir = o, repo, py, workdir
        self.tag = o.tag

    def env(self) -> dict:
        e = {"PYTHONPATH": self.repo, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
             "ENK_RUN": self.tag}
        for k in PASS_ENV:
            if os.environ.get(k):
                e[k] = os.environ[k]
        for kv in self.o.env or []:
            k, v = kv.split("=", 1)
            e[k] = v
        return e

    def shell(self, argv: list) -> str:
        exports = " ".join(f"{k}={q(v)}" for k, v in self.env().items())
        return f"cd {q(self.repo)} && export {exports} && exec " + " ".join(q(a) for a in argv)

    def pid_dir(self, node: Node) -> str:
        return f"{self.workdir}/pids/{node.name}"

    def tagged_py(self, role: str | None = None) -> list[str]:
        a = [self.py, "-X", f"enk_run={self.tag}"]
        return a + (["-X", f"enk_role={role}"] if role else [])

    def role(self, node: Node, role: str, inner: list, slot: int | None = None, gpu_mode: str = "cpu",
             nvml: bool = False) -> str:
        w = self.tagged_py() + [f"{self.repo}/scripts/cluster.py", "role", "--tag", self.tag, "--name", role,
                                "--pid-dir", self.pid_dir(node), "--ttl", str(int(self.o.ttl))]
        if slot is not None:
            w += ["--slot", str(slot), "--gpu-mode", gpu_mode]
        if nvml:
            w.append("--nvml")
        return self.shell(w + ["--"] + inner)

    def node_info(self, node: Node) -> str:
        a = self.tagged_py() + [f"{self.repo}/scripts/cluster.py", "node-info", "--node", node.name]
        if self.o.iface:
            a += ["--iface", self.o.iface]
        return self.shell(a)

    def cleanup(self, node: Node) -> str:
        """Kill this run's process groups (pid files) and anything else tagged, then report strays."""
        t = self.tag
        pat = f"enk_run=[{t[0]}]{t[1:]}"          # the [x] trick: never matches this shell itself
        d = q(self.pid_dir(node))
        return (
            f"for f in {d}/*.pid; do [ -f \"$f\" ] || continue; p=$(cat \"$f\" 2>/dev/null); "
            f"case \"$p\" in \"\"|*[!0-9]*) continue;; esac; "
            f"if ps -o command= -p \"$p\" 2>/dev/null | grep -q -- {q(pat)}; then "
            f"kill -CONT -- -\"$p\" 2>/dev/null; kill -KILL -- -\"$p\" 2>/dev/null; kill -KILL \"$p\" 2>/dev/null; "
            f"elif ! kill -0 \"$p\" 2>/dev/null && kill -0 -- -\"$p\" 2>/dev/null; then "
            f"kill -CONT -- -\"$p\" 2>/dev/null; kill -KILL -- -\"$p\" 2>/dev/null; fi; done; "
            f"pkill -CONT -f {q(pat)} 2>/dev/null; pkill -KILL -f {q(pat)} 2>/dev/null; sleep 1; "
            f"if pgrep -f {q(pat)} >/dev/null 2>&1; then echo '@@STRAY'; ps -o pid=,command= -p "
            f"\"$(pgrep -f {q(pat)} | tr '\\n' ',' | sed 's/,$//')\"; else echo '@@CLEAN'; fi"
        )


# ===================================================================== layout
@dataclass
class DonorRole:
    id: str
    node: int
    slot: int
    port: int
    job: str
    gpu_name: str = "cpu"
    gpu_mem_mib: float = 0.0


def _short1(name: str) -> str:
    host = name.split("@")[-1]
    if re.match(r"^\d+(\.\d+){3}$", host):           # IPv4: keep every octet (10.0.0.12 -> 10-0-0-12)
        return host.replace(".", "-")
    return re.sub(r"[^A-Za-z0-9-]", "", host.split(".")[0])[:24] or "node"


def assign_shorts(nodes: list[Node]):
    """Unique, filesystem- and id-safe short name per node (used in role ids, job ids, log names)."""
    seen = {}
    for n in nodes:
        base = _short1(n.name)
        k = seen.get(base, 0)
        seen[base] = k + 1
        n.short = base if k == 0 else f"{base}-{k}"


def short(n: Node) -> str:
    return n.short or _short1(n.name)


def make_layout(o, nodes: list[Node]) -> list[DonorRole]:
    """Anchor on node 0 slot 0; donors on every other slot. Ids interleave nodes (remote first) so
    the planner's id tie-break spreads primaries across nodes rather than piling onto node 0."""
    per_node = []
    for n in nodes:
        slots = [s for s in range(n.gpus) if not (n.idx == 0 and (s == 0 or o.no_anchor_node_donors))]
        per_node.append((n, slots))
    order = per_node[1:] + per_node[:1]
    cells, depth = [], max((len(s) for _, s in per_node), default=0)
    for k in range(depth):
        for n, slots in order:
            if k < len(slots):
                cells.append((n, slots[k]))
    if o.max_donors:
        cells = cells[:o.max_donors]
    out = []
    for k, (n, s) in enumerate(cells):
        did = f"d{k:02d}-{short(n)}-g{s}"
        job = {"node": n.job or f"job-{short(n)}", "gpu": f"job-{did}", "single": "job-all"}[o.jobs]
        name = n.gpu_names[s] if s < len(n.gpu_names) else ("cpu" if o.device_mode == "cpu" else "GPU?")
        mem = n.gpu_mem_mib[s] if s < len(n.gpu_mem_mib) else 0.0
        out.append(DonorRole(did, n.idx, s, o.donor_port + k, job, name, mem))
    return out


# ===================================================================== the run
class Run:
    def __init__(self, o):
        self.o = o
        self.launcher = make_launcher(o)
        self.t0 = time.time()
        self.results: dict[int, dict] = {}
        self.procs: dict[str, tuple] = {}         # role -> (Popen, node, log path, kind)
        self.expected_exit: set[str] = set()
        self.notes: list[str] = []
        self.nodes: list[Node] = []
        self.donors: list[DonorRole] = []
        self.cleanup_report: dict = {}
        self.finished = False
        self._lock = threading.Lock()
        remote = o.launcher == "ssh"
        self.repo = o.remote_repo or REPO
        self.py = o.remote_python or (f"{self.repo}/.venv/bin/python" if (remote and o.sync) else sys.executable)
        self.workdir = o.remote_workdir or (f"/tmp/enkhora-{o.tag}" if remote else os.path.join(o.out, "work"))
        self.cmds = Commands(o, self.repo, self.py, self.workdir)
        self.logdir = os.path.join(o.out, "logs")

    # ------------------------------------------------------------ bookkeeping
    def stage(self, n: int, status: str, headline: str = "", reason: str = "", **details):
        r = {"stage": n, "name": STAGES[n], "status": status, "headline": headline, "reason": reason, **details}
        self.results[n] = r
        say(f"stage {n} {STAGES[n]}: {status}  {headline}{('  (' + reason + ')') if reason else ''}")
        return r

    def want(self, n: int) -> bool:
        return n in self.o.stage_set

    def registry(self):
        reg = {"tag": self.o.tag, "launcher": self.o.launcher, "repo": self.repo, "python": self.py,
               "workdir": self.workdir, "nodes": [{"idx": n.idx, "name": n.name, "target": n.target} for n in self.nodes],
               "srun_args": self.o.srun_args, "srun_cpus": self.o.srun_cpus, "jobid": self.o.jobid,
               "ssh_opts": self.o.ssh_opts, **self.launcher.registry()}
        os.makedirs(self.o.out, exist_ok=True)
        try:
            os.remove(os.path.join(self.o.out, ".cleanup_done"))     # a reused --out must not look cleaned
        except FileNotFoundError:
            pass
        with open(os.path.join(self.o.out, "launch.json"), "w") as f:
            json.dump(reg, f, indent=1)

    # ------------------------------------------------------------ main
    def execute(self):
        o = self.o
        os.makedirs(self.logdir, exist_ok=True)
        self.nodes = self.launcher.discover()
        assign_shorts(self.nodes)
        say(f"run {o.tag}: launcher {o.launcher}, {len(self.nodes)} node(s): {', '.join(n.name for n in self.nodes)}")
        say(f"results -> {o.out}")
        self.registry()
        if o.launcher == "ssh" and o.sync:
            self.sync()
        self.inventory()
        if o.device_mode == "cuda" and self.nodes and self.nodes[0].gpus == 0:
            raise SystemExit(f"node 0 ({self.nodes[0].name}) has no usable GPU for the anchor; "
                             "put a GPU node first (hostfile order / --nodelist) or use --device-mode cpu")
        self.donors = make_layout(o, self.nodes)
        plan = self.driver_plan()
        with open(os.path.join(o.out, "plan.json"), "w") as f:
            json.dump(plan, f, indent=1)
        if not self.donors:
            for n in (1, 2, 3):
                if self.want(n):
                    self.stage(n, "FAIL", reason="no donor slots (need >= 2 GPUs in total, or more --local-gpus)")
        elif self.o.stage_set & {1, 2, 3}:
            self.start_roles()
            self.run_driver(plan)
        self.kill_everything("roles")
        if self.want(4):
            self.colocate()

    def finish(self):
        if self.finished:
            return
        self.finished = True
        try:
            self.kill_everything("final")
        except Exception as e:
            self.notes.append(f"cleanup error: {e!r}")
        if "anchor" in self.procs:            # interrupted mid-driver: keep whatever it reported
            got = parse_driver_log(self.procs["anchor"][2], os.path.join(self.o.out, "artifacts"))
            for n in (1, 2, 3):
                if self.want(n) and n not in self.results:
                    r = got.get(n) or {"status": "FAIL", "reason": "interrupted before the stage reported"}
                    self.stage(n, r.pop("status"), r.pop("headline", ""), r.pop("reason", ""),
                               **{k: v for k, v in r.items() if k not in ("stage", "name")})
        try:
            self.collect()
        finally:
            try:
                open(os.path.join(self.o.out, ".cleanup_done"), "w").close()
            except OSError:
                pass

    # ------------------------------------------------------------ ssh --sync
    def sync(self):
        o = self.o
        say(f"syncing {REPO} -> {self.repo} on {len(self.nodes)} host(s) and installing the env")
        install = self.install_cmd()

        def one(n):
            rc, out = self.launcher.run(n, f"mkdir -p {q(self.repo)}", "mkdir", 60)
            if rc == 0:
                rc, out = run_capture(self.launcher.rsync_argv(n, self.repo), o.sync_timeout)
            if rc == 0:
                rc, out = self.launcher.run(n, install, "install", o.sync_timeout)
            with open(os.path.join(self.logdir, f"sync-{n.name}.log"), "w") as f:
                f.write(out)
            return rc

        rcs = parallel({n.name: (lambda n=n: one(n)) for n in self.nodes}, o.sync_timeout + 120)
        bad = [k for k, v in rcs.items() if v != 0]
        if bad:
            self.notes.append(f"--sync failed on {bad}; see logs/sync-*.log")
            say(f"WARNING: sync failed on {bad}")

    def install_cmd(self) -> str:
        """Remote env install for --sync: uv if present, else venv + pip; records this checkout's git SHA."""
        sha = git_rev(REPO)
        return (f"mkdir -p {q(self.repo)} && cd {q(self.repo)} && echo {q(json.dumps(sha))} > .enkhora_rev && "
                "if command -v uv >/dev/null 2>&1; then "
                "{ [ -x .venv/bin/python ] || uv venv --python 3.12 .venv; } && "
                "uv pip install --python .venv/bin/python -q -e '.[gpu]'; "
                "else { [ -x .venv/bin/python ] || python3 -m venv .venv; } && "
                ".venv/bin/python -m pip install -q -e '.[gpu]'; fi")

    # ------------------------------------------------------------ stage 0
    def inventory(self):
        o = self.o
        say("stage 0: inventory on every node")
        jobs = {n.name: (lambda n=n: self.launcher.run(n, self.cmds.node_info(n), "info", o.inventory_timeout))
                for n in self.nodes}
        res = parallel(jobs, o.inventory_timeout + 30)
        problems, infos = [], {}
        for n in self.nodes:
            rc, out = res.get(n.name) or (1, "no result")
            with open(os.path.join(self.logdir, f"inventory-{n.name}.log"), "w") as f:
                f.write(out)
            info = next((json.loads(ln[len("@@NODEINFO "):]) for ln in out.splitlines()
                         if ln.startswith("@@NODEINFO ")), None)
            if info is None:
                problems.append(f"{n.name}: inventory failed (rc {rc}): {out.strip()[-300:]}")
                info = {}
            n.info = info
            infos[n.name] = info
        self.resolve_nodes()
        # consistency checks
        hashes = {n.name: n.info.get("code_hash") for n in self.nodes if n.info}
        shas = {n.name: (n.info.get("git") or {}).get("sha") for n in self.nodes if n.info}
        torchv = {n.name: (n.info.get("torch") or {}).get("version") for n in self.nodes if n.info}
        if len(set(hashes.values())) > 1:
            problems.append(f"code differs between nodes: {hashes}")
        if len({v for v in shas.values() if v}) > 1:
            problems.append(f"git SHA differs between nodes: {shas}")
        if len(set(torchv.values())) > 1:
            problems.append(f"torch version differs between nodes: {torchv}")
        local_hash = code_fingerprint(REPO)
        if o.launcher != "local" and hashes and local_hash not in hashes.values():
            problems.append(f"no node runs this checkout's code (local {local_hash[:12]}; use --sync or update)")
        for n in self.nodes:
            t = n.info.get("torch") or {}
            if n.info.get("nvidia_smi_gpus") and not t.get("cuda_available") and o.launcher != "local":
                problems.append(f"{n.name}: nvidia-smi sees GPUs but torch has no CUDA ({t.get('version')})")
            if o.launcher == "slurm" and n.info and o.device_mode_requested != "cpu" and \
                    not n.info.get("cuda_visible_devices") and not n.info.get("nvidia_smi_gpus"):
                problems.append(f"{n.name}: the srun step sees no GPUs; try --srun-args='--gres=gpu:N'")
        gpu_models = sorted({g for n in self.nodes for g in n.gpu_names[:n.gpus]})
        mixed = len(gpu_models) > 1
        rows = [{"node": n.name, "addr": n.addr, "gpus": n.gpus,
                 "gpu": ", ".join(sorted(set(n.gpu_names[:n.gpus]))) or "-",
                 "mem_mib": n.gpu_mem_mib[0] if n.gpu_mem_mib else None,
                 "driver": n.info.get("driver"), "torch": (n.info.get("torch") or {}).get("version"),
                 "cuda": (n.info.get("torch") or {}).get("cuda"),
                 "ib": n.info.get("ib_summary") or "-", "git": ((n.info.get("git") or {}).get("sha") or "-")[:12],
                 "code": (n.info.get("code_hash") or "-")[:12]} for n in self.nodes]
        with open(os.path.join(o.out, "inventory.json"), "w") as f:
            json.dump(infos, f, indent=1)
        total = sum(n.gpus for n in self.nodes)
        head = (f"{len(self.nodes)} nodes, {total} {'GPU' if o.device_mode == 'cuda' else 'CPU'} slots; "
                f"models: {', '.join(gpu_models) or 'none'}; code {'consistent' if len(set(hashes.values())) == 1 else 'MISMATCH'}")
        self.mixed = mixed and o.device_mode == "cuda"
        self.gpu_models = gpu_models
        st = "FAIL" if problems else "PASS"
        if o.device_mode == "cpu" and not problems:
            self.notes.append("CPU dry run: no CUDA devices are used; GPU-only numbers are not meaningful")
        self.stage(0, st, head, "; ".join(problems), nodes=rows, mixed_gpu_generations=self.mixed,
                   device_mode=o.device_mode)

    def resolve_nodes(self):
        """GPU counts, advertised addresses, job ids and the device mode, from the inventory."""
        o = self.o
        any_cuda = any((n.info.get("torch") or {}).get("cuda_available") for n in self.nodes)
        if o.device_mode == "auto":
            o.device_mode = "cuda" if (any_cuda and o.launcher != "local") else "cpu"
        for n in self.nodes:
            gl = n.info.get("visible_gpus") or []
            if o.device_mode == "cuda":
                if not (n.info.get("torch") or {}).get("cuda_available"):
                    gl = []                     # torch cannot use this node's GPUs: no roles there
                n.gpus = min(o.gpus_per_node or n.slots_hint or len(gl), len(gl))
                n.gpu_names = [g.get("name", "GPU?") for g in gl]
                n.gpu_mem_mib = [float(g.get("memory_mib") or 0) for g in gl]
            else:
                n.gpus = o.gpus_per_node or n.slots_hint or o.local_gpus
                n.gpu_names = ["cpu"] * n.gpus
                n.gpu_mem_mib = [0.0] * n.gpus
            n.addr = (n.addr_hint or (n.info.get("iface_ip") if o.iface else None) or
                      (n.info.get("primary_ip") if o.advertise == "ip" else None) or
                      (n.target.split("@")[-1] if o.launcher == "ssh" else n.name))
            if o.launcher == "local":
                n.addr = "127.0.0.1"
            if o.iface and not n.info.get("iface_ip") and o.launcher != "local":
                self.notes.append(f"{n.name}: interface {o.iface} not found; advertising {n.addr}")
            n.job = n.job_hint or ""

    # ------------------------------------------------------------ roles
    def bind_host(self):
        return self.o.bind_host or ("127.0.0.1" if self.o.launcher == "local" else "0.0.0.0")

    def expert_demand(self):
        """Planner demand in GPU-equivalents. Half the donors' total kappa: primaries spread over at
        least half the donors, and standbys keep compute headroom so a kappa shrink can be a pure
        routing-table edit."""
        return self.o.expert_demand or max(1.0, round(0.5 * len(self.donors), 2))

    def control_argv(self):
        o = self.o
        demand = self.expert_demand()
        return self.cmds.tagged_py("control") + [
            "-m", "enkhora", "-v", "control", "--host", self.bind_host(), "--port", str(o.control_port),
            "--expert-demand", str(demand), "--spread-g", str(o.spread_g), "--alpha", "1.0",
            "--copy-rate-mbps", "1000000", "--max-replicas", str(o.max_replicas)]

    def agent_argv(self, n: Node):
        return self.cmds.tagged_py("agent") + [
            f"{self.repo}/scripts/cluster.py", "agent", "--host", self.bind_host(),
            "--port", str(self.o.agent_port + n.idx), "--pid-dir", self.cmds.pid_dir(n), "--tag", self.o.tag,
            "--node", n.name]

    def donor_argv(self, d: DonorRole):
        o, n = self.o, self.nodes[d.node]
        a = self.cmds.tagged_py(d.id) + [
            "-m", "enkhora", "-v", "donor", "--id", d.id, "--job", d.job, "--node", n.name,
            "--control", f"{self.nodes[0].addr}:{o.control_port}", "--host", self.bind_host(),
            "--advertise-host", n.addr, "--port", str(d.port), "--kappa", "1.0",
            "--device", "cuda:0" if o.device_mode == "cuda" else "cpu"]
        if o.device_mode == "cuda":
            guard = o.guard_mib if o.guard_mib is not None else 2048
            mem = o.mem_mib or max(1024.0, 0.85 * d.gpu_mem_mib - guard - 512)
            a += ["--guard-mib", str(guard), "--slack-mib", "512", "--mem-mib", f"{mem:.0f}"]
        else:
            a += ["--mem-mib", str(o.mem_mib or 64), "--guard-mib", str(o.guard_mib or 0.25), "--slack-mib", "0.25",
                  "--threads", "1"]
        return a

    def use_nvml(self):
        return self.o.device_mode == "cuda" and self.o.nvml in ("auto", "on")

    def driver_argv(self, plan):
        return self.cmds.tagged_py("anchor") + [f"{self.repo}/scripts/cluster_anchor.py", "--plan-b64", pack(plan)]

    def role_specs(self, plan=None):
        """(role, node, shell command, kind) for every long-running role, in start order."""
        n0, specs = self.nodes[0], []
        gm = self.o.device_mode
        specs.append(("control", n0, self.cmds.role(n0, "control", self.control_argv()), "control"))
        for n in self.nodes:
            specs.append((f"agent-{short(n)}", n, self.cmds.role(n, f"agent-{short(n)}", self.agent_argv(n)),
                          "agent"))
        for d in self.donors:
            n = self.nodes[d.node]
            specs.append((d.id, n, self.cmds.role(n, d.id, self.donor_argv(d), slot=d.slot if gm == "cuda" else None,
                                                  gpu_mode=gm, nvml=self.use_nvml()), "donor"))
        if plan is not None:
            specs.append(("anchor", n0, self.cmds.role(n0, "anchor", self.driver_argv(plan),
                                                         slot=0 if gm == "cuda" else None, gpu_mode=gm), "anchor"))
        return specs

    def launch(self, role, node, cmd, kind):
        log = os.path.join(self.logdir, f"{role}.log")
        with open(log, "wb") as f:              # truncate: a reused --out must not mix runs
            f.write(f"# {now_iso()} launching {role} on {node.name} via {self.o.launcher}\n".encode())
        p = self.launcher.start(node, cmd, role, log)
        with self._lock:
            self.procs[role] = (p, node, log, kind)
        return p

    def start_roles(self):
        say(f"starting control plane + {len(self.nodes)} agent(s) + {len(self.donors)} donor(s)")
        stagger = {"ssh": 0.3, "slurm": 0.1}.get(self.o.launcher, 0.0)   # stay under sshd MaxStartups
        for role, node, cmd, kind in self.role_specs():
            self.launch(role, node, cmd, kind)
            time.sleep(stagger)
        time.sleep(1.0)
        dead = [r for r, (p, *_rest) in self.procs.items() if p.poll() is not None]
        if dead:
            self.notes.append(f"roles exited right after launch: {dead}")
            say(f"WARNING: exited right after launch: {dead} (see logs/)")

    def driver_plan(self) -> dict:
        o = self.o
        n0 = self.nodes[0] if self.nodes else Node(0, "?", "?")
        anchor_gpu = n0.gpu_names[0] if n0.gpu_names else "cpu"
        return {
            "tag": o.tag, "model": o.model, "bundle_size": o.bundle_size, "dtype": o.dtype, "seed": o.seed,
            "device": "cuda:0" if o.device_mode == "cuda" else "cpu", "anchor_gpu": anchor_gpu,
            "bind_host": self.bind_host(), "advertise": n0.addr, "anchor_port": o.anchor_port,
            "control": [n0.addr, o.control_port],
            "master_dir": o.master_dir or self.default_master_dir(),
            "donors": [{"id": d.id, "node": d.node, "node_name": self.nodes[d.node].name,
                        "addr": [self.nodes[d.node].addr, d.port], "slot": d.slot, "gpu": d.gpu_name,
                        "job": d.job, "remote": d.node != 0} for d in self.donors],
            "agents": [{"node": n.idx, "name": n.name, "addr": [n.addr, o.agent_port + n.idx]} for n in self.nodes],
            "stages": sorted(o.stage_set & {1, 2, 3}), "mixed": getattr(self, "mixed", False),
            "audit_atol": o.audit_atol,
            "fabric": {"tokens": o.fabric_tokens, "bytes_per_token": o.fabric_bytes_per_token,
                       "iters": o.fabric_iters},
            "batches": o.batches, "max_new": o.max_new, "correctness_batch": o.correctness_batch,
            "chaos_batch": o.chaos_batch, "chaos_max_new": o.chaos_max_new,
            "timeouts": {"startup": o.startup_timeout, "placement": o.placement_timeout},
            "ref_cache_bundles": o.ref_cache_bundles,
            "planner": {"expert_demand": self.expert_demand(), "spread_g": o.spread_g,
                        "max_replicas": o.max_replicas, "alpha": 1.0, "copy_rate_Bps": 1e12},
        }

    def default_master_dir(self):
        if self.o.model.startswith("tiny"):
            return f"{self.workdir}/master-{self.o.model}-bs{self.o.bundle_size}"
        slug = re.sub(r"[^A-Za-z0-9._-]", "_", self.o.model)
        return f"{self.repo}/.cache/enkhora-master/{slug}-bs{self.o.bundle_size}-{self.o.dtype or 'auto'}"

    # ------------------------------------------------------------ driver (stages 1-3)
    def run_driver(self, plan):
        o = self.o
        say(f"starting anchor driver on {self.nodes[0].name} (model {o.model}); live progress below")
        p = self.launch("anchor", self.nodes[0], self.role_specs(plan)[-1][2], "anchor")
        log = self.procs["anchor"][2]
        stop = threading.Event()
        tail = threading.Thread(target=tail_file, args=(log, stop, "  anchor| "), daemon=True)
        tail.start()
        deadline = time.monotonic() + o.driver_timeout
        while p.poll() is None and time.monotonic() < deadline:
            time.sleep(0.5)
        timed_out = p.poll() is None
        if timed_out:
            self.notes.append(f"anchor driver exceeded --driver-timeout {o.driver_timeout}s; killed")
            self.signal_role("anchor", "KILL")
            try:
                p.wait(15)
            except subprocess.TimeoutExpired:
                _killpg(p.pid)
        time.sleep(0.5)
        stop.set()
        tail.join(2)
        self.expected_exit.add("anchor")
        got = parse_driver_log(log, os.path.join(o.out, "artifacts"))
        for n in (1, 2, 3):
            if not self.want(n):
                continue
            if n in got:
                r = got[n]
                self.stage(n, r.pop("status"), r.pop("headline", ""), r.pop("reason", ""),
                           **{k: v for k, v in r.items() if k not in ("stage", "name")})
            else:
                why = "driver timed out" if timed_out else f"driver exited (rc {p.returncode}) before reporting"
                self.stage(n, "FAIL", reason=f"{why}; see logs/anchor.log")
        if 3 in got:
            self.expected_exit.update(d.id for d in self.donors)
            self.expected_exit.add("control")

    def signal_role(self, role, sig):
        """Signal a role from the orchestrator (used on timeouts), via the launcher."""
        _, node, _, _ = self.procs[role]
        pid = f"{self.cmds.pid_dir(node)}/{role}.pid"
        cmd = (f"p=$(cat {q(pid)} 2>/dev/null) && ps -o command= -p \"$p\" | grep -q -- {q('enk_run=' + self.o.tag)}"
               f" && kill -{sig} -- -\"$p\"")
        return self.launcher.run(node, cmd, f"sig-{role}", 30)

    # ------------------------------------------------------------ stage 4
    def colocate(self):
        o = self.o
        if o.device_mode != "cuda" and not o.colocate_cpu_smoke:
            self.stage(4, "SKIP", reason="needs CUDA (experiment 3 is a GPU time-slicing measurement)")
            return
        picks = []
        if o.device_mode == "cuda":
            for model in self.gpu_models:
                cands = [(n, s) for n in self.nodes for s in range(n.gpus) if n.gpu_names[s] == model]
                cands.sort(key=lambda c: (c[0].idx == 0 and c[1] == 0, c[0].idx, c[1]))  # avoid the anchor GPU
                if cands:
                    picks.append((model, *cands[0]))
        else:
            picks.append(("cpu (smoke)", self.nodes[0], None))
        runs, ok_all = [], True
        for model, n, s in picks:
            role = f"colocate-{short(n)}-g{s if s is not None else 'cpu'}"
            if o.device_mode == "cuda":
                bench = ["--device", "cuda:0", "--windows", str(o.colocate_windows), "--window-s",
                         str(o.colocate_window_s)]
            else:
                bench = ["--device", "cpu", "--allow-cpu", "--host-matmul", "192", "--hidden", "256", "--inter",
                         "128", "--experts", "4", "--tokens", "8", "--windows", "1", "--window-s", "0.4",
                         "--kappas", "0.05,0.25", "--dtype", "float32"]
            bench += shlex.split(o.colocate_args or "")
            inner = self.cmds.tagged_py(role) + [f"{self.repo}/bench/colocate.py", *bench]
            cmd = self.cmds.role(n, role, inner, slot=s, gpu_mode=o.device_mode)
            say(f"stage 4: colocate on {n.name} slot {s} ({model})")
            p = self.launch(role, n, cmd, "colocate")
            try:
                p.wait(o.colocate_timeout)
            except subprocess.TimeoutExpired:
                self.signal_role(role, "KILL")
                try:
                    p.wait(15)
                except subprocess.TimeoutExpired:
                    _killpg(p.pid)
            self.expected_exit.add(role)
            with open(self.procs[role][2], errors="replace") as f:
                parsed = parse_colocate(f.read())
            parsed.update(gpu=model, node=n.name, slot=s, rc=p.returncode)
            ok = p.returncode == 0 and parsed["kappas"]
            ok_all &= bool(ok)
            runs.append(parsed)
        head = "; ".join(f"{r['gpu']}: idle-ctx {fmt(r.get('idle_slowdown_pct'))}%, first kappa >5%: "
                         f"{r.get('first_over_cap')}" for r in runs)
        if o.device_mode != "cuda":
            head = "CPU smoke only (numbers meaningless): " + head
        self.stage(4, "PASS" if ok_all else "FAIL", head, "" if ok_all else "colocate run failed; see logs/colocate-*",
                   runs=runs)

    # ------------------------------------------------------------ cleanup
    def kill_everything(self, phase: str):
        """Kill every tagged process on every node, terminate local launcher clients, verify."""
        if not self.nodes:
            return
        say(f"cleanup ({phase}): killing this run's processes on {len(self.nodes)} node(s)")
        rep = cleanup_nodes(self.launcher, self.cmds, self.nodes, self.o.cleanup_timeout)
        with self._lock:
            procs = list(self.procs.items())
        exits = {}
        for role, (p, _n, _l, _kind) in procs:
            if p.poll() is None:
                _killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(3)
                except subprocess.TimeoutExpired:
                    _killpg(p.pid)
                    try:
                        p.wait(3)
                    except subprocess.TimeoutExpired:
                        pass
            exits[role] = p.returncode
        rep["slurm"] = self.launcher.extra_cleanup(self.o.tag)
        rep["client_exit_codes"] = exits
        self.cleanup_report = rep
        stray = {k: v for k, v in rep["nodes"].items() if v != "clean"}
        if stray:
            say(f"WARNING: stray processes after cleanup: {stray}")

    # ------------------------------------------------------------ stage 5
    def collect(self):
        logs = sorted(glob.glob(os.path.join(self.logdir, "*.log")))
        tracebacks = {}
        for p in logs:
            try:
                with open(p, errors="replace") as f:
                    c = f.read().count("Traceback (most recent call last)")
            except OSError:
                c = 0
            if c:
                tracebacks[os.path.basename(p)] = c
        hints = {"step creation temporarily disabled": "srun steps could not start concurrently: the job's GPUs "
                 "may not be shareable between steps on this Slurm version; try --srun-args='--gres=gpu:N'",
                 "Address already in use": "a port is taken on some node: pick others with --control-port/"
                 "--anchor-port/--donor-port/--agent-port",
                 "Permission denied (publickey": "ssh needs passwordless keys to every host (--ssh-opts '-i KEY')",
                 "CUDA out of memory": "GPU OOM: lower --batches / --ref-cache-bundles, or use larger GPUs"}
        for p in logs:
            try:
                with open(p, errors="replace") as f:
                    txt = f.read()
            except OSError:
                continue
            for pat, hint in hints.items():
                if pat in txt and hint not in self.notes:
                    self.notes.append(hint + f" (seen in logs/{os.path.basename(p)})")
        missing = [r for r in self.procs if not os.path.exists(os.path.join(self.logdir, f"{r}.log"))]
        stray = {k: v for k, v in (self.cleanup_report.get("nodes") or {}).items() if v != "clean"}
        problems = []
        if missing:
            problems.append(f"missing logs: {missing}")
        if stray:
            problems.append(f"stray processes: {stray}")
        if self.want(5):
            head = f"{len(logs)} logs; cleanup {'clean on all nodes' if not stray and self.cleanup_report else 'NOT CLEAN'}"
            if tracebacks:
                head += f"; tracebacks in {', '.join(sorted(tracebacks))}"
            self.stage(5, "FAIL" if problems else "PASS", head, "; ".join(problems),
                       tracebacks=tracebacks, cleanup=self.cleanup_report)
        write_summary(self, logs)


def cleanup_nodes(launcher, cmds, nodes, timeout) -> dict:
    jobs = {n.name: (lambda n=n: launcher.run(n, cmds.cleanup(n), "cleanup", timeout)) for n in nodes}
    res = parallel(jobs, timeout + 15)
    out = {}
    for n in nodes:
        rc, txt = res.get(n.name) or (1, "")
        if "@@CLEAN" in txt:
            out[n.name] = "clean"
        elif "@@STRAY" in txt:
            out[n.name] = "stray: " + txt.split("@@STRAY", 1)[1].strip()[:500]
        else:
            out[n.name] = f"unknown (rc {rc}): {txt.strip()[-300:]}"
    return {"nodes": out}


def parallel(jobs: dict, timeout: float) -> dict:
    out, threads = {}, []

    def wrap(k, fn):
        try:
            out[k] = fn()
        except Exception as e:     # noqa: BLE001
            out[k] = (1, repr(e))

    for k, fn in jobs.items():
        t = threading.Thread(target=wrap, args=(k, fn), daemon=True)
        t.start()
        threads.append(t)
    end = time.monotonic() + timeout
    for t in threads:
        t.join(max(0.0, end - time.monotonic()))
    return out


def tail_file(path, stop: threading.Event, prefix: str):
    """Echo progress lines of a growing log to the console."""
    pos = 0
    while True:
        try:
            with open(path, errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except OSError:
            chunk = ""
        for ln in chunk.splitlines():
            if ln.startswith("@@RESULT "):
                try:
                    r = json.loads(ln[9:])
                    print(f"{prefix}=> stage {r.get('stage')} {r.get('status')}: {r.get('headline', '')}", flush=True)
                except ValueError:
                    pass
            elif ln.startswith("[") or "Traceback" in ln or "Error" in ln:
                print(prefix + ln[:300], flush=True)
        if stop.is_set():
            return
        time.sleep(0.5)


def parse_driver_log(path, art_dir) -> dict:
    got = {}
    os.makedirs(art_dir, exist_ok=True)
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return got
    for ln in lines:
        if ln.startswith("@@RESULT "):
            try:
                r = json.loads(ln[9:])
                got[int(r["stage"])] = r
            except (ValueError, KeyError):
                pass
        elif ln.startswith("@@ARTIFACT "):
            try:
                _, name, body = ln.split(" ", 2)
                if SAFE.match(name):
                    with open(os.path.join(art_dir, name), "w") as f:
                        f.write(json.dumps(json.loads(body), indent=1) if name.endswith(".json") else json.loads(body))
            except ValueError:
                pass
    return got


def parse_colocate(text: str) -> dict:
    ks = []
    for m in re.finditer(r"kappa\s+([\d.]+)\s+slowdown\s+(-?[\d.]+)% ± ([\d.]+)\s+duty\s+([\d.]+)%\s+call p50 ([\d.]+) ms"
                         r"\s+p99 ([\d.]+) ms", text):
        ks.append({"kappa": float(m[1]), "slowdown_pct": float(m[2]), "sd_pct": float(m[3]),
                   "duty_pct": float(m[4]), "call_p50_ms": float(m[5]), "call_p99_ms": float(m[6])})
    idle = re.search(r"idle second context: slowdown (-?[\d.]+)%", text)
    over = re.search(r"first kappa over the 5% cap: (\S+)", text)
    return {"kappas": ks, "idle_slowdown_pct": float(idle[1]) if idle else None,
            "first_over_cap": over[1] if over else None, "kill_criterion_hit": "KILL CRITERION HIT" in text}


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


# ===================================================================== summary
def write_summary(run: Run, logs: list[str]):
    o = run.o
    res = [run.results[k] for k in sorted(run.results)]
    overall = "PASS" if res and all(r["status"] in ("PASS", "SKIP") for r in res) else "FAIL"
    summ = {"run": {"tag": o.tag, "launcher": o.launcher, "model": o.model, "bundle_size": o.bundle_size,
                    "device_mode": o.device_mode, "started": datetime.datetime.fromtimestamp(run.t0).isoformat(timespec="seconds"),
                    "duration_s": round(time.time() - run.t0, 1), "git": git_rev(REPO),
                    "nodes": [n.name for n in run.nodes], "donors": len(run.donors), "overall": overall,
                    "argv": sys.argv[1:]},
            "stages": res, "notes": run.notes, "cleanup": run.cleanup_report,
            "logs": [os.path.relpath(p, o.out) for p in logs]}
    with open(os.path.join(o.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=str)
    L = [f"# Enkhora cluster validation — {o.tag}", "",
         f"- overall: **{overall}**", f"- launcher: {o.launcher}; nodes: {', '.join(n.name for n in run.nodes)}",
         f"- model: {o.model} (bundle size {o.bundle_size}); device mode: {o.device_mode}; donors: {len(run.donors)}",
         f"- git: {summ['run']['git'].get('sha', '-')}{' (dirty)' if summ['run']['git'].get('dirty') else ''}; "
         f"duration {summ['run']['duration_s']} s", "",
         "| Stage | Status | Headline | Notes |", "|---|---|---|---|"]
    for r in res:
        L.append(f"| {r['stage']} {r['name']} | **{r['status']}** | {md(r.get('headline'))} | {md(r.get('reason'))} |")
    for k in sorted(set(STAGES) - set(run.results)):
        L.append(f"| {k} {STAGES[k]} | not run | | |")
    L.append("")
    if run.notes:
        L += ["## Notes", ""] + [f"- {md(n)}" for n in run.notes] + [""]
    r0 = run.results.get(0)
    if r0 and r0.get("nodes"):
        L += ["## Inventory", "", "| node | addr | gpus | gpu | MiB | driver | torch | cuda | ib | git | code |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in r0["nodes"]:
            L.append("| " + " | ".join(md(fmt(x.get(k))) for k in
                                       ("node", "addr", "gpus", "gpu", "mem_mib", "driver", "torch", "cuda", "ib",
                                        "git", "code")) + " |")
        L.append("")
    r1 = run.results.get(1)
    if r1 and r1.get("links"):
        L += ["## Fabric (stage 1)", "", "| donor | node | gpu | remote | ping p50/p99 ms | payload p50 ms by size | "
              "exec rtt p50/p99 ms (net of kernel) | fetch MB/s |", "|---|---|---|---|---|---|---|---|"]
        for x in r1["links"]:
            pl = ", ".join(f"{k}: {fmt(v.get('p50'), 3)}" for k, v in (x.get("payload") or {}).items())
            ex = ", ".join(f"T{k}: {fmt(v.get('p50'), 3)}/{fmt(v.get('p99'), 3)} ({fmt(v.get('net_p50'), 3)})"
                           for k, v in (x.get("exec") or {}).items())
            L.append(f"| {x['id']} | {x['node']} | {md(x['gpu'])} | {x['remote']} | "
                     f"{fmt((x.get('ping') or {}).get('p50'), 3)}/{fmt((x.get('ping') or {}).get('p99'), 3)} | "
                     f"{pl} | {ex} | {fmt(x.get('fetch_MBps'), 0)} |")
        if r1.get("groups"):
            L += ["", "| group | links | ping p50 | ping p99 | largest payload p50 | largest payload p99 |",
                  "|---|---|---|---|---|---|"]
            for g, v in r1["groups"].items():
                L.append(f"| {md(g)} | {v['links']} | {fmt(v['ping_p50'], 3)} | {fmt(v['ping_p99'], 3)} | "
                         f"{fmt(v.get('payload_p50'), 3)} | {fmt(v.get('payload_p99'), 3)} |")
        L.append("")
    r2 = run.results.get(2)
    if r2 and r2.get("sweep"):
        L += ["## Distributed decode (stage 2)", "",
              "| batch | new tok | tok/s dist | tok/s local | layer ms p50/p95/p99 | tokens eq | logits | max abs diff "
              "| served_by | hedges | link errs |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in r2["sweep"]:
            L.append(f"| {x['batch']} | {x['max_new']} | {fmt(x['tok_s_dist'], 1)} | {fmt(x.get('tok_s_local'), 1)} | "
                     f"{fmt(x['layer_ms']['p50'], 3)}/{fmt(x['layer_ms']['p95'], 3)}/{fmt(x['layer_ms']['p99'], 3)} | "
                     f"{x.get('tokens_equal')} | {x.get('logits')} | {fmt(x.get('max_abs_logit_diff'), 6)} | "
                     f"{md(json.dumps(x.get('served_by')))} | {x.get('hedges', 0)} | {x.get('link_errors', 0)} |")
        L.append("")
        if r2.get("placement"):
            L += [f"Placement: {md(json.dumps(r2['placement']))}", ""]
    r3 = run.results.get(3)
    if r3 and r3.get("events"):
        L += ["## Chaos (stage 3)", "", "| layer call | event | result |", "|---|---|---|"]
        for e in r3["events"]:
            L.append(f"| {e.get('layer')} | {md(e['event'])} | {md(e.get('result'))} |")
        L += ["", f"Compare: {md(json.dumps(r3.get('compare')))}", "",
              f"Dispatcher: {md(json.dumps(r3.get('dispatcher')))}", ""]
    r4 = run.results.get(4)
    if r4 and r4.get("runs"):
        L += ["## Co-location (stage 4)", "", "| gpu | node | kappa | slowdown % | ± | duty % | call p50 ms | p99 ms |",
              "|---|---|---|---|---|---|---|---|"]
        for x in r4["runs"]:
            for k in x["kappas"]:
                L.append(f"| {md(x['gpu'])} | {x['node']} | {k['kappa']} | {k['slowdown_pct']} | {k['sd_pct']} | "
                         f"{k['duty_pct']} | {k['call_p50_ms']} | {k['call_p99_ms']} |")
        L.append("")
    L += ["## Files", "", "- `summary.json` (everything above, machine-readable), `plan.json`, `inventory.json`, "
          "`launch.json`", "- `logs/<role>.log`: every role's stdout/stderr (streamed back through the launcher)",
          "- `artifacts/`: decoded text samples and per-link fabric details", ""]
    text = "\n".join(L)
    with open(os.path.join(o.out, "summary.md"), "w") as f:
        f.write(text)
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100, flush=True)
    return overall


def md(s) -> str:
    return "" if s is None else str(s).replace("|", "\\|").replace("\n", " ")


# ===================================================================== print-plan
def print_plan(o):
    run = Run(o)
    run.nodes = run.launcher.discover()
    assign_shorts(run.nodes)
    for n in run.nodes:
        n.info = {}
    if o.device_mode == "auto":
        o.device_mode = "cpu" if o.launcher == "local" else "cuda"
    for n in run.nodes:
        k = o.gpus_per_node or n.slots_hint or (o.local_gpus if o.launcher == "local" else o.fake_gpus)
        n.gpus = k
        n.gpu_names = [("cpu" if o.device_mode == "cpu" else "GPU?")] * k
        n.gpu_mem_mib = [81559.0 if o.device_mode == "cuda" else 0.0] * k
        n.addr = n.addr_hint or (f"<{o.iface} IPv4 of {n.name}>" if o.iface else
                                 ("127.0.0.1" if o.launcher == "local" else n.target.split("@")[-1]))
        n.job = n.job_hint or ""
    run.mixed = False
    run.donors = make_layout(o, run.nodes)
    plan = run.driver_plan()
    P = print
    P(f"# plan for run {o.tag}: launcher={o.launcher} device_mode={o.device_mode} model={o.model}")
    P(f"# repo on nodes: {run.repo}   python on nodes: {run.py}   workdir on nodes: {run.workdir}")
    P(f"# results: {o.out}")
    P("\n## nodes")
    for n in run.nodes:
        P(f"  node {n.idx}: {n.name:<14} target={n.target:<18} addr={n.addr:<22} slots={n.gpus}"
          f"{'   <- control plane + anchor (slot 0)' if n.idx == 0 else ''}")
    P("\n## roles")
    P(f"  control   {run.nodes[0].name}:{o.control_port} (bind {run.bind_host()})")
    P(f"  anchor    {run.nodes[0].name} slot 0, master store server :{o.anchor_port}")
    for n in run.nodes:
        P(f"  agent     {n.name}:{o.agent_port + n.idx}")
    for d in run.donors:
        P(f"  donor     {d.id:<22} {run.nodes[d.node].name} slot {d.slot} :{d.port} job={d.job}")
    P("\n## stage 0 inventory (one per node, captured stdout)")
    for n in run.nodes:
        P("  $ " + shlex.join(run.launcher.argv(n, run.cmds.node_info(n), "info")))
    if o.launcher == "ssh" and o.sync:
        P("\n## ssh --sync (per host)")
        for n in run.nodes:
            P("  $ " + shlex.join(run.launcher.rsync_argv(n, run.repo)))
            P("  $ " + shlex.join(run.launcher.argv(n, run.install_cmd(), "install")))
    P("\n## long-running roles (stdout/stderr -> results/<run>/logs/<role>.log)")
    for role, node, cmd, _kind in run.role_specs(plan):
        argv = run.launcher.argv(node, cmd, role)
        s = shlex.join(argv)
        s = re.sub(r"--plan-b64 \S+", f"--plan-b64 <{len(pack(plan))} chars: plan.json>", s)
        P(f"  [{role}]\n  $ {s}")
    P("\n## chaos signals (stage 3): the anchor driver sends {op: signal, roles, sig} to the agent on the")
    P("## target node over TCP; the agent reads <pid-dir>/<role>.pid and calls kill(2). Equivalent manual command:")
    if run.donors:
        d = next((x for x in run.donors if x.node != 0), run.donors[0])
        n = run.nodes[d.node]
        pid = f"{run.cmds.pid_dir(n)}/{d.id}.pid"
        P("  $ " + shlex.join(run.launcher.argv(n, f"kill -STOP $(cat {q(pid)})", "sig")))
    P("\n## stage 4 colocate (one GPU per distinct GPU model; example for node 1 slot 0)")
    if len(run.nodes) > 1 or run.donors:
        n = run.nodes[1] if len(run.nodes) > 1 else run.nodes[0]
        inner = run.cmds.tagged_py("colocate") + [f"{run.repo}/bench/colocate.py", "--device", "cuda:0",
                                                  "--windows", str(o.colocate_windows), "--window-s",
                                                  str(o.colocate_window_s)]
        P("  $ " + shlex.join(run.launcher.argv(n, run.cmds.role(n, "colocate", inner, slot=0,
                                                                 gpu_mode=o.device_mode), "colocate")))
    P("\n## cleanup (per node; also run by the bash EXIT trap via `cluster.py cleanup <results dir>`)")
    for n in run.nodes:
        P("  $ " + shlex.join(run.launcher.argv(n, run.cmds.cleanup(n), "cleanup")))
    if o.launcher == "slurm":
        P(f"  $ squeue --steps --jobs=$SLURM_JOB_ID --noheader --format='%i %j'   # then scancel steps named enk-{o.tag}-*")
    P("\n## driver plan (plan.json passed to the anchor)")
    P(json.dumps(plan, indent=1))


# ===================================================================== node-side subcommands
def git_rev(repo) -> dict:
    rc, sha = run_capture(["git", "-C", repo, "rev-parse", "HEAD"], 10)
    if rc == 0:
        rc2, st = run_capture(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"], 10)
        return {"sha": sha.strip(), "dirty": bool(st.strip()) if rc2 == 0 else None, "source": "git"}
    try:
        with open(os.path.join(repo, ".enkhora_rev")) as f:
            return {**json.loads(f.read()), "source": ".enkhora_rev"}
    except (OSError, ValueError):
        return {"sha": None, "dirty": None, "source": "none"}


def code_fingerprint(repo) -> str:
    files = sorted(set(glob.glob(os.path.join(repo, "enkhora", "**", "*.py"), recursive=True) +
                       glob.glob(os.path.join(repo, "scripts", "*.py")) + glob.glob(os.path.join(repo, "scripts", "*.sh")) +
                       glob.glob(os.path.join(repo, "bench", "*.py")) + [os.path.join(repo, "pyproject.toml")]))
    h = hashlib.sha256()
    for p in files:
        if os.path.isfile(p):
            h.update(os.path.relpath(p, repo).encode())
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def _iface_ip(iface: str) -> str | None:
    if shutil.which("ip"):
        rc, out = run_capture(["ip", "-o", "-4", "addr", "show", "dev", iface], 10)
        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", out) if rc == 0 else None
        if m:
            return m[1]
    if shutil.which("ifconfig"):
        rc, out = run_capture(["ifconfig", iface], 10)
        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", out) if rc == 0 else None
        if m:
            return m[1]
    return None


def _primary_ip() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return None


def cmd_node_info(a):
    info = {"node": a.node, "hostname": socket.gethostname(), "fqdn": socket.getfqdn(),
            "python": sys.version.split()[0], "executable": sys.executable, "repo": REPO, "time": now_iso(),
            "git": git_rev(REPO), "code_hash": code_fingerprint(REPO), "primary_ip": _primary_ip(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm": {k: v for k, v in os.environ.items() if k in ("SLURM_JOB_ID", "SLURM_STEP_ID",
                                                                   "SLURM_JOB_GPUS", "SLURM_STEP_GPUS")}}
    if a.iface:
        info["iface"], info["iface_ip"] = a.iface, _iface_ip(a.iface)
    gpus = []
    if shutil.which("nvidia-smi"):
        rc, out = run_capture(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,uuid,pci.bus_id",
                               "--format=csv,noheader,nounits"], 30)
        if rc == 0:
            for ln in out.strip().splitlines():
                f = [x.strip() for x in ln.split(",")]
                if len(f) >= 6:
                    gpus.append({"index": int(f[0]), "name": f[1], "memory_mib": float(f[2]), "driver": f[3],
                                 "uuid": f[4], "pci": f[5]})
        rc, topo = run_capture(["nvidia-smi", "topo", "-m"], 30)
        info["topo"] = topo[:6000] if rc == 0 else None
    info["nvidia_smi_gpus"] = gpus
    info["driver"] = gpus[0]["driver"] if gpus else None
    vis = info["cuda_visible_devices"]
    if gpus and vis is not None:
        sel = []
        for e in [x.strip() for x in vis.split(",") if x.strip()]:
            m = next((g for g in gpus if g["uuid"] == e or (e.isdigit() and g["index"] == int(e))), None)
            sel.append(m or {"index": e, "name": gpus[0]["name"], "memory_mib": gpus[0]["memory_mib"]})
        info["visible_gpus"] = sel
    else:
        info["visible_gpus"] = gpus
    try:
        import torch

        t = {"version": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
             "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0}
        try:
            import transformers

            t["transformers"] = transformers.__version__
        except Exception as e:  # noqa: BLE001
            t["transformers"] = f"import failed: {e!r}"
    except Exception as e:  # noqa: BLE001
        t = {"version": None, "error": repr(e)}
    info["torch"] = t
    if t.get("cuda_available") and t.get("device_count") is not None and info["visible_gpus"]:
        info["visible_gpus"] = info["visible_gpus"][:t["device_count"]]
    try:
        import pynvml  # noqa: F401

        info["pynvml"] = True
    except Exception:  # noqa: BLE001
        info["pynvml"] = False
    ib = []
    for rate in sorted(glob.glob("/sys/class/infiniband/*/ports/*/rate")):
        try:
            with open(rate) as f:
                r = f.read().strip()
            with open(os.path.join(os.path.dirname(rate), "state")) as f:
                s = f.read().strip()
            ll = ""
            llp = os.path.join(os.path.dirname(rate), "link_layer")
            if os.path.exists(llp):
                with open(llp) as f:
                    ll = f.read().strip()
            dev = rate.split("/")[4]
            ib.append(f"{dev}:{r}:{s.split(':')[-1].strip()}:{ll}")
        except OSError:
            pass
    info["ib_ports"] = ib
    info["ib_summary"] = "; ".join(x for x in ib if "ACTIVE" in x) or ("none active" if ib else None)
    if shutil.which("ibstat"):
        rc, out = run_capture(["ibstat"], 20)
        info["ibstat"] = out[:4000]
    if shutil.which("ip"):
        rc, out = run_capture(["ip", "-o", "-4", "addr"], 10)
        info["ip_addr"] = out[:4000]
    elif shutil.which("ifconfig"):
        rc, out = run_capture(["ifconfig"], 10)
        info["ip_addr"] = "\n".join(ln for ln in out.splitlines() if "inet " in ln or not ln.startswith(("\t", " ")))[:4000]
    print("@@NODEINFO " + json.dumps(info), flush=True)


def _resolve_nvml_index(sel: str) -> tuple[int | None, str]:
    """NVML index of the GPU that CUDA_VISIBLE_DEVICES=sel selects, matched by UUID."""
    try:
        import pynvml
    except Exception as e:  # noqa: BLE001
        return None, f"pynvml unavailable ({e!r})"
    uuid = sel if sel.startswith(("GPU-", "MIG-")) else None
    if uuid is None:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=sel)
        try:
            out = subprocess.run([sys.executable, "-c", "import torch;print(torch.cuda.get_device_properties(0).uuid)"],
                                 env=env, capture_output=True, text=True, timeout=120)
            uuid = "GPU-" + out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
        except Exception:  # noqa: BLE001
            uuid = None
    if uuid is None:
        return None, "could not read the device UUID"
    try:
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            u = pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i))
            u = u.decode() if isinstance(u, bytes) else u
            if u == uuid:
                return i, f"uuid {uuid}"
    except Exception as e:  # noqa: BLE001
        return None, f"nvml error {e!r}"
    return None, f"uuid {uuid} not found by NVML"


def cmd_role(a):
    cmd = list(a.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd or not SAFE.match(a.name):
        raise SystemExit("role: need a safe --name and a command after --")
    os.makedirs(a.pid_dir, exist_ok=True)
    try:
        os.setsid()                    # lead a process group: cleanup kills -PGID (children included)
    except OSError:
        pass                           # already a group leader (pgid == pid)
    env = dict(os.environ)
    if a.slot is not None and a.gpu_mode == "cuda":
        vis = env.get("CUDA_VISIBLE_DEVICES")
        ents = [x.strip() for x in vis.split(",") if x.strip()] if vis else None
        if ents is not None and a.slot >= len(ents):
            raise SystemExit(f"role {a.name}: slot {a.slot} but CUDA_VISIBLE_DEVICES={vis}")
        sel = ents[a.slot] if ents is not None else str(a.slot)
        env["CUDA_VISIBLE_DEVICES"] = sel
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print(f"[role] {a.name}: slot {a.slot} -> CUDA_VISIBLE_DEVICES={sel} (inherited {vis})", flush=True)
        if a.nvml:
            idx, why = _resolve_nvml_index(sel)
            if idx is None:
                print(f"[role] {a.name}: NVML watchdog disabled, falling back to the simulated probe: {why}", flush=True)
            else:
                cmd += ["--nvml", str(idx)]
                print(f"[role] {a.name}: NVML index {idx} ({why})", flush=True)
    pidfile = os.path.join(a.pid_dir, f"{a.name}.pid")
    with open(pidfile + ".tmp", "w") as f:
        f.write(str(os.getpid()))
    os.replace(pidfile + ".tmp", pidfile)
    if a.ttl > 0:
        signal.alarm(a.ttl)            # survives exec: a hard lifetime cap even if every cleanup fails
    print(f"[role] {a.name}: pid {os.getpid()} on {socket.gethostname()}: {shlex.join(cmd)[:400]}", flush=True)
    os.execve(cmd[0], cmd, env)


def agent_signal(pid_dir: str, tag: str, roles: list | None, prefix: str | None, sig: str) -> dict:
    if sig not in SIGNALS:
        return {"ok": False, "error": f"signal {sig} not allowed"}
    names = list(roles or [])
    if prefix:
        names += [os.path.basename(p)[:-4] for p in glob.glob(os.path.join(pid_dir, "*.pid"))
                  if os.path.basename(p).startswith(prefix)]
    done, errors = {}, {}
    for r in names:
        if not SAFE.match(r):
            errors[r] = "bad role name"
            continue
        try:
            with open(os.path.join(pid_dir, f"{r}.pid")) as f:
                pid = int(f.read().strip())
            ps = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
            if f"enk_run={tag}" not in ps.stdout:
                errors[r] = "not running"
                continue
            os.kill(pid, getattr(signal, "SIG" + sig))
            done[r] = pid
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            errors[r] = repr(e)
    return {"ok": not errors or bool(done), "signaled": done, "errors": errors}


def cmd_agent(a):
    import asyncio

    sys.path.insert(0, REPO)
    from enkhora.transport import serve

    async def main():
        async def on_msg(conn, h, t):
            op = h.get("op")
            if op == "ping":
                await conn.send({"rid": h.get("rid"), "ok": True, "node": a.node})
            elif op == "signal":
                res = await asyncio.get_running_loop().run_in_executor(
                    None, agent_signal, a.pid_dir, a.tag, h.get("roles"), h.get("prefix"), str(h.get("sig")))
                print(f"[agent] {a.node}: {h.get('sig')} {h.get('roles') or h.get('prefix')} -> {res}", flush=True)
                await conn.send({"rid": h.get("rid"), **res})
            else:
                await conn.send({"rid": h.get("rid"), "ok": False, "error": "unknown op"})

        await serve(a.host, a.port, on_msg)
        print(f"[agent] {a.node} listening on {a.host}:{a.port} (pid dir {a.pid_dir})", flush=True)
        await asyncio.Event().wait()

    asyncio.run(main())


def cmd_cleanup(a):
    """Rebuild the launcher from launch.json and kill everything the run left behind."""
    path = os.path.join(a.run_dir, "launch.json")
    if not os.path.exists(path):
        print(f"cleanup: no {path}; nothing launched")
        return
    with open(path) as f:
        reg = json.load(f)
    o = argparse.Namespace(launcher=reg["launcher"], tag=reg["tag"], hosts=None, nodelist=None, print_plan=False,
                           jobid=reg.get("jobid"), srun_args=reg.get("srun_args"), srun_cpus=reg.get("srun_cpus"),
                           ssh_opts=reg.get("ssh_opts") if isinstance(reg.get("ssh_opts"), str) else None,
                           local_nodes=0, local_gpus=0, env=[], ttl=0, iface=None)
    launcher = {"local": LocalLauncher, "ssh": SshLauncher, "slurm": SlurmLauncher}[o.launcher](o)
    if o.launcher == "ssh" and isinstance(reg.get("ssh_opts"), list):
        launcher.opts = reg["ssh_opts"]
    nodes = [Node(n["idx"], n["name"], n["target"]) for n in reg["nodes"]]
    cmds = Commands(o, reg["repo"], reg["python"], reg["workdir"])
    rep = cleanup_nodes(launcher, cmds, nodes, a.timeout)
    rep["slurm"] = launcher.extra_cleanup(o.tag)
    print(json.dumps(rep, indent=1))
    open(os.path.join(a.run_dir, ".cleanup_done"), "w").close()


# ===================================================================== CLI
def ints(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if x.strip()]


def add_run_args(p):
    g = p.add_argument_group("launcher")
    g.add_argument("--launcher", choices=["auto", "local", "slurm", "ssh"], default="auto",
                   help="auto: slurm if SLURM_JOB_ID is set, ssh if --hosts is given, else local")
    g.add_argument("--hosts", help="ssh hostfile: '[user@]host [slots=N] [addr=IP] [job=ID]' per line; first = node 0")
    g.add_argument("--nodelist", help="slurm: node list expression (default $SLURM_JOB_NODELIST)")
    g.add_argument("--jobid", help="slurm: job id when not running inside the allocation")
    g.add_argument("--srun-args", default="", help="extra srun args for every step, e.g. '--gres=gpu:4'")
    g.add_argument("--srun-cpus", help="--cpus-per-task for each step (default $SLURM_CPUS_PER_TASK if set)")
    g.add_argument("--ssh-opts", default="", help="extra ssh options, e.g. '-i ~/.ssh/key -p 2222'")
    g.add_argument("--sync", action="store_true", help="ssh: rsync the repo and install .venv on every host first")
    g.add_argument("--sync-timeout", type=float, default=1800)
    g.add_argument("--remote-repo", help="repo path on the nodes (default: this checkout's path)")
    g.add_argument("--remote-python", help="python on the nodes (default: this interpreter, or <repo>/.venv with --sync)")
    g.add_argument("--remote-workdir", help="per-run scratch on the nodes (pid files, tiny masters)")
    g.add_argument("--local-nodes", type=int, default=3, help="local: number of pretend nodes")
    g.add_argument("--local-gpus", type=int, default=2, help="local: pretend GPU slots per node")
    g.add_argument("--env", action="append", help="K=V exported to every role (repeatable)")
    g = p.add_argument_group("layout and network")
    g.add_argument("--gpus-per-node", type=int, help="override the GPU count found by the inventory")
    g.add_argument("--no-anchor-node-donors", action="store_true", help="no donors on node 0's other GPUs")
    g.add_argument("--max-donors", type=int)
    g.add_argument("--jobs", choices=["node", "gpu", "single"], default="node",
                   help="anti-affinity job id per donor: one per node (default), per GPU, or one for all")
    g.add_argument("--iface", help="advertise this interface's IPv4 (e.g. ib0) instead of the hostname")
    g.add_argument("--advertise", choices=["name", "ip"], default="name")
    g.add_argument("--bind-host", help="listen address (default 0.0.0.0; local: 127.0.0.1)")
    g.add_argument("--control-port", type=int, default=7400)
    g.add_argument("--anchor-port", type=int, default=7401)
    g.add_argument("--donor-port", type=int, default=7410, help="donor k listens on this + k")
    g.add_argument("--agent-port", type=int, default=7470, help="agent on node i listens on this + i")
    g = p.add_argument_group("model and placement")
    g.add_argument("--model", help=f"tiny | tiny-olmoe | HF id (default: tiny locally, else {DEFAULT_HF_MODEL})")
    g.add_argument("--bundle-size", type=int, default=8)
    g.add_argument("--dtype", help="default bfloat16 for HF models on GPU")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--master-dir", help="master store dir on node 0 (reused when it exists)")
    g.add_argument("--device-mode", choices=["auto", "cuda", "cpu"], default="auto")
    g.add_argument("--expert-demand", type=float, help="planner GPU-equivalents (default 0.75 x donors)")
    g.add_argument("--spread-g", type=int, default=2)
    g.add_argument("--max-replicas", type=int, default=3)
    g.add_argument("--nvml", choices=["auto", "on", "off"], default="auto", help="NVML watchdog on GPU donors")
    g.add_argument("--guard-mib", type=float)
    g.add_argument("--mem-mib", type=float, help="donor memory budget (default 85%% of the GPU)")
    g.add_argument("--ref-cache-bundles", type=int, help="bundles the all-local reference keeps on the anchor GPU")
    g = p.add_argument_group("workload")
    g.add_argument("--stages", default="0,1,2,3,4,5")
    g.add_argument("--batches", help="batch-size sweep (default 1,8,32,64; local 1,4,8)")
    g.add_argument("--max-new", type=int, help="new tokens per sweep run (default 32; local 16)")
    g.add_argument("--correctness-batch", type=int, default=8)
    g.add_argument("--chaos-batch", type=int, default=8)
    g.add_argument("--chaos-max-new", type=int, help="tokens in the chaos generation (default 64; tiny 96)")
    g.add_argument("--audit-atol", type=float, default=0.1,
                   help="max |logit diff| allowed when GPU generations are mixed (bitwise otherwise)")
    g.add_argument("--fabric-tokens", default="1,8,32,64")
    g.add_argument("--fabric-bytes-per-token", type=int, default=14336, help="design: 7168 x bf16")
    g.add_argument("--fabric-iters", type=int, help="round trips per measurement (default 200; local 40)")
    g.add_argument("--colocate-windows", type=int, default=4)
    g.add_argument("--colocate-window-s", type=float, default=1.5)
    g.add_argument("--colocate-args", default="", help="extra args for bench/colocate.py (e.g. --host-cmd ...)")
    g.add_argument("--no-colocate-cpu-smoke", dest="colocate_cpu_smoke", action="store_false",
                   help="in CPU mode, SKIP stage 4 instead of a CPU smoke run of the harness")
    g = p.add_argument_group("timeouts (seconds)")
    g.add_argument("--inventory-timeout", type=float, default=180)
    g.add_argument("--startup-timeout", type=float, default=300, help="donors reachable")
    g.add_argument("--placement-timeout", type=float, help="default 1800; tiny 90")
    g.add_argument("--driver-timeout", type=float, help="whole anchor driver (default 7200; tiny 900)")
    g.add_argument("--colocate-timeout", type=float, default=1800)
    g.add_argument("--cleanup-timeout", type=float, default=45)
    g.add_argument("--ttl", type=float, help="hard lifetime of every launched process (default driver timeout + 1h)")
    g = p.add_argument_group("output")
    g.add_argument("--out", help="results dir (default results/cluster-<timestamp>)")
    g.add_argument("--print-plan", action="store_true", help="print the exact commands and exit (launches nothing)")
    g.add_argument("--fake-gpus", type=int, default=4, help="--print-plan: GPUs per node when not in the hostfile")
    g.add_argument("--tag", help=argparse.SUPPRESS)


def finalize(o):
    o.tag = o.tag or time.strftime("%Y%m%d-%H%M%S") + f"-{random.randrange(16 ** 4):04x}"
    if not SAFE.match(o.tag):
        raise SystemExit("bad --tag")
    o.out = os.path.abspath(o.out or os.path.join(REPO, "results", f"cluster-{o.tag}"))
    local = o.launcher == "local" or (o.launcher == "auto" and not os.environ.get("SLURM_JOB_ID")
                                      and not o.nodelist and not o.hosts)
    o.model = o.model or ("tiny" if local else DEFAULT_HF_MODEL)
    tiny = o.model.startswith("tiny")
    o.stage_set = set(ints(o.stages))
    o.device_mode_requested = o.device_mode
    o.batches = ints(o.batches or ("1,4,8" if local else "1,8,32,64"))
    o.max_new = o.max_new or (16 if local else 32)
    o.chaos_max_new = o.chaos_max_new or (96 if tiny else 64)
    o.fabric_tokens = ints(o.fabric_tokens)
    o.fabric_iters = o.fabric_iters or (40 if local else 200)
    o.placement_timeout = o.placement_timeout or (90 if tiny else 1800)
    o.driver_timeout = o.driver_timeout or (900 if tiny else 7200)
    o.ttl = o.ttl or (o.driver_timeout + o.colocate_timeout + 3600)
    if o.dtype is None and not tiny and o.device_mode != "cpu" and not local:
        o.dtype = "bfloat16"
    return o


def main(argv=None):
    p = argparse.ArgumentParser(prog="cluster.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="sub", required=True)
    r = sub.add_parser("run", help="run the cluster validation")
    add_run_args(r)
    c = sub.add_parser("cleanup", help="kill everything a run launched")
    c.add_argument("run_dir")
    c.add_argument("--timeout", type=float, default=45)
    n = sub.add_parser("node-info")
    n.add_argument("--node", default="")
    n.add_argument("--iface")
    ag = sub.add_parser("agent")
    ag.add_argument("--host", default="0.0.0.0")
    ag.add_argument("--port", type=int, required=True)
    ag.add_argument("--pid-dir", required=True)
    ag.add_argument("--tag", required=True)
    ag.add_argument("--node", default="")
    ro = sub.add_parser("role")
    ro.add_argument("--tag", required=True)
    ro.add_argument("--name", required=True)
    ro.add_argument("--pid-dir", required=True)
    ro.add_argument("--ttl", type=int, default=0)
    ro.add_argument("--slot", type=int)
    ro.add_argument("--gpu-mode", default="cpu")
    ro.add_argument("--nvml", action="store_true")
    ro.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    if a.sub == "node-info":
        return cmd_node_info(a)
    if a.sub == "agent":
        return cmd_agent(a)
    if a.sub == "role":
        return cmd_role(a)
    if a.sub == "cleanup":
        return cmd_cleanup(a)
    o = finalize(a)
    if o.print_plan:
        return print_plan(o)
    run = Run(o)

    def on_signal(sig, _frame):
        raise KeyboardInterrupt(f"signal {sig}")

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)
    atexit.register(run.finish)
    rc = 1
    try:
        run.execute()
        rc = 0
    except KeyboardInterrupt as e:
        run.notes.append(f"interrupted ({e}); partial results")
        say(f"interrupted ({e}); cleaning up")
        rc = 130
    except SystemExit as e:
        run.notes.append(f"aborted: {e}")
        say(f"aborted: {e}")
        rc = 2
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        run.notes.append(f"orchestrator error: {e!r}")
        rc = 1
    finally:
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)      # do not interrupt the cleanup itself
        run.finish()
    if rc == 0:
        res = run.results.values()
        rc = 0 if res and all(r["status"] in ("PASS", "SKIP") for r in res) else 3
    return rc


if __name__ == "__main__":
    sys.exit(main() or 0)
