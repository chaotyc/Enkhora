"""A whole system on one machine: control plane (thread), donors (real subprocesses,
so they can be SIGKILLed), anchor (this process). Used by the demo and the tests."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

from .anchor.anchor import Anchor, make_anchor_model
from .anchor.dispatcher import DispatchConfig
from .control.ctl import admin_async
from .control.planner import PlannerConfig
from .control.server import ControlPlane
from .transport import LoopThread


@dataclass
class DonorSpec:
    id: str
    job: str
    kappa: float = 1.0
    mem_mib: float = 4.0            # simulated GPU headroom handed to the donor (M before guard)
    life_s: float | None = None
    limit_s: float | None = None
    corrupt: float = 0.0
    aimd: bool = False
    sensitivity: float = 0.0
    extra: list[str] = field(default_factory=list)


class LocalCluster:
    def __init__(self, workdir: str, donors: list[DonorSpec], model: str = "tiny", bundle_size: int = 8,
                 planner: PlannerConfig | None = None, dispatch: DispatchConfig | None = None,
                 seed: int = 0, threads: int = 1, log_dir: str | None = None,
                 device: str = "cpu", dtype: str | None = None, model_loader=None):
        """``device``/``dtype`` place the anchor (donor devices go in ``DonorSpec.extra``).
        ``model_loader`` is an optional ``() -> (model, MasterStore)`` replacing
        ``make_anchor_model`` (e.g. a model already exported and moved to the GPU)."""
        self.workdir = workdir
        self.specs = donors
        self.model_name, self.bundle_size, self.seed = model, bundle_size, seed
        self.device, self.dtype, self.model_loader = device, dtype, model_loader
        self.planner_cfg = planner or PlannerConfig(copy_rate_Bps=1e12, alpha=1.0)
        self.dispatch_cfg = dispatch or DispatchConfig()
        self.threads = threads
        self.log_dir = log_dir or os.path.join(workdir, "logs")
        self.procs: dict[str, subprocess.Popen] = {}
        self.control: ControlPlane | None = None
        self.control_loop: LoopThread | None = None
        self.anchor: Anchor | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self, wait_replicas: int = 1, timeout: float = 60.0, spawn_donors: bool = True):
        os.makedirs(self.log_dir, exist_ok=True)
        self.start_control()
        if self.model_loader is not None:
            model, master = self.model_loader()
        else:
            model, master = make_anchor_model(self.model_name, os.path.join(self.workdir, "master"),
                                              self.bundle_size, seed=self.seed, device=self.device, dtype=self.dtype)
        self.anchor = Anchor(model, master, control=("127.0.0.1", self.control.port), device=self.device,
                             cfg=self.dispatch_cfg)
        if spawn_donors:
            self.start_donors(wait_replicas, timeout)
        return self

    def start_donors(self, wait_replicas: int = 1, timeout: float = 60.0):
        for s in self.specs:
            self.spawn(s)
        if wait_replicas and not self.anchor.wait_for_coverage(wait_replicas, timeout):
            raise TimeoutError(f"placement did not reach {wait_replicas} replica(s): {self.status()['coverage']}")
        return self

    def start_control(self, port: int = 0):
        self.control_loop = LoopThread("control")
        self.control = ControlPlane(port=port, cfg=self.planner_cfg)
        self.control_loop.call(self.control.start())

    def stop_control(self):
        """Simulate the control plane dying."""
        self.control_loop.call(self.control.stop())
        self.control_loop.stop()

    def spawn(self, s: DonorSpec):
        cmd = [sys.executable, "-m", "enkhora", "donor", "--id", s.id, "--job", s.job,
               "--control", f"127.0.0.1:{self.control.port}", "--kappa", str(s.kappa),
               "--mem-mib", str(s.mem_mib), "--threads", str(self.threads), *s.extra]
        if s.life_s is not None:
            cmd += ["--life-s", str(s.life_s)]
        if s.limit_s is not None:
            cmd += ["--limit-s", str(s.limit_s)]
        if s.corrupt:
            cmd += ["--corrupt", str(s.corrupt)]
        if s.aimd:
            cmd += ["--aimd", "--sensitivity", str(s.sensitivity)]
        log = open(os.path.join(self.log_dir, f"{s.id}.log"), "w")
        self.procs[s.id] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    def kill(self, donor_id: str, sig=signal.SIGKILL):
        p = self.procs[donor_id]
        p.send_signal(sig)
        if sig in (signal.SIGKILL, signal.SIGTERM):
            p.wait(5)

    def stop(self):
        if self.anchor:
            self.anchor.close()
        for p in self.procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGKILL)
                p.wait(5)
        if self.control_loop and not self.control_loop.loop.is_closed() and self.control_loop.thread.is_alive():
            try:
                self.control_loop.call(self.control.stop(), timeout=3)
            except Exception:
                pass
            self.control_loop.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ------------------------------------------------------------ helpers
    def admin(self, cmd: str, **kw) -> dict:
        return self.control_loop.call(admin_async(("127.0.0.1", self.control.port), cmd, **kw), timeout=10)

    def status(self) -> dict:
        return self.admin("status")["status"]

    def wait(self, pred, timeout: float = 20.0, poll: float = 0.05):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            st = self.status()
            if pred(st):
                return st
            time.sleep(poll)
        raise TimeoutError("condition not reached")
