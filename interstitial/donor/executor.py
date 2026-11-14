"""Donor process: one per donor GPU, with its own CUDA context (never MPS).

Holds bundle slabs, executes expert calls as a pure function, keeps no state
between calls, and protects the host with the watchdog and the token bucket.
In v1 the node agent's loops (watchdog, governor) run as threads/tasks inside this
process; splitting them into a privileged per-node agent is a v2 item.

Ops served to anchors and peers:  exec, fetch, ping
Pushes from the control plane:    assign, set_kappa, host_alloc, evict_all, exit
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import time

import torch

from ..experts import BundleWeights, bundle_forward, checksum
from ..transport import RpcClient, serve, server_port
from .governor import AIMD, OnOffEstimator, TokenBucket
from .slabs import SlabManager
from .watchdog import SimulatedProbe, Watchdog

log = logging.getLogger(__name__)


class Donor:
    def __init__(self, donor_id: str, job_id: str, control: tuple[str, int] | None,
                 host: str = "127.0.0.1", port: int = 0, device: str = "cpu",
                 kappa: float = 1.0, life_s: float = math.inf, limit_s: float = math.inf,
                 node: str = "", probe=None, guard_bytes: int = 0, slack_bytes: int = 0,
                 poll_s: float = 0.02, copy_rate_Bps: float = 1e9, status_s: float = 0.5,
                 corrupt: float = 0.0, aimd: bool = False, advertise_host: str | None = None):
        self.id, self.job, self.node = donor_id, job_id, node
        self.control_addr = control
        self.host, self.port = host, port
        self.advertise_host = advertise_host or host
        self.device = torch.device(device)
        self.life_s, self.limit_s = life_s, limit_s
        self.copy_rate = copy_rate_Bps
        self.status_s = status_s
        self.corrupt = corrupt
        self.aimd_enabled = aimd

        self.slabs = SlabManager(device)
        self.bucket = TokenBucket(kappa)
        self.probe = probe or SimulatedProbe(total=1 << 40)
        self.watchdog = Watchdog(self.probe, self.slabs, guard_bytes, slack_bytes, poll_s,
                                 on_evict=lambda _: self._mark_dirty_threadsafe())
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="exec")
        self.failed: set[int] = set()
        self.stats = {"calls": 0, "refused": 0, "missing": 0, "exec_ms": 0.0, "loaded_bytes": 0, "fetch_served": 0}
        self.busy_ms = 0.0
        self._peers: dict[tuple[str, int], RpcClient] = {}
        self._control: RpcClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dirty: asyncio.Event | None = None
        self._exit: asyncio.Event | None = None
        self._load_q: asyncio.Queue | None = None
        self._tasks: list[asyncio.Task] = []

    # ---------------------------------------------------------------- lifecycle
    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._dirty, self._exit, self._load_q = asyncio.Event(), asyncio.Event(), asyncio.Queue()
        self.server = await serve(self.host, self.port, self._on_message)
        self.port = server_port(self.server)
        self.watchdog.start()
        spawn = self._loop.create_task
        self._tasks = [spawn(self._loader()), spawn(self._status_loop())]
        if self.control_addr:
            self._tasks.append(spawn(self._control_loop()))
        if self.aimd_enabled:
            self._tasks.append(spawn(self._governor_loop()))
        log.info("donor %s (job %s) serving on %s:%d", self.id, self.job, self.host, self.port)
        return self

    async def run_forever(self):
        await self.start()
        await self._exit.wait()
        await self.shutdown()

    async def shutdown(self):
        self.watchdog.stop()
        for t in self._tasks:
            t.cancel()
        self.server.close()
        for c in list(self._peers.values()) + ([self._control] if self._control else []):
            await c.close()
        self.slabs.drop_all()

    def _mark_dirty_threadsafe(self):
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._dirty.set)

    # ---------------------------------------------------------------- control plane
    def _hello(self) -> dict:
        return {"op": "hello_donor", "donor": self.id, "job": self.job, "node": self.node,
                "exec": [self.advertise_host, self.port], "kappa": self.bucket.kappa,
                "mem": self.watchdog.budget(), "life_s": _num(self.life_s), "limit_s": _num(self.limit_s),
                "held": {str(b): s for b, s in self.slabs.snapshot().items()},
                "roles": {str(b): r for b, r in self.slabs.roles().items()}}

    async def _control_loop(self):
        host, port = self.control_addr
        while True:
            try:
                self._control = RpcClient(host, port, on_push=self._on_control)
                await self._control.connect(timeout=2.0)
                await self._control.push(self._hello())
                while self._control.connected:
                    await asyncio.sleep(0.2)
            except (OSError, ConnectionError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(1.0)     # control plane down: keep serving, retry

    def _status(self) -> dict:
        return {"op": "status", "donor": self.id, "mem": self.watchdog.budget(), "kappa": self.bucket.kappa,
                "held": {str(b): s for b, s in self.slabs.snapshot().items()},
                "failed": sorted(self.failed), "stats": dict(self.stats),
                "evicted": len(self.watchdog.evicted_total)}

    async def _status_loop(self):
        while True:
            try:
                await asyncio.wait_for(self._dirty.wait(), self.status_s)
            except asyncio.TimeoutError:
                pass
            self._dirty.clear()
            await asyncio.sleep(0.01)   # coalesce bursts
            if self._control and self._control.connected:
                try:
                    await self._control.push(self._status())
                    self.failed.clear()
                except (ConnectionError, OSError):
                    pass

    def _on_control(self, h: dict, t: dict):
        op = h.get("op")
        if op == "assign":
            for b in h.get("drop", []):
                self.slabs.drop(int(b))
            self.slabs.set_roles({int(b): r for b, r in h.get("roles", {}).items()})
            for item in h.get("load", []):
                if self.slabs.reserve(item["bundle"], item["nbytes"], item.get("role", "standby")):
                    self._load_q.put_nowait(item)
        elif op == "set_kappa":
            self.bucket.set_kappa(float(h["kappa"]))
        elif op == "host_alloc":
            if isinstance(self.probe, SimulatedProbe):
                self.probe.host_used = int(h["bytes"])
        elif op == "evict_all":
            self.slabs.drop_all()
        elif op == "exit":
            self._exit.set()
        self._dirty.set()

    # ---------------------------------------------------------------- loading
    async def _peer(self, addr: tuple[str, int]) -> RpcClient:
        c = self._peers.get(addr)
        if c is None or not c.connected:
            c = await RpcClient(*addr).connect(timeout=2.0)
            self._peers[addr] = c
        return c

    async def _loader(self):
        """Sequential, throttled copies. Sources are store donors or the master."""
        while True:
            item = await self._load_q.get()
            b = item["bundle"]
            if self.slabs.snapshot().get(b) != "loading":
                continue
            t0 = time.monotonic()
            try:
                if self.watchdog.budget() < 0:
                    raise MemoryError("no headroom")
                c = await self._peer(tuple(item["source"]))
                h, t = await asyncio.wait_for(c.request({"op": "fetch", "bundle": b}), 60)
                if not h.get("ok"):
                    raise ConnectionError(h.get("error", "fetch refused"))
                w = BundleWeights(b, h["layer"], h["experts"], t["gate_up"].clone(), t["down"].clone())
                if checksum(w.gate_up, w.down) != item["sha256"]:
                    raise ValueError(f"checksum mismatch for bundle {b}")
                if self.slabs.set_ready(b, w):
                    self.stats["loaded_bytes"] += w.nbytes
                wait = w.nbytes / self.copy_rate - (time.monotonic() - t0)
                if wait > 0:
                    await asyncio.sleep(wait)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("donor %s: load of bundle %d failed: %r", self.id, b, e)
                self.slabs.drop(b)
                self.failed.add(b)
            self._dirty.set()

    # ---------------------------------------------------------------- data plane
    async def _on_message(self, conn, h: dict, t: dict):
        op = h.get("op")
        if op == "exec":
            await self._exec(conn, h, t)
        elif op == "fetch":
            w = self.slabs.get_ready(int(h["bundle"]))
            if w is None:
                await conn.send({"rid": h["rid"], "ok": False, "error": "not held"})
            else:
                self.stats["fetch_served"] += 1
                await conn.send({"rid": h["rid"], "ok": True, "layer": w.layer, "experts": w.experts},
                                {"gate_up": w.gate_up, "down": w.down})
        elif op == "ping":
            await conn.send({"rid": h["rid"], "ok": True, "donor": self.id})

    async def _exec(self, conn, h: dict, t: dict):
        arrive = time.monotonic()
        self.stats["calls"] += 1
        if not self.bucket.try_acquire():
            self.stats["refused"] += 1
            reason = "paused" if self.bucket.paused else "budget"
            await conn.send({"rid": h["rid"], "ok": False, "refused": reason,
                             "budget_ms": self.bucket.remaining_ms()})
            return
        present, missing = [], []
        for i, b in enumerate(h["bundles"]):
            w = self.slabs.get_ready(b)
            (present if w is not None else missing).append((i, b, w))
        self.stats["missing"] += len(missing)
        out, offs, wait_ms, exec_ms = await self._loop.run_in_executor(
            self._pool, self._compute, h, t, present, arrive)
        self.bucket.charge(exec_ms)
        self.busy_ms += exec_ms
        self.stats["exec_ms"] += exec_ms
        await conn.send({"rid": h["rid"], "ok": True, "epoch": h.get("epoch"),
                         "done": [b for _, b, _ in present], "missing": [b for _, b, _ in missing],
                         "wait_ms": wait_ms, "exec_ms": exec_ms, "budget_ms": self.bucket.remaining_ms()},
                        {"out": out, "offsets": offs})

    def _compute(self, h, t, present, arrive):
        start = time.monotonic()
        offs = t["offsets"].tolist()
        hidden = t["hidden"].to(self.device)
        rows, experts, gates = t["rows"], t["experts"], t["gates"].to(self.device)
        outs, lens = [], [0]
        for i, b, w in present:
            s, e = offs[i], offs[i + 1]
            _, p = bundle_forward(w, h["act"], hidden, rows[s:e], experts[s:e], gates[s:e], h["chunk_rows"])
            if self.corrupt:
                p = p + self.corrupt        # fault injection: a silently wrong donor
            outs.append(p.cpu())
            lens.append(lens[-1] + p.shape[0])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        D = hidden.shape[1]
        out = torch.cat(outs) if outs else torch.empty(0, D, dtype=hidden.dtype)
        end = time.monotonic()
        return out, torch.tensor(lens, dtype=torch.int64), (start - arrive) * 1e3, (end - start) * 1e3

    # ---------------------------------------------------------------- governor
    async def _governor_loop(self):
        """On/off windows + AIMD on κ. Needs a probe with host_activity()."""
        est, aimd = OnOffEstimator(), AIMD(kappa0=min(self.bucket.kappa, 0.01))
        self.bucket.set_kappa(aimd.kappa)
        while True:
            lend, secs = est.next_window()
            self.bucket.paused = not lend
            busy0, t0, samples = self.busy_ms, time.monotonic(), []
            while time.monotonic() - t0 < secs:
                await asyncio.sleep(0.05)
                el = (time.monotonic() - t0) * 1e3
                self.probe.set_duty((self.busy_ms - busy0) / el if el else 0.0)
                a = self.probe.host_activity()
                if a is not None:
                    samples.append(a)
            self.probe.set_duty(0.0)
            if samples:
                est.record(lend, sum(samples) / len(samples))
            if est.ready() and len(est.on) == len(est.off):
                self.bucket.set_kappa(aimd.update(est.slowdown()))
                self._dirty.set()


def _num(x: float):
    return None if math.isinf(x) else x
