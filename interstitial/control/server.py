"""Control plane: donor registry, planner, routing table. Never on the per-token path.

If it dies, decode continues on the last routing table and only placement freezes.

Connections are typed by their first message:
  hello_donor   a donor (then pushes status; receives assign / set_kappa / ...)
  hello_anchor  an anchor (sends the bundle manifest; receives routing epochs)
  admin         one request/response command from the ctl CLI
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field

from ..routing import RoutingTable
from ..transport import Conn, serve, server_port
from .planner import ExpSurvival, PBundle, PDonor, PlannerConfig, plan

log = logging.getLogger(__name__)


@dataclass
class DonorRec:
    id: str
    conn: Conn
    job: str
    node: str
    exec_addr: tuple[str, int]
    kappa: float
    mem: int
    life_s: float
    limit_s: float
    joined: float
    held: dict[int, str] = field(default_factory=dict)
    last_seen: float = 0.0
    leaving: bool = False
    stats: dict = field(default_factory=dict)

    def survival(self, now: float) -> ExpSurvival:
        age = now - self.joined
        return ExpSurvival(mean_s=self.life_s, cap_s=max(0.0, self.limit_s - age))


class ControlPlane:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, cfg: PlannerConfig | None = None,
                 heartbeat_timeout_s: float = 3.0, drain_s: float = 0.2, debounce_s: float = 0.05):
        self.host, self.port = host, port
        self.cfg = cfg or PlannerConfig()
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.drain_s = drain_s
        self.debounce_s = debounce_s
        self.donors: dict[str, DonorRec] = {}
        self.anchors: dict[Conn, dict] = {}
        self.bundles: dict[int, dict] = {}       # manifest entries
        self.master_addr: tuple[str, int] | None = None
        self.placement: dict[int, list[str]] = {}
        self.roles: dict[str, dict[int, str]] = {}
        self.prev_primary: dict[int, str] = {}
        self.quarantined: set[str] = set()
        self.table = RoutingTable()
        self.events: list[tuple[float, str]] = []
        self.counters = {"replans": 0, "loads_issued": 0, "bytes_issued": 0, "drops_issued": 0, "epochs": 0}
        self._replan = None

    # ------------------------------------------------------------ lifecycle
    async def start(self):
        self._replan = asyncio.Event()
        self.server = await serve(self.host, self.port, self._on_message, self._on_close)
        self.port = server_port(self.server)
        loop = asyncio.get_running_loop()
        self._tasks = [loop.create_task(self._replan_loop()), loop.create_task(self._reaper())]
        log.info("control plane on %s:%d", self.host, self.port)
        return self

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        self.server.close()
        for d in self.donors.values():
            d.conn.close()
        for c in self.anchors:
            c.close()

    def _event(self, msg: str):
        self.events.append((time.time(), msg))
        del self.events[:-500]
        log.info(msg)

    def kick(self):
        if self._replan is not None:
            self._replan.set()

    # ------------------------------------------------------------ messages
    async def _on_message(self, conn: Conn, h: dict, t: dict):
        op = h.get("op")
        if op == "hello_donor":
            did = h["donor"]
            if did in self.quarantined:
                await conn.send({"op": "evict_all"})
                await conn.send({"op": "exit"})
                return
            old = self.donors.get(did)
            if old and old.conn is not conn:
                old.conn.close()
            now = time.monotonic()
            self.donors[did] = DonorRec(
                id=did, conn=conn, job=h["job"], node=h.get("node", ""), exec_addr=tuple(h["exec"]),
                kappa=float(h["kappa"]), mem=int(h["mem"]),
                life_s=math.inf if h.get("life_s") is None else float(h["life_s"]),
                limit_s=math.inf if h.get("limit_s") is None else float(h["limit_s"]),
                joined=now, held={int(b): s for b, s in h.get("held", {}).items()}, last_seen=now)
            self._event(f"donor {did} joined (job {h['job']}, kappa {h['kappa']}, mem {h['mem'] / 2**20:.1f} MiB)")
            self.kick()
        elif op == "status":
            d = self.donors.get(h["donor"])
            if d is None or d.conn is not conn:
                return
            d.last_seen = time.monotonic()
            held = {int(b): s for b, s in h["held"].items()}
            lost = [b for b in d.held if b not in held]
            changed = held != d.held or d.kappa != h["kappa"] or abs(d.mem - h["mem"]) > 0
            if lost:
                self._event(f"donor {d.id} no longer holds {lost}")
            if d.kappa != h["kappa"]:
                self._event(f"donor {d.id} kappa {d.kappa} -> {h['kappa']}")
            d.held, d.kappa, d.mem, d.stats = held, float(h["kappa"]), int(h["mem"]), h.get("stats", {})
            if changed:
                await self.publish()
                self.kick()
        elif op == "hello_anchor":
            self.anchors[conn] = {"id": h.get("anchor", "anchor")}
            self.master_addr = tuple(h["master"])
            self.bundles = {int(b["id"]): b for b in h["bundles"]}
            self._event(f"anchor {h.get('anchor')} attached: {len(self.bundles)} bundles, master {self.master_addr}")
            await conn.send({"op": "routing", "table": self.table.to_json()})
            self.kick()
        elif op == "quarantine":
            await self.quarantine(h["donor"])
        elif op == "admin":
            await conn.send({"rid": h.get("rid"), **(await self._admin(h))})

    def _on_close(self, conn: Conn):
        for did, d in list(self.donors.items()):
            if d.conn is conn:
                self._donor_gone(did, "connection closed")
        self.anchors.pop(conn, None)

    def _donor_gone(self, did: str, why: str):
        if self.donors.pop(did, None) is None:
            return
        self._event(f"donor {did} gone ({why})")
        asyncio.get_running_loop().create_task(self.publish())
        self.kick()

    async def _reaper(self):
        while True:
            await asyncio.sleep(self.heartbeat_timeout_s / 3)
            now = time.monotonic()
            for did, d in list(self.donors.items()):
                if now - d.last_seen > self.heartbeat_timeout_s:
                    d.conn.close()
                    self._donor_gone(did, "heartbeat timeout")

    async def quarantine(self, did: str):
        self.quarantined.add(did)
        d = self.donors.get(did)
        self._event(f"donor {did} quarantined")
        if d:
            try:
                await d.conn.send({"op": "evict_all"})
                await d.conn.send({"op": "exit"})
            except Exception:
                pass
            self._donor_gone(did, "quarantined")

    # ------------------------------------------------------------ planning
    async def _replan_loop(self):
        while True:
            await self._replan.wait()
            await asyncio.sleep(self.debounce_s)
            self._replan.clear()
            try:
                await self.replan()
            except Exception:
                log.exception("replan failed")

    def _pdonors(self, now: float) -> list[PDonor]:
        return [PDonor(id=d.id, job=d.job, mem=d.mem, kappa=d.kappa, survival=d.survival(now),
                       node=d.node, held=dict(d.held), ending=d.leaving)
                for d in self.donors.values()]

    def _pbundles(self) -> list[PBundle]:
        return [PBundle(b, m["layer"], m["nbytes"], m.get("heat", 1.0)) for b, m in self.bundles.items()]

    async def replan(self):
        if not self.bundles or self.master_addr is None:
            return
        now = time.monotonic()
        p = plan(self._pbundles(), self._pdonors(now), self.cfg, self.prev_primary)
        self.counters["replans"] += 1
        self.placement, self.prev_primary = p.placement, p.primary
        self.roles = p.roles
        drops_later: dict[str, list[int]] = {}
        for did, d in self.donors.items():
            loads = []
            for b, src in p.loads.get(did, []):
                m = self.bundles[b]
                addr = self.donors[src].exec_addr if src in self.donors else self.master_addr
                loads.append({"bundle": b, "nbytes": m["nbytes"], "sha256": m["sha256"],
                              "source": list(addr), "role": p.roles.get(did, {}).get(b, "standby")})
                d.held[b] = "loading"
                self.counters["bytes_issued"] += m["nbytes"]
            self.counters["loads_issued"] += len(loads)
            drops = p.drops.get(did, [])
            if drops:
                drops_later[did] = drops
                for b in drops:
                    d.held.pop(b, None)
            if loads or drops or self.roles.get(did) != p.roles.get(did):
                try:
                    await d.conn.send({"op": "assign", "load": loads,
                                       "roles": {str(b): r for b, r in p.roles.get(did, {}).items()}})
                except Exception:
                    pass
        await self.publish()     # dropped replicas leave the table first (draining) ...
        if drops_later:
            self.counters["drops_issued"] += sum(map(len, drops_later.values()))
            asyncio.get_running_loop().call_later(self.drain_s, lambda: asyncio.ensure_future(self._send_drops(drops_later)))

    async def _send_drops(self, drops: dict[str, list[int]]):
        for did, bs in drops.items():       # ... and are freed after in-flight calls finish
            d = self.donors.get(did)
            if d:
                try:
                    await d.conn.send({"op": "assign", "drop": bs})
                except Exception:
                    pass

    async def publish(self):
        replicas = {}
        for b, ds in self.placement.items():
            ready = tuple(x for x in ds if x in self.donors and self.donors[x].held.get(b) == "ready")
            if ready:
                replicas[b] = ready
        donors = {d: self.donors[d].exec_addr for r in replicas.values() for d in r}
        if replicas == self.table.replicas and donors == self.table.donors:
            return
        self.table = RoutingTable(self.table.epoch + 1, replicas, donors)
        self.counters["epochs"] += 1
        msg = {"op": "routing", "table": self.table.to_json()}
        for c in list(self.anchors):
            try:
                await c.send(msg)
            except Exception:
                self.anchors.pop(c, None)

    # ------------------------------------------------------------ admin
    def status(self) -> dict:
        cov = self.table.coverage(list(self.bundles))
        return {
            "epoch": self.table.epoch, "coverage": cov,
            "tier_bytes": sum(m["nbytes"] for m in self.bundles.values()),
            "donors": {d.id: {"job": d.job, "kappa": d.kappa, "mem": d.mem, "leaving": d.leaving,
                              "ready": sorted(b for b, s in d.held.items() if s == "ready"),
                              "loading": sorted(b for b, s in d.held.items() if s == "loading"),
                              "primary_for": sorted(b for b, r in self.roles.get(d.id, {}).items() if r == "primary"),
                              "stats": d.stats}
                       for d in self.donors.values()},
            "quarantined": sorted(self.quarantined), "counters": dict(self.counters),
            "events": [m for _, m in self.events[-20:]],
        }

    async def _admin(self, h: dict) -> dict:
        cmd = h.get("cmd")
        target = h.get("donor")
        d = self.donors.get(target) if target else None
        if cmd == "status":
            return {"ok": True, "status": self.status()}
        if target and d is None:
            return {"ok": False, "error": f"unknown donor {target}"}
        if cmd == "set_kappa":
            await d.conn.send({"op": "set_kappa", "kappa": float(h["kappa"])})
        elif cmd == "host_alloc":
            await d.conn.send({"op": "host_alloc", "bytes": int(h["bytes"])})
        elif cmd == "leave":
            d.leaving = True
            self._event(f"donor {d.id} leaving: planned drain")
            self.kick()
            asyncio.get_running_loop().create_task(self._finish_leave(d.id, float(h.get("timeout_s", 30))))
        elif cmd == "kill_switch":
            for x in ([d] if d else list(self.donors.values())):
                await x.conn.send({"op": "evict_all"})
            self._event(f"kill switch: {target or 'all donors'}")
        elif cmd == "replan":
            self.kick()
        else:
            return {"ok": False, "error": f"unknown command {cmd}"}
        return {"ok": True}

    async def _finish_leave(self, did: str, timeout_s: float):
        """Planned drain: wait until every bundle the donor holds has another ready replica."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            d = self.donors.get(did)
            if d is None:
                return
            covered = all(any(x != did and x in self.donors and self.donors[x].held.get(b) == "ready"
                              for x in self.placement.get(b, [])) for b in d.held)
            if covered:
                break
            await asyncio.sleep(0.1)
        d = self.donors.get(did)
        if d:
            self.placement = {b: [x for x in ds if x != did] for b, ds in self.placement.items()}
            await self.publish()
            await asyncio.sleep(self.drain_s)
            try:
                await d.conn.send({"op": "evict_all"})
                await d.conn.send({"op": "exit"})
            except Exception:
                pass
