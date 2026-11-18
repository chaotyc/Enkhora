"""Gate-side dispatcher: one scatter-gather round per MoE layer (section 6).

Per layer:
  1. build (token, expert, gate) pairs per bundle from the router's top-k
  2. choose a replica per bundle from the routing table (read once: epochs swap
     only between layers)
  3. scatter one message per donor; a token's hidden row is sent once per donor
  4. run anchor-assigned bundles locally while replies are in flight
  5. gather; on refusal, missing bundle or broken link, escalate at once to the
     next rung; on a missed deadline, hedge to the next rung and keep the original
     alive -- first answer wins
  6. merge partials in ascending bundle id

The last rung is always the anchor itself, so every bundle resolves.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import random
import time
from collections import Counter, deque
from dataclasses import dataclass, field

import torch

from ..experts import build_route, bundle_forward, merge_partials
from ..routing import RoutingTable
from ..store import MasterStore
from ..transport import LoopThread, RpcClient
from .fallback import FallbackStore

log = logging.getLogger(__name__)
ANCHOR = "@anchor"


@dataclass
class DispatchConfig:
    chunk_rows: int = 4096           # kernel-length cap (part of the numeric contract)
    deadline_init_s: float = 0.5
    deadline_min_s: float = 0.02
    deadline_max_s: float = 2.0
    deadline_mult: float = 3.0       # deadline = mult * p95 of the donor's recent RTTs
    reconnect_backoff_s: float = 1.0
    miss_limit: int = 2              # consecutive deadline misses before the circuit opens
    connect_timeout_s: float = 2.0
    low_budget_ms: float = 1.0       # reply feedback: steer away below this remaining budget
    avoid_s: float = 0.05
    audit_rate: float = 0.0          # fraction of remote bundle results recomputed locally
    audit_atol: float = 0.0          # 0 => bitwise (same device class on both sides)
    prefill_local_tokens: int | None = None   # run layers with more tokens than this on the anchor
    seed: int = 0


@dataclass
class DonorLink:
    id: str
    addr: tuple[str, int]
    client: RpcClient | None = None
    connecting: bool = False
    down_until: float = 0.0
    avoid_until: float = 0.0
    quarantined: bool = False
    misses: int = 0
    rtts: deque = field(default_factory=lambda: deque(maxlen=256))

    def usable(self, now: float) -> bool:
        return (not self.quarantined and self.client is not None and self.client.connected
                and now >= self.down_until)

    def deadline(self, cfg: DispatchConfig) -> float:
        if len(self.rtts) < 16:
            return cfg.deadline_init_s
        s = sorted(self.rtts)
        p95 = s[int(0.95 * (len(s) - 1))]
        return min(cfg.deadline_max_s, max(cfg.deadline_min_s, cfg.deadline_mult * p95))


@dataclass
class _Msg:
    donor: str
    bundles: list[int]
    sent: float
    deadline: float
    hedged: bool = False


class Dispatcher:
    def __init__(self, master: MasterStore, fallback: FallbackStore, loop: LoopThread,
                 cfg: DispatchConfig | None = None, on_quarantine=None):
        self.master = master
        self.fallback = fallback
        self.loop = loop
        self.cfg = cfg or DispatchConfig()
        self.act = master.spec.act
        self.e2b = {layer: master.expert_to_bundle(layer) for layer in master.layers}
        self.table = RoutingTable()
        self.links: dict[str, DonorLink] = {}
        self.on_quarantine = on_quarantine
        self.rng = random.Random(self.cfg.seed)
        self.stats = Counter()
        self.rungs = Counter()                 # where each bundle-call was finally served
        self.layer_ms: deque = deque(maxlen=4096)
        self.epochs_seen: set[int] = set()
        self.on_layer = None                   # test/demo hook: called with the layer-call count

    # ------------------------------------------------------------ routing table
    def set_table(self, table: RoutingTable):
        """Called from the loop thread. Takes effect at the next layer boundary."""
        for d, addr in table.donors.items():
            link = self.links.get(d)
            if link is None or link.addr != addr:
                if link and link.client:
                    self.loop.submit(link.client.close())
                self.links[d] = DonorLink(d, addr)
            self._ensure_connected(self.links[d])
        self.table = table

    def _ensure_connected(self, link: DonorLink):
        if link.connecting or link.quarantined or (link.client and link.client.connected):
            return
        if time.monotonic() < link.down_until:
            return
        link.connecting = True

        async def _go():
            try:
                c = RpcClient(*link.addr)
                await c.connect(self.cfg.connect_timeout_s)
                link.client = c
            except Exception:
                link.down_until = time.monotonic() + self.cfg.reconnect_backoff_s
            finally:
                link.connecting = False

        self.loop.submit(_go())

    def _deadline_miss(self, donor: str):
        """Circuit breaker: repeated deadline misses disconnect the donor for a backoff.
        Closing the connection fails its stuck requests instead of leaking them."""
        link = self.links.get(donor)
        if link is None:
            return
        link.misses += 1
        link.avoid_until = time.monotonic() + self.cfg.avoid_s
        if link.misses >= self.cfg.miss_limit and link.client is not None:
            self.stats["circuit_open"] += 1
            link.misses = 0
            link.down_until = time.monotonic() + self.cfg.reconnect_backoff_s
            client, link.client = link.client, None
            self.loop.submit(client.close())

    def _mark_down(self, link: DonorLink):
        link.down_until = time.monotonic() + self.cfg.reconnect_backoff_s
        if link.client is not None and not link.client.connected:
            link.client = None

    # ------------------------------------------------------------ one layer
    @torch.no_grad()
    def run_layer(self, layer: int, hidden: torch.Tensor, topk_idx: torch.Tensor,
                  topk_w: torch.Tensor) -> torch.Tensor:
        if self.on_layer is not None:
            self.on_layer(self.stats["layers"])
        t_start = time.monotonic()
        table = self.table                       # one epoch for the whole layer
        self.epochs_seen.add(table.epoch)
        T, D = hidden.shape
        route = build_route(topk_idx, topk_w, self.e2b[layer])
        hidden_cpu = hidden.detach().cpu() if hidden.device.type != "cpu" else hidden
        local_only = self.cfg.prefill_local_tokens is not None and T > self.cfg.prefill_local_tokens
        now = time.monotonic()

        cands: dict[int, list[str]] = {}
        for b in route.bundles:
            reps = [] if local_only else list(table.replicas_for(b))
            for d in reps:
                if d in self.links:
                    self._ensure_connected(self.links[d])
            ok = [d for d in reps if d in self.links and self.links[d].usable(now)]
            if len(ok) < len(reps):
                self.stats["replicas_skipped"] += len(reps) - len(ok)   # link down / quarantined
            ok.sort(key=lambda d: self.links[d].avoid_until > now)   # stable: soft-avoided last
            cands[b] = ok + [ANCHOR]

        nxt = {b: 0 for b in route.bundles}
        active = Counter()          # un-hedged pending attempts per bundle
        partials: dict[int, torch.Tensor] = {}
        pending: dict[cf.Future, _Msg] = {}

        def escalate(bs: list[int]):
            groups: dict[str, list[int]] = {}
            now = time.monotonic()
            for b in bs:
                if b in partials:
                    continue
                while True:
                    target = cands[b][min(nxt[b], len(cands[b]) - 1)]
                    nxt[b] += 1
                    if target == ANCHOR or self.links[target].usable(now):
                        break
                groups.setdefault(target, []).append(b)
            for target, g in groups.items():
                if target != ANCHOR:
                    self._send(layer, table.epoch, target, g, route, hidden_cpu, pending, active)
            for b in groups.get(ANCHOR, []):
                if b not in partials:
                    partials[b] = self._local(b, route, hidden)

        escalate(list(route.bundles))

        while pending and len(partials) < len(route.bundles):
            now = time.monotonic()
            live = [m.deadline for m in pending.values() if not m.hedged]
            timeout = max(0.0, min(live) - now) if live else None
            done, _ = cf.wait(list(pending), timeout=timeout, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                msg = pending.pop(fut)
                retry = self._handle_reply(fut, msg, route, partials, hidden, active)
                escalate([b for b in retry if active[b] == 0])
            now = time.monotonic()
            for fut, msg in list(pending.items()):
                if not msg.hedged and now >= msg.deadline:
                    msg.hedged = True               # stays pending: it may still win
                    for b in msg.bundles:
                        active[b] -= 1
                    self._deadline_miss(msg.donor)
                    open_ = [b for b in msg.bundles if b not in partials]
                    if open_:
                        self.stats["hedges"] += 1
                        escalate(open_)
            # Invariant guard: every open bundle has a live attempt or is escalated now.
            stuck = [b for b in route.bundles if b not in partials and active[b] == 0]
            if stuck:
                escalate(stuck)
        # pending futures that lost the race are abandoned; their replies are discarded

        out = merge_partials(T, D, hidden.dtype, hidden.device, route, partials)
        self.layer_ms.append((time.monotonic() - t_start) * 1e3)
        self.stats["layers"] += 1
        self.stats["bundle_calls"] += len(route.bundles)
        return out

    def _send(self, layer, epoch, donor, bundles, route, hidden_cpu, pending, active):
        link = self.links[donor]
        U = torch.unique(torch.cat([route.urows[b] for b in bundles]), sorted=True)
        rows, exps, gates, offs = [], [], [], [0]
        for b in bundles:
            r, e, g = route.pairs[b]
            rows.append(torch.searchsorted(U, r).to(torch.int32))
            exps.append(e.to(torch.int32))
            gates.append(g.detach().cpu())
            offs.append(offs[-1] + len(r))
        header = {"op": "exec", "epoch": epoch, "layer": layer, "bundles": bundles,
                  "act": self.act, "chunk_rows": self.cfg.chunk_rows}
        tensors = {"hidden": hidden_cpu.index_select(0, U), "rows": torch.cat(rows), "experts": torch.cat(exps),
                   "gates": torch.cat(gates), "offsets": torch.tensor(offs, dtype=torch.int64)}
        now = time.monotonic()
        fut = self.loop.submit(link.client.request(header, tensors))
        pending[fut] = _Msg(donor, list(bundles), now, now + link.deadline(self.cfg))
        for b in bundles:
            active[b] += 1
        self.stats["messages"] += 1

    def _handle_reply(self, fut, msg: _Msg, route, partials, hidden, active) -> list[int]:
        """Absorb one reply. Returns bundles that need another rung now."""
        if not msg.hedged:
            for b in msg.bundles:
                active[b] -= 1
        link = self.links.get(msg.donor)
        try:
            h, t = fut.result()
        except Exception:
            self.stats["link_errors"] += 1
            if link:
                self._mark_down(link)
            return [b for b in msg.bundles if b not in partials]
        if not h.get("ok"):
            self.stats["refusals"] += 1
            if link:
                link.avoid_until = time.monotonic() + self.cfg.avoid_s
            return [b for b in msg.bundles if b not in partials]
        rtt = time.monotonic() - msg.sent
        if link:
            link.rtts.append(rtt)
            link.misses = 0
            if h.get("budget_ms", 1e9) < self.cfg.low_budget_ms:
                link.avoid_until = time.monotonic() + self.cfg.avoid_s
        offs = t["offsets"].tolist()
        for i, b in enumerate(h["done"]):
            if b in partials:
                self.stats["late_discarded"] += 1
                continue
            p = t["out"][offs[i]:offs[i + 1]]
            if self.cfg.audit_rate and self.rng.random() < self.cfg.audit_rate:
                p = self._audit(msg.donor, b, p, route, hidden)
                if p is None:
                    continue
            partials[b] = p
            self.rungs["donor"] += 1
        self.stats["missing"] += len(h.get("missing", []))
        return [b for b in h.get("missing", []) if b not in partials]

    def _audit(self, donor, b, p, route, hidden):
        """Silent-error guard: recompute on the anchor and compare."""
        self.stats["audits"] += 1
        ref = self._local(b, route, hidden, count=False)
        p_dev = p.to(ref.device, ref.dtype)
        same = torch.equal(p_dev, ref) if self.cfg.audit_atol == 0 else \
            torch.allclose(p_dev, ref, atol=self.cfg.audit_atol, rtol=0)
        if same:
            return p
        self.stats["audit_failures"] += 1
        log.error("audit mismatch from donor %s on bundle %d: quarantining", donor, b)
        link = self.links.get(donor)
        if link and not link.quarantined:
            link.quarantined = True
            if self.on_quarantine:
                self.on_quarantine(donor)
        self.rungs["anchor:audit"] += 1
        return ref

    def _local(self, b, route, hidden, count: bool = True) -> torch.Tensor:
        w, rung = self.fallback.get(b)
        rows, ex, g = route.pairs[b]
        _, p = bundle_forward(w, self.act, hidden, rows, ex, g, self.cfg.chunk_rows)
        if count:
            self.rungs[f"anchor:{rung}"] += 1
        return p

    # ------------------------------------------------------------ reporting
    def report(self) -> dict:
        ms = sorted(self.layer_ms)
        pct = (lambda q: round(ms[int(q * (len(ms) - 1))], 3)) if ms else (lambda q: None)
        return {"stats": dict(self.stats), "served_by": dict(self.rungs),
                "layer_ms": {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99)},
                "epochs_seen": sorted(self.epochs_seen),
                "quarantined": [d for d, l in self.links.items() if l.quarantined]}
