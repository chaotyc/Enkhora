"""Expert placement: incremental greedy assignment, run on events (section 5).

Pure function of (bundles, donors with current holdings, config). The same code
runs in the live control plane and in the trace simulator.

Hard constraints: memory fits M; primaries' expected compute fits κ; no two
replicas of a bundle in one Slurm job; zero replicas is legal (master exists).
Priorities: coverage, then redundancy where fragile, then locality and hedging.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol


class Survival(Protocol):
    def p_alive(self, t: float) -> float: ...
    def expected_min(self, horizon: float) -> float: ...


@dataclass
class ExpSurvival:
    """S(t) = exp(-t / mean), hard-capped at the remaining time limit."""

    mean_s: float = math.inf
    cap_s: float = math.inf

    def p_alive(self, t: float) -> float:
        if t >= self.cap_s:
            return 0.0
        if math.isinf(self.mean_s):
            return 1.0
        return math.exp(-t / self.mean_s)

    def expected_min(self, horizon: float) -> float:
        h = min(horizon, self.cap_s)
        if math.isinf(self.mean_s):
            return h
        return self.mean_s * (1 - math.exp(-h / self.mean_s))


@dataclass
class PBundle:
    id: int
    layer: int
    nbytes: int
    heat: float = 1.0


@dataclass
class PDonor:
    id: str
    job: str
    mem: int                         # M: total bytes we may hold on this GPU
    kappa: float                     # compute budget, fraction of wall time
    survival: Survival = field(default_factory=ExpSurvival)
    rtt_ms: float = 0.0
    node: str = ""
    held: dict[int, str] = field(default_factory=dict)   # bundle -> "loading" | "ready"
    ending: bool = False             # leaving, or predicted end inside the lead time


@dataclass
class PlannerConfig:
    spread_g: int = 2                # donors per layer for primaries
    horizon_s: float = 3600.0        # H
    alpha: float = 10.0              # admission safety multiple
    copy_rate_Bps: float = 1e9       # throttled fill rate
    expert_demand: float = 1.0       # total expert compute needed, GPU-equivalents
    kappa_exec_min: float = 0.05     # below this a donor only stores
    max_replicas: int = 3
    hysteresis: float = 0.25         # a new primary must beat the current one by this margin
    end_lead_s: float = 300.0
    end_p: float = 0.5               # P(alive at lead) below this => treat as ending


@dataclass
class Plan:
    placement: dict[int, list[str]]          # bundle -> donors, primary first
    primary: dict[int, str]
    roles: dict[str, dict[int, str]]         # donor -> bundle -> "primary" | "standby"
    loads: dict[str, list[tuple[int, str | None]]]   # donor -> [(bundle, source donor | None=master)]
    drops: dict[str, list[int]]
    no_exec: list[int]                       # bundles without a compute-backed primary
    uncovered: list[int]                     # bundles with zero donor replicas
    admitted: list[str]

    def bytes_to_load(self, bundles: dict[int, PBundle]) -> int:
        return sum(bundles[b].nbytes for ls in self.loads.values() for b, _ in ls)


def admissible(d: PDonor, cfg: PlannerConfig) -> bool:
    """E[min(L, H)] >= alpha * t_load(M)."""
    if d.ending or d.mem <= 0:
        return False
    return d.survival.expected_min(cfg.horizon_s) >= cfg.alpha * d.mem / cfg.copy_rate_Bps


def is_ending(d: PDonor, cfg: PlannerConfig) -> bool:
    return d.ending or d.survival.p_alive(cfg.end_lead_s) < cfg.end_p


def plan(bundles: list[PBundle], donors: list[PDonor], cfg: PlannerConfig,
         prev_primary: dict[int, str] | None = None) -> Plan:
    prev_primary = prev_primary or {}
    B = {b.id: b for b in bundles}
    D = {d.id: d for d in donors}
    total_heat = sum(b.heat for b in bundles) or 1.0
    cost = {b.id: cfg.expert_demand * b.heat / total_heat for b in bundles}

    ending = {d.id: is_ending(d, cfg) for d in donors}
    admit = {d.id: (not ending[d.id]) and admissible(d, cfg) for d in donors}
    exec_ok = {d.id: d.kappa >= cfg.kappa_exec_min and not ending[d.id] for d in donors}
    exec_score = {d.id: d.kappa / (1.0 + d.rtt_ms) for d in donors}
    stab = {d.id: d.survival.expected_min(cfg.horizon_s) for d in donors}
    fragility = {d.id: 1.0 - d.survival.p_alive(cfg.horizon_s) for d in donors}

    hold: dict[int, set[str]] = defaultdict(set)
    mem_used: dict[str, int] = defaultdict(int)
    for d in donors:
        for b in d.held:
            if b in B:
                hold[b].add(d.id)
                mem_used[d.id] += B[b].nbytes
    comp_used: dict[str, float] = defaultdict(float)
    jobs_of = lambda b: {D[x].job for x in hold[b]}  # noqa: E731
    loads: dict[str, list[int]] = defaultdict(list)
    drops: dict[str, list[int]] = defaultdict(list)
    primary: dict[int, str] = {}

    by_cost = sorted(B, key=lambda b: (-cost[b], b))

    def comp_room(d, b):
        return comp_used[d] + cost[b] <= D[d].kappa + 1e-12

    def mem_room(d, b):
        return mem_used[d] + B[b].nbytes <= D[d].mem

    # A. Roles for replicas that already exist. Swapping roles moves no bytes.
    def assign_existing_primaries():
        primary.clear()
        comp_used.clear()
        for b in by_cost:
            cands = [d for d in hold[b] if exec_ok[d] and comp_room(d, b)]
            if not cands:
                continue
            best = max(cands, key=lambda d: (exec_score[d], d))
            cur = prev_primary.get(b)
            if cur in cands and exec_score[cur] * (1 + cfg.hysteresis) >= exec_score[best]:
                best = cur
            primary[b] = best
            comp_used[best] += cost[b]

    assign_existing_primaries()

    # B. Trim donors whose M shrank: drop extras/standbys first, primaries last.
    trimmed = False
    for d in donors:
        if mem_used[d.id] <= d.mem:
            continue
        mine = [b for b in d.held if b in B]
        mine.sort(key=lambda b: (primary.get(b) == d.id, -len(hold[b]), b))
        while mem_used[d.id] > d.mem and mine:
            b = mine.pop(0)
            hold[b].discard(d.id)
            mem_used[d.id] -= B[b].nbytes
            drops[d.id].append(b)
            trimmed = True
    if trimmed:
        assign_existing_primaries()

    def add(d, b):
        hold[b].add(d)
        mem_used[d] += B[b].nbytes
        loads[d].append(b)

    # C. New primaries, layer by layer, spread across the g best execute donors.
    exec_rank = sorted((d.id for d in donors if exec_ok[d.id] and admit[d.id]),
                       key=lambda x: (-exec_score[x], x))
    need = defaultdict(list)
    for b in sorted(B):
        if b not in primary:
            need[B[b].layer].append(b)
    for layer in sorted(need):
        bs = need[layer]
        chosen = [d for d in exec_rank if any(mem_room(d, b) and comp_room(d, b) for b in bs)][:cfg.spread_g]
        rest = [d for d in exec_rank if d not in chosen]
        for i, b in enumerate(bs):
            order = chosen[i % len(chosen):] + chosen[:i % len(chosen)] + rest if chosen else rest
            for d in order:
                if mem_room(d, b) and comp_room(d, b) and D[d].job not in jobs_of(b):
                    add(d, b)
                    primary[b] = d
                    comp_used[d] += cost[b]
                    break

    def live_count(b):
        return sum(1 for x in hold[b] if not ending[x])

    # D. Standbys: most fragile primaries first, on the most stable donor in another job.
    stable_rank = sorted((d.id for d in donors if admit[d.id]),
                         key=lambda x: (-stab[x], D[x].kappa, x))

    def frag(b):
        p = primary.get(b)
        return fragility[p] if p else 2.0   # no exec primary: most in need of a copy

    for b in sorted(B, key=lambda b: (-frag(b), -cost[b], b)):
        while live_count(b) < 2:
            js = jobs_of(b)
            pick = next((d for d in stable_rank if d not in hold[b] and D[d].job not in js and mem_room(d, b)), None)
            if pick is None:
                break
            add(pick, b)

    # E. Leftovers: third+ replicas on execute donors, for load spreading and hedging.
    if cfg.max_replicas >= 3 and all(live_count(b) >= 2 for b in B):
        for b in by_cost:
            while live_count(b) < cfg.max_replicas:
                js = jobs_of(b)
                pick = next((d for d in exec_rank if d not in hold[b] and D[d].job not in js and mem_room(d, b)), None)
                if pick is None:
                    break
                add(pick, b)

    # Output: ordered placement, roles, sources for loads.
    placement, roles = {}, defaultdict(dict)
    for b in sorted(B):
        p = primary.get(b)
        others = sorted((x for x in hold[b] if x != p),
                        key=lambda x: (ending[x], not exec_ok[x], -exec_score[x], x))
        placement[b] = ([p] if p else []) + others
        for x in placement[b]:
            roles[x][b] = "primary" if x == p else "standby"

    def source(b, target):
        ready = [x for x in hold[b] if x != target and b in D[x].held and D[x].held[b] == "ready"]
        store = [x for x in ready if not exec_ok[x]]
        return min(store, key=lambda x: D[x].rtt_ms) if store else None   # never source from executors

    load_out = {d: [(b, source(b, d)) for b in bs] for d, bs in loads.items()}
    return Plan(placement=placement, primary=dict(primary), roles=dict(roles), loads=load_out,
                drops=dict(drops), no_exec=[b for b in sorted(B) if b not in primary],
                uncovered=[b for b in sorted(B) if not hold[b]],
                admitted=[d for d in D if admit[d]])
