"""Experiment 1: replay a trace through the planner, predictor and eviction rules.

Reports replica coverage over time, fallback exposure, GiB re-replicated per hour,
oracle versus predictor. Decision it makes: if predicted coverage at P10 falls
below one copy, the default becomes a smaller or 4-bit model.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass

from ..control.planner import PBundle, PDonor, PlannerConfig, plan
from .survival import OracleCurve, SurvivalModel
from .trace import INTERVAL_S, Trace, load_csv, synthetic

GiB = 1 << 30


@dataclass
class SimConfig:
    bundle_bytes: float = 1.41e9      # DeepSeek-V3: one routing group of 32 experts at 8 bits
    n_layers: int = 58
    bundles_per_layer: int = 8
    guard_gib: float = 12.0
    ctx_gib: float = 1.0              # CUDA context + activation arena per donor
    warmup_s: float = 1800.0          # ring 1: a job must have run this long to lend
    quiet_util: float = 0.25
    kappa_quiet: float = 0.25
    lend_unallocated: bool = False    # accounting averages overstate live idle GPUs; opt in with --unallocated
    max_peak_frac: float = 0.25       # donor screen: peak VRAM under 25% of capacity
    train_frac: float = 0.5


def _gpu_timelines(trace: Trace):
    """Per interval and GPU: when the current idle stretch started and when it ends (oracle)."""
    gpus = sorted(trace.samples[0])
    n = len(trace)
    idle_since = [dict() for _ in range(n)]
    next_busy = [dict() for _ in range(n)]
    for g in gpus:
        start = None
        for i in range(n):
            if not trace.samples[i][g].job:
                start = trace.times[i] if start is None else start
                idle_since[i][g] = start
            else:
                start = None
        nb = math.inf
        for i in range(n - 1, -1, -1):
            s = trace.samples[i][g]
            if s.job:
                nb = s.job_start
            else:
                next_busy[i][g] = nb
    return idle_since, next_busy


def fit_predictor(trace: Trace, upto: int) -> SurvivalModel:
    jobs, idle_runs = {}, []
    gpus = sorted(trace.samples[0])
    end_t = trace.times[upto - 1] + INTERVAL_S if upto else 0
    for i in range(upto):
        for s in trace.samples[i].values():
            if s.job and s.job_end <= end_t:
                jobs[s.job] = {"partition": s.partition, "limit_s": s.limit_s, "account": s.account,
                               "duration_s": s.job_end - s.job_start}
    for g in gpus:
        run = 0
        for i in range(upto):
            if not trace.samples[i][g].job:
                run += 1
            elif run:
                idle_runs.append(run * INTERVAL_S)
                run = 0
    return SurvivalModel().fit(list(jobs.values()), idle_runs or [INTERVAL_S])


def replay(trace: Trace, sc: SimConfig, pc: PlannerConfig, mode: str, model: SurvivalModel | None,
           start: int, idle_since, next_busy) -> list[dict]:
    bundles = [PBundle(l * sc.bundles_per_layer + j, l, int(sc.bundle_bytes))
               for l in range(sc.n_layers) for j in range(sc.bundles_per_layer)]
    tier = sum(b.nbytes for b in bundles)
    nb = len(bundles)
    held: dict[str, dict[int, str]] = {}
    prev_primary: dict[int, str] = {}
    rows = []
    for i in range(start, len(trace)):
        t = trace.times[i]
        donors, four_h = [], 0
        for g, s in trace.samples[i].items():
            if s.job:
                elapsed = t - s.job_start
                if elapsed < sc.warmup_s or s.peak_mem_gib > sc.max_peak_frac * s.mem_gib:
                    continue
                mem = (s.mem_gib - s.peak_mem_gib - sc.guard_gib - sc.ctx_gib) * GiB
                kappa = sc.kappa_quiet if s.mean_util < sc.quiet_util else 0.0
                surv = OracleCurve(s.job_end - t) if mode == "oracle" else \
                    model.curve("job", s.partition, s.limit_s, elapsed, s.account)
                did, job, true_rem = f"{g}/{s.job}", s.job, s.job_end - t
            else:
                if not sc.lend_unallocated:
                    continue
                mem = (s.mem_gib - sc.ctx_gib) * GiB
                kappa = 1.0
                since = idle_since[i][g]
                surv = OracleCurve(next_busy[i][g] - t) if mode == "oracle" else \
                    model.curve("idle", elapsed_s=t - since)
                did, job, true_rem = f"{g}/idle{int(since)}", f"idle:{g}", next_busy[i][g] - t
            if mem < sc.bundle_bytes:
                continue
            four_h += true_rem >= 4 * 3600
            donors.append(PDonor(did, job, int(mem), kappa, surv, held=held.get(did, {})))
        ids = {d.id for d in donors}

        # Departures since the last interval: data plane fails over before the planner backfills.
        before = {b for h in held.values() for b in h}
        gone = [d for d in held if d not in ids]
        lost_primary = sum(1 for b, d in prev_primary.items() if d in gone)
        for d in gone:
            del held[d]
        after = {b for h in held.values() for b in h}
        exposed = len(before - after)

        p = plan(bundles, donors, pc, prev_primary)
        loaded = p.bytes_to_load({b.id: b for b in bundles})
        for d, bs in p.drops.items():
            for b in bs:
                held.get(d, {}).pop(b, None)
        for d, ls in p.loads.items():
            held.setdefault(d, {}).update({b: "ready" for b, _ in ls})
        prev_primary = p.primary
        counts = {}
        for h in held.values():
            for b in h:
                counts[b] = counts.get(b, 0) + 1
        admitted = set(p.admitted)
        rows.append({
            "t": t, "donors": len(donors), "admitted": len(admitted),
            "pooled_gib": sum(d.mem for d in donors if d.id in admitted) / GiB,
            "copies": sum(counts.values()) * sc.bundle_bytes / tier,
            "cov1": sum(c >= 1 for c in counts.values()) / nb,
            "cov2": sum(c >= 2 for c in counts.values()) / nb,
            "no_exec": len(p.no_exec) / nb,
            "exposed": exposed / nb, "primary_lost": lost_primary / nb,
            "loaded_gib": loaded / GiB, "first": i == start,
            "donors_4h": four_h,
        })
    return rows


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))] if s else float("nan")


def summarize(rows: list[dict]) -> dict:
    steady = [r for r in rows if not r["first"]]
    hours = len(steady) * INTERVAL_S / 3600 or 1
    out = {}
    for k in ("donors", "donors_4h", "admitted", "pooled_gib", "copies", "cov1", "cov2", "no_exec"):
        out[k] = {"p10": pct([r[k] for r in rows], 0.1), "p50": pct([r[k] for r in rows], 0.5),
                  "p90": pct([r[k] for r in rows], 0.9)}
    out["exposed_mean"] = sum(r["exposed"] for r in steady) / max(1, len(steady))
    out["primary_lost_mean"] = sum(r["primary_lost"] for r in steady) / max(1, len(steady))
    out["gib_rereplicated_per_hour"] = sum(r["loaded_gib"] for r in steady) / hours
    out["frac_intervals_below_one_copy"] = sum(r["copies"] < 1.0 for r in rows) / max(1, len(rows))
    return out


def add_args(p):
    p.add_argument("--trace", help="CSV in the schema of sim/trace.py; default: synthetic")
    p.add_argument("--days", type=float, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write-trace", help="also write the (synthetic) trace CSV here")
    p.add_argument("--out", help="write the full report JSON here")
    p.add_argument("--expert-demand", type=float, default=2.0, help="GPU-equivalents of expert compute")
    p.add_argument("--kappa-quiet", type=float, default=0.25)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--spread-g", type=int, default=2)
    p.add_argument("--max-replicas", type=int, default=3)
    p.add_argument("--unallocated", action="store_true", help="also lend unallocated GPUs (count unverified)")
    p.add_argument("--max-peak-frac", type=float, default=0.25, help="donor screen on peak VRAM fraction")
    p.add_argument("--low-mem-share", type=float, default=0.30, help="synthetic: share of jobs under 25%% VRAM")
    p.add_argument("--bits", type=int, default=8, choices=[4, 8], help="expert weight width")


def run(trace: Trace, sc: SimConfig, pc: PlannerConfig) -> dict:
    split = int(len(trace) * sc.train_frac)
    model = fit_predictor(trace, split) if split else fit_predictor(trace, len(trace))
    idle_since, next_busy = _gpu_timelines(trace)
    rep = {"config": {"sim": asdict(sc), "planner": asdict(pc)}, "intervals": len(trace) - split}
    for mode in ("oracle", "predictor"):
        t0 = time.monotonic()
        rows = replay(trace, sc, pc, mode, model, split, idle_since, next_busy)
        rep[mode] = summarize(rows)
        rep[mode]["replay_s"] = round(time.monotonic() - t0, 2)
        rep[mode + "_rows"] = rows
    p10 = rep["predictor"]["copies"]["p10"]
    rep["decision"] = ("OK: predicted P10 coverage is %.2f copies" % p10 if p10 >= 1.0 else
                       "SWITCH: predicted P10 coverage %.2f < 1 copy -> default to a 4-bit or smaller model" % p10)
    return rep


def print_report(rep: dict):
    o, p = rep["oracle"], rep["predictor"]
    print(f"replayed {rep['intervals']} intervals ({rep['intervals'] * INTERVAL_S / 3600:.0f} h)\n")
    print(f"{'metric':34s} {'oracle':>22s} {'predictor':>22s}")
    for k, label in [("donors", "qualifying donors"), ("donors_4h", "donors with 4h+ left (true)"),
                     ("admitted", "admitted donors"),
                     ("pooled_gib", "pooled VRAM after guard (GiB)"), ("copies", "copies of expert tier"),
                     ("cov1", "bundles with >=1 replica"), ("cov2", "bundles with >=2 replicas"),
                     ("no_exec", "bundles w/o compute primary")]:
        f = lambda s: f"{s['p10']:.2f} / {s['p50']:.2f} / {s['p90']:.2f}"  # noqa: E731
        print(f"{label + ' (P10/P50/P90)':34s} {f(o[k]):>22s} {f(p[k]):>22s}")
    for k, label in [("exposed_mean", "fallback exposure per interval"),
                     ("primary_lost_mean", "primaries lost per interval"),
                     ("gib_rereplicated_per_hour", "GiB re-replicated per hour"),
                     ("frac_intervals_below_one_copy", "intervals below one copy")]:
        print(f"{label:34s} {o[k]:>22.3f} {p[k]:>22.3f}")
    print("\n" + rep["decision"])


def main(a):
    trace = load_csv(a.trace) if a.trace else synthetic(days=a.days, seed=a.seed, low_mem_share=a.low_mem_share)
    if a.write_trace:
        from .trace import write_csv

        write_csv(trace, a.write_trace)
    sc = SimConfig(kappa_quiet=a.kappa_quiet, lend_unallocated=a.unallocated, max_peak_frac=a.max_peak_frac,
                   bundle_bytes=1.41e9 * a.bits / 8)
    pc = PlannerConfig(spread_g=a.spread_g, expert_demand=a.expert_demand, alpha=a.alpha,
                       max_replicas=a.max_replicas, copy_rate_Bps=1e9)
    rep = run(trace, sc, pc)
    print_report(rep)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f)
