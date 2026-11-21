"""Traces of GPU occupancy in 15-minute intervals.

CSV schema (one row per GPU per interval; empty ``job`` = unallocated GPU)::

    t            seconds since trace start (interval start)
    gpu          stable GPU id, e.g. gpu-node12:2
    node         node name
    fabric       EDR | HDR | NDR
    mem_gib      GPU memory capacity
    job          Slurm job id ('' if unallocated)
    job_start    job start, seconds since trace start
    job_end      job end, seconds since trace start (used only by the oracle)
    limit_s      requested time limit in seconds
    partition    Slurm partition
    account      pseudonymized account
    peak_mem_gib sampled high-water mark of the job on this GPU in this interval
    mean_util    kernel-active fraction in this interval (0..1)

To replay a real cluster, convert its accounting and GPU telemetry export (e.g. Slurm
``sacct`` job records joined with per-GPU memory and utilization samples) to this
schema. ``synthetic`` generates a trace with the same schema for a hypothetical
mid-sized cluster, so the pipeline runs without real data.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass

INTERVAL_S = 900


@dataclass
class Sample:
    gpu: str
    node: str
    fabric: str
    mem_gib: float
    job: str
    job_start: float
    job_end: float
    limit_s: float
    partition: str
    account: str
    peak_mem_gib: float
    mean_util: float


@dataclass
class Trace:
    times: list[int]
    samples: list[dict[str, Sample]]          # per interval: gpu -> sample

    def __len__(self):
        return len(self.times)


FIELDS = ["t", "gpu", "node", "fabric", "mem_gib", "job", "job_start", "job_end", "limit_s",
          "partition", "account", "peak_mem_gib", "mean_util"]


def load_csv(path: str) -> Trace:
    by_t: dict[int, dict[str, Sample]] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            t = int(float(r["t"]))
            s = Sample(r["gpu"], r["node"], r.get("fabric", ""), float(r["mem_gib"]), r.get("job", ""),
                       float(r["job_start"] or 0), float(r["job_end"] or 0), float(r["limit_s"] or 0),
                       r.get("partition", ""), r.get("account", ""), float(r["peak_mem_gib"] or 0),
                       float(r["mean_util"] or 0))
            by_t.setdefault(t, {})[s.gpu] = s
    times = sorted(by_t)
    return Trace(times, [by_t[t] for t in times])


def write_csv(trace: Trace, path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for t, snap in zip(trace.times, trace.samples):
            for s in snap.values():
                w.writerow([t, s.gpu, s.node, s.fabric, s.mem_gib, s.job, s.job_start, s.job_end, s.limit_s,
                            s.partition, s.account, round(s.peak_mem_gib, 3), round(s.mean_util, 4)])


def synthetic(days: float = 14, seed: int = 0, target_alloc: float = 0.8,
              low_mem_share: float = 0.30, low_util_share: float = 0.20) -> Trace:
    """A hypothetical cluster: 128 GPUs over three generations; jobs of 1/2/4 GPUs;
    ~80% allocated on average.

    ``low_mem_share`` (jobs peaking under 25% VRAM) and ``low_util_share`` (share of
    GPU-hours under 25% compute, about a fifth) are rough, illustrative assumptions
    that put a few dozen qualifying donors in a typical interval. Synthetic numbers
    are placeholders for your own cluster's data, not evidence."""
    rng = random.Random(seed)
    gpus = []   # (gpu, node, fabric, mem)
    def add_nodes(prefix, n_nodes, per, fabric, mem):
        for i in range(n_nodes):
            for g in range(per):
                gpus.append((f"{prefix}{i:02d}:{g}", f"{prefix}{i:02d}", fabric, mem))
    add_nodes("a40n", 10, 4, "EDR", 40.0)
    add_nodes("a80n", 12, 4, "HDR", 80.0)
    add_nodes("h100n", 10, 4, "NDR", 80.0)
    by_node: dict[str, list[str]] = {}
    for g, n, _, _ in gpus:
        by_node.setdefault(n, []).append(g)
    meta = {g: (n, f, m) for g, n, f, m in gpus}

    mean_dur = 3 * 3600 * math.exp(1.1 ** 2 / 2)            # lognormal(median 3h, sigma 1.1)
    mean_gpus = 0.6 * 1 + 0.25 * 2 + 0.15 * 4
    lam = target_alloc * len(gpus) / (mean_dur * mean_gpus)   # jobs per second
    n_int = int(days * 86400 / INTERVAL_S)
    owner: dict[str, str] = {}
    jobs: dict[str, dict] = {}
    times, samples = [], []
    jid = 0
    accounts = [f"acct{i:03d}" for i in range(60)]
    acct_ratio = {a: rng.uniform(0.2, 0.9) for a in accounts}   # habitual actual/requested runtime

    for i in range(n_int):
        t = i * INTERVAL_S
        for j in [j for j, info in jobs.items() if info["end"] <= t]:
            for g in jobs[j]["gpus"]:
                owner.pop(g, None)
            del jobs[j]
        diurnal = 1 + 0.25 * math.sin(2 * math.pi * (t % 86400) / 86400)
        for _ in range(_poisson(rng, lam * INTERVAL_S * diurnal)):
            k = rng.choices([1, 2, 4], [0.6, 0.25, 0.15])[0]
            nodes = [n for n, gs in by_node.items() if sum(g not in owner for g in gs) >= k]
            if not nodes:
                continue
            node = rng.choice(nodes)
            take = [g for g in by_node[node] if g not in owner][:k]
            acct = rng.choice(accounts)
            limit = rng.choice([4, 8, 12, 24, 48]) * 3600
            part = "short" if limit <= 4 * 3600 else "gpu"
            if rng.random() < 0.12:
                dur = rng.uniform(60, 1800)                          # early crash / typo
            else:
                dur = min(limit, rng.lognormvariate(math.log(limit * acct_ratio[acct]), 0.6))
            start = t + rng.uniform(0, INTERVAL_S)
            jid += 1
            name = str(1_000_000 + jid)
            memf = rng.uniform(0.02, 0.25) if rng.random() < low_mem_share else rng.uniform(0.25, 0.97)
            low_util_p = low_util_share * (1.15 if memf < 0.25 else 0.9)
            util = rng.uniform(0.0, 0.25) if rng.random() < low_util_p else rng.uniform(0.25, 1.0)
            jobs[name] = {"gpus": take, "start": start, "end": start + dur, "limit": limit, "part": part,
                          "acct": acct, "memf": memf, "util": util}
            for g in take:
                owner[g] = name
        snap = {}
        for g, (node, fabric, mem) in meta.items():
            j = owner.get(g)
            if j is None:
                snap[g] = Sample(g, node, fabric, mem, "", 0, 0, 0, "", "", 0.0, 0.0)
                continue
            info = jobs[j]
            warm = min(1.0, (t + INTERVAL_S - info["start"]) / 1800)
            peak = mem * info["memf"] * (0.6 + 0.4 * warm) * rng.uniform(0.9, 1.0)
            util = min(1.0, max(0.0, info["util"] + rng.gauss(0, 0.05)))
            snap[g] = Sample(g, node, fabric, mem, j, info["start"], info["end"], info["limit"],
                             info["part"], info["acct"], peak, util)
        times.append(t)
        samples.append(snap)
    return Trace(times, samples)


def _poisson(rng: random.Random, mu: float) -> int:
    L, k, p = math.exp(-mu), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1
