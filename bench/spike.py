"""Experiment 4: reclaim spike test.

The host allocates in steps of rising size and speed while our slabs are loaded.
Measures time to free, and host allocation failures over many trials. Pass is
zero host failures for steps inside the guard. Sets the guard and poll interval.

GPU mode (real allocations, NVML probe; the host is a separate process):
    python bench/spike.py --device cuda:0 --guard-gib 12 --poll-ms 20 --trials 200

Simulated mode (CPU; exercises the watchdog logic and the residual-risk model):
    python bench/spike.py --simulate --trials 2000

Residual risk without ring 3: host allocations totalling more than the guard inside
one poll interval can still fail (a single step larger than the guard is the special
case the design doc names). The simulated report splits trials by that burst size.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import sys
import threading
import time

import torch

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))
from interstitial.donor.slabs import SlabManager  # noqa: E402
from interstitial.donor.watchdog import SimulatedProbe, Watchdog  # noqa: E402
from interstitial.experts import BundleWeights  # noqa: E402

GiB = 1 << 30
MiB = 1 << 20


def fill_slabs(slabs: SlabManager, device, total_bytes: int, slab_bytes: int, alloc: bool):
    n = total_bytes // slab_bytes
    for b in range(n):
        role = "primary" if b % 3 == 0 else "standby"
        slabs.reserve(b, slab_bytes, role)
        if alloc:
            el = slab_bytes // 4 // 2
            w = BundleWeights(b, 0, [0], torch.empty(1, el, 1, device=device), torch.empty(1, 1, el, device=device))
        else:
            w = BundleWeights(b, 0, [0], torch.empty(1, 0, 1), torch.empty(1, 1, 0))
        slabs.set_ready(b, w)
        if not alloc:
            slabs._slabs[b].nbytes = slab_bytes     # simulated: account bytes without allocating
    return n


def simulated(a):
    rng = random.Random(a.seed)
    rows = []
    for trial in range(a.trials):
        total = int(a.gpu_gib * GiB)
        probe = SimulatedProbe(total=total, host_used=int(a.host_base_gib * GiB))
        slabs = SlabManager()
        guard, slack = int(a.guard_gib * GiB), int(a.slack_gib * GiB)
        fill_slabs(slabs, "cpu", max(0, total - probe.host_used - guard - slack), int(a.slab_gib * GiB), alloc=False)
        wd = Watchdog(probe, slabs, guard, slack, poll_s=a.poll_ms / 1e3)
        step = rng.uniform(0.1, a.max_step_gib) * GiB
        # The host's own growth always fits the GPU: a failure is one *we* caused.
        n_steps = min(a.steps, int((total - probe.host_used - GiB) // step))
        # Host allocates `step` at random gaps; the watchdog polls at a random phase.
        phase = rng.uniform(0, wd.poll_s)
        t, next_poll, burst, worst, failed = 0.0, phase, 0.0, 0.0, 0
        for _ in range(n_steps):
            t += rng.expovariate(1.0 / (a.mean_gap_ms / 1e3))
            while next_poll <= t:
                wd.check()
                next_poll += wd.poll_s
                burst = 0.0                       # host bytes allocated since the last poll
            burst += step
            worst = max(worst, burst)
            if probe.free_bytes(slabs.used_bytes()) < step:
                failed = 1
                break
            probe.host_used += int(step)
        rows.append((worst / GiB, failed))
    report(rows, a.guard_gib)


def host_proc(device, steps_gib, gap_s, out: mp.Queue):
    torch.cuda.set_device(device)
    held, fails = [], 0
    for s in steps_gib:
        time.sleep(gap_s)
        try:
            held.append(torch.empty(int(s * GiB), dtype=torch.uint8, device=device))
        except torch.cuda.OutOfMemoryError:
            fails += 1
            break
    out.put(fails)


def gpu(a):
    from interstitial.donor.watchdog import NvmlProbe

    rng = random.Random(a.seed)
    idx = int(a.device.split(":")[1]) if ":" in a.device else 0
    rows, free_ms = [], []
    ctx = mp.get_context("spawn")
    for trial in range(a.trials):
        slabs = SlabManager(a.device)
        probe = NvmlProbe(idx)
        guard, slack = int(a.guard_gib * GiB), int(a.slack_gib * GiB)
        headroom = probe.free_bytes(0) - guard - slack
        fill_slabs(slabs, a.device, max(0, headroom), int(a.slab_gib * GiB), alloc=True)
        evict_times = []
        wd = Watchdog(probe, slabs, guard, slack, poll_s=a.poll_ms / 1e3, on_evict=lambda bs: evict_times.append(time.monotonic()))
        wd.start()
        step = rng.uniform(0.1, a.max_step_gib)
        n_steps = min(a.steps, int((probe.total / GiB - a.host_base_gib - 1) // step))
        q = ctx.Queue()
        p = ctx.Process(target=host_proc, args=(a.device, [step] * n_steps, a.mean_gap_ms / 1e3, q))
        t0 = time.monotonic()
        p.start()
        fails = q.get()
        p.join()
        wd.stop()
        if evict_times:
            free_ms.append((evict_times[0] - t0) * 1e3)
        slabs.drop_all()
        rows.append((step, fails))
    report(rows, a.guard_gib, unit="step")     # on hardware we see steps, not the poll-relative burst
    if free_ms:
        free_ms.sort()
        print(f"time to first free (ms): p50 {free_ms[len(free_ms) // 2]:.1f}  p99 {free_ms[int(0.99 * (len(free_ms) - 1))]:.1f}")


def report(rows, guard_gib, unit="burst"):
    """rows: (largest host allocation burst between two watchdog polls in GiB, failed)."""
    inside = [f for s, f in rows if s <= guard_gib]
    outside = [f for s, f in rows if s > guard_gib]
    print(f"trials {len(rows)}: {unit} <= guard: {sum(inside)} failures / {len(inside)}   "
          f"{unit} > guard: {sum(outside)} failures / {len(outside)}  (residual risk without ring 3)")
    print("PASS" if sum(inside) == 0 else f"FAIL: host failures with {unit} inside the guard")


def main():
    import logging

    logging.basicConfig(level=logging.ERROR)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--steps", type=int, default=40, help="host allocations per trial")
    ap.add_argument("--gpu-gib", type=float, default=80)
    ap.add_argument("--host-base-gib", type=float, default=10)
    ap.add_argument("--guard-gib", type=float, default=12)
    ap.add_argument("--slack-gib", type=float, default=2)
    ap.add_argument("--slab-gib", type=float, default=1.41)
    ap.add_argument("--poll-ms", type=float, default=20)
    ap.add_argument("--max-step-gib", type=float, default=16)
    ap.add_argument("--mean-gap-ms", type=float, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    simulated(a) if a.simulate else gpu(a)


if __name__ == "__main__":
    main()
