"""End-to-end demo on one machine.

Three donors in three Slurm jobs:
  d1  unallocated GPU      κ=1.0   short expected life   (fast, fragile)
  d2  quiet host           κ=0.5   long life
  d3  busy host            κ=0.0   longest life           (store donor)

Phases:
  A  distributed decode == all-local reference, bitwise
  B  κ shrink on d1 swaps primaries to d2 by a routing-table edit: zero bytes moved
  C  chaos at fixed layer counts during one generation:
       SIGKILL d1 · SIGSTOP d2 (hang -> hedges) · host memory spike on d3 (watchdog
       evicts) · SIGCONT d2 · kill the control plane
     -> output still bitwise equal to the reference
"""

from __future__ import annotations

import json
import random
import signal
import tempfile
import time

from .anchor.anchor import compare
from .anchor.dispatcher import DispatchConfig
from .cluster import DonorSpec, LocalCluster
from .control.planner import PlannerConfig


def _say(msg):
    print(msg, flush=True)


def run_demo(workdir: str | None = None, verbose: bool = False) -> bool:
    workdir = workdir or tempfile.mkdtemp(prefix="interstitial-demo-")
    mib = 1 << 20
    donors = [
        DonorSpec("d1", "job-idle-a", kappa=1.0, mem_mib=4, life_s=1800, extra=["--guard-mib", "0.25", "--slack-mib", "0.25"]),
        DonorSpec("d2", "job-quiet-b", kappa=0.5, mem_mib=4, life_s=36000, extra=["--guard-mib", "0.25", "--slack-mib", "0.25"]),
        DonorSpec("d3", "job-busy-c", kappa=0.0, mem_mib=4, life_s=72000, extra=["--guard-mib", "0.25", "--slack-mib", "0.25"]),
    ]
    planner = PlannerConfig(expert_demand=0.5, alpha=1.0, copy_rate_Bps=1e12, max_replicas=3)
    dispatch = DispatchConfig(deadline_init_s=0.25, deadline_min_s=0.03)
    rng = random.Random(0)
    prompts = [[1] + [rng.randrange(2, 256) for _ in range(rng.randint(3, 12))] for _ in range(6)]
    max_new = 96
    ok = True

    _say(f"workdir {workdir}")
    with LocalCluster(workdir, donors, planner=planner, dispatch=dispatch) as c:
        st = c.wait(lambda s: s["coverage"]["ge2"] == s["coverage"]["bundles"] and
                    all(not d["loading"] for d in s["donors"].values()))
        _say(f"placement: {st['coverage']}  tier {st['tier_bytes'] / mib:.1f} MiB, epoch {st['epoch']}")
        for did, d in st["donors"].items():
            _say(f"  {did}: kappa {d['kappa']}  ready {len(d['ready'])}  primary for {len(d['primary_for'])}")

        a = c.anchor
        ref = a.generate_local(prompts, max_new)
        out = a.generate(prompts, max_new)
        r = compare(ref, out)
        _say(f"\n[A] distributed vs reference: {r}")
        ok &= r["logits_bitwise"]

        before = c.status()["counters"]["bytes_issued"]
        c.admin("set_kappa", donor="d1", kappa=0.0)
        st = c.wait(lambda s: not s["donors"]["d1"]["primary_for"] and s["donors"]["d1"]["kappa"] == 0.0)
        moved = st["counters"]["bytes_issued"] - before
        _say(f"\n[B] d1 kappa -> 0: primaries now d2={len(st['donors']['d2']['primary_for'])}, "
             f"d1={len(st['donors']['d1']['primary_for'])}; bytes moved {moved}")
        ok &= moved == 0
        c.admin("set_kappa", donor="d1", kappa=1.0)
        st = c.wait(lambda s: s["donors"]["d1"]["primary_for"])
        _say(f"    d1 kappa -> 1: d1 primary again for {len(st['donors']['d1']['primary_for'])}; "
             f"bytes moved {st['counters']['bytes_issued'] - before}")

        _say("\n[C] chaos during one generation")
        events = {
            30: ("SIGKILL d1 (primary, unallocated GPU reclaimed)", lambda: c.kill("d1")),
            80: ("SIGSTOP d2 (donor hangs, connection stays open)", lambda: c.kill("d2", signal.SIGSTOP)),
            120: ("host on d3 grows by 3.5 MiB (watchdog evicts)",
                  lambda: c.admin("host_alloc", donor="d3", bytes=int(3.5 * mib))),
            200: ("SIGCONT d2", lambda: c.kill("d2", signal.SIGCONT)),
            260: ("control plane dies", c.stop_control),
        }
        base = a.dispatcher.stats["layers"]
        fired = []

        def hook(n):
            ev = events.get(n - base)
            if ev and ev[0] not in fired:
                fired.append(ev[0])
                _say(f"    layer call {n - base:4d}: {ev[0]}")
                ev[1]()

        a.dispatcher.stats.clear()
        a.dispatcher.rungs.clear()
        base = 0
        a.dispatcher.on_layer = hook
        t0 = time.monotonic()
        out = a.generate(prompts, max_new)
        dt = time.monotonic() - t0
        a.dispatcher.on_layer = None
        r = compare(ref, out)
        _say(f"    {len(fired)}/{len(events)} events fired; generation took {dt:.2f}s")
        _say(f"    distributed vs reference: {r}")
        ok &= r["logits_bitwise"] and len(fired) == len(events)
        rep = a.dispatcher.report()
        _say("    dispatcher: " + json.dumps(rep))
        _say("\nRESULT: " + ("PASS - zero corrupted sequences" if ok else "FAIL"))
    return ok
