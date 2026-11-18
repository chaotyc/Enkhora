"""CLI: enkhora {control,donor,anchor,export,ctl,demo,sim}"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys


def _addr(s: str) -> tuple[str, int]:
    h, p = s.rsplit(":", 1)
    return h, int(p)


def cmd_control(a):
    from .control.planner import PlannerConfig
    from .control.server import ControlPlane

    cfg = PlannerConfig(spread_g=a.spread_g, expert_demand=a.expert_demand, alpha=a.alpha,
                        copy_rate_Bps=a.copy_rate_mbps * 1e6, max_replicas=a.max_replicas)

    async def main():
        cp = await ControlPlane(a.host, a.port, cfg).start()
        print(f"control plane listening on {a.host}:{cp.port}", flush=True)
        await asyncio.Event().wait()

    asyncio.run(main())


def cmd_donor(a):
    import torch

    from .donor.executor import Donor
    from .donor.watchdog import NvmlProbe, SimulatedProbe

    if a.threads:
        torch.set_num_threads(a.threads)
    mib = 1 << 20
    if a.nvml is not None:
        probe = NvmlProbe(a.nvml)
    else:
        probe = SimulatedProbe(total=int((a.mem_mib + a.guard_mib + a.slack_mib) * mib), sensitivity=a.sensitivity)
    d = Donor(a.id, a.job, _addr(a.control) if a.control else None, host=a.host, port=a.port,
              device=a.device, kappa=a.kappa, life_s=a.life_s, limit_s=a.limit_s, node=a.node,
              probe=probe, guard_bytes=int(a.guard_mib * mib), slack_bytes=int(a.slack_mib * mib),
              copy_rate_Bps=a.copy_rate_mbps * 1e6, corrupt=a.corrupt, aimd=a.aimd,
              advertise_host=a.advertise_host)
    asyncio.run(d.run_forever())


def cmd_export(a):
    from .model.hf import export_master, load_model

    m = load_model(a.model, seed=a.seed, dtype=a.dtype)
    ms = export_master(m, a.out, a.bundle_size, {"name": a.model, "seed": a.seed})
    print(f"wrote {len(ms.bundles)} bundles ({ms.tier_bytes() / 2**30:.2f} GiB) to {a.out}")


def cmd_anchor(a):
    import random
    import time

    from .anchor.anchor import Anchor, make_anchor_model
    from .anchor.dispatcher import DispatchConfig

    model, master = make_anchor_model(a.model, a.master, a.bundle_size, seed=a.seed, device=a.device, dtype=a.dtype)
    cfg = DispatchConfig(audit_rate=a.audit_rate, chunk_rows=a.chunk_rows)
    anc = Anchor(model, master, control=_addr(a.control) if a.control else None, host=a.host, port=a.port,
                 device=a.device, cfg=cfg, advertise_host=a.advertise_host)
    print(f"anchor master store serving on {a.host}:{anc.port}", flush=True)
    if a.control:
        ok = anc.wait_for_coverage(1, a.wait_s)
        print("placement ready" if ok else "placement incomplete; anchor fallback will serve the rest", flush=True)
    rng = random.Random(a.seed)
    V = model.config.vocab_size
    prompts = [[rng.randrange(2, V) for _ in range(rng.randint(4, 16))] for _ in range(a.batch)]
    for i in range(a.rounds):
        t0 = time.monotonic()
        toks, _ = anc.generate(prompts, a.max_new)
        dt = time.monotonic() - t0
        print(json.dumps({"round": i, "tok_per_s": round(toks.numel() / dt, 1), **anc.dispatcher.report()}), flush=True)
    anc.close()


def cmd_ctl(a):
    from .control.ctl import admin

    kw = {}
    if a.donor:
        kw["donor"] = a.donor
    if a.cmd == "set_kappa":
        kw["kappa"] = a.value
    if a.cmd == "host_alloc":
        kw["bytes"] = int(a.value * (1 << 20))
    print(json.dumps(admin(_addr(a.control), a.cmd, **kw), indent=1))


def cmd_demo(a):
    from .demo import run_demo

    sys.exit(0 if run_demo(a.workdir, verbose=a.verbose) else 1)


def cmd_sim(a):
    from .sim.replay import main as sim_main

    sim_main(a)


def main(argv=None):
    p = argparse.ArgumentParser(prog="enkhora")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="sub", required=True)

    c = sub.add_parser("control", help="run the control plane")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=7400)
    c.add_argument("--spread-g", type=int, default=2)
    c.add_argument("--expert-demand", type=float, default=1.0, help="GPU-equivalents of expert compute")
    c.add_argument("--alpha", type=float, default=10.0)
    c.add_argument("--copy-rate-mbps", type=float, default=1000.0)
    c.add_argument("--max-replicas", type=int, default=3)
    c.set_defaults(fn=cmd_control)

    d = sub.add_parser("donor", help="run a donor executor on one GPU (or CPU)")
    d.add_argument("--id", required=True)
    d.add_argument("--job", required=True, help="Slurm job id (anti-affinity domain)")
    d.add_argument("--node", default="")
    d.add_argument("--control", help="host:port of the control plane")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--advertise-host")
    d.add_argument("--port", type=int, default=0)
    d.add_argument("--device", default="cpu")
    d.add_argument("--kappa", type=float, default=1.0)
    d.add_argument("--life-s", type=float, default=math.inf)
    d.add_argument("--limit-s", type=float, default=math.inf)
    d.add_argument("--mem-mib", type=float, default=1024, help="simulated headroom (ignored with --nvml)")
    d.add_argument("--guard-mib", type=float, default=0.0)
    d.add_argument("--slack-mib", type=float, default=0.0)
    d.add_argument("--nvml", type=int, default=None, help="use NVML on this GPU index for the watchdog")
    d.add_argument("--copy-rate-mbps", type=float, default=1e6)
    d.add_argument("--threads", type=int, default=0)
    d.add_argument("--corrupt", type=float, default=0.0, help="fault injection: add this to every output")
    d.add_argument("--aimd", action="store_true", help="run the on/off-window AIMD governor")
    d.add_argument("--sensitivity", type=float, default=0.0, help="simulated host slowdown per unit duty")
    d.set_defaults(fn=cmd_donor)

    e = sub.add_parser("export", help="export a model's experts to a master store")
    e.add_argument("--model", default="tiny")
    e.add_argument("--out", required=True)
    e.add_argument("--bundle-size", type=int, default=8)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--dtype")
    e.set_defaults(fn=cmd_export)

    an = sub.add_parser("anchor", help="run an anchor with a synthetic batch workload")
    an.add_argument("--model", default="tiny")
    an.add_argument("--master", required=True)
    an.add_argument("--bundle-size", type=int, default=8)
    an.add_argument("--seed", type=int, default=0)
    an.add_argument("--dtype")
    an.add_argument("--device", default="cpu")
    an.add_argument("--control")
    an.add_argument("--host", default="127.0.0.1")
    an.add_argument("--advertise-host")
    an.add_argument("--port", type=int, default=7401)
    an.add_argument("--batch", type=int, default=8)
    an.add_argument("--max-new", type=int, default=32)
    an.add_argument("--rounds", type=int, default=3)
    an.add_argument("--audit-rate", type=float, default=0.0)
    an.add_argument("--chunk-rows", type=int, default=4096)
    an.add_argument("--wait-s", type=float, default=60)
    an.set_defaults(fn=cmd_anchor)

    t = sub.add_parser("ctl", help="admin commands")
    t.add_argument("cmd", choices=["status", "set_kappa", "host_alloc", "leave", "kill_switch", "replan"])
    t.add_argument("donor", nargs="?")
    t.add_argument("value", nargs="?", type=float, help="kappa, or MiB for host_alloc")
    t.add_argument("--control", default="127.0.0.1:7400")
    t.set_defaults(fn=cmd_ctl)

    m = sub.add_parser("demo", help="end-to-end demo on this machine")
    m.add_argument("--workdir", default=None)
    m.set_defaults(fn=cmd_demo)

    s = sub.add_parser("sim", help="trace-driven replay of planner + predictor")
    from .sim.replay import add_args

    add_args(s)
    s.set_defaults(fn=cmd_sim)

    a = p.parse_args(argv)
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)   # kill -USR1 <pid> dumps stacks
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    a.fn(a)


if __name__ == "__main__":
    main()
