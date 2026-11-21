"""Projected maximum token throughput on the design doc's illustrative
~128-GPU shared cluster, for batched (offline) jobs.

A back-of-envelope model built from the design doc's own figures (sections 3, 6 and 10).
Replace the assumptions with measured numbers as experiments 3, 5 and 6 land.

    python bench/throughput.py
    python bench/throughput.py --in-tokens 4000 --out-tokens 200 --anchors 2

Throughput is the minimum of four ceilings:
  1. Donor expert compute: GPU-time supply / expert GPU-time per output token.
  2. Anchor decode: KV cache bounds concurrent sequences; each advances one token per step.
  3. Anchor trunk: attention + shared layers for every prompt and output token.
  4. Anchor network: 14.3 KB per token per donor per layer, each way.
Plus a coverage penalty: bundles with no compute-capable copy are run on the anchor
by streaming their weights over PCIe every step.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

GiB = 1 << 30


@dataclass
class Assumptions:
    # Model: DeepSeek-V3-class, as in the design doc
    layers: int = 58
    experts: int = 256
    topk: int = 8
    hidden: int = 7168
    inter: int = 2048
    expert_bytes: float = 44e6           # one expert at 8 bits
    tier_gib: float = 609                # routed-expert tier at 8 bits
    trunk_bytes: float = 17e9            # dense trunk at 8 bits
    trunk_flops_per_tok: float = 2 * 16.6e9
    kv_bytes_per_tok: float = 61 * 576 * 2   # MLA latent, bf16
    # Donor GPUs (A100-class equivalents)
    donor_bw: float = 1.5e12             # B/s
    donor_flops: float = 150e12          # effective bf16 (about half of peak)
    prefill_ms_per_tok: float = 0.4      # expert GPU-ms per prompt token (doc's assumption)
    # Anchor (H100 80 GB, one per anchor)
    anchor_mem: float = 80e9
    anchor_reserve: float = 8e9          # workspace, activations, fallback cache
    anchor_bw: float = 3.35e12
    anchor_flops: float = 400e12
    anchor_nic: float = 50e9 * 0.7       # NDR 400 Gb/s at 70% efficiency
    anchor_pcie: float = 25e9
    # Path
    rtt_s: float = 50e-6
    wait_s: float = 1e-3                 # time-slice wait on a shared donor
    spread_g: int = 2                    # donors per layer
    # Supply on the design doc's illustrative cluster (median hour; assumptions, not data)
    quiet_hosts: int = 10
    quiet_mem_gib: float = 50
    unalloc_mem_gib: float = 60          # mix of 40 and 80 GB GPUs, after context
    kappa_exec_min: float = 0.05         # planner: below this a donor only stores


def distinct_experts(a: Assumptions, B: float) -> float:
    return a.experts * (1 - (1 - a.topk / a.experts) ** B)


def expert_layer_s(a: Assumptions, B: float) -> float:
    """GPU-seconds for one MoE layer's experts at batch B (bandwidth vs arithmetic)."""
    stream = distinct_experts(a, B) * a.expert_bytes / a.donor_bw
    math = B * a.topk * 6 * a.hidden * a.inter / a.donor_flops
    return max(stream, math)


def project(a: Assumptions, kappa: float, unalloc: int, anchors: int, p_in: int, p_out: int) -> dict:
    ratio = p_in / p_out
    # Supply
    quiet_exec = a.quiet_hosts if kappa >= a.kappa_exec_min else 0
    supply = quiet_exec * kappa + unalloc * 1.0                      # GPU-equivalents
    exec_mem = quiet_exec * a.quiet_mem_gib + unalloc * a.unalloc_mem_gib
    coverage = min(1.0, exec_mem / a.tier_gib)

    # Anchor concurrency from KV cache
    live_ctx = p_in + p_out / 2
    kv_room = a.anchor_mem - a.trunk_bytes - a.anchor_reserve
    B = max(1, int(kv_room / (live_ctx * a.kv_bytes_per_tok)))

    # Decode step time on one anchor
    trunk = max(B * a.trunk_flops_per_tok / a.anchor_flops, a.trunk_bytes / a.anchor_bw)
    attn = B * live_ctx * a.kv_bytes_per_tok / a.anchor_bw
    hop = a.layers * (a.rtt_s + (a.wait_s if kappa < 1 else 0) + expert_layer_s(a, B) / a.spread_g)
    uncovered = (1 - coverage) * a.tier_gib * GiB / a.anchor_pcie     # weights over PCIe each step
    t_step = trunk + attn + hop + uncovered

    # Ceilings (output tokens / s)
    cost_out = a.layers * expert_layer_s(a, B) / B + ratio * a.prefill_ms_per_tok / 1e3
    c_compute = supply * coverage / cost_out if cost_out else float("inf")
    c_decode = anchors * B / t_step
    trunk_tok = a.trunk_flops_per_tok / a.anchor_flops               # anchor-s per token, any phase
    c_trunk = anchors / (ratio * trunk_tok + (trunk + attn) / B)       # prefill + decode steps
    per_tok_wire = a.layers * 2 * a.hidden * 2 * a.spread_g          # bytes per processed token
    c_net = anchors * a.anchor_nic / ((1 + ratio) * per_tok_wire)
    ceilings = {"donor compute": c_compute, "anchor decode": c_decode,
                "anchor trunk": c_trunk, "anchor network": c_net}
    bind = min(ceilings, key=ceilings.get)
    out = ceilings[bind]
    return dict(kappa=kappa, unalloc=unalloc, supply=supply, coverage=coverage, B=B,
                t_step=t_step, out_tps=out, out_day=out * 86400, total_day=out * (1 + ratio) * 86400,
                bind=bind, ceilings=ceilings)


def fmt(n: float) -> str:
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.0f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-tokens", type=int, default=2000, help="prompt tokens per request")
    ap.add_argument("--out-tokens", type=int, default=500, help="output tokens per request")
    ap.add_argument("--anchors", type=int, default=2)
    ap.add_argument("--kappas", default="0,0.02,0.1,0.25")
    ap.add_argument("--unalloc", default="0,6,12,24", help="usable unallocated GPUs")
    ap.add_argument("--fp8-kv", action="store_true")
    ap.add_argument("--bits", type=int, default=8, choices=(4, 8), help="expert weight precision")
    a = ap.parse_args()
    base = Assumptions()
    if a.fp8_kv:
        base = replace(base, kv_bytes_per_tok=base.kv_bytes_per_tok / 2)
    if a.bits == 4:
        base = replace(base, expert_bytes=base.expert_bytes / 2, tier_gib=base.tier_gib / 2)
    kappas = [float(x) for x in a.kappas.split(",")]
    unallocs = [int(x) for x in a.unalloc.split(",")]
    print(f"requests {a.in_tokens} in / {a.out_tokens} out, {a.anchors} anchor GPU(s), {a.bits}-bit experts\n")
    print("output tokens per day (binding ceiling)")
    print("quiet-host κ \\ unallocated GPUs " + "".join(f"{u:>22}" for u in unallocs))
    rows = []
    for k in kappas:
        cells = []
        for u in unallocs:
            r = project(base, k, u, a.anchors, a.in_tokens, a.out_tokens)
            rows.append(r)
            flag = "" if r["coverage"] >= 1 else f" cov {r['coverage']:.0%}"
            cells.append(f"{fmt(r['out_day'])} ({r['bind'].split()[-1]}{flag})")
        print(f"κ = {k:<28}" + "".join(f"{c:>22}" for c in cells))
    r0 = project(base, 0.25, 24, a.anchors, a.in_tokens, a.out_tokens)
    print(f"\nanchor batch {r0['B']} sequences, decode step {r0['t_step'] * 1e3:.0f} ms")
    print("ceilings at κ 0.25, 24 unallocated (output tok/s): "
          + ", ".join(f"{n} {v:,.0f}" for n, v in r0["ceilings"].items()))


if __name__ == "__main__":
    main()
