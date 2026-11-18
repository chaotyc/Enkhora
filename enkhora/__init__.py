"""Enkhora v1: anchored trunk + scavenged expert tier for MoE inference.

Layout
------
wire, transport     frame codec and asyncio RPC (no pickle on the wire)
experts             the expert kernel contract shared by donors and the anchor
store               master copy of the expert tier on the anchor's disk
routing             versioned bundle -> replica map
model               HF adapter (Qwen3-MoE, OLMoE) and the tiny test model
control             planner (pure), control-plane service, admin CLI
donor               slab manager, token bucket + AIMD governor, watchdog, executor
anchor              dispatcher (scatter/gather, hedging, audit), fallback ladder
sim                 trace-driven replay of the planner and lifetime predictor
"""

__version__ = "0.1.0"
