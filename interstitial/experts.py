"""The expert kernel contract, shared by every executor (donors and the anchor).

A donor holding bundle b computes, for each token t routed into b,

    partial_b[t] = sum_{e in TopK(t) ∩ b, ascending e} g_e(t) * E_e(h_t)

and the anchor merges ``partial_b`` in ascending bundle id. Because each bundle's
inputs are fixed for a layer step (the dispatcher sends a whole bundle to one
replica), and the reduction order is fixed, a step served by a standby, a hedge or
the anchor's fallback is bitwise identical to the normal step on the same device
type and software stack. That turns "zero corrupted sequences" into an equality test.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

ACTS = {
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}


@dataclass(frozen=True)
class ExpertSpec:
    hidden: int
    inter: int
    act: str = "silu"
    dtype: str = "float32"

    def expert_bytes(self) -> int:
        elt = torch.empty(0, dtype=getattr(torch, self.dtype)).element_size()
        return 3 * self.hidden * self.inter * elt


@dataclass
class BundleWeights:
    """One placement unit: a group of experts from one layer.

    gate_up: (n, 2*I, D)   down: (n, D, I)   -- the HF v5 fused-expert layout.
    """

    bundle_id: int
    layer: int
    experts: list[int]
    gate_up: torch.Tensor
    down: torch.Tensor
    _slot: dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._slot = {e: i for i, e in enumerate(self.experts)}

    @property
    def nbytes(self) -> int:
        return self.gate_up.nbytes + self.down.nbytes

    def slot(self, expert: int) -> int:
        return self._slot[expert]

    def to(self, device) -> "BundleWeights":
        return BundleWeights(self.bundle_id, self.layer, list(self.experts),
                             self.gate_up.to(device), self.down.to(device))

    def checksum(self) -> str:
        return checksum(self.gate_up, self.down)


def checksum(*tensors: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().contiguous().cpu()
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def bundle_forward(w: BundleWeights, act: str, hidden: torch.Tensor, rows: torch.Tensor,
                   experts: torch.Tensor, gates: torch.Tensor, chunk_rows: int = 4096):
    """Compute one bundle's partial sums.

    hidden: (n, D) rows available to this call. rows/experts/gates: (P,) one entry
    per (token, expert) pair routed into this bundle; rows index ``hidden``.
    Returns (urows, partial) with urows sorted ascending and partial (len(urows), D).

    ``chunk_rows`` bounds rows per matmul (the kernel-length cap). It is part of the
    numeric contract: every executor must use the same value.
    """
    fn = ACTS[act]
    rows = rows.to(torch.int64)
    experts = experts.to(torch.int64)
    urows = torch.unique(rows, sorted=True)
    partial = torch.zeros(len(urows), hidden.shape[1], dtype=hidden.dtype, device=hidden.device)
    if rows.numel() == 0:
        return urows, partial
    # Canonical pair order: by expert, then row. Independent of how pairs arrived.
    key = experts * (int(rows.max()) + 1) + rows
    order = torch.argsort(key, stable=True)
    rows, experts, gates = rows[order], experts[order], gates[order]
    pos = torch.searchsorted(urows, rows)
    uexp, counts = torch.unique_consecutive(experts, return_counts=True)
    start = 0
    for e, c in zip(uexp.tolist(), counts.tolist()):
        s = w.slot(e)
        gu_w, dn_w = w.gate_up[s], w.down[s]
        for a in range(start, start + c, chunk_rows):
            b = min(a + chunk_rows, start + c)
            r, p, g = rows[a:b], pos[a:b], gates[a:b]
            x = hidden.index_select(0, r.to(hidden.device))
            gate, up = F.linear(x, gu_w).chunk(2, dim=-1)
            y = F.linear(fn(gate) * up, dn_w)
            y = y * g.to(y.device, y.dtype)[:, None]
            p = p.to(partial.device)
            partial[p] = partial[p] + y
        start += c
    return urows, partial


@dataclass
class LayerRoute:
    """Per-bundle (token, expert, gate) pairs for one MoE layer step."""

    bundles: list[int]                      # ascending
    pairs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]  # b -> rows, experts, gates
    urows: dict[int, torch.Tensor]          # b -> sorted unique token rows


def build_route(topk_idx: torch.Tensor, topk_w: torch.Tensor, expert_to_bundle: torch.Tensor) -> LayerRoute:
    """Group a layer's (token, expert) pairs by bundle. expert_to_bundle: (E,) int64."""
    T, k = topk_idx.shape
    idx = topk_idx.reshape(-1).to(torch.int64).cpu()
    gates = topk_w.reshape(-1)
    rows = torch.arange(T, dtype=torch.int64).repeat_interleave(k)
    bun = expert_to_bundle[idx]
    order = torch.argsort(bun * (len(expert_to_bundle) * (T + 1)) + idx * (T + 1) + rows, stable=True)
    rows, idx, bun = rows[order], idx[order], bun[order]
    gates = gates[order.to(gates.device)]
    ub, counts = torch.unique_consecutive(bun, return_counts=True)
    pairs, urows, start = {}, {}, 0
    for b, c in zip(ub.tolist(), counts.tolist()):
        r = rows[start:start + c]
        pairs[b] = (r, idx[start:start + c], gates[start:start + c])
        urows[b] = torch.unique(r, sorted=True)
        start += c
    return LayerRoute(bundles=ub.tolist(), pairs=pairs, urows=urows)


def merge_partials(T: int, D: int, dtype, device, route: LayerRoute,
                   partials: dict[int, torch.Tensor]) -> torch.Tensor:
    """Sum bundle partials into (T, D) in ascending bundle id — the fixed reduction order."""
    out = torch.zeros(T, D, dtype=dtype, device=device)
    for b in route.bundles:
        r = route.urows[b].to(device)
        out[r] = out[r] + partials[b].to(device, dtype)
    return out


def split_experts(num_experts: int, bundle_size: int) -> list[list[int]]:
    """v1 bundling: contiguous expert ranges. Replace with co-activation clusters
    once routing traces exist (experiment 2)."""
    return [list(range(a, min(a + bundle_size, num_experts))) for a in range(0, num_experts, bundle_size)]
