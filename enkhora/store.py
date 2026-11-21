"""Master store: the full expert tier on the anchor's local disk.

This is the only copy the system depends on. Donor replicas are caches of it, so
reclaiming a donor is always a drop, never a save.

    <root>/manifest.json
    <root>/bundles/b000042.safetensors   (tensors: gate_up, down)
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch
from safetensors.torch import load_file, save_file

from .experts import BundleWeights, ExpertSpec


@dataclass
class BundleMeta:
    id: int
    layer: int
    experts: list[int]
    nbytes: int
    sha256: str
    file: str
    heat: float = 1.0


class MasterStore:
    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "manifest.json")) as f:
            m = json.load(f)
        self.spec = ExpertSpec(**m["spec"])
        self.num_experts: int = m["num_experts"]
        self.bundle_size: int = m["bundle_size"]
        self.model: dict = m.get("model", {})
        self.bundles: dict[int, BundleMeta] = {b["id"]: BundleMeta(**b) for b in m["bundles"]}
        self.layers: list[int] = sorted({b.layer for b in self.bundles.values()})

    def load(self, bundle_id: int) -> BundleWeights:
        meta = self.bundles[bundle_id]
        t = load_file(os.path.join(self.root, meta.file))
        return BundleWeights(meta.id, meta.layer, list(meta.experts), t["gate_up"], t["down"])

    def expert_to_bundle(self, layer: int) -> torch.Tensor:
        """(E,) map from expert index to global bundle id for one layer."""
        m = torch.full((self.num_experts,), -1, dtype=torch.int64)
        for b in self.bundles.values():
            if b.layer == layer:
                m[torch.tensor(b.experts)] = b.id
        if (m < 0).any():
            raise ValueError(f"layer {layer} has unbundled experts")
        return m

    def manifest(self) -> list[dict]:
        return [asdict(b) for b in self.bundles.values()]

    def tier_bytes(self) -> int:
        return sum(b.nbytes for b in self.bundles.values())

    @staticmethod
    def write(root: str, spec: ExpertSpec, num_experts: int, bundle_size: int,
              bundles: Iterable[BundleWeights], model: dict | None = None) -> "MasterStore":
        os.makedirs(os.path.join(root, "bundles"), exist_ok=True)
        metas = []
        for w in bundles:
            rel = os.path.join("bundles", f"b{w.bundle_id:06d}.safetensors")
            save_file({"gate_up": w.gate_up.contiguous(), "down": w.down.contiguous()},
                      os.path.join(root, rel))
            metas.append(asdict(BundleMeta(w.bundle_id, w.layer, list(w.experts), w.nbytes, w.checksum(), rel)))
        with open(os.path.join(root, "manifest.json"), "w") as f:
            json.dump({"spec": asdict(spec), "num_experts": num_experts, "bundle_size": bundle_size,
                       "model": model or {}, "bundles": metas}, f, indent=1)
        return MasterStore(root)
