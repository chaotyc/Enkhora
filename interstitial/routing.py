"""Versioned routing table: bundle -> ordered replica list (primary first).

Anchors swap tables between layers, never inside one. Every data-plane message
carries the epoch it was routed under; a donor that no longer holds a bundle
refuses, so a stale table costs one retry inside the same layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoutingTable:
    epoch: int = 0
    replicas: dict[int, tuple[str, ...]] = field(default_factory=dict)
    donors: dict[str, tuple[str, int]] = field(default_factory=dict)   # donor id -> exec address

    def replicas_for(self, bundle: int) -> tuple[str, ...]:
        return self.replicas.get(bundle, ())

    def to_json(self) -> dict:
        return {"epoch": self.epoch,
                "replicas": {str(b): list(r) for b, r in self.replicas.items()},
                "donors": {d: list(a) for d, a in self.donors.items()}}

    @staticmethod
    def from_json(j: dict) -> "RoutingTable":
        return RoutingTable(epoch=j["epoch"],
                            replicas={int(b): tuple(r) for b, r in j["replicas"].items()},
                            donors={d: (a[0], int(a[1])) for d, a in j["donors"].items()})

    def coverage(self, bundles) -> dict:
        n = len(bundles)
        counts = [len(self.replicas.get(b, ())) for b in bundles]
        return {"bundles": n,
                "ge1": sum(c >= 1 for c in counts),
                "ge2": sum(c >= 2 for c in counts),
                "replicas": sum(counts)}
