"""Slab manager: donor VRAM held as one slab per bundle replica.

Lifecycle: loading -> ready (checksum verified) -> gone. Draining happens at the
control plane (the replica leaves the routing table before the drop is sent).
A hard eviction jumps straight from ready to gone; in-flight calls holding a
reference finish (bounded by the kernel-length cap) and then the memory is freed.

v1 allocates each slab as ordinary tensors. Fixed-size chunked slabs (so a free is
a handful of calls and never fragments) are a v2 item.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from ..experts import BundleWeights


@dataclass
class Slab:
    bundle: int
    nbytes: int
    state: str = "loading"          # loading | ready
    role: str = "standby"           # primary | standby
    weights: BundleWeights | None = None


class SlabManager:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self._slabs: dict[int, Slab] = {}
        self._lock = threading.Lock()
        self.evictions = 0

    def reserve(self, bundle: int, nbytes: int, role: str = "standby") -> bool:
        with self._lock:
            if bundle in self._slabs:
                return False
            self._slabs[bundle] = Slab(bundle, nbytes, "loading", role)
            return True

    def set_ready(self, bundle: int, weights: BundleWeights) -> bool:
        with self._lock:
            s = self._slabs.get(bundle)
            if s is None:               # dropped while loading
                return False
            s.weights = weights.to(self.device)
            s.nbytes = weights.nbytes
            s.state = "ready"
            return True

    def get_ready(self, bundle: int) -> BundleWeights | None:
        with self._lock:
            s = self._slabs.get(bundle)
            return s.weights if s is not None and s.state == "ready" else None

    def set_roles(self, roles: dict[int, str]):
        with self._lock:
            for b, r in roles.items():
                if b in self._slabs:
                    self._slabs[b].role = r

    def drop(self, bundle: int) -> bool:
        with self._lock:
            s = self._slabs.pop(bundle, None)
        if s is not None:
            self._release()
        return s is not None

    def drop_all(self) -> int:
        with self._lock:
            n = len(self._slabs)
            self._slabs.clear()
        self._release()
        return n

    def evict_until(self, need_free: int, free_now) -> list[int]:
        """Hard-evict standbys first, then primaries, until ``free_now() >= need_free``."""
        out = []
        while free_now() < need_free:
            with self._lock:
                if not self._slabs:
                    break
                victim = min(self._slabs.values(),
                             key=lambda s: (s.role == "primary", s.state == "ready", -s.nbytes, s.bundle))
                del self._slabs[victim.bundle]
            out.append(victim.bundle)
            self.evictions += 1
            self._release()
        return out

    def _release(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def used_bytes(self) -> int:
        with self._lock:
            return sum(s.nbytes for s in self._slabs.values())

    def snapshot(self) -> dict[int, str]:
        with self._lock:
            return {b: s.state for b, s in self._slabs.items()}

    def roles(self) -> dict[int, str]:
        with self._lock:
            return {b: s.role for b, s in self._slabs.items()}
