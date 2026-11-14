"""Rungs 3-5 of the fallback ladder, and the all-local reference executor.

    rung 3  anchor expert cache   bundle already on the anchor device
    rung 4  anchor host RAM       copy over PCIe (no-op on CPU)
    rung 5  master on disk        read, copy, execute

Weights come to the anchor on these rungs. Slower, but needs nothing from the rest
of the cluster, so the floor of the system is an ordinary offloading MoE server.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import torch

from ..experts import BundleWeights, build_route, bundle_forward, merge_partials
from ..store import MasterStore


class FallbackStore:
    def __init__(self, master: MasterStore, device="cpu", cache_bundles: int = 16,
                 ram_bundles: int | None = None):
        self.master = master
        self.device = torch.device(device)
        self.cache_bundles = cache_bundles
        self.ram_bundles = ram_bundles
        self._cache: OrderedDict[int, BundleWeights] = OrderedDict()
        self._ram: OrderedDict[int, BundleWeights] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = {"cache": 0, "ram": 0, "disk": 0}

    def _lru_put(self, d: OrderedDict, cap, key, val):
        d[key] = val
        d.move_to_end(key)
        while cap is not None and len(d) > cap:
            d.popitem(last=False)

    def get(self, b: int) -> tuple[BundleWeights, str]:
        with self._lock:
            if b in self._cache:
                self._cache.move_to_end(b)
                self.hits["cache"] += 1
                return self._cache[b], "cache"
            rung = "ram"
            w = self._ram.get(b)
            if w is None:
                rung = "disk"
                w = self.master.load(b)
                if self.device.type == "cuda":
                    w = BundleWeights(w.bundle_id, w.layer, w.experts, w.gate_up.pin_memory(), w.down.pin_memory())
                self._lru_put(self._ram, self.ram_bundles, b, w)
            else:
                self._ram.move_to_end(b)
            dw = w.to(self.device)
            if self.cache_bundles > 0:
                self._lru_put(self._cache, self.cache_bundles, b, dw)
            self.hits[rung] += 1
            return dw, rung

    def preload_ram(self):
        for b in self.master.bundles:
            if b not in self._ram:
                self._lru_put(self._ram, self.ram_bundles, b, self.master.load(b))


class LocalExecutor:
    """Executes every bundle on this process. The bitwise reference for the dispatcher,
    and the 'every donor gone' floor."""

    def __init__(self, master: MasterStore, device="cpu", chunk_rows: int = 4096, cache_bundles: int | None = None):
        self.master = master
        self.act = master.spec.act
        self.chunk_rows = chunk_rows
        n = len(master.bundles) if cache_bundles is None else cache_bundles
        self.store = FallbackStore(master, device, cache_bundles=n)
        self.e2b = {layer: master.expert_to_bundle(layer) for layer in master.layers}

    def run_layer(self, layer, hidden, topk_idx, topk_w):
        route = build_route(topk_idx, topk_w, self.e2b[layer])
        partials = {}
        for b in route.bundles:
            w, _ = self.store.get(b)
            rows, ex, g = route.pairs[b]
            _, partials[b] = bundle_forward(w, self.act, hidden, rows, ex, g, self.chunk_rows)
        return merge_partials(hidden.shape[0], hidden.shape[1], hidden.dtype, hidden.device, route, partials)
