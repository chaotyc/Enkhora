"""Anchor: trunk + KV cache + gate/dispatch + expert cache + master store server.

All per-request state lives here. Donors only ever see hidden rows and return
partial sums, so donor loss can cost latency but never corrupt a sequence.
"""

from __future__ import annotations

import asyncio
import logging

import torch

from ..model.hf import generate, patch_model, set_executor
from ..routing import RoutingTable
from ..store import MasterStore
from ..transport import LoopThread, RpcClient, serve, server_port
from .dispatcher import DispatchConfig, Dispatcher
from .fallback import FallbackStore, LocalExecutor

log = logging.getLogger(__name__)


class Anchor:
    def __init__(self, model, master: MasterStore, control: tuple[str, int] | None = None,
                 host: str = "127.0.0.1", port: int = 0, device: str = "cpu",
                 cfg: DispatchConfig | None = None, cache_bundles: int = 16,
                 anchor_id: str = "anchor-0", advertise_host: str | None = None):
        self.id = anchor_id
        self.master = master
        self.model = patch_model(model)
        self.device = device
        self.loop = LoopThread("anchor-io")
        self.fallback = FallbackStore(master, device, cache_bundles=cache_bundles)
        self.dispatcher = Dispatcher(master, self.fallback, self.loop, cfg, on_quarantine=self._report_quarantine)
        set_executor(self.model, self.dispatcher)
        self.control_addr = control
        self.host, self.port = host, port
        self.advertise_host = advertise_host or host
        self._control: RpcClient | None = None
        self.loop.call(self._start())

    async def _start(self):
        self.server = await serve(self.host, self.port, self._on_message)
        self.port = server_port(self.server)
        if self.control_addr:
            self._control_task = asyncio.get_running_loop().create_task(self._control_loop())

    # ------------------------------------------------------------ master store server
    async def _on_message(self, conn, h, t):
        if h.get("op") == "fetch":
            b = int(h["bundle"])
            w = await asyncio.get_running_loop().run_in_executor(None, self.master.load, b)
            await conn.send({"rid": h["rid"], "ok": True, "layer": w.layer, "experts": w.experts},
                            {"gate_up": w.gate_up, "down": w.down})
        elif h.get("op") == "ping":
            await conn.send({"rid": h["rid"], "ok": True})

    # ------------------------------------------------------------ control plane link
    async def _control_loop(self):
        host, port = self.control_addr
        while True:
            try:
                self._control = RpcClient(host, port, on_push=self._on_push)
                await self._control.connect(timeout=2.0)
                await self._control.push({"op": "hello_anchor", "anchor": self.id,
                                          "master": [self.advertise_host, self.port],
                                          "bundles": self.master.manifest()})
                while self._control.connected:
                    await asyncio.sleep(0.2)
                log.warning("control plane lost; decoding continues on epoch %d", self.dispatcher.table.epoch)
            except (OSError, ConnectionError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(1.0)

    def _on_push(self, h, t):
        if h.get("op") == "routing":
            table = RoutingTable.from_json(h["table"])
            if table.epoch >= self.dispatcher.table.epoch:
                self.dispatcher.set_table(table)

    def _report_quarantine(self, donor: str):
        if self._control and self._control.connected:
            self.loop.submit(self._control.push({"op": "quarantine", "donor": donor}))

    # ------------------------------------------------------------ serving
    def generate(self, prompts: list[list[int]], max_new_tokens: int, pad_id: int = 0):
        return generate(self.model, prompts, max_new_tokens, pad_id=pad_id)

    def generate_local(self, prompts, max_new_tokens, pad_id: int = 0, executor: LocalExecutor | None = None):
        """Reference: every bundle on this process with the same kernel contract.
        Pass ``executor`` to reuse one (e.g. with all bundles preloaded on the device)."""
        ref = executor or LocalExecutor(self.master, self.device, self.dispatcher.cfg.chunk_rows)
        set_executor(self.model, ref)
        try:
            return generate(self.model, prompts, max_new_tokens, pad_id=pad_id)
        finally:
            set_executor(self.model, self.dispatcher)

    def wait_for_coverage(self, min_replicas: int = 1, timeout: float = 30.0) -> bool:
        import time

        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            tab = self.dispatcher.table
            if all(len(tab.replicas_for(b)) >= min_replicas for b in self.master.bundles) and \
                    all(self.dispatcher.links.get(d) and self.dispatcher.links[d].usable(time.monotonic())
                        for d in tab.donors):
                return True
            time.sleep(0.05)
        return False

    def close(self):
        async def _close():
            if self._control:
                await self._control.close()
            for link in self.dispatcher.links.values():
                if link.client:
                    await link.client.close()
            self.server.close()

        try:
            self.loop.call(_close(), timeout=5)
        except Exception:
            pass
        self.loop.stop()


def make_anchor_model(name: str, master_dir: str, bundle_size: int, seed: int = 0, device="cpu",
                      dtype: str | None = None):
    """Load a model and its master store, exporting the store if it does not exist yet.

    The experts are dropped (``patch_model``) before the move to ``device``, so only the
    trunk ever reaches the anchor's GPU."""
    import os

    from ..model.hf import export_master, load_model

    model = load_model(name, seed=seed, dtype=dtype, device="cpu")
    if not os.path.exists(os.path.join(master_dir, "manifest.json")):
        export_master(model, master_dir, bundle_size, {"name": name, "seed": seed})
    master = MasterStore(master_dir)
    return patch_model(model).to(device), master


@torch.no_grad()
def compare(a, b) -> dict:
    (ta, la), (tb, lb) = a, b
    return {"tokens_equal": bool(torch.equal(ta, tb)), "logits_bitwise": bool(torch.equal(la, lb)),
            "max_abs_logit_diff": float((la - lb).abs().max())}
