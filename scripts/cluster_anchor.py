#!/usr/bin/env python3
"""Anchor-side driver for scripts/cluster.py: stage 1 (fabric), 2 (distributed decode), 3 (chaos).

Runs on node 0 as the anchor process: dense trunk, KV cache, dispatcher and master-store
server, built from the library (Anchor, LocalExecutor, compare, Dispatcher.report,
admin_async). Donors and the control plane are separate ``python -m enkhora`` processes
started by the orchestrator. Progress lines start with ``[``; each stage ends with one
``@@RESULT {json}`` line, and larger outputs with ``@@ARTIFACT <name> <json>``. The
orchestrator parses both from this role's log, so nothing needs a shared filesystem.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from cluster import unpack  # noqa: E402
from enkhora.anchor.anchor import Anchor, compare  # noqa: E402
from enkhora.anchor.dispatcher import DispatchConfig  # noqa: E402
from enkhora.anchor.fallback import LocalExecutor  # noqa: E402
from enkhora.control.ctl import admin_async  # noqa: E402
from enkhora.experts import bundle_forward  # noqa: E402
from enkhora.model.hf import (export_master, find_moe_blocks, generate, load_model, patch_model,  # noqa: E402
                              set_executor, tiny_qwen3_moe)
from enkhora.store import MasterStore  # noqa: E402
from enkhora.transport import LoopThread, RpcClient  # noqa: E402

TEXT_PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue:",
    "Write a haiku about distributed computing:",
    "def fibonacci(n):",
    "The three laws of thermodynamics are",
    "Translate to German: The weather is nice today.",
    "A mixture-of-experts language model routes each token to",
    "Q: What is 17 times 23?\nA:",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def emit(stage, status, headline="", reason="", **kw):
    print("@@RESULT " + json.dumps({"stage": stage, "status": status, "headline": headline, "reason": reason, **kw},
                                   default=str), flush=True)


def artifact(name, obj):
    print(f"@@ARTIFACT {name} {json.dumps(obj, default=str)}", flush=True)


def stats_ms(xs):
    if not xs:
        return {"n": 0, "p50": None, "p99": None}
    s = sorted(xs)
    return {"n": len(s), "p50": round(s[len(s) // 2], 4), "p99": round(s[int(0.99 * (len(s) - 1))], 4),
            "min": round(s[0], 4)}


def tiny_olmoe(seed=0):
    """A small random-weight OLMoE with the real HF architecture (exercises the OLMoE patch path)."""
    from transformers import OlmoeConfig, OlmoeForCausalLM

    cfg = OlmoeConfig(vocab_size=256, hidden_size=64, intermediate_size=32, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, num_experts=32, num_experts_per_tok=4,
                      max_position_embeddings=512, pad_token_id=0, bos_token_id=1, eos_token_id=None,
                      tie_word_embeddings=False, initializer_range=0.08)
    torch.manual_seed(seed)
    model = OlmoeForCausalLM(cfg)
    g = torch.Generator().manual_seed(seed + 1)
    for _, block in find_moe_blocks(model):
        w = block.gate.weight
        w.data = torch.randn(w.shape, generator=g) / cfg.hidden_size ** 0.5 * 4
    return model.eval()


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class Driver:
    def __init__(self, plan):
        self.p = plan
        self.io = LoopThread("driver-io")
        self.ctrl = tuple(plan["control"])
        self.donors = {d["id"]: d for d in plan["donors"]}
        self.agents = {a["node"]: a for a in plan["agents"]}
        self.device = plan["device"]
        self.mixed = bool(plan["mixed"])
        self.atol = float(plan["audit_atol"])
        self.anchor = self.model = self.master = self.tok = self.ref_exec = None
        self.model_error = None
        self.placement = None

    # ------------------------------------------------------------ plumbing
    def call(self, coro, timeout=30.0):
        return self.io.call(coro, timeout)

    def admin(self, cmd, **kw):
        return self.call(admin_async(self.ctrl, cmd, **kw), 20)

    def status(self):
        return self.admin("status")["status"]

    def agent(self, node: int, **req) -> dict:
        addr = tuple(self.agents[node]["addr"])

        async def go():
            c = await RpcClient(*addr).connect(5)
            try:
                h, _ = await asyncio.wait_for(c.request(req), 20)
                return h
            finally:
                await c.close()

        return self.call(go(), 30)

    def signal(self, role: str, sig: str) -> dict:
        node = self.donors[role]["node"] if role in self.donors else 0
        r = self.agent(node, op="signal", roles=[role], sig=sig)
        if not r.get("signaled"):
            raise RuntimeError(f"agent could not {sig} {role}: {r.get('errors') or r}")
        return r

    # ------------------------------------------------------------ stage 1a: transport only
    def wait_donors(self):
        deadline = time.monotonic() + self.p["timeouts"]["startup"]
        pending = set(self.donors)
        log(f"waiting for {len(pending)} donor(s) to listen (<= {self.p['timeouts']['startup']:.0f}s)")

        async def ping(addr):
            c = await RpcClient(*addr).connect(2)
            try:
                await asyncio.wait_for(c.request({"op": "ping"}), 5)
            finally:
                await c.close()

        while pending and time.monotonic() < deadline:
            for d in sorted(pending):
                try:
                    self.call(ping(tuple(self.donors[d]["addr"])), 10)
                    pending.discard(d)
                except Exception:  # noqa: BLE001
                    pass
            if pending:
                time.sleep(1.0)
        return sorted(pending)

    async def _probe(self, addr, tokens, bpt, iters):
        c = await RpcClient(*addr).connect(5)
        try:
            for _ in range(10):
                await asyncio.wait_for(c.request({"op": "ping"}), 10)
            ping = []
            for _ in range(iters):
                t0 = time.perf_counter()
                await asyncio.wait_for(c.request({"op": "ping"}), 10)
                ping.append((time.perf_counter() - t0) * 1e3)
            payload, raw = {}, {}
            for T in tokens:
                x = torch.zeros(T, max(1, bpt // 2), dtype=torch.bfloat16)
                rt = []
                for i in range(max(10, iters // 2) + 3):
                    t0 = time.perf_counter()
                    await asyncio.wait_for(c.request({"op": "ping"}, {"x": x}), 20)
                    if i >= 3:
                        rt.append((time.perf_counter() - t0) * 1e3)
                key = f"{T * bpt // 1024}KiB"
                payload[key] = {"tokens": T, "bytes": T * bpt, **stats_ms(rt)}
                raw[key] = rt
            return {"ping": stats_ms(ping), "payload": payload}, ping, raw
        finally:
            await c.close()

    def fabric_transport(self):
        f = self.p["fabric"]
        links, raw = [], {}
        for d in sorted(self.donors.values(), key=lambda d: d["id"]):
            try:
                res, ping, praw = self.call(self._probe(tuple(d["addr"]), f["tokens"], f["bytes_per_token"],
                                                        f["iters"]), 600)
                raw[d["id"]] = (ping, praw)
                links.append({"id": d["id"], "node": d["node_name"], "gpu": d["gpu"], "remote": d["remote"], **res})
                log(f"fabric {d['id']:<22} ping p50 {res['ping']['p50']:.3f} ms p99 {res['ping']['p99']:.3f} ms; "
                    + ", ".join(f"{k} p50 {v['p50']:.3f}" for k, v in res["payload"].items()))
            except Exception as e:  # noqa: BLE001
                links.append({"id": d["id"], "node": d["node_name"], "gpu": d["gpu"], "remote": d["remote"],
                              "error": repr(e)})
                log(f"fabric {d['id']}: FAILED {e!r}")
        return links, raw

    @staticmethod
    def groups(links, raw):
        out = {}
        for x in links:
            if x["id"] not in raw:
                continue
            g = f"{x['gpu']} / {'remote node' if x['remote'] else 'anchor node'}"
            e = out.setdefault(g, {"links": 0, "ping": [], "payload": []})
            ping, praw = raw[x["id"]]
            e["links"] += 1
            e["ping"] += ping
            if praw:
                e["payload"] += praw[list(praw)[-1]]
        res = {}
        for g, e in out.items():
            ps, pl = stats_ms(e["ping"]), stats_ms(e["payload"])
            res[g] = {"links": e["links"], "ping_p50": ps["p50"], "ping_p99": ps["p99"],
                      "payload_p50": pl["p50"], "payload_p99": pl["p99"]}
        return res

    # ------------------------------------------------------------ model + anchor
    def build(self):
        p = self.p
        name, bs, seed = p["model"], p["bundle_size"], p["seed"]
        log(f"loading model {name} (dtype {p['dtype'] or 'default'}) on CPU")
        t0 = time.monotonic()
        if name == "tiny":
            model = tiny_qwen3_moe(seed=seed)
        elif name == "tiny-olmoe":
            model = tiny_olmoe(seed=seed)
        else:
            model = load_model(name, seed=seed, dtype=p["dtype"], device="cpu")
            from transformers import AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(name)
        md = p["master_dir"]
        if os.path.exists(os.path.join(md, "manifest.json")):
            ms = MasterStore(md)
            if ms.bundle_size != bs or ms.model.get("name") != name:
                raise RuntimeError(f"master store {md} is for {ms.model.get('name')} bs {ms.bundle_size}; "
                                   f"pass another --master-dir")
            log(f"reusing master store {md}")
        else:
            tmp = f"{md}.tmp-{os.getpid()}"
            log(f"exporting master store to {md}")
            export_master(model, tmp, bs, {"name": name, "seed": seed})
            if os.path.exists(md):
                shutil.rmtree(md)
            os.replace(tmp, md)
        self.master = MasterStore(md)
        patch_model(model)                 # drop expert weights before the trunk goes to the GPU
        self.model = model.to(self.device)
        tier = self.master.tier_bytes()
        log(f"model ready in {time.monotonic() - t0:.1f}s: {len(self.master.layers)} MoE layers, "
            f"{len(self.master.bundles)} bundles, tier {tier / 2**20:.1f} MiB")
        cfg = DispatchConfig(audit_rate=0.0, audit_atol=self.atol if self.mixed else 0.0)
        self.anchor = Anchor(self.model, self.master, control=self.ctrl, host=p["bind_host"], port=p["anchor_port"],
                             device=self.device, cfg=cfg, advertise_host=p["advertise"])
        log(f"anchor serving the master store on {p['bind_host']}:{self.anchor.port} "
            f"(advertised {p['advertise']}); control plane {self.ctrl[0]}:{self.ctrl[1]}")

    def wait_placement(self):
        t0 = time.monotonic()
        timeout = self.p["timeouts"]["placement"]
        ok = self.anchor.wait_for_coverage(1, timeout)
        st, quiet = None, 0
        jobs = len({d["job"] for d in self.donors.values()})
        while time.monotonic() - t0 < timeout:
            st = self.status()
            loading = any(d["loading"] for d in st["donors"].values())
            cov = st["coverage"]
            if not loading and cov["ge1"] == cov["bundles"] and (cov["ge2"] == cov["bundles"] or jobs < 2):
                break
            quiet = quiet + 1 if not loading else 0
            if quiet >= 10:
                break                  # nothing loading and nothing more coming
            time.sleep(0.5)
        self.anchor.wait_for_coverage(1, 30)
        st = st or self.status()
        secs = time.monotonic() - t0
        loaded = st["counters"]["bytes_issued"]
        self.placement = {"seconds": round(secs, 1), "coverage": st["coverage"], "epoch": st["epoch"],
                          "tier_MiB": round(st["tier_bytes"] / 2**20, 1), "loaded_MiB": round(loaded / 2**20, 1),
                          "fill_MBps": round(loaded / 1e6 / max(secs, 1e-3), 1),
                          "donors": {k: {"ready": len(v["ready"]), "primary": len(v["primary_for"])}
                                     for k, v in st["donors"].items()},
                          "full_coverage": bool(ok)}
        log(f"placement after {secs:.1f}s: {st['coverage']}, loaded {loaded / 2**20:.1f} MiB")
        if not any(v["primary_for"] for v in st["donors"].values()):
            nb = len(self.master.bundles)
            self.placement["warning"] = (f"planner assigned no compute primaries: expert demand "
                                         f"{self.p['planner']['expert_demand']} is too high for {nb} bundles "
                                         f"(each bundle costs more than one donor's kappa); pass --expert-demand "
                                         f"<= {0.5 * nb:g}. Standby replicas still serve.")
            log("WARNING: " + self.placement["warning"])
        return ok

    # ------------------------------------------------------------ stage 1b: real exec round trips
    async def _exec_probe(self, addr, b, tokens, iters):
        meta = self.master.bundles[b]
        spec = self.master.spec
        dtype = getattr(torch, spec.dtype)
        w = self.master.load(b).to(self.device)
        c = await RpcClient(*addr).connect(5)
        out, worst = {}, 0.0
        bitwise = True
        try:
            for T in tokens:
                g = torch.Generator().manual_seed(T)
                hidden = (torch.randn(T, spec.hidden, generator=g) * 0.5).to(dtype)
                k = min(2, len(meta.experts))
                rows = torch.arange(T, dtype=torch.int32).repeat_interleave(k)
                exps = torch.tensor([meta.experts[(t + j) % len(meta.experts)] for t in range(T) for j in range(k)],
                                    dtype=torch.int32)
                gates = torch.rand(T * k, generator=g).to(dtype)
                header = {"op": "exec", "epoch": 0, "layer": meta.layer, "bundles": [b], "act": spec.act,
                          "chunk_rows": self.anchor.dispatcher.cfg.chunk_rows}
                tens = {"hidden": hidden, "rows": rows, "experts": exps, "gates": gates,
                        "offsets": torch.tensor([0, T * k], dtype=torch.int64)}
                rtt, net, refused, last = [], [], 0, None
                for i in range(max(10, iters // 4) + 3):
                    t0 = time.perf_counter()
                    h, t = await asyncio.wait_for(c.request(header, tens), 30)
                    dt = (time.perf_counter() - t0) * 1e3
                    if not h.get("ok"):
                        refused += 1
                        continue
                    last = (h, t)
                    if i >= 3:
                        rtt.append(dt)
                        net.append(dt - h.get("exec_ms", 0.0) - h.get("wait_ms", 0.0))
                entry = {**stats_ms(rtt), "net_p50": stats_ms(net)["p50"], "refused": refused}
                if last is not None:
                    h, t = last
                    if b in h.get("done", []):
                        o = t["out"][t["offsets"][0]:t["offsets"][1]]
                        _, ref = bundle_forward(w, spec.act, hidden.to(self.device), rows, exps, gates,
                                                self.anchor.dispatcher.cfg.chunk_rows)
                        ref = ref.cpu()
                        same = torch.equal(o, ref)
                        diff = float((o.float() - ref.float()).abs().max()) if o.numel() else 0.0
                        entry.update(bitwise=same, max_abs_diff=diff)
                        bitwise &= same
                        worst = max(worst, diff)
                    else:
                        entry["missing"] = True
                out[str(T)] = entry
            t0 = time.perf_counter()
            mbps = []
            for _ in range(3):
                t0 = time.perf_counter()
                h, t = await asyncio.wait_for(c.request({"op": "fetch", "bundle": b}), 120)
                if h.get("ok"):
                    mbps.append(meta.nbytes / 1e6 / (time.perf_counter() - t0))
            return out, (max(mbps) if mbps else None), bitwise, worst
        finally:
            await c.close()

    def fabric_exec(self, links):
        st = self.status()
        f = self.p["fabric"]
        by_id = {x["id"]: x for x in links}
        problems = []
        for did, d in sorted(st["donors"].items()):
            if did not in by_id or not d["ready"]:
                continue
            b = d["ready"][0]
            try:
                ex, mbps, bitwise, worst = self.call(self._exec_probe(tuple(self.donors[did]["addr"]), b, f["tokens"],
                                                                      f["iters"]), 900)
                by_id[did].update(exec=ex, exec_bundle=b, fetch_MBps=mbps, exec_bitwise=bitwise, exec_max_abs_diff=worst)
                ok = bitwise or (self.mixed and worst <= self.atol)
                if not ok:
                    problems.append(f"{did}: exec output differs from the anchor's (max {worst:g})")
                log(f"exec   {did:<22} bundle {b}: " + ", ".join(
                    f"T{k} p50 {v['p50']:.3f} ms (net {v['net_p50']:.3f})" for k, v in ex.items() if v["p50"]) +
                    f"; fetch {mbps or 0:.0f} MB/s; output {'bitwise' if bitwise else f'max diff {worst:g}'}")
            except Exception as e:  # noqa: BLE001
                problems.append(f"{did}: exec probe failed {e!r}")
        return problems

    # ------------------------------------------------------------ decode helpers
    def prompts(self, B, seed=0):
        if self.tok is not None:
            out = []
            for i in range(B):
                ids = list(self.tok(TEXT_PROMPTS[i % len(TEXT_PROMPTS)])["input_ids"])
                out.append([int(x) for x in ids] or [1])
            return out
        rng = random.Random(seed * 7919 + B)
        V = min(self.model.config.vocab_size, 256)
        return [[1] + [rng.randrange(2, V) for _ in range(rng.randint(3, 12))] for _ in range(B)]

    def gen_local(self, prompts, max_new):
        if self.ref_exec is None:
            n = self.p.get("ref_cache_bundles")
            if n is None and self.device.startswith("cuda"):
                free = torch.cuda.mem_get_info()[0]
                per = self.master.tier_bytes() / max(1, len(self.master.bundles))
                n = max(1, min(len(self.master.bundles), int(0.5 * free / per)))
                log(f"reference executor caches {n}/{len(self.master.bundles)} bundles on {self.device}")
            self.ref_exec = LocalExecutor(self.master, self.device, self.anchor.dispatcher.cfg.chunk_rows,
                                          cache_bundles=n)
        set_executor(self.model, self.ref_exec)
        try:
            return generate(self.model, prompts, max_new)
        finally:
            set_executor(self.model, self.anchor.dispatcher)

    def timed(self, fn, *a):
        sync(self.device)
        t0 = time.monotonic()
        r = fn(*a)
        sync(self.device)
        return r, time.monotonic() - t0

    def check(self, ref, out) -> dict:
        r = compare(ref, out)
        if not self.mixed:
            r["ok"] = r["tokens_equal"] and r["logits_bitwise"]
            r["mode"] = "bitwise"
            return r
        (ta, la), (tb, lb) = ref, out
        div = (ta != tb).any(dim=0).nonzero()
        first = int(div[0]) if len(div) else ta.shape[1]
        steps = min(first + 1, la.shape[0])
        d = float((la[:steps].float() - lb[:steps].float()).abs().max())
        r.update(mode=f"atol {self.atol}", first_divergent_step=first if first < ta.shape[1] else None,
                 max_abs_logit_diff_prefix=d, token_match=float((ta == tb).float().mean()), ok=d <= self.atol)
        return r

    def reset_dispatch(self):
        dsp = self.anchor.dispatcher
        dsp.stats.clear()
        dsp.rungs.clear()
        dsp.layer_ms.clear()

    # ------------------------------------------------------------ stage 2
    def stage2(self):
        p, dsp = self.p, self.anchor.dispatcher
        max_new = p["max_new"]
        log("stage 2: warm-up")
        warm = self.prompts(2, seed=99)
        self.anchor.generate(warm, 2)
        self.gen_local(warm, 2)
        B0 = p["correctness_batch"]
        pr = self.prompts(B0, seed=1)
        self.reset_dispatch()
        out, t_out = self.timed(self.anchor.generate, pr, max_new)
        ref, t_ref = self.timed(self.gen_local, pr, max_new)
        main = self.check(ref, out)
        main.update(batch=B0, served_by=dict(dsp.rungs))
        log(f"stage 2: correctness B={B0}: {json.dumps(main)}")
        if self.tok is not None:
            rows = []
            for i in range(min(B0, len(TEXT_PROMPTS))):
                rows.append({"prompt": TEXT_PROMPTS[i], "distributed": self.tok.decode(out[0][i].tolist()),
                             "reference_equal": bool(torch.equal(out[0][i], ref[0][i]))})
            artifact("text_samples.json", rows)
            md = "\n".join(f"### Prompt {i + 1}\n\n> {r['prompt']}\n\n{r['distributed']}\n\n"
                           f"(identical to the all-local reference: {r['reference_equal']})\n" for i, r in enumerate(rows))
            artifact("text_samples.md", md)
        sweep, problems = [], []
        if not main["ok"]:
            problems.append(f"correctness run B={B0} differs: {main}")
        for B in p["batches"]:
            pr = self.prompts(B, seed=B)
            self.reset_dispatch()
            out, t_d = self.timed(self.anchor.generate, pr, max_new)
            rep = dsp.report()
            ref, t_l = self.timed(self.gen_local, pr, max_new)
            c = self.check(ref, out)
            ntok = B * max_new
            row = {"batch": B, "max_new": max_new, "tok_s_dist": ntok / t_d, "tok_s_local": ntok / t_l,
                   "step_ms_dist": 1e3 * t_d / max_new, "layer_ms": rep["layer_ms"], "served_by": rep["served_by"],
                   "tokens_equal": c["tokens_equal"], "logits": "bitwise" if c["logits_bitwise"] else
                   ("within atol" if c["ok"] else "DIFFERENT"),
                   "max_abs_logit_diff": c.get("max_abs_logit_diff_prefix", c["max_abs_logit_diff"]),
                   "hedges": rep["stats"].get("hedges", 0), "link_errors": rep["stats"].get("link_errors", 0),
                   "refusals": rep["stats"].get("refusals", 0), "messages": rep["stats"].get("messages", 0),
                   "ok": c["ok"]}
            sweep.append(row)
            log(f"stage 2: B={B:<3} dist {row['tok_s_dist']:.1f} tok/s, local {row['tok_s_local']:.1f} tok/s, "
                f"layer p50/p99 {rep['layer_ms']['p50']}/{rep['layer_ms']['p99']} ms, {row['logits']}, "
                f"served_by {rep['served_by']}")
            if not c["ok"]:
                problems.append(f"B={B}: output differs ({c})")
            if not rep["served_by"].get("donor"):
                problems.append(f"B={B}: no bundle was served by a donor")
        best = max(sweep, key=lambda r: r["tok_s_dist"]) if sweep else None
        head = (f"{'bitwise' if not self.mixed else f'within atol {self.atol}'} vs all-local: "
                f"{'yes' if not problems else 'NO'}; ")
        if best:
            b1 = sweep[0]
            head += (f"B={best['batch']}: {best['tok_s_dist']:.1f} tok/s dist vs {best['tok_s_local']:.1f} local; "
                     f"B={b1['batch']} layer p50/p99 {b1['layer_ms']['p50']}/{b1['layer_ms']['p99']} ms")
        if self.placement and self.placement.get("warning"):
            head += "; WARNING: " + self.placement["warning"]
        emit(2, "FAIL" if problems else "PASS", head, "; ".join(problems), correctness=main, sweep=sweep,
             placement=self.placement, mixed=self.mixed)

    # ------------------------------------------------------------ stage 3
    def predict_kappa_moves(self, st) -> dict:
        """Bytes the control plane's (pure) planner would load if each primary-holding donor went to
        kappa 0, from the live status. A zero means its standbys can absorb its primaries by a
        routing-table edit alone, which is the design's claim for a kappa shrink."""
        from enkhora.control.planner import PBundle, PDonor, PlannerConfig, plan

        cfg = PlannerConfig(**self.p["planner"])
        B = [PBundle(m.id, m.layer, m.nbytes, m.heat) for m in self.master.bundles.values()]
        Bd = {b.id: b for b in B}
        prev = {b: d for d, v in st["donors"].items() for b in v["primary_for"]}
        out = {}
        for x, vx in st["donors"].items():
            if not vx["primary_for"]:
                continue
            ds = [PDonor(id=d, job=v["job"], mem=int(v["mem"]), kappa=0.0 if d == x else float(v["kappa"]),
                         held={**{int(b): "ready" for b in v["ready"]}, **{int(b): "loading" for b in v["loading"]}})
                  for d, v in st["donors"].items()]
            try:
                out[x] = plan(B, ds, cfg, prev).bytes_to_load(Bd)
            except Exception:  # noqa: BLE001
                out[x] = None
        return out

    def stage3(self):
        p, dsp = self.p, self.anchor.dispatcher
        B, N = p["chaos_batch"], p["chaos_max_new"]
        pr = self.prompts(B, seed=3)
        self.wait_placement()
        log(f"stage 3: reference for chaos run (B={B}, {N} new tokens)")
        ref = self.gen_local(pr, N)
        total = len(self.master.layers) * N
        state = {"kappa": None, "killed": set(), "stopped": None, "left": None}
        check = {}

        def alive(st):
            return [d for d in st["donors"] if d not in state["killed"] and d != state["left"]]

        def pick(st, exclude, prefer_remote=True, need_primary=False):
            c = [d for d in alive(st) if d not in exclude and d in self.donors]
            if need_primary:
                c = [d for d in c if st["donors"][d]["primary_for"]] or c
            c.sort(key=lambda d: (not (self.donors[d]["remote"] == prefer_remote),
                                  -len(st["donors"][d]["primary_for"]), -len(st["donors"][d]["ready"]), d))
            return c[0] if c else None

        def ev_kappa():
            st = self.status()
            pred = self.predict_kappa_moves(st)
            zero = [d for d, b in pred.items() if b == 0]
            k = pick(st, set(st["donors"]) - set(zero), prefer_remote=False, need_primary=True) if zero else \
                pick(st, set(), prefer_remote=False, need_primary=True)
            if k is None:
                return "skipped: no donor"
            check["kappa_predicted_bytes"] = pred.get(k)
            check["kappa_zero_move_possible"] = bool(zero)
            n_prim = len(st["donors"][k]["primary_for"])
            before = st["counters"]["bytes_issued"]
            self.admin("set_kappa", donor=k, kappa=0.0)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 30:
                s2 = self.status()
                d = s2["donors"].get(k)
                if d and d["kappa"] == 0.0 and not d["primary_for"]:
                    break
                time.sleep(0.1)
            s2 = self.status()
            moved = s2["counters"]["bytes_issued"] - before
            state["kappa"] = k
            check["kappa_bytes_moved"] = moved
            check["kappa_swapped"] = not s2["donors"].get(k, {}).get("primary_for")
            return f"{k}: kappa 1 -> 0, primaries {n_prim} -> {len(s2['donors'].get(k, {}).get('primary_for', []))}, " \
                   f"bytes moved {moved} (planner predicted {pred.get(k)})"

        def ev_kill():
            st = self.status()
            x = pick(st, {state["kappa"]}, prefer_remote=True, need_primary=True)
            if x is None:
                return "skipped: needs a second donor"
            self.signal(x, "KILL")
            state["killed"].add(x)
            return f"SIGKILL {x} on {self.donors[x]['node_name']} (primary for {len(st['donors'][x]['primary_for'])})"

        def ev_stop():
            st = self.status()
            y = pick(st, {state["kappa"]} | state["killed"], prefer_remote=True)
            if y is None:
                y = pick(st, state["killed"], prefer_remote=True)
            if y is None:
                return "skipped: no donor left to hang"
            self.signal(y, "STOP")
            state["stopped"] = y
            return f"SIGSTOP {y} on {self.donors[y]['node_name']} (hung donor, connection stays open)"

        def ev_cont():
            y = state["stopped"]
            if not y:
                return "skipped: nothing stopped"
            self.signal(y, "CONT")
            state["stopped"] = None
            return f"SIGCONT {y}"

        def ev_leave():
            st = self.status()
            z = state["kappa"] if state["kappa"] in alive(st) else pick(st, state["killed"])
            if z is None:
                return "skipped: no donor to drain"
            self.admin("leave", donor=z)
            t0 = time.monotonic()
            gone = False
            while time.monotonic() - t0 < 60:
                if z not in self.status()["donors"]:
                    gone = True
                    break
                time.sleep(0.2)
            state["left"] = z
            check["leave_completed"] = gone
            return f"planned leave of {z}: {'drained and exited' if gone else 'NOT finished in 60s'} " \
                   f"after {time.monotonic() - t0:.1f}s"

        def ev_control():
            self.signal("control", "KILL")
            return "SIGKILL control plane (decode continues on the last routing epoch)"

        def ev_all():
            killed = []
            for node in sorted({d["node"] for d in self.donors.values()}):
                roles = [d for d, v in self.donors.items() if v["node"] == node]
                r = self.agent(node, op="signal", roles=roles, sig="KILL")
                killed += list(r.get("signaled", {}))
            state["killed"].update(killed)
            return f"SIGKILL every remaining donor ({len(killed)}): anchor floor"

        plan = [(0.06, "set_kappa -> 0 on a primary donor (routing-table edit)", ev_kappa),
                (0.15, "SIGKILL a primary donor (remote node preferred)", ev_kill),
                (0.28, "SIGSTOP another donor (hung)", ev_stop),
                (0.42, "SIGCONT the hung donor", ev_cont),
                (0.52, "ctl leave (planned drain)", ev_leave),
                (0.66, "kill the control plane", ev_control),
                (0.82, "kill all donors", ev_all)]
        at, last = {}, 0
        for frac, name, fn in plan:
            k = max(last + 1, int(frac * total))
            at[k] = (name, fn)
            last = k
        events = []

        def hook(n):
            ev = at.get(n)
            if ev is None:
                return
            name, fn = ev
            t0 = time.monotonic()
            try:
                res = fn()
                status = "skipped" if str(res).startswith("skipped") else "ok"
            except Exception as e:  # noqa: BLE001
                res, status = f"ERROR {e!r}", "error"
            events.append({"layer": n, "event": name, "result": res, "status": status,
                           "blocked_s": round(time.monotonic() - t0, 2)})
            log(f"stage 3: layer call {n:5d}: {name}: {res}")

        self.reset_dispatch()
        dsp.epochs_seen.clear()
        dsp.on_layer = hook
        t0 = time.monotonic()
        try:
            out = self.anchor.generate(pr, N)
        finally:
            dsp.on_layer = None
            if state["stopped"]:
                try:
                    self.signal(state["stopped"], "CONT")
                except Exception:  # noqa: BLE001
                    pass
        dt = time.monotonic() - t0
        c = self.check(ref, out)
        rep = dsp.report()
        problems = []
        if not c["ok"]:
            problems.append("output differs from the reference")
        if len(events) < len(plan):
            problems.append(f"only {len(events)}/{len(plan)} events fired")
        errs = [e for e in events if e["status"] == "error"]
        if errs:
            problems.append(f"{len(errs)} event(s) errored")
        if check.get("kappa_bytes_moved") and check.get("kappa_zero_move_possible"):
            problems.append(f"set_kappa moved {check['kappa_bytes_moved']} bytes (planner predicted "
                            f"{check.get('kappa_predicted_bytes')})")
        elif check.get("kappa_bytes_moved"):
            head_note = (f"no donor could shed its primaries without a load under this placement; "
                         f"set_kappa moved {check['kappa_bytes_moved']} bytes as the planner predicted")
            check["note"] = head_note
        if check.get("kappa_swapped") is False:
            problems.append("set_kappa did not swap primaries")
        if check.get("leave_completed") is False:
            problems.append("planned leave did not finish")
        skipped = [e["event"] for e in events if e["status"] == "skipped"]
        head = (f"{'exact' if c['ok'] else 'NOT EXACT'} ({c['mode']}) through {sum(e['status'] == 'ok' for e in events)}"
                f"/{len(plan)} events in {dt:.1f}s; hedges {rep['stats'].get('hedges', 0)}, circuit opens "
                f"{rep['stats'].get('circuit_open', 0)}, link errors {rep['stats'].get('link_errors', 0)}, "
                f"kappa bytes moved {check.get('kappa_bytes_moved')}")
        if skipped:
            head += f"; skipped: {', '.join(skipped)}"
        if check.get("note"):
            head += "; " + check["note"]
        emit(3, "FAIL" if problems else "PASS", head, "; ".join(problems), events=events, compare=c,
             dispatcher=rep, checks=check, seconds=round(dt, 2), layer_calls=total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-b64", required=True)
    a = ap.parse_args()
    plan = unpack(a.plan_b64)
    stages = set(plan["stages"])
    log(f"anchor driver on {os.uname().nodename} pid {os.getpid()}: device {plan['device']}, "
        f"{len(plan['donors'])} donors, stages {sorted(stages)}, mixed GPU generations: {plan['mixed']}")
    if plan["device"].startswith("cuda"):
        log(f"anchor GPU: {torch.cuda.get_device_name(0)}")
    d = Driver(plan)
    links, raw, f_problems = [], {}, []
    unreachable = d.wait_donors()
    if unreachable:
        log(f"unreachable donors: {unreachable}")
    if 1 in stages:
        try:
            links, raw = d.fabric_transport()
            f_problems += [f"{x['id']}: {x['error']}" for x in links if "error" in x]
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            f_problems.append(f"transport probe crashed: {e!r}")
    try:
        d.build()
        d.wait_placement()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        d.model_error = repr(e)
    if 1 in stages:
        if d.anchor is not None:
            try:
                f_problems += d.fabric_exec(links)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                f_problems.append(f"exec probe crashed: {e!r}")
        else:
            f_problems.append(f"exec probe not run: model/anchor failed ({d.model_error})")
        if unreachable:
            f_problems.insert(0, f"unreachable donors: {unreachable}")
        grp = Driver.groups(links, raw)
        artifact("fabric.json", {"links": links, "groups": grp})
        rem = [x for x in links if x["remote"] and "ping" in x] or [x for x in links if "ping" in x]
        head = ""
        if rem:
            pooled = [v for x in rem for v in raw.get(x["id"], ([], {}))[0]]
            big = list(rem[0]["payload"])[-1] if rem[0].get("payload") else None
            pl = [v for x in rem for v in (raw.get(x["id"], ([], {}))[1].get(big, []) if big else [])]
            ps, pls = stats_ms(pooled), stats_ms(pl)
            head = (f"{len(links)} links; {'remote' if rem[0]['remote'] else 'local'} ping p50 {ps['p50']} ms "
                    f"p99 {ps['p99']} ms; {big} payload p50 {pls['p50']} ms p99 {pls['p99']} ms")
            ex = [x["exec"] for x in rem if x.get("exec")]
            if ex:
                k = sorted(ex[0], key=int)[min(1, len(ex[0]) - 1)]
                vals = sorted(e[k]["p50"] for e in ex if e.get(k, {}).get("p50") is not None)
                if vals:
                    head += f"; exec T={k} p50 {vals[len(vals) // 2]} ms (median link)"
        emit(1, "FAIL" if f_problems else "PASS", head, "; ".join(f_problems), links=links, groups=grp)
    for st, fn in ((2, d.stage2), (3, d.stage3)):
        if st not in stages:
            continue
        if d.anchor is None:
            emit(st, "FAIL", reason=f"model/anchor setup failed: {d.model_error}")
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            emit(st, "FAIL", reason=f"crashed: {e!r}")
    if d.anchor is not None:
        d.anchor.close()
    d.io.stop()
    log("anchor driver done")


if __name__ == "__main__":
    main()
