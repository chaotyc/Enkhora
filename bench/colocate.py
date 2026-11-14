"""Experiment 3 (the gating number): host slowdown vs κ under co-located expert kernels.

A host process runs a GPU workload and reports its throughput. This process holds
a bundle in its own CUDA context (never MPS) and injects expert calls paced by the
token bucket at κ, with the kernel-length cap. For each κ we alternate randomized
on/off windows and report slowdown = 1 - host_rate(on) / host_rate(off).

Also measured: the cost of a mostly idle second context (κ = 0, process alive),
our own per-call wait and execution times, and the duty cycle actually achieved.

    python bench/colocate.py --device cuda:0 --kappas 0.005,0.01,0.02,0.05,0.1,0.25 --out colocate.csv

Kill criterion (design doc section 11): if typical host codes hit 5% slowdown below
about κ = 2%, busy hosts become store-only. Use the build list's three host codes
via --host-cmd (a command that prints one throughput number per line to stdout).

``--allow-cpu`` runs the harness on CPU for smoke testing only: CPU contention is
not GPU time-slicing and the numbers mean nothing.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import queue
import random
import statistics
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))
from interstitial.donor.governor import TokenBucket  # noqa: E402
from interstitial.experts import BundleWeights, bundle_forward  # noqa: E402


def synthetic_host(device: str, n: int, q: mp.Queue, stop: mp.Event):
    """A training-like host: back-to-back matmuls; reports iterations every 50 ms."""
    torch.manual_seed(0)
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    it, last = 0, time.monotonic()
    while not stop.is_set():
        c = a @ b
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        it += 1
        now = time.monotonic()
        if now - last >= 0.05:
            q.put((now, it / (now - last)))
            it, last = 0, now
        del c


def command_host(cmd: str, q: mp.Queue, stop: threading.Event):
    """A real host code: one throughput number per stdout line."""
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        if stop.is_set():
            break
        try:
            q.put((time.monotonic(), float(line.strip().split()[-1])))
        except ValueError:
            continue
    p.terminate()


def make_bundle(device, n_experts, D, I, dtype):
    g = torch.Generator().manual_seed(0)
    gu = (torch.randn(n_experts, 2 * I, D, generator=g) / D ** 0.5).to(device, dtype)
    dn = (torch.randn(n_experts, D, I, generator=g) / I ** 0.5).to(device, dtype)
    return BundleWeights(0, 0, list(range(n_experts)), gu, dn)


class Injector:
    def __init__(self, w, device, tokens, topk, chunk_rows, dtype):
        self.w, self.device, self.chunk_rows = w, device, chunk_rows
        T, E = tokens, len(w.experts)
        self.hidden = torch.randn(T, w.gate_up.shape[2], device=device, dtype=dtype)
        pairs = torch.stack([torch.randperm(E)[:topk] for _ in range(T)])
        self.rows = torch.arange(T).repeat_interleave(topk)
        self.exps = pairs.reshape(-1)
        self.gates = torch.rand(T * topk, device=device, dtype=dtype)
        self.on = threading.Event()
        self.stop = threading.Event()
        self.bucket = TokenBucket(0.0)
        self.calls: list[tuple[float, float]] = []      # (wait_ms, exec_ms)
        self.busy_ms = 0.0

    def call(self):
        t0 = time.monotonic()
        bundle_forward(self.w, "silu", self.hidden, self.rows, self.exps, self.gates, self.chunk_rows)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return (time.monotonic() - t0) * 1e3

    def run(self):
        while not self.stop.is_set():
            if not self.on.is_set() or not self.bucket.try_acquire():
                time.sleep(0.0005)
                continue
            t_req = time.monotonic()
            ms = self.call()
            self.bucket.charge(ms)
            self.busy_ms += ms
            self.calls.append(((time.monotonic() - t_req) * 1e3 - ms, ms))


def measure(q, inj, kappa, windows, win_s, rng):
    inj.bucket.set_kappa(kappa)
    rates = {True: [], False: []}
    duty = []
    for _ in range(windows):
        order = [True, False]
        rng.shuffle(order)
        for lend in order:
            if lend:
                inj.on.set()
            else:
                inj.on.clear()
            while True:                       # drop samples from before this window
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            b0, t0, samples = inj.busy_ms, time.monotonic(), []
            while time.monotonic() - t0 < win_s:
                try:
                    samples.append(q.get(timeout=win_s)[1])
                except queue.Empty:
                    pass
            if samples:
                rates[lend].append(statistics.mean(samples[1:] or samples))
            if lend:
                duty.append((inj.busy_ms - b0) / (win_s * 1e3))
    inj.on.clear()
    on, off = statistics.mean(rates[True]), statistics.mean(rates[False])
    per = [1 - a / b for a, b in zip(rates[True], rates[False]) if b > 0]
    return {"kappa": kappa, "slowdown": 1 - on / off,
            "slowdown_sd": statistics.stdev(per) if len(per) > 1 else 0.0,
            "duty": statistics.mean(duty) if duty else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--host-matmul", type=int, default=4096)
    ap.add_argument("--host-cmd", help="real host code printing a throughput number per line")
    ap.add_argument("--kappas", default="0.005,0.01,0.02,0.05,0.1,0.25")
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--experts", type=int, default=32, help="experts in the injected bundle")
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--chunk-rows", type=int, default=4096)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not a.device.startswith("cuda") and not a.allow_cpu:
        sys.exit("experiment 3 needs a GPU (pass --allow-cpu to smoke-test the harness)")
    dtype = getattr(torch, a.dtype)

    ctx = mp.get_context("spawn")
    q, stop = ctx.Queue(), ctx.Event()
    if a.host_cmd:
        tstop = threading.Event()
        host = threading.Thread(target=command_host, args=(a.host_cmd, q, tstop), daemon=True)
    else:
        host = ctx.Process(target=synthetic_host, args=(a.device, a.host_matmul, q, stop), daemon=True)
    host.start()
    time.sleep(3.0)                            # host warm-up

    w = make_bundle(a.device, a.experts, a.hidden, a.inter, dtype)
    inj = Injector(w, a.device, a.tokens, a.topk, a.chunk_rows, dtype)
    for _ in range(3):
        inj.call()                             # warm kernels
    th = threading.Thread(target=inj.run, daemon=True)
    th.start()
    rng = random.Random(0)

    results = [dict(measure(q, inj, 0.0, a.windows, a.window_s, rng), kappa=0.0, note="idle second context")]
    for k in [float(x) for x in a.kappas.split(",")]:
        inj.calls.clear()
        r = measure(q, inj, k, a.windows, a.window_s, rng)
        ms = sorted(c[1] for c in inj.calls) or [0.0]
        waits = sorted(c[0] for c in inj.calls) or [0.0]
        r.update(call_ms_p50=ms[len(ms) // 2], call_ms_p99=ms[int(0.99 * (len(ms) - 1))],
                 wait_ms_p50=waits[len(waits) // 2], calls=len(inj.calls))
        results.append(r)
        print(f"kappa {k:6.3f}  slowdown {100 * r['slowdown']:6.2f}% ± {100 * r['slowdown_sd']:.2f}  "
              f"duty {100 * r['duty']:5.2f}%  call p50 {r['call_ms_p50']:.2f} ms  p99 {r['call_ms_p99']:.2f} ms",
              flush=True)
    inj.stop.set()
    stop.set()
    if a.host_cmd:
        tstop.set()
    over = [r["kappa"] for r in results[1:] if r["slowdown"] > 0.05]
    print(f"\nidle second context: slowdown {100 * results[0]['slowdown']:.2f}%")
    print("first kappa over the 5% cap:", over[0] if over else "none in grid")
    if over and over[0] <= 0.02:
        print("KILL CRITERION HIT: 5% slowdown below kappa 2% -> busy hosts become store-only")
    if a.out:
        keys = sorted({k for r in results for k in r})
        with open(a.out, "w", newline="") as f:
            wr = csv.DictWriter(f, keys)
            wr.writeheader()
            wr.writerows(results)


if __name__ == "__main__":
    main()
