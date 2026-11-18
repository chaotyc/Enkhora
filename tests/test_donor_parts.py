import random

import torch

from enkhora.donor.governor import AIMD, OnOffEstimator, TokenBucket
from enkhora.donor.slabs import SlabManager
from enkhora.donor.watchdog import SimulatedProbe, Watchdog
from enkhora.experts import BundleWeights


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def bw(b, n=100):
    return BundleWeights(b, 0, [0], torch.zeros(1, n // 8, 1), torch.zeros(1, 1, n // 8))


def test_token_bucket_spends_kappa():
    clk = Clock()
    tb = TokenBucket(0.25, window_ms=100, clock=clk)
    assert tb.capacity == 25 and tb.try_acquire()
    tb.charge(30)                         # debt allowed for one call
    assert not tb.try_acquire()
    clk.t += 0.024                        # 24 ms wall -> 6 ms of kernel budget, pays the 5 ms debt
    assert tb.try_acquire()
    tb.set_kappa(0.0)
    assert not tb.try_acquire() or tb.remaining_ms() <= 0
    tb2 = TokenBucket(1.0, clock=clk)
    tb2.paused = True
    assert not tb2.try_acquire()


def test_aimd_converges_to_tolerated_kappa():
    """Closed loop against a simulated host whose slowdown = sensitivity * duty."""
    rng = random.Random(0)
    sens = 0.5          # host loses 0.5% of throughput per 1% of GPU time we take
    est, aimd = OnOffEstimator(pairs=2, rng=rng), AIMD(kappa0=0.01, add=0.01, mult=0.5, tolerance=0.05)
    history = []
    for _ in range(400):
        lend, _ = est.next_window()
        duty = aimd.kappa if lend else 0.0
        est.record(lend, 1.0 - sens * duty + rng.gauss(0, 0.002))
        if est.ready() and len(est.on) == len(est.off):
            history.append(aimd.update(est.slowdown()))
    tail = history[-50:]
    # 5% slowdown cap => kappa ~ 0.1; AIMD saw-tooths around it
    assert 0.03 < sum(tail) / len(tail) < 0.12
    assert max(tail) < 0.2


def test_watchdog_evicts_standbys_first_and_restores_high_watermark():
    slabs = SlabManager()
    for b, role in [(1, "primary"), (2, "standby"), (3, "standby"), (4, "primary")]:
        slabs.reserve(b, 100, role)
        slabs.set_ready(b, bw(b))
    probe = SimulatedProbe(total=1000)
    wd = Watchdog(probe, slabs, guard_bytes=200, slack_bytes=100)
    assert wd.check() == [] and wd.budget() == 1000 - 300
    probe.host_used = 550                 # slabs are 96 B each: free = 1000 - 550 - 384 = 66 < low(200)
    out = wd.check()
    assert out[:2] == [2, 3] or out[:2] == [3, 2]
    assert probe.free_bytes(slabs.used_bytes()) >= 300
    assert len(out) == 3 and out[2] in (1, 4)                # standbys, then one primary


def test_slab_dropped_while_loading_is_not_resurrected():
    s = SlabManager()
    assert s.reserve(7, 100)
    assert s.drop(7)
    assert not s.set_ready(7, bw(7))
    assert s.get_ready(7) is None and s.used_bytes() == 0
