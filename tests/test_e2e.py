"""Experiment 5 on one machine: split inference with real donor processes.

The correctness claim is exact: every run below must produce logits bitwise equal
to the all-local reference, whatever happens to donors mid-generation.
"""

import random
import signal
import time

import pytest
import torch

from enkhora.anchor.anchor import compare
from enkhora.anchor.dispatcher import DispatchConfig
from enkhora.cluster import DonorSpec, LocalCluster
from enkhora.control.planner import PlannerConfig
from enkhora.model.hf import generate, tiny_qwen3_moe

rng = random.Random(7)
PROMPTS = [[1] + [rng.randrange(2, 256) for _ in range(rng.randint(3, 10))] for _ in range(4)]
MAX_NEW = 24
GUARD = ["--guard-mib", "0.25", "--slack-mib", "0.25"]


def specs():
    return [DonorSpec("d1", "job-a", kappa=1.0, mem_mib=4, life_s=1800, extra=GUARD),
            DonorSpec("d2", "job-b", kappa=0.5, mem_mib=4, life_s=36000, extra=GUARD),
            DonorSpec("d3", "job-c", kappa=0.0, mem_mib=4, life_s=72000, extra=GUARD)]


def make_cluster(path, donors=None, **dispatch):
    d = dict(deadline_init_s=0.25, deadline_min_s=0.03)
    d.update(dispatch)
    return LocalCluster(str(path), donors or specs(),
                        planner=PlannerConfig(expert_demand=0.5, alpha=1.0, copy_rate_Bps=1e12, max_replicas=3),
                        dispatch=DispatchConfig(**d))


def settled(s):
    c = s["coverage"]
    return c["ge2"] == c["bundles"] and all(not d["loading"] for d in s["donors"].values())


@pytest.fixture
def cluster(tmp_path):
    c = make_cluster(tmp_path).start()
    c.wait(settled)
    c.ref = c.anchor.generate_local(PROMPTS, MAX_NEW)
    try:
        yield c
    finally:
        c.stop()


def at_layer(c, n, fn):
    c.anchor.dispatcher.stats.clear()
    c.anchor.dispatcher.rungs.clear()
    fired = []

    def hook(k):
        if k == n and not fired:
            fired.append(k)
            fn()

    c.anchor.dispatcher.on_layer = hook
    return fired


def test_split_matches_reference_and_hf(cluster):
    out = cluster.anchor.generate(PROMPTS, MAX_NEW)
    assert compare(cluster.ref, out)["logits_bitwise"]
    served = cluster.anchor.dispatcher.rungs
    assert served["donor"] > 0 and not any(k.startswith("anchor") for k in served)
    hf = generate(tiny_qwen3_moe(seed=0), PROMPTS, MAX_NEW)
    assert torch.equal(hf[0], out[0]) and (hf[1] - out[1]).abs().max() < 1e-4


def test_sigkill_primary_mid_generation(cluster):
    assert cluster.status()["donors"]["d1"]["primary_for"]
    e0 = cluster.anchor.dispatcher.table.epoch
    fired = at_layer(cluster, 20, lambda: cluster.kill("d1"))
    out = cluster.anchor.generate(PROMPTS, MAX_NEW)
    assert fired and compare(cluster.ref, out)["logits_bitwise"]
    # d1's loss surfaced mid-generation: as a link error, a skipped replica, or a new epoch
    s = cluster.anchor.dispatcher.stats
    assert s["link_errors"] + s["replicas_skipped"] >= 1 or max(cluster.anchor.dispatcher.epochs_seen) > e0
    st = cluster.wait(lambda s: "d1" not in s["donors"] and s["coverage"]["ge1"] == s["coverage"]["bundles"])
    assert len(st["donors"]["d2"]["primary_for"]) == st["coverage"]["bundles"]   # standbys promoted


def test_hung_donors_are_hedged(cluster):
    def stop_both():
        cluster.kill("d1", signal.SIGSTOP)
        cluster.kill("d2", signal.SIGSTOP)

    fired = at_layer(cluster, 10, stop_both)
    try:
        out = cluster.anchor.generate(PROMPTS, MAX_NEW)
    finally:
        cluster.kill("d1", signal.SIGCONT)
        cluster.kill("d2", signal.SIGCONT)
    assert fired and compare(cluster.ref, out)["logits_bitwise"]
    s = cluster.anchor.dispatcher.stats
    assert s["hedges"] >= 1 and s["circuit_open"] >= 1


def test_control_plane_death_does_not_stop_decode(cluster):
    cluster.stop_control()
    out = cluster.anchor.generate(PROMPTS, MAX_NEW)
    assert compare(cluster.ref, out)["logits_bitwise"]
    assert cluster.anchor.dispatcher.rungs["donor"] > 0


def test_every_donor_gone_is_the_floor(cluster):
    for d in ("d1", "d2", "d3"):
        cluster.kill(d)
    out = cluster.anchor.generate(PROMPTS, MAX_NEW)
    assert compare(cluster.ref, out)["logits_bitwise"]


def test_host_memory_spike_evicts_standbys_and_replans(cluster):
    mib = 1 << 20
    cluster.admin("host_alloc", donor="d3", bytes=int(3.5 * mib))
    st = cluster.wait(lambda s: len(s["donors"]["d3"]["ready"]) <= 3)
    assert st["donors"]["d3"]["stats"] is not None
    out = cluster.anchor.generate(PROMPTS, MAX_NEW)
    assert compare(cluster.ref, out)["logits_bitwise"]
    st = cluster.status()
    assert st["coverage"]["ge1"] == st["coverage"]["bundles"]


def test_planned_leave_drains_before_exit(cluster):
    cluster.admin("leave", donor="d1")
    st = cluster.wait(lambda s: "d1" not in s["donors"], timeout=30)
    assert st["coverage"]["ge1"] == st["coverage"]["bundles"]
    cluster.anchor.dispatcher.rungs.clear()
    out = cluster.anchor.generate(PROMPTS, MAX_NEW)
    assert compare(cluster.ref, out)["logits_bitwise"]
    assert not any(k.startswith("anchor") for k in cluster.anchor.dispatcher.rungs)   # no fallback needed


def test_kappa_shrink_moves_no_bytes(cluster):
    before = cluster.status()["counters"]["bytes_issued"]
    cluster.admin("set_kappa", donor="d1", kappa=0.0)
    st = cluster.wait(lambda s: not s["donors"]["d1"]["primary_for"] and s["donors"]["d1"]["kappa"] == 0.0)
    assert st["counters"]["bytes_issued"] == before
    assert len(st["donors"]["d2"]["primary_for"]) == st["coverage"]["bundles"]


def test_prefill_on_anchor_only(cluster):
    cluster.anchor.dispatcher.cfg.prefill_local_tokens = 8
    out = cluster.anchor.generate(PROMPTS, MAX_NEW)
    assert compare(cluster.ref, out)["logits_bitwise"]
    assert any(k.startswith("anchor") for k in cluster.anchor.dispatcher.rungs)


def test_silent_corruption_is_caught_by_audit(tmp_path):
    donors = [DonorSpec("bad", "job-a", kappa=1.0, mem_mib=4, life_s=1800, corrupt=1e-3, extra=GUARD),
              DonorSpec("good", "job-b", kappa=1.0, mem_mib=4, life_s=36000, extra=GUARD)]
    c = make_cluster(tmp_path, donors, audit_rate=1.0).start()
    try:
        c.wait(lambda s: s["coverage"]["ge2"] == s["coverage"]["bundles"])
        ref = c.anchor.generate_local(PROMPTS, MAX_NEW)
        out = c.anchor.generate(PROMPTS, MAX_NEW)
        assert compare(ref, out)["logits_bitwise"]
        assert c.anchor.dispatcher.stats["audit_failures"] >= 1
        st = c.wait(lambda s: "bad" in s["quarantined"] and "bad" not in s["donors"])
        assert "bad" in c.anchor.dispatcher.report()["quarantined"]
    finally:
        c.stop()
