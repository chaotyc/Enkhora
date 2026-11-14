import copy

import pytest
import torch

from interstitial.anchor.fallback import LocalExecutor
from interstitial.experts import BundleWeights, build_route, bundle_forward, merge_partials
from interstitial.model.hf import export_master, find_moe_blocks, generate, patch_model, tiny_qwen3_moe
from interstitial.store import MasterStore


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    model = tiny_qwen3_moe(seed=0)
    root = str(tmp_path_factory.mktemp("master"))
    master = export_master(copy.deepcopy(model), root, bundle_size=8)
    return model, master


def test_export_manifest_and_checksums(tiny):
    model, master = tiny
    assert len(master.bundles) == 4 * 32 // 8
    assert master.spec.hidden == 64 and master.spec.inter == 32
    for b, meta in master.bundles.items():
        w = master.load(b)
        assert w.checksum() == meta.sha256
        assert w.experts == list(range((b % 4) * 8, (b % 4) * 8 + 8))
    reopened = MasterStore(master.root)
    assert reopened.tier_bytes() == master.tier_bytes()


def test_bundle_partials_sum_to_hf_layer_output(tiny):
    model, master = tiny
    layer, block = find_moe_blocks(model)[1]
    torch.manual_seed(3)
    h = torch.randn(37, 64)
    with torch.no_grad():
        _, w, idx = block.gate(h)
        want = block.experts(h, idx, w)
    ex = LocalExecutor(master)
    got = ex.run_layer(layer, h, idx, w)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-5)


def test_bundle_forward_is_independent_of_pair_order(tiny):
    _, master = tiny
    w = master.load(5)
    torch.manual_seed(1)
    h = torch.randn(20, 64)
    rows = torch.randint(0, 20, (40,))
    exps = torch.tensor(w.experts)[torch.randint(0, 8, (40,))]
    # one pair per (row, expert)
    key = torch.unique(rows * 1000 + exps)
    rows, exps = key // 1000, key % 1000
    g = torch.rand(len(rows))
    u1, p1 = bundle_forward(w, "silu", h, rows, exps, g)
    perm = torch.randperm(len(rows))
    u2, p2 = bundle_forward(w, "silu", h, rows[perm], exps[perm], g[perm])
    assert torch.equal(u1, u2) and torch.equal(p1, p2)


def test_subset_rows_give_bitwise_identical_partials(tiny):
    """A donor receives only the hidden rows it needs, re-indexed. Same bits."""
    _, master = tiny
    layer = master.layers[0]
    torch.manual_seed(2)
    h = torch.randn(50, 64)
    idx = torch.randint(0, 32, (50, 4))
    wts = torch.rand(50, 4)
    route = build_route(idx, wts, master.expert_to_bundle(layer))
    for b in route.bundles:
        rows, ex, g = route.pairs[b]
        full_u, full_p = bundle_forward(master.load(b), "silu", h, rows, ex, g)
        U = route.urows[b]
        sub_u, sub_p = bundle_forward(master.load(b), "silu", h[U], torch.searchsorted(U, rows), ex, g)
        assert torch.equal(full_p, sub_p)


def test_merge_is_ascending_bundle_order(tiny):
    _, master = tiny
    route = build_route(torch.tensor([[0, 9]]), torch.tensor([[0.5, 0.5]]), master.expert_to_bundle(master.layers[0]))
    parts = {b: torch.full((1, 4), float(b + 1)) for b in route.bundles}
    out = merge_partials(1, 4, torch.float32, "cpu", route, parts)
    assert torch.equal(out, torch.full((1, 4), float(sum(b + 1 for b in route.bundles))))


def test_local_executor_matches_hf_generation(tiny):
    model, master = tiny
    ref = copy.deepcopy(model)
    patched = patch_model(copy.deepcopy(model), LocalExecutor(master))
    assert not any(hasattr(b, "experts") for _, b in find_moe_blocks(patched))
    prompts = [[1, 5, 9, 33], [1, 7], [1, 100, 101, 102, 103, 104]]
    t0, l0 = generate(ref, prompts, 10)
    t1, l1 = generate(patched, prompts, 10)
    assert torch.equal(t0, t1)
    assert (l0 - l1).abs().max() < 1e-4
