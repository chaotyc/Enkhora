import math

from interstitial.control.planner import ExpSurvival, PBundle, PDonor, PlannerConfig, admissible, plan

MB = 1 << 20


def bundles(layers=4, per=4, size=10 * MB):
    return [PBundle(l * per + j, l, size) for l in range(layers) for j in range(per)]


def cfg(**kw):
    base = dict(copy_rate_Bps=1e12, alpha=1.0, expert_demand=1.0, max_replicas=2)
    base.update(kw)
    return PlannerConfig(**base)


def donor(id, job, mem_mb, kappa, life=math.inf, held=None, **kw):
    return PDonor(id, job, mem_mb * MB, kappa, ExpSurvival(life), held=held or {}, **kw)


def apply(donors, p):
    for d in donors:
        for b in p.drops.get(d.id, []):
            d.held.pop(b, None)
        for b, _ in p.loads.get(d.id, []):
            d.held[b] = "ready"


def check_invariants(bs, donors, p):
    D = {d.id: d for d in donors}
    B = {b.id: b for b in bs}
    for d in donors:
        held = set(d.held) - set(p.drops.get(d.id, [])) | {b for b, _ in p.loads.get(d.id, [])}
        assert sum(B[b].nbytes for b in held) <= d.mem, f"memory over on {d.id}"
    for b, ds in p.placement.items():
        jobs = [D[x].job for x in ds]
        assert len(jobs) == len(set(jobs)), f"anti-affinity broken for bundle {b}"


def test_cover_everything_with_primary_and_standby():
    bs = bundles()
    donors = [donor("fast", "j1", 200, 1.0, life=1800), donor("quiet", "j2", 200, 0.5, life=36000),
              donor("store", "j3", 200, 0.0, life=72000)]
    p = plan(bs, donors, cfg())
    check_invariants(bs, donors, p)
    assert not p.uncovered and not p.no_exec
    assert all(len(p.placement[b.id]) == 2 for b in bs)
    # primaries spread across the g=2 execute donors, store donor never primary
    assert {p.primary[b.id] for b in bs} == {"fast", "quiet"}
    # fragile primaries get standbys on the most stable donor (pair fragile with stable)
    for b in bs:
        if p.primary[b.id] == "fast":
            assert p.placement[b.id][1] == "store"


def test_anti_affinity_by_job_even_across_gpus():
    bs = bundles(layers=2, per=2)
    donors = [donor("g0", "job4gpu", 500, 1.0), donor("g1", "job4gpu", 500, 1.0), donor("g2", "job4gpu", 500, 1.0)]
    p = plan(bs, donors, cfg())
    check_invariants(bs, donors, p)
    assert all(len(r) == 1 for r in p.placement.values())    # one job => one replica max


def test_kappa_shrink_swaps_roles_without_moving_bytes():
    bs = bundles()
    donors = [donor("a", "j1", 500, 1.0), donor("b", "j2", 500, 1.0)]
    p1 = plan(bs, donors, cfg())
    apply(donors, p1)
    assert any(p1.primary[b.id] == "a" for b in bs)
    donors[0].kappa = 0.0                      # host on "a" woke up
    p2 = plan(bs, donors, cfg(), p1.primary)
    assert p2.bytes_to_load({b.id: b for b in bs}) == 0 and not p2.drops
    assert all(p2.primary[b.id] == "b" for b in bs)


def test_hysteresis_keeps_current_primary():
    bs = bundles(layers=1, per=2)
    donors = [donor("a", "j1", 500, 0.9), donor("b", "j2", 500, 1.0)]
    for d in donors:
        d.held = {b.id: "ready" for b in bs}
    p = plan(bs, donors, cfg(expert_demand=0.1), prev_primary={b.id: "a" for b in bs})
    assert all(p.primary[b.id] == "a" for b in bs)


def test_memory_shrink_drops_standbys_first():
    bs = bundles(layers=2, per=2)
    donors = [donor("a", "j1", 100, 1.0), donor("b", "j2", 100, 1.0)]
    p1 = plan(bs, donors, cfg())
    apply(donors, p1)
    prim_a = {b for b, d in p1.primary.items() if d == "a"}
    donors[0].mem = 2 * 10 * MB                # host grew: room for two bundles only
    p2 = plan(bs, donors, cfg(), p1.primary)
    check_invariants(bs, donors, p2)
    kept = set(donors[0].held) - set(p2.drops["a"])
    assert kept == prim_a                      # standbys went, primaries stayed


def test_donor_loss_rereplicates_from_store_donor():
    bs = bundles(layers=2, per=2)
    donors = [donor("fast", "j1", 100, 1.0, life=600), donor("quiet", "j2", 100, 0.5),
              donor("store", "j3", 100, 0.0, life=1e6)]
    c = cfg(max_replicas=3, expert_demand=0.4)      # "quiet" alone can execute everything
    p1 = plan(bs, donors, c)
    apply(donors, p1)
    survivors = [d for d in donors if d.id != "fast"]
    p2 = plan(bs, survivors, c, p1.primary)
    check_invariants(bs, survivors, p2)
    assert not p2.uncovered
    assert all(p2.primary[b.id] == "quiet" for b in bs)
    for d, loads in p2.loads.items():
        for b, src in loads:
            assert src in (None, "store")      # never copy from an executor


def test_admission_rejects_short_life():
    c = PlannerConfig(alpha=10, copy_rate_Bps=1e9)
    long_ = donor("l", "j1", 47 * 1024, 1.0, life=3600)
    short = donor("s", "j2", 47 * 1024, 1.0, life=60)
    assert admissible(long_, c) and not admissible(short, c)
    # the doc's example: ~50 s to fill 47 GiB at 1 GB/s, alpha 10 => ~8 min of life
    bar = 10 * 47 * 1024 * MB / 1e9
    assert 450 < bar < 550


def test_zero_donors_is_legal():
    bs = bundles()
    p = plan(bs, [], cfg())
    assert p.uncovered == [b.id for b in bs] and not p.loads


def test_ending_donor_is_replaced_not_counted():
    bs = bundles(layers=1, per=2)
    donors = [donor("old", "j1", 100, 1.0, ending=True), donor("new", "j2", 100, 1.0), donor("st", "j3", 100, 0.0)]
    donors[0].held = {b.id: "ready" for b in bs}
    p = plan(bs, donors, cfg())
    assert all(p.primary[b.id] == "new" for b in bs)
    assert all(p.placement[b.id][-1] == "old" for b in bs)   # kept as a last resort until gone


def test_compute_budget_limits_primaries():
    bs = bundles(layers=1, per=4)
    donors = [donor("small", "j1", 500, 0.5)]                # can execute half of the demand
    p = plan(bs, donors, cfg(expert_demand=1.0))
    assert len([b for b in bs if p.primary.get(b.id) == "small"]) == 2
    assert len(p.no_exec) == 2
