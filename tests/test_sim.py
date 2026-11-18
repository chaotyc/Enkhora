import math

from enkhora.control.planner import PlannerConfig
from enkhora.sim.replay import SimConfig, run
from enkhora.sim.survival import EmpiricalCurve, OracleCurve, SurvivalModel
from enkhora.sim.trace import load_csv, synthetic, write_csv


def test_trace_csv_roundtrip(tmp_path):
    tr = synthetic(days=0.5, seed=1)
    p = tmp_path / "t.csv"
    write_csv(tr, str(p))
    back = load_csv(str(p))
    assert back.times == tr.times and len(back.samples[3]) == 131
    g = next(iter(tr.samples[10]))
    assert back.samples[10][g].job == tr.samples[10][g].job


def test_empirical_survival_conditions_on_elapsed():
    m = SurvivalModel(min_bucket=1).fit(
        [{"partition": "p", "limit_s": 86400, "duration_s": d, "account": "a"} for d in [60] * 50 + [36000] * 50],
        idle_gaps=[900, 1800])
    fresh = m.curve("job", "p", 86400, elapsed_s=0, account="a")
    seasoned = m.curve("job", "p", 86400, elapsed_s=1800, account="a")
    assert fresh.p_alive(600) < 0.6           # half die in the first minute
    assert seasoned.p_alive(600) == 1.0        # survivors of the first half hour are a different population
    capped = m.curve("job", "p", 3600, elapsed_s=3000, account="a")
    assert capped.p_alive(601) == 0.0          # never beyond the time limit
    assert OracleCurve(100).expected_min(3600) == 100
    assert math.isclose(EmpiricalCurve([1e9], 0, math.inf).expected_min(3600), 3600)


def test_replay_runs_and_reports():
    tr = synthetic(days=2, seed=0)
    rep = run(tr, SimConfig(), PlannerConfig(expert_demand=2.0))
    for mode in ("oracle", "predictor"):
        s = rep[mode]
        assert s["donors"]["p50"] > 0
        assert 0 <= s["cov1"]["p10"] <= 1 and s["copies"]["p50"] > 0
        assert s["gib_rereplicated_per_hour"] >= 0
        rows = rep[mode + "_rows"]
        assert len(rows) == rep["intervals"]
    assert rep["decision"].startswith(("OK", "SWITCH"))


def test_four_bit_halves_the_tier():
    tr = synthetic(days=1, seed=0)
    r8 = run(tr, SimConfig(), PlannerConfig(expert_demand=2.0))
    r4 = run(tr, SimConfig(bundle_bytes=1.41e9 / 2), PlannerConfig(expert_demand=2.0))
    assert r4["oracle"]["copies"]["p50"] >= r8["oracle"]["copies"]["p50"]
