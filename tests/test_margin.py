"""Tests for forecast safety margins (residual-quantile uplift, safety factor)."""

import random

import pytest

from capacity_planner.planner import (
    plan_fleet,
    residual_quantile_margin,
    select_forecaster,
)
from capacity_planner.simulator import Machine
from capacity_planner.sweep import identical_fleet
from capacity_planner.workload import generate_workload

MACHINE = Machine("node", cpu=16, ram=64, gpu=2, cost_per_hour=4.0)
SIZES = list(range(1, 9))


def _history(seed: int = 3, hours: float = 18.0) -> list[float]:
    return [j.arrival for j in generate_workload(seed=seed, hours=hours)]


def _plan(history, seed=3, **kw):
    return plan_fleet(
        history, identical_fleet(MACHINE), SIZES, 6.0, 0.05,
        bucket_size=1.0, seed=seed, **kw,
    )


def test_margin_never_reduces_planned_counts():
    history = _history()
    base = _plan(history)
    for kw in ({"margin_quantile": 0.9}, {"safety_factor": 0.2},
               {"margin_quantile": 0.9, "safety_factor": 0.2}):
        margined = _plan(history, **kw)
        assert margined["planned_counts"] is not None
        for m, b in zip(margined["planned_counts"], base["planned_counts"]):
            assert m >= b - 1e-9
        assert margined["margin_uplift_total"] >= -1e-9


def test_quantile_margin_clamped_at_zero():
    # Forecaster that over-predicts everywhere -> negative p90 residual.
    assert residual_quantile_margin([-5.0, -3.0, -1.0, -0.5], 0.9) == 0.0
    assert residual_quantile_margin([], 0.9) == 0.0


def test_p90_covers_known_noise():
    # Clear noise model: level 10 + uniform(-3, 3) noise over 100 buckets.
    rng = random.Random(42)
    series = [10.0 + rng.uniform(-3.0, 3.0) for _ in range(100)]
    sel = select_forecaster(series, horizon=6)
    resid = sel["backtest_residuals"]
    assert len(resid) == 20  # last 20% of 100
    margin = residual_quantile_margin(resid, 0.9)
    assert margin > 0  # noise is symmetric; p90 residual must be positive
    # actual <= predicted + margin  <=>  residual <= margin
    coverage = sum(1 for r in resid if r <= margin + 1e-9) / len(resid)
    assert coverage >= 0.85


def test_safety_factor_uplifts_exactly():
    history = _history()
    base = _plan(history)
    sf = _plan(history, safety_factor=0.2)
    assert sf["safety_factor"] == 0.2
    assert sf["margin_jobs_per_bucket"] == 0.0
    for m, b in zip(sf["planned_counts"], base["planned_counts"]):
        assert m == pytest.approx(b * 1.2)


def test_margins_compose_in_documented_order():
    history = _history()
    both = _plan(history, margin_quantile=0.9, safety_factor=0.2)
    uplift = both["margin_jobs_per_bucket"]
    raw = both["forecast"]
    assert uplift >= 0.0
    for m, r in zip(both["planned_counts"], raw):
        # margined = (forecast + quantile_uplift) * (1 + factor)
        assert m == pytest.approx((r + uplift) * 1.2)
    assert both["margin_uplift_total"] == pytest.approx(
        sum(both["planned_counts"]) - sum(raw)
    )


def test_margin_validation():
    history = _history()
    for bad_q in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            _plan(history, margin_quantile=bad_q)
    with pytest.raises(ValueError):
        _plan(history, safety_factor=-0.1)
    with pytest.raises(ValueError):
        residual_quantile_margin([1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        residual_quantile_margin([1.0, 2.0], 1.0)


def test_margin_deterministic_given_seed():
    history = _history()
    kw = {"margin_quantile": 0.9, "safety_factor": 0.2}
    a = _plan(history, **kw)
    b = _plan(history, **kw)
    assert a["planned_counts"] == b["planned_counts"]
    assert a["n_machines"] == b["n_machines"]
    assert a["margin_jobs_per_bucket"] == b["margin_jobs_per_bucket"]


def test_margin_diagnostics_present_without_margin():
    plan = _plan(_history())
    assert plan["margin_quantile"] is None
    assert plan["safety_factor"] == 0.0
    assert plan["margin_jobs_per_bucket"] == 0.0
    assert plan["planned_counts"] == pytest.approx(plan["forecast"])
    assert plan["margin_uplift_total"] == pytest.approx(0.0)


def test_margin_never_shrinks_fleet_pick():
    # Wider grid: the p90 margin on this bursty workload is conservative
    # enough that no fleet of <=8 meets the SLO on the margined workload.
    history = _history()
    sizes = list(range(1, 17))
    base_pick = plan_fleet(
        history, identical_fleet(MACHINE), sizes, 6.0, 0.05,
        bucket_size=1.0, seed=3,
    )["n_machines"]
    assert base_pick is not None
    for kw in ({"margin_quantile": 0.9}, {"safety_factor": 0.2}):
        pick = plan_fleet(
            history, identical_fleet(MACHINE), sizes, 6.0, 0.05,
            bucket_size=1.0, seed=3, **kw,
        )["n_machines"]
        # None = even the largest swept fleet misses the SLO on the margined
        # workload: the conservative extreme, never a shrink.
        assert (pick if pick is not None else float("inf")) >= base_pick
    # And on this workload the p90 margin strictly grows the pick.
    p90_pick = plan_fleet(
        history, identical_fleet(MACHINE), sizes, 6.0, 0.05,
        bucket_size=1.0, seed=3, margin_quantile=0.9,
    )["n_machines"]
    assert p90_pick is not None and p90_pick > base_pick
