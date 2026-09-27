"""Tests for the forecast-driven planning loop."""

import pytest

from capacity_planner.forecast import bucket_arrivals
from capacity_planner.planner import (
    cost_of_being_wrong,
    evaluate_plan,
    plan_fleet,
    scale_workload_to_counts,
    select_forecaster,
)
from capacity_planner.simulator import Job, Machine
from capacity_planner.sweep import cheapest_meeting_slo, identical_fleet, sweep_fleet_sizes
from capacity_planner.workload import generate_workload

MACHINE = Machine("node", cpu=16, ram=64, gpu=2, cost_per_hour=4.0)


def _fixed_job(arrival: float, i: int) -> Job:
    return Job(
        name=f"fixed-{i}",
        arrival=arrival,
        cpu=4.0,
        ram=8.0,
        duration=2.0,
        wait_deadline=float("inf"),
        priority=0,
    )


def test_select_forecaster_trended_prefers_smoothing():
    # Clean upward trend: exponential smoothing tracks it, moving average lags.
    series = [float(i) for i in range(1, 22)]
    sel = select_forecaster(series, horizon=4)
    assert sel["name"] == "exponential_smoothing"
    assert sel["backtest_mae"] < 5.0
    assert len(sel["forecast"]) == 4
    # Forecast continues upward, not flat at the historical mean.
    assert sel["forecast"][-1] > sum(series) / len(series)


def test_select_forecaster_rejects_short_series():
    with pytest.raises(ValueError):
        select_forecaster([1.0, 2.0, 3.0], horizon=2)


def test_scale_preserves_bucket_counts():
    base = generate_workload(seed=11, hours=6.0)
    targets = [0.0, 5.0, 20.0, 3.0, 0.0, 8.0]
    scaled = scale_workload_to_counts(base, targets, bucket_size=1.0, seed=2)
    got = bucket_arrivals([j.arrival for j in scaled], 1.0, end_time=6.0)
    assert got == [0, 5, 20, 3, 0, 8, 0]  # end_time=6.0 -> 7 buckets
    # Deterministic for a fixed seed.
    again = scale_workload_to_counts(base, targets, bucket_size=1.0, seed=2)
    assert [j.arrival for j in again] == [j.arrival for j in scaled]


def test_plan_fleet_picks_finite():
    history = [j.arrival for j in generate_workload(seed=3, hours=12.0)]
    plan = plan_fleet(
        history, identical_fleet(MACHINE), sizes=list(range(1, 7)),
        horizon=6.0, slo=0.05, seed=3,
    )
    assert plan["n_machines"] is not None
    assert 1 <= plan["n_machines"] <= 6
    assert plan["forecaster"] in ("moving_average", "exponential_smoothing")
    assert plan["backtest_mae"] >= 0
    assert plan["planned_total"] >= 0
    assert abs(plan["forecast_total"] - plan["planned_total"]) <= max(
        1, 0.05 * plan["forecast_total"]
    )


def test_perfect_forecast_matches_oracle():
    # Constant-rate demand: the forecast nails it, so the forecast plan
    # should pick (essentially) the same fleet as the oracle.
    history = [i * 0.25 for i in range(72)]  # 4/hr for 18h
    actual = [_fixed_job(18.0 + i * 0.25, i) for i in range(24)]
    sizes = list(range(1, 7))
    oracle = cheapest_meeting_slo(
        sweep_fleet_sizes(actual, identical_fleet(MACHINE), sizes, horizon=6.0),
        0.05,
    )
    plan = plan_fleet(
        history, identical_fleet(MACHINE), sizes,
        horizon=6.0, slo=0.05, seed=9,
    )
    assert plan["n_machines"] is not None
    assert abs(plan["n_machines"] - oracle["n_machines"]) <= 1


def test_inflated_forecast_overprovisions():
    base = generate_workload(seed=5, hours=6.0)
    true_counts = bucket_arrivals([j.arrival for j in base], 1.0, end_time=6.0)
    sizes = list(range(1, 9))
    pick_true = cheapest_meeting_slo(
        sweep_fleet_sizes(base, identical_fleet(MACHINE), sizes, horizon=6.0), 0.05
    )
    big = scale_workload_to_counts(
        base, [3.0 * c for c in true_counts], 1.0, seed=1
    )
    pick_big = cheapest_meeting_slo(
        sweep_fleet_sizes(big, identical_fleet(MACHINE), sizes, horizon=6.0), 0.05
    )
    assert pick_big["n_machines"] >= pick_true["n_machines"]
    assert pick_big["cost"] >= pick_true["cost"]
    # Realized on the true workload, the bigger fleet breaches no more.
    r_true = evaluate_plan(pick_true["n_machines"], identical_fleet(MACHINE), base, 6.0)
    r_big = evaluate_plan(pick_big["n_machines"], identical_fleet(MACHINE), base, 6.0)
    assert r_big["breach_rate"] <= r_true["breach_rate"]


def test_evaluate_plan_sane():
    workload = generate_workload(seed=21, hours=6.0)
    pt = evaluate_plan(3, identical_fleet(MACHINE), workload, horizon=6.0)
    assert pt["cost"] == pytest.approx(3 * 4.0 * 6.0)
    assert 0.0 <= pt["breach_rate"] <= 1.0
    assert pt["n_machines"] == 3
    with pytest.raises(ValueError):
        evaluate_plan(0, identical_fleet(MACHINE), workload, horizon=6.0)


def test_cost_of_being_wrong_signs():
    oracle = {"cost": 500.0, "breach_rate": 0.02, "n_machines": 5}
    over = {"cost": 700.0, "breach_rate": 0.0, "n_machines": 7}
    under = {"cost": 400.0, "breach_rate": 0.10, "n_machines": 4}
    err_over = cost_of_being_wrong(oracle, over)
    assert err_over["extra_cost"] == pytest.approx(200.0)
    assert err_over["breach_delta"] == pytest.approx(-0.02)
    err_under = cost_of_being_wrong(oracle, under)
    assert err_under["extra_cost"] == pytest.approx(-100.0)
    assert err_under["breach_delta"] == pytest.approx(0.08)


def test_plan_fleet_validation():
    with pytest.raises(ValueError):
        plan_fleet([], identical_fleet(MACHINE), [1, 2], 6.0, 0.05)
    with pytest.raises(ValueError):
        plan_fleet([1.0, 2.0], identical_fleet(MACHINE), [1, 2], 6.0, 1.5)
