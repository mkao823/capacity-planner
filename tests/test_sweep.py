"""Tests for the fleet-size sweep and knee helpers."""

import pytest

from capacity_planner.simulator import Machine
from capacity_planner.sweep import (
    cheapest_meeting_slo,
    find_knee,
    identical_fleet,
    sweep_fleet_sizes,
)
from capacity_planner.workload import generate_workload

TEMPLATE = Machine("node", cpu=16, ram=64, gpu=2, cost_per_hour=4.0)
HORIZON = 24.0


@pytest.fixture(scope="module")
def workload():
    return generate_workload(seed=11, hours=24.0)


def test_sweep_returns_one_point_per_size(workload):
    sizes = [2, 4, 6]
    points = sweep_fleet_sizes(workload, identical_fleet(TEMPLATE), sizes, horizon=HORIZON)
    assert [p["n_machines"] for p in points] == sizes
    for p in points:
        assert {"cost", "breach_rate", "avg_wait_admitted", "makespan"} <= p.keys()


def test_cost_nondecreasing_with_fleet_size(workload):
    sizes = [2, 3, 4, 5, 6]
    points = sweep_fleet_sizes(workload, identical_fleet(TEMPLATE), sizes, horizon=HORIZON)
    costs = [p["cost"] for p in points]
    assert all(b >= a - 1e-9 for a, b in zip(costs, costs[1:]))


def test_breach_rate_nonincreasing_with_fleet_size(workload):
    sizes = [2, 3, 4, 5, 6]
    points = sweep_fleet_sizes(workload, identical_fleet(TEMPLATE), sizes, horizon=HORIZON)
    breaches = [p["breach_rate"] for p in points]
    assert all(b <= a + 1e-9 for a, b in zip(breaches, breaches[1:]))


def test_sweep_rejects_bad_sizes(workload):
    with pytest.raises(ValueError):
        sweep_fleet_sizes(workload, identical_fleet(TEMPLATE), [0], horizon=HORIZON)


def _synthetic(points):
    return [{"cost": c, "breach_rate": b} for c, b in points]


def test_cheapest_meeting_slo_picks_cheapest_feasible():
    points = _synthetic([(1, 0.9), (2, 0.12), (3, 0.10), (4, 0.09), (5, 0.085)])
    pick = cheapest_meeting_slo(points, 0.10)
    assert pick["cost"] == 3  # first point at/below 10%


def test_cheapest_meeting_slo_none_when_infeasible():
    points = _synthetic([(1, 0.9), (2, 0.5)])
    assert cheapest_meeting_slo(points, 0.10) is None


def test_find_knee_lands_interior_on_l_curve():
    points = _synthetic([(1, 0.9), (2, 0.12), (3, 0.10), (4, 0.09), (5, 0.085)])
    knee = find_knee(points)
    assert knee is not None
    assert knee["cost"] not in (1, 5)  # strictly interior, not an endpoint


def test_find_knee_needs_three_points():
    assert find_knee([]) is None
    assert find_knee(_synthetic([(1, 0.5)])) is None
    assert find_knee(_synthetic([(1, 0.5), (2, 0.4)])) is None


def test_workload_is_seedable_and_shaped():
    a = generate_workload(seed=3)
    b = generate_workload(seed=3)
    c = generate_workload(seed=4)
    assert [j.arrival for j in a] == [j.arrival for j in b]
    assert [j.arrival for j in a] != [j.arrival for j in c]
    assert len(a) > 100  # a full day has real volume
    assert all(0 <= j.arrival < 24.0 for j in a)
    assert {j.priority for j in a} == {0, 1, 2}
    assert any(j.gpu > 0 for j in a)
