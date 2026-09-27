"""End-to-end smoke tests: a tiny fleet, a tiny workload, real invariants."""

from capacity_planner.cost import slo_stats, total_cost, tradeoff_point
from capacity_planner.forecast import (
    bucket_arrivals,
    exponential_smoothing_forecast,
    moving_average_forecast,
)
from capacity_planner.simulator import Job, Machine, Simulator


def make_fleet() -> list[Machine]:
    return [
        Machine(name="m1", cpu=8, ram=32, gpu=1, cost_per_hour=2.50),
        Machine(name="m2", cpu=8, ram=32, gpu=0, cost_per_hour=1.20),
    ]


def make_jobs() -> list[Job]:
    return [
        Job(name="j1", arrival=0.0, cpu=4, ram=8, gpu=1, duration=2.0, wait_deadline=1.0),
        Job(name="j2", arrival=0.5, cpu=4, ram=8, gpu=0, duration=1.0, wait_deadline=1.0),
        Job(name="j3", arrival=1.0, cpu=8, ram=16, gpu=0, duration=1.5, wait_deadline=2.0),
        # Hopeless: needs more GPU than any single machine has.
        Job(name="j4", arrival=1.5, cpu=2, ram=4, gpu=4, duration=1.0, wait_deadline=5.0),
        # Low priority, arrives while fleet is busy.
        Job(
            name="j5",
            arrival=0.2,
            cpu=8,
            ram=32,
            gpu=0,
            duration=0.5,
            wait_deadline=10.0,
            priority=5,
        ),
    ]


def test_simulation_end_to_end_invariants():
    machines = make_fleet()
    jobs = make_jobs()
    result = Simulator(machines).run(jobs)

    # Every job is either scheduled or explicitly rejected — nothing vanishes.
    accounted = {j.name for j, _, _ in result.scheduled} | {
        j.name for j in result.rejected
    }
    assert accounted == {j.name for j in jobs}

    # The hopeless job is rejected, never silently dropped or force-fit.
    assert any(j.name == "j4" for j in result.rejected)

    # No over-allocation: peak usage never exceeded capacity on any machine.
    for m in machines:
        assert m.peak_cpu <= m.cpu
        assert m.peak_ram <= m.ram
        assert m.peak_gpu <= m.gpu

    # Utilization is a sane fraction.
    for util in result.utilization.values():
        for v in util.values():
            assert 0.0 <= v <= 1.0

    # Wait times are non-negative and line up with scheduled jobs.
    assert len(result.wait_times) == result.n_scheduled
    assert all(w >= 0 for w in result.wait_times)


def test_cost_and_slo_stats_are_consistent():
    machines = make_fleet()
    result = Simulator(machines).run(make_jobs())

    cost = total_cost(machines, result.makespan)
    assert cost == (2.50 + 1.20) * result.makespan
    assert cost >= 0

    stats = slo_stats(result, wait_deadline=1.0)
    assert 0.0 <= stats["breach_rate"] <= 1.0
    assert stats["avg_wait_admitted"] >= 0
    # p95 is a nearest-rank percentile: always within [min, max] of waits.
    assert 0.0 <= stats["p95_wait_admitted"] <= max(result.wait_times)

    point = tradeoff_point(machines, result, wait_deadline=1.0)
    assert point["cost"] == cost
    assert point["breach_rate"] == stats["breach_rate"]
    # Default horizon is the result makespan.
    assert point["horizon"] == result.makespan
    assert point["makespan"] == result.makespan


def test_forecast_baselines():
    arrivals = [0.1, 0.4, 1.2, 1.8, 2.3, 3.1]
    buckets = bucket_arrivals(arrivals, bucket_size=1.0)
    assert buckets == [2, 2, 1, 1]

    ma = moving_average_forecast([2.0, 2.0, 1.0, 1.0], window=2, horizon=3)
    assert ma == [1.0, 1.0, 1.0]

    # Flat series: exponential smoothing converges to the level.
    ses = exponential_smoothing_forecast([5.0, 5.0, 5.0], alpha=0.5, horizon=2)
    assert ses == [5.0, 5.0]

    # Alpha=1 tracks the last observation exactly.
    ses2 = exponential_smoothing_forecast([1.0, 2.0, 9.0], alpha=1.0, horizon=1)
    assert ses2 == [9.0]


def test_empty_workload_does_not_crash():
    result = Simulator(make_fleet()).run([])
    assert result.n_scheduled == 0
    assert result.n_rejected == 0
    assert result.makespan == 0.0
    stats = slo_stats(result, wait_deadline=1.0)
    assert stats["breach_rate"] == 0.0
