"""Regression tests for the Codex review fix round.

Each test pins one corrected behavior: queue discipline, phase ordering,
deadline expiries, makespan semantics, per-job SLO accounting, the p95
estimator, input validation, and rejection reasons.
"""

import pytest

from capacity_planner.cost import slo_stats, total_cost, tradeoff_point
from capacity_planner.forecast import bucket_arrivals
from capacity_planner.simulator import Job, Machine, SimulationResult, Simulator


def one_machine() -> list[Machine]:
    return [Machine("m", cpu=8, ram=32, cost_per_hour=2.0)]


def starts_by_name(result: SimulationResult) -> dict[str, float]:
    return {job.name: start for job, _, start in result.scheduled}


def rejection_by_name(result: SimulationResult):
    return {r.job.name: r for r in result.rejections}


# --- Item 1: arrivals never bypass the queue ---------------------------------


def test_arrival_does_not_bypass_queue():
    machines = one_machine()
    jobs = [
        Job("j1", arrival=0.0, cpu=4, ram=8, duration=10.0),
        # Needs the whole machine; only 4 cpu free -> queues.
        Job("j2", arrival=1.0, cpu=8, ram=8, duration=1.0),
        # Fits in the 4 free cpu, but must NOT jump ahead of j2.
        Job("j3", arrival=2.0, cpu=4, ram=8, duration=1.0),
    ]
    result = Simulator(machines).run(jobs)
    starts = starts_by_name(result)
    assert result.n_rejected == 0
    assert starts["j1"] == 0.0
    assert starts["j2"] == 10.0  # after j1 completes
    # j3 waited for the queue to drain instead of bypassing it on arrival.
    assert starts["j3"] == 11.0
    assert starts["j3"] > starts["j2"]


# --- Item 10: strict priority, no backfill ------------------------------------


def test_no_backfill_behind_blocked_head():
    # Same scenario as above doubles as the backfill test: j2 is the queue
    # head and cannot be placed, so j3 — which would fit — is not backfilled
    # around it. It starts only after j2 has run.
    machines = one_machine()
    jobs = [
        Job("j1", arrival=0.0, cpu=4, ram=8, duration=10.0),
        Job("j2", arrival=1.0, cpu=8, ram=8, duration=1.0),
        Job("j3", arrival=2.0, cpu=4, ram=8, duration=1.0),
    ]
    result = Simulator(machines).run(jobs)
    starts = starts_by_name(result)
    assert starts["j3"] != 2.0  # a backfilling scheduler would start j3 at 2.0
    assert starts["j3"] == 11.0


def test_higher_priority_runs_first():
    machines = one_machine()
    jobs = [
        Job("blocker", arrival=0.0, cpu=8, ram=8, duration=5.0),
        Job("low", arrival=1.0, cpu=8, ram=8, duration=1.0, priority=9),
        Job("high", arrival=1.5, cpu=8, ram=8, duration=1.0, priority=0),
    ]
    result = Simulator(machines).run(jobs)
    starts = starts_by_name(result)
    assert starts["high"] == 5.0
    assert starts["low"] == 6.0


# --- Item 2: completions before arrivals at the same timestamp ----------------


def test_completion_before_arrival_same_timestamp():
    machines = one_machine()
    jobs = [
        Job("j1", arrival=0.0, cpu=8, ram=8, duration=5.0),
        # Arrives exactly when j1 completes, with a zero wait deadline:
        # the freed capacity must be visible, so it is placed immediately.
        Job("j2", arrival=5.0, cpu=8, ram=8, duration=1.0, wait_deadline=0.0),
    ]
    result = Simulator(machines).run(jobs)
    assert result.n_rejected == 0
    assert starts_by_name(result)["j2"] == 5.0
    assert all(w == 0.0 for w in result.wait_times)


def test_zero_deadline_rejected_when_no_capacity_frees():
    machines = one_machine()
    jobs = [
        Job("j1", arrival=0.0, cpu=8, ram=8, duration=6.0),  # completes at 6, not 5
        Job("j2", arrival=5.0, cpu=8, ram=8, duration=1.0, wait_deadline=0.0),
    ]
    result = Simulator(machines).run(jobs)
    assert result.n_rejected == 1
    rej = rejection_by_name(result)["j2"]
    assert rej.reason == "deadline"
    assert rej.time == 5.0


# --- Item 3: deadline expiry is an explicit event ------------------------------


def test_deadline_rejection_fires_at_deadline():
    machines = one_machine()
    jobs = [
        Job("blocker", arrival=0.0, cpu=8, ram=8, duration=10.0),
        Job("victim", arrival=1.0, cpu=8, ram=8, duration=1.0, wait_deadline=3.0),
    ]
    result = Simulator(machines).run(jobs)
    assert result.n_rejected == 1
    rej = rejection_by_name(result)["victim"]
    assert rej.reason == "deadline"
    # Rejected at arrival + deadline (4.0), not at the next drain (10.0).
    assert rej.time == pytest.approx(4.0)


def test_job_expiring_at_completion_time_gets_placed():
    machines = one_machine()
    jobs = [
        Job("j1", arrival=0.0, cpu=8, ram=8, duration=5.0),
        # Deadline expires exactly when j1 completes; drain runs before
        # expiry, so the freed slot saves it. wait == deadline is admitted.
        Job("j2", arrival=1.0, cpu=8, ram=8, duration=1.0, wait_deadline=4.0),
    ]
    result = Simulator(machines).run(jobs)
    assert result.n_rejected == 0
    assert starts_by_name(result)["j2"] == 5.0


# --- Item 4: makespan excludes pre-workload idle --------------------------------


def test_makespan_excludes_pre_workload_idle():
    machines = one_machine()
    jobs = [Job("late", arrival=100.0, cpu=8, ram=8, duration=1.0)]
    result = Simulator(machines).run(jobs)
    assert result.t_start == 100.0
    assert result.t_end == 101.0
    assert result.makespan == pytest.approx(1.0)
    # Utilization is measured over the active window, not the idle prefix.
    assert result.utilization["m"]["cpu"] == pytest.approx(1.0)


def test_tradeoff_point_accepts_explicit_horizon():
    machines = one_machine()
    result = Simulator(machines).run(
        [Job("late", arrival=100.0, cpu=8, ram=8, duration=1.0)]
    )
    default = tradeoff_point(machines, result)
    assert default["horizon"] == pytest.approx(result.makespan)
    assert default["cost"] == pytest.approx(total_cost(machines, result.makespan))
    # A shared horizon makes fleet configs comparable like-for-like.
    shared = tradeoff_point(machines, result, horizon=24.0)
    assert shared["horizon"] == 24.0
    assert shared["cost"] == pytest.approx(2.0 * 24.0)
    assert shared["makespan"] == pytest.approx(result.makespan)


# --- Item 5: per-job deadline breach accounting ---------------------------------


def test_per_job_deadline_breach_accounting():
    machines = one_machine()
    jobs = [
        Job("blocker", arrival=0.0, cpu=8, ram=8, duration=2.0, wait_deadline=10.0),
        # Waits 2h; fine against its own infinite deadline...
        Job("patient", arrival=0.0, cpu=8, ram=8, duration=1.0),
        # ...but rejected at its own 0.5h deadline (t=1.0).
        Job("impatient", arrival=0.5, cpu=8, ram=8, duration=1.0, wait_deadline=0.5),
    ]
    result = Simulator(machines).run(jobs)
    assert result.n_rejected == 1

    per_job = slo_stats(result)
    # Only the rejected job breaches against per-job deadlines.
    assert per_job["breach_rate"] == pytest.approx(1 / 3)
    # Admitted waits are labeled as admitted-job latency.
    assert per_job["avg_wait_admitted"] == pytest.approx(1.0)
    assert per_job["p95_wait_admitted"] == pytest.approx(2.0)

    # A global override measures every job against one threshold instead.
    override = slo_stats(result, wait_deadline=1.0)
    # patient waited 2h > 1h, impatient was rejected.
    assert override["breach_rate"] == pytest.approx(2 / 3)


def test_rejected_jobs_count_as_breaches():
    machines = one_machine()
    jobs = [
        # Hopeless: rejected outright, counts as a breach.
        Job("hopeless", arrival=0.0, cpu=64, ram=8, duration=1.0),
        Job("fine", arrival=0.0, cpu=8, ram=8, duration=1.0),
    ]
    result = Simulator(machines).run(jobs)
    stats = slo_stats(result)
    assert stats["breach_rate"] == pytest.approx(1 / 2)


# --- Item 6: small-sample-sane p95 ----------------------------------------------


def _result_with_waits(waits: list[float]) -> SimulationResult:
    scheduled = [
        (Job(f"j{i}", arrival=0.0, cpu=1, ram=1, duration=1.0), "m", w)
        for i, w in enumerate(waits)
    ]
    return SimulationResult(
        scheduled=scheduled,
        rejections=[],
        wait_times=list(waits),
        utilization={},
    )


def test_p95_single_sample_equals_sample():
    stats = slo_stats(_result_with_waits([5.0]))
    assert stats["p95_wait_admitted"] == 5.0


def test_p95_never_exceeds_max_wait():
    for waits in (
        [1.0, 2.0],
        [0.0, 0.0, 10.0],
        [3.0, 1.0, 2.0],
        [7.0, 7.0, 7.0, 7.0, 7.0, 0.0],
        [0.5],
    ):
        p95 = slo_stats(_result_with_waits(list(waits)))["p95_wait_admitted"]
        assert min(waits) <= p95 <= max(waits), waits


# --- Item 7: input validation ----------------------------------------------------


def test_simulator_rejects_invalid_inputs():
    machines = one_machine()
    with pytest.raises(ValueError):
        Simulator(machines).run([Job("j", arrival=-1.0, cpu=1, ram=1)])
    with pytest.raises(ValueError):
        Simulator(machines).run(
            [Job("j", arrival=0.0, cpu=1, ram=1, duration=0.0)]
        )
    with pytest.raises(ValueError):
        Simulator(machines).run(
            [Job("j", arrival=0.0, cpu=1, ram=1, duration=-2.0)]
        )
    with pytest.raises(ValueError):
        Simulator(machines).run(
            [Job("j", arrival=0.0, cpu=float("nan"), ram=1)]
        )
    with pytest.raises(ValueError):
        Simulator(machines).run(
            [Job("j", arrival=float("inf"), cpu=1, ram=1)]
        )
    with pytest.raises(ValueError):
        Simulator(machines).run(
            [Job("j", arrival=0.0, cpu=1, ram=1, wait_deadline=float("nan"))]
        )
    # Infinite wait_deadline is the legitimate default — allowed.
    Simulator(machines).run(
        [Job("j", arrival=0.0, cpu=1, ram=1, wait_deadline=float("inf"))]
    )
    with pytest.raises(ValueError):
        Simulator([Machine("m", cpu=8, ram=32), Machine("m", cpu=4, ram=16)])
    with pytest.raises(ValueError):
        Simulator([Machine("bad", cpu=-2, ram=32)])
    with pytest.raises(ValueError):
        Simulator([Machine("bad", cpu=8, ram=32, cost_per_hour=float("nan"))])


# --- Item 8: bucket_arrivals validation ------------------------------------------


def test_bucket_arrivals_validates_before_empty_return():
    with pytest.raises(ValueError):
        bucket_arrivals([], 0.0)
    with pytest.raises(ValueError):
        bucket_arrivals([], -1.0)


def test_bucket_arrivals_rejects_bad_times():
    with pytest.raises(ValueError):
        bucket_arrivals([-0.5], 1.0)  # no silent negative-index wrap
    with pytest.raises(ValueError):
        bucket_arrivals([float("nan")], 1.0)
    with pytest.raises(ValueError):
        bucket_arrivals([float("inf")], 1.0)


def test_bucket_arrivals_end_time_gives_trailing_zeros():
    assert bucket_arrivals([0.1], 1.0, end_time=2.5) == [1, 0, 0]
    assert bucket_arrivals([], 1.0, end_time=1.0) == [0, 0]
    assert bucket_arrivals([], 1.0) == []
    # Unchanged behavior without end_time.
    assert bucket_arrivals([0.1, 0.4, 1.2, 1.8, 2.3, 3.1], 1.0) == [2, 2, 1, 1]
    with pytest.raises(ValueError):
        bucket_arrivals([0.1], 1.0, end_time=-1.0)


# --- Item 9: rejection reasons are distinguishable ---------------------------------


def test_rejection_reasons_are_distinguishable():
    machines = one_machine()
    jobs = [
        # Too big for any machine: hopeless at arrival.
        Job("hopeless", arrival=0.0, cpu=64, ram=8, duration=1.0),
        Job("blocker", arrival=0.0, cpu=8, ram=8, duration=10.0),
        # Queued, deadline expires at 1.0 + 2.0 = 3.0 while blocked.
        Job("doomed", arrival=1.0, cpu=8, ram=8, duration=1.0, wait_deadline=2.0),
    ]
    result = Simulator(machines).run(jobs)
    by_name = rejection_by_name(result)
    assert set(by_name) == {"hopeless", "doomed"}
    assert by_name["hopeless"].reason == "hopeless"
    assert by_name["hopeless"].time == 0.0
    assert by_name["doomed"].reason == "deadline"
    assert by_name["doomed"].time == pytest.approx(3.0)
    # The two reasons are different values, not a single boolean.
    assert by_name["hopeless"].reason != by_name["doomed"].reason
