"""Priority preemption: a blocked high-priority head may displace
strictly-lower-priority running jobs (opt-in via preemption_enabled)."""

import pytest

from capacity_planner.simulator import Job, Machine, Simulator


def _m(**kw):
    base = dict(name="m", cpu=4.0, ram=8.0, cost_per_hour=1.0)
    base.update(kw)
    return Machine(**base)


def _j(name, arrival, priority, duration=10.0, cpu=4.0, deadline=float("inf")):
    return Job(
        name=name,
        arrival=arrival,
        cpu=cpu,
        ram=4.0,
        duration=duration,
        wait_deadline=deadline,
        priority=priority,
    )


def _starts(result, name):
    return [t for job, _m, t in result.scheduled if job.name == name]


def test_preemption_fires_when_head_blocked():
    lo = _j("lo", 0.0, priority=1, duration=10.0)
    hi = _j("hi", 1.0, priority=0, duration=1.0)
    res = Simulator([_m()], preemption_enabled=True).run([lo, hi])
    assert res.n_preemptions == 1
    assert lo.preemptions == 1
    # hi started immediately at its arrival; lo was displaced once
    assert _starts(res, "hi") == [1.0]
    assert res.wait_times[res.scheduled.index((hi, "m", 1.0))] == 0.0
    # one schedule entry per job: the resume is a continuation, not a
    # re-admission — so SLO stats never double-count or mislabel execution
    # time as wait
    assert _starts(res, "lo") == [0.0]
    assert res.n_scheduled == 2
    assert res.wait_times == [0.0, 0.0]
    # checkpoint: lo ran 0..1 (1h), resumed at 2 with 9h left -> done at 11
    assert res.t_end == pytest.approx(11.0)


def test_resume_never_counts_as_breach():
    # The resume placement must not be mistaken for an admission: waits
    # measure time to first admission, so lo (started at t=0, resumed at
    # t=2) reads as a 0h wait even under a strict global deadline.
    from capacity_planner.cost import slo_stats

    lo = _j("lo", 0.0, priority=1, duration=10.0)
    hi = _j("hi", 1.0, priority=0, duration=1.0)
    res = Simulator([_m()], preemption_enabled=True).run([lo, hi])
    stats = slo_stats(res, wait_deadline=1.0)
    assert stats["breach_rate"] == pytest.approx(0.0)
    assert stats["avg_wait_admitted"] == pytest.approx(0.0)


def test_preempted_work_counted_once_in_utilization():
    # lo: 4cpu x (1h + 9h) = 40 cpu-h; hi: 4cpu x 1h = 4 cpu-h.
    # makespan 11 on a 4-cpu machine -> cpu utilization exactly 1.0.
    lo = _j("lo", 0.0, priority=1, duration=10.0)
    hi = _j("hi", 1.0, priority=0, duration=1.0)
    res = Simulator([_m()], preemption_enabled=True).run([lo, hi])
    assert res.utilization["m"]["cpu"] == pytest.approx(1.0)


def test_no_preemption_when_capacity_free():
    lo = _j("lo", 0.0, priority=1, duration=5.0, cpu=4.0)
    hi = _j("hi", 1.0, priority=0, duration=1.0, cpu=4.0)
    res = Simulator([_m(cpu=8.0)], preemption_enabled=True).run([lo, hi])
    assert res.n_preemptions == 0
    assert _starts(res, "hi") == [1.0]
    assert _starts(res, "lo") == [0.0]


def test_never_preempts_equal_priority():
    h1 = _j("h1", 0.0, priority=0, duration=5.0)
    h2 = _j("h2", 1.0, priority=0, duration=1.0)
    res = Simulator([_m()], preemption_enabled=True).run([h1, h2])
    assert res.n_preemptions == 0
    assert _starts(res, "h2") == [5.0]  # waited for h1


def test_lower_priority_number_cannot_preempt():
    # priority 2 is *lower* than priority 1: it must not displace it.
    lo = _j("lo", 0.0, priority=1, duration=5.0)
    lower = _j("lower", 1.0, priority=2, duration=1.0)
    res = Simulator([_m()], preemption_enabled=True).run([lo, lower])
    assert res.n_preemptions == 0
    assert _starts(res, "lower") == [5.0]


def test_victim_selection_lowest_priority_first():
    a = _j("a", 0.0, priority=1, duration=10.0, cpu=4.0)
    b = _j("b", 0.0, priority=2, duration=10.0, cpu=4.0)
    hi = _j("hi", 1.0, priority=0, duration=1.0, cpu=4.0)
    res = Simulator([_m(cpu=8.0)], preemption_enabled=True).run([a, b, hi])
    assert res.n_preemptions == 1
    assert b.preemptions == 1  # lowest priority displaced...
    assert a.preemptions == 0  # ...while the higher-priority job kept running
    assert _starts(res, "hi") == [1.0]


def test_victim_tiebreak_largest_remaining():
    # Same priority and cpu: the job with more remaining is displaced first.
    a = _j("a", 0.0, priority=1, duration=10.0, cpu=4.0)  # remaining 9 at t=1
    b = _j("b", 0.0, priority=1, duration=2.0, cpu=4.0)  # remaining 1 at t=1
    hi = _j("hi", 1.0, priority=0, duration=1.0, cpu=4.0)
    res = Simulator([_m(cpu=8.0)], preemption_enabled=True).run([a, b, hi])
    assert a.preemptions == 1
    assert b.preemptions == 0


def test_preemption_cap_prevents_ping_pong():
    # lo is displaced twice, becomes immune, then high-priority arrivals
    # must wait instead of preempting forever.
    lo = _j("lo", 0.0, priority=1, duration=30.0)
    hs = [_j(f"h{i}", float(t), priority=0, duration=5.0) for i, t in enumerate([1, 7, 13, 19])]
    res = Simulator([_m()], preemption_enabled=True).run([lo, *hs])
    assert res.n_preemptions == 2
    assert lo.preemptions == 2
    # h2 (arrived 13) could not displace the immune lo: it waited for lo's
    # 12..40 run to finish.
    assert _starts(res, "h2") == [40.0]
    assert _starts(res, "h3") == [45.0]
    assert res.t_end == pytest.approx(50.0)


def test_preempted_past_deadline_rejected_immediately():
    lo = _j("lo", 0.0, priority=1, duration=10.0, deadline=2.0)
    hi = _j("hi", 5.0, priority=0, duration=1.0)
    res = Simulator([_m()], preemption_enabled=True).run([lo, hi])
    assert res.n_preemptions == 1
    reasons = {r.job.name: (r.reason, r.time) for r in res.rejections}
    # lo was displaced at t=5, already past its deadline=2 -> rejected now
    assert reasons["lo"] == ("deadline", 5.0)


def test_default_off_bit_for_bit_equivalence():
    lo = _j("lo", 0.0, priority=1, duration=10.0)
    hi = _j("hi", 1.0, priority=0, duration=1.0)
    jobs_a = [lo, hi]
    default = Simulator([_m()]).run(jobs_a)
    # fresh objects: the first run mutated runtime state
    lo2 = _j("lo", 0.0, priority=1, duration=10.0)
    hi2 = _j("hi", 1.0, priority=0, duration=1.0)
    explicit_off = Simulator([_m()], preemption_enabled=False).run([lo2, hi2])
    assert [(j.name, m, t) for j, m, t in default.scheduled] == [
        (j.name, m, t) for j, m, t in explicit_off.scheduled
    ]
    assert [(r.job.name, r.reason) for r in default.rejections] == [
        (r.job.name, r.reason) for r in explicit_off.rejections
    ]
    assert default.wait_times == explicit_off.wait_times
    assert default.n_preemptions == 0 == explicit_off.n_preemptions
    # and with the flag off, the high-priority job simply waits
    assert _starts(explicit_off, "hi") == [10.0]


def test_preemption_enabled_must_be_bool():
    with pytest.raises(ValueError):
        Simulator([_m()], preemption_enabled="yes")


def test_repeat_runs_reset_preemption_state():
    lo = _j("lo", 0.0, priority=1, duration=10.0)
    hi = _j("hi", 1.0, priority=0, duration=1.0)
    sim = Simulator([_m()], preemption_enabled=True)
    r1 = sim.run([lo, hi])
    r2 = sim.run([lo, hi])
    assert r1.n_preemptions == r2.n_preemptions == 1
    assert [t for _, _, t in r1.scheduled] == [t for _, _, t in r2.scheduled]
