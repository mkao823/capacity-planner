"""Spot / preemptible machines: revocation evicts and requeues, deadlines are
measured from the original arrival, on-demand behavior is untouched, and
mix sweeps price heterogeneous fleets like-for-like."""

import pytest

from capacity_planner.simulator import Job, Machine, Simulator
from capacity_planner.sweep import (
    identical_fleet,
    spot_fleet,
    sweep_mixes,
)
from capacity_planner.workload import generate_workload

NODE = {"cpu": 16, "ram": 64, "gpu": 2, "cost_per_hour": 4.0}


def spot_machine(name="spot", rate=100.0, **kw):
    args = dict(NODE)
    args.update(kw)
    return Machine(
        name,
        preemptible=True,
        revocation_rate=rate,
        **{k: v for k, v in args.items()},
    )


def test_revocation_evicts_and_requeues():
    # Ridiculous revocation rate: the job is evicted almost immediately, then
    # the machine is effectively unusable (5min gap >> 36s mean inter-revoke),
    # so a 1h-deadline job bounces between evictions until its original
    # deadline expires. The point: eviction happened, the attempt left no
    # accounting trace, and capacity was never inflated.
    m = spot_machine(rate=100.0)
    job = Job("j", arrival=0.0, cpu=4, ram=8, duration=5.0, wait_deadline=1.0)
    result = Simulator([m], seed=0).run([job])

    assert len(result.evictions) >= 1
    assert result.evictions[0].machine == "spot"
    assert result.evictions[0].time > 0
    # Evicted attempts are erased: nothing ran to completion here.
    assert m.used_cpu_hours == pytest.approx(0.0)
    # No over-allocation from the orphaned completion event.
    assert m.free_cpu <= m.cpu + 1e-9
    assert m.peak_cpu <= m.cpu + 1e-9


def test_evicted_job_completes_cleanly():
    # Moderate rate: evicted once, then completes. Exactly one schedule
    # entry, one wait entry, and a single run's worth of used hours.
    m = spot_machine(rate=0.5)
    job = Job("j", arrival=0.0, cpu=4, ram=8, duration=2.0,
              wait_deadline=float("inf"))
    result = Simulator([m], seed=1).run([job])

    assert len(result.evictions) >= 1
    assert result.n_scheduled == 1
    assert result.n_rejected == 0
    assert len(result.wait_times) == 1
    assert m.used_cpu_hours == pytest.approx(job.cpu * job.duration)
    # Wait is measured from the original arrival, including the false start.
    assert result.wait_times[0] == pytest.approx(
        result.scheduled[0][2] - job.arrival
    )


def test_evicted_job_deadline_measured_from_original_arrival():
    # Deadline 1h: evicted almost immediately, machine unusable afterwards,
    # so the deadline (from the ORIGINAL arrival) expires while queued.
    m = spot_machine(rate=100.0)
    job = Job("j", arrival=0.0, cpu=4, ram=8, duration=5.0, wait_deadline=1.0)
    result = Simulator([m], seed=0).run([job])

    assert result.n_scheduled == 0
    assert len(result.rejections) == 1
    rej = result.rejections[0]
    assert rej.reason == "deadline"
    assert rej.time >= job.arrival + job.wait_deadline - 1e-9


def test_revocation_gap_takes_machine_offline():
    # After a revocation the machine fits nothing until the gap elapses.
    # Finite deadline so the bouncing job terminates via deadline rejection.
    m = spot_machine(rate=100.0, revocation_gap=1.0)
    job = Job("j", arrival=0.0, cpu=4, ram=8, duration=0.5, wait_deadline=0.5)
    Simulator([m], seed=0).run([job])
    t_revoke = m.unavailable_until - m.revocation_gap
    assert t_revoke >= 0
    assert not m.fits(job, t_revoke + 0.5 * m.revocation_gap)
    assert m.fits(job, t_revoke + m.revocation_gap)


def test_ondemand_unaffected_by_preemption_codepath():
    # preemptible=True with revocation_rate=0 must behave exactly like a
    # plain on-demand machine: no revocation events are ever scheduled.
    workload = [
        Job(f"j{i}", arrival=float(i) * 0.5, cpu=2, ram=4, duration=1.0,
            wait_deadline=2.0)
        for i in range(8)
    ]
    base = Simulator([Machine("m", **NODE)], seed=42).run(workload)
    spotty = Simulator(
        [Machine("m", **NODE, preemptible=True, revocation_rate=0.0)], seed=42
    ).run(workload)

    assert base.scheduled == spotty.scheduled
    assert base.n_rejected == spotty.n_rejected
    assert base.wait_times == spotty.wait_times
    assert base.evictions == spotty.evictions == []


def test_revocation_is_seeded_and_reproducible():
    workload = generate_workload(seed=3, hours=6.0)
    template = Machine("n", **NODE)
    factory = spot_fleet(template, revocation_rate=0.5)
    fleet_a = [factory(i) for i in range(3)]
    fleet_b = [factory(i) for i in range(3)]
    ra = Simulator(fleet_a, seed=11).run(list(workload))
    rb = Simulator(fleet_b, seed=11).run(list(workload))

    assert [(e.job.name, e.machine, e.time) for e in ra.evictions] == [
        (e.job.name, e.machine, e.time) for e in rb.evictions
    ]
    assert [s[2] for s in ra.scheduled] == [s[2] for s in rb.scheduled]


def test_sweep_mixes_one_point_per_mix():
    workload = generate_workload(seed=5, hours=4.0)
    template = Machine("node", **NODE)
    od = identical_fleet(template)
    spot = spot_fleet(template, revocation_rate=0.5)
    mixes = [
        {"on-demand": (od, 4), "spot": (spot, 0)},
        {"on-demand": (od, 2), "spot": (spot, 4)},
        {"on-demand": (od, 0), "spot": (spot, 6)},
    ]
    points = sweep_mixes(workload, mixes, horizon=4.0, seed=9)

    assert len(points) == 3
    assert points[0]["mix"] == {"on-demand": 4, "spot": 0}
    assert points[1]["mix"] == {"on-demand": 2, "spot": 4}
    assert points[2]["mix"] == {"on-demand": 0, "spot": 6}
    for p in points:
        assert set(p["mix"]) == {"on-demand", "spot"}


def test_sweep_mixes_rejects_empty_and_negative():
    workload = generate_workload(seed=5, hours=2.0)
    template = Machine("node", **NODE)
    od = identical_fleet(template)
    with pytest.raises(ValueError):
        sweep_mixes(workload, [{"on-demand": (od, 0), "spot": (od, 0)}])
    with pytest.raises(ValueError):
        sweep_mixes(workload, [{"on-demand": (od, -1)}])


def test_spot_cheaper_but_breachier_than_ondemand():
    # Same total machine count; spot mix must win on cost and lose on SLO
    # when revocations actually bite.
    workload = generate_workload(seed=8, hours=8.0)
    template = Machine("node", **NODE)
    od = identical_fleet(template)
    spot = spot_fleet(template, discount=0.3, revocation_rate=1.0)
    points = sweep_mixes(
        workload,
        [
            {"on-demand": (od, 6), "spot": (spot, 0)},
            {"on-demand": (od, 3), "spot": (spot, 3)},
        ],
        horizon=8.0,
        seed=21,
    )
    pure, mixed = points

    assert pure["mix"] == {"on-demand": 6, "spot": 0}
    assert mixed["cost"] < pure["cost"]
    assert mixed["breach_rate"] >= pure["breach_rate"]
    # Sanity: the discount actually applied — mixed cost is well below the
    # all-on-demand price for 6 machines.
    assert mixed["cost"] < 0.75 * pure["cost"]


def test_spot_fleet_validates_discount():
    template = Machine("node", **NODE)
    with pytest.raises(ValueError):
        spot_fleet(template, discount=0)
    with pytest.raises(ValueError):
        spot_fleet(template, discount=-0.5)


def test_machine_validates_preemption_fields():
    with pytest.raises(ValueError):
        Simulator([Machine("m", **NODE, revocation_rate=-1.0)])
    with pytest.raises(ValueError):
        Simulator([Machine("m", **NODE, preemptible="yes")])  # type: ignore[arg-type]


def test_stale_completion_cannot_finish_retry_early():
    # The evicted attempt's completion event must not release the retry's
    # capacity: job2 (arriving while the retry runs) may only start once
    # the retry truly finishes.
    m = Machine(
        "spot",
        cpu=8,
        ram=16,
        cost_per_hour=1.0,
        preemptible=True,
        revocation_rate=0.3,
    )
    j1 = Job("j1", arrival=0.0, cpu=8, ram=8, duration=10.0,
             wait_deadline=float("inf"))
    j2 = Job("j2", arrival=10.5, cpu=8, ram=8, duration=1.0,
             wait_deadline=float("inf"))
    result = Simulator([m], seed=8).run([j1, j2])

    assert len(result.evictions) >= 1  # j1 evicted mid-run, retried
    j1_start = next(t for s, _, t in result.scheduled if s.name == "j1")
    j2_starts = [t for s, _, t in result.scheduled if s.name == "j2"]
    assert len(j2_starts) == 1  # counted exactly once
    # j2 starts only after j1's retry runs its full duration — the stale
    # completion at t=10 (evicted attempt) freed nothing.
    assert j2_starts[0] >= j1_start + j1.duration - 1e-9


def test_queued_job_retried_when_gap_ends():
    # No unrelated events after the outage: the machine-return wakeup alone
    # must get the queued job running again, exactly when the gap ends.
    # (Without the wakeup, j2 would strand with the machine idle.)
    m = spot_machine(rate=1.0, revocation_gap=1.0)
    j1 = Job("j1", arrival=0.0, cpu=4, ram=8, duration=0.5, wait_deadline=0.3)
    j2 = Job("j2", arrival=0.05, cpu=4, ram=8, duration=0.5,
             wait_deadline=float("inf"))
    result = Simulator([m], seed=31).run([j1, j2])

    assert [e.job.name for e in result.evictions] == ["j1"]
    assert result.n_scheduled == 1  # j2 ran to completion; j1 was rejected
    j2_start = next(t for s, _, t in result.scheduled if s.name == "j2")
    assert j2_start == pytest.approx(result.evictions[0].time + m.revocation_gap)


def test_revoke_during_outage_does_not_extend_gap():
    # Two revocations inside one gap: the machine still returns on the
    # first revocation's schedule.
    m = spot_machine(rate=2.0, revocation_gap=1.0)
    job = Job("j", arrival=0.0, cpu=4, ram=8, duration=1.0, wait_deadline=0.5)
    result = Simulator([m], seed=0).run([job])

    assert len(result.evictions) == 1
    assert result.n_rejected == 1  # evicted past its deadline
    assert m.unavailable_until == pytest.approx(
        result.evictions[0].time + m.revocation_gap
    )


def test_repeat_runs_are_reproducible():
    workload = generate_workload(seed=3, hours=6.0)
    template = Machine("n", **NODE)
    factory = spot_fleet(template, revocation_rate=0.5)
    sim = Simulator([factory(i) for i in range(3)], seed=11)
    ra = sim.run(list(workload))
    rb = sim.run(list(workload))  # same Simulator, seed re-applied

    assert [(e.job.name, e.machine, e.time) for e in ra.evictions] == [
        (e.job.name, e.machine, e.time) for e in rb.evictions
    ]
    assert [s[2] for s in ra.scheduled] == [s[2] for s in rb.scheduled]
    assert ra.n_rejected == rb.n_rejected


def test_duplicate_job_objects_rejected():
    job = Job("j", arrival=0.0, cpu=1, ram=1, duration=0.5)
    with pytest.raises(ValueError, match="duplicate Job objects"):
        Simulator([Machine("m", **NODE)]).run([job, job])
