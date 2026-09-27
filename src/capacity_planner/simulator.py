"""Discrete-event core: machines with CPU/RAM/GPU capacity, jobs arriving over
time with resource demands and durations, and a strict-priority allocation
loop that records utilization, wait times, and rejected jobs.

Simulator semantics (the contract tests pin down):

- Every arrival joins a single queue; one allocator drains it. Immediate
  placement is just the special case of an empty queue — a new arrival never
  jumps ahead of jobs already waiting.
- Each timestamp is processed in explicit phases: (1) completions release
  capacity, (2) revocations evict running jobs and take machines offline,
  (3) machine returns wake the queue when a revocation gap ends,
  (4) arrivals enqueue, (5) the queue drains once with the full picture,
  (6) deadline expiries reject whatever is still queued past its
  deadline.
  Freed capacity is therefore visible to same-timestamp arrivals, and a job
  expiring at the exact moment capacity frees still gets placed.
- The queue drains in strict priority order — sorted by (priority, arrival) —
  with head-of-line blocking: if the head job cannot be placed, nothing
  behind it is tried. No backfilling, by design, so priority is never
  violated by a smaller job sneaking past.
- Each queued job with a finite wait_deadline gets an explicit expiry event
  at arrival + wait_deadline, so deadline rejections are recorded at the
  deadline itself, not whenever the next unrelated event happens to drain
  the queue. A job placed exactly at its deadline (wait == deadline) is
  admitted, not rejected.
- Preemptible (spot) machines are revoked as a Poisson process with rate
  `revocation_rate` revocations/hour (seeded via `Simulator(seed=...)` for
  reproducibility; the seed is re-applied on every run). On revocation the
  machine evicts every running job — partial progress is lost, and the
  attempt is erased from utilization and schedule records, so only each
  job's final successful run counts — then goes offline for
  `revocation_gap` hours before returning. A revocation that lands while
  the machine is already offline is a no-op: it evicts nothing and does not
  extend the outage. The machine's return is an explicit wakeup event so
  queued jobs are reconsidered exactly when the gap ends. Evicted jobs
  rejoin the queue with their original arrival times: deadlines are still
  measured from the original arrival, and a job requeued already past its
  deadline is rejected immediately. Each placement carries an allocation
  attempt token, so a stale completion event from an evicted attempt can
  never release a retry's capacity early. The revocation process is chained
  unconditionally, but the event loop stops once no live work remains
  (empty queue, nothing running, no arrivals/completions/expiries pending)
  — revocations and machine returns alone never keep the simulation alive
  past the end of the workload. Each job must be a distinct object;
  duplicate Job identities are rejected. Note the simulator never gives up
  on a job: with an infinite wait deadline on a machine revoked faster
  than the job can complete, evict/requeue cycles continue indefinitely —
  keep deadlines finite (or revocation rates realistic) so every run
  terminates.
- makespan = t_end - t_start, where t_start is the first event time. Idle
  time before the workload starts is not charged to the fleet, so fleet
  configurations are comparable like-for-like.
- Rejections are recorded with a reason and a timestamp:
  "hopeless" (demand exceeds every machine, rejected at arrival),
  "deadline" (wait deadline expired while queued, rejected at the deadline),
  "stranded-at-end" (still queued when the event stream ended — only
  possible with an infinite wait_deadline).
"""

from __future__ import annotations

import heapq
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Machine:
    name: str
    cpu: float
    ram: float
    gpu: float = 0.0
    cost_per_hour: float = 0.0
    # Spot/preemptible config. A preemptible machine is revoked as a Poisson
    # process with rate `revocation_rate` (revocations/hour); on revocation
    # running jobs are evicted (partial progress lost) and the machine is
    # unavailable for `revocation_gap` hours. On-demand machines
    # (preemptible=False) are never revoked. Revocations are scheduled iff
    # preemptible and revocation_rate > 0.
    preemptible: bool = False
    revocation_rate: float = 0.0
    revocation_gap: float = 5.0 / 60.0  # 5 minutes, in hours
    # runtime state (managed by Simulator)
    free_cpu: float = field(init=False)
    free_ram: float = field(init=False)
    free_gpu: float = field(init=False)
    used_cpu_hours: float = field(init=False, default=0.0)
    used_ram_hours: float = field(init=False, default=0.0)
    used_gpu_hours: float = field(init=False, default=0.0)
    peak_cpu: float = field(init=False, default=0.0)
    peak_ram: float = field(init=False, default=0.0)
    peak_gpu: float = field(init=False, default=0.0)
    unavailable_until: float = field(init=False, default=0.0)
    # Running jobs keyed by id(job): (job, allocation attempt id). The
    # attempt token distinguishes a retry from the stale completion event of
    # the evicted attempt it replaced.
    _running: dict[int, tuple["Job", int]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        # Full runtime reset, so repeat runs (and dataclasses.replace copies)
        # start from a clean machine.
        self.free_cpu = self.cpu
        self.free_ram = self.ram
        self.free_gpu = self.gpu
        self.used_cpu_hours = 0.0
        self.used_ram_hours = 0.0
        self.used_gpu_hours = 0.0
        self.peak_cpu = 0.0
        self.peak_ram = 0.0
        self.peak_gpu = 0.0
        self.unavailable_until = 0.0
        self._running = {}

    def fits(self, job: "Job", t: float = 0.0) -> bool:
        """Capacity check at time t. A machine inside its post-revocation
        unavailability gap fits nothing."""
        if t < self.unavailable_until:
            return False
        return (
            self.free_cpu >= job.cpu
            and self.free_ram >= job.ram
            and self.free_gpu >= job.gpu
        )

    def could_ever_fit(self, job: "Job") -> bool:
        return self.cpu >= job.cpu and self.ram >= job.ram and self.gpu >= job.gpu

    def allocate(self, job: "Job", t: float = 0.0, attempt: int = 0) -> None:
        assert self.fits(job, t), f"over-allocation on {self.name}: {job}"
        self.free_cpu -= job.cpu
        self.free_ram -= job.ram
        self.free_gpu -= job.gpu
        self.used_cpu_hours += job.cpu * job.duration
        self.used_ram_hours += job.ram * job.duration
        self.used_gpu_hours += job.gpu * job.duration
        self.peak_cpu = max(self.peak_cpu, self.cpu - self.free_cpu)
        self.peak_ram = max(self.peak_ram, self.ram - self.free_ram)
        self.peak_gpu = max(self.peak_gpu, self.gpu - self.free_gpu)
        self._running[id(job)] = (job, attempt)

    def release(self, job: "Job") -> None:
        self.free_cpu += job.cpu
        self.free_ram += job.ram
        self.free_gpu += job.gpu

    def _untrack(self, job: "Job") -> None:
        self._running.pop(id(job), None)

    def _active_attempt(self, job: "Job") -> int | None:
        """Attempt id of the job's currently running placement, if any."""
        entry = self._running.get(id(job))
        return entry[1] if entry is not None else None

    def evict(self, job: "Job") -> None:
        """Undo a placement after a revocation: free the capacity and erase
        the attempt's used-hours accounting. Partial progress is lost — only
        the job's final successful run counts toward utilization."""
        self.release(job)
        self.used_cpu_hours -= job.cpu * job.duration
        self.used_ram_hours -= job.ram * job.duration
        self.used_gpu_hours -= job.gpu * job.duration
        self._untrack(job)

    def utilization(self, makespan: float) -> dict[str, float]:
        if makespan <= 0:
            return {"cpu": 0.0, "ram": 0.0, "gpu": 0.0}

        def frac(used: float, cap: float) -> float:
            if not cap:
                return 0.0
            # Clamp: float accumulation can land one ulp above 1.0 on a fully
            # busy machine; utilization is a fraction by definition.
            return min(1.0, max(0.0, used / (cap * makespan)))

        return {
            "cpu": frac(self.used_cpu_hours, self.cpu),
            "ram": frac(self.used_ram_hours, self.ram),
            "gpu": frac(self.used_gpu_hours, self.gpu),
        }


@dataclass
class Job:
    name: str
    arrival: float  # hours since t=0
    cpu: float
    ram: float
    gpu: float = 0.0
    duration: float = 1.0  # hours of wall-clock hold time
    wait_deadline: float = float("inf")  # max hours willing to wait in queue
    priority: int = 0  # lower number = scheduled first when draining the queue


RejectionReason = Literal["hopeless", "deadline", "stranded-at-end"]


@dataclass
class Rejection:
    job: Job
    reason: RejectionReason
    time: float  # simulation time at which the job was rejected


@dataclass
class Eviction:
    job: Job
    machine: str  # name of the machine that was revoked
    time: float  # simulation time of the revocation


@dataclass
class SimulationResult:
    scheduled: list[tuple[Job, str, float]]  # (job, machine_name, start_time)
    rejections: list[Rejection]
    wait_times: list[float]  # admitted-job waits, in schedule order
    utilization: dict[str, dict[str, float]]  # machine -> resource -> [0,1]
    evictions: list[Eviction] = field(default_factory=list)
    t_start: float = 0.0  # time of the first event
    t_end: float = 0.0  # time of the last meaningful event (arrival,
    # completion, or rejection — no-op expiry checks don't extend the window)

    @property
    def makespan(self) -> float:
        """Active window of the workload: idle time before the first event
        is not charged to the fleet."""
        return self.t_end - self.t_start

    @property
    def rejected(self) -> list[Job]:
        return [r.job for r in self.rejections]

    @property
    def n_scheduled(self) -> int:
        return len(self.scheduled)

    @property
    def n_rejected(self) -> int:
        return len(self.rejections)


def _check_number(
    value: object,
    name: str,
    *,
    allow_inf: bool = False,
    minimum: float = 0.0,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    v = float(value)
    if math.isnan(v):
        raise ValueError(f"{name} must not be NaN")
    if math.isinf(v):
        if not allow_inf:
            raise ValueError(f"{name} must be finite, got {value!r}")
        return v
    if minimum_inclusive and v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    if not minimum_inclusive and v <= minimum:
        raise ValueError(f"{name} must be > {minimum}, got {value!r}")
    return v


def _validate_machine(machine: Machine) -> None:
    _check_number(machine.cpu, f"machine {machine.name!r} cpu")
    _check_number(machine.ram, f"machine {machine.name!r} ram")
    _check_number(machine.gpu, f"machine {machine.name!r} gpu")
    _check_number(machine.cost_per_hour, f"machine {machine.name!r} cost_per_hour")
    if not isinstance(machine.preemptible, bool):
        raise ValueError(
            f"machine {machine.name!r} preemptible must be a bool, "
            f"got {machine.preemptible!r}"
        )
    _check_number(
        machine.revocation_rate, f"machine {machine.name!r} revocation_rate"
    )
    _check_number(machine.revocation_gap, f"machine {machine.name!r} revocation_gap")


def _validate_job(job: Job) -> None:
    _check_number(job.arrival, f"job {job.name!r} arrival")
    _check_number(
        job.duration, f"job {job.name!r} duration", minimum=0.0, minimum_inclusive=False
    )
    _check_number(job.cpu, f"job {job.name!r} cpu")
    _check_number(job.ram, f"job {job.name!r} ram")
    _check_number(job.gpu, f"job {job.name!r} gpu")
    _check_number(job.wait_deadline, f"job {job.name!r} wait_deadline", allow_inf=True)
    _check_number(job.priority, f"job {job.name!r} priority")


# Event phases within a single timestamp (lower runs first). Revocations sit
# between completions and arrivals: a revocation shrinks capacity before new
# arrivals see it, and a job completing at the exact revocation instant is
# released first, not evicted. A machine return is just a wakeup — the drain
# below reconsiders the queue with the machine available again — so it runs
# before arrivals too.
_PHASE_COMPLETE = 0
_PHASE_REVOKE = 1
_PHASE_RECOVER = 2
_PHASE_ARRIVAL = 3
_PHASE_EXPIRE = 4


class Simulator:
    """Strict-priority allocator over a fixed fleet, with optional preemptible
    (spot) machines.

    Event loop: every arrival joins a single queue; each timestamp is
    processed in phases (completions, revocations, machine returns,
    arrivals, one queue drain, deadline expiries). The queue drains in
    (priority, arrival) order with head-of-line blocking — no backfilling.
    Jobs that cannot ever fit are rejected as hopeless at arrival; queued
    jobs whose wait_deadline expires are rejected at the deadline via an
    explicit expiry event; anything still queued when the event stream ends
    is rejected as stranded-at-end.

    Preemptible machines are revoked as a Poisson process with their
    `revocation_rate`; pass `seed` for reproducible revocation times (the
    seed is re-applied on every run, so repeat runs of one Simulator agree).
    """

    def __init__(self, machines: list[Machine], seed: int | None = None):
        names = [m.name for m in machines]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate machine names: {names}")
        for m in machines:
            _validate_machine(m)
        self.machines = machines
        self._seed = seed

    def run(self, jobs: list[Job]) -> SimulationResult:
        for job in jobs:
            _validate_job(job)
        if len({id(job) for job in jobs}) != len(jobs):
            raise ValueError(
                "duplicate Job objects in workload: each job must be a "
                "distinct object (attempt tracking keys on identity)"
            )
        # Fresh per-run state: the seed is re-applied so repeat runs of one
        # Simulator reproduce the same revocation schedule.
        self._seq = itertools.count()
        self._attempt = itertools.count()
        # Dedicated RNG for the revocation process: same seed -> same
        # revocation times, so spot experiments are reproducible.
        self._rng = random.Random(self._seed)
        for m in self.machines:  # reset runtime state for repeat runs
            m.__post_init__()

        events: list[tuple[float, int, int, str, object]] = []
        for job in jobs:
            heapq.heappush(
                events, (job.arrival, _PHASE_ARRIVAL, next(self._seq), "arrival", job)
            )

        # Seed the revocation Poisson process for each preemptible machine.
        # The first revocation lands after the workload starts so idle time
        # before the first arrival isn't polluted with revocation events.
        if jobs:
            t0 = min(job.arrival for job in jobs)
            for m in self.machines:
                if m.preemptible and m.revocation_rate > 0:
                    heapq.heappush(
                        events,
                        (
                            t0 + self._rng.expovariate(m.revocation_rate),
                            _PHASE_REVOKE,
                            next(self._seq),
                            "revoke",
                            m,
                        ),
                    )

        queue: list[Job] = []
        scheduled: list[tuple[Job, str, float]] = []
        rejections: list[Rejection] = []
        evictions: list[Eviction] = []
        wait_times: list[float] = []
        t_start: float | None = None
        t_end = 0.0

        def try_place(job: Job, t: float) -> bool:
            for m in self.machines:
                if m.fits(job, t):
                    attempt = next(self._attempt)
                    m.allocate(job, t, attempt)
                    heapq.heappush(
                        events,
                        (
                            t + job.duration,
                            _PHASE_COMPLETE,
                            next(self._seq),
                            "complete",
                            (m, job, attempt),
                        ),
                    )
                    scheduled.append((job, m.name, t))
                    wait_times.append(t - job.arrival)
                    return True
            return False

        def drain_queue(t: float) -> None:
            # Strict priority: head-of-line blocking, no backfill. If the
            # head job cannot be placed, nothing behind it is tried.
            queue.sort(key=lambda j: (j.priority, j.arrival))
            while queue:
                if try_place(queue[0], t):
                    del queue[0]
                else:
                    break

        def dequeue(job: Job) -> bool:
            for i, q in enumerate(queue):
                if q is job:
                    del queue[i]
                    return True
            return False

        def remove_placement(job: Job) -> None:
            # An evicted job's earlier placement is erased: the schedule and
            # wait records describe only final successful runs, so every job
            # is counted exactly once.
            for i in range(len(scheduled) - 1, -1, -1):
                if scheduled[i][0] is job:
                    del scheduled[i]
                    del wait_times[i]
                    return

        def chain_revoke(m: Machine, t: float) -> None:
            # The Poisson process continues for the whole run: every
            # processed revocation schedules its successor. The event loop's
            # liveness guard (below) stops the chain once no real work
            # remains, so revocations alone can never keep the simulation
            # alive past the end of the workload.
            heapq.heappush(
                events,
                (
                    t + self._rng.expovariate(m.revocation_rate),
                    _PHASE_REVOKE,
                    next(self._seq),
                    "revoke",
                    m,
                ),
            )

        def revoke_machine(m: Machine, t: float) -> bool:
            """Revoke one preemptible machine at time t: evict its running
            jobs (they requeue with their original arrival times), take the
            machine offline for its revocation_gap, and schedule both its
            return and the next revocation. Returns True if any job was
            evicted."""
            if t < m.unavailable_until:
                # Revoked while already offline: nothing is running, and the
                # outage must not be extended — the machine returns on its
                # original schedule.
                chain_revoke(m, t)
                return False
            running = [job for job, _ in m._running.values()]
            for job in running:
                m.evict(job)
                evictions.append(Eviction(job=job, machine=m.name, time=t))
                remove_placement(job)
                if (
                    math.isfinite(job.wait_deadline)
                    and t - job.arrival > job.wait_deadline
                ):
                    # Evicted after its deadline already passed: it rejoins
                    # the queue past its limit, so reject it now at the
                    # revocation time rather than letting it wait pointlessly.
                    rejections.append(Rejection(job, "deadline", t))
                else:
                    queue.append(job)
                    if math.isfinite(job.wait_deadline):
                        heapq.heappush(
                            events,
                            (
                                job.arrival + job.wait_deadline,
                                _PHASE_EXPIRE,
                                next(self._seq),
                                "expire",
                                job,
                            ),
                        )
            m.unavailable_until = t + m.revocation_gap
            # Wakeup so the queue is reconsidered when the machine returns —
            # placement only happens at event timestamps.
            heapq.heappush(
                events,
                (
                    m.unavailable_until,
                    _PHASE_RECOVER,
                    next(self._seq),
                    "recover",
                    m,
                ),
            )
            chain_revoke(m, t)
            return len(running) > 0

        def live_work_pending() -> bool:
            # Revocations and machine returns alone are not real work: once
            # the queue is empty, nothing is running, and no arrivals,
            # completions, or expiries remain, the simulation is over even if
            # revocation events are still scheduled.
            if queue or any(m._running for m in self.machines):
                return True
            return any(
                kind in ("arrival", "complete", "expire")
                for _, _, _, kind, _ in events
            )

        while events and live_work_pending():
            t = events[0][0]
            if t_start is None:
                t_start = t
            batch: list[tuple[float, int, int, str, object]] = []
            while events and events[0][0] == t:
                batch.append(heapq.heappop(events))
            # t_end tracks the last event that did real work (an arrival, a
            # completion, an eviction, or a rejection). No-op expiry checks
            # and idle revocations must not stretch the window with idle
            # tail time.
            meaningful = False

            # Phase 1: completions release capacity first. A completion is
            # honored only if its allocation attempt is still the job's
            # active one: evicted attempts leave stale completion events
            # behind, and a retry may already be running under a newer
            # attempt — releasing on the stale event would free the retry's
            # capacity early.
            for _, phase, _, kind, payload in batch:
                if phase == _PHASE_COMPLETE:
                    m, job, attempt = payload  # type: ignore[misc]
                    if m._active_attempt(job) == attempt:
                        m.release(job)
                        m._untrack(job)
                        meaningful = True

            # Phase 2: revocations evict running jobs and take the machine
            # offline for its gap. Capacity shrinks before arrivals see it.
            for _, phase, _, kind, payload in batch:
                if phase == _PHASE_REVOKE:
                    m = payload  # type: ignore[assignment]
                    if revoke_machine(m, t):
                        meaningful = True

            # Phase 3: machine returns are just a wakeup — the drain below
            # reconsiders the queue with the machine available again.
            # (No-op when the queue is empty or nothing fits.)
            for _, phase, _, kind, _payload in batch:
                if phase == _PHASE_RECOVER:
                    continue  # wakeup only; the drain does the work

            # Phase 4: arrivals always join the queue (hopeless ones are
            # rejected outright). Finite-deadline jobs get an expiry event.
            for _, phase, _, kind, payload in batch:
                if phase == _PHASE_ARRIVAL:
                    meaningful = True
                    job = payload  # type: ignore[assignment]
                    if not any(m.could_ever_fit(job) for m in self.machines):
                        rejections.append(Rejection(job, "hopeless", t))
                    else:
                        queue.append(job)
                        if math.isfinite(job.wait_deadline):
                            heapq.heappush(
                                events,
                                (
                                    job.arrival + job.wait_deadline,
                                    _PHASE_EXPIRE,
                                    next(self._seq),
                                    "expire",
                                    job,
                                ),
                            )

            # Phase 5: one drain with the full picture (freed capacity +
            # new arrivals, minus revoked machines). Immediate placement is
            # the empty-queue case.
            drain_queue(t)

            # Phase 6: expiries — jobs still queued at their deadline are
            # rejected now, at the deadline, not at some later event.
            for _, phase, _, kind, payload in batch:
                if phase == _PHASE_EXPIRE:
                    job = payload  # type: ignore[assignment]
                    if dequeue(job):
                        rejections.append(Rejection(job, "deadline", t))
                        meaningful = True

            if meaningful:
                t_end = t

        # Anything still queued at end of time had an infinite deadline and
        # simply never found room: stranded-at-end.
        for job in queue:
            rejections.append(Rejection(job, "stranded-at-end", t_end))

        if t_start is None:  # empty workload
            t_start = 0.0
        makespan = t_end - t_start
        utilization = {m.name: m.utilization(makespan) for m in self.machines}
        return SimulationResult(
            scheduled=scheduled,
            rejections=rejections,
            wait_times=wait_times,
            utilization=utilization,
            evictions=evictions,
            t_start=t_start,
            t_end=t_end,
        )
