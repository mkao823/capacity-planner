"""Discrete-event core: machines with CPU/RAM/GPU capacity, jobs arriving over
time with resource demands and durations, and a strict-priority allocation
loop that records utilization, wait times, and rejected jobs.

Simulator semantics (the contract tests pin down):

- Every arrival joins a single queue; one allocator drains it. Immediate
  placement is just the special case of an empty queue — a new arrival never
  jumps ahead of jobs already waiting.
- Each timestamp is processed in explicit phases: (1) completions release
  capacity, (2) arrivals enqueue, (3) the queue drains once with the full
  picture, (4) deadline expiries reject whatever is still queued past its
  deadline. Freed capacity is therefore visible to same-timestamp arrivals,
  and a job expiring at the exact moment capacity frees still gets placed.
- The queue drains in strict priority order — sorted by (priority, arrival) —
  with head-of-line blocking: if the head job cannot be placed, nothing
  behind it is tried. No backfilling, by design, so priority is never
  violated by a smaller job sneaking past.
- Each queued job with a finite wait_deadline gets an explicit expiry event
  at arrival + wait_deadline, so deadline rejections are recorded at the
  deadline itself, not whenever the next unrelated event happens to drain
  the queue. A job placed exactly at its deadline (wait == deadline) is
  admitted, not rejected.
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
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Machine:
    name: str
    cpu: float
    ram: float
    gpu: float = 0.0
    cost_per_hour: float = 0.0
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

    def __post_init__(self) -> None:
        self.free_cpu = self.cpu
        self.free_ram = self.ram
        self.free_gpu = self.gpu

    def fits(self, job: "Job") -> bool:
        return (
            self.free_cpu >= job.cpu
            and self.free_ram >= job.ram
            and self.free_gpu >= job.gpu
        )

    def could_ever_fit(self, job: "Job") -> bool:
        return self.cpu >= job.cpu and self.ram >= job.ram and self.gpu >= job.gpu

    def allocate(self, job: "Job") -> None:
        assert self.fits(job), f"over-allocation on {self.name}: {job}"
        self.free_cpu -= job.cpu
        self.free_ram -= job.ram
        self.free_gpu -= job.gpu
        self.used_cpu_hours += job.cpu * job.duration
        self.used_ram_hours += job.ram * job.duration
        self.used_gpu_hours += job.gpu * job.duration
        self.peak_cpu = max(self.peak_cpu, self.cpu - self.free_cpu)
        self.peak_ram = max(self.peak_ram, self.ram - self.free_ram)
        self.peak_gpu = max(self.peak_gpu, self.gpu - self.free_gpu)

    def release(self, job: "Job") -> None:
        self.free_cpu += job.cpu
        self.free_ram += job.ram
        self.free_gpu += job.gpu

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
class SimulationResult:
    scheduled: list[tuple[Job, str, float]]  # (job, machine_name, start_time)
    rejections: list[Rejection]
    wait_times: list[float]  # admitted-job waits, in schedule order
    utilization: dict[str, dict[str, float]]  # machine -> resource -> [0,1]
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


# Event phases within a single timestamp (lower runs first).
_PHASE_COMPLETE = 0
_PHASE_ARRIVAL = 1
_PHASE_EXPIRE = 2


class Simulator:
    """Strict-priority allocator over a fixed fleet.

    Event loop: every arrival joins a single queue; each timestamp is
    processed in phases (completions, arrivals, one queue drain, deadline
    expiries). The queue drains in (priority, arrival) order with
    head-of-line blocking — no backfilling. Jobs that cannot ever fit are
    rejected as hopeless at arrival; queued jobs whose wait_deadline expires
    are rejected at the deadline via an explicit expiry event; anything still
    queued when the event stream ends is rejected as stranded-at-end.
    """

    def __init__(self, machines: list[Machine]):
        names = [m.name for m in machines]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate machine names: {names}")
        for m in machines:
            _validate_machine(m)
        self.machines = machines
        self._seq = itertools.count()

    def run(self, jobs: list[Job]) -> SimulationResult:
        for job in jobs:
            _validate_job(job)
        for m in self.machines:  # reset runtime state for repeat runs
            m.__post_init__()
            m.used_cpu_hours = m.used_ram_hours = m.used_gpu_hours = 0.0
            m.peak_cpu = m.peak_ram = m.peak_gpu = 0.0

        events: list[tuple[float, int, int, str, object]] = []
        for job in jobs:
            heapq.heappush(
                events, (job.arrival, _PHASE_ARRIVAL, next(self._seq), "arrival", job)
            )

        queue: list[Job] = []
        scheduled: list[tuple[Job, str, float]] = []
        rejections: list[Rejection] = []
        wait_times: list[float] = []
        t_start: float | None = None
        t_end = 0.0

        def try_place(job: Job, t: float) -> bool:
            for m in self.machines:
                if m.fits(job):
                    m.allocate(job)
                    heapq.heappush(
                        events,
                        (
                            t + job.duration,
                            _PHASE_COMPLETE,
                            next(self._seq),
                            "complete",
                            (m, job),
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

        while events:
            t = events[0][0]
            if t_start is None:
                t_start = t
            batch: list[tuple[float, int, int, str, object]] = []
            while events and events[0][0] == t:
                batch.append(heapq.heappop(events))
            # t_end tracks the last event that did real work (an arrival, a
            # completion, or a rejection). No-op expiry checks for already
            # placed jobs must not stretch the window with idle tail time.
            meaningful = False

            # Phase 1: completions release capacity first.
            for _, phase, _, kind, payload in batch:
                if phase == _PHASE_COMPLETE:
                    m, job = payload  # type: ignore[misc]
                    m.release(job)
                    meaningful = True

            # Phase 2: arrivals always join the queue (hopeless ones are
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

            # Phase 3: one drain with the full picture (freed capacity +
            # new arrivals). Immediate placement is the empty-queue case.
            drain_queue(t)

            # Phase 4: expiries — jobs still queued at their deadline are
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
            t_start=t_start,
            t_end=t_end,
        )
