"""Discrete-event core: machines with CPU/RAM/GPU capacity, jobs arriving over
time with resource demands and durations, and a priority-aware FIFO allocation
loop that records utilization, wait times, and rejected jobs."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field


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
        return {
            "cpu": self.used_cpu_hours / (self.cpu * makespan) if self.cpu else 0.0,
            "ram": self.used_ram_hours / (self.ram * makespan) if self.ram else 0.0,
            "gpu": self.used_gpu_hours / (self.gpu * makespan) if self.gpu else 0.0,
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


@dataclass
class SimulationResult:
    makespan: float
    scheduled: list[tuple[Job, str, float]]  # (job, machine_name, start_time)
    rejected: list[Job]
    wait_times: list[float]
    utilization: dict[str, dict[str, float]]  # machine -> resource -> [0,1]

    @property
    def n_scheduled(self) -> int:
        return len(self.scheduled)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)


class Simulator:
    """Priority-aware FIFO allocator over a fixed fleet.

    Event loop: on each job arrival, try to place it immediately. If no machine
    has room, the job waits in a queue drained (by priority, then arrival order)
    on every job completion. A job that cannot ever fit on any machine, or that
    has waited past its wait_deadline, is rejected.
    """

    def __init__(self, machines: list[Machine]):
        self.machines = machines
        self._seq = itertools.count()

    def run(self, jobs: list[Job]) -> SimulationResult:
        for m in self.machines:  # reset runtime state for repeat runs
            m.__post_init__()
            m.used_cpu_hours = m.used_ram_hours = m.used_gpu_hours = 0.0
            m.peak_cpu = m.peak_ram = m.peak_gpu = 0.0

        events: list[tuple[float, int, str, object]] = []
        for job in jobs:
            heapq.heappush(events, (job.arrival, next(self._seq), "arrival", job))

        queue: list[Job] = []
        scheduled: list[tuple[Job, str, float]] = []
        rejected: list[Job] = []
        wait_times: list[float] = []
        now = 0.0

        def try_place(job: Job, t: float) -> bool:
            for m in self.machines:
                if m.fits(job):
                    m.allocate(job)
                    heapq.heappush(
                        events, (t + job.duration, next(self._seq), "complete", (m, job))
                    )
                    scheduled.append((job, m.name, t))
                    wait_times.append(t - job.arrival)
                    return True
            return False

        def drain_queue(t: float) -> None:
            queue.sort(key=lambda j: (j.priority, j.arrival))
            remaining: list[Job] = []
            for job in queue:
                if t - job.arrival > job.wait_deadline:
                    rejected.append(job)
                    continue
                if not try_place(job, t):
                    remaining.append(job)
            queue[:] = remaining

        while events:
            t, _, kind, payload = heapq.heappop(events)
            now = t
            if kind == "arrival":
                job = payload  # type: ignore[assignment]
                if not any(m.could_ever_fit(job) for m in self.machines):
                    rejected.append(job)  # demand exceeds every machine: hopeless
                elif not try_place(job, t):
                    queue.append(job)
            else:  # "complete"
                m, job = payload  # type: ignore[misc]
                m.release(job)
                drain_queue(t)

        # Anything still queued at end of time missed its chance: reject it.
        for job in queue:
            rejected.append(job)

        utilization = {m.name: m.utilization(now) for m in self.machines}
        return SimulationResult(
            makespan=now,
            scheduled=scheduled,
            rejected=rejected,
            wait_times=wait_times,
            utilization=utilization,
        )
