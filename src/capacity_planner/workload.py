"""Synthetic workload generator: diurnal arrivals, mixed job sizes.

Produces a seeded, reproducible stream of Jobs shaped like real service
traffic: arrivals follow a sinusoidal daily pattern (busy afternoons, quiet
nights) with noise, and jobs come in small/medium/large/gpu classes with
different resource demands, durations, deadlines, and priorities.

The generator is the demand side of the capacity-planning loop: pair it
with the simulator and cost model to ask "what fleet do I need for *this*
demand?"
"""

from __future__ import annotations

import math
import random

from .simulator import Job

# (name, weight, cpu_range, ram_range, gpu, duration_range_h, deadline_h, priority)
# Demands are sized against the example fleet machine (16 cpu / 64 ram / 2 gpu):
# a "large" job takes half a machine, so packing behavior actually matters.
_CLASSES = [
    ("small", 0.55, (1, 2), (2, 4), 0.0, (0.25, 1.0), 0.5, 0),
    ("medium", 0.30, (4, 4), (8, 16), 0.0, (1.0, 3.0), 1.0, 1),
    ("large", 0.10, (8, 8), (32, 32), 0.0, (2.0, 6.0), 4.0, 2),
    ("gpu", 0.05, (4, 4), (16, 16), 1.0, (0.5, 2.0), 0.5, 0),
]


def _rate(t: float, base: float, amplitude: float, peak_hour: float) -> float:
    """Instantaneous arrival rate (jobs/hour) at hour t of the day."""
    # sin peaks at t == peak_hour: sin(2*pi*(peak-peak+6)/24) = sin(pi/2) = 1
    return base + amplitude * math.sin(2 * math.pi * (t - peak_hour + 6) / 24.0)


def generate_workload(
    seed: int = 0,
    hours: float = 24.0,
    base_rate: float = 12.0,
    amplitude: float = 8.0,
    peak_hour: float = 15.0,
) -> list[Job]:
    """Generate a seeded job stream over `hours` hours.

    Arrivals are a non-homogeneous Poisson process (thinning): candidate
    arrivals at the peak rate are accepted with probability rate(t)/peak.
    Each arrival draws a job class (small/medium/large/gpu) with fixed
    weights; demands and durations are uniform within the class ranges.
    """
    if hours <= 0:
        raise ValueError("hours must be positive")
    if base_rate < 0 or amplitude < 0 or base_rate < amplitude:
        raise ValueError("need 0 <= amplitude <= base_rate for a non-negative rate")
    rng = random.Random(seed)

    peak_rate = base_rate + amplitude
    arrivals: list[float] = []
    t = 0.0
    while True:
        t += rng.expovariate(peak_rate)
        if t >= hours:
            break
        if rng.random() < _rate(t, base_rate, amplitude, peak_hour) / peak_rate:
            arrivals.append(t)

    weights = [c[1] for c in _CLASSES]
    jobs: list[Job] = []
    for i, at in enumerate(arrivals):
        name, _, cpu_r, ram_r, gpu, dur_r, deadline, prio = rng.choices(
            _CLASSES, weights=weights, k=1
        )[0]
        jobs.append(
            Job(
                name=f"{name}-{i}",
                arrival=at,
                cpu=rng.uniform(*cpu_r),
                ram=rng.uniform(*ram_r),
                gpu=gpu,
                duration=rng.uniform(*dur_r),
                wait_deadline=deadline,
                priority=prio,
            )
        )
    return jobs
