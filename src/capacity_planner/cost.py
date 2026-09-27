"""Cost model: turn a simulated schedule into dollars and SLO stats, so fleet
size becomes a cost-vs-SLO tradeoff curve instead of a guess."""

from __future__ import annotations

import math
import statistics

from .simulator import Machine, SimulationResult


def total_cost(machines: list[Machine], horizon_hours: float) -> float:
    """Fleet cost = sum over machines of cost_per_hour * horizon_hours.

    Conservative: machines are paid for the whole window (reserved capacity),
    which is the right model for owned hardware or committed reservations.
    Pass the same horizon for every fleet configuration you compare so the
    comparison is like-for-like.
    """
    if horizon_hours < 0:
        raise ValueError("horizon_hours must be non-negative")
    return sum(m.cost_per_hour * horizon_hours for m in machines)


def _percentile(data: list[float], pct: float) -> float:
    """Nearest-rank percentile: rank = ceil(pct/100 * n), clamped to [1, n].

    Unlike statistics.quantiles' exclusive estimator, this never exceeds the
    sample max (or drops below the min) on small samples, and returns the
    single observation for n=1.
    """
    if not data:
        return 0.0
    ordered = sorted(data)
    rank = min(max(math.ceil(pct / 100 * len(ordered)), 1), len(ordered))
    return ordered[rank - 1]


def slo_stats(
    result: SimulationResult, wait_deadline: float | None = None
) -> dict[str, float]:
    """SLO summary for one simulated fleet configuration.

    Breach accounting: a job breaches if its wait exceeds its applicable
    deadline, or if it was rejected outright. By default (wait_deadline=None)
    each job is measured against its own wait_deadline; pass an explicit
    wait_deadline to override every job's deadline with one global threshold
    (e.g. to measure admitted latency against a stricter SLO than the jobs
    were scheduled under).

    Note: the simulator rejects queued jobs at their deadline via explicit
    expiry events, so with per-job deadlines every admitted job waited within
    its own deadline by construction — breaches in that mode come from
    rejections. The global override is what surfaces admitted-job latency
    pressure.

    avg_wait_admitted / p95_wait_admitted are admitted-job latency: waits of
    scheduled jobs only (rejected jobs have no wait to average).
    """
    n = result.n_scheduled + result.n_rejected
    if n == 0:
        return {"breach_rate": 0.0, "avg_wait_admitted": 0.0, "p95_wait_admitted": 0.0}
    if wait_deadline is not None and (
        not isinstance(wait_deadline, (int, float))
        or isinstance(wait_deadline, bool)
        or math.isnan(wait_deadline)
        or wait_deadline < 0
    ):
        raise ValueError(f"wait_deadline override must be a non-negative number, got {wait_deadline!r}")

    breached = 0
    waits: list[float] = []
    for job, _, start in result.scheduled:
        w = start - job.arrival
        waits.append(w)
        limit = wait_deadline if wait_deadline is not None else job.wait_deadline
        if w > limit:
            breached += 1
    breached += result.n_rejected
    return {
        "breach_rate": breached / n,
        "avg_wait_admitted": statistics.fmean(waits) if waits else 0.0,
        "p95_wait_admitted": _percentile(waits, 95),
    }


def tradeoff_point(
    machines: list[Machine],
    result: SimulationResult,
    wait_deadline: float | None = None,
    horizon: float | None = None,
) -> dict[str, float]:
    """One point on the cost-vs-SLO curve.

    `horizon` is the window the fleet is priced over; it defaults to the
    result's makespan, but pass an explicit shared horizon when comparing
    fleet configurations so cost is like-for-like across different
    makespans.
    """
    stats = slo_stats(result, wait_deadline)
    h = result.makespan if horizon is None else horizon
    if h < 0:
        raise ValueError("horizon must be non-negative")
    return {
        "cost": total_cost(machines, h),
        "breach_rate": stats["breach_rate"],
        "avg_wait_admitted": stats["avg_wait_admitted"],
        "p95_wait_admitted": stats["p95_wait_admitted"],
        "makespan": result.makespan,
        "horizon": h,
    }
