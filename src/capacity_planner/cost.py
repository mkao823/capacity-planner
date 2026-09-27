"""Cost model: turn a simulated schedule into dollars and SLO stats, so fleet
size becomes a cost-vs-SLO tradeoff curve instead of a guess."""

from __future__ import annotations

import statistics

from .simulator import Machine, SimulationResult


def total_cost(machines: list[Machine], makespan_hours: float) -> float:
    """Fleet cost = sum over machines of cost_per_hour * makespan.

    Conservative: machines are paid for the whole window (reserved capacity),
    which is the right model for owned hardware or committed reservations.
    """
    if makespan_hours < 0:
        raise ValueError("makespan_hours must be non-negative")
    return sum(m.cost_per_hour * makespan_hours for m in machines)


def slo_stats(result: SimulationResult, wait_deadline: float) -> dict[str, float]:
    """SLO summary for one simulated fleet configuration.

    A job breaches SLO if it waited longer than `wait_deadline` or was
    rejected outright.
    """
    n = result.n_scheduled + result.n_rejected
    if n == 0:
        return {"breach_rate": 0.0, "avg_wait": 0.0, "p95_wait": 0.0}
    breached = sum(1 for w in result.wait_times if w > wait_deadline)
    breached += result.n_rejected
    waits = result.wait_times or [0.0]
    return {
        "breach_rate": breached / n,
        "avg_wait": statistics.fmean(result.wait_times) if result.wait_times else 0.0,
        "p95_wait": statistics.quantiles(waits, n=100)[94]
        if len(waits) > 1
        else waits[0],
    }


def tradeoff_point(
    machines: list[Machine], result: SimulationResult, wait_deadline: float
) -> dict[str, float]:
    """One point on the cost-vs-SLO curve: {'cost': ..., 'breach_rate': ...}.

    Sweep fleet sizes, collect these, and you have the curve a capacity plan
    is actually decided on.
    """
    stats = slo_stats(result, wait_deadline)
    return {
        "cost": total_cost(machines, result.makespan),
        "breach_rate": stats["breach_rate"],
        "avg_wait": stats["avg_wait"],
        "p95_wait": stats["p95_wait"],
        "makespan": result.makespan,
    }
