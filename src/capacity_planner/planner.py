"""Planning against forecasts: close the capacity-planning loop.

The sweep in sweep.py plans against *known* demand — a cheat. Real planners
only ever see history. This module:

1. picks a demand forecaster by backtesting on history (select_forecaster),
2. turns the forecast into a planned workload (scale_workload_to_counts),
3. sizes the fleet against the planned workload (plan_fleet),
4. evaluates the chosen fleet against held-out actual demand (evaluate_plan),

so forecast error can be priced in dollars and SLO points
(cost_of_being_wrong).

Caveats, stated plainly:
- The forecast predicts bucket *counts* only — not the job mix or the
  intra-bucket timing shape. The planned workload is synthesized by
  resampling, which smooths burstiness relative to real Poisson traffic, so
  raw-forecast picks skew slightly optimistic. `plan_fleet` therefore plans
  against a *margined* forecast by default when asked: a residual-quantile
  uplift from the holdout backtest and/or a flat safety factor (see
  plan_fleet). Real planners do the same — nobody provisions against the
  point forecast; they provision against an upper prediction interval.
- Shape error matters as much as level error: a front-loaded evening looks
  the same as a flat afternoon in bucket totals but schedules very
  differently.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from .cost import tradeoff_point
from .forecast import (
    bucket_arrivals,
    exponential_smoothing_forecast,
    moving_average_forecast,
)
from .simulator import Job, Simulator
from .sweep import FleetFactory, cheapest_meeting_slo, sweep_fleet_sizes
from .workload import generate_workload

# Forecaster candidates tried by select_forecaster: (method, params).
# Small grid on purpose — these are baselines any fancier model must beat.
_CANDIDATES: list[tuple[str, dict]] = [
    ("moving_average", {"window": 6}),
    ("moving_average", {"window": 12}),
    ("moving_average", {"window": 24}),
    ("exponential_smoothing", {"alpha": 0.1}),
    ("exponential_smoothing", {"alpha": 0.3}),
    ("exponential_smoothing", {"alpha": 0.5}),
]


def _fit_forecast(
    name: str, params: dict, train: list[float], horizon: int
) -> list[float]:
    if name == "moving_average":
        return moving_average_forecast(train, params["window"], horizon)
    return exponential_smoothing_forecast(train, params["alpha"], horizon)


def select_forecaster(series: Sequence[float], horizon: int) -> dict:
    """Pick the best baseline forecaster by holdout backtest.

    Fits every candidate in _CANDIDATES on the first 80% of `series`, scores
    MAE on the last 20%, and refits the winner on the full series to forecast
    `horizon` steps. Ties break toward the simpler model (moving average).

    Returns {"name", "params", "backtest_mae", "backtest_residuals", "forecast",
    "candidate_mae": {candidate: mae}}. backtest_residuals are the winner's
    holdout errors (actual - predicted), used downstream for margin sizing.
    """
    series = [float(x) for x in series]
    if len(series) < 4:
        raise ValueError(
            f"need at least 4 observations to backtest, got {len(series)}"
        )
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError(f"horizon must be a positive int of steps, got {horizon!r}")
    if any(not math.isfinite(x) or x < 0 for x in series):
        raise ValueError("series must contain finite non-negative counts")

    split = max(1, int(len(series) * 0.8))
    train, test = series[:split], series[split:]

    scored: list[tuple[float, int, str, dict, list[float]]] = []
    # (mae, tiebreak, name, params, residuals)
    for name, params in _CANDIDATES:
        fc = _fit_forecast(name, params, train, len(test))
        resid = [a - p for a, p in zip(test, fc)]
        mae = sum(abs(r) for r in resid) / len(resid)
        tiebreak = 0 if name == "moving_average" else 1
        scored.append((mae, tiebreak, name, params, resid))
    scored.sort(key=lambda r: (r[0], r[1]))
    best_mae, _, best_name, best_params, best_resid = scored[0]

    return {
        "name": best_name,
        "params": dict(best_params),
        "backtest_mae": best_mae,
        "backtest_residuals": best_resid,
        "forecast": _fit_forecast(best_name, best_params, series, horizon),
        "candidate_mae": {
            f"{n}({','.join(f'{k}={v}' for k, v in p.items())})": m
            for m, _, n, p, _ in scored
        },
    }


def _clone_at(job: Job, arrival: float, name: str) -> Job:
    """Copy a job's demand profile to a new arrival time (synthetic filler)."""
    return Job(
        name=name,
        arrival=arrival,
        cpu=job.cpu,
        ram=job.ram,
        gpu=job.gpu,
        duration=job.duration,
        wait_deadline=job.wait_deadline,
        priority=job.priority,
    )


def scale_workload_to_counts(
    base_jobs: Sequence[Job],
    counts: Sequence[float],
    bucket_size: float,
    seed: int = 0,
) -> list[Job]:
    """Resample a base workload so each bucket holds the forecasted count.

    For each bucket: subsample when the base has more jobs than forecasted,
    keep-all plus clones (same demand profile, re-drawn arrival inside the
    bucket) when it has fewer. The job-class mix is preserved in expectation
    because every filler job is cloned from the base workload. Deterministic
    for a fixed seed.
    """
    if not base_jobs:
        raise ValueError("base_jobs must not be empty")
    if bucket_size <= 0:
        raise ValueError(f"bucket_size must be positive, got {bucket_size!r}")
    base_jobs = list(base_jobs)
    rng = random.Random(seed)
    n_buckets = len(counts)
    by_bucket: list[list[Job]] = [[] for _ in range(n_buckets)]
    for job in base_jobs:
        b = int(job.arrival // bucket_size)
        if 0 <= b < n_buckets:
            by_bucket[b].append(job)

    planned: list[Job] = []
    for b, (jobs_b, target) in enumerate(zip(by_bucket, counts)):
        n_target = max(0, int(round(target)))
        start = b * bucket_size
        if n_target <= len(jobs_b):
            planned.extend(rng.sample(jobs_b, n_target))
        else:
            planned.extend(jobs_b)
            for k in range(n_target - len(jobs_b)):
                src = rng.choice(base_jobs)
                planned.append(
                    _clone_at(
                        src,
                        start + rng.random() * bucket_size,
                        f"synth-b{b}-{k}",
                    )
                )
    planned.sort(key=lambda j: j.arrival)
    return planned


def residual_quantile_margin(residuals: Sequence[float], quantile: float) -> float:
    """Upper-quantile margin from holdout residuals (actual - predicted).

    Nearest-rank: the smallest value m such that at least `quantile` of the
    residuals are <= m. Planning against forecast + m therefore covers the
    historical under-prediction rate by construction — the poor man's upper
    prediction interval, straight from the backtest we already run.

    Clamped at 0: a margin never *reduces* the plan. If the backtest says
    the forecaster over-predicted at this quantile, the margin is zero.
    """
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile!r}")
    residuals = [float(r) for r in residuals]
    if not residuals:
        return 0.0
    if any(not math.isfinite(r) for r in residuals):
        raise ValueError("residuals must be finite")
    ordered = sorted(residuals)
    rank = min(max(math.ceil(quantile * len(ordered)), 1), len(ordered))
    return max(0.0, ordered[rank - 1])


def plan_fleet(
    history: Sequence[float],
    make_machine: FleetFactory,
    sizes: Sequence[int],
    horizon: float,
    slo: float,
    bucket_size: float = 1.0,
    seed: int = 0,
    wait_deadline: float | None = None,
    margin_quantile: float | None = None,
    safety_factor: float = 0.0,
) -> dict:
    """Size a fleet from forecasted demand, not known demand.

    `history` is past job arrival timestamps (hours). They are bucketed,
    the best baseline forecaster is picked by backtest, and the forecast is
    turned into a planned workload that the fleet sweep sizes against.

    Safety margins (the optimism-bias fix): the forecast predicts bucket
    counts only and resampling smooths burstiness, so raw-forecast plans
    skew optimistic. Two independent, composable margins, applied in order:

      margined_bucket = (forecast_bucket + quantile_uplift) * (1 + safety_factor)

    - margin_quantile=q: after the holdout backtest, take the q-th upper
      quantile of residuals (actual - predicted) via nearest-rank, floored
      at 0, and add it to every forecast bucket. q=0.9 means: plan against
      a demand level the backtest under-predicted at most 10% of the time.
      A single global quantile is applied per bucket — simple, documented,
      and honest about the small holdout sample.
    - safety_factor=f: flat headroom multiplier; f=0.2 is +20% on every
      bucket. Use when you want a policy number ("always 20% headroom")
      rather than a data-driven one.

    A margin never shrinks the plan: the quantile uplift is floored at 0
    and the factor is non-negative.

    Returns a diagnostics dict with keys: n_machines (None when no fleet
    size meets the SLO), point, points, forecaster, forecaster_params,
    backtest_mae, forecast, forecast_total, history_total, planned_total,
    plus margin diagnostics: margin_quantile, safety_factor,
    margin_jobs_per_bucket (the quantile uplift applied), planned_counts
    (the margined per-bucket forecast), margin_uplift_total.
    """
    history = list(history)
    if not history:
        raise ValueError("history must not be empty")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon!r}")
    if not 0 <= slo <= 1:
        raise ValueError(f"slo must be a breach rate in [0, 1], got {slo!r}")
    if margin_quantile is not None and not 0 < margin_quantile < 1:
        raise ValueError(
            f"margin_quantile must be in (0, 1), got {margin_quantile!r}"
        )
    if not math.isfinite(safety_factor) or safety_factor < 0:
        raise ValueError(
            f"safety_factor must be a non-negative number, got {safety_factor!r}"
        )

    hist_end = max(history)
    series = bucket_arrivals(history, bucket_size, end_time=hist_end)
    n_buckets = max(1, math.ceil(horizon / bucket_size))
    sel = select_forecaster(series, n_buckets)

    # Margin: quantile uplift from the holdout residuals, then flat factor.
    uplift = (
        residual_quantile_margin(sel["backtest_residuals"], margin_quantile)
        if margin_quantile is not None
        else 0.0
    )
    planned_counts = [c + uplift for c in sel["forecast"]]
    if safety_factor:
        planned_counts = [c * (1.0 + safety_factor) for c in planned_counts]

    # Base workload rich enough to subsample from: ~2x the margined total.
    total_planned = sum(planned_counts)
    base_hours = n_buckets * bucket_size
    base_rate = max(2.0, 2.0 * total_planned / base_hours)
    base = generate_workload(
        seed=seed,
        hours=base_hours,
        base_rate=base_rate,
        amplitude=base_rate / 2,
    )
    planned = scale_workload_to_counts(base, planned_counts, bucket_size, seed + 1)

    points = sweep_fleet_sizes(
        planned, make_machine, sizes, horizon=horizon, wait_deadline=wait_deadline
    )
    pick = cheapest_meeting_slo(points, slo)
    return {
        "n_machines": pick["n_machines"] if pick else None,
        "point": pick,
        "points": points,
        "forecaster": sel["name"],
        "forecaster_params": sel["params"],
        "backtest_mae": sel["backtest_mae"],
        "forecast": sel["forecast"],
        "forecast_total": sum(sel["forecast"]),
        "history_total": len(history),
        "planned_total": len(planned),
        "margin_quantile": margin_quantile,
        "safety_factor": safety_factor,
        "margin_jobs_per_bucket": uplift,
        "planned_counts": planned_counts,
        "margin_uplift_total": total_planned - sum(sel["forecast"]),
    }


def evaluate_plan(
    n_machines: int,
    make_machine: FleetFactory,
    actual_workload: Sequence[Job],
    horizon: float,
    wait_deadline: float | None = None,
) -> dict:
    """Run a chosen fleet against held-out actual demand: what really happened."""
    if not isinstance(n_machines, int) or isinstance(n_machines, bool) or n_machines <= 0:
        raise ValueError(f"n_machines must be a positive int, got {n_machines!r}")
    fleet = [make_machine(i) for i in range(n_machines)]
    result = Simulator(fleet).run(list(actual_workload))
    point = tradeoff_point(
        fleet, result, wait_deadline=wait_deadline, horizon=horizon
    )
    point["n_machines"] = n_machines
    return point


def cost_of_being_wrong(oracle_point: dict, realized_point: dict) -> dict:
    """Price the forecast error: oracle plan vs what the forecast plan did.

    Positive extra_cost = over-provisioned (paid for idle machines);
    positive breach_delta = under-provisioned (SLO damage vs the oracle).
    """
    extra = realized_point["cost"] - oracle_point["cost"]
    base = oracle_point["cost"]
    return {
        "extra_cost": extra,
        "extra_cost_pct": (extra / base * 100.0) if base else 0.0,
        "breach_delta": realized_point["breach_rate"] - oracle_point["breach_rate"],
        "oracle_n_machines": oracle_point["n_machines"],
        "realized_n_machines": realized_point["n_machines"],
        "oracle_cost": oracle_point["cost"],
        "realized_cost": realized_point["cost"],
    }


# Re-export the factory helper most callers need next to plan_fleet.
__all__ = [
    "select_forecaster",
    "residual_quantile_margin",
    "scale_workload_to_counts",
    "plan_fleet",
    "evaluate_plan",
    "cost_of_being_wrong",
]
