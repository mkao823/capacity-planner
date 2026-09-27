"""Demand forecasting: turn a history of job arrivals into a forward-looking
demand curve the simulator and cost model can plan against.

Starts simple on purpose — moving average and exponential smoothing are
honest baselines. A fancier model adapter slot is stubbed below.
"""

from __future__ import annotations

import math


def _check_arrival_time(t: object) -> float:
    if isinstance(t, bool) or not isinstance(t, (int, float)):
        raise ValueError(f"arrivals must be numbers, got {t!r}")
    v = float(t)
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"arrivals must be finite, got {t!r}")
    if v < 0:
        raise ValueError(f"arrivals must be non-negative, got {t!r}")
    return v


def bucket_arrivals(
    arrivals: list[float], bucket_size: float, end_time: float | None = None
) -> list[int]:
    """Count arrivals per fixed-size time bucket (hours).

    `end_time` optionally extends the bucket range so trailing zero-demand
    buckets are representable (e.g. planning a window that outlasts the last
    observed arrival). Negative or non-finite arrival times raise ValueError
    instead of silently wrapping into the wrong bucket.
    """
    if (
        isinstance(bucket_size, bool)
        or not isinstance(bucket_size, (int, float))
        or math.isnan(bucket_size)
        or math.isinf(bucket_size)
        or bucket_size <= 0
    ):
        raise ValueError(f"bucket_size must be a positive finite number, got {bucket_size!r}")
    if end_time is not None:
        end_time = _check_arrival_time(end_time)
    times = [_check_arrival_time(t) for t in arrivals]
    n_buckets = 0
    if times:
        n_buckets = int(max(times) // bucket_size) + 1
    if end_time is not None:
        n_buckets = max(n_buckets, int(end_time // bucket_size) + 1)
    counts = [0] * n_buckets
    for t in times:
        counts[int(t // bucket_size)] += 1
    return counts


def moving_average_forecast(
    series: list[float], window: int, horizon: int
) -> list[float]:
    """Forecast = mean of the last `window` observations, repeated for `horizon` steps."""
    if not series:
        raise ValueError("series must not be empty")
    if window <= 0 or horizon <= 0:
        raise ValueError("window and horizon must be positive")
    level = sum(series[-window:]) / min(window, len(series))
    return [level] * horizon


def exponential_smoothing_forecast(
    series: list[float], alpha: float, horizon: int
) -> list[float]:
    """Simple exponential smoothing (no trend/seasonality)."""
    if not series:
        raise ValueError("series must not be empty")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    level = series[0]
    for x in series[1:]:
        level = alpha * x + (1 - alpha) * level
    return [level] * horizon


class ProphetForecaster:
    """Adapter slot for a real statistical model (Prophet / ARIMA / etc.).

    Not implemented yet — the two functions above are the baselines any fancier
    model has to beat. Implement fit()/predict() here when the dependency is
    available.
    """

    def fit(self, series: list[float]) -> "ProphetForecaster":  # noqa: D102
        raise NotImplementedError(
            "ProphetForecaster is a stub: install `prophet` and implement fit/predict."
        )

    def predict(self, horizon: int) -> list[float]:  # noqa: D102
        raise NotImplementedError(
            "ProphetForecaster is a stub: install `prophet` and implement fit/predict."
        )
