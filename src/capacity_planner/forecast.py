"""Demand forecasting: turn a history of job arrivals into a forward-looking
demand curve the simulator and cost model can plan against.

Starts simple on purpose — moving average and exponential smoothing are
honest baselines. A fancier model adapter slot is stubbed below.
"""

from __future__ import annotations


def bucket_arrivals(arrivals: list[float], bucket_size: float) -> list[int]:
    """Count arrivals per fixed-size time bucket (hours)."""
    if not arrivals:
        return []
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    n_buckets = int(max(arrivals) // bucket_size) + 1
    counts = [0] * n_buckets
    for t in arrivals:
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
