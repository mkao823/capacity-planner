"""Fleet sweeps: run one workload against many fleet configurations and
collect the cost-vs-SLO tradeoff curve.

`sweep_fleet_sizes` covers the homogeneous case (N identical machines per
point). `sweep_mixes` generalizes to heterogeneous fleets — e.g. on-demand
vs spot mixes — over the same shared horizon so every point is like-for-like.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence

from .cost import tradeoff_point
from .simulator import Job, Machine, Simulator

# Maps a machine index to a fresh Machine. Factories must return a new
# instance per call — the simulator mutates machines as it runs.
FleetFactory = Callable[[int], Machine]


def identical_fleet(template: Machine) -> FleetFactory:
    """Build a factory that stamps out identical copies of `template`.

    Every field is carried over (including spot config), so this is also
    how you build on-demand fleets from a plain template.
    """

    def make(i: int) -> Machine:
        return dataclasses.replace(template, name=f"{template.name}-{i}")

    return make


def spot_fleet(
    template: Machine,
    *,
    discount: float = 0.3,
    revocation_rate: float = 0.2,
    revocation_gap: float = 5.0 / 60.0,
) -> FleetFactory:
    """Stamp out preemptible (spot) copies of `template`.

    Same capacity as the template, priced at `discount * cost_per_hour`
    (`discount` is a price multiplier — typical real-world spot pricing is
    ~0.3), revoked as a Poisson process with rate `revocation_rate`
    revocations/hour, returning after `revocation_gap` hours. Use a
    different template name per group in a mix so machine names stay unique.
    """
    def _check_nonneg(label: str, value: float, *, positive: bool = False) -> None:
        ok = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            and not (positive and value == 0)
        )
        if not ok:
            need = "a positive number" if positive else "a non-negative number"
            raise ValueError(f"{label} must be {need}, got {value!r}")

    _check_nonneg("discount", discount, positive=True)
    _check_nonneg("revocation_rate", revocation_rate)
    _check_nonneg("revocation_gap", revocation_gap)

    def make(i: int) -> Machine:
        return dataclasses.replace(
            template,
            name=f"{template.name}-spot-{i}",
            cost_per_hour=template.cost_per_hour * discount,
            preemptible=True,
            revocation_rate=revocation_rate,
            revocation_gap=revocation_gap,
        )

    return make


def sweep_fleet_sizes(
    workload: Sequence[Job],
    make_machine: FleetFactory,
    sizes: Sequence[int],
    horizon: float | None = None,
    wait_deadline: float | None = None,
) -> list[dict]:
    """Simulate `workload` once per fleet size; return one tradeoff point each.

    `horizon` is the window every fleet is priced over. Pass an explicit
    shared horizon (e.g. the workload's 24h day) so cost is like-for-like
    across fleet sizes; leaving it None prices each point over its own
    makespan, which mixes billing windows.
    """
    points: list[dict] = []
    for n in sizes:
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError(f"fleet sizes must be positive ints, got {n!r}")
        fleet = [make_machine(i) for i in range(n)]
        result = Simulator(fleet).run(list(workload))
        point = tradeoff_point(
            fleet, result, wait_deadline=wait_deadline, horizon=horizon
        )
        point["n_machines"] = n
        points.append(point)
    return points


def sweep_mixes(
    workload: Sequence[Job],
    mixes: Sequence[dict[str, tuple[FleetFactory, int]]],
    horizon: float | None = None,
    wait_deadline: float | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Simulate `workload` once per fleet mix; return one tradeoff point each.

    A mix maps a group label to a (factory, count) pair, e.g.
    {"on-demand": (od_factory, 4), "spot": (spot_factory, 6)}.
    Counts may be zero (so pure fleets are expressible as a mix); every mix
    needs at least one machine, and factories in a mix must produce unique
    names (identical_fleet/spot_fleet with distinct template names do).
    `seed` goes to every Simulator so revocation draws are reproducible.
    """
    points: list[dict] = []
    for mix in mixes:
        fleet: list[Machine] = []
        counts: dict[str, int] = {}
        for label, (make, n) in mix.items():
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                raise ValueError(f"mix counts must be non-negative ints, got {n!r}")
            counts[label] = n
            fleet.extend(make(i) for i in range(n))
        if not fleet:
            raise ValueError("each mix must include at least one machine")
        result = Simulator(fleet, seed=seed).run(list(workload))
        point = tradeoff_point(
            fleet, result, wait_deadline=wait_deadline, horizon=horizon
        )
        point["mix"] = counts
        points.append(point)
    return points


def cheapest_meeting_slo(
    points: Sequence[dict], max_breach_rate: float
) -> dict | None:
    """Cheapest tradeoff point with breach_rate <= max_breach_rate, else None."""
    feasible = [p for p in points if p["breach_rate"] <= max_breach_rate]
    return min(feasible, key=lambda p: p["cost"]) if feasible else None


def find_knee(points: Sequence[dict]) -> dict | None:
    """Max-curvature knee of the cost-vs-breach curve (kneedle-style).

    Normalizes cost and breach_rate to [0, 1] and returns the point farthest
    from the chord joining the first and last points — the spot where adding
    machines stops buying much SLO. Returns None for fewer than 3 points.
    """
    if len(points) < 3:
        return None
    costs = [p["cost"] for p in points]
    breaches = [p["breach_rate"] for p in points]

    def norm(v: float, lo: float, hi: float) -> float:
        return 0.0 if hi == lo else (v - lo) / (hi - lo)

    c0, c1 = min(costs), max(costs)
    b0, b1 = min(breaches), max(breaches)
    x0, y0 = norm(costs[0], c0, c1), norm(breaches[0], b0, b1)
    x1, y1 = norm(costs[-1], c0, c1), norm(breaches[-1], b0, b1)
    dx, dy = x1 - x0, y1 - y0
    denom = math.hypot(dx, dy)

    best: dict | None = None
    best_d = -1.0
    for p, c, b in zip(points, costs, breaches):
        x, y = norm(c, c0, c1), norm(b, b0, b1)
        d = abs(dy * x - dx * y + x1 * y0 - y1 * x0) / denom if denom else 0.0
        if d > best_d:
            best, best_d = p, d
    return best
