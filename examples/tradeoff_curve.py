#!/usr/bin/env python3
"""Cost-vs-SLO tradeoff curve: the decision artifact a capacity planner makes.

Generates a synthetic 24h workload, sweeps fleet sizes from under- to
over-provisioned, and writes:
  examples/tradeoff.csv  — one row per fleet size
  examples/tradeoff.png  — cost (x) vs breach rate (y), knee + SLO pick marked

Run from the repo root (package installed, viz extra for the plot):
    pip install -e ".[viz]"
    python examples/tradeoff_curve.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")  # headless: render to file, no display needed
import matplotlib.pyplot as plt

from capacity_planner.simulator import Machine
from capacity_planner.sweep import (
    cheapest_meeting_slo,
    find_knee,
    identical_fleet,
    sweep_fleet_sizes,
)
from capacity_planner.workload import generate_workload

# --- knobs -----------------------------------------------------------------
SEED = 7
HOURS = 24.0
MACHINE = Machine("node", cpu=16, ram=64, gpu=2, cost_per_hour=4.0)
SIZES = list(range(2, 17))  # under-provisioned -> over-provisioned
SLO_MAX_BREACH = 0.05  # the SLO we're buying: breach_rate <= 5%
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    workload = generate_workload(seed=SEED, hours=HOURS)
    print(f"workload: {len(workload)} jobs over {HOURS:.0f}h (seed={SEED})")

    points = sweep_fleet_sizes(
        workload, identical_fleet(MACHINE), SIZES, horizon=HOURS
    )

    csv_path = os.path.join(HERE, "tradeoff.csv")
    fieldnames = [
        "n_machines",
        "cost",
        "breach_rate",
        "avg_wait_admitted",
        "p95_wait_admitted",
        "makespan",
        "horizon",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)
    print(f"wrote {csv_path}")

    print(f"\n{'n':>3} {'cost':>8} {'breach':>8}")
    for p in points:
        print(f"{p['n_machines']:>3} {p['cost']:>8.1f} {p['breach_rate']:>8.3f}")

    knee = find_knee(points)
    pick = cheapest_meeting_slo(points, SLO_MAX_BREACH)
    if knee:
        print(
            f"\nknee of the curve: {knee['n_machines']} machines "
            f"(cost {knee['cost']:.0f}, breach {knee['breach_rate']:.3f})"
        )
    if pick:
        print(
            f"cheapest fleet meeting {SLO_MAX_BREACH:.0%} breach SLO: "
            f"{pick['n_machines']} machines (cost {pick['cost']:.0f}, "
            f"breach {pick['breach_rate']:.3f})"
        )
    else:
        print(f"\nno fleet size meets the {SLO_MAX_BREACH:.0%} breach SLO — widen SIZES")

    costs = [p["cost"] for p in points]
    breaches = [p["breach_rate"] for p in points]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(costs, breaches, "o-", color="#4c78a8", label="fleet sizes")
    for p in points[::3]:  # label every third point to avoid clutter
        ax.annotate(
            f"{p['n_machines']}", (p["cost"], p["breach_rate"]),
            textcoords="offset points", xytext=(4, 6), fontsize=8, color="#555",
        )
    ax.axhline(SLO_MAX_BREACH, color="#e45756", linestyle="--", linewidth=1,
               label=f"SLO breach <= {SLO_MAX_BREACH:.0%}")
    if knee:
        ax.scatter([knee["cost"]], [knee["breach_rate"]], s=120,
                   facecolor="none", edgecolor="#f58518", linewidth=2,
                   label=f"knee: {knee['n_machines']} machines", zorder=5)
    if pick:
        ax.scatter([pick["cost"]], [pick["breach_rate"]], s=120, marker="*",
                   color="#54a24b", label=f"SLO pick: {pick['n_machines']} machines",
                   zorder=5)
    ax.set_xlabel(f"fleet cost over {HOURS:.0f}h ($)")
    ax.set_ylabel("SLO breach rate")
    ax.set_title("Cost vs SLO: how many machines do we need?")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    png_path = os.path.join(HERE, "tradeoff.png")
    fig.savefig(png_path, dpi=120)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
