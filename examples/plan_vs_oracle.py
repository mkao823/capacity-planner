#!/usr/bin/env python3
"""Forecast plan vs oracle plan: what does being wrong cost?

Generates one 24h workload, then pretends the last 6h are the unknown future:
  - oracle:  sizes the fleet against the actual future workload (the cheat)
  - forecast: sizes the fleet from a forecast fit on the first 18h of history
  - naive:   sizes the fleet as if every future hour hits the historical peak

Each plan is evaluated against the actual future. Writes:
  examples/plan_vs_oracle.png — bar chart of realized 6h cost per plan

Run from the repo root (package installed, viz extra for the plot):
    pip install -e ".[viz]"
    python examples/plan_vs_oracle.py
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless: render to file, no display needed
import matplotlib.pyplot as plt

from capacity_planner.forecast import bucket_arrivals
from capacity_planner.planner import (
    cost_of_being_wrong,
    evaluate_plan,
    plan_fleet,
    scale_workload_to_counts,
)
from capacity_planner.simulator import Machine
from capacity_planner.sweep import (
    cheapest_meeting_slo,
    identical_fleet,
    sweep_fleet_sizes,
)
from capacity_planner.workload import generate_workload

# --- knobs -----------------------------------------------------------------
SEED = 7
TOTAL_HOURS = 24.0
HISTORY_HOURS = 18.0
FUTURE_HOURS = TOTAL_HOURS - HISTORY_HOURS
MACHINE = Machine("node", cpu=16, ram=64, gpu=2, cost_per_hour=4.0)
SIZES = list(range(2, 13))  # under-provisioned -> over-provisioned
SLO_MAX_BREACH = 0.05
BUCKET = 1.0  # hourly demand buckets
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    workload = generate_workload(seed=SEED, hours=TOTAL_HOURS)
    history = [j.arrival for j in workload if j.arrival < HISTORY_HOURS]
    actual = [j for j in workload if j.arrival >= HISTORY_HOURS]
    print(
        f"workload: {len(workload)} jobs / {TOTAL_HOURS:.0f}h "
        f"(history {len(history)} arrivals in {HISTORY_HOURS:.0f}h, "
        f"future {len(actual)} jobs in {FUTURE_HOURS:.0f}h)"
    )
    factory = identical_fleet(MACHINE)

    # Oracle: plan against the actual future (the cheat no real planner gets).
    oracle_points = sweep_fleet_sizes(
        actual, factory, SIZES, horizon=FUTURE_HOURS
    )
    oracle = cheapest_meeting_slo(oracle_points, SLO_MAX_BREACH)

    # Forecast plan: history -> forecast -> planned workload -> fleet pick,
    # then evaluated against what actually happened.
    plan = plan_fleet(
        history, factory, SIZES, FUTURE_HOURS, SLO_MAX_BREACH,
        bucket_size=BUCKET, seed=SEED,
    )
    print(
        f"forecaster: {plan['forecaster']}{plan['forecaster_params']} "
        f"(backtest MAE {plan['backtest_mae']:.2f} jobs/bucket)"
    )
    realized = evaluate_plan(
        plan["n_machines"], factory, actual, FUTURE_HOURS
    )

    # Naive plan: provision as if every future hour hits the historical peak.
    peak = max(bucket_arrivals(history, BUCKET, end_time=HISTORY_HOURS))
    n_future_buckets = max(1, int(FUTURE_HOURS // BUCKET))
    base = generate_workload(seed=SEED + 99, hours=FUTURE_HOURS,
                             base_rate=2 * peak, amplitude=peak)
    naive_workload = scale_workload_to_counts(
        base, [float(peak)] * n_future_buckets, BUCKET, seed=SEED + 100
    )
    naive_points = sweep_fleet_sizes(
        naive_workload, factory, SIZES, horizon=FUTURE_HOURS
    )
    naive_pick = cheapest_meeting_slo(naive_points, SLO_MAX_BREACH)
    naive = evaluate_plan(naive_pick["n_machines"], factory, actual, FUTURE_HOURS)

    rows = [
        ("oracle (cheat)", oracle),
        ("forecast plan", realized),
        ("naive (peak-always)", naive),
    ]
    print(f"\n{'plan':<20} {'fleet':>5} {'cost/6h':>8} {'breach':>7}")
    for name, pt in rows:
        print(
            f"{name:<20} {pt['n_machines']:>5} "
            f"${pt['cost']:>7.0f} {pt['breach_rate']:>6.1%}"
        )

    err = cost_of_being_wrong(oracle, realized)
    print(
        f"\ncost of being wrong vs oracle: "
        f"${err['extra_cost']:+.0f} ({err['extra_cost_pct']:+.1f}%), "
        f"breach delta {err['breach_delta']:+.1%}"
    )

    # Plot: realized cost per plan, breach rate annotated.
    names = [n for n, _ in rows]
    costs = [pt["cost"] for _, pt in rows]
    breaches = [pt["breach_rate"] for _, pt in rows]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, costs, color=colors)
    ax.set_ylabel("realized fleet cost over 6h ($)")
    ax.set_title("Planning against forecasts: what being wrong costs")
    for bar, b in zip(bars, breaches):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(costs) * 0.02,
            f"breach {b:.1%}",
            ha="center",
            fontsize=10,
        )
    fig.tight_layout()
    out = os.path.join(HERE, "plan_vs_oracle.png")
    fig.savefig(out, dpi=100)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
