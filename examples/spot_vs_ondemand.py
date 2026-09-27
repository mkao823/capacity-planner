#!/usr/bin/env python3
"""Spot vs on-demand: where the real cost wins live.

Fixed synthetic 24h workload; sweeps a grid of (on-demand, spot) mixes and
writes:
  examples/spot_vs_ondemand.png — heatmap of 24h fleet cost, each cell
      annotated with its breach rate; the cheapest mix meeting the SLO is
      starred. Read it as: how much on-demand base do you need, and how
      much cheap-but-revocable spot can you layer on top?

Run from the repo root (package installed, viz extra for the plot):
    pip install -e ".[viz]"
    python examples/spot_vs_ondemand.py
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless: render to file, no display needed
import matplotlib.pyplot as plt

from capacity_planner.simulator import Machine
from capacity_planner.sweep import (
    cheapest_meeting_slo,
    identical_fleet,
    spot_fleet,
    sweep_mixes,
)
from capacity_planner.workload import generate_workload

# --- knobs -----------------------------------------------------------------
SEED = 7
HOURS = 24.0
NODE = Machine("node", cpu=16, ram=64, gpu=2, cost_per_hour=4.0)
SPOT_DISCOUNT = 0.3  # spot costs 30% of on-demand per hour
REVOCATION_RATE = 0.2  # per spot machine per hour (~1 revoke / 5h)
OD_COUNTS = list(range(2, 9))  # on-demand machines
SPOT_COUNTS = list(range(0, 13))  # spot machines
SLO_MAX_BREACH = 0.05  # the SLO we're buying: breach_rate <= 5%
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    workload = generate_workload(seed=SEED, hours=HOURS)
    print(f"workload: {len(workload)} jobs over {HOURS:.0f}h (seed={SEED})")

    od_factory = identical_fleet(NODE)
    spot_factory = spot_fleet(
        NODE, discount=SPOT_DISCOUNT, revocation_rate=REVOCATION_RATE
    )
    mixes = [
        {"on-demand": (od_factory, n_od), "spot": (spot_factory, n_spot)}
        for n_od in OD_COUNTS
        for n_spot in SPOT_COUNTS
    ]
    points = sweep_mixes(workload, mixes, horizon=HOURS, seed=SEED)
    print(f"swept {len(points)} mixes")

    by_mix = {(p["mix"]["on-demand"], p["mix"]["spot"]): p for p in points}
    cost_grid = [
        [by_mix[(od, sp)]["cost"] for sp in SPOT_COUNTS] for od in OD_COUNTS
    ]
    breach_grid = [
        [by_mix[(od, sp)]["breach_rate"] for sp in SPOT_COUNTS]
        for od in OD_COUNTS
    ]

    pick = cheapest_meeting_slo(points, SLO_MAX_BREACH)
    od_only = [p for p in points if p["mix"]["spot"] == 0]
    od_pick = cheapest_meeting_slo(od_only, SLO_MAX_BREACH)

    print(f"\ncheapest mix meeting {SLO_MAX_BREACH:.0%} breach SLO:")
    if pick:
        m = pick["mix"]
        print(
            f"  {m['on-demand']} on-demand + {m['spot']} spot "
            f"(cost ${pick['cost']:.0f}, breach {pick['breach_rate']:.1%})"
        )
    if od_pick:
        m = od_pick["mix"]
        print(
            f"cheapest all-on-demand meeting SLO: {m['on-demand']} machines "
            f"(cost ${od_pick['cost']:.0f}, breach {od_pick['breach_rate']:.1%})"
        )
    if pick and od_pick:
        saved = od_pick["cost"] - pick["cost"]
        print(f"spot saves ${saved:.0f} "
              f"({saved / od_pick['cost']:.0%} of the on-demand bill)")

    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(cost_grid, origin="lower", aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(SPOT_COUNTS)), SPOT_COUNTS)
    ax.set_yticks(range(len(OD_COUNTS)), OD_COUNTS)
    ax.set_xlabel("spot machines (30% price, ~1 revoke / 5h each)")
    ax.set_ylabel("on-demand machines")
    ax.set_title("24h fleet cost by mix — cell labels are SLO breach rate")
    for i, od in enumerate(OD_COUNTS):
        for j, sp in enumerate(SPOT_COUNTS):
            ax.text(
                j,
                i,
                f"{breach_grid[i][j]:.0%}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if cost_grid[i][j] < 900 else "white",
            )
    if pick:
        m = pick["mix"]
        ax.plot(
            SPOT_COUNTS.index(m["spot"]),
            OD_COUNTS.index(m["on-demand"]),
            marker="*",
            markersize=20,
            markeredgecolor="black",
            markerfacecolor="gold",
            linestyle="",
            label="cheapest mix meeting SLO",
        )
        ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="fleet cost over 24h ($)")
    fig.tight_layout()
    png_path = os.path.join(HERE, "spot_vs_ondemand.png")
    fig.savefig(png_path, dpi=110)
    print(f"\nwrote {png_path}")


if __name__ == "__main__":
    main()
