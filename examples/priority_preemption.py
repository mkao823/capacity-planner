#!/usr/bin/env python3
"""Priority preemption: protecting the VIP class.

A low-priority bulk class saturates the fleet; a few high-priority (VIP)
jobs arrive mid-congestion with tight deadlines. The same fleet runs twice
— preemption off vs on — and we compare breach rate per priority class:

  examples/priority_preemption.png — grouped bars of breach rate by class,
      off vs on. Expectation: VIP breach collapses to ~0% with preemption
      on, the bulk class pays a small price, and fleet cost is unchanged
      (same machines, same horizon).

Run from the repo root (package installed, viz extra for the plot):
    pip install -e ".[viz]"
    python examples/priority_preemption.py
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless: render to file, no display needed
import matplotlib.pyplot as plt

from capacity_planner.cost import tradeoff_point
from capacity_planner.simulator import Job, Machine, Simulator

# --- knobs -----------------------------------------------------------------
N_MACHINES = 3
NODE_CPU, NODE_RAM, NODE_COST = 4.0, 16.0, 1.0  # $/machine-hour
BULK_N = 18  # low-priority jobs, priority=1, arrive t=0..2.55
VIP_N = 4  # high-priority jobs, priority=0, arrive mid-congestion at t=3
VIP_ARRIVAL = 3.0
HORIZON = 24.0  # shared pricing window so cost is like-for-like
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))


def build_workload() -> list[Job]:
    jobs = [
        Job(
            name=f"bulk-{i}",
            arrival=i * 0.15,
            cpu=2.0,
            ram=4.0,
            duration=6.0,
            wait_deadline=6.0,
            priority=1,
        )
        for i in range(BULK_N)
    ]
    jobs += [
        Job(
            name=f"vip-{i}",
            arrival=VIP_ARRIVAL + i * 0.01,
            cpu=2.0,
            ram=4.0,
            duration=1.0,
            wait_deadline=2.0,
            priority=0,
        )
        for i in range(VIP_N)
    ]
    return jobs


def breach_by_priority(result) -> dict[int, tuple[int, int]]:
    """priority -> (rejected, total). The schedule holds one entry per
    admitted job, so no deduping is needed."""
    totals: dict[int, int] = {}
    rejected: dict[int, int] = {}
    for job, _m, _t in result.scheduled:
        totals[job.priority] = totals.get(job.priority, 0) + 1
    for r in result.rejections:
        totals[r.job.priority] = totals.get(r.job.priority, 0) + 1
        rejected[r.job.priority] = rejected.get(r.job.priority, 0) + 1
    return {p: (rejected.get(p, 0), totals[p]) for p in sorted(totals)}


def main() -> None:
    fleet = [
        Machine(f"node-{i}", cpu=NODE_CPU, ram=NODE_RAM, cost_per_hour=NODE_COST)
        for i in range(N_MACHINES)
    ]
    runs = {}
    for label, enabled in (("off", False), ("on", True)):
        result = Simulator(fleet, preemption_enabled=enabled).run(
            build_workload()
        )
        point = tradeoff_point(fleet, result, horizon=HORIZON)
        runs[label] = (result, point)
        by_prio = breach_by_priority(result)
        # avg wait of first admission per class
        waits: dict[int, list[float]] = {}
        for job, _m, t in result.scheduled:
            waits.setdefault(job.priority, []).append(t - job.arrival)
        print(f"--- preemption {label} ---")
        print(f"  preemptions performed: {result.n_preemptions}")
        for p, (rej, total) in by_prio.items():
            name = "VIP " if p == 0 else "bulk"
            w = waits.get(p, [])
            avg = f", avg wait {sum(w) / len(w):.2f}h" if w else ", none admitted"
            print(f"  {name}(prio {p}): {rej}/{total} breached "
                  f"({rej / total:.1%}{avg})")
        print(f"  fleet cost over {HORIZON:.0f}h: ${point['cost']:.0f}")

    labels = ["VIP (prio 0)", "bulk (prio 1)"]
    off_rates = [breach_by_priority(runs["off"][0])[p][0] / breach_by_priority(runs["off"][0])[p][1] for p in (0, 1)]
    on_rates = [breach_by_priority(runs["on"][0])[p][0] / breach_by_priority(runs["on"][0])[p][1] for p in (0, 1)]

    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar([i - w / 2 for i in x], off_rates, w, label="preemption off")
    b2 = ax.bar([i + w / 2 for i in x], on_rates, w, label="preemption on")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("SLO breach rate")
    ax.set_title("Priority preemption: who pays when the fleet is full?")
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    out = os.path.join(HERE, "priority_preemption.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
