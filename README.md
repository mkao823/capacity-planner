# capacity-planner

Compute capacity planning: demand forecasting + a headroom optimizer with cost-vs-SLO tradeoff curves.

## The problem

Allocating compute is an inventory allocation problem. A datacenter (or a GPU pool serving free-tier inference traffic) is a warehouse with finite stock: machines with fixed CPU, RAM, and GPU. Jobs are orders arriving stochastically, each demanding a bundle of resources for some duration. Some jobs matter more than others — priority tiers — and some can't wait past a deadline without breaching an SLO.

This is the same shape as allocating stock to satisfy supply and demand: finite supply, uncertain demand, competing claimants, and a cost for getting it wrong. Under-provision and jobs queue, wait, and breach SLOs. Over-provision and you're burning money on idle machines. The job of a capacity planner is to sit exactly on that tradeoff curve — and to know where the curve is.

## What this does

Three stages, wired end to end:

1. **Forecast** (`forecast.py`) — turn a history of job arrivals into a forward demand curve. Starts with honest baselines (moving average, exponential smoothing); an adapter slot is stubbed for a real statistical model. `bucket_arrivals` validates inputs (bucket size first, even for empty input) and accepts an optional `end_time` so trailing zero-demand buckets are representable.
2. **Simulate** (`simulator.py`) — a discrete-event allocator over a fixed fleet. Every arrival joins a single queue drained by one allocator (immediate placement is just the empty-queue case — arrivals never jump the queue). Each timestamp runs in phases: completions release capacity, then arrivals enqueue, then the queue drains once, then deadline expiries fire. The drain is strict priority order — `(priority, arrival)` with head-of-line blocking, no backfilling — so a smaller job never sneaks past a blocked higher-priority one. Jobs that can't ever fit are rejected `hopeless` at arrival; queued jobs get an explicit expiry event and are rejected `deadline` at the deadline itself; anything left when the event stream ends is `stranded-at-end`. `makespan = t_end − t_start`, so pre-workload idle time isn't charged to the fleet.
3. **Price** (`cost.py`) — turn a simulated schedule into dollars and SLO stats. Sweep fleet sizes, collect one `tradeoff_point` per configuration, and you have the cost-vs-breach-rate curve a capacity decision is actually made on. SLOs are measured per job against its own `wait_deadline` by default (pass a global `wait_deadline` to override); rejected jobs always count as breaches, and `avg/p95_wait_admitted` are admitted-job latency only. p95 uses a nearest-rank percentile that stays within the observed range on small samples.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
pip install -e .
python -m pytest
```

```python
from capacity_planner.simulator import Job, Machine, Simulator
from capacity_planner.cost import tradeoff_point

fleet = [Machine("gpu-node", cpu=16, ram=64, gpu=2, cost_per_hour=4.0)]
jobs = [Job("infer-1", arrival=0.0, cpu=4, ram=16, gpu=1, duration=0.5, wait_deadline=0.1)]
result = Simulator(fleet).run(jobs)
print(tradeoff_point(fleet, result))  # per-job deadlines; pass wait_deadline=... to override
print(tradeoff_point(fleet, result, horizon=24.0))  # shared horizon for like-for-like fleet comparisons
```

## Status

Early scaffold: the core loop works and is tested. Next steps are a real demand generator from the forecast, fleet-size sweep tooling, and utilization plots.
