# capacity-planner

Compute capacity planning: demand forecasting + a headroom optimizer with cost-vs-SLO tradeoff curves.

## The problem

Allocating compute is an inventory allocation problem. A datacenter (or a GPU pool serving free-tier inference traffic) is a warehouse with finite stock: machines with fixed CPU, RAM, and GPU. Jobs are orders arriving stochastically, each demanding a bundle of resources for some duration. Some jobs matter more than others — priority tiers — and some can't wait past a deadline without breaching an SLO.

This is the same shape as allocating stock to satisfy supply and demand: finite supply, uncertain demand, competing claimants, and a cost for getting it wrong. Under-provision and jobs queue, wait, and breach SLOs. Over-provision and you're burning money on idle machines. The job of a capacity planner is to sit exactly on that tradeoff curve — and to know where the curve is.

## What this does

Three stages, wired end to end:

1. **Forecast** (`forecast.py`) — turn a history of job arrivals into a forward demand curve. Starts with honest baselines (moving average, exponential smoothing); an adapter slot is stubbed for a real statistical model.
2. **Simulate** (`simulator.py`) — a discrete-event allocator over a fixed fleet. Jobs arrive, get placed on machines with free capacity, queue when the fleet is full (drained by priority, then arrival order), and are rejected when they can't ever fit or wait past their deadline. Records utilization, wait times, and rejections.
3. **Price** (`cost.py`) — turn a simulated schedule into dollars and SLO stats. Sweep fleet sizes, collect one `tradeoff_point` per configuration, and you have the cost-vs-breach-rate curve a capacity decision is actually made on.

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
print(tradeoff_point(fleet, result, wait_deadline=0.1))
```

## Status

Early scaffold: the core loop works and is tested. Next steps are a real demand generator from the forecast, fleet-size sweep tooling, and utilization plots.
