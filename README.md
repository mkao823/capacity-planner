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

## Tradeoff curve

The decision artifact: sweep fleet sizes against one workload and plot cost vs SLO breach rate.

```bash
pip install -e ".[viz]"          # matplotlib for the plot
python examples/tradeoff_curve.py
```

This generates a seeded 24h workload (`workload.py`: sinusoidal diurnal
arrivals via Poisson thinning, mixed small/medium/large/gpu job classes),
sweeps fleets of 2–16 identical machines (`sweep.py`, one
`tradeoff_point` per size over a shared 24h horizon so cost is
like-for-like), and writes `examples/tradeoff.csv` plus
`examples/tradeoff.png`.

How to read the curve: x-axis is fleet cost over the horizon, y-axis is
SLO breach rate. The **knee** (circled, max-curvature point) is where
adding machines stops buying much SLO — the economic sweet spot. The
**star** is the cheapest fleet meeting the breach-rate SLO (5% by
default). Everything right of the knee is over-provisioning; everything
left of the star breaches the SLO. `cheapest_meeting_slo` and `find_knee`
are importable from `capacity_planner.sweep` for your own analyses.

## Planning against forecasts

The tradeoff curve above cheats: it sizes the fleet against the exact
workload it will face. Real planners only ever see history. `planner.py`
closes the loop:

```bash
python examples/plan_vs_oracle.py
```

This generates one 24h workload, then treats the last 6h as the unknown
future. Three plans compete, all evaluated against what actually happened:

- **oracle** — sizes against the actual future (the cheat no real planner gets)
- **forecast plan** — `plan_fleet` buckets the first 18h of history, picks the
  best baseline forecaster by holdout backtest (moving average vs exponential
  smoothing over a small param grid), turns the forecast into a planned
  workload, and sizes the fleet against that
- **naive** — sizes as if every future hour hits the historical peak

`evaluate_plan` runs the chosen fleet against the held-out future;
`cost_of_being_wrong` prices the error in dollars and SLO points. On the
default seed the forecast plan lands one machine under the oracle (saving
$24 but breaching 5.2% vs the 5% SLO) while the naive plan over-provisions
by 25% — the shape of the tradeoff forecast error actually creates.

Caveat, stated in the module docstring too: the planned workload is
resampled, which smooths burstiness, so picks skew slightly optimistic —
apply a safety margin in practice.

## Status

Working end to end: synthetic diurnal workloads, discrete-event
simulation, fleet-size sweeps, cost-vs-SLO tradeoff curves with
knee/SLO-pick analysis, and a forecast-driven planning loop with
oracle-vs-forecast cost-of-error accounting — 40 tests green. Next steps:
machine-type sweeps (heterogeneous fleets, spot vs on-demand mixes) and
utilization plots.
