# Rough Pricing

A self-contained Python research lab for **derivative pricing and calibration under stochastic and rough volatility models**. The project is intentionally experimental: just the simulation, analytics, calibration, and experiment scripts needed to study roughness empirically.

## Core focus

Rough volatility path simulation, live market calibration, and model comparison. The library implements multiple simulation schemes for each model family — from baseline Euler discretisations to high-accuracy methods — so that convergence rates and computational trade-offs can be measured directly. See [`kernels/rough_model_sim.md`](src/roughvol/kernels/rough_model_sim.md) for the full mathematical reference.

## Models

| Model | Schemes | Calibration |
|---|---|---|
| GBM | exact log-Euler | yes |
| Heston | Euler full-truncation | yes |
| Rough Bergomi | `volterra-midpoint` · `blp-hybrid` · `exact-gaussian` | yes |
| Rough Heston | `volterra-euler` · `markovian-lift` · `bayer-breneis` | yes |

All models implement the `PathModel` protocol and run through the unified `MonteCarloEngine`.

### Rough Bergomi simulation schemes

| Scheme | Complexity | Notes |
|---|---|---|
| `volterra-midpoint` | O(n²) | Midpoint Volterra quadrature; fast, mild singularity bias |
| `blp-hybrid` | O(n log n) | Bennedsen-Lunde-Pakkanen: near-field exact + far-field FFT; default for calibration |
| `exact-gaussian` | O(n³) precomp | Cholesky of joint RL-fBM covariance; benchmark quality |

Select with `RoughBergomiModel(..., scheme="blp-hybrid")`.

### Rough Heston simulation schemes

| Scheme | Complexity | Notes |
|---|---|---|
| `volterra-euler` | O(n²) | Direct Volterra history accumulation; positivity by clipping |
| `markovian-lift` | O(N·n) | N-factor sum-of-exponentials kernel; exponential integrator for stability |
| `bayer-breneis` | O(N·n) | Order-2 weak scheme: 3-point Gauss-Hermite innovation for dW₂ + Strang splitting for log-spot; arXiv:2310.04146 |

Select with `RoughHestonModel(..., scheme="bayer-breneis", n_factors=8)`.

The `bayer-breneis` scheme replaces the Gaussian variance driver with z ∈ {−√3, 0, +√3}, P = {1/6, 2/3, 1/6} (matching 5 moments of N(0,1)) and uses Strang splitting for log-spot to symmetrise the Itô correction across V_i and V_{i+1}.

## Instruments

- European vanilla options (call and put)

## Key packages

| Package | Purpose |
|---|---|
| `roughvol.models` | GBM, Heston, Rough Bergomi (3 schemes), Rough Heston (3 schemes) |
| `roughvol.kernels` | Pure math: Volterra midpoint weights, exact Cholesky, Rough Heston kernel, Markovian lift fit |
| `roughvol.sim` | Brownian motion primitives and fBM / Volterra driver simulation |
| `roughvol.engines` | Monte Carlo pricing engine |
| `roughvol.analytics` | Black-Scholes closed-form pricing, implied vol, delta |
| `roughvol.calibration` | IV-space MC calibration (L-BFGS-B) and windowed calibration toolbox |
| `roughvol.data` | Live market data loader via yfinance (spot, rates, OTM option surface) |
| `roughvol.experiments` | Runnable scripts (calibration, convergence, model comparison, rough estimate) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas matplotlib
pip install yfinance          # required for the live-data demos
```

## Experiments

The runnable entry points are now organized by purpose under `roughvol.experiments.<purpose>`.

Plotting scripts follow two rules:

- each script produces exactly one figure
- each figure is written into `output/<purpose>/...`

Workflow ownership is split in two:

- each purpose folder owns the simulation / calibration / data-building logic
- `roughvol.experiments.ensemble` owns the multi-plot pipelines

The older top-level scripts in `roughvol.experiments` remain as compatibility entry points.

### Basics

`roughvol.experiments.basics`

```bash
python -m roughvol.experiments.basics.run_vanilla
```

This is a console-only sanity check for vanilla pricing, so it does not write a figure.

### Model comparison

`roughvol.experiments.model_comparison`

```bash
python -m roughvol.experiments.model_comparison.run_model_lab
```

This is also console-first: it prints surface-fit and hedge-PnL diagnostics for the model lab.

### Calibration

`roughvol.experiments.calibration`

The calibration workflow now lives in `roughvol.experiments.calibration.run_calibration_demo`.
It fetches live data, calibrates the models, and returns structured per-ticker results for the one-plot scripts below.

```bash
python -m roughvol.experiments.calibration.run_calibration_demo
python -m roughvol.experiments.calibration.plot_iv_smile
python -m roughvol.experiments.calibration.plot_rmse_bars
python -m roughvol.experiments.calibration.plot_simulated_paths
python -m roughvol.experiments.calibration.plot_surface
python -m roughvol.experiments.ensemble.run_calibration_pipeline
```

`run_calibration_demo` is the workflow/data script.
Each `plot_*` script writes one figure into `output/calibration/`.
`ensemble.run_calibration_pipeline` runs the workflow once and renders the full calibration figure set.

Outputs:

- `output/calibration/calibration_demo_iv_smile.png`
- `output/calibration/calibration_demo_rmse_bars.png`
- `output/calibration/calibration_demo_paths.png`
- `output/calibration/calibration_demo_surface.png`

### Convergence

`roughvol.experiments.convergence`

The convergence workflow lives in `roughvol.experiments.convergence.run_rough_vol`.
The one-plot scripts render one figure each, and the ensemble pipeline renders the standard pair.

```bash
python -m roughvol.experiments.convergence.run_rough_vol
python -m roughvol.experiments.convergence.plot_error
python -m roughvol.experiments.convergence.plot_timing
python -m roughvol.experiments.convergence.plot_efficiency_rh
python -m roughvol.experiments.ensemble.run_convergence_pipeline
```

Outputs:

- `output/convergence/rough_vol_error.png`
- `output/convergence/rough_vol_timing.png`
- `output/convergence/rough_heston_efficiency.png`

#### Scheme efficiency: accuracy vs wall-clock time

![Rough Heston efficiency](output/convergence/rough_heston_efficiency.png)

Each panel fixes the Hurst exponent H and plots wall-clock time (log scale) against absolute
pricing error relative to the characteristic-function benchmark, for N = 32 factors and a
Heston control variate. Points toward the **lower right** are better — lower error achieved
in less time. Each curve is one simulation scheme; the labels show the number of time steps
at each point.

**H = 0.1** (mildly rough): Volterra Euler dominates. Its exact kernel accumulation incurs no
Markovian approximation error, so it reaches sub-0.06 error at n = 256 steps in ~15 s — below
the factor floor that limits the Markovian-lift schemes even at n = 1024.

**H = 0.01** (extremely rough): the picture inverts. The kernel K(t) ~ t^{-0.49} is nearly
maximally singular; Volterra Euler stalls above 0.3 error at any tested step count because the
O(n²) cost makes a fine enough grid prohibitive. The Markovian lift and Bayer-Breneis schemes,
whose N = 32 exponential basis captures the singularity, converge monotonically and reach
comparable error 3× faster.

**H = 0.05** (transition regime): both families are competitive. This is the roughness range
most commonly calibrated to short-dated equity options (7–30 day expiries), where the choice
of scheme is a genuine engineering decision.

### Rough estimate

`roughvol.experiments.rough_estimate`

The roughness workflow lives in `roughvol.experiments.rough_estimate.run_empirical_roughness_demo`.
It builds per-ticker reports from yfinance-based spot and option data. Intraday history defaults to `1m`; `--period` is chosen conservatively from the interval unless you override it; and intraday roughness estimation is session-aware, excluding overnight gaps and using non-overlapping realized-variance blocks.

Single-ticker and multi-ticker views:

```bash
python -m roughvol.experiments.rough_estimate.run_empirical_roughness_demo SPY AAPL MSFT
python -m roughvol.experiments.rough_estimate.plot_realized_vol SPY AAPL MSFT
python -m roughvol.experiments.rough_estimate.plot_roughness_regression SPY AAPL MSFT
python -m roughvol.experiments.rough_estimate.plot_atm_term_structure SPY AAPL MSFT
python -m roughvol.experiments.rough_estimate.plot_simulation SPY AAPL MSFT
python -m roughvol.experiments.rough_estimate.plot_recent_window_triptych SPY AAPL
```

Cross-sectional views:

```bash
python -m roughvol.experiments.rough_estimate.plot_scaling_law --top-n 50
python -m roughvol.experiments.rough_estimate.plot_cross_section_summary --top-n 50
python -m roughvol.experiments.rough_estimate.plot_hurst_histogram --top-n 50
python -m roughvol.experiments.rough_estimate.plot_hurst_rankings --top-n 50
python -m roughvol.experiments.rough_estimate.plot_hurst_sector --top-n 100
python -m roughvol.experiments.ensemble.run_rough_estimate_pipeline SPY AAPL --hurst-hist-top-n 50
```

Representative outputs in `output/rough_estimate/`:

- `empirical_roughness_realized_vol.png`
- `empirical_roughness_roughness_regression.png`
- `empirical_roughness_atm_term_structure.png`
- `empirical_roughness_simulation.png`
- `empirical_roughness_recent_window_triptych.png`
- `empirical_roughness_scaling_law.png`
- `empirical_roughness_cross_section_summary.png`
- `empirical_roughness_hurst_histogram_top50.png`
- `empirical_roughness_hurst_rankings_top50.png`
- `empirical_roughness_hurst_by_sector_top100.png`
- `empirical_roughness_cache.json`

The empirical roughness pipeline:

- de-seasonalizes intraday returns by time of day
- builds non-overlapping realized-variance blocks
- estimates `H` from the scaling law of `log(RV)`
- uses the current option chain for ATM term-structure context
- can reuse cached cross-sectional H estimates via `--cache-path` and `--refresh-cache`

`run_empirical_roughness_demo` is the workflow/data script.
Each `plot_*` script writes one figure into `output/rough_estimate/`.
`ensemble.run_rough_estimate_pipeline` runs the workflow once and renders the default empirical roughness figure set.
