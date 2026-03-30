# Rough Pricing

A Python library for derivative pricing and calibration under stochastic and rough volatility models, with a gRPC service layer designed to plug into a larger pricing engine.

## Core focus

Rough volatility path simulation and model comparison. The library implements multiple simulation schemes for each model family — from baseline Euler discretisations to high-accuracy methods — so that convergence rates and computational trade-offs can be measured directly. See [`kernels/rough_model_sim.md`](src/roughvol/kernels/rough_model_sim.md) for the full mathematical reference.

## Models

| Model | Schemes | Calibration |
|---|---|---|
| GBM | exact log-Euler | yes |
| Heston | Euler full-truncation | yes |
| Rough Bergomi | `volterra-midpoint` · `blp-hybrid` · `exact-gaussian` | yes |
| Rough Heston | `volterra-euler` · `markovian-lift` | — |

All models implement the `PathModel` protocol and run through the unified `MonteCarloEngine`.

### Rough Bergomi simulation schemes

| Scheme | Section | Complexity | Notes |
|---|---|---|---|
| `volterra-midpoint` (default) | §2.5 | O(n²) | Midpoint Volterra quadrature; fast, mild singularity bias |
| `blp-hybrid` | §2.3 | O(n log n) | Bennedsen-Lunde-Pakkanen: near-field exact + far-field FFT; practical workhorse |
| `exact-gaussian` | §2.1 | O(n³) precomp | Cholesky of joint RL-fBM covariance; benchmark quality |

Select with `RoughBergomiModel(..., scheme="blp-hybrid")`.

### Rough Heston simulation schemes

| Scheme | Section | Complexity | Notes |
|---|---|---|---|
| `volterra-euler` (default) | §3.1 | O(n²) | Direct Volterra history accumulation; positivity by clipping |
| `markovian-lift` | §3.5 | O(N·n) | N-factor sum-of-exponentials kernel; exponential integrator for stability |

Select with `RoughHestonModel(..., scheme="markovian-lift", n_factors=8)`.

## Instruments

- European vanilla options
- Arithmetic Asian options

## Key packages

| Package | Purpose |
|---|---|
| `roughvol.models` | GBM, Heston, Rough Bergomi (3 schemes), Rough Heston (2 schemes) |
| `roughvol.kernels` | Volterra kernel weights: midpoint, BLP hybrid, exact Cholesky, Rough Heston kernel, Markovian lift fit |
| `roughvol.engines` | Monte Carlo pricing engine |
| `roughvol.sim` | Brownian motion primitives (increments, correlated BMs) |
| `roughvol.analytics` | Black-Scholes closed-form pricing, implied vol, delta |
| `roughvol.service` | Calibration, windowed calibration toolbox, gRPC server |
| `roughvol.lab` | Model comparison: vol surface fit and delta-hedge PnL |
| `roughvol.experiments` | Runnable scripts including scheme convergence study |

## Setup

```bash
python3 bootstrap/setup.py
# or
make setup
```

## Usage

```bash
make test                            # run test suite (25 tests)
make proto-python                    # regenerate gRPC stubs
make serve                           # start gRPC server

# Experiments
python -m roughvol.experiments.run_vanilla
python -m roughvol.experiments.run_asian
python -m roughvol.experiments.run_compare_gbm_heston
python -m roughvol.experiments.run_model_lab        # vol surface + delta-hedge benchmark
python -m roughvol.experiments.run_rough_vol_convergence  # scheme convergence study
```

## Convergence experiment

`run_rough_vol_convergence` runs all schemes at increasing `n_steps` and plots:

1. **rBergomi error vs n_steps** (log-log) — compares `volterra-midpoint`, `blp-hybrid`, and `exact-gaussian` against a dense reference; BLP should show a steeper convergence slope than midpoint.
2. **Wall-clock time vs n_steps** — illustrates the O(n²) vs O(n log n) scaling difference.
3. **Rough Heston price vs n_steps** — `volterra-euler` and `markovian-lift` converging to the reference.