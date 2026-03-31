# Rough Pricing

A self-contained Python research lab for **derivative pricing and calibration under stochastic and rough volatility models**. The project is intentionally minimal: just the simulation, analytics, calibration, and experiment scripts needed to study roughness empirically.

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

- European vanilla options (call and put)

## Key packages

| Package | Purpose |
|---|---|
| `roughvol.models` | GBM, Heston, Rough Bergomi (3 schemes), Rough Heston (2 schemes) |
| `roughvol.kernels` | Pure math: Volterra midpoint weights, exact Cholesky, Rough Heston kernel, Markovian lift fit |
| `roughvol.sim` | Brownian motion primitives (`brownian.py`) and fBM / Volterra driver simulation (`volterra.py`) |
| `roughvol.engines` | Monte Carlo pricing engine |
| `roughvol.analytics` | Black-Scholes closed-form pricing, implied vol, delta |
| `roughvol.calibration` | Calibration routines and windowed calibration toolbox |
| `roughvol.lab` | Model comparison: vol surface fit and delta-hedge PnL |
| `roughvol.experiments` | Runnable scripts including scheme convergence study |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # or: pip install numpy scipy pandas matplotlib
```

## Usage

```bash
# Experiments
python -m roughvol.experiments.run_vanilla             # basic vanilla pricing check
python -m roughvol.experiments.run_model_lab           # vol surface fit + delta-hedge PnL
python -m roughvol.experiments.run_rough_vol_convergence  # scheme convergence study
```

## Convergence experiment

`run_rough_vol_convergence` runs all rBergomi schemes at increasing `n_steps` and plots:

1. **rBergomi weak error vs n_steps** (log-log) — `volterra-midpoint`, `blp-hybrid`, and `exact-gaussian` measured against a dense `exact-gaussian` reference at n=32. BLP converges to the same limit as exact at far fewer steps than midpoint.
2. **Wall-clock time vs n_steps** — illustrates the O(n²) vs O(n log n) vs O(n³) scaling difference.

The reference is `exact-gaussian` at n=32 (already converged, unlike midpoint at n=256 which still carries a large discretisation bias at small H).
