# Rough Pricing

A self-contained Python research lab for **derivative pricing and calibration under stochastic and rough volatility models**. The project is intentionally minimal: just the simulation, analytics, calibration, and experiment scripts needed to study roughness empirically.

## Core focus

Rough volatility path simulation, live market calibration, and model comparison. The library implements multiple simulation schemes for each model family — from baseline Euler discretisations to high-accuracy methods — so that convergence rates and computational trade-offs can be measured directly. See [`kernels/rough_model_sim.md`](src/roughvol/kernels/rough_model_sim.md) for the full mathematical reference.

## Models

| Model | Schemes | Calibration |
|---|---|---|
| GBM | exact log-Euler | yes |
| Heston | Euler full-truncation | yes |
| Rough Bergomi | `volterra-midpoint` · `blp-hybrid` · `exact-gaussian` | yes |
| Rough Heston | `volterra-euler` · `markovian-lift` | — |

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

Select with `RoughHestonModel(..., scheme="markovian-lift", n_factors=8)`.

## Instruments

- European vanilla options (call and put)

## Key packages

| Package | Purpose |
|---|---|
| `roughvol.models` | GBM, Heston, Rough Bergomi (3 schemes), Rough Heston (2 schemes) |
| `roughvol.kernels` | Pure math: Volterra midpoint weights, exact Cholesky, Rough Heston kernel, Markovian lift fit |
| `roughvol.sim` | Brownian motion primitives and fBM / Volterra driver simulation |
| `roughvol.engines` | Monte Carlo pricing engine |
| `roughvol.analytics` | Black-Scholes closed-form pricing, implied vol, delta |
| `roughvol.calibration` | IV-space MC calibration (L-BFGS-B) and windowed calibration toolbox |
| `roughvol.data` | Live market data loader via yfinance (spot, rates, OTM option surface) |
| `roughvol.lab` | Model comparison: vol surface fit and delta-hedge PnL |
| `roughvol.experiments` | Runnable scripts |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas matplotlib
pip install yfinance          # required for the live-data demos
```

## Experiments

```bash
# Basic vanilla pricing check
python -m roughvol.experiments.run_vanilla

# Vol surface fit + delta-hedge PnL across models
python -m roughvol.experiments.run_model_lab

# Scheme convergence study (rBergomi weak error and wall-clock time vs n_steps)
python -m roughvol.experiments.run_rough_vol_convergence

# Live calibration demo: GBM, Heston, Rough Bergomi vs real market data
python -m roughvol.experiments.run_calibration_demo

# Empirical roughness demo: realized-vol proxy, ATM IV term structure,
# and matched rough-vs-Brownian volatility simulation from yfinance data
python -m roughvol.experiments.run_empirical_roughness_demo

# Multiple tickers are supported
python -m roughvol.experiments.run_empirical_roughness_demo SPY AAPL MSFT

# Intraday frequency is configurable and defaults to 1-minute data
python -m roughvol.experiments.run_empirical_roughness_demo SPY NVDA --interval 5m --rv-block-size 78
```

## Calibration demo

`run_calibration_demo` fetches live option chains from yfinance and calibrates three models against real equity options (SPY, AAPL, MSFT by default). It produces four figures:

| Figure | Content |
|---|---|
| `calibration_demo_iv_smile.png` | Market IV smile vs calibrated model smiles per ticker |
| `calibration_demo_rmse_bars.png` | IV RMSE bar chart (models × tickers) |
| `calibration_demo_paths.png` | Simulated spot paths: GBM vs Rough Bergomi |
| `calibration_demo_surface.png` | Vol surface heatmaps: market vs three models |

### Calibration design

- **Loss function**: IV-space MSE — `mean((σ_model − σ_market)²)` — weights all strikes equally regardless of dollar price, so OTM options carry full signal
- **Option selection**: OTM-only surface (puts below spot, calls above spot) to avoid ITM noise; 30–90 day maturity window; stratified sampling across `(maturity, strike)` for each model
- **Rough Bergomi scheme**: `blp-hybrid` during both optimisation and visualisation — O(n log n) with exact near-field kernel treatment for accurate short-maturity skew
- **Optimiser**: L-BFGS-B via `scipy.optimize.minimize`

### Empirical findings (SPY, April 2026)

| Model | IV RMSE | Calibrated params |
|---|---|---|
| GBM | ~7 vol pts | σ ≈ 0.22 |
| Heston | ~4 vol pts | κ≈2, ξ≈1.5, ρ≈−0.79 |
| Rough Bergomi | ~5 vol pts | H≈0.08, η≈2.3, ρ≈−0.98 |

The recovered Hurst exponent H ≈ 0.08 is consistent with the empirical literature (Gatheral et al. 2018 found H ≈ 0.1 for S&P 500 realised volatility).

## Empirical roughness demo

`run_empirical_roughness_demo` fetches historical prices and the current listed option chain from yfinance for one or more tickers (`SPY` by default). It then:

- builds non-overlapping realized-variance blocks from close-to-close returns at the chosen sampling interval
- de-seasonalizes intraday returns by time-of-day before constructing those blocks
- estimates the Hurst exponent from the log-log scaling of `log(realized variance)`
- extracts one near-ATM implied-vol quote per expiry to show the current option term structure
- simulates matched rough and Brownian lognormal volatility paths using the empirical H estimate

The historical-price interval is configurable with `--interval` and now defaults to `1m`. Because yfinance limits how far back intraday data can go, the script automatically picks a conservative default lookback period from the interval unless you override it with `--period`.

For intraday runs, the roughness estimator is session-aware: overnight close-to-open gaps are excluded from the return stream, the plot does not draw artificial lines across market closures, and the realized-variance series is built from non-overlapping `--rv-block-size` observation blocks rather than overlapping rolling windows.

It now saves four grouped figure files across the requested ticker set:

- `empirical_roughness_realized_vol.png`
- `empirical_roughness_roughness_regression.png`
- `empirical_roughness_atm_term_structure.png`
- `empirical_roughness_simulation.png`

Each figure contains one panel per ticker, so the outputs are split by topic rather than mixed into one per-ticker dashboard.

The realized-volatility figure now gives each ticker a full-sample block-volatility panel and a separate high-frequency local-vol zoom over the last four hours so local roughness is easier to inspect visually.

You can also ask for a cross-sectional Hurst histogram across a large-cap stock universe ranked by live market cap, for example:

```bash
python -m roughvol.experiments.run_empirical_roughness_demo --hurst-hist-top-n 50
python -m roughvol.experiments.run_empirical_roughness_demo --hurst-hist-top-n 100
```

When enabled, the script adds a histogram figure such as `empirical_roughness_hurst_histogram_top50.png` showing the distribution of estimated Hurst exponents across the top `N` names that were successfully processed.

The script also keeps a lightweight JSON cache of estimated Hurst summaries at `empirical_roughness_cache.json` by default, so repeated histogram runs can reuse prior estimates instead of recomputing every stock each time. You can override or bypass that behavior with:

```bash
python -m roughvol.experiments.run_empirical_roughness_demo --hurst-hist-top-n 100 --cache-path custom_cache.json
python -m roughvol.experiments.run_empirical_roughness_demo --hurst-hist-top-n 100 --refresh-cache
```

## Convergence experiment

`run_rough_vol_convergence` runs all rBergomi schemes at increasing `n_steps` and plots:

1. **Weak error vs n_steps** (log-log) — measured against a dense `exact-gaussian` reference. BLP converges to the same limit as exact at far fewer steps than midpoint.
2. **Wall-clock time vs n_steps** — illustrates the O(n²) vs O(n log n) vs O(n³) scaling difference.
