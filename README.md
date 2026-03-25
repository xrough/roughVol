# Rough Pricing

A Python library for derivative pricing and calibration under stochastic and rough volatility models, with a gRPC service layer designed to plug into a larger pricing engine.

## Core focus

Rough volatility path simulation and model comparison. The central piece is the **Rough Bergomi** model — implemented end-to-end from the fractional Brownian kernel through Monte Carlo pricing, calibration, and volatility-surface benchmarking against GBM and Heston.

## Models

| Model | Path simulation | Calibration |
|---|---|---|
| GBM | yes | yes |
| Heston | yes | yes |
| Rough Bergomi | yes | yes |

All models share the same `PathModel` protocol and run through a unified Monte Carlo engine.

## Instruments

- European vanilla options
- Arithmetic Asian options

## Key packages

| Package | Purpose |
|---|---|
| `roughvol.models` | GBM, Heston, and Rough Bergomi path simulators |
| `roughvol.kernels` | Fractional Brownian motion kernel for rBergomi |
| `roughvol.engines` | Monte Carlo pricing engine |
| `roughvol.analytics` | Black-Scholes closed-form pricing, implied vol, delta |
| `roughvol.service` | Calibration, windowed calibration toolbox, gRPC server |
| `roughvol.lab` | Model comparison: vol surface fit and delta-hedge PnL |

## Setup

```bash
python3 bootstrap/setup.py
# or
make setup
```

## Usage

```bash
make test                            # run test suite
make proto-python                    # regenerate gRPC stubs
make serve                           # start gRPC server

python -m roughvol.experiments.run_model_lab        # benchmark models
python -m roughvol.experiments.run_vanilla
python -m roughvol.experiments.run_asian
python -m roughvol.experiments.run_compare_gbm_heston
```
