# Stochastic and Rough Volatility Lab

Python framework for derivative pricing, calibration, and model comparison under classical and stochastic-volatility models, with a gRPC/proto boundary intended to plug into a larger pricing engine.

## Current Status

The project is no longer just a minimal Monte Carlo sandbox. It currently includes:

- Monte Carlo pricing for vanilla and Asian options
- Black-Scholes analytics for pricing, implied volatility, and delta
- Path models for `GBM`, `Heston`, and `Rough Bergomi`
- Calibration utilities for `BS`, `GBM_MC`, `HESTON`, and `ROUGH_BERGOMI`
- A gRPC service with proto definitions for pricing, implied vol, and calibration
- A windowed calibration toolbox for time-bucketed parameter updates
- A research lab for comparing models by volatility-surface fit and delta-hedging PnL

What is not implemented yet:

- A true rough-volatility path model in the production model registry
- C++ integration code on the engine side
- Persistent calibration state outside the Python service process

## Repository Layout

```text
.
├── Makefile
├── README.md
├── bootstrap/
│   └── setup.py
├── generated/
│   └── python/
├── notebooks/
├── proto/
│   └── rough_pricing.proto
├── src/
│   └── roughvol/
│       ├── analytics/      # Black-Scholes pricing, IV, delta
│       ├── data/           # market-data schema/provider scaffolding
│       ├── engines/        # Monte Carlo engine
│       ├── experiments/    # runnable research scripts
│       ├── instruments/    # vanilla and Asian contracts
│       ├── lab/            # model comparison and hedge-PnL lab
│       ├── models/         # GBM and Heston path models
│       ├── service/        # calibration + gRPC service layer
│       ├── sim/            # Brownian drivers
│       └── types.py        # shared contracts and containers
└── tests/
```

## Setup

Clone the repo and create the environment:

```bash
git clone https://github.com/jixh-KPZ-1020/Rough-Pricing.git
cd Rough-Pricing
python3 bootstrap/setup.py
```

Or use the `Makefile`:

```bash
make setup
```

## Development Commands

```bash
make test
make lint
make proto-python
make serve
```

`make proto-python` regenerates the Python stubs from [`proto/rough_pricing.proto`](proto/rough_pricing.proto).

## Implemented Components

### Core Types

[`src/roughvol/types.py`](src/roughvol/types.py) defines the project-wide contracts:

- `MarketData`
- `SimConfig`
- `PathBundle`
- `PriceResult`
- `Instrument` and `PathModel` protocols

This layer is the interface boundary between models, pricing engines, and instruments.

### Models

Implemented models:

- `GBM_Model`
- `HestonModel`
- `RoughBergomiModel`

Both expose `simulate_paths(...) -> PathBundle` and run through the common Monte Carlo engine.

### Instruments

Implemented instruments:

- European vanilla options
- Arithmetic Asian options

The path container supports interpolation through `spot_at(...)`, which is used to handle observation dates that do not align exactly with the simulation grid.

### Calibration

[`src/roughvol/service/calibration.py`](src/roughvol/service/calibration.py) provides:

- `BSCalibrator`
- `MCCalibrator`
- factory helpers for `GBM_MC` and `HESTON`

[`src/roughvol/service/toolbox.py`](src/roughvol/service/toolbox.py) adds a windowed calibration layer intended for proto-connected engine updates:

- fixed calibration window selection
- update throttling via `update_interval_ms`
- in-process caching of the last calibration snapshot

### gRPC Service

[`proto/rough_pricing.proto`](proto/rough_pricing.proto) and [`src/roughvol/service/`](src/roughvol/service/) expose:

- `MCPrice`
- `BSPrice`
- `ImpliedVol`
- `Calibrate`
- `UpdateCalibrationWindow`

`UpdateCalibrationWindow` is the calibration-toolbox entrypoint for latency-tolerant parameter refreshes over a fixed time window.

### Research Lab

[`src/roughvol/lab/model_comparison.py`](src/roughvol/lab/model_comparison.py) supports model benchmarking on:

- price surface fit
- implied-volatility fit
- discrete delta-hedging PnL

The main example runner is:

```bash
python -m roughvol.experiments.run_model_lab
```

Other experiment entrypoints:

```bash
python -m roughvol.experiments.run_vanilla
python -m roughvol.experiments.run_asian
python -m roughvol.experiments.run_compare_gbm_heston
```

## Testing

Run the full test suite with:

```bash
pytest
```

The test suite currently covers:

- Monte Carlo engine sanity and reproducibility
- Asian option support
- windowed calibration toolbox behavior
- lab comparison output

## Near-Term Direction

The next meaningful extensions are:

1. Add a true rough-volatility model to `src/roughvol/models/` and register it in calibration and lab workflows.
2. Define the C++ consumer contract for `UpdateCalibrationWindow`, including asset identifiers, scheduling, and fallback behavior.
3. Move calibration snapshots from in-memory cache to durable storage if the service must survive restarts.
