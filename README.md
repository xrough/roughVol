# Stochastic and Rough Volatility Lab

**Numerical Methods for Advanced Volatility Modeling in Python**

This repository is a **research-grade Python framework** for derivative pricing under modern volatility models.

---

## Installation

```bash
git clone https://github.com/jixh-KPZ-1020/Rough-Pricing.git
```

## Venv

After cloning:

```bash
git clone <your-repo>
cd <your-repo>
python3 scripts/setup.py
```
---

## The current stage

```text
We have built so far a simple pricing project with the following structure tree:
.
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
├── Schedule.text
├── notebooks
│   ├── Py_note.ipynb
│   ├── pricing_research.ipynb
│   └── roughvol_research copy.ipynb
├── pyproject.toml
├── scripts
│   └── setup.py
├── src
│   └── roughvol
│       ├── analytics
│       │   └── black_scholes_formula.py
│       ├── engines
│       │   └── mc.py
│       ├── experiments
│       │   └── run_surface.py
│       ├── instruments
│       │   └── vanilla.py
│       ├── logging_utils.py
│       ├── models
│       │   └── GBM_model.py
│       ├── sim
│       │   └── brownian.py
│       └── types.py
└── tests
    ├── test_MC.py
    ├── test_antithetic.py
    └── test_sanity.py
```
## Next steps for this branch

### Types

```text
types.py
└── Core architecture
    ├── Data containers (dataclasses: concrete schemas)
    │   ├── MarketData
    │   │   ├── spot: float, spot price at zero
    │   │   ├── rate: float
    │   │   ├── div_yield: float
    │   │   ├── discount_curve: Any | None
    │   │   ├── forward_variance_curve: Callable[[ArrayF], ArrayF] | None # conditional expectation of variance process under the risk-neutral measure. 
    │   │   └── metadata: Mapping[str, Any]
    │   │
    │   ├── SimConfig
    │   │   ├── n_paths: int
    │   │   ├── maturity: float
    │   │   ├── n_steps: int | None
    │   │   ├── time_grid: ArrayF | None
    │   │   ├── seed: int | None
    │   │   ├── antithetic: bool
    │   │   ├── scheme: str
    │   │   ├── store_paths: bool
    │   │   ├── metadata: Mapping[str, Any]
    │   │   └── method: grid() -> ArrayF
    │   │
    │   ├── PathBundle
    │   │   ├── t: ArrayF
    │   │   ├── state: Mapping[str, ArrayF]
    │   │   │   └── required key: "spot" -> ArrayF (n_paths, n_times)
    │   │   ├── extras: Mapping[str, ArrayF]
    │   │   ├── metadata: Mapping[str, Any]
    │   │   ├── method: __post_init__()  (validates invariants)
    │   │   └── properties (derived facts)
    │   │       ├── spot -> ArrayF
    │   │       ├── spot_T -> ArrayF
    │   │       ├── n_paths -> int
    │   │       ├── n_times -> int
    │   │       └── get(name: str) -> ArrayF
    │   │
    │   └── PriceResult
    │       ├── price: float
    │       ├── stderr: float
    │       ├── ci95: tuple[float, float]
    │       ├── n_paths: int
    │       ├── n_steps: int
    │       ├── seed: int | None
    │       └── metadata: Mapping[str, Any]
    │
    ├── Capability boundaries (Protocols: behavioral contracts)
    │   ├── Instrument
    │   │   ├── maturity: float
    │   │   └── payoff(paths: PathBundle) -> ArrayF
    │   │
    │   ├── TerminalInstrument (legacy support)
    │   │   ├── maturity: float
    │   │   └── payoff_terminal(spot_T: ArrayF) -> ArrayF
    │   │
    │   └── PathModel
    │       └── simulate_paths(market: MarketData, sim: SimConfig, rng: Generator) -> PathBundle
    │
    └── Adapter / utility functions (glue)
        ├── compute_payoff(instrument, paths) -> ArrayF
        │   ├── tries Instrument.payoff(paths)
        │   ├── else tries TerminalInstrument.payoff_terminal(spot_T)
        │   └── else tries legacy payoff(spot_T)
        ├── make_rng(seed) -> Generator
        └── flat_discount_factor(rate, t) -> float
```