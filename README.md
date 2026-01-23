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

## The current stage and the next-step design

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

