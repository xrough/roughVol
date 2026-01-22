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

---

## What This Project Does

This project implements and studies three advanced extensions to standard volatility models:

### 1. Markovian Approximations of Rough Volatility

Turning non-Markovian rough models into finite-dimensional, tractable systems.

### 2. Multilevel Monte Carlo (MLMC)

Reducing the computational cost of Monte Carlo pricing while controlling error.

### 3. Rough Heston Pricing via Fourier Methods

Using fractional dynamics and fast transform-based pricing instead of brute-force simulation.

All methods are validated through **convergence studies**, **performance benchmarks**, and **reproducible experiments**.

---

## Why This Project Exists

Classic models like Black–Scholes and Heston are computationally convenient but empirically weak.
Rough volatility models are empirically strong but numerically difficult.

This project explores how to make modern models usable in practice by addressing:

* Computational cost
* Numerical stability
* Approximation error
* Algorithmic complexity
* Reproducibility

---

## Core Features

### Models

* Black–Scholes (baseline)
* Heston
* Rough Bergomi (Monte Carlo)
* Rough Heston (transform-based)
* Markovian-lifted rough models

### Pricing Engines

* Standard Monte Carlo
* Multilevel Monte Carlo (MLMC)
* Fourier/COS pricing

### Instruments

* Vanilla European options (primary focus)
* Path-dependent options (optional extensions)

### Analytics

* Implied volatility
* Greeks
* Variance reduction
* Classical calibration (non-ML)

---

## Key Contributions

### 1. Markovian Lift of Rough Volatility

Transforms rough, non-Markovian models into finite-dimensional systems that are:

* Faster to simulate
* Easier to calibrate
* Compatible with PDE-based methods

Includes full **accuracy vs speed tradeoff analysis**.

---

### 2. Multilevel Monte Carlo (MLMC)

Implements MLMC for rough volatility pricing, including:

* Level coupling
* Adaptive sampling
* Bias/variance control

Demonstrates **real complexity reduction** compared to naive Monte Carlo.

---

### 3. Rough Heston with Transform Pricing

Implements a semi-analytic pricing route for rough Heston:

* Fractional dynamics solvers
* Fourier-based pricing (COS method)
* Stability and convergence analysis

---

## Repository Structure

```
rough-volatility-lab/
  src/roughvol/
    models/
    kernels/
    sim/
    engines/
    instruments/
    analytics/
    experiments/
  tests/
  docs/
  notebooks/
```

---

## Reproducible Experiments

This project emphasizes **numerical credibility**.

Key experiments:

* MLMC vs standard Monte Carlo cost comparison
* Markovian lift accuracy vs number of factors
* Rough Heston solver stability and convergence
* Implied volatility surface generation

All experiments are fully scriptable and configuration-driven.

---



