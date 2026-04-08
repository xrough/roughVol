"""Plot: Rough Heston error vs n_factors at fixed n_steps.

Diagnoses whether the Markovian approximation error (factor count) or the
time-discretization error (step count) is the dominant term at a given grid.

Run:
    python -m roughvol.experiments.convergence.plot_factors_rh
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import time

from roughvol.engines.mc import MonteCarloEngine
from roughvol.experiments._paths import output_path
from roughvol.experiments.convergence.run_rough_vol import (
    INSTRUMENT,
    MARKET_RH,
    N_PATHS_TEST_RH,
    RH_PARAMS,
    load_or_compute_rh_benchmark,
)
from roughvol.models.rough_heston_model import RoughHestonModel

# Fixed time grid; vary only n_factors.
N_STEPS_FIXED = 128
TEST_FACTORS  = [4, 8, 16, 32, 64]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_factors_sweep() -> dict:
    """Sweep n_factors for markovian-lift and bayer-breneis at fixed n_steps."""
    print(f"n_steps fixed at {N_STEPS_FIXED},  n_paths={N_PATHS_TEST_RH:,}")
    print(f"n_factors sweep: {TEST_FACTORS}\n")

    benchmark = load_or_compute_rh_benchmark()

    results = {
        "benchmark_price": benchmark.price,
        "n_steps":         N_STEPS_FIXED,
        "schemes":         {},
    }

    for scheme in ("markovian-lift", "bayer-breneis"):
        print(f"\n  Scheme: {scheme}")
        print(f"  {'n_factors':>10}  {'price':>10}  {'stderr':>8}  {'|err|':>8}  {'t(s)':>7}")
        print(f"  {'-'*52}")
        prices, stderrs, errors, times, factors = [], [], [], [], []
        for nf in TEST_FACTORS:
            model   = RoughHestonModel(**RH_PARAMS, scheme=scheme, n_factors=nf)
            engine  = MonteCarloEngine(
                n_paths=N_PATHS_TEST_RH, n_steps=N_STEPS_FIXED, seed=42, antithetic=True,
            )
            t0     = time.perf_counter()
            result = engine.price(model=model, instrument=INSTRUMENT, market=MARKET_RH)
            elapsed = time.perf_counter() - t0
            error   = abs(result.price - benchmark.price)
            prices.append(result.price)
            stderrs.append(result.stderr)
            errors.append(error)
            times.append(elapsed)
            factors.append(nf)
            print(
                f"  {nf:>10}  {result.price:>10.5f}  {result.stderr:>8.5f}"
                f"  {error:>8.5f}  {elapsed:>7.3f}",
            )
        results["schemes"][scheme] = {
            "factors": factors,
            "prices":  prices,
            "stderrs": stderrs,
            "errors":  errors,
            "times":   times,
        }

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_factors_sweep(results: dict, out: str | None = None) -> None:
    """Error vs n_factors for markovian-lift and bayer-breneis."""
    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    style_map = {
        "markovian-lift": ("C3", "D--", "Markovian lift"),
        "bayer-breneis":  ("C2", "s:",  "Bayer-Breneis"),
    }

    for scheme, data in results["schemes"].items():
        color, fmt, label = style_map[scheme]
        factors = np.array(data["factors"], dtype=float)
        errors  = np.maximum(np.array(data["errors"], dtype=float), 1e-9)
        stderrs = np.array(data["stderrs"], dtype=float)
        ax.loglog(factors, errors, fmt, color=color, label=label, linewidth=1.8, markersize=7)
        # MC noise band (±2σ)
        ax.fill_between(
            factors,
            np.maximum(errors - 2 * stderrs, 1e-9),
            errors + 2 * stderrs,
            color=color, alpha=0.12,
        )

    # Reference line at 2× MC noise floor (one representative stderr)
    first_scheme = next(iter(results["schemes"].values()))
    noise_floor  = 2.0 * np.mean(first_scheme["stderrs"])
    ax.axhline(noise_floor, color="0.5", linestyle=":", linewidth=1.2, label=f"2σ MC noise ≈ {noise_floor:.4f}")

    ax.set_title(
        f"Rough Heston: error vs n_factors  (n_steps={results['n_steps']}, "
        f"n_paths={N_PATHS_TEST_RH:,})"
    )
    ax.set_xlabel("n_factors")
    ax.set_ylabel("Absolute error vs CF benchmark")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = out or output_path("convergence", "rough_heston_factors.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    plot_factors_sweep(run_factors_sweep())


if __name__ == "__main__":
    main()
