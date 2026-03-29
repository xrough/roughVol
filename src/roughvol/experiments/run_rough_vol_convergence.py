"""Convergence experiment: compare rough-vol simulation schemes.

Produces three panels:
  1. rBergomi price error vs n_steps (log-log) — midpoint vs BLP hybrid
  2. rBergomi wall-clock time vs n_steps — shows O(n²) vs O(n log n) scaling
  3. Rough Heston price vs n_steps — volterra-euler vs markovian-lift

Run with:
    python -m roughvol.experiments.run_rough_vol_convergence
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from roughvol.engines.mc import MonteCarloEngine
from roughvol.instruments.vanilla import VanillaOption
from roughvol.models.rough_bergomi_model import RoughBergomiModel
from roughvol.models.rough_heston_model import RoughHestonModel
from roughvol.types import MarketData

# ---- Market and instrument shared across experiments ----
MARKET_RB = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
INSTRUMENT = VanillaOption(strike=100.0, maturity=1.0, is_call=True)

# ---- Rough Bergomi parameters (regime with visible roughness: H=0.1) ----
RB_PARAMS = dict(hurst=0.1, eta=1.9, rho=-0.7, xi0=0.04)

# ---- Rough Heston parameters ----
MARKET_RH = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
RH_PARAMS = dict(hurst=0.1, lam=0.3, theta=0.04, nu=0.5, rho=-0.7, v0=0.04)

N_PATHS_BENCH = 80_000   # benchmark simulation paths
N_PATHS_TEST = 20_000    # convergence test paths (each data point)
N_STEPS_BENCH = 1024     # benchmark grid (used as "truth")
TEST_STEPS = [8, 16, 32, 64, 128, 256]


def _price(model, n_steps: int, n_paths: int, seed: int) -> tuple[float, float, float]:
    """Return (price, stderr, elapsed_seconds)."""
    engine = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps, seed=seed, antithetic=True)
    t0 = time.perf_counter()
    res = engine.price(model=model, instrument=INSTRUMENT, market=MARKET_RB)
    elapsed = time.perf_counter() - t0
    return res.price, res.stderr, elapsed


def run_rough_bergomi_convergence() -> dict:
    """Run rBergomi convergence study. Returns dict with results."""
    print("=" * 60)
    print("Rough Bergomi: scheme convergence vs n_steps")
    print("=" * 60)

    # ---- Benchmark (dense midpoint) ----
    bench_model = RoughBergomiModel(**RB_PARAMS, scheme="volterra-midpoint")
    print(f"  Computing benchmark: n_steps={N_STEPS_BENCH}, n_paths={N_PATHS_BENCH} ...", end=" ")
    bench_engine = MonteCarloEngine(
        n_paths=N_PATHS_BENCH, n_steps=N_STEPS_BENCH, seed=0, antithetic=True
    )
    t0 = time.perf_counter()
    bench_res = bench_engine.price(model=bench_model, instrument=INSTRUMENT, market=MARKET_RB)
    bench_elapsed = time.perf_counter() - t0
    ref_price = bench_res.price
    print(f"price={ref_price:.5f}  ({bench_elapsed:.1f}s)")

    results = {"ref_price": ref_price, "bench_steps": N_STEPS_BENCH, "schemes": {}}

    for scheme in ["volterra-midpoint", "blp-hybrid", "exact-gaussian"]:
        print(f"\n  Scheme: {scheme}")
        prices, errors, times = [], [], []
        for n in TEST_STEPS:
            model = RoughBergomiModel(**RB_PARAMS, scheme=scheme)
            p, se, elapsed = _price(model, n_steps=n, n_paths=N_PATHS_TEST, seed=42)
            err = abs(p - ref_price)
            prices.append(p)
            errors.append(err)
            times.append(elapsed)
            print(f"    n={n:4d}  price={p:.5f}  |err|={err:.5f}  t={elapsed:.3f}s")
        results["schemes"][scheme] = {"prices": prices, "errors": errors, "times": times}

    return results


def run_rough_heston_convergence() -> dict:
    """Run rough Heston convergence study (volterra-euler vs markovian-lift)."""
    print("\n" + "=" * 60)
    print("Rough Heston: scheme convergence vs n_steps")
    print("=" * 60)

    # Benchmark with markovian-lift at fine grid
    bench_model = RoughHestonModel(**RH_PARAMS, scheme="markovian-lift", n_factors=12)
    bench_engine = MonteCarloEngine(
        n_paths=N_PATHS_BENCH, n_steps=512, seed=0, antithetic=True
    )
    print(f"  Computing benchmark: n_steps=512, n_paths={N_PATHS_BENCH} ...", end=" ")
    t0 = time.perf_counter()
    bench_res = bench_engine.price(model=bench_model, instrument=INSTRUMENT, market=MARKET_RH)
    bench_elapsed = time.perf_counter() - t0
    ref_price_rh = bench_res.price
    print(f"price={ref_price_rh:.5f}  ({bench_elapsed:.1f}s)")

    results = {"ref_price": ref_price_rh, "schemes": {}}

    for scheme, n_factors in [("volterra-euler", 8), ("markovian-lift", 8)]:
        print(f"\n  Scheme: {scheme}")
        prices, errors, times = [], [], []
        for n in TEST_STEPS:
            model = RoughHestonModel(**RH_PARAMS, scheme=scheme, n_factors=n_factors)
            engine = MonteCarloEngine(n_paths=N_PATHS_TEST, n_steps=n, seed=42, antithetic=True)
            t0 = time.perf_counter()
            res = engine.price(model=model, instrument=INSTRUMENT, market=MARKET_RH)
            elapsed = time.perf_counter() - t0
            err = abs(res.price - ref_price_rh)
            prices.append(res.price)
            errors.append(err)
            times.append(elapsed)
            print(f"    n={n:4d}  price={res.price:.5f}  |err|={err:.5f}  t={elapsed:.3f}s")
        results["schemes"][scheme] = {"prices": prices, "errors": errors, "times": times}

    return results


def plot_results(rb_results: dict, rh_results: dict) -> None:
    """Generate three-panel convergence figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Rough Volatility Simulation: Scheme Convergence", fontsize=13, y=1.02)

    steps = np.array(TEST_STEPS, dtype=float)

    # Colour/style map
    style_map = {
        "volterra-midpoint": ("C0", "o-", "Volterra midpoint (§2.5)"),
        "blp-hybrid":        ("C1", "s--", "BLP hybrid (§2.3)"),
        "exact-gaussian":    ("C2", "^:", "Exact Gaussian (§2.1)"),
        "volterra-euler":    ("C0", "o-", "Volterra Euler (§3.1)"),
        "markovian-lift":    ("C3", "D--", "Markovian lift (§3.5)"),
    }

    # ------------------------------------------------------------------
    # Panel 1: rBergomi error vs n_steps (log-log)
    # ------------------------------------------------------------------
    ax1 = axes[0]
    for scheme, data in rb_results["schemes"].items():
        col, fmt, label = style_map[scheme]
        errors = np.array(data["errors"])
        # Replace zero errors (can happen if price==ref exactly) with small floor
        errors = np.maximum(errors, 1e-7)
        ax1.loglog(steps, errors, fmt, color=col, label=label, linewidth=1.8, markersize=7)

    # Reference slope guide-lines: O(n^{H+0.5}) and O(n^1.5) (H=0.1 → O(n^0.6))
    H = RB_PARAMS["hurst"]
    n_ref = np.array([steps[0], steps[-1]])
    # Normalise guide lines to pass through the first midpoint data point
    mp_err0 = max(rb_results["schemes"]["volterra-midpoint"]["errors"][0], 1e-7)
    ax1.loglog(n_ref, mp_err0 * (n_ref / steps[0]) ** (-(H + 0.5)),
               "k:", linewidth=1, label=f"O(n^{{-{H+0.5:.1f}}}) guide")
    blp_err0 = max(rb_results["schemes"]["blp-hybrid"]["errors"][0], 1e-7)
    ax1.loglog(n_ref, blp_err0 * (n_ref / steps[0]) ** (-1.5),
               "k--", linewidth=1, label="O(n^{-1.5}) guide")

    ax1.set_xlabel("n_steps")
    ax1.set_ylabel("|price - reference|")
    ax1.set_title("rBergomi: Error vs n_steps\n(reference = midpoint n=1024)")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 2: rBergomi wall-clock time vs n_steps (log-log)
    # ------------------------------------------------------------------
    ax2 = axes[1]
    for scheme, data in rb_results["schemes"].items():
        col, fmt, label = style_map[scheme]
        ax2.loglog(steps, data["times"], fmt, color=col, label=label, linewidth=1.8, markersize=7)

    # O(n^2) and O(n log n) guide lines
    t0_ref = rb_results["schemes"]["volterra-midpoint"]["times"][0]
    ax2.loglog(n_ref, t0_ref * (n_ref / steps[0]) ** 2, "k:", linewidth=1, label="O(n²) guide")
    ax2.loglog(n_ref,
               t0_ref * (n_ref / steps[0]) * np.log2(n_ref / steps[0] + 1),
               "k--", linewidth=1, label="O(n log n) guide")

    ax2.set_xlabel("n_steps")
    ax2.set_ylabel("Wall-clock time (s)")
    ax2.set_title("rBergomi: Timing vs n_steps")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 3: Rough Heston price convergence
    # ------------------------------------------------------------------
    ax3 = axes[2]
    ref_rh = rh_results["ref_price"]
    for scheme, data in rh_results["schemes"].items():
        col, fmt, label = style_map[scheme]
        prices = np.array(data["prices"])
        ax3.semilogx(steps, prices, fmt, color=col, label=label, linewidth=1.8, markersize=7)

    ax3.axhline(ref_rh, color="gray", linestyle="--", linewidth=1, label=f"Reference ({ref_rh:.4f})")
    ax3.set_xlabel("n_steps")
    ax3.set_ylabel("MC price")
    ax3.set_title("Rough Heston: Price vs n_steps\n(reference = lift n=512)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rough_vol_convergence.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved to: rough_vol_convergence.png")
    plt.show()


def main() -> None:
    rb_results = run_rough_bergomi_convergence()
    rh_results = run_rough_heston_convergence()
    plot_results(rb_results, rh_results)


if __name__ == "__main__":
    main()
