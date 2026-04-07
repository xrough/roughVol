"""Convergence workflow for rough-volatility simulation schemes."""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roughvol.engines.mc import MonteCarloEngine
from roughvol.experiments._paths import output_path
from roughvol.instruments.vanilla import VanillaOption
from roughvol.models.rough_bergomi_model import RoughBergomiModel
from roughvol.models.rough_heston_model import RoughHestonModel
from roughvol.types import MarketData

MARKET_RB = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
INSTRUMENT = VanillaOption(strike=100.0, maturity=1.0, is_call=True)
RB_PARAMS = dict(hurst=0.1, eta=1.9, rho=-0.7, xi0=0.04)
MARKET_RH = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
RH_PARAMS = dict(hurst=0.1, lam=0.3, theta=0.04, nu=0.5, rho=-0.7, v0=0.04)
N_PATHS_BENCH = 500_000
N_PATHS_TEST = 1_000_000
N_STEPS_BENCH = 32
TEST_STEPS = [8, 16, 32, 64]
TEST_STEPS_EXACT = [8, 16, 32]

# Rough Heston uses lower path counts for a quick first-pass run.
# volterra-euler is O(n_paths × n_steps²), so high step counts are expensive.
# Raise these back to N_PATHS_BENCH / N_PATHS_TEST once the scheme is validated.
N_PATHS_BENCH_RH = 20_000
N_PATHS_TEST_RH = 50_000
TEST_STEPS_RH = [4, 8, 16, 32]


def price(model: object, n_steps: int, n_paths: int, seed: int) -> tuple[float, float, float]:
    engine = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps, seed=seed, antithetic=True)
    started = time.perf_counter()
    result = engine.price(model=model, instrument=INSTRUMENT, market=MARKET_RB)
    return result.price, result.stderr, time.perf_counter() - started


def run_rough_bergomi_convergence() -> dict:
    print("=" * 60)
    print("Rough Bergomi: scheme convergence vs n_steps")
    print("=" * 60)

    bench_model = RoughBergomiModel(**RB_PARAMS, scheme="exact-gaussian")
    print(
        f"  Computing benchmark: exact-gaussian n_steps={N_STEPS_BENCH}, "
        f"n_paths={N_PATHS_BENCH} ...",
        end=" ",
    )
    bench_engine = MonteCarloEngine(
        n_paths=N_PATHS_BENCH,
        n_steps=N_STEPS_BENCH,
        seed=0,
        antithetic=True,
    )
    started = time.perf_counter()
    bench_result = bench_engine.price(model=bench_model, instrument=INSTRUMENT, market=MARKET_RB)
    bench_elapsed = time.perf_counter() - started
    ref_price = bench_result.price
    print(f"price={ref_price:.5f}  ({bench_elapsed:.1f}s)")

    results = {"ref_price": ref_price, "bench_steps": N_STEPS_BENCH, "schemes": {}}
    for scheme in ["volterra-midpoint", "blp-hybrid", "exact-gaussian"]:
        steps = TEST_STEPS_EXACT if scheme == "exact-gaussian" else TEST_STEPS
        print(f"\n  Scheme: {scheme}  (steps={steps})")
        prices, errors, times, ns = [], [], [], []
        for n_steps in steps:
            model = RoughBergomiModel(**RB_PARAMS, scheme=scheme)
            price_value, _, elapsed = price(model, n_steps=n_steps, n_paths=N_PATHS_TEST, seed=42)
            error = abs(price_value - ref_price)
            prices.append(price_value)
            errors.append(error)
            times.append(elapsed)
            ns.append(n_steps)
            print(f"    n={n_steps:4d}  price={price_value:.5f}  |err|={error:.5f}  t={elapsed:.3f}s")
        results["schemes"][scheme] = {"prices": prices, "errors": errors, "times": times, "steps": ns}
    return results


def run_rough_heston_convergence() -> dict:
    print("\n" + "=" * 60)
    print("Rough Heston: scheme convergence vs n_steps")
    print("=" * 60)

    bench_model = RoughHestonModel(**RH_PARAMS, scheme="markovian-lift", n_factors=8)
    bench_engine = MonteCarloEngine(n_paths=N_PATHS_BENCH_RH, n_steps=64, seed=0, antithetic=True)
    print(f"  Computing benchmark: n_steps=64, n_paths={N_PATHS_BENCH_RH} ...", end=" ")
    started = time.perf_counter()
    bench_result = bench_engine.price(model=bench_model, instrument=INSTRUMENT, market=MARKET_RH)
    bench_elapsed = time.perf_counter() - started
    ref_price = bench_result.price
    print(f"price={ref_price:.5f}  ({bench_elapsed:.1f}s)")

    results = {"ref_price": ref_price, "schemes": {}}
    for scheme, n_factors in [("volterra-euler", 8), ("markovian-lift", 8), ("bayer-breneis", 8)]:
        print(f"\n  Scheme: {scheme}")
        prices, errors, times, ns = [], [], [], []
        for n_steps in TEST_STEPS_RH:
            model = RoughHestonModel(**RH_PARAMS, scheme=scheme, n_factors=n_factors)
            engine = MonteCarloEngine(n_paths=N_PATHS_TEST_RH, n_steps=n_steps, seed=42, antithetic=True)
            started = time.perf_counter()
            result = engine.price(model=model, instrument=INSTRUMENT, market=MARKET_RH)
            elapsed = time.perf_counter() - started
            error = abs(result.price - ref_price)
            prices.append(result.price)
            errors.append(error)
            times.append(elapsed)
            ns.append(n_steps)
            print(f"    n={n_steps:4d}  price={result.price:.5f}  |err|={error:.5f}  t={elapsed:.3f}s")
        results["schemes"][scheme] = {"prices": prices, "errors": errors, "times": times, "steps": ns}
    return results


def plot_error_panel(rb_results: dict, out: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    steps = np.array(TEST_STEPS, dtype=float)
    style_map = {
        "volterra-midpoint": ("C0", "o-", "Volterra midpoint (§2.5)"),
        "blp-hybrid": ("C1", "s--", "BLP hybrid (§2.3)"),
        "exact-gaussian": ("C2", "^:", "Exact Gaussian (§2.1)"),
        "volterra-euler": ("C0", "o-", "Volterra Euler (§3.1)"),
        "markovian-lift": ("C3", "D--", "Markovian lift (§3.5)"),
    }

    n_ref = np.array([steps[0], steps[-1]], dtype=float)
    for scheme, data in rb_results["schemes"].items():
        color, fmt, label = style_map[scheme]
        scheme_steps = np.array(data["steps"], dtype=float)
        errors = np.maximum(np.array(data["errors"]), 1e-7)
        ax.loglog(scheme_steps, errors, fmt, color=color, label=label, linewidth=1.8, markersize=7)

    hurst = RB_PARAMS["hurst"]
    midpoint_error = max(rb_results["schemes"]["volterra-midpoint"]["errors"][0], 1e-7)
    ax.loglog(n_ref, midpoint_error * (n_ref / n_ref[0]) ** (-(hurst + 0.5)), "k:", linewidth=1, label=f"O(n^{{-{hurst+0.5:.1f}}}) guide")
    hybrid_error = max(rb_results["schemes"]["blp-hybrid"]["errors"][0], 1e-7)
    ax.loglog(n_ref, hybrid_error * (n_ref / n_ref[0]) ** (-1.5), "k--", linewidth=1, label="O(n^{-1.5}) guide")

    ax.set_title("rBergomi price error vs n_steps")
    ax.set_xlabel("n_steps")
    ax.set_ylabel("Absolute pricing error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out or output_path("convergence", "rough_vol_error.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_timing_panel(rb_results: dict, out: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    style_map = {
        "volterra-midpoint": ("C0", "o-", "Volterra midpoint"),
        "blp-hybrid": ("C1", "s--", "BLP hybrid"),
        "exact-gaussian": ("C2", "^:", "Exact Gaussian"),
    }
    for scheme, data in rb_results["schemes"].items():
        color, fmt, label = style_map[scheme]
        ax.loglog(data["steps"], data["times"], fmt, color=color, label=label, linewidth=1.8, markersize=7)

    ax.set_title("rBergomi wall-clock time vs n_steps")
    ax.set_xlabel("n_steps")
    ax.set_ylabel("Wall-clock time (s)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out or output_path("convergence", "rough_vol_timing.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_error_panel_rh(rh_results: dict, out: str | None = None) -> None:
    """Log-log error plot for the three Rough Heston schemes."""
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    style_map = {
        "volterra-euler":  ("C0", "o-",  "Volterra Euler (O(n²))"),
        "markovian-lift":  ("C3", "D--", "Markovian lift (O(N·n))"),
        "bayer-breneis":   ("C2", "s:",  "Bayer-Breneis (order-2 weak)"),
    }

    steps_all = np.array(TEST_STEPS_RH, dtype=float)
    n_ref = np.array([steps_all[0], steps_all[-1]], dtype=float)

    for scheme, data in rh_results["schemes"].items():
        color, fmt, label = style_map[scheme]
        scheme_steps = np.array(data["steps"], dtype=float)
        errors = np.maximum(np.array(data["errors"]), 1e-7)
        ax.loglog(scheme_steps, errors, fmt, color=color, label=label, linewidth=1.8, markersize=7)

    # Order-1 reference line anchored on volterra-euler
    if "volterra-euler" in rh_results["schemes"]:
        ve_errors = rh_results["schemes"]["volterra-euler"]["errors"]
        anchor = max(ve_errors[0], 1e-7)
        ax.loglog(n_ref, anchor * (n_ref / n_ref[0]) ** (-1.0), "k:", linewidth=1, label="O(n⁻¹) guide")

    # Order-2 reference line anchored on bayer-breneis
    if "bayer-breneis" in rh_results["schemes"]:
        bb_errors = rh_results["schemes"]["bayer-breneis"]["errors"]
        anchor2 = max(bb_errors[0], 1e-7)
        ax.loglog(n_ref, anchor2 * (n_ref / n_ref[0]) ** (-2.0), "k--", linewidth=1, label="O(n⁻²) guide")

    ax.set_title("Rough Heston price error vs n_steps")
    ax.set_xlabel("n_steps")
    ax.set_ylabel("Absolute pricing error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out or output_path("convergence", "rough_heston_error.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_timing_panel_rh(rh_results: dict, out: str | None = None) -> None:
    """Wall-clock timing plot for the three Rough Heston schemes."""
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    style_map = {
        "volterra-euler":  ("C0", "o-",  "Volterra Euler"),
        "markovian-lift":  ("C3", "D--", "Markovian lift"),
        "bayer-breneis":   ("C2", "s:",  "Bayer-Breneis"),
    }

    for scheme, data in rh_results["schemes"].items():
        color, fmt, label = style_map[scheme]
        ax.loglog(data["steps"], data["times"], fmt, color=color, label=label, linewidth=1.8, markersize=7)

    ax.set_title("Rough Heston wall-clock time vs n_steps")
    ax.set_xlabel("n_steps")
    ax.set_ylabel("Wall-clock time (s)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out or output_path("convergence", "rough_heston_timing.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    run_rough_bergomi_convergence()
    run_rough_heston_convergence()


if __name__ == "__main__":
    main()
