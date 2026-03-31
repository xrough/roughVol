"""Convergence experiment: compare rough Bergomi simulation schemes.

Produces two panels:
  1. rBergomi price error vs n_steps (log-log) — midpoint vs BLP hybrid vs Exact Gaussian
  2. rBergomi wall-clock time vs n_steps — shows O(n²) vs O(n log n) scaling

The experiment answers: given a fixed MC budget, which discretization scheme reaches the smallest pricing error the fastest?

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

# ---------------------------------------------------------------------------
# Market and instrument
# ---------------------------------------------------------------------------
# ATM call, 1-year maturity, continuous rate 5%, no dividends.
# ATM is the most sensitive strike to the volatility surface shape, making
# discretization errors most visible.
MARKET_RB = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
INSTRUMENT = VanillaOption(strike=100.0, maturity=1.0, is_call=True)

# ---------------------------------------------------------------------------
# Rough Bergomi model parameters
# ---------------------------------------------------------------------------
# H=0.1 is deep in the rough regime (H << 0.5), where the Volterra kernel
# t^{H-0.5} is most singular and discretization errors are largest.
# eta=1.9 gives strong vol-of-vol; rho=-0.7 gives a pronounced leverage skew.
# xi0=0.04 sets the flat initial forward variance curve (ATM vol ≈ 20%).
RB_PARAMS = dict(hurst=0.1, eta=1.9, rho=-0.7, xi0=0.04)

# ---------------------------------------------------------------------------
# Rough Heston model parameters (kept for run_rough_heston_convergence)
# ---------------------------------------------------------------------------
MARKET_RH = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
RH_PARAMS = dict(hurst=0.1, lam=0.3, theta=0.04, nu=0.5, rho=-0.7, v0=0.04)

# ---------------------------------------------------------------------------
# Simulation budget constants
# ---------------------------------------------------------------------------
# N_PATHS_BENCH: large path count for the reference price so its MC standard
#   error is small (stderr ∝ 1/√N ≈ 0.01 of a single-path std at 10k paths).
#   This makes the reference a stable target that is not itself noisy.
N_PATHS_BENCH = 500_000

# N_PATHS_TEST: smaller path count per test data point. We run one simulation
#   per (scheme, n_steps) combination, so keeping this lower avoids excessive
#   total runtime while still giving a readable error estimate.
N_PATHS_TEST = 1_000_000

# N_STEPS_BENCH: the grid used for the exact-gaussian benchmark.
#   Exact-gaussian samples the Volterra kernel exactly (no kernel discretization
#   error), so n=32 is already converged — the residual spot-SDE discretization
#   error at n=32 is negligible compared to the midpoint scheme's error at n=8..64.
#   The midpoint scheme will show a visible offset from this reference because
#   it converges slowly (O(n^{-0.6})) and is still biased even at n=64.
N_STEPS_BENCH = 32

# TEST_STEPS: coarse grids swept in the convergence study (x-axis of plots).
#   Powers of 2 give evenly-spaced points on a log scale.
TEST_STEPS = [8, 16, 32, 64]

# TEST_STEPS_EXACT: the exact-gaussian scheme requires building and solving an
#   n×n Cholesky system (O(n³) precomputation, O(n²) memory), so it becomes
#   prohibitively slow at large n.
TEST_STEPS_EXACT = [8, 16, 32]


def _price(model, n_steps: int, n_paths: int, seed: int) -> tuple[float, float, float]:
    """Price INSTRUMENT under `model` via MC and return (price, stderr, elapsed_s).

    antithetic=True halves MC variance at no extra path cost by pairing each
    path with its Brownian-motion negation.
    """
    engine = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps, seed=seed, antithetic=True)
    t0 = time.perf_counter()
    res = engine.price(model=model, instrument=INSTRUMENT, market=MARKET_RB)
    elapsed = time.perf_counter() - t0
    return res.price, res.stderr, elapsed


def run_rough_bergomi_convergence() -> dict:
    """Run the rBergomi scheme convergence study.

    Procedure
    ---------
    1. Compute a reference price using the volterra-midpoint scheme at high
       resolution (N_STEPS_BENCH, N_PATHS_BENCH). This is our
       proxy for the true option price as no analytical formula exists for rBergomi.

    2. For each candidate scheme at each n in TEST_STEPS, compute a MC price
       with a separate random seed as we want weak-error rate. The absolute error |p - ref_price|
       is the discretization error plus MC noise.

    3. Record wall-clock times to expose the computational cost scaling of each
       scheme as n grows.

    Returns a dict:
        {
          "ref_price":   float,          # benchmark price
          "bench_steps": int,            # N_STEPS_BENCH
          "schemes": {
            scheme_name: {
              "prices": [...],           # MC prices at each n
              "errors": [...],           # |price - ref_price| at each n
              "times":  [...],           # wall-clock seconds at each n
              "steps":  [...],           # n_steps values tested
            }
          }
        }
    """
    print("=" * 60)
    print("Rough Bergomi: scheme convergence vs n_steps")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Reference price with exact-gaussian at n=32.
    # Exact-gaussian samples the Volterra process from its true Gaussian
    # distribution (Cholesky of the RL fBM covariance), so it has no kernel
    # discretization error. At n=32 the residual spot-SDE Euler error is small
    # enough to serve as a reliable proxy for the true price.
    # The midpoint scheme will appear offset from this reference because it
    # converges slowly (O(n^{-0.6})) and is still biased at the tested grids.
    # ------------------------------------------------------------------
    bench_model = RoughBergomiModel(**RB_PARAMS, scheme="exact-gaussian")
    print(f"  Computing benchmark: exact-gaussian n_steps={N_STEPS_BENCH}, n_paths={N_PATHS_BENCH} ...", end=" ")
    bench_engine = MonteCarloEngine(
        n_paths=N_PATHS_BENCH, n_steps=N_STEPS_BENCH, seed=0, antithetic=True
    )
    t0 = time.perf_counter()
    bench_res = bench_engine.price(model=bench_model, instrument=INSTRUMENT, market=MARKET_RB)
    bench_elapsed = time.perf_counter() - t0
    ref_price = bench_res.price
    print(f"price={ref_price:.5f}  ({bench_elapsed:.1f}s)")

    results = {"ref_price": ref_price, "bench_steps": N_STEPS_BENCH, "schemes": {}}

    # ------------------------------------------------------------------
    # Step 2 & 3: Sweep each scheme over its test grid
    # ------------------------------------------------------------------
    for scheme in ["volterra-midpoint", "blp-hybrid", "exact-gaussian"]:
        # exact-gaussian has O(n³) Cholesky precomputation and O(n²) memory;
        # restrict it to TEST_STEPS_EXACT to keep runtime manageable.
        steps = TEST_STEPS_EXACT if scheme == "exact-gaussian" else TEST_STEPS
        print(f"\n  Scheme: {scheme}  (steps={steps})")
        prices, errors, times, ns = [], [], [], []
        for n in steps:
            model = RoughBergomiModel(**RB_PARAMS, scheme=scheme)
            p, se, elapsed = _price(model, n_steps=n, n_paths=N_PATHS_TEST, seed=42)
            # Absolute error vs reference: measures discretization bias + MC noise.
            # Once error flattens (stops decreasing), discretization error is below
            # the MC noise floor and adding more steps gives no further benefit.
            err = abs(p - ref_price)
            prices.append(p)
            errors.append(err)
            times.append(elapsed)
            ns.append(n)
            print(f"    n={n:4d}  price={p:.5f}  |err|={err:.5f}  t={elapsed:.3f}s")
        results["schemes"][scheme] = {"prices": prices, "errors": errors, "times": times, "steps": ns}

    return results


def run_rough_heston_convergence() -> dict:
    """Run rough Heston convergence study (volterra-euler vs markovian-lift)."""
    print("\n" + "=" * 60)
    print("Rough Heston: scheme convergence vs n_steps")
    print("=" * 60)

    # Benchmark with markovian-lift at fine grid
    bench_model = RoughHestonModel(**RH_PARAMS, scheme="markovian-lift", n_factors=8)
    bench_engine = MonteCarloEngine(
        n_paths=N_PATHS_BENCH, n_steps=128, seed=0, antithetic=True
    )
    print(f"  Computing benchmark: n_steps=128, n_paths={N_PATHS_BENCH} ...", end=" ")
    t0 = time.perf_counter()
    bench_res = bench_engine.price(model=bench_model, instrument=INSTRUMENT, market=MARKET_RH)
    bench_elapsed = time.perf_counter() - t0
    ref_price_rh = bench_res.price
    print(f"price={ref_price_rh:.5f}  ({bench_elapsed:.1f}s)")

    results = {"ref_price": ref_price_rh, "schemes": {}}

    for scheme, n_factors in [("volterra-euler", 8), ("markovian-lift", 8)]:
        print(f"\n  Scheme: {scheme}")
        prices, errors, times, ns = [], [], [], []
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
            ns.append(n)
            print(f"    n={n:4d}  price={res.price:.5f}  |err|={err:.5f}  t={elapsed:.3f}s")
        results["schemes"][scheme] = {"prices": prices, "errors": errors, "times": times, "steps": ns}

    return results


def plot_results(rb_results: dict) -> None:
    """Generate two-panel convergence figure and save to rough_vol_convergence.png.

    Panel 1 — Error vs n_steps (log-log):
        Shows how fast each scheme's pricing error shrinks as the time grid
        is refined. Steeper downward slope = faster convergence order.
        Dashed guide-lines mark the theoretical rates:
          - O(n^{-(H+0.5)}) for the midpoint scheme (weak order H+0.5 ≈ 0.6)
          - O(n^{-1.5})     for the BLP hybrid scheme (higher order near-field)

    Panel 2 — Wall-clock time vs n_steps (log-log):
        Shows the computational cost of each scheme.
        Guide-lines mark O(n²) and O(n log n) scalings so you can read off
        which complexity class each scheme belongs to.

    Reading the two panels together reveals the accuracy/speed tradeoff:
    a scheme that is slower per step (right panel) may need far fewer steps
    to hit a target error (left panel), potentially winning overall.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Rough Volatility Simulation: Scheme Convergence", fontsize=13, y=1.02)

    steps = np.array(TEST_STEPS, dtype=float)

    # Consistent colour/marker/label per scheme across both panels.
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

    # Anchor points for the guide-line segments (leftmost and rightmost n tested).
    n_ref_lo = steps[0]
    n_ref_hi = steps[-1]
    n_ref = np.array([n_ref_lo, n_ref_hi], dtype=float)

    for scheme, data in rb_results["schemes"].items():
        col, fmt, label = style_map[scheme]
        s = np.array(data.get("steps", TEST_STEPS), dtype=float)
        # Clip errors to 1e-7 so log scale doesn't break on near-zero values
        # (can happen when MC noise causes the error to be spuriously tiny).
        errors = np.maximum(np.array(data["errors"]), 1e-7)
        ax1.loglog(s, errors, fmt, color=col, label=label, linewidth=1.8, markersize=7)

    # Theoretical convergence rate guide-lines, anchored to the first data point
    # of each scheme so they pass through the actual curve for easy slope reading.
    H = RB_PARAMS["hurst"]

    # Midpoint weak order: O(n^{-(H+0.5)}) — from the Riemann-sum error of the
    # Volterra kernel integral with H-dependent singularity at the origin.
    mp_err0 = max(rb_results["schemes"]["volterra-midpoint"]["errors"][0], 1e-7)
    ax1.loglog(n_ref, mp_err0 * (n_ref / n_ref_lo) ** (-(H + 0.5)),
               "k:", linewidth=1, label=f"O(n^{{-{H+0.5:.1f}}}) guide")

    # BLP hybrid weak order: O(n^{-1.5}) — the near-field exact correction
    # eliminates the leading singularity, boosting convergence to 3/2.
    blp_err0 = max(rb_results["schemes"]["blp-hybrid"]["errors"][0], 1e-7)
    ax1.loglog(n_ref, blp_err0 * (n_ref / n_ref_lo) ** (-1.5),
               "k--", linewidth=1, label="O(n^{-1.5}) guide")

    ax1.set_xlabel("n_steps")
    ax1.set_ylabel("|price - reference|")
    ax1.set_title(f"rBergomi: Error vs n_steps\n(reference = exact-gaussian n={N_STEPS_BENCH})")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 2: rBergomi wall-clock time vs n_steps (log-log)
    # ------------------------------------------------------------------
    ax2 = axes[1]
    for scheme, data in rb_results["schemes"].items():
        col, fmt, label = style_map[scheme]
        s = np.array(data.get("steps", TEST_STEPS), dtype=float)
        ax2.loglog(s, data["times"], fmt, color=col, label=label, linewidth=1.8, markersize=7)

    # Guide-lines anchored to the midpoint scheme's first timing measurement.
    # O(n²): midpoint and exact-gaussian both require O(n²) operations
    #   (midpoint: full n×n weight matrix; exact: n×n Cholesky).
    # O(n log n): BLP hybrid uses FFT convolution for the far-field,
    #   reducing cost from O(n²) to O(n log n).
    t0_ref = rb_results["schemes"]["volterra-midpoint"]["times"][0]
    ax2.loglog(n_ref, t0_ref * (n_ref / n_ref_lo) ** 2,
               "k:", linewidth=1, label="O(n²) guide")
    ax2.loglog(n_ref,
               t0_ref * (n_ref / n_ref_lo) * np.log2(n_ref / n_ref_lo + 1),
               "k--", linewidth=1, label="O(n log n) guide")

    ax2.set_xlabel("n_steps")
    ax2.set_ylabel("Wall-clock time (s)")
    ax2.set_title("rBergomi: Timing vs n_steps")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("rough_vol_convergence.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved to: rough_vol_convergence.png")
    plt.show()


def main() -> None:
    rb_results = run_rough_bergomi_convergence()
    plot_results(rb_results)


if __name__ == "__main__":
    main()
