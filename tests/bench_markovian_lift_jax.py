"""Benchmark: JAX lax.scan vs NumPy loop for the Markovian-lift scheme.

Run with:
    python tests/bench_markovian_lift_jax.py

The first JAX call triggers XLA compilation; subsequent calls are fast.
The script times both, reports wall-clock seconds, and checks that the
two paths produce prices within Monte Carlo noise of each other.
"""

from __future__ import annotations

import time

import numpy as np

from roughvol.engines.mc import MonteCarloEngine
from roughvol.instruments.vanilla import VanillaOption
from roughvol.models.rough_heston_model import RoughHestonModel, _JAX_AVAILABLE
from roughvol.types import MarketData

# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------
MARKET   = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
OPTION   = VanillaOption(strike=100.0, maturity=1.0, is_call=True)
RH_PARAMS = dict(hurst=0.1, lam=0.3, theta=0.04, nu=0.5, rho=-0.7, v0=0.04, n_factors=8)

CONFIGS = [
    # (n_paths, n_steps)
    (10_000,  64),
    (50_000,  64),
    (50_000, 128),
    (200_000, 64)
]


def _price(scheme: str, n_paths: int, n_steps: int, seed: int = 0) -> tuple[float, float, float]:
    model  = RoughHestonModel(**RH_PARAMS, scheme=scheme)
    engine = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps, seed=seed, antithetic=True)
    t0 = time.perf_counter()
    result = engine.price(model=model, instrument=OPTION, market=MARKET)
    return result.price, result.stderr, time.perf_counter() - t0


def main() -> None:
    if not _JAX_AVAILABLE:
        print("JAX is not installed — install with:  pip install jax")
        return

    print("Markovian-lift benchmark: JAX lax.scan vs NumPy loop")
    print("=" * 65)

    # Warm-up JAX: first call triggers XLA compilation.
    print("Warming up JAX (compiling XLA kernel) ...", end=" ", flush=True)
    _price("markovian-lift", n_paths=1_000, n_steps=16, seed=1)
    print("done.\n")

    header = f"{'n_paths':>10}  {'n_steps':>7}  {'numpy (s)':>10}  {'jax (s)':>10}  {'speedup':>8}  {'|Δprice|':>10}"
    print(header)
    print("-" * len(header))

    for n_paths, n_steps in CONFIGS:
        p_np,  se_np,  t_np  = _price("markovian-lift-numpy", n_paths, n_steps)
        p_jax, se_jax, t_jax = _price("markovian-lift",       n_paths, n_steps)

        speedup   = t_np / t_jax if t_jax > 0 else float("inf")
        delta     = abs(p_np - p_jax)
        noise_tol = 4.0 * np.hypot(se_np, se_jax)   # 4-sigma threshold
        flag      = "" if delta < noise_tol else "  *** MISMATCH"

        print(
            f"{n_paths:>10,}  {n_steps:>7}  {t_np:>10.3f}  {t_jax:>10.3f}"
            f"  {speedup:>7.1f}x  {delta:>10.5f}{flag}"
        )

    print()
    print("Note: JAX time includes any residual JIT overhead on the first")
    print("      benchmarked call. Subsequent calls at the same shape are faster.")


if __name__ == "__main__":
    main()
