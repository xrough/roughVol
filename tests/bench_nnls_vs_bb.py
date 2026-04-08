"""Benchmark: NNLS vs Bayer-Breneis factor weights across factor counts.

Tests whether BB's super-polynomial convergence in N overtakes NNLS at large N.
Uses the same BMs, same JAX scan path, same n_steps=256 grid.

Run with:
    python tests/bench_nnls_vs_bb.py
"""

from __future__ import annotations

import time

import numpy as np

from roughvol.analytics.rough_heston_pricer import reliable_rough_heston_call_price_cf
from roughvol.kernels.rough_heston import _bayer_breneis_weights, _nnls_weights
from roughvol.models.rough_heston_model import _JAX_AVAILABLE

if not _JAX_AVAILABLE:
    raise SystemExit("JAX is required for this benchmark.")

import jax.numpy as jnp

from roughvol.models.rough_heston_model import _jax_markovian_lift_scan

# ---------------------------------------------------------------------------
# Model / market parameters  (same as run_rough_vol.py)
# ---------------------------------------------------------------------------
HURST  = 0.1
LAM    = 0.3
THETA  = 0.04
NU     = 0.5
RHO    = -0.7
V0     = 0.04
S0     = 100.0
K      = 100.0
T      = 1.0
RATE   = 0.05
DIV    = 0.0

N_PATHS  = 200_000
N_STEPS  = 256
SEED     = 42

TEST_N   = [4, 8, 16, 32, 64, 128, 256]


def _get_cf_benchmark() -> float:
    print("Computing CF benchmark ...", end=" ", flush=True)
    t0 = time.perf_counter()
    result = reliable_rough_heston_call_price_cf(
        spot=S0, strike=K, maturity=T, rate=RATE, div=DIV,
        hurst=HURST, lam=LAM, theta=THETA, nu=NU, rho=RHO, v0=V0,
    )
    print(f"done in {time.perf_counter()-t0:.1f}s  (price={result.price:.5f})")
    return result.price


def _draw_bms(n_paths: int, n_steps: int, dt: float, seed: int) -> tuple:
    """Draw shared correlated Brownian increments."""
    rng   = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)
    z1    = rng.standard_normal((n_paths, n_steps))
    z2    = rng.standard_normal((n_paths, n_steps))
    sqrt_1m_rho2 = np.sqrt(1.0 - RHO ** 2)
    dW2   = z2 * sqrt_dt
    dW1   = RHO * dW2 + sqrt_1m_rho2 * z1 * sqrt_dt
    return dW1, dW2


def _price_with_weights(
    dW1: np.ndarray,
    dW2: np.ndarray,
    dt_arr: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
) -> tuple[float, float]:
    """Price using the JAX scan path with given (w, x) factor weights."""
    S, _, _ = _jax_markovian_lift_scan(
        jnp.asarray(dW1), jnp.asarray(dW2), jnp.asarray(dt_arr),
        jnp.asarray(w), jnp.asarray(x),
        float(LAM), float(THETA), float(NU), float(V0),
        float(S0), float(RATE), float(DIV),
    )
    payoffs = np.asarray(jnp.maximum(S[:, -1] - K, 0.0))
    discount = np.exp(-RATE * T)
    price = float(discount * payoffs.mean())
    stderr = float(discount * payoffs.std() / np.sqrt(len(payoffs)))
    return price, stderr


def main() -> None:
    cf_price = _get_cf_benchmark()

    dt  = T / N_STEPS
    dt_arr = np.full(N_STEPS, dt)

    print(f"\nDrawing {N_PATHS:,} paths × {N_STEPS} steps ...", end=" ", flush=True)
    dW1, dW2 = _draw_bms(N_PATHS, N_STEPS, dt, SEED)
    print("done.\n")

    # Warm-up JAX JIT
    print("JIT warm-up ...", end=" ", flush=True)
    w0, x0 = _nnls_weights(HURST, 4)
    _price_with_weights(dW1[:1000], dW2[:1000], dt_arr, w0, x0)
    print("done.\n")

    hdr = f"{'method':<16}  {'N':>4}  {'price':>10}  {'stderr':>8}  {'error':>10}  {'err/se':>8}"
    print(hdr)
    print("-" * len(hdr))

    for N in TEST_N:
        for method, fn in [("nnls", _nnls_weights), ("bayer-breneis", _bayer_breneis_weights)]:
            w, x = fn(HURST, N)
            price, se = _price_with_weights(dW1, dW2, dt_arr, w, x)
            err = abs(price - cf_price)
            print(f"{method:<16}  {N:>4}  {price:>10.5f}  {se:>8.5f}  {err:>10.5f}  {err/se:>8.2f}x")
        print()


if __name__ == "__main__":
    main()
