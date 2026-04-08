"""Convergence simulation runner for rough-volatility schemes.

This module owns the simulation logic only — no plotting.

Quick correctness pass (seconds):
    python -m roughvol.experiments.convergence.run_rough_vol --quick

Full comparison (~5 min):
    python -m roughvol.experiments.convergence.run_rough_vol

Each ``plot_*.py`` in this package imports the relevant ``run_*`` function
and config constants it needs to annotate plots.

Boundary
--------
* ``run_rough_vol.py``  — simulation, CV, caching.  No matplotlib.
* ``plot_*.py``         — own their plot function, call ``run_*`` for data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from roughvol.analytics.heston_pricer import HestonCallPrice, heston_call_price
from roughvol.analytics.rough_heston_pricer import (
    RoughHestonBenchmarkPrice,
    reliable_rough_heston_call_price_cf,
)
from roughvol.engines.mc import MonteCarloEngine
from roughvol.experiments._paths import output_path
from roughvol.instruments.vanilla import VanillaOption
from roughvol.kernels.rough_heston import markovian_lift_weights
from roughvol.models.rough_bergomi_model import RoughBergomiModel
from roughvol.models.rough_heston_model import RoughHestonModel, _JAX_AVAILABLE
from roughvol.sim.brownian import correlated_brownian_increments
from roughvol.types import MarketData, make_rng

if _JAX_AVAILABLE:
    import jax.numpy as jnp
    from roughvol.models.rough_heston_model import (
        _jax_markovian_lift_scan,
        _jax_ml_terminal_scan,
    )

# ---------------------------------------------------------------------------
# Shared instrument
# ---------------------------------------------------------------------------
INSTRUMENT = VanillaOption(strike=100.0, maturity=1.0, is_call=True)

# ---------------------------------------------------------------------------
# Rough Bergomi config
# ---------------------------------------------------------------------------
MARKET_RB        = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
RB_PARAMS        = dict(hurst=0.1, eta=1.9, rho=-0.7, xi0=0.04)
N_PATHS_BENCH    = 500_000
N_PATHS_TEST     = 1_000_000
N_STEPS_BENCH    = 32
TEST_STEPS       = [8, 16, 32, 64]
TEST_STEPS_EXACT = [8, 16, 32]

# ---------------------------------------------------------------------------
# Rough Heston config
# ---------------------------------------------------------------------------
MARKET_RH = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
RH_PARAMS = dict(hurst=0.01, lam=0.3, theta=0.04, nu=0.5, rho=-0.7, v0=0.04)

# Per-scheme step grids.
# VE is O(n_paths × n_steps²) — cap at 256.
# ML and BB with JAX are O(N·n) — extend to 1024.
TEST_STEPS_RH_VE   = [32, 64, 128, 256]
TEST_STEPS_RH_FAST = [32, 64, 128, 256, 512, 1024]

# Path counts.  VE is too slow for 200K; ML/BB with JAX handle it fine.
N_PATHS_VE   = 50_000
N_PATHS_FAST = 200_000

# Quick-pass params for a fast correctness check.
N_PATHS_QUICK    = 5_000
QUICK_STEPS_VE   = [16, 32]
QUICK_STEPS_FAST = [16, 32, 64]

def _cf_benchmark_settings(hurst: float) -> dict:
    """Return CF benchmark solver settings appropriate for the given Hurst parameter.

    At very small H the fractional Riccati solution diverges for large u;
    use a finer grid and a conservative Fourier integration upper limit.
    """
    small_h = hurst < 0.08
    return {
        "riccati_steps_grid": (800, 1200) if small_h else (400, 600),
        "integration_limits": (40.0, 60.0) if small_h else (100.0, 150.0),
        "integration_epsabs": 1e-8,
        "integration_epsrel": 1e-6,
        "martingale_tol":     5e-4,
        "stability_tol":      5e-3,
    }

# Module-level defaults (used when hurst override is not provided).
RH_CF_BENCHMARK = _cf_benchmark_settings(RH_PARAMS["hurst"])

# ---------------------------------------------------------------------------
# GH5 constants (Bayer-Breneis variance-driver innovation)
# ---------------------------------------------------------------------------
_GH5_NODES = np.array([-2.85697001, -1.35562618, 0.0, 1.35562618, 2.85697001])
_GH5_PROBS = np.array([0.01125741, 0.22207592, 0.53333333, 0.22207592, 0.01125741])
_GH5_CDF   = np.cumsum(_GH5_PROBS[:-1])   # 4 breakpoints for np.select


# ===========================================================================
# Cache helpers
# ===========================================================================

def _cache_path(name: str) -> Path:
    return Path(output_path("convergence", name))


def _make_key(inputs: dict) -> str:
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Rough Heston CF benchmark cache
# ---------------------------------------------------------------------------

def _rh_benchmark_inputs(rh_params: dict | None = None) -> dict:
    params = {**RH_PARAMS, **(rh_params or {})}
    cf_settings = _cf_benchmark_settings(params["hurst"])
    return {
        "spot": MARKET_RH.spot, "strike": INSTRUMENT.strike,
        "maturity": INSTRUMENT.maturity,
        "rate": MARKET_RH.rate, "div": MARKET_RH.div_yield,
        **params,
        **{k: list(v) if isinstance(v, tuple) else v
           for k, v in cf_settings.items()},
    }


def load_or_compute_rh_benchmark(
    rh_params: dict | None = None,
    *,
    verbose: bool = True,
) -> RoughHestonBenchmarkPrice:
    """Return CF benchmark, loading from disk cache when inputs are unchanged."""
    params     = {**RH_PARAMS, **(rh_params or {})}
    cf_settings = _cf_benchmark_settings(params["hurst"])
    key        = _make_key(_rh_benchmark_inputs(rh_params))
    cache_path = _cache_path(f"rh_benchmark_cache_{key}.json")

    if cache_path.exists():
        with cache_path.open() as f:
            cached = json.load(f)
        if cached.get("key") == key:
            if verbose:
                print(
                    f"  Loaded RH benchmark from cache  "
                    f"(price={cached['price']:.5f},  "
                    f"martingale_err={cached['martingale_error']:.2e},  "
                    f"stability_err={cached['stability_error']:.2e})"
                )
            return RoughHestonBenchmarkPrice(**{k: cached[k] for k in (
                "price", "p1", "p2", "integration_error",
                "martingale_error", "stability_error",
                "riccati_steps", "integration_limit",
            )})

    if verbose:
        print(
            "  Computing RH benchmark: fractional Riccati + Fourier inversion ...",
            end=" ", flush=True,
        )
    t0        = time.perf_counter()
    benchmark = reliable_rough_heston_call_price_cf(
        spot=MARKET_RH.spot, strike=INSTRUMENT.strike, maturity=INSTRUMENT.maturity,
        rate=MARKET_RH.rate, div=MARKET_RH.div_yield,
        hurst=params["hurst"], lam=params["lam"], theta=params["theta"],
        nu=params["nu"], rho=params["rho"], v0=params["v0"],
        **cf_settings,
    )
    if verbose:
        print(
            f"price={benchmark.price:.5f}  ({time.perf_counter()-t0:.1f}s)  "
            f"martingale_err={benchmark.martingale_error:.2e}  "
            f"stability_err={benchmark.stability_error:.2e}"
        )

    payload = {"key": key, **{k: getattr(benchmark, k) for k in (
        "price", "p1", "p2", "integration_error",
        "martingale_error", "stability_error", "riccati_steps", "integration_limit",
    )}}
    with cache_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return benchmark


# ---------------------------------------------------------------------------
# Heston CF price cache  (control-variate anchor)
# ---------------------------------------------------------------------------

def _heston_cf_inputs(rh_params: dict | None = None) -> dict:
    params = {**RH_PARAMS, **(rh_params or {})}
    return {
        "spot": MARKET_RH.spot, "strike": INSTRUMENT.strike,
        "maturity": INSTRUMENT.maturity,
        "rate": MARKET_RH.rate, "div": MARKET_RH.div_yield,
        "kappa": params["lam"],  "theta": params["theta"],
        "sigma": params["nu"],   "rho":   params["rho"],
        "v0":    params["v0"],
    }


def load_or_compute_heston_cf_price(
    rh_params: dict | None = None,
    *,
    verbose: bool = True,
) -> float:
    """Return Heston CF call price (the CV anchor), cached on disk."""
    key        = _make_key(_heston_cf_inputs(rh_params))
    cache_path = _cache_path("heston_cf_cache.json")

    if cache_path.exists():
        with cache_path.open() as f:
            cached = json.load(f)
        if cached.get("key") == key:
            if verbose:
                print(f"  Loaded Heston CF price from cache  (price={cached['price']:.5f})")
            return float(cached["price"])

    if verbose:
        print("  Computing Heston CF price ...", end=" ", flush=True)
    t0     = time.perf_counter()
    inp    = _heston_cf_inputs(rh_params)
    result = heston_call_price(
        spot=inp["spot"], strike=inp["strike"], maturity=inp["maturity"],
        rate=inp["rate"], div=inp["div"],
        kappa=inp["kappa"], theta=inp["theta"], sigma=inp["sigma"],
        rho=inp["rho"], v0=inp["v0"],
    )
    if verbose:
        print(
            f"price={result.price:.5f}  ({time.perf_counter()-t0:.3f}s)  "
            f"martingale_err={result.martingale_error:.2e}"
        )

    with cache_path.open("w") as f:
        json.dump({"key": key, "price": result.price}, f, indent=2)
    return result.price


# ===========================================================================
# Simulation helpers
# ===========================================================================

def _heston_euler_from_bm(
    dW1: np.ndarray,
    dW2: np.ndarray,
    dt: np.ndarray,
    *,
    kappa: float,
    theta: float,
    nu: float,
    v0: float,
    S0: float,
    r: float,
    q: float,
) -> np.ndarray:
    """Euler-full-truncation for Heston using pre-generated BM increments.

    Parameters
    ----------
    dW1, dW2 : (n_paths, n_steps)  correlated BM increments (spot, variance)
    dt       : (n_steps,)          time-step sizes

    Returns
    -------
    S_T : (n_paths,)  terminal spot values
    """
    n_paths, n_steps = dW1.shape
    V = np.full(n_paths, v0)
    S = np.full(n_paths, S0)
    for i in range(n_steps):
        v_pos = np.maximum(V, 0.0)
        V = np.maximum(
            v_pos + kappa * (theta - v_pos) * dt[i] + nu * np.sqrt(v_pos) * dW2[:, i],
            0.0,
        )
        S *= np.exp((r - q - 0.5 * v_pos) * dt[i] + np.sqrt(v_pos) * dW1[:, i])
    return S


def _sample_bb_brownians(
    n_paths: int,
    n_steps: int,
    dt: np.ndarray,
    rho: float,
    rng: np.random.Generator,
    antithetic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Bayer-Breneis GH5 BM increments (dW1, dW2).

    dW2 uses the 5-point Gauss-Hermite discrete distribution (9-moment match).
    dW1 = rho*dW2 + sqrt(1-rho²)*dW_perp  where dW_perp is Gaussian.
    """
    half       = n_paths // 2 if antithetic else n_paths
    sqrt_dt    = np.sqrt(dt)
    sqrt_1mrho = np.sqrt(max(1.0 - rho ** 2, 0.0))

    U       = rng.uniform(size=(half, n_steps))
    z2_base = np.select(
        [U < _GH5_CDF[0], U < _GH5_CDF[1], U < _GH5_CDF[2], U < _GH5_CDF[3]],
        [_GH5_NODES[0],   _GH5_NODES[1],   _GH5_NODES[2],   _GH5_NODES[3]],
        default=_GH5_NODES[4],
    )
    z_perp_base = rng.standard_normal(size=(half, n_steps))

    if antithetic:
        z2     = np.concatenate([z2_base,    -z2_base],    axis=0)
        z_perp = np.concatenate([z_perp_base, -z_perp_base], axis=0)
    else:
        z2, z_perp = z2_base, z_perp_base

    dW2 = z2    * sqrt_dt[None, :]
    dW1 = rho * dW2 + sqrt_1mrho * z_perp * sqrt_dt[None, :]
    return dW1, dW2


def _apply_cv(
    pv_rh: np.ndarray,
    pv_h: np.ndarray,
    heston_cf_price: float,
) -> tuple[float, float]:
    """Optimal control-variate adjustment.  Returns (price_cv, stderr_cv)."""
    cov_mat = np.cov(pv_rh, pv_h)
    beta    = cov_mat[0, 1] / max(cov_mat[1, 1], 1e-15)
    pv_cv   = pv_rh - beta * (pv_h - heston_cf_price)
    n       = len(pv_cv)
    return float(np.mean(pv_cv)), float(np.std(pv_cv, ddof=1) / np.sqrt(n))


def _run_one(
    *,
    scheme: str,
    n_steps: int,
    n_paths: int,
    seed: int,
    use_cv: bool,
    heston_cf_price: float | None,
    n_factors: int = 32,
    rh_params: dict | None = None,
) -> tuple[float, float, float]:
    """Simulate one (scheme, n_steps) cell.  Returns (price, stderr, elapsed_s)."""

    params = {**RH_PARAMS, **(rh_params or {})}

    T  = float(INSTRUMENT.maturity)
    K  = float(INSTRUMENT.strike)
    S0 = float(MARKET_RH.spot)
    r  = float(MARKET_RH.rate)
    q  = float(MARKET_RH.div_yield)
    df = float(np.exp(-r * T))

    model = RoughHestonModel(**params, scheme=scheme, n_factors=n_factors)

    # ------------------------------------------------------------------
    # Plain MC path (no CV, or scheme not supported for CV)
    # ------------------------------------------------------------------
    if not use_cv or heston_cf_price is None:
        engine = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps, seed=seed, antithetic=True)
        t0     = time.perf_counter()
        result = engine.price(model=model, instrument=INSTRUMENT, market=MARKET_RH)
        return result.price, result.stderr, time.perf_counter() - t0

    # ------------------------------------------------------------------
    # CV path: generate BMs externally, simulate RH + Heston on same BMs
    # ------------------------------------------------------------------
    H   = float(model.hurst)
    lam = float(model.lam);   theta = float(model.theta)
    nu  = float(model.nu);    v0    = float(model.v0)
    rho = float(model.rho)

    t  = np.linspace(0.0, T, n_steps + 1)
    dt = np.diff(t)
    rng = make_rng(seed)

    t0 = time.perf_counter()

    if scheme == "markovian-lift":
        dW1, dW2 = correlated_brownian_increments(
            n_paths=n_paths, n_steps=n_steps, dt=1.0,
            rho=rho, rng=rng, antithetic=True,
        )
        dW1 *= np.sqrt(dt)[None, :]
        dW2 *= np.sqrt(dt)[None, :]
        w, x = markovian_lift_weights(H, n_factors=n_factors)
        if _JAX_AVAILABLE:
            S_rh_T, _ = _jax_ml_terminal_scan(
                jnp.asarray(dW1), jnp.asarray(dW2), jnp.asarray(dt),
                jnp.asarray(w),   jnp.asarray(x),
                lam, theta, nu, v0, S0, r, q,
            )
            S_rh = np.asarray(S_rh_T)[:, None]   # (n_paths, 1) — only terminal needed
        else:
            S_rh, _, _ = model._markovian_lift_numpy(
                t=t, dt=dt, n_steps=n_steps, n_paths=n_paths,
                H=H, lam=lam, theta=theta, nu=nu, v0=v0,
                S0=S0, r=r, q=q, dW1=dW1, dW2=dW2,
            )

    elif scheme == "bayer-breneis":
        dW1, dW2 = _sample_bb_brownians(
            n_paths=n_paths, n_steps=n_steps, dt=dt, rho=rho, rng=rng,
        )
        w, x = markovian_lift_weights(H, n_factors=n_factors, method="bayer-breneis")
        if _JAX_AVAILABLE:
            S_rh_T, _ = _jax_ml_terminal_scan(
                jnp.asarray(dW1), jnp.asarray(dW2), jnp.asarray(dt),
                jnp.asarray(w),   jnp.asarray(x),
                lam, theta, nu, v0, S0, r, q,
            )
            S_rh = np.asarray(S_rh_T)[:, None]
        else:
            S_rh, _, _ = model._markovian_lift_numpy(
                t=t, dt=dt, n_steps=n_steps, n_paths=n_paths,
                H=H, lam=lam, theta=theta, nu=nu, v0=v0,
                S0=S0, r=r, q=q, dW1=dW1, dW2=dW2,
            )

    elif scheme == "volterra-euler":
        dW1, dW2 = correlated_brownian_increments(
            n_paths=n_paths, n_steps=n_steps, dt=1.0,
            rho=rho, rng=rng, antithetic=True,
        )
        dW1 *= np.sqrt(dt)[None, :]
        dW2 *= np.sqrt(dt)[None, :]
        S_rh, _ = model._volterra_euler(
            t=t, dt=dt, n_steps=n_steps, n_paths=n_paths,
            H=H, lam=lam, theta=theta, nu=nu, v0=v0,
            S0=S0, r=r, q=q, dW1=dW1, dW2=dW2,
        )

    else:
        raise ValueError(f"CV not supported for scheme {scheme!r}")

    elapsed = time.perf_counter() - t0

    # Heston control on same BMs
    S_h_T = _heston_euler_from_bm(
        dW1=dW1, dW2=dW2, dt=dt,
        kappa=lam, theta=theta, nu=nu, v0=v0, S0=S0, r=r, q=q,
    )

    pv_rh = df * np.maximum(S_rh[:, -1] - K, 0.0)
    pv_h  = df * np.maximum(S_h_T       - K, 0.0)

    price, stderr = _apply_cv(pv_rh, pv_h, heston_cf_price)
    return price, stderr, elapsed


# ===========================================================================
# Richardson successive differences
# ===========================================================================

def _compute_richardson(schemes: dict[str, dict]) -> dict[str, dict]:
    """For consecutive power-of-2 step pairs, compute |p(2n) - p(n)|."""
    out = {}
    for scheme, data in schemes.items():
        steps, prices = data["steps"], data["prices"]
        rich_steps, rich_diffs = [], []
        for i in range(len(steps) - 1):
            if steps[i + 1] == 2 * steps[i]:
                rich_steps.append(steps[i + 1])
                rich_diffs.append(abs(prices[i + 1] - prices[i]))
        out[scheme] = {"steps": rich_steps, "diffs": rich_diffs}
    return out


# ===========================================================================
# Simulation runners
# ===========================================================================

def _price_rb(model, n_steps, n_paths, seed):
    engine  = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps, seed=seed, antithetic=True)
    t0      = time.perf_counter()
    result  = engine.price(model=model, instrument=INSTRUMENT, market=MARKET_RB)
    return result.price, result.stderr, time.perf_counter() - t0


def run_rough_bergomi_convergence() -> dict:
    """Run rBergomi convergence across schemes; return results dict."""
    print("=" * 60)
    print("Rough Bergomi: scheme convergence vs n_steps")
    print("=" * 60)

    bench_model  = RoughBergomiModel(**RB_PARAMS, scheme="exact-gaussian")
    bench_engine = MonteCarloEngine(
        n_paths=N_PATHS_BENCH, n_steps=N_STEPS_BENCH, seed=0, antithetic=True,
    )
    print(
        f"  Computing benchmark: exact-gaussian n_steps={N_STEPS_BENCH}, "
        f"n_paths={N_PATHS_BENCH} ...",
        end=" ",
    )
    t0           = time.perf_counter()
    bench_result = bench_engine.price(model=bench_model, instrument=INSTRUMENT, market=MARKET_RB)
    ref_price    = bench_result.price
    print(f"price={ref_price:.5f}  ({time.perf_counter()-t0:.1f}s)")

    results = {"ref_price": ref_price, "bench_steps": N_STEPS_BENCH, "schemes": {}}
    for scheme in ["volterra-midpoint", "blp-hybrid", "exact-gaussian"]:
        steps = TEST_STEPS_EXACT if scheme == "exact-gaussian" else TEST_STEPS
        print(f"\n  Scheme: {scheme}  (steps={steps})")
        prices, errors, times, ns = [], [], [], []
        for n_steps in steps:
            model     = RoughBergomiModel(**RB_PARAMS, scheme=scheme)
            pv, _, el = _price_rb(model, n_steps=n_steps, n_paths=N_PATHS_TEST, seed=42)
            error = abs(pv - ref_price)
            prices.append(pv); errors.append(error)
            times.append(el); ns.append(n_steps)
            print(f"    n={n_steps:4d}  price={pv:.5f}  |err|={error:.5f}  t={el:.3f}s")
        results["schemes"][scheme] = {
            "prices": prices, "errors": errors, "times": times, "steps": ns,
        }
    return results


def run_rough_heston_convergence(
    *,
    quick: bool = False,
    n_paths_ve: int | None = None,
    n_paths_fast: int | None = None,
    steps_ve: list[int] | None = None,
    steps_fast: list[int] | None = None,
    use_cv: bool = True,
    seed: int = 42,
    hurst: float | None = None,
) -> dict:
    """Run rHeston convergence with optional Heston control variate.

    Parameters
    ----------
    quick       : Use small params for a fast correctness check.
    n_paths_ve  : Path count for Volterra-Euler (default 50K; full 50K, quick 5K).
    n_paths_fast: Path count for Markovian-lift and Bayer-Breneis
                  (default 200K; quick 5K).
    steps_ve    : Step grid for VE  (default TEST_STEPS_RH_VE / QUICK_STEPS_VE).
    steps_fast  : Step grid for ML and BB (default TEST_STEPS_RH_FAST / QUICK).
    use_cv      : Apply Heston control variate to reduce MC variance.
    seed        : RNG seed (same seed → same paths across restarts).
    hurst       : Override Hurst parameter (default: use RH_PARAMS["hurst"]).
    """
    rh_params = {"hurst": hurst} if hurst is not None else None

    if quick:
        n_paths_ve   = n_paths_ve   or N_PATHS_QUICK
        n_paths_fast = n_paths_fast or N_PATHS_QUICK
        steps_ve     = steps_ve     or QUICK_STEPS_VE
        steps_fast   = steps_fast   or QUICK_STEPS_FAST
    else:
        n_paths_ve   = n_paths_ve   or N_PATHS_VE
        n_paths_fast = n_paths_fast or N_PATHS_FAST
        steps_ve     = steps_ve     or TEST_STEPS_RH_VE
        steps_fast   = steps_fast   or TEST_STEPS_RH_FAST

    _hurst = hurst if hurst is not None else RH_PARAMS["hurst"]

    # ------------------------------------------------------------------
    # Results cache — keyed on all simulation inputs
    # ------------------------------------------------------------------
    _params = {**RH_PARAMS, **(rh_params or {})}
    sim_key_inputs = {
        **_params,
        "n_paths_ve": n_paths_ve, "n_paths_fast": n_paths_fast,
        "steps_ve": steps_ve, "steps_fast": steps_fast,
        "use_cv": use_cv, "seed": seed, "quick": quick,
        "n_factors": 32,
    }
    sim_key       = _make_key(sim_key_inputs)
    sim_cache_path = _cache_path(f"rh_sim_cache_{sim_key}.json")

    if sim_cache_path.exists():
        with sim_cache_path.open() as f:
            cached = json.load(f)
        print(f"\n  Loaded simulation results from cache  (H={_hurst})")
        return cached

    print("\n" + "=" * 60)
    print(f"Rough Heston: scheme convergence vs n_steps  (H={_hurst})")
    print(f"  quick={quick}  use_cv={use_cv}")
    print(f"  n_paths VE={n_paths_ve:,}  fast={n_paths_fast:,}")
    print("=" * 60)

    benchmark        = load_or_compute_rh_benchmark(rh_params)
    heston_cf_price  = load_or_compute_heston_cf_price(rh_params) if use_cv else None

    if use_cv and heston_cf_price is not None:
        print(f"  CV anchor: Heston CF = {heston_cf_price:.5f}  "
              f"(RH benchmark = {benchmark.price:.5f})")

    scheme_cfg = [
        ("volterra-euler",  32, steps_ve,    n_paths_ve),
        ("markovian-lift",  32, steps_fast,  n_paths_fast),
        ("bayer-breneis",   32, steps_fast,  n_paths_fast),
    ]
    cv_tag = "+CV" if use_cv else ""

    results: dict = {
        "benchmark_price":   benchmark.price,
        "heston_cf_price":   heston_cf_price,
        "use_cv":            use_cv,
        "hurst":             _hurst,
        "n_paths_ve":        n_paths_ve,
        "n_paths_fast":      n_paths_fast,
        "schemes":           {},
        "diagnostics":       {},
        "richardson":        {},
    }

    for scheme, n_factors, steps, n_paths in scheme_cfg:
        print(f"\n  Scheme: {scheme}{cv_tag}  "
              f"(n_paths={n_paths:,}, steps={steps})")
        print(f"  {'n_steps':>8}  {'price':>10}  {'stderr':>8}  {'|err|':>8}  {'t(s)':>7}")
        print(f"  {'-'*50}")
        prices, stderrs, errors, times, ns = [], [], [], [], []
        for n_steps in steps:
            price, stderr, elapsed = _run_one(
                scheme=scheme, n_steps=n_steps, n_paths=n_paths,
                seed=seed, use_cv=use_cv, heston_cf_price=heston_cf_price,
                n_factors=n_factors, rh_params=rh_params,
            )
            error = abs(price - benchmark.price)
            prices.append(price);   stderrs.append(stderr)
            errors.append(error);   times.append(elapsed); ns.append(n_steps)
            print(
                f"  {n_steps:>8}  {price:>10.5f}  {stderr:>8.5f}"
                f"  {error:>8.5f}  {elapsed:>7.3f}",
            )
        results["schemes"][scheme] = {
            "prices": prices, "stderrs": stderrs,
            "errors": errors, "times": times, "steps": ns,
        }

    results["diagnostics"] = compute_rh_scheme_diagnostics(results["schemes"])
    print("\n  Cross-scheme diagnostics:")
    for idx, n_steps in enumerate(results["diagnostics"]["steps"]):
        spread  = results["diagnostics"]["pairwise_spread"][idx]
        noise   = results["diagnostics"]["max_pairwise_noise"][idx]
        zscore  = results["diagnostics"]["max_pairwise_zscore"][idx]
        print(
            f"  n={n_steps:4d}  spread={spread:.5f}  "
            f"noise={noise:.5f}  spread/noise={zscore:.2f}",
        )

    results["richardson"] = _compute_richardson(results["schemes"])

    with sim_cache_path.open("w") as f:
        json.dump(results, f, indent=2)

    return results


# ===========================================================================
# Cross-scheme diagnostics
# ===========================================================================

def compute_rh_scheme_diagnostics(schemes: dict[str, dict]) -> dict[str, list]:
    """Cross-scheme price spread vs MC noise at each shared step count."""
    steps = sorted({int(s) for sd in schemes.values() for s in sd.get("steps", [])})
    pairwise_spread, max_pairwise_noise, max_pairwise_zscore, median_price = [], [], [], []

    for step in steps:
        estimates: list[tuple[str, float, float]] = []
        for sname, sd in schemes.items():
            if step not in sd.get("steps", []):
                continue
            idx = sd["steps"].index(step)
            estimates.append((sname, float(sd["prices"][idx]), float(sd["stderrs"][idx])))

        prices = [p for _, p, _ in estimates]
        pairwise_spread.append(max(prices) - min(prices))
        median_price.append(float(np.median(np.asarray(prices, dtype=float))))

        pair_noises, pair_zscores = [], []
        for li in range(len(estimates)):
            _, lp, ls = estimates[li]
            for ri in range(li + 1, len(estimates)):
                _, rp, rs = estimates[ri]
                pn = float(np.hypot(ls, rs))
                pair_noises.append(pn)
                if pn > 0.0:
                    pair_zscores.append(abs(lp - rp) / pn)

        max_pairwise_noise.append(max(pair_noises) if pair_noises else 0.0)
        max_pairwise_zscore.append(max(pair_zscores) if pair_zscores else 0.0)

    return {
        "steps":               steps,
        "pairwise_spread":     pairwise_spread,
        "max_pairwise_noise":  max_pairwise_noise,
        "max_pairwise_zscore": max_pairwise_zscore,
        "median_price":        median_price,
    }


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Rough vol convergence runner")
    ap.add_argument("--quick", action="store_true",
                    help="Quick correctness pass (5K paths, 2-3 step values)")
    ap.add_argument("--no-cv", dest="use_cv", action="store_false",
                    help="Disable Heston control variate")
    args = ap.parse_args()

    run_rough_bergomi_convergence()
    run_rough_heston_convergence(quick=args.quick, use_cv=args.use_cv)


if __name__ == "__main__":
    main()
