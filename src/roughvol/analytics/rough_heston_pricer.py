"""Characteristic-function pricing for the rough Heston model.

Implements the El Euch-Rosenbaum fractional Riccati representation and solves
it numerically with the predictor-corrector fractional Adams method described
in Section 5 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
import cmath
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


@dataclass(frozen=True)
class RoughHestonBenchmarkPrice:
    """Fourier benchmark price for a European option under rough Heston."""

    price: float
    p1: float
    p2: float
    integration_error: float
    martingale_error: float = 0.0
    stability_error: float = 0.0
    riccati_steps: int = 0
    integration_limit: float = 0.0


def rough_heston_fractional_riccati_rhs(
    u: complex,
    h: complex,
    *,
    lam: float,
    nu: float,
    rho: float,
) -> complex:
    """Quadratic nonlinearity in the El Euch-Rosenbaum fractional Riccati equation."""
    # El Euch & Rosenbaum (2019) eq. (2.5):
    # F(u, h) = -u²/2 - iu/2 + (iρνu − λ)·h + ν²/2 · h²
    return (
        0.5 * (-u * u - 1j * u)
        + (1j * u * rho * nu - lam) * h
        + 0.5 * nu ** 2 * h * h
    )


def solve_fractional_riccati(
    u: complex,
    *,
    maturity: float,
    hurst: float,
    lam: float,
    nu: float,
    rho: float,
    n_steps: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the fractional Riccati equation on a uniform grid."""
    if maturity <= 0.0:
        return np.array([0.0], dtype=float), np.array([0.0 + 0.0j], dtype=complex)
    if not (0.0 < hurst < 0.5):
        raise ValueError("hurst must lie in (0, 0.5)")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    alpha = hurst + 0.5
    delta = maturity / n_steps
    times = np.linspace(0.0, maturity, n_steps + 1)
    values = np.zeros(n_steps + 1, dtype=complex)
    rhs_values = np.zeros(n_steps + 1, dtype=complex)

    gamma_alpha_1 = gamma(alpha + 1.0)
    gamma_alpha_2 = gamma(alpha + 2.0)
    delta_alpha = delta ** alpha

    for k in range(n_steps):
        predictor = 0.0 + 0.0j
        for j in range(k + 1):
            b = (
                delta_alpha
                / gamma_alpha_1
                * ((k - j + 1) ** alpha - (k - j) ** alpha)
            )
            predictor += b * rhs_values[j]

        corrected = 0.0 + 0.0j
        a0 = (
            delta_alpha
            / gamma_alpha_2
            * (k ** (alpha + 1.0) - (k - alpha) * (k + 1) ** alpha)
        )
        corrected += a0 * rhs_values[0]
        for j in range(1, k + 1):
            aj = (
                delta_alpha
                / gamma_alpha_2
                * (
                    (k - j + 2) ** (alpha + 1.0)
                    + (k - j) ** (alpha + 1.0)
                    - 2.0 * (k - j + 1) ** (alpha + 1.0)
                )
            )
            corrected += aj * rhs_values[j]

        a_last = delta_alpha / gamma_alpha_2
        values[k + 1] = corrected + a_last * rough_heston_fractional_riccati_rhs(
            u,
            predictor,
            lam=lam,
            nu=nu,
            rho=rho,
        )
        if not cmath.isfinite(values[k + 1]):
            values[k + 1 :] = complex(float("nan"), float("nan"))
            break
        rhs_values[k + 1] = rough_heston_fractional_riccati_rhs(
            u,
            values[k + 1],
            lam=lam,
            nu=nu,
            rho=rho,
        )

    return times, values


def _fractional_trapezoid_weights(order: float, n_steps: int) -> np.ndarray:
    """Fractional Adams trapezoid weights for I^order f(T) on a uniform grid."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    k = n_steps - 1
    weights = np.zeros(n_steps + 1, dtype=float)
    weights[0] = (
        k ** (order + 1.0) - (k - order) * (k + 1) ** order
    )
    for j in range(1, k + 1):
        weights[j] = (
            (k - j + 2) ** (order + 1.0)
            + (k - j) ** (order + 1.0)
            - 2.0 * (k - j + 1) ** (order + 1.0)
        )
    weights[-1] = 1.0
    return weights


def _fractional_integral_terminal(
    values: np.ndarray,
    *,
    maturity: float,
    order: float,
) -> complex:
    """Compute I^order f(T) from grid values using fractional Adams trapezoid weights."""
    if maturity <= 0.0:
        return 0.0 + 0.0j
    if not (0.0 < order <= 1.0):
        raise ValueError("order must lie in (0, 1]")

    n_steps = len(values) - 1
    delta = maturity / n_steps
    weights = _fractional_trapezoid_weights(order, n_steps)
    return delta**order / gamma(order + 2.0) * np.sum(weights * values)


def rough_heston_log_price_cf(
    u: complex,
    *,
    spot: float,
    maturity: float,
    rate: float,
    div: float,
    hurst: float,
    lam: float,
    theta: float,
    nu: float,
    rho: float,
    v0: float,
    riccati_steps: int = 400,
) -> complex:
    """Characteristic function of log S_T under rough Heston."""
    if spot <= 0.0:
        raise ValueError("spot must be positive")

    times, h_values = solve_fractional_riccati(
        u,
        maturity=maturity,
        hurst=hurst,
        lam=lam,
        nu=nu,
        rho=rho,
        n_steps=riccati_steps,
    )
    i1h = _fractional_integral_terminal(h_values, maturity=maturity, order=1.0)
    i1_minus_alpha_h = _fractional_integral_terminal(
        h_values,
        maturity=maturity,
        order=0.5 - hurst,
    )
    log_return_cf = np.exp(theta * lam * i1h + v0 * i1_minus_alpha_h)
    drift_cf = np.exp(1j * u * (math.log(spot) + (rate - div) * maturity))
    return drift_cf * log_return_cf


def rough_heston_call_price_cf(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div: float,
    hurst: float,
    lam: float,
    theta: float,
    nu: float,
    rho: float,
    v0: float,
    riccati_steps: int = 400,
    integration_limit: float = 150.0,
    integration_epsabs: float = 1e-8,
    integration_epsrel: float = 1e-6,
) -> RoughHestonBenchmarkPrice:
    """Price a European call via Fourier inversion of the rough Heston characteristic function."""
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if maturity <= 0.0:
        intrinsic = max(spot - strike, 0.0)
        return RoughHestonBenchmarkPrice(price=intrinsic, p1=float(spot > strike), p2=float(spot > strike), integration_error=0.0)

    log_strike = math.log(strike)
    def phi(z: complex) -> complex:
        return rough_heston_log_price_cf(
            z,
            spot=spot,
            maturity=maturity,
            rate=rate,
            div=div,
            hurst=hurst,
            lam=lam,
            theta=theta,
            nu=nu,
            rho=rho,
            v0=v0,
            riccati_steps=riccati_steps,
        )

    phi_minus_i = phi(-1j)

    def integrand_p1(x: float) -> float:
        try:
            u = complex(x, 0.0)
            value = np.exp(-1j * u * log_strike) * phi(u - 1j) / (1j * u * phi_minus_i)
            result = float(np.real(value))
            return result if math.isfinite(result) else 0.0
        except (OverflowError, ZeroDivisionError):
            return 0.0

    def integrand_p2(x: float) -> float:
        try:
            u = complex(x, 0.0)
            value = np.exp(-1j * u * log_strike) * phi(u) / (1j * u)
            result = float(np.real(value))
            return result if math.isfinite(result) else 0.0
        except (OverflowError, ZeroDivisionError):
            return 0.0

    lower = 1e-8
    p1_int, p1_err = quad(integrand_p1, lower, integration_limit, epsabs=integration_epsabs, epsrel=integration_epsrel, limit=400)
    p2_int, p2_err = quad(integrand_p2, lower, integration_limit, epsabs=integration_epsabs, epsrel=integration_epsrel, limit=400)

    p1 = 0.5 + p1_int / math.pi
    p2 = 0.5 + p2_int / math.pi
    forward = float(np.real(phi_minus_i))
    price = math.exp(-rate * maturity) * (forward * p1 - strike * p2)
    expected_forward = spot * math.exp((rate - div) * maturity)
    martingale_error = abs(forward - expected_forward) / max(expected_forward, 1e-12)
    return RoughHestonBenchmarkPrice(
        price=float(np.real(price)),
        p1=float(np.real(p1)),
        p2=float(np.real(p2)),
        integration_error=float(abs(p1_err) + abs(p2_err)),
        martingale_error=float(martingale_error),
        riccati_steps=riccati_steps,
        integration_limit=integration_limit,
    )


def reliable_rough_heston_call_price_cf(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div: float,
    hurst: float,
    lam: float,
    theta: float,
    nu: float,
    rho: float,
    v0: float,
    riccati_steps_grid: tuple[int, ...] = (400, 600),
    integration_limits: tuple[float, ...] = (100.0, 150.0),
    integration_epsabs: float = 1e-8,
    integration_epsrel: float = 1e-6,
    martingale_tol: float = 5e-4,
    stability_tol: float = 5e-3,
) -> RoughHestonBenchmarkPrice:
    """Compute a benchmark price and verify that it is numerically stable."""
    candidates: list[RoughHestonBenchmarkPrice] = []
    seen: set[tuple[int, float]] = set()
    evaluation_grid = [
        (riccati_steps_grid[0], integration_limits[0]),
        (riccati_steps_grid[-1], integration_limits[0]),
        (riccati_steps_grid[-1], integration_limits[-1]),
    ]
    for riccati_steps, integration_limit in evaluation_grid:
        key = (riccati_steps, integration_limit)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            rough_heston_call_price_cf(
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                div=div,
                hurst=hurst,
                lam=lam,
                theta=theta,
                nu=nu,
                rho=rho,
                v0=v0,
                riccati_steps=riccati_steps,
                integration_limit=integration_limit,
                integration_epsabs=integration_epsabs,
                integration_epsrel=integration_epsrel,
            )
        )

    benchmark = candidates[-1]
    stability_error = max(abs(candidate.price - benchmark.price) for candidate in candidates)
    benchmark = RoughHestonBenchmarkPrice(
        price=benchmark.price,
        p1=benchmark.p1,
        p2=benchmark.p2,
        integration_error=benchmark.integration_error,
        martingale_error=benchmark.martingale_error,
        stability_error=float(stability_error),
        riccati_steps=benchmark.riccati_steps,
        integration_limit=benchmark.integration_limit,
    )

    if benchmark.martingale_error > martingale_tol:
        raise RuntimeError(
            f"Rough Heston benchmark failed martingale check: "
            f"{benchmark.martingale_error:.3e} > {martingale_tol:.3e}",
        )
    if benchmark.stability_error > stability_tol:
        raise RuntimeError(
            f"Rough Heston benchmark failed stability check: "
            f"{benchmark.stability_error:.3e} > {stability_tol:.3e}",
        )
    return benchmark
