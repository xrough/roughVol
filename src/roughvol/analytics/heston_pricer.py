"""Standard Heston (1993) call price via characteristic function.

Used as a control-variate anchor in rough Heston convergence experiments.
The 'D2' formulation (Albrecher et al. 2007) is used to avoid branch-cut
discontinuities that affect the original Heston formula.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad


@dataclass(frozen=True)
class HestonCallPrice:
    price: float
    p1: float
    p2: float
    integration_error: float
    martingale_error: float = 0.0


def heston_log_cf(
    u: complex,
    *,
    spot: float,
    maturity: float,
    rate: float,
    div: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
) -> complex:
    """Characteristic function of log(S_T) under Heston (1993).

    Parameters
    ----------
    kappa : mean-reversion speed
    theta : long-run variance
    sigma : vol-of-vol
    rho   : spot-variance correlation
    v0    : initial variance
    """
    T = float(maturity)
    b   = kappa - rho * sigma * 1j * u
    d   = cmath.sqrt(b * b + sigma * sigma * (u * u + 1j * u))
    g   = (b - d) / (b + d)

    exp_dT          = cmath.exp(-d * T)
    one_m_g_exp_dT  = 1.0 - g * exp_dT
    one_m_g         = 1.0 - g

    C = (kappa * theta / (sigma * sigma)) * (
        (b - d) * T - 2.0 * cmath.log(one_m_g_exp_dT / one_m_g)
    )
    D = ((b - d) / (sigma * sigma)) * (1.0 - exp_dT) / one_m_g_exp_dT

    return cmath.exp(1j * u * (math.log(spot) + (rate - div) * T) + C + D * v0)


def heston_call_price(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    integration_limit: float = 200.0,
    integration_epsabs: float = 1e-10,
    integration_epsrel: float = 1e-8,
) -> HestonCallPrice:
    """Price a European call under Heston (1993) via Gil-Pelaez Fourier inversion."""
    if maturity <= 0.0:
        intrinsic = max(spot - strike, 0.0)
        return HestonCallPrice(
            price=intrinsic, p1=float(spot > strike), p2=float(spot > strike),
            integration_error=0.0,
        )

    log_K = math.log(strike)

    def phi(z: complex) -> complex:
        return heston_log_cf(
            z, spot=spot, maturity=maturity, rate=rate, div=div,
            kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0,
        )

    phi_minus_i = phi(-1j)

    def integrand_p1(x: float) -> float:
        u   = complex(x, 0.0)
        val = cmath.exp(-1j * u * log_K) * phi(u - 1j) / (1j * u * phi_minus_i)
        return val.real

    def integrand_p2(x: float) -> float:
        u   = complex(x, 0.0)
        val = cmath.exp(-1j * u * log_K) * phi(u) / (1j * u)
        return val.real

    lower = 1e-8
    p1_int, p1_err = quad(
        integrand_p1, lower, integration_limit,
        epsabs=integration_epsabs, epsrel=integration_epsrel, limit=400,
    )
    p2_int, p2_err = quad(
        integrand_p2, lower, integration_limit,
        epsabs=integration_epsabs, epsrel=integration_epsrel, limit=400,
    )

    p1      = 0.5 + p1_int / math.pi
    p2      = 0.5 + p2_int / math.pi
    forward = float(phi_minus_i.real)
    price   = math.exp(-rate * maturity) * (forward * p1 - strike * p2)

    expected_forward = spot * math.exp((rate - div) * maturity)
    martingale_error = abs(forward - expected_forward) / max(expected_forward, 1e-12)

    return HestonCallPrice(
        price=float(price),
        p1=float(p1),
        p2=float(p2),
        integration_error=float(abs(p1_err) + abs(p2_err)),
        martingale_error=float(martingale_error),
    )
