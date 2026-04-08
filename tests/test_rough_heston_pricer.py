from __future__ import annotations

import math

from roughvol.analytics.black_scholes_formula import bs_price
from roughvol.analytics.rough_heston_pricer import (
    reliable_rough_heston_call_price_cf,
    rough_heston_call_price_cf,
    solve_fractional_riccati,
)


def test_fractional_riccati_solver_returns_complex_grid():
    times, values = solve_fractional_riccati(
        0.3 + 0.0j,
        maturity=0.5,
        hurst=0.1,
        lam=0.7,
        nu=0.4,
        rho=-0.6,
        n_steps=32,
    )

    assert len(times) == 33
    assert len(values) == 33
    assert values[0] == 0.0


def test_rough_heston_cf_price_matches_black_scholes_when_variance_is_constant():
    spot = 100.0
    strike = 100.0
    maturity = 0.5
    rate = 0.03
    div = 0.01
    variance = 0.04
    volatility = math.sqrt(variance)

    benchmark = rough_heston_call_price_cf(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        div=div,
        hurst=0.1,
        lam=1.7,
        theta=variance,
        nu=0.0,
        rho=-0.6,
        v0=variance,
        riccati_steps=400,
        integration_limit=80.0,
        integration_epsabs=1e-8,
        integration_epsrel=1e-7,
    )

    bs = bs_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        div=div,
        vol=volatility,
        is_call=True,
    )

    assert abs(benchmark.price - bs) < 5e-3
    assert benchmark.integration_error >= 0.0
    assert benchmark.martingale_error < 5e-4


def test_reliable_rough_heston_benchmark_runs_stability_checks():
    benchmark = reliable_rough_heston_call_price_cf(
        spot=100.0,
        strike=100.0,
        maturity=0.25,
        rate=0.02,
        div=0.0,
        hurst=0.1,
        lam=1.5,
        theta=0.04,
        nu=0.0,
        rho=-0.5,
        v0=0.04,
        riccati_steps_grid=(200, 300),
        integration_limits=(60.0, 80.0),
        martingale_tol=2e-3,
        stability_tol=2e-2,
    )

    assert benchmark.martingale_error < 2e-3
    assert benchmark.stability_error < 2e-2
