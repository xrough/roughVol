from __future__ import annotations

import numpy as np
import pytest

from roughvol.analytics.black_scholes_formula import bs_price
from roughvol.engines.mc import MonteCarloEngine
from roughvol.instruments.vanilla import VanillaOption
from roughvol.models.rough_heston_model import RoughHestonModel
from roughvol.types import MarketData, SimConfig, make_rng


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _market() -> MarketData:
    return MarketData(spot=100.0, rate=0.0, div_yield=0.0)


def _base_model(scheme: str = "volterra-euler", n_factors: int = 8) -> RoughHestonModel:
    return RoughHestonModel(
        hurst=0.1,
        lam=0.3,
        theta=0.04,
        nu=0.5,
        rho=-0.7,
        v0=0.04,
        scheme=scheme,
        n_factors=n_factors,
    )


# ---------------------------------------------------------------------------
# Test 1: Path shapes are correct for both schemes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["volterra-euler", "markovian-lift"])
def test_rough_heston_path_shapes(scheme):
    model = _base_model(scheme)
    market = _market()
    sim = SimConfig(n_paths=8, maturity=1.0, n_steps=6, seed=42, antithetic=False)
    paths = model.simulate_paths(market=market, sim=sim, rng=make_rng(sim.seed))

    assert paths.spot.shape == (8, 7), f"spot shape wrong: {paths.spot.shape}"
    assert paths.get("var").shape == (8, 7), f"var shape wrong: {paths.get('var').shape}"
    assert np.all(paths.spot > 0.0), "spot paths must be positive"
    assert np.all(paths.get("var") >= 0.0), "variance paths must be non-negative"


# ---------------------------------------------------------------------------
# Test 2: Trivial grid (n_times == 1) returns correct shapes
# ---------------------------------------------------------------------------

def test_rough_heston_single_time_point():
    model = _base_model()
    market = _market()
    sim = SimConfig(n_paths=4, maturity=1.0, n_steps=0, seed=1, antithetic=False)
    paths = model.simulate_paths(market=market, sim=sim, rng=make_rng(sim.seed))
    assert paths.spot.shape == (4, 1)
    assert np.allclose(paths.spot[:, 0], 100.0)


# ---------------------------------------------------------------------------
# Test 3: nu=0 → deterministic vol → BS price inside 95% CI
# ---------------------------------------------------------------------------

def test_rough_heston_nu_zero_matches_black_scholes():
    """With nu=0, variance is deterministic: V_t = v0 for all t (lam=0 case).

    When lam=0 and nu=0, V_t = v0 + 0 = v0 everywhere (no mean reversion, no
    diffusion), so the stock is just GBM with vol = sqrt(v0).
    """
    v0 = 0.04
    sigma = np.sqrt(v0)
    model = RoughHestonModel(
        hurst=0.1, lam=0.0, theta=v0, nu=0.0, rho=0.0, v0=v0,
        scheme="volterra-euler",
    )
    market = MarketData(spot=100.0, rate=0.02, div_yield=0.0)
    instrument = VanillaOption(strike=100.0, maturity=1.0, is_call=True)
    engine = MonteCarloEngine(n_paths=40_000, n_steps=64, seed=99, antithetic=True)

    mc = engine.price(model=model, instrument=instrument, market=market)
    bs = bs_price(
        spot=market.spot,
        strike=instrument.strike,
        maturity=instrument.maturity,
        rate=market.rate,
        div=market.div_yield,
        vol=sigma,
        is_call=instrument.is_call,
    )

    assert mc.ci95[0] <= bs <= mc.ci95[1], (
        f"BS price {bs:.6f} not in 95% CI [{mc.ci95[0]:.6f}, {mc.ci95[1]:.6f}]"
    )


# ---------------------------------------------------------------------------
# Test 4: Markovian lift prices agree with volterra-euler within 2σ
# ---------------------------------------------------------------------------

def test_rough_heston_markovian_lift_consistent_with_euler():
    """Both schemes should give consistent ATM call prices within 2 standard errors."""
    market = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
    instrument = VanillaOption(strike=100.0, maturity=0.5, is_call=True)

    engine = MonteCarloEngine(n_paths=20_000, n_steps=64, seed=7, antithetic=True)

    model_euler = _base_model("volterra-euler")
    model_lift = _base_model("markovian-lift", n_factors=8)

    res_euler = engine.price(model=model_euler, instrument=instrument, market=market)
    res_lift = engine.price(model=model_lift, instrument=instrument, market=market)

    # 2-sigma combined confidence: price difference should be < 2*(se1 + se2)
    combined_se = 2.0 * (res_euler.stderr + res_lift.stderr)
    diff = abs(res_lift.price - res_euler.price)
    assert diff <= combined_se, (
        f"Prices differ by {diff:.5f}, which exceeds 2*(se_euler+se_lift)={combined_se:.5f}. "
        f"Euler: {res_euler.price:.5f} ± {res_euler.stderr:.5f}, "
        f"Lift: {res_lift.price:.5f} ± {res_lift.stderr:.5f}"
    )


# ---------------------------------------------------------------------------
# Test 5: Invalid parameters raise ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_kwargs,match", [
    ({"hurst": 0.6}, "hurst"),
    ({"hurst": 0.0}, "hurst"),
    ({"lam": -0.1}, "lam"),
    ({"nu": -1.0}, "nu"),
    ({"v0": -0.01}, "v0"),
    ({"rho": 1.5}, "rho"),
    ({"scheme": "unknown-scheme"}, "scheme"),
])
def test_rough_heston_invalid_params_raise(bad_kwargs, match):
    kwargs = dict(hurst=0.1, lam=0.3, theta=0.04, nu=0.5, rho=-0.7, v0=0.04)
    kwargs.update(bad_kwargs)
    model = RoughHestonModel(**kwargs)
    sim = SimConfig(n_paths=4, maturity=1.0, n_steps=4, seed=0, antithetic=False)
    with pytest.raises(ValueError, match=match):
        model.simulate_paths(market=_market(), sim=sim, rng=make_rng(0))
