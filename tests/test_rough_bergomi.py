from __future__ import annotations

import numpy as np

from roughvol.analytics.black_scholes_formula import bs_price
from roughvol.engines.mc import MonteCarloEngine
from roughvol.instruments.vanilla import VanillaOption
from roughvol.models.rough_bergomi_model import RoughBergomiModel
from roughvol.types import MarketData, SimConfig, make_rng


def test_rough_bergomi_paths_include_variance_and_driver_state():
    market = MarketData(
        spot=100.0,
        rate=0.0,
        div_yield=0.0,
        forward_variance_curve=lambda t: np.full_like(t, 0.09, dtype=float),
    )
    model = RoughBergomiModel(hurst=0.1, eta=0.0, rho=-0.7, xi0=0.04)
    sim = SimConfig(n_paths=6, maturity=1.0, n_steps=4, seed=7, antithetic=False)

    paths = model.simulate_paths(market=market, sim=sim, rng=make_rng(sim.seed))

    assert paths.spot.shape == (6, 5)
    assert paths.get("var").shape == (6, 5)
    assert paths.get("Y").shape == (6, 5)
    assert np.allclose(paths.get("var"), 0.09)


def test_rough_bergomi_eta_zero_matches_black_scholes_limit():
    market = MarketData(spot=100.0, rate=0.01, div_yield=0.0)
    model = RoughBergomiModel(hurst=0.1, eta=0.0, rho=-0.8, xi0=0.04)
    instrument = VanillaOption(strike=100.0, maturity=1.0, is_call=True)
    engine = MonteCarloEngine(
        n_paths=40_000,
        n_steps=64,
        seed=123,
        antithetic=True,
    )

    mc = engine.price(model=model, instrument=instrument, market=market)
    bs = bs_price(
        spot=market.spot,
        strike=instrument.strike,
        maturity=instrument.maturity,
        rate=market.rate,
        div=market.div_yield,
        vol=np.sqrt(model.xi0),
        is_call=instrument.is_call,
    )

    assert mc.ci95[0] <= bs <= mc.ci95[1]
