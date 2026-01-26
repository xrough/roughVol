'''
Tests for the MC engine.
'''

import numpy as np

from roughvol.engines.mc import MonteCarloEngine
from roughvol.types import PriceResult, MarketData

from roughvol.models.GBM_model import GBM_Model
from roughvol.instruments.vanilla import VanillaOption
from roughvol.analytics.black_scholes_formula import bs_price  # deterministic benchmark


def test_reproducibility_same_seed():
    market = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
    model = GBM_Model(sigma=0.2)
    inst = VanillaOption(strike=100.0, maturity=1.0, is_call=True)

    e1 = MonteCarloEngine(n_paths=200_000, n_steps=200, seed=123, antithetic=False)
    e2 = MonteCarloEngine(n_paths=200_000, n_steps=200, seed=123, antithetic=False)

    r1 = e1.price(model=model, instrument=inst, market=market)
    r2 = e2.price(model=model, instrument=inst, market=market)

    # Use exact equality if you expect bitwise determinism; allclose is safer across platforms.
    assert r1.price == r2.price
    assert r1.stderr == r2.stderr
    assert r1.ci95 == r2.ci95


def test_ci_and_stderr_sanity():
    market = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
    model = GBM_Model(sigma=0.2)
    inst = VanillaOption(strike=100.0, maturity=1.0, is_call=True)

    eng = MonteCarloEngine(n_paths=200_000, n_steps=200, seed=123, antithetic=False)
    res = eng.price(model=model, instrument=inst, market=market)

    assert isinstance(res, PriceResult)
    assert res.stderr > 0.0
    assert res.ci95[0] < res.price < res.ci95[1]

    # T=0 deterministic payoff tests
    inst_itm = VanillaOption(strike=90.0, maturity=0.0, is_call=True)
    inst_otm = VanillaOption(strike=110.0, maturity=0.0, is_call=True)

    r_itm = eng.price(model=model, instrument=inst_itm, market=market)
    r_otm = eng.price(model=model, instrument=inst_otm, market=market)

    # At maturity=0, payoff is intrinsic: max(S0-K,0) for calls
    assert r_itm.price == 10.0
    assert r_itm.stderr == 0.0
    assert r_itm.ci95 == (10.0, 10.0)

    assert r_otm.price == 0.0
    assert r_otm.stderr == 0.0
    assert r_otm.ci95 == (0.0, 0.0)


def test_convergence_stderr_shrinks():
    market = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
    model = GBM_Model(sigma=0.2)
    inst = VanillaOption(strike=100.0, maturity=1.0, is_call=True)

    Ns = [2_000, 10_000, 50_000, 200_000]
    stderrs = []
    for N in Ns:
        eng = MonteCarloEngine(n_paths=N, n_steps=200, seed=123, antithetic=False)
        stderrs.append(eng.price(model=model, instrument=inst, market=market).stderr)

    # Not strictly monotone due to randomness, but should trend down strongly.
    assert stderrs[-1] < stderrs[0]


def test_bs_price_inside_mc_ci_optional(): 
    '''
    Black–Scholes benchmark inside MC CI.
    '''
    market = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
    model = GBM_Model(sigma=0.2)
    inst = VanillaOption(strike=100.0, maturity=1.0, is_call=True)

    eng = MonteCarloEngine(n_paths=200_000, n_steps=200, seed=123, antithetic=False)
    mc = eng.price(model=model, instrument=inst, market=market)

    bs = bs_price(
        spot=market.spot,
        strike=inst.strike,
        maturity=inst.maturity,
        rate=market.rate,
        div=market.div_yield,
        vol=model.sigma,
        is_call=inst.is_call,
    )

    assert mc.ci95[0] <= bs <= mc.ci95[1]


def test_antithetic_reduces_stderr_odd_paths():
    '''
    Antithetic test robust to odd n_paths when the model uses a remainder path.

    We run multiple seeds and compare median CI width, which is stable.
    '''
    
    market = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
    model = GBM_Model(sigma=0.2)
    inst = VanillaOption(strike=100.0, maturity=1.0, is_call=True)

    n_paths = 200_001  # odd => remainder logic exercised
    n_steps = 200

    seeds = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]

    widths_plain = []
    widths_anti = []

    for seed in seeds:
        eng_plain = MonteCarloEngine(
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            antithetic=False,
        )
        eng_anti = MonteCarloEngine(
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            antithetic=True,
        )

        res_plain = eng_plain.price(model=model, instrument=inst, market=market)
        res_anti = eng_anti.price(model=model, instrument=inst, market=market)

        widths_plain.append(res_plain.ci95[1] - res_plain.ci95[0])
        widths_anti.append(res_anti.ci95[1] - res_anti.ci95[0])

    widths_plain = np.asarray(widths_plain, dtype=float)
    widths_anti = np.asarray(widths_anti, dtype=float)

    med_plain = float(np.median(widths_plain))
    med_anti = float(np.median(widths_anti))

    # Regression-style check with tiny tolerance.
    tol = 1e-12
    assert med_anti <= med_plain + tol, (
        "Expected antithetic median CI width <= plain median CI width. "
        f"median_plain={med_plain}, median_anti={med_anti}, "
        f"plain_widths={widths_plain.tolist()}, anti_widths={widths_anti.tolist()}"
    )
