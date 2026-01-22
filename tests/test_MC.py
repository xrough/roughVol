'''
Tests for the MC engine.
'''

import numpy as np

from roughvol.engines.mc import MonteCarloEngine
from roughvol.types import PriceResult

# Adjust these imports to your actual module paths:
from roughvol.models.black_scholes import BlackScholesModel
from roughvol.instruments.vanilla import VanillaOption
from roughvol.analytics.black_scholes import bs_price # have to use the deterministic BS formula! 

'''
以下是seed可重复性测试函数，统计量sanity check等模型的测试函数。
'''

def test_reproducibility_same_seed():
    model = BlackScholesModel(spot0=100.0, rate=0.05, vol=0.2)
    inst = VanillaOption(strike=100.0, maturity=1.0, is_call=True)

    e1 = MonteCarloEngine(n_paths=200_000, n_steps=200, seed=123)
    e2 = MonteCarloEngine(n_paths=200_000, n_steps=200, seed=123)

    r1 = e1.price(model=model, instrument=inst)
    r2 = e2.price(model=model, instrument=inst)

    assert r1.price == r2.price
    assert r1.stderr == r2.stderr
    assert r1.ci95 == r2.ci95


def test_ci_and_stderr_sanity():
    model = BlackScholesModel(spot0=100.0, rate=0.05, vol=0.2)
    inst = VanillaOption(strike=100.0, maturity=1.0, is_call=True)
    eng = MonteCarloEngine(n_paths=200_000, n_steps=200, seed=123)

    res = eng.price(model=model, instrument=inst)
    assert isinstance(res, PriceResult)
    assert res.stderr > 0.0
    assert res.ci95[0] < res.price < res.ci95[1]


def test_T0_is_intrinsic_call():
    model = BlackScholesModel(spot0=100.0, rate=0.05, vol=0.2)

    inst_itm = VanillaOption(strike=90.0, maturity=0.0, is_call=True)
    inst_otm = VanillaOption(strike=110.0, maturity=0.0, is_call=True)

    eng = MonteCarloEngine(n_paths=200_000, n_steps=200, seed=123)

    r_itm = eng.price(model=model, instrument=inst_itm)
    r_otm = eng.price(model=model, instrument=inst_otm)

    assert r_itm.price == 10.0
    assert r_itm.stderr == 0.0
    assert r_itm.ci95 == (10.0, 10.0)

    assert r_otm.price == 0.0
    assert r_otm.stderr == 0.0
    assert r_otm.ci95 == (0.0, 0.0)


def test_convergence_stderr_shrinks():
    model = BlackScholesModel(spot0=100.0, rate=0.05, vol=0.2)
    inst = VanillaOption(strike=100.0, maturity=1.0, is_call=True)

    Ns = [2_000, 10_000, 50_000, 200_000]
    stderrs = []
    for N in Ns:
        eng = MonteCarloEngine(n_paths=N, n_steps=200, seed=123)
        stderrs.append(eng.price(model=model, instrument=inst).stderr)

    # Not strictly monotone due to randomness, but should trend down strongly.
    assert stderrs[-1] < stderrs[0]


def test_bs_price_inside_mc_ci_optional():
    """
    Optional but recommended: Black–Scholes benchmark inside MC CI.
    If your repo does not expose bs_price_vanilla, you can delete this test.
    """
    model = BlackScholesModel(spot0=100.0, rate=0.05, vol=0.2)
    inst = VanillaOption(strike=100.0, maturity=1.0, is_call=True)
    eng = MonteCarloEngine(n_paths=200_000, n_steps=200, seed=123)

    mc = eng.price(model=model, instrument=inst)
    bs = bs_price_vanilla(
        spot0=model.spot0,
        strike=inst.strike,
        maturity=inst.maturity,
        rate=model.rate,
        vol=model.vol,
        is_call=inst.is_call,
    )

    assert mc.ci95[0] <= bs <= mc.ci95[1]
