'''
Test examing that antithetic sampling reduces variation.

'''
import numpy as np

from roughvol.engines.mc import MonteCarloEngine  # adjust path to your engine V2
from roughvol.models.GBM_model import GBM_Model              # adjust module name if different
from roughvol.instruments.vanilla import VanillaOption # adjust module name if different


import numpy as np

from roughvol.engines.mc import MonteCarloEngine
from roughvol.models.GBM_model import GBM_Model
from roughvol.instruments.vanilla import VanillaOption
from roughvol.types import MarketData


def test_antithetic_reduces_stderr_odd_paths():
    market = MarketData(spot=100.0, rate=0.05, div_yield=0.0)
    model = GBM_Model(sigma=0.2)
    inst = VanillaOption(strike=100.0, maturity=1.0, is_call=True)

    # Odd number of paths to exercise "remainder" logic inside the model
    n_paths = 200_001
    n_steps = 200

    # Multiple seeds => robustness against Monte Carlo noise
    seeds = [11, 22, 33, 44, 55, 66, 77]

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

    # Robust comparison: median width
    med_plain = float(np.median(widths_plain))
    med_anti = float(np.median(widths_anti))

    # Allow a small numerical tolerance; we mainly want to catch regressions
    tol = 1e-12
    assert med_anti <= med_plain + tol, (
        "Expected antithetic median CI width <= plain median CI width. "
        f"median_plain={med_plain}, median_anti={med_anti}, "
        f"plain_widths={widths_plain.tolist()}, anti_widths={widths_anti.tolist()}"
    )