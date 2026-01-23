'''
Test examing that antithetic sampling reduces variation.

'''
import numpy as np

from roughvol.engines.mc import MonteCarloEngine  # adjust path to your engine V2
from roughvol.models.GBM_model import GBM_Model              # adjust module name if different
from roughvol.instruments.vanilla import VanillaOption # adjust module name if different


def test_antithetic_reduces_stderr():
    model = GBM_Model(spot0=100.0, rate=0.05, div=0.0, vol=0.2)
    inst = VanillaOption(strike=100.0, maturity=1.0, is_call=True)

    # Use even n_paths for antithetic
    n_paths = 200_000
    n_steps = 200
    seed = 123

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

    res_plain = eng_plain.price(model=model, instrument=inst)
    res_anti = eng_anti.price(model=model, instrument=inst)

    # CI width comparison is usually more stable than raw stderr comparison
    width_plain = res_plain.ci95[1] - res_plain.ci95[0]
    width_anti = res_anti.ci95[1] - res_anti.ci95[0]

    assert width_anti < width_plain, (
        f"Expected antithetic CI width smaller. "
        f"plain={width_plain}, anti={width_anti}"
    )
