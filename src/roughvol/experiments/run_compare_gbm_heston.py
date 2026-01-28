'''
Run script: pricing comparison under GBM vs Heston for vanilla + Asian options.
'''

from __future__ import annotations

from dataclasses import asdict
import numpy as np

from roughvol.engines.mc import MonteCarloEngine
from roughvol.types import MarketData

from roughvol.models.GBM_model import GBM_Model
from roughvol.models.heston_model import HestonModel

# ---- Instruments: update these imports to match your actual class names ----
from roughvol.instruments.vanilla import VanillaOption as EuropeanOption
from roughvol.instruments.asian import AsianArithmeticOption as AsianOption


def _print_result(label: str, res) -> None:
    print(f"{label}")
    print(f"  price  : {res.price:.6f}")
    print(f"  stderr : {res.stderr:.6f}")
    print(f"  ci95   : ({res.ci95[0]:.6f}, {res.ci95[1]:.6f})")
    print(f"  n_paths: {res.n_paths} | n_steps: {res.n_steps} | seed: {res.seed}")
    if res.metadata:
        # show only the key engine knobs
        keys = ["df", "antithetic", "scheme", "store_paths", "requested_n_paths", "realized_n_paths"]
        meta = {k: res.metadata[k] for k in keys if k in res.metadata}
        if meta:
            print(f"  meta   : {meta}")
    print()


def main() -> None:
    # ----------------------------
    # Market
    # ----------------------------
    market = MarketData(
        spot=100.0,
        rate=0.02,
        div_yield=0.00,
    )

    # ----------------------------
    # Instruments (vanilla + Asian)
    # ----------------------------
    T = 1.0
    K = 100.0

    euro_call = EuropeanOption(maturity=T, strike=K)
    asian_call = AsianOption(maturity=T, strike=K)

    # ----------------------------
    # Models
    # ----------------------------
    # GBM baseline
    sigma = 0.20
    gbm = GBM_Model(sigma=sigma)

    # Heston example parameters (illustrative; tune as needed)
    heston = HestonModel(
        kappa=2.0,
        theta=sigma**2,   # long-run variance roughly matches GBM variance
        xi=0.60,          # vol-of-vol
        rho=-0.70,        # leverage effect
        v0=sigma**2,      # start at long-run
    )

    # ----------------------------
    # Engine
    # ----------------------------
    # Important: antithetic=True requires even n_paths (your BM generator enforces it) :contentReference[oaicite:2]{index=2}
    engine = MonteCarloEngine(
        n_paths=200_000,
        n_steps=252,
        seed=0,
        antithetic=True,
        scheme="euler",       # GBM can ignore; Heston uses euler-ft internally in our plan
        store_paths=True,     # required for Asian
    )

    # ----------------------------
    # Run pricing
    # ----------------------------
    print("=== Pricing comparison: GBM vs Heston ===\n")
    print(f"Market: spot={market.spot}, r={market.rate}, q={market.div_yield}")
    print(f"Engine: n_paths={engine.n_paths}, n_steps={engine.n_steps}, antithetic={engine.antithetic}, seed={engine.seed}")
    print()

    # Vanilla European
    res_gbm_euro = engine.price(model=gbm, instrument=euro_call, market=market)
    res_hes_euro = engine.price(model=heston, instrument=euro_call, market=market)

    _print_result("GBM  | European Call", res_gbm_euro)
    _print_result("HES  | European Call", res_hes_euro)
    print(f"Diff (HES - GBM) European: {res_hes_euro.price - res_gbm_euro.price:.6f}\n")

    # Asian
    res_gbm_asian = engine.price(model=gbm, instrument=asian_call, market=market)
    res_hes_asian = engine.price(model=heston, instrument=asian_call, market=market)

    _print_result("GBM  | Asian Call", res_gbm_asian)
    _print_result("HES  | Asian Call", res_hes_asian)
    print(f"Diff (HES - GBM) Asian: {res_hes_asian.price - res_gbm_asian.price:.6f}\n")


if __name__ == "__main__":
    main()
