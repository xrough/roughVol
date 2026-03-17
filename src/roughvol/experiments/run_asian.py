'''
Example script to run a GBM Monte Carlo surface experiment 
with prescribed market parameters and Asian Call.
'''

from __future__ import annotations

import argparse
import numpy as np

from roughvol.models.GBM_model import GBM_Model
from roughvol.instruments.asian import AsianArithmeticOption
from roughvol.engines.mc import MonteCarloEngine

from roughvol.types import MarketData  
from roughvol.analytics.black_scholes_formula import implied_vol

def main() -> None:
    ap = argparse.ArgumentParser(description="GBM Monte Carlo surface.")
    # market arguments
    ap.add_argument("--spot", type=float, default=100.0) # spot_0
    ap.add_argument("--rate", type=float, default=0.02)
    ap.add_argument("--div", type=float, default=0.00)
    
    # model arguments
    ap.add_argument("--sigma", type=float, default=0.20)
    
    # engine arguments
    ap.add_argument("--n_paths", type=int, default=200_000)
    ap.add_argument("--n_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--antithetic", action="store_true") # careful with bool type in argparse!
    
    # instrument arguments
    ap.add_argument("--maturity", type=float, default=1.0)
    ap.add_argument("--is_call", type=bool, default=True)
    ap.add_argument("--strike", type=float, nargs="+", default=[100.0,110.0,120.0])

    args = ap.parse_args()

    # Market inputs now live in MarketData
    market = MarketData(
        spot=args.spot,
        rate=args.rate,
        div_yield=args.div,
    )
    
    # Model now holds only the dynamics parameter(s)
    model = GBM_Model(sigma=args.sigma)

    # Engine controls simulation config defaults (n_paths, n_steps, seed, antithetic)
    engine = MonteCarloEngine(
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        seed=args.seed,
        antithetic=args.antithetic,
    )

    print("strike, price, stderr")
    for k in args.strike:
        opt = AsianArithmeticOption(maturity=args.maturity, strike=k, callput="call" if args.is_call else "put")

        # pass model, instrument and market into engine now -> result.
        res = engine.price(model=model, instrument=opt, market=market)

        print(f"{k:8.3f}, {res.price:12.6f}, {res.stderr:10.6f}")

if __name__ == "__main__":
    main()
