'''
Example script to run a GBM Monte Carlo surface experiment with prescribed market parameters.
'''

from __future__ import annotations

import argparse
import numpy as np

from roughvol.models.GBM_model import GBM_Model
from roughvol.instruments.vanilla import VanillaOption
from roughvol.engines.mc import MonteCarloEngine

from roughvol.types import MarketData  # NEW
from roughvol.analytics.black_scholes_formula import implied_vol


def parse_strikes(raw: str) -> list[float]:
    strikes: list[float] = []
    parts = raw.replace(",", " ").split()

    for p in parts:
        if ":" in p:
            # Range format: start:end:step
            try:
                start, end, step = map(float, p.split(":"))
                x = start
                while x <= end + 1e-12:
                    strikes.append(round(x, 10))
                    x += step
            except Exception:
                raise ValueError(f"Invalid range format: {p}")
        else:
            strikes.append(float(p))

    return sorted(set(strikes))


def get_user_strikes() -> list[float]:
    while True:
        raw = input("\nEnter strikes (e.g. 80 90 100 or 80:120:5):\n> ")
        try:
            strikes = parse_strikes(raw)
            if len(strikes) == 0:
                raise ValueError("No strikes entered.")
            return strikes
        except Exception as e:
            print("Invalid input:", e)


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
    strikes = get_user_strikes()
    
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

    print("strike, price, stderr, implied_vol")
    for k in strikes:
        opt = VanillaOption(strike=k, maturity=args.maturity, is_call=args.is_call)

        # pass market into engine now -> result.
        res = engine.price(model=model, instrument=opt, market=market)

        iv = implied_vol(
            price=res.price,
            spot=args.spot,
            strike=k,
            maturity=args.maturity,
            rate=args.rate,
            div=args.div,
            is_call=True,
        )

        print(f"{k:8.3f}, {res.price:12.6f}, {res.stderr:10.6f}, {iv:10.6f}")


if __name__ == "__main__":
    main()
