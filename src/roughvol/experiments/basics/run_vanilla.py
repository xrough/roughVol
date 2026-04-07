"""Console-based vanilla pricing sanity check."""

from __future__ import annotations

import argparse

from roughvol.analytics.black_scholes_formula import implied_vol
from roughvol.engines.mc import MonteCarloEngine
from roughvol.instruments.vanilla import VanillaOption
from roughvol.models.GBM_model import GBM_Model
from roughvol.types import MarketData


def parse_strikes(raw: str) -> list[float]:
    strikes: list[float] = []
    for part in raw.replace(",", " ").split():
        if ":" in part:
            start, end, step = map(float, part.split(":"))
            value = start
            while value <= end + 1e-12:
                strikes.append(round(value, 10))
                value += step
        else:
            strikes.append(float(part))
    return sorted(set(strikes))


def get_user_strikes() -> list[float]:
    while True:
        raw = input("\nEnter strikes (e.g. 80 90 100 or 80:120:5):\n> ")
        try:
            strikes = parse_strikes(raw)
        except Exception as exc:
            print("Invalid input:", exc)
            continue
        if strikes:
            return strikes
        print("Invalid input: no strikes entered.")


def main() -> None:
    parser = argparse.ArgumentParser(description="GBM Monte Carlo surface.")
    parser.add_argument("--spot", type=float, default=100.0)
    parser.add_argument("--rate", type=float, default=0.02)
    parser.add_argument("--div", type=float, default=0.00)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--n_paths", type=int, default=200_000)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--antithetic", action="store_true")
    parser.add_argument("--maturity", type=float, default=1.0)
    parser.add_argument("--is_call", type=bool, default=True)

    strikes = get_user_strikes()
    args = parser.parse_args()

    market = MarketData(spot=args.spot, rate=args.rate, div_yield=args.div)
    model = GBM_Model(sigma=args.sigma)
    engine = MonteCarloEngine(
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        seed=args.seed,
        antithetic=args.antithetic,
    )

    print("strike, price, stderr, implied_vol")
    for strike in strikes:
        option = VanillaOption(strike=strike, maturity=args.maturity, is_call=args.is_call)
        result = engine.price(model=model, instrument=option, market=market)
        iv = implied_vol(
            price=result.price,
            spot=args.spot,
            strike=strike,
            maturity=args.maturity,
            rate=args.rate,
            div=args.div,
            is_call=True,
        )
        print(f"{strike:8.3f}, {result.price:12.6f}, {result.stderr:10.6f}, {iv:10.6f}")


if __name__ == "__main__":
    main()
