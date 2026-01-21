from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from roughvol.types import Instrument, PathModel, PriceResult


@dataclass(frozen=True)
class MonteCarloEngine:
    '''
    Monte Carlo pricer for instruments whose payoff depends on terminal spot S_T.

    Design:
    - model produces paths
    - instrument turns terminal spots into payoffs
    - engine discounts and aggregates
    '''
    n_paths: int = 200_000
    n_steps: int = 200
    seed: int | None = 0

    def price(self, *, model: PathModel, instrument: Instrument) -> PriceResult:
        # 1) RNG lives in the engine so runs are reproducible by seed
        rng = np.random.default_rng(self.seed)

        # 2) simulate full paths under the chosen model
        paths = model.simulate_paths(
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            maturity=instrument.maturity,
            rng=rng,
        )

        # 3) terminal spots (European payoff)
        spot_T = paths[:, -1] #-1指倒数第一个

        # 4) compute pathwise payoffs using the instrument 
        payoffs = instrument.payoff(spot_T) # 不需要明确class 

        # 5) discount to time 0
        # For now assume model exposes constant risk-free rate as attribute `rate`
        r = float(getattr(model, "rate", 0.0)) # getattr是自带的函数，返回model.rate若存在，反之返回0.0
        disc = np.exp(-r * instrument.maturity)

        discounted = disc * payoffs # 折损

        # 6) estimator: mean and standard error
        price = float(discounted.mean())
        stderr = float(discounted.std(ddof=1) / np.sqrt(self.n_paths)) # .std(ddof=1): sample standard deviation

        return PriceResult(
            price=price,
            stderr=stderr,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            seed=self.seed,
        )
