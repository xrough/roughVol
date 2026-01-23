'''
蒙特卡洛引擎，加入antithetic取样的选择，以及SeedSequence。
'''
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from roughvol.types import Instrument, PathModel, PriceResult
from roughvol.sim.brownian import brownian_increments, brownian_increments_antithetic


@dataclass(frozen=True)
class MonteCarloEngine:
    '''
    Drop-in Monte Carlo engine (terminal payoff), with:
    - SeedSequence RNG discipline
    - Optional antithetic variates (if model supports simulate_paths_antithetic)    
    '''
    
    n_paths: int = 200_000
    n_steps: int = 200
    seed: int | None = 0
    antithetic: bool = True # 现在的引擎可选antithetic或者原始的mc。
    
    # 确认随机生成seed
    def _make_rng(self) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(np.random.SeedSequence(self.seed))

    def price(self, *, model: PathModel, instrument: Instrument) -> PriceResult:
        # sanity checks
        if self.n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if instrument.maturity < 0:
            raise ValueError("maturity must be non-negative")

        # Deterministic: T=0 => no simulation noise
        if instrument.maturity == 0.0:
            spot0 = float(getattr(model, "spot0", np.nan))
            if not np.isfinite(spot0):
                raise ValueError("model must expose spot0 for maturity=0 pricing.")
            spot_T = np.array([spot0], dtype=float)
            payoff0 = float(np.asarray(instrument.payoff(spot_T), dtype=float)[0])
            return PriceResult(
                price=payoff0,
                stderr=0.0,
                ci95=(payoff0, payoff0),
                n_paths=1,
                n_steps=1,
                seed=self.seed,
            )

        rng = self._make_rng()
        dt = float(instrument.maturity) / int(self.n_steps)

        # Engine-controlled increments (model is independent of antithetic)
        if self.antithetic:
            print("[MC] using antithetic variates")
            if self.n_paths % 2 != 0:
                raise ValueError("n_paths must be even when antithetic=True.")
            dW = brownian_increments_antithetic(
                n_paths=self.n_paths,
                n_steps=self.n_steps,
                dt=dt,
                rng=rng,
            )
        else:
            dW = brownian_increments(
                n_paths=self.n_paths,
                n_steps=self.n_steps,
                dt=dt,
                rng=rng,
            )

        # Model consumes dW; if your model signature does not accept dW yet, add it.
        paths = model.simulate_paths(
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            maturity=instrument.maturity,
            rng=rng,
            dW=dW,
        )

        paths = np.asarray(paths, dtype=float)
        spot_T = paths[:, -1]
        payoffs = np.asarray(instrument.payoff(spot_T), dtype=float)

        # Discounting: require model.rate (or switch to model.discount_factor)
        if not hasattr(model, "rate"):
            raise ValueError("model must expose a `rate` attribute for discounting.")
        r = float(model.rate)
        disc = float(np.exp(-r * float(instrument.maturity)))
        discounted = disc * payoffs

        n = discounted.size
        price = float(discounted.mean())

        if n < 2:
            stderr = 0.0
            ci95 = (price, price)
        else:
            std = float(discounted.std(ddof=1))
            stderr = std / np.sqrt(n)
            ci95 = (price - 1.96 * stderr, price + 1.96 * stderr)

        return PriceResult(
            price=price,
            stderr=float(stderr),
            ci95=(float(ci95[0]), float(ci95[1])),
            n_paths=n,
            n_steps=self.n_steps,
            seed=self.seed,
        )
