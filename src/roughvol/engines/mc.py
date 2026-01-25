'''
MC engine for possibly path-dependent instruments and various models.

Key contracts (see roughvol.types):
- model.simulate_paths(market=..., sim=..., rng=...) -> PathBundle
- payoff resolved via compute_payoff(instrument, paths)
- discounting via MarketData.rate and flat_discount_factor
'''
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


from roughvol.types import (
    Instrument,
    TerminalInstrument,
    PathModel,
    MarketData,
    SimConfig,
    PriceResult,
    PathBundle,
    ArrayF,
    compute_payoff,
    make_rng,
    flat_discount_factor,
)



@dataclass(frozen=True)
class MonteCarloEngine:
    '''
    Monte Carlo engine (PathBundle-native).

    Notes:
    - n_paths/n_steps/seed/antithetic are engine defaults.
      The engine packages them into SimConfig and passes to the model.
    - Antithetic handling is a model responsibility via sim.antithetic. 
    '''
    
    n_paths: int = 200_000
    n_steps: int = 200
    seed: int | None = 0
    antithetic: bool = True
    scheme: str = "euler"
    store_paths: bool = True

    def price(
        self,
        *,
        model: PathModel,
        instrument: Instrument | TerminalInstrument,
        market: MarketData,
    ) -> PriceResult:
        # --- sanity checks (engine-level) ---
        if self.n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if instrument.maturity < 0:
            raise ValueError("instrument.maturity must be non-negative")
        if market.spot <= 0:
            raise ValueError("market.spot must be positive")

        # --- Build simulation config (the model consumes this) ---
        sim = SimConfig(
            n_paths=int(self.n_paths),
            maturity=float(instrument.maturity),
            n_steps=int(self.n_steps),
            seed=self.seed,
            antithetic=bool(self.antithetic),
            scheme=str(self.scheme),
            store_paths=bool(self.store_paths),
        )  # SimConfig.grid() defined in types.py 

        rng = make_rng(sim.seed)  # standardized RNG helper 

        # --- Simulate paths ---
        paths: PathBundle = model.simulate_paths(market=market, sim=sim, rng=rng)  

        # --- Compute payoff (supports path-dependent + terminal-only legacy) ---
        payoff: ArrayF = compute_payoff(instrument, paths)  
        payoff = np.asarray(payoff, dtype=float).reshape(-1)

        # --- Discount ---
        df = flat_discount_factor(float(market.rate), float(instrument.maturity))  # 
        pv = df * payoff

        # --- MC stats ---
        n = int(pv.size)
        if n == 0:
            raise ValueError("No payoffs produced (n_paths=0?)")

        price = float(pv.mean())
        if n > 1:
            stderr = float(pv.std(ddof=1) / np.sqrt(n))
            ci95 = (price - 1.96 * stderr, price + 1.96 * stderr)
        else:
            stderr = 0.0
            ci95 = (price, price)

        return PriceResult(
            price=price,
            stderr=stderr,
            ci95=(float(ci95[0]), float(ci95[1])),
            n_paths=n,
            n_steps=int(paths.n_times - 1),  # realized from PathBundle grid 
            seed=self.seed,
            metadata={
                "df": df,
                "antithetic": bool(sim.antithetic),
                "scheme": sim.scheme,
                "store_paths": bool(sim.store_paths),
            },
        )