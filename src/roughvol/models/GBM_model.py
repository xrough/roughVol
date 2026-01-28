'''
建模underlying asset的随机过程，随机性由MC引擎传导，随机元素的生成单独包含在sim中。
'''

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from roughvol.sim.brownian import brownian_increments # generate paths of BMs with given increment 
from roughvol.types import ArrayF, PathBundle, MarketData, SimConfig

# 由于包含simulate_paths这个函数，GBM_Model属于PathModel这个Class。

@dataclass(frozen=True)
class GBM_Model: 
    '''
    Geometric Brownian Motion (risk-neutral), contract: PathModel.
    - sigma: constant volatility coefficient.
    - simulate_paths: simulate paths with input from market and sim config and output PathBundle.
    '''
    
    # Prescribed constant multiplicative constant of the diffusion.
    sigma: float 
    
    def simulate_paths( # Method prescribed by PathModel Protocol.
        self,
        *,
        market: MarketData,
        sim: SimConfig,
        rng: np.random.Generator,
    ) -> PathBundle:
        # ---- Validate model parameter ----
        sigma = float(self.sigma) # read sigma from the model.
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative.")

        # ---- Read market inputs ----
        spot0 = float(market.spot)
        r = float(market.rate)
        q = float(market.div_yield)

        if spot0 <= 0.0:
            raise ValueError("market.spot must be positive.")
        
        # ---- Build time grid from the configuration: SimConfig ----
        t = np.asarray(sim.grid(), dtype=float)  # shape (n_times,)
        n_times = int(t.size)
        if n_times < 1:
            raise ValueError("sim.grid() must return a non-empty time grid.")
        
        # ---- Read simulation controls (number of paths) ----
        n_paths = int(sim.n_paths)
        if n_paths <= 0:
            raise ValueError("sim.n_paths must be positive.")

        # If grid has only time 0 (maturity=0 case)
        if n_times == 1:
            spot = np.full((n_paths, 1), spot0, dtype=float)
            return PathBundle(
                t=t,
                state={"spot": spot},
                metadata={"model": "GBM", "scheme": "exact"},
            )

        dt = np.diff(t)  # shape (n_times-1,)
        if np.any(dt < 0.0):
            raise ValueError("Time grid must be increasing (dt >= 0).")

        n_steps = n_times - 1 # increment number = grid points - 1.
        
        # ---- Generate Brownian increments dW ----
        # Convention: brownian_increments returns sqrt(dt) * Z, so diffusion is sigma * dW.
        use_antithetic = bool(getattr(sim, "antithetic", False))

        dW = np.empty((n_paths, n_steps), dtype=float)
        
        if use_antithetic and (n_paths % 2 != 0):
            raise ValueError("n_paths must be even when antithetic=True.")

        if np.allclose(dt, dt[0]):
            dW[:, :] = brownian_increments(
                n_paths=n_paths,
                n_steps=n_steps,
                dt=float(dt[0]),
                rng=rng,
                antithetic=use_antithetic,
            )
        else:
            for j in range(n_steps):
                dW[:, j:j+1] = brownian_increments(
                    n_paths=n_paths,
                    n_steps=1,
                    dt=float(dt[j]),
                    rng=rng,
                    antithetic=use_antithetic,
                )

        # ---- Simulate GBM paths (exact log scheme) ----
        spot = np.empty((n_paths, n_times), dtype=float)
        spot[:, 0] = spot0

        for j in range(n_steps):
            dt_j = float(dt[j])
            drift = (r - q - 0.5 * sigma * sigma) * dt_j
            diffusion = sigma * dW[:, j]  # dW already includes sqrt(dt_j)
            spot[:, j + 1] = spot[:, j] * np.exp(drift + diffusion)

        # ---- Package into PathBundle ----
        return PathBundle(
            t=t,
            state={"spot": spot},
            extras={"dW": dW}, # include the Brownian increments for calibrations. 
            metadata={
                "model": "GBM",
                "scheme": "exact",
                "sigma": sigma,
                "antithetic": use_antithetic,
            },
        )

