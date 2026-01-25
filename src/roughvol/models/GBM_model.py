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
    Geometric Brownian Motion (risk-neutral):
        dS_t = (r - q) S_t dt + sigma S_t dW_t

    Model parameter:
        sigma: constant volatility coefficient.

    MarketData provides:
        spot, rate, div_yield
    
    SimConfig provides:
        n_paths, grid(), (optionally) antithetic
    '''
    sigma: float # Prescribed constant multiplicative constant of the diffusion.
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
        if np.any(dt <= 0.0):
            raise ValueError("Time grid must be strictly increasing (dt > 0).")

        n_steps = n_times - 1 # increment number = grid points - 1.
        
        # ---- Generate Brownian increments dW ----
        # Convention: brownian_increments returns sqrt(dt) * Z, so diffusion is sigma * dW.
        # Handle (optional) antithetic variates if sim.antithetic exists.
        use_antithetic = bool(getattr(sim, "antithetic", False))

        dW = np.empty((n_paths, n_steps), dtype=float)

        if not use_antithetic:
            # If dt is constant, one call is enough; otherwise generate stepwise.
            if np.allclose(dt, dt[0]):
                dW[:, :] = brownian_increments(
                    n_paths=n_paths, n_steps=n_steps, dt=float(dt[0]), rng=rng
                )
            else:
                for j in range(n_steps):
                    dW[:, j:j+1] = brownian_increments(
                        n_paths=n_paths, n_steps=1, dt=float(dt[j]), rng=rng
                    )
        else:
            # Antithetic: generate half and mirror. If odd n_paths, add one extra path.
            half = n_paths // 2
            remainder = n_paths - 2 * half

            if np.allclose(dt, dt[0]):
                dW_half = brownian_increments(
                    n_paths=half, n_steps=n_steps, dt=float(dt[0]), rng=rng
                )
            else:
                dW_half = np.empty((half, n_steps), dtype=float)
                for j in range(n_steps):
                    dW_half[:, j:j+1] = brownian_increments(
                        n_paths=half, n_steps=1, dt=float(dt[j]), rng=rng
                    )

            dW[:half, :] = dW_half
            dW[half:2 * half, :] = -dW_half
            
            # Statistically redundant but easier to match the size if n_paths is odd.
            if remainder:
                # One additional independent path (or more, but remainder can only be 1 here)
                if np.allclose(dt, dt[0]):
                    dW_extra = brownian_increments(
                        n_paths=remainder, n_steps=n_steps, dt=float(dt[0]), rng=rng
                    )
                else:
                    dW_extra = np.empty((remainder, n_steps), dtype=float)
                    for j in range(n_steps):
                        dW_extra[:, j:j+1] = brownian_increments(
                            n_paths=remainder, n_steps=1, dt=float(dt[j]), rng=rng
                        )
                dW[2 * half:, :] = dW_extra

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

