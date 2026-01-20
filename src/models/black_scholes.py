'''

'''

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from roughvol.sim.BM import brownian_increments # generate paths of BMs with given increment 
from roughvol.types import ArrayF

'''
参数self有些神奇,此处指代包含simulate_paths的class,具体实用时先定义BlackScholesModel,如model=BlackScholesModel(...),
然后通过paths = model.simulate_paths(...)调用.
'''

@dataclass(frozen=True)
class BlackScholesModel:
    spot0: float # S_0
    rate: float # risk free rate
    div: float # dividend
    vol: float # constant vol
    def simulate_paths(
        self,
        *,
        n_paths: int,
        n_steps: int,
        maturity: float,
        rng: np.random.Generator,
    ) -> ArrayF:
        # Basic validation
        if self.spot0 <= 0:
            raise ValueError("spot0 must be positive.")
        if self.vol < 0:
            raise ValueError("vol must be non-negative.")
        if maturity <= 0:
            raise ValueError("maturity must be positive.")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")

        dt = maturity / n_steps

        # Brownian increments: shape (n_paths, n_steps)
        dW = brownian_increments(n_paths=n_paths, n_steps=n_steps, dt=dt, rng=rng) # fixed seed

        # Exact log-Euler (exact discretization for GBM)
        drift = (self.rate - self.div - 0.5 * self.vol**2) * dt
        diffusion = self.vol * dW

        logS = np.empty((n_paths, n_steps + 1), dtype=float) # shape of log price
        logS[:, 0] = np.log(self.spot0) #初值
        logS[:, 1:] = logS[:, [0]] + np.cumsum(drift + diffusion, axis=1) # cumsum: cummulation sum

        return np.exp(logS)
