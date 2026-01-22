'''
建模underlying asset的随机过程，随机性由MC引擎传导，随机元素的生成单独包含在sim中。
'''

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from roughvol.sim.BM import brownian_increments # generate paths of BMs with given increment 
from roughvol.types import ArrayF

'''
参数self有些神奇，此处指代包含simulate_paths的class。
'''

@dataclass(frozen=True)
class GBM_Model: # 由于包含simulate_paths这个函数，属于PathModel这个Class。
    spot0: float # S_0
    rate: float # risk free rate
    div: float # dividend
    vol: float # constant vol
    def simulate_paths( #this is a method, which is just a function within a class.
        self,
        *,
        n_paths: int,
        n_steps: int,
        maturity: float,
        rng: np.random.Generator,
        dW: ArrayF | None = None,
    ) -> ArrayF:
        # Basic validation
        if self.spot0 <= 0:
            raise ValueError("spot0 must be positive.")
        if self.vol < 0:
            raise ValueError("vol must be non-negative.")
        if maturity < 0:
            raise ValueError("maturity must be nonnegative.")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        
        # --- NEW: zero maturity short-circuit ---
        if maturity == 0:
            return np.full((n_paths, 1), self.spot0, dtype=float)

        dt = maturity / n_steps

        # If engine did not provide increments, generate plain ones (no antithetic here)
        if dW is None:
            dW = brownian_increments(n_paths=n_paths, n_steps=n_steps, dt=dt, rng=rng)
        else:
            dW = np.asarray(dW, dtype=float)
            if dW.shape != (n_paths, n_steps):
                raise ValueError(f"dW must have shape {(n_paths, n_steps)}, got {dW.shape}")

        # Exact log-Euler (exact discretization for GBM)
        drift = (self.rate - self.div - 0.5 * self.vol**2) * dt
        diffusion = self.vol * dW

        logS = np.empty((n_paths, n_steps + 1), dtype=float) # shape of log price
        logS[:, 0] = np.log(self.spot0) #初值
        logS[:, 1:] = logS[:, [0]] + np.cumsum(drift + diffusion, axis=1) # cumsum: cummulation sum

        return np.exp(logS)
