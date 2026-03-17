from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from roughvol.types import PathBundle, MarketData, SimConfig
from roughvol.sim.brownian import correlated_brownian_increments


@dataclass(frozen=True)
class HestonModel:
    """
    Heston stochastic volatility model (risk-neutral).

    Parameters
    ----------
    kappa : mean reversion speed
    theta : long-run variance
    xi    : vol-of-vol
    rho   : correlation in [-1, 1]
    v0    : initial variance
    """
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    def simulate_paths(
        self,
        *,
        market: MarketData,
        sim: SimConfig,
        rng: np.random.Generator,
    ) -> PathBundle:

        # ---- Validate parameters ----
        kappa = float(self.kappa)
        theta = float(self.theta)
        xi = float(self.xi)
        rho = float(self.rho)
        v0 = float(self.v0)

        if kappa < 0 or theta < 0 or xi < 0 or v0 < 0:
            raise ValueError("Heston parameters must be non-negative.")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("rho must be in [-1,1].")

        # ---- Market inputs ----
        S0 = float(market.spot)
        r = float(market.rate)
        q = float(market.div_yield)

        if S0 <= 0.0:
            raise ValueError("market.spot must be positive.")

        # ---- Time grid ----
        t = np.asarray(sim.grid(), dtype=float)
        n_times = t.size
        if n_times < 1:
            raise ValueError("Empty time grid.")

        n_paths = int(sim.n_paths)
        if n_paths <= 0:
            raise ValueError("sim.n_paths must be positive.")

        antithetic = bool(sim.antithetic)
        if antithetic and (n_paths % 2 != 0):
            raise ValueError("n_paths must be even when antithetic=True.")

        # Maturity = 0 shortcut
        if n_times == 1:
            S = np.full((n_paths, 1), S0)
            v = np.full((n_paths, 1), v0)
            return PathBundle(
                t=t,
                state={"spot": S, "var": v},
                metadata={"model": "Heston", "scheme": "euler-ft"},
            )

        dt = np.diff(t)
        if np.any(dt <= 0.0):
            raise ValueError("Time grid must be strictly increasing.")

        n_steps = n_times - 1

        # ---- Correlated Brownian increments ----
        if np.allclose(dt, dt[0]):
            dW_S, dW_v = correlated_brownian_increments(
                n_paths=n_paths,
                n_steps=n_steps,
                dt=float(dt[0]),
                rho=rho,
                rng=rng,
                antithetic=antithetic,
            )
        else:
            dW_S = np.empty((n_paths, n_steps))
            dW_v = np.empty((n_paths, n_steps))
            for j in range(n_steps):
                dW_S[:, j:j+1], dW_v[:, j:j+1] = correlated_brownian_increments(
                    n_paths=n_paths,
                    n_steps=1,
                    dt=float(dt[j]),
                    rho=rho,
                    rng=rng,
                    antithetic=antithetic,
                )

        # ---- Allocate paths ----
        S = np.empty((n_paths, n_times))
        v = np.empty((n_paths, n_times))
        S[:, 0] = S0
        v[:, 0] = v0

        # ---- Euler full truncation ----
        for j in range(n_steps):
            dt_j = float(dt[j])
            v_pos = np.maximum(v[:, j], 0.0)

            v_next = (
                v[:, j]
                + kappa * (theta - v_pos) * dt_j
                + xi * np.sqrt(v_pos) * dW_v[:, j]
            )
            v[:, j + 1] = np.maximum(v_next, 0.0)

            drift = (r - q - 0.5 * v_pos) * dt_j
            diffusion = np.sqrt(v_pos) * dW_S[:, j]
            S[:, j + 1] = S[:, j] * np.exp(drift + diffusion)

        return PathBundle(
            t=t,
            state={"spot": S, "var": v},
            extras={"dW_S": dW_S, "dW_v": dW_v},
            metadata={
                "model": "Heston",
                "scheme": "euler-full-truncation",
                "kappa": kappa,
                "theta": theta,
                "xi": xi,
                "rho": rho,
                "v0": v0,
                "antithetic": antithetic,
            },
        )
