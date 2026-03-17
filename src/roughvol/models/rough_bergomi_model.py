from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roughvol.kernels import rough_bergomi_midpoint_weights
from roughvol.sim.brownian import correlated_brownian_increments
from roughvol.types import MarketData, PathBundle, SimConfig


@dataclass(frozen=True)
class RoughBergomiModel:
    """Approximate rough Bergomi model with midpoint Volterra discretisation.

    Parameters
    ----------
    hurst : roughness parameter H in (0, 0.5)
    eta   : volatility-of-volatility
    rho   : correlation between the variance driver and spot driver
    xi0   : flat initial forward variance level, used when market.forward_variance_curve is absent
    """

    hurst: float
    eta: float
    rho: float
    xi0: float

    def simulate_paths(
        self,
        *,
        market: MarketData,
        sim: SimConfig,
        rng: np.random.Generator,
    ) -> PathBundle:
        H = float(self.hurst)
        eta = float(self.eta)
        rho = float(self.rho)
        xi0 = float(self.xi0)

        if not (0.0 < H < 0.5):
            raise ValueError("hurst must lie in (0, 0.5).")
        if eta < 0.0:
            raise ValueError("eta must be non-negative.")
        if xi0 < 0.0:
            raise ValueError("xi0 must be non-negative.")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("rho must be in [-1, 1].")

        S0 = float(market.spot)
        r = float(market.rate)
        q = float(market.div_yield)
        if S0 <= 0.0:
            raise ValueError("market.spot must be positive.")

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

        if n_times == 1:
            S = np.full((n_paths, 1), S0, dtype=float)
            xi_curve = np.full((1,), xi0, dtype=float)
            return PathBundle(
                t=t,
                state={
                    "spot": S,
                    "var": np.broadcast_to(xi_curve, (n_paths, 1)).copy(),
                    "Y": np.zeros((n_paths, 1), dtype=float),
                },
                metadata={"model": "RoughBergomi", "scheme": "volterra-midpoint"},
            )

        dt = np.diff(t)
        if np.any(dt <= 0.0):
            raise ValueError("Time grid must be strictly increasing.")

        n_steps = n_times - 1
        dW_y, dW_s = correlated_brownian_increments(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=1.0,
            rho=rho,
            rng=rng,
            antithetic=antithetic,
        )
        dW_y *= np.sqrt(dt)[None, :]
        dW_s *= np.sqrt(dt)[None, :]

        weights = rough_bergomi_midpoint_weights(t, H)
        Y = np.zeros((n_paths, n_times), dtype=float)
        Y[:, 1:] = dW_y @ weights.T

        xi_curve = _forward_variance_curve(t, market, xi0)
        var = np.empty((n_paths, n_times), dtype=float)
        var[:, 0] = xi_curve[0]
        variance_correction = np.power(t[1:], 2.0 * H)
        var[:, 1:] = xi_curve[1:][None, :] * np.exp(
            eta * Y[:, 1:] - 0.5 * (eta ** 2) * variance_correction[None, :]
        )

        S = np.empty((n_paths, n_times), dtype=float)
        S[:, 0] = S0
        for j in range(n_steps):
            v = np.maximum(var[:, j], 0.0)
            drift = (r - q - 0.5 * v) * dt[j]
            diffusion = np.sqrt(v) * dW_s[:, j]
            S[:, j + 1] = S[:, j] * np.exp(drift + diffusion)

        return PathBundle(
            t=t,
            state={"spot": S, "var": var, "Y": Y},
            extras={
                "dW_y": dW_y,
                "dW_s": dW_s,
                "forward_variance_curve": np.broadcast_to(xi_curve, (n_paths, n_times)),
            },
            metadata={
                "model": "RoughBergomi",
                "scheme": "volterra-midpoint",
                "hurst": H,
                "eta": eta,
                "rho": rho,
                "xi0": xi0,
                "antithetic": antithetic,
            },
        )


def _forward_variance_curve(t: np.ndarray, market: MarketData, xi0: float) -> np.ndarray:
    if market.forward_variance_curve is None:
        return np.full_like(t, xi0, dtype=float)

    xi_curve = np.asarray(market.forward_variance_curve(t), dtype=float)
    if xi_curve.shape != t.shape:
        raise ValueError("forward_variance_curve must return an array aligned with t.")
    if np.any(xi_curve < 0.0):
        raise ValueError("forward_variance_curve must be non-negative.")
    return xi_curve
