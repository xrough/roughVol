from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from roughvol.kernels.rough_heston import markovian_lift_weights, rough_heston_kernel
from roughvol.sim.brownian import correlated_brownian_increments
from roughvol.types import MarketData, PathBundle, SimConfig

_VALID_SCHEMES = ("volterra-euler", "markovian-lift")


@dataclass(frozen=True)
class RoughHestonModel:
    """Rough Heston stochastic volatility model with selectable simulation scheme.

    The variance process satisfies the stochastic Volterra equation:
        V_t = V_0 + integral_0^t K(t-s) [lam*(theta - V_s) ds + nu*sqrt(V_s) dW_s^(2)]

    where  K(t) = t^(H-1/2) / Gamma(H+1/2),  H in (0, 0.5).

    The stock follows:
        dS_t/S_t = sqrt(V_t) dW_t^(1),   d<W^(1), W^(2)>_t = rho dt.

    Parameters
    ----------
    hurst     : roughness H in (0, 0.5)
    lam       : mean-reversion speed (kappa in standard notation)
    theta     : long-run variance
    nu        : vol-of-vol
    rho       : correlation between spot and variance drivers
    v0        : initial variance
    scheme    : one of:
        - ``"volterra-euler"``   (default) — direct O(n²) Euler, full history (§3.1)
        - ``"markovian-lift"``   — N-factor Markov lift, O(N·n) Euler (§3.5)
    n_factors : number of exponential factors for ``"markovian-lift"`` (default 8)
    """

    hurst: float
    lam: float
    theta: float
    nu: float
    rho: float
    v0: float
    scheme: str = "volterra-euler"
    n_factors: int = 8

    def simulate_paths(
        self,
        *,
        market: MarketData,
        sim: SimConfig,
        rng: np.random.Generator,
    ) -> PathBundle:
        scheme = str(self.scheme)
        if scheme not in _VALID_SCHEMES:
            raise ValueError(f"Unknown scheme {scheme!r}. Valid: {_VALID_SCHEMES}")

        H = float(self.hurst)
        lam = float(self.lam)
        theta = float(self.theta)
        nu = float(self.nu)
        rho = float(self.rho)
        v0 = float(self.v0)

        if not (0.0 < H < 0.5):
            raise ValueError("hurst must lie in (0, 0.5).")
        if lam < 0.0:
            raise ValueError("lam must be non-negative.")
        if theta < 0.0:
            raise ValueError("theta must be non-negative.")
        if nu < 0.0:
            raise ValueError("nu must be non-negative.")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("rho must be in [-1, 1].")
        if v0 < 0.0:
            raise ValueError("v0 must be non-negative.")

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
            V = np.full((n_paths, 1), v0, dtype=float)
            return PathBundle(
                t=t,
                state={"spot": S, "var": V},
                metadata={"model": "RoughHeston", "scheme": scheme},
            )

        dt = np.diff(t)
        if np.any(dt <= 0.0):
            raise ValueError("Time grid must be strictly increasing.")

        n_steps = n_times - 1

        # Correlated BM increments
        dW1, dW2 = correlated_brownian_increments(
            n_paths=n_paths, n_steps=n_steps, dt=1.0,
            rho=rho, rng=rng, antithetic=antithetic,
        )
        dW1 *= np.sqrt(dt)[None, :]
        dW2 *= np.sqrt(dt)[None, :]

        if scheme == "volterra-euler":
            S, V = self._volterra_euler(
                t=t, dt=dt, n_steps=n_steps, n_paths=n_paths,
                H=H, lam=lam, theta=theta, nu=nu, v0=v0,
                S0=S0, r=r, q=q, dW1=dW1, dW2=dW2,
            )
            extras = {"dW1": dW1, "dW2": dW2}
        else:  # markovian-lift
            S, V, Y_factors = self._markovian_lift(
                t=t, dt=dt, n_steps=n_steps, n_paths=n_paths,
                H=H, lam=lam, theta=theta, nu=nu, v0=v0,
                S0=S0, r=r, q=q, dW1=dW1, dW2=dW2,
            )
            extras = {"dW1": dW1, "dW2": dW2, "Y_factors": Y_factors}

        return PathBundle(
            t=t,
            state={"spot": S, "var": V},
            extras=extras,
            metadata={
                "model": "RoughHeston", "scheme": scheme,
                "hurst": H, "lam": lam, "theta": theta,
                "nu": nu, "rho": rho, "v0": v0, "antithetic": antithetic,
            },
        )

    # ------------------------------------------------------------------
    # Scheme implementations
    # ------------------------------------------------------------------

    def _volterra_euler(
        self, *, t, dt, n_steps, n_paths, H, lam, theta, nu, v0,
        S0, r, q, dW1, dW2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Direct Volterra Euler — O(n²) — §3.1."""
        n_times = n_steps + 1

        # Precompute full Volterra kernel matrix K[i, j] = K(t_{i+1} - t_j) for j <= i
        K = np.zeros((n_steps, n_steps), dtype=float)
        for i in range(n_steps):
            lags = t[i + 1] - t[: i + 1]  # t_{i+1} - t_j, j = 0,...,i
            K[i, : i + 1] = rough_heston_kernel(lags, H)

        V = np.empty((n_paths, n_times), dtype=float)
        S = np.empty((n_paths, n_times), dtype=float)
        V[:, 0] = v0
        S[:, 0] = S0

        for i in range(n_steps):
            v_pos = np.maximum(V[:, : i + 1], 0.0)  # (n_paths, i+1), clipped

            # Drift sum: sum_{j=0}^{i} K[i,j] * lam*(theta - V_j) * dt_j
            drift_integ = np.dot(v_pos * 0.0 + lam * (theta - V[:, : i + 1]), K[i, : i + 1] * dt[: i + 1])
            # Diffusion sum: sum_{j=0}^{i} K[i,j] * nu * sqrt(V_j+) * dW2_j
            diff_integ = np.dot(nu * np.sqrt(v_pos) * dW2[:, : i + 1], K[i, : i + 1])

            V[:, i + 1] = np.maximum(v0 + drift_integ + diff_integ, 0.0)

            v_i = np.maximum(V[:, i], 0.0)
            S[:, i + 1] = S[:, i] * np.exp(
                (r - q - 0.5 * v_i) * dt[i] + np.sqrt(v_i) * dW1[:, i]
            )

        return S, V

    def _markovian_lift(
        self, *, t, dt, n_steps, n_paths, H, lam, theta, nu, v0,
        S0, r, q, dW1, dW2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Markovian N-factor exponential integrator — O(N·n) — §3.5.

        Uses an exponential integrator for the linear -x_m*Y_m decay term so that
        the scheme is unconditionally stable even when x_m >> 1/dt (which is the case
        for the high-frequency factors in the geometric-spacing approximation).

        Update rule per factor m:
            decay_m   = exp(-x_m * dt)
            Y_m^{i+1} = decay_m * Y_m^i
                        + (1 - decay_m)/x_m * lam*(theta - V_i)          [drift integral]
                        + nu * sqrt(V_i+) * exp(-x_m*dt/2) * dW2_i       [noise, midpoint]
        """
        N = int(self.n_factors)
        n_times = n_steps + 1

        # Fit kernel approximation
        w, x = markovian_lift_weights(H, n_factors=N)  # (N,), (N,)

        Y = np.zeros((n_paths, N), dtype=float)
        V = np.empty((n_paths, n_times), dtype=float)
        S = np.empty((n_paths, n_times), dtype=float)
        V[:, 0] = v0
        S[:, 0] = S0

        Y_history = np.zeros((n_paths, N, n_times), dtype=float)

        for i in range(n_steps):
            dt_i = dt[i]
            v_pos = np.maximum(V[:, i], 0.0)  # (n_paths,)

            # Exponential integrator for linear part (unconditionally stable)
            decay = np.exp(-x * dt_i)               # (N,)
            inv_x = np.where(x > 1e-12, 1.0 / x, dt_i)  # safe inverse
            drift_scale = (1.0 - decay) * inv_x      # (N,) = integral_0^dt exp(-x*s) ds

            # Factor update:
            #   decay term:  decay_m * Y_m
            #   drift term:  drift_scale_m * lam*(theta - V_i)
            #   noise term:  nu * sqrt(V_i+) * exp(-x_m * dt/2) * dW2_i
            noise_scale = np.exp(-x * dt_i * 0.5)   # midpoint approximation for noise integral
            Y = (
                Y * decay[None, :]
                + drift_scale[None, :] * (lam * (theta - v_pos[:, None]))
                + noise_scale[None, :] * nu * np.sqrt(v_pos)[:, None] * dW2[:, i : i + 1]
            )

            V[:, i + 1] = np.maximum(v0 + Y @ w, 0.0)

            S[:, i + 1] = S[:, i] * np.exp(
                (r - q - 0.5 * v_pos) * dt_i + np.sqrt(v_pos) * dW1[:, i]
            )
            Y_history[:, :, i + 1] = Y

        return S, V, Y_history
