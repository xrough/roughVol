from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from roughvol.kernels.rough_heston import markovian_lift_weights, rough_heston_kernel
from roughvol.sim.brownian import correlated_brownian_increments
from roughvol.types import MarketData, PathBundle, SimConfig

_VALID_SCHEMES = ("volterra-euler", "markovian-lift", "markovian-lift-numpy", "bayer-breneis")

# ---------------------------------------------------------------------------
# Optional JAX backend for Markovian lift
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp

    # Required for float64 precision in financial calculations.
    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def _jax_markovian_lift_scan(dW1, dW2, dt, w, x, lam, theta, nu, v0, S0, r, q):
        """JAX lax.scan kernel — full path history.

        Returns S (n_paths, n_times), V (n_paths, n_times), Y_history (n_paths, N, n_times).
        Use only when the full trajectory is needed (factor-sweep diagnostics).
        For pricing use _jax_ml_terminal_scan to avoid the O(n_paths·N·n_steps) allocation.
        """
        n_paths = dW1.shape[0]
        N = w.shape[0]

        Y0     = jnp.zeros((n_paths, N))
        V0     = jnp.full((n_paths,), v0)
        log_S0 = jnp.full((n_paths,), jnp.log(S0))

        def step(carry, xs):
            Y, log_S, V_i = carry
            dW1_i, dW2_i, dt_i = xs

            v_pos  = jnp.maximum(V_i, 0.0)
            sqrt_v = jnp.sqrt(v_pos)

            decay       = jnp.exp(-x * dt_i)
            inv_x       = jnp.where(x > 1e-12, 1.0 / x, dt_i)
            drift_scale = (1.0 - decay) * inv_x
            noise_scale = jnp.exp(-x * dt_i * 0.5)

            Y_new = (
                Y * decay
                + drift_scale * (lam * (theta - v_pos[:, None]))
                + noise_scale * nu * sqrt_v[:, None] * dW2_i[:, None]
            )
            V_new     = jnp.maximum(v0 + Y_new @ w, 0.0)
            log_S_new = log_S + (r - q - 0.5 * v_pos) * dt_i + sqrt_v * dW1_i

            return (Y_new, log_S_new, V_new), (V_new, log_S_new, Y_new)

        (_, _, _), (V_steps, log_S_steps, Y_steps) = jax.lax.scan(
            step, (Y0, log_S0, V0), (dW1.T, dW2.T, dt)
        )

        V_full = jnp.concatenate([V0[None], V_steps], axis=0).T
        S_full = jnp.exp(jnp.concatenate([log_S0[None], log_S_steps], axis=0).T)
        Y0_hist = jnp.zeros((1, n_paths, N))
        Y_full  = jnp.transpose(
            jnp.concatenate([Y0_hist, Y_steps], axis=0), (1, 2, 0)
        )

        return S_full, V_full, Y_full

    @jax.jit
    def _jax_ml_terminal_scan(dW1, dW2, dt, w, x, lam, theta, nu, v0, S0, r, q):
        """JAX lax.scan — terminal values only (no Y_history allocation).

        Returns S_T (n_paths,) and V_T (n_paths,).
        O(n_paths·N) memory instead of O(n_paths·N·n_steps).
        """
        n_paths = dW1.shape[0]
        N = w.shape[0]

        Y0     = jnp.zeros((n_paths, N))
        V0     = jnp.full((n_paths,), v0)
        log_S0 = jnp.full((n_paths,), jnp.log(S0))

        def step(carry, xs):
            Y, log_S, V_i = carry
            dW1_i, dW2_i, dt_i = xs

            v_pos  = jnp.maximum(V_i, 0.0)
            sqrt_v = jnp.sqrt(v_pos)

            decay       = jnp.exp(-x * dt_i)
            inv_x       = jnp.where(x > 1e-12, 1.0 / x, dt_i)
            drift_scale = (1.0 - decay) * inv_x
            noise_scale = jnp.exp(-x * dt_i * 0.5)

            Y_new = (
                Y * decay
                + drift_scale * (lam * (theta - v_pos[:, None]))
                + noise_scale * nu * sqrt_v[:, None] * dW2_i[:, None]
            )
            V_new     = jnp.maximum(v0 + Y_new @ w, 0.0)
            log_S_new = log_S + (r - q - 0.5 * v_pos) * dt_i + sqrt_v * dW1_i

            return (Y_new, log_S_new, V_new), None

        (_, log_S_T, V_T), _ = jax.lax.scan(
            step, (Y0, log_S0, V0), (dW1.T, dW2.T, dt)
        )

        return jnp.exp(log_S_T), V_T

    _JAX_AVAILABLE = True

except ImportError:
    _JAX_AVAILABLE = False

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
        - ``"volterra-euler"``   (default) — direct O(n²) Euler, full history
        - ``"markovian-lift"``   — N-factor Markov lift, O(N·n) Euler
        - ``"bayer-breneis"``    — N-factor Markov lift with 5-point Gauss-Hermite
                                   innovation for dW₂ (order-2 weak);
                                   arXiv:2310.04146
    n_factors : number of exponential factors for ``"markovian-lift"`` and
                ``"bayer-breneis"`` (default 8)
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

        if scheme == "bayer-breneis":
            # Generates its own innovations internally (3-point + Gaussian)
            S, V, Y_factors = self._bayer_breneis(
                t=t, dt=dt, n_steps=n_steps, n_paths=n_paths,
                H=H, lam=lam, theta=theta, nu=nu, v0=v0,
                S0=S0, r=r, q=q, rho=rho, antithetic=antithetic, rng=rng,
            )
            extras = {"Y_factors": Y_factors}
        else:
            # Correlated BM increments (volterra-euler and markovian-lift)
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
            elif scheme == "markovian-lift":
                S, V, Y_factors = self._markovian_lift(
                    t=t, dt=dt, n_steps=n_steps, n_paths=n_paths,
                    H=H, lam=lam, theta=theta, nu=nu, v0=v0,
                    S0=S0, r=r, q=q, dW1=dW1, dW2=dW2,
                )
                extras = {"dW1": dW1, "dW2": dW2, "Y_factors": Y_factors}
            else:  # markovian-lift-numpy
                S, V, Y_factors = self._markovian_lift_numpy(
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
        """Markovian N-factor exponential integrator — O(N·n).

        Dispatches to the JAX ``lax.scan`` kernel when JAX is available,
        otherwise falls back to the NumPy loop (``_markovian_lift_numpy``).
        Use scheme ``"markovian-lift-numpy"`` to force the NumPy path.
        """
        if _JAX_AVAILABLE:
            N = int(self.n_factors)
            w, x = markovian_lift_weights(H, n_factors=N)
            S, V, Y_history = _jax_markovian_lift_scan(
                jnp.asarray(dW1), jnp.asarray(dW2), jnp.asarray(dt),
                jnp.asarray(w),   jnp.asarray(x),
                float(lam), float(theta), float(nu), float(v0),
                float(S0),  float(r),     float(q),
            )
            return np.asarray(S), np.asarray(V), np.asarray(Y_history)

        return self._markovian_lift_numpy(
            t=t, dt=dt, n_steps=n_steps, n_paths=n_paths,
            H=H, lam=lam, theta=theta, nu=nu, v0=v0,
            S0=S0, r=r, q=q, dW1=dW1, dW2=dW2,
        )

    def _markovian_lift_numpy(
        self, *, t, dt, n_steps, n_paths, H, lam, theta, nu, v0,
        S0, r, q, dW1, dW2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """NumPy reference implementation of the Markovian lift — O(N·n).

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

            decay       = np.exp(-x * dt_i)
            inv_x       = np.where(x > 1e-12, 1.0 / x, dt_i)
            drift_scale = (1.0 - decay) * inv_x
            noise_scale = np.exp(-x * dt_i * 0.5)

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

    def _bayer_breneis(
        self, *, t, dt, n_steps, n_paths, H, lam, theta, nu, v0,
        S0, r, q, rho, antithetic, rng,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bayer-Breneis (2024) order-2 weak scheme — O(N·n).

        Two improvements over plain markovian-lift:

        (a) Exact matrix-exponential integrator for deterministic drift (same
            formula as markovian-lift: decay = exp(-x·dt), drift scale =
            (1 − exp(-x·dt))/x).

        (b) 5-point Gauss-Hermite innovation for the variance driver dW₂:

                z ∈ {±2.857, ±1.356, 0}
                P ≈ {0.01126, 0.22208, 0.53333, 0.22208, 0.01126}

            This discrete distribution matches the first 9 moments of N(0,1).
            Compared to the 3-point rule (4 moments), the 5-point rule gives
            substantially better weak convergence for option prices (which are
            non-smooth functionals and require higher-moment accuracy).

            The 3-point rule has P(z=0) = 2/3 = 66.7 %, meaning only 1/3 of
            steps have active variance noise; the 5-point rule reduces that to
            53.3 %, which is much more Gaussian-like and yields monotone
            convergence for option prices in our numerical experiments.

        When JAX is available the time loop is replaced by ``lax.scan`` over
        the same XLA kernel used by ``markovian-lift``.  The GH5 innovations
        are precomputed in NumPy and passed in as ordinary arrays — the scan
        body does not care whether dW₁/dW₂ are Gaussian or discrete.

        References
        ----------
        Bayer & Breneis (2024), "Efficient Option Pricing in the Rough Heston
        Model Using Weak Simulation Schemes", arXiv:2310.04146.
        """
        N = int(self.n_factors)
        n_times = n_steps + 1

        w, x = markovian_lift_weights(H, n_factors=N, method="bayer-breneis")

        # 5-point Gauss-Hermite nodes and cumulative probabilities for N(0,1).
        # Nodes are roots of the degree-5 probabilists' Hermite polynomial,
        # weights are the corresponding Gauss-Hermite quadrature weights.
        _GH5_NODES = np.array([-2.85697001, -1.35562618, 0.0,
                                 1.35562618,  2.85697001])
        _GH5_PROBS = np.array([0.01125741, 0.22207592, 0.53333333,
                                0.22207592, 0.01125741])
        # CDF breakpoints for inverse-CDF sampling (exclude last since np.select
        # uses 'default' for the final bin)
        _GH5_CDF = np.cumsum(_GH5_PROBS[:-1])   # [0.01126, 0.23333, 0.76667, 0.98874]

        # Base sample size (half when antithetic so we can mirror)
        half = n_paths // 2 if antithetic else n_paths

        # Variance driver: 5-point discrete innovations via inverse CDF
        U = rng.uniform(size=(half, n_steps))
        z2_base = np.select(
            [U < _GH5_CDF[0], U < _GH5_CDF[1], U < _GH5_CDF[2], U < _GH5_CDF[3]],
            [_GH5_NODES[0],   _GH5_NODES[1],   _GH5_NODES[2],   _GH5_NODES[3]],
            default=_GH5_NODES[4],
        )  # (half, n_steps)
        if antithetic:
            # 5-point distribution is symmetric → -z2_base is the antithetic mirror
            z2 = np.concatenate([z2_base, -z2_base], axis=0)
        else:
            z2 = z2_base

        # Independent Gaussian driver for spot (W⊥, uncorrelated with W₂)
        z_perp_base = rng.standard_normal(size=(half, n_steps))
        if antithetic:
            z_perp = np.concatenate([z_perp_base, -z_perp_base], axis=0)
        else:
            z_perp = z_perp_base

        sqrt_1m_rho2 = np.sqrt(max(1.0 - rho ** 2, 0.0))

        # Precompute full increment arrays (n_paths, n_steps).
        # W₁ = ρ W₂ + √(1-ρ²) W⊥,  W₂ drawn from the GH5 discrete law.
        sqrt_dt = np.sqrt(dt)                                          # (n_steps,)
        dW2 = z2    * sqrt_dt[None, :]                                 # (n_paths, n_steps)
        dW1 = rho * dW2 + sqrt_1m_rho2 * z_perp * sqrt_dt[None, :]   # (n_paths, n_steps)

        # ------------------------------------------------------------------
        # JAX fast path: reuse the same lax.scan kernel as markovian-lift.
        # The kernel is innovation-agnostic — it only sees dW1 / dW2 arrays.
        # ------------------------------------------------------------------
        if _JAX_AVAILABLE:
            S, V, Y_history = _jax_markovian_lift_scan(
                jnp.asarray(dW1), jnp.asarray(dW2), jnp.asarray(dt),
                jnp.asarray(w),   jnp.asarray(x),
                float(lam), float(theta), float(nu), float(v0),
                float(S0),  float(r),     float(q),
            )
            return np.asarray(S), np.asarray(V), np.asarray(Y_history)

        # ------------------------------------------------------------------
        # NumPy fallback loop
        # ------------------------------------------------------------------
        Y = np.zeros((n_paths, N), dtype=float)
        V = np.empty((n_paths, n_times), dtype=float)
        S = np.empty((n_paths, n_times), dtype=float)
        V[:, 0] = v0
        S[:, 0] = S0
        log_S = np.full(n_paths, np.log(S0), dtype=float)

        Y_history = np.zeros((n_paths, N, n_times), dtype=float)

        for i in range(n_steps):
            dt_i  = float(dt[i])
            v_pos = np.maximum(V[:, i], 0.0)  # (n_paths,)

            decay       = np.exp(-x * dt_i)
            inv_x       = np.where(x > 1e-12, 1.0 / x, dt_i)
            drift_scale = (1.0 - decay) * inv_x
            noise_scale = np.exp(-x * dt_i * 0.5)

            Y = (
                Y * decay[None, :]
                + drift_scale[None, :] * (lam * (theta - v_pos[:, None]))
                + noise_scale[None, :] * nu * np.sqrt(v_pos)[:, None] * dW2[:, i : i + 1]
            )
            V[:, i + 1] = np.maximum(v0 + Y @ w, 0.0)
            Y_history[:, :, i + 1] = Y

            log_S += (r - q - 0.5 * v_pos) * dt_i + np.sqrt(v_pos) * dW1[:, i]
            S[:, i + 1] = np.exp(log_S)

        return S, V, Y_history
