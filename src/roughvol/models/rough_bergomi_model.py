from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roughvol.sim.brownian import correlated_brownian_increments
from roughvol.sim.volterra import simulate_blp, simulate_exact, simulate_midpoint
from roughvol.types import MarketData, PathBundle, SimConfig

_VALID_SCHEMES = ("volterra-midpoint", "exact-gaussian", "blp-hybrid")

# ---------------------------------------------------------------------------
# Optional JAX backend for the spot path time loop
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def _jax_var_and_spot(Y, xi_curve, t, dt, dW_s, H, eta, S0, r, q):
        """JAX lax.scan kernel — variance and spot paths.

        All array inputs should be JAX arrays.  H, eta, S0, r, q are scalars.
        Returns var (n_paths, n_times) and S (n_paths, n_times) as JAX arrays.
        """
        n_paths = Y.shape[0]

        # Variance is fully vectorised — no sequential dependence.
        variance_correction = jnp.power(t[1:], 2.0 * H)
        var_rest = xi_curve[1:][None, :] * jnp.exp(
            eta * Y[:, 1:] - 0.5 * (eta ** 2) * variance_correction[None, :]
        )
        var0 = jnp.full((n_paths, 1), xi_curve[0])
        var = jnp.concatenate([var0, var_rest], axis=1)  # (n_paths, n_times)

        # Spot path via lax.scan — replaces the Python for-loop over time steps.
        log_S0_vec = jnp.full(n_paths, jnp.log(S0))

        def step(log_S, xs):
            v_j, dW_s_j, dt_j = xs
            v = jnp.maximum(v_j, 0.0)
            log_S_new = log_S + (r - q - 0.5 * v) * dt_j + jnp.sqrt(v) * dW_s_j
            return log_S_new, log_S_new

        # Scan over (n_steps,) leading axis; each slice is (n_paths,) or scalar.
        _, log_S_steps = jax.lax.scan(
            step, log_S0_vec,
            (var[:, :-1].T, dW_s.T, dt),  # shapes: (n_steps, n_paths), (n_steps, n_paths), (n_steps,)
        )
        # log_S_steps: (n_steps, n_paths)
        log_S_all = jnp.concatenate([log_S0_vec[None, :], log_S_steps], axis=0)  # (n_times, n_paths)
        S = jnp.exp(log_S_all).T  # (n_paths, n_times)

        return var, S

    _JAX_AVAILABLE = True

except ImportError:
    _JAX_AVAILABLE = False


@dataclass(frozen=True)
class RoughBergomiModel:
    """Rough Bergomi model with selectable simulation scheme.

    Parameters
    ----------
    hurst  : roughness parameter H in (0, 0.5)
    eta    : volatility-of-volatility
    rho    : correlation between the variance driver and spot driver
    xi0    : flat initial forward variance level, used when market.forward_variance_curve is absent
    scheme : simulation scheme; one of:
        - ``"volterra-midpoint"``  (default) — O(n²) midpoint Volterra quadrature (§2.5)
        - ``"exact-gaussian"``    — O(n³) precomp Cholesky of joint fBM covariance (§2.1 benchmark)
        - ``"blp-hybrid"``        — O(n log n) Bennedsen-Lunde-Pakkanen hybrid scheme (§2.3)
    blp_kappa : near-field cutoff for the BLP scheme (default 10)
    """

    hurst: float
    eta: float
    rho: float
    xi0: float
    scheme: str = "volterra-midpoint"
    blp_kappa: int = 10

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
                metadata={"model": "RoughBergomi", "scheme": scheme},
            )

        dt = np.diff(t)
        if np.any(dt <= 0.0):
            raise ValueError("Time grid must be strictly increasing.")

        n_steps = n_times - 1
        xi_curve = _forward_variance_curve(t, market, xi0)

        # ------------------------------------------------------------------
        # Dispatch to selected scheme
        # ------------------------------------------------------------------
        if scheme == "volterra-midpoint":
            return self._simulate_volterra_midpoint(
                t=t, dt=dt, n_steps=n_steps, n_paths=n_paths,
                antithetic=antithetic, H=H, eta=eta, rho=rho, xi0=xi0,
                S0=S0, r=r, q=q, xi_curve=xi_curve, rng=rng,
            )
        elif scheme == "exact-gaussian":
            return self._simulate_exact_gaussian(
                t=t, dt=dt, n_steps=n_steps, n_paths=n_paths,
                antithetic=antithetic, H=H, eta=eta, rho=rho, xi0=xi0,
                S0=S0, r=r, q=q, xi_curve=xi_curve, rng=rng,
            )
        else:  # blp-hybrid
            return self._simulate_blp_hybrid(
                t=t, dt=dt, n_steps=n_steps, n_paths=n_paths,
                antithetic=antithetic, H=H, eta=eta, rho=rho, xi0=xi0,
                S0=S0, r=r, q=q, xi_curve=xi_curve, rng=rng,
            )

    # ------------------------------------------------------------------
    # Private scheme implementations
    # ------------------------------------------------------------------

    def _simulate_volterra_midpoint(
        self, *, t, dt, n_steps, n_paths, antithetic, H, eta, rho, xi0,
        S0, r, q, xi_curve, rng,
    ) -> PathBundle:
        n_times = n_steps + 1
        dW_y, dW_s = correlated_brownian_increments(
            n_paths=n_paths, n_steps=n_steps, dt=1.0,
            rho=rho, rng=rng, antithetic=antithetic,
        )
        dW_y *= np.sqrt(dt)[None, :]
        dW_s *= np.sqrt(dt)[None, :]

        Y = simulate_midpoint(dW_y, t, H)

        var, S = _var_and_spot(Y, xi_curve, t, dt, dW_s, H, eta, S0, r, q, n_steps, n_paths)

        return PathBundle(
            t=t,
            state={"spot": S, "var": var, "Y": Y},
            extras={
                "dW_y": dW_y,
                "dW_s": dW_s,
                "forward_variance_curve": np.broadcast_to(xi_curve, (n_paths, n_times)),
            },
            metadata={
                "model": "RoughBergomi", "scheme": "volterra-midpoint",
                "hurst": H, "eta": eta, "rho": rho, "xi0": xi0, "antithetic": antithetic,
            },
        )

    def _simulate_exact_gaussian(
        self, *, t, dt, n_steps, n_paths, antithetic, H, eta, rho, xi0,
        S0, r, q, xi_curve, rng,
    ) -> PathBundle:
        """Exact Gaussian simulation via Cholesky of joint (Ytilde, W) covariance (§2.1)."""
        n_times = n_steps + 1

        Y, dW_y = simulate_exact(t, H, n_paths, antithetic=antithetic, rng=rng)

        # Orthogonal BM increments for spot; combine with variance-driver BM for correlation
        dWperp = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)[None, :]
        dW_s = rho * dW_y + np.sqrt(1.0 - rho ** 2) * dWperp

        var, S = _var_and_spot(Y, xi_curve, t, dt, dW_s, H, eta, S0, r, q, n_steps, n_paths)

        return PathBundle(
            t=t,
            state={"spot": S, "var": var, "Y": Y},
            extras={
                "dW_y": dW_y,
                "dW_s": dW_s,
                "forward_variance_curve": np.broadcast_to(xi_curve, (n_paths, n_times)),
            },
            metadata={
                "model": "RoughBergomi", "scheme": "exact-gaussian",
                "hurst": H, "eta": eta, "rho": rho, "xi0": xi0, "antithetic": antithetic,
            },
        )

    def _simulate_blp_hybrid(
        self, *, t, dt, n_steps, n_paths, antithetic, H, eta, rho, xi0,
        S0, r, q, xi_curve, rng,
    ) -> PathBundle:
        """BLP hybrid scheme (§2.3): near-field exact + far-field FFT convolution."""
        n_times = n_steps + 1
        kappa = int(self.blp_kappa)

        # Generate correlated raw BM increments (variance driver dW_y, spot driver dW_s)
        dW_y, dW_s = correlated_brownian_increments(
            n_paths=n_paths, n_steps=n_steps, dt=1.0,
            rho=rho, rng=rng, antithetic=antithetic,
        )
        dW_y *= np.sqrt(dt)[None, :]
        dW_s *= np.sqrt(dt)[None, :]

        # BLP scheme returns Y of shape (n_paths, n_times)
        Y = simulate_blp(dW_y, t, H, kappa=kappa, rng=rng)

        var, S = _var_and_spot(Y, xi_curve, t, dt, dW_s, H, eta, S0, r, q, n_steps, n_paths)

        return PathBundle(
            t=t,
            state={"spot": S, "var": var, "Y": Y},
            extras={
                "dW_y": dW_y,
                "dW_s": dW_s,
                "forward_variance_curve": np.broadcast_to(xi_curve, (n_paths, n_times)),
            },
            metadata={
                "model": "RoughBergomi", "scheme": "blp-hybrid",
                "hurst": H, "eta": eta, "rho": rho, "xi0": xi0,
                "antithetic": antithetic, "blp_kappa": kappa,
            },
        )


def _var_and_spot(
    Y: np.ndarray,
    xi_curve: np.ndarray,
    t: np.ndarray,
    dt: np.ndarray,
    dW_s: np.ndarray,
    H: float,
    eta: float,
    S0: float,
    r: float,
    q: float,
    n_steps: int,
    n_paths: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute variance and spot paths given the Volterra driver Y.

    Dispatches to the JAX lax.scan kernel when JAX is available, otherwise
    falls back to the NumPy loop.
    """
    if _JAX_AVAILABLE:
        var, S = _jax_var_and_spot(
            jnp.asarray(Y), jnp.asarray(xi_curve), jnp.asarray(t),
            jnp.asarray(dt), jnp.asarray(dW_s),
            float(H), float(eta), float(S0), float(r), float(q),
        )
        return np.asarray(var), np.asarray(S)

    # NumPy fallback
    n_times = n_steps + 1
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

    return var, S


def _forward_variance_curve(t: np.ndarray, market: MarketData, xi0: float) -> np.ndarray:
    if market.forward_variance_curve is None:
        return np.full_like(t, xi0, dtype=float)

    xi_curve = np.asarray(market.forward_variance_curve(t), dtype=float)
    if xi_curve.shape != t.shape:
        raise ValueError("forward_variance_curve must return an array aligned with t.")
    if np.any(xi_curve < 0.0):
        raise ValueError("forward_variance_curve must be non-negative.")
    return xi_curve
