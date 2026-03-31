"""Fractional Brownian motion (Volterra driver) simulation for rough volatility models.

Mirrors sim/brownian.py for standard Brownian motion: this module is concerned
only with simulating the Volterra process Y (the fBM driver), independent of
model parameters (eta, rho, xi0).

Three schemes are provided:

  simulate_midpoint  — O(n²) midpoint quadrature (fast, biased for small n)
  simulate_exact     — O(n³) exact Cholesky of the joint fBM covariance
  simulate_blp       — O(n log n) Bennedsen-Lunde-Pakkanen hybrid scheme

All three return Y of shape (n_paths, n_times) with Y[:, 0] = 0.
"""
from __future__ import annotations

import numpy as np

from roughvol.kernels.rough_bergomi import rough_bergomi_midpoint_weights
from roughvol.kernels.rough_bergomi_exact import rough_bergomi_exact_cholesky
from roughvol.types import ArrayF


# ---------------------------------------------------------------------------
# Scheme 1: volterra-midpoint  O(n²)
# ---------------------------------------------------------------------------

def simulate_midpoint(
    dW_y: ArrayF,
    t: ArrayF,
    hurst: float,
) -> ArrayF:
    """Simulate the Volterra driver Y using midpoint quadrature.

    Y_{t_i} = sqrt(2H) * sum_{j=0}^{i-1} (t_i - mid_j)^{H-0.5} * dW_y_j

    where mid_j is the midpoint of [t_j, t_{j+1}].

    Parameters
    ----------
    dW_y  : (n_paths, n_steps) Brownian increments for the variance driver
    t     : 1D time grid of shape (n_steps + 1,), t[0] = 0
    hurst : Hurst exponent H in (0, 0.5)

    Returns
    -------
    Y : (n_paths, n_steps + 1) Volterra driver paths, Y[:, 0] = 0
    """
    dW_y = np.asarray(dW_y, dtype=float)
    t = np.asarray(t, dtype=float)
    n_paths, n_steps = dW_y.shape

    weights = rough_bergomi_midpoint_weights(t, hurst)  # (n_steps, n_steps) upper-triangular
    Y = np.zeros((n_paths, n_steps + 1), dtype=float)
    Y[:, 1:] = dW_y @ weights.T
    return Y


# ---------------------------------------------------------------------------
# Scheme 2: exact-gaussian  O(n³ precompute, O(n²) per sample)
# ---------------------------------------------------------------------------

def simulate_exact(
    t: ArrayF,
    hurst: float,
    n_paths: int,
    *,
    antithetic: bool = False,
    rng: np.random.Generator,
) -> tuple[ArrayF, ArrayF]:
    """Simulate the Volterra driver Y exactly via Cholesky of the joint fBM covariance.

    Samples (Ytilde, W) jointly from their true Gaussian distribution.  W is
    the standard Brownian motion that also drives the Volterra integral, so its
    increments dW_y can be recovered directly from the simulation without any
    additional approximation.

    Parameters
    ----------
    t         : 1D time grid of shape (n_steps + 1,), t[0] = 0
    hurst     : Hurst exponent H in (0, 0.5)
    n_paths   : number of simulation paths
    antithetic: if True, produce n_paths//2 base paths and mirror them
    rng       : numpy Generator for reproducibility

    Returns
    -------
    Y     : (n_paths, n_steps + 1) Volterra driver, Y[:, 0] = 0
    dW_BM : (n_paths, n_steps) increments of the underlying BM W
    """
    t = np.asarray(t, dtype=float)
    n_steps = t.size - 1

    # O(n³) precomputation: build Cholesky factor of the joint 2n × 2n covariance
    # Block layout: [Ytilde (rows 0:n), W (rows n:2n)]
    L = rough_bergomi_exact_cholesky(t, hurst)  # (2n, 2n)
    n = n_steps

    if antithetic:
        half = n_paths // 2
        Z_half = rng.standard_normal((2 * n, half))
        X_half = L @ Z_half                    # (2n, half)
        Wtilde_pos = X_half[:n, :].T           # (half, n)
        W_pos = X_half[n:, :].T               # (half, n)
        # Antithetic negation: both Ytilde and W are linear in Z
        Ytilde = np.concatenate([Wtilde_pos, -Wtilde_pos], axis=0)  # (n_paths, n)
        W_full = np.concatenate([W_pos, -W_pos], axis=0)            # (n_paths, n)
    else:
        Z = rng.standard_normal((2 * n, n_paths))
        X = L @ Z                    # (2n, n_paths)
        Ytilde = X[:n, :].T          # (n_paths, n)
        W_full = X[n:, :].T          # (n_paths, n)

    # Prepend zero to W_full to get level paths, then difference to get increments
    W_levels = np.concatenate([np.zeros((n_paths, 1)), W_full], axis=1)  # (n_paths, n+1)
    dW_BM = np.diff(W_levels, axis=1)  # (n_paths, n_steps)

    # Prepend zero column to get Y on the full time grid
    Y = np.zeros((n_paths, n_steps + 1), dtype=float)
    Y[:, 1:] = Ytilde

    return Y, dW_BM


# ---------------------------------------------------------------------------
# Scheme 3: blp-hybrid  O(n log n)
# ---------------------------------------------------------------------------

def simulate_blp(
    dW_y: ArrayF,
    t: ArrayF,
    hurst: float,
    kappa: int = 10,
    rng: np.random.Generator | None = None,
) -> ArrayF:
    """Simulate the Volterra driver Y using the Bennedsen-Lunde-Pakkanen hybrid scheme.

    Splits the Volterra integral into:
      - Near field  (lags k = 1,...,kappa): exact bivariate Gaussian sampling,
        correctly handling the kernel singularity at lag 0.
      - Far field   (lags k > kappa): midpoint-quadrature step function,
        computed via FFT linear convolution in O(n log n).

    Parameters
    ----------
    dW_y  : (n_paths, n_steps) Brownian increments for the variance driver
    t     : 1D time grid of shape (n_steps + 1,), t[0] = 0
    hurst : Hurst exponent H in (0, 0.5)
    kappa : near-field cutoff; default 10
    rng   : numpy Generator for near-field sampling; uses default_rng() if None

    Returns
    -------
    Y : (n_paths, n_steps + 1) Volterra driver paths, Y[:, 0] = 0
    """
    H = float(hurst)
    alpha = H - 0.5  # kernel exponent; alpha in (-0.5, 0)

    t = np.asarray(t, dtype=float)
    dW_y = np.asarray(dW_y, dtype=float)
    n_paths, n_steps = dW_y.shape

    if t.size != n_steps + 1:
        raise ValueError("t must have n_steps + 1 elements.")
    if not (0.0 < H < 0.5):
        raise ValueError("hurst must lie in (0, 0.5).")
    if kappa < 0:
        raise ValueError("kappa must be non-negative.")

    if rng is None:
        rng = np.random.default_rng()

    dt = np.diff(t)
    scale = np.sqrt(2.0 * H)  # L_g prefactor from the kernel normalisation

    # ------------------------------------------------------------------
    # Pre-compute near-field covariances for lags k = 1, ..., min(kappa, n_steps)
    #
    # For lag k the kernel interval is [t_{i-k}, t_{i-k+1}].
    # On a uniform grid with step dt_val:
    #   Var(W^n_{i-k,k}) = [(k·dt)^{2H} - ((k-1)·dt)^{2H}] / (2H)
    #   Cov(W^n_{i-k,k}, dW_{i-k}) = [(k·dt)^{H+0.5} - ((k-1)·dt)^{H+0.5}] / (H+0.5)
    # ------------------------------------------------------------------
    dt_val = dt[0]  # reference step (first interval; approximation for non-uniform grids)
    k_near = min(kappa, n_steps)

    near_var = np.empty(k_near)
    near_cov = np.empty(k_near)
    for k in range(1, k_near + 1):
        near_var[k - 1] = ((k * dt_val) ** (2 * H) - ((k - 1) * dt_val) ** (2 * H)) / (2 * H)
        near_cov[k - 1] = (
            (k * dt_val) ** (alpha + 1) - ((k - 1) * dt_val) ** (alpha + 1)
        ) / (alpha + 1)

    # ------------------------------------------------------------------
    # Far-field midpoint weights for lags k = kappa+1, ..., n_steps
    # Midpoint of [t_{i-k}, t_{i-k+1}] relative to t_i ≈ (k - 0.5)·dt_val
    # ------------------------------------------------------------------
    if k_near < n_steps:
        lags_far = np.arange(k_near + 1, n_steps + 1, dtype=float)
        mids_far = (lags_far - 0.5) * dt_val
        weights_far = scale * np.power(mids_far, alpha)  # (n_far,)
    else:
        weights_far = np.empty(0)

    Y = np.zeros((n_paths, n_steps + 1), dtype=float)

    # ---- Far-field via FFT convolution ----
    # Y_far[:, i] = sum_{k=k_near+1}^{i} weights_far[k-k_near-1] * dW_y[:, i-k]
    if weights_far.size > 0:
        n_fft = int(2 ** np.ceil(np.log2(2 * n_steps)))
        w_pad = np.zeros(n_fft)
        w_pad[: weights_far.size] = weights_far

        dW_pad = np.zeros((n_paths, n_fft))
        dW_pad[:, :n_steps] = dW_y

        W_fft = np.fft.rfft(w_pad)
        DW_fft = np.fft.rfft(dW_pad, axis=1)
        conv = np.fft.irfft(W_fft[None, :] * DW_fft, n=n_fft, axis=1)  # (n_paths, n_fft)

        for i in range(k_near + 1, n_steps + 1):
            idx = i - k_near - 1
            if 0 <= idx < n_steps:
                Y[:, i] += conv[:, idx]

    # ---- Near-field: bivariate Gaussian sampling ----
    # W^n_{i-k,k} | dW_{i-k} ~ N(beta_k * dW_{i-k}, cond_var_k)
    for k_idx in range(k_near):
        k = k_idx + 1
        var_k = near_var[k_idx]
        cov_k = near_cov[k_idx]
        dt_k = dt_val

        beta_k = cov_k / dt_k
        cond_var_k = max(var_k - cov_k ** 2 / dt_k, 0.0)
        cond_std_k = np.sqrt(cond_var_k)

        n_valid = n_steps - k + 1
        if n_valid <= 0:
            continue

        dW_lagged = dW_y[:, :n_valid]  # (n_paths, n_valid)

        if cond_std_k > 0.0:
            eps = rng.standard_normal((n_paths, n_valid)) * cond_std_k
        else:
            eps = np.zeros((n_paths, n_valid))

        W_near_k = beta_k * dW_lagged + eps  # (n_paths, n_valid)
        Y[:, k : n_steps + 1] += scale * W_near_k

    return Y
