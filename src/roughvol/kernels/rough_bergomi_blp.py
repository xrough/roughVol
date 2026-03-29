from __future__ import annotations

import numpy as np

from roughvol.types import ArrayF


def rough_bergomi_blp_driver(
    dW: ArrayF,
    t: ArrayF,
    hurst: float,
    kappa: int = 10,
    rng: np.random.Generator | None = None,
) -> ArrayF:
    """Simulate the Volterra driver Ytilde for rough Bergomi using the BLP hybrid scheme.

    Implements the Bennedsen-Lunde-Pakkanen (2017) hybrid scheme, which splits the
    Volterra integral into:
      - Near field (lags k = 1,...,kappa): exact bivariate Gaussian sampling of the
        stochastic integral over each cell, accounting for the singularity correctly.
      - Far field (lags k > kappa): midpoint-quadrature step function approximation,
        computed via FFT linear convolution for O(n log n) cost.

    Parameters
    ----------
    dW    : (n_paths, n_steps) array of Brownian increments for the variance driver
    t     : 1D time grid including t_0 = 0, shape (n_steps + 1,)
    hurst : Hurst parameter H in (0, 0.5)
    kappa : near-field cutoff (number of lags handled exactly); default 10
    rng   : numpy Generator used to sample the near-field stochastic integrals;
            if None, uses np.random.default_rng()

    Returns
    -------
    Y : (n_paths, n_steps + 1) Volterra driver paths with Y[:, 0] = 0
    """
    H = float(hurst)
    alpha = H - 0.5  # kernel exponent; alpha in (-0.5, 0)

    t = np.asarray(t, dtype=float)
    dW = np.asarray(dW, dtype=float)
    n_paths, n_steps = dW.shape

    if t.size != n_steps + 1:
        raise ValueError("t must have n_steps + 1 elements.")
    if not (0.0 < H < 0.5):
        raise ValueError("hurst must lie in (0, 0.5).")
    if kappa < 0:
        raise ValueError("kappa must be non-negative.")

    if rng is None:
        rng = np.random.default_rng()

    dt = np.diff(t)
    scale = np.sqrt(2.0 * H)  # L_g prefactor

    # ------------------------------------------------------------------
    # Pre-compute near-field covariances for lags k = 1, ..., min(kappa, n_steps)
    # For lag k the interval is [t_{i-k}, t_{i-k+1}].
    # Stochastic integral  W^n_{i-k,k} = integral_{t_{i-k}}^{t_{i-k+1}} (t_i - s)^alpha dW_s
    #
    # For a UNIFORM grid with step dt_val:
    #   Var(W^n_{..}) = integral_{t_{i-k}}^{t_{i-k+1}} (t_i-s)^(2*alpha) ds
    #                 = [ (k*dt_val)^(2H) - ((k-1)*dt_val)^(2H) ] / (2H)
    #   Cov(W^n_{..}, dW_{i-k}) = integral_{t_{i-k}}^{t_{i-k+1}} (t_i-s)^alpha ds
    #                            = [ (k*dt_val)^(alpha+1) - ((k-1)*dt_val)^(alpha+1) ] / (alpha+1)
    #
    # For simplicity we use the first step's dt for non-uniform grids (good approximation).
    # ------------------------------------------------------------------
    dt_val = dt[0]  # reference step size (handles approximately non-uniform grids)
    k_near = min(kappa, n_steps)

    near_var = np.empty(k_near)
    near_cov = np.empty(k_near)
    for k in range(1, k_near + 1):
        near_var[k - 1] = ((k * dt_val) ** (2 * H) - ((k - 1) * dt_val) ** (2 * H)) / (2 * H)
        near_cov[k - 1] = (
            (k * dt_val) ** (alpha + 1) - ((k - 1) * dt_val) ** (alpha + 1)
        ) / (alpha + 1)

    # ------------------------------------------------------------------
    # Far-field: midpoint weights for lags k = kappa+1, ..., n_steps
    # w_k = sqrt(2H) * (t_i - mid_{i-k})^alpha  ≈  sqrt(2H) * (k * dt_val)^alpha
    # (we use the midpoint of the k-th lag interval relative to t_i)
    # ------------------------------------------------------------------
    if k_near < n_steps:
        lags_far = np.arange(k_near + 1, n_steps + 1, dtype=float)  # k = kappa+1,...,n
        # Midpoint of interval [t_{i-k}, t_{i-k+1}] relative to t_i ≈ (k - 0.5) * dt_val
        mids_far = (lags_far - 0.5) * dt_val
        weights_far = scale * np.power(mids_far, alpha)  # (n_far,)
    else:
        weights_far = np.empty(0)

    # ------------------------------------------------------------------
    # Accumulate Volterra driver Y (n_paths, n_steps+1)
    # Y[:, 0] = 0 by convention; Y[:, i] corresponds to time t_i for i >= 1.
    # ------------------------------------------------------------------
    Y = np.zeros((n_paths, n_steps + 1), dtype=float)

    # ---- Far-field via FFT convolution ----
    # Y_far[i] = sum_{k=k_near+1}^{i} weights_far[k - k_near - 1] * dW[:, i-k]
    # This is a causal convolution of dW with weights_far (padded with zeros).
    if weights_far.size > 0:
        n_fft = int(2 ** np.ceil(np.log2(2 * n_steps)))  # next power of 2 for efficiency
        # Pad weights_far and dW for FFT linear convolution
        w_pad = np.zeros(n_fft)
        w_pad[: weights_far.size] = weights_far

        dW_pad = np.zeros((n_paths, n_fft))
        dW_pad[:, :n_steps] = dW

        W_fft = np.fft.rfft(w_pad)          # (n_fft//2+1,)
        DW_fft = np.fft.rfft(dW_pad, axis=1)  # (n_paths, n_fft//2+1)

        conv = np.fft.irfft(W_fft[None, :] * DW_fft, n=n_fft, axis=1)  # (n_paths, n_fft)
        # conv[:, i-1] = sum_{k=0}^{i-1} weights_far[k] * dW[:, i-1-k]
        # We want Y_far for time index i (i=1,...,n_steps), which uses lags k_near+1,...,i:
        # Y_far[:, i] = sum_{k=k_near+1}^{i} weights_far[k-k_near-1] * dW[:, i-k]
        #             = conv[:, i - k_near - 1]  (offset by k_near+1)
        for i in range(k_near + 1, n_steps + 1):
            idx = i - k_near - 1
            if 0 <= idx < n_steps:
                Y[:, i] += conv[:, idx]

    # ---- Near-field: bivariate Gaussian sampling ----
    # For each lag k and each time i >= k, sample the stochastic integral W^n_{i-k,k}
    # jointly with dW[:, i-k] using the precomputed covariance.
    for k_idx in range(k_near):
        k = k_idx + 1  # lag 1-indexed
        var_k = near_var[k_idx]
        cov_k = near_cov[k_idx]
        dt_k = dt_val

        # Joint covariance [[var_k, cov_k], [cov_k, dt_k]]
        # Conditional: W^n_{..} | dW_{i-k} ~ N(cov_k/dt_k * dW_{i-k}, var_k - cov_k^2/dt_k)
        beta_k = cov_k / dt_k  # regression coefficient
        cond_var_k = max(var_k - cov_k ** 2 / dt_k, 0.0)
        cond_std_k = np.sqrt(cond_var_k)

        # Valid time indices for this lag: i = k, k+1, ..., n_steps
        n_valid = n_steps - k + 1
        if n_valid <= 0:
            continue

        # dW[:, i-k] for i = k,...,n_steps  →  dW[:, 0:n_valid]  (columns 0..n_valid-1)
        dW_lagged = dW[:, :n_valid]  # (n_paths, n_valid)

        # Sample residuals ~ N(0, cond_var_k)
        if cond_std_k > 0.0:
            eps = rng.standard_normal((n_paths, n_valid)) * cond_std_k
        else:
            eps = np.zeros((n_paths, n_valid))

        W_near_k = beta_k * dW_lagged + eps  # (n_paths, n_valid)

        # Accumulate into Y[:, k:n_steps+1] with the L_g = sqrt(2H) factor
        Y[:, k : n_steps + 1] += scale * W_near_k

    return Y
