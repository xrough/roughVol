from __future__ import annotations

import math

import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import gamma, roots_legendre

from roughvol.types import ArrayF


def rough_heston_kernel(t: ArrayF, hurst: float) -> ArrayF:
    """Evaluate the rough Heston Volterra kernel K(t) = t^(H-1/2) / Gamma(H+1/2)."""
    H = float(hurst)
    if not (0.0 < H < 0.5):
        raise ValueError("hurst must lie in (0, 0.5).")
    t = np.asarray(t, dtype=float)
    return np.where(t > 0.0, np.power(np.maximum(t, 0.0), H - 0.5) / gamma(H + 0.5), 0.0)


def _bayer_breneis_weights(hurst: float, n_factors: int) -> tuple[ArrayF, ArrayF]:
    """Bayer-Breneis (2024) spectral quadrature for the Markovian lift.

    Approximates K(t) = t^(H-½)/Γ(H+½) using the exact spectral representation

        K(t) = c_H · ∫₀^∞ x^(-H-½) · e^{-xt} dx,   c_H = 1/(Γ(H+½)·Γ(½-H))

    The integral is discretised by Gauss-Legendre quadrature on a sequence of
    geometrically-spaced sub-intervals.  The resulting nodes x_i and weights
    w_i = c_H · x_i^(-H-½) · (GL weight for interval) satisfy

        K(t) ≈ Σ_i w_i · exp(-x_i · t)

    Parameters
    ----------
    hurst     : H ∈ (0, 0.5)
    n_factors : target number of factors N

    Returns
    -------
    w : weights  (non-negative, analytically derived)
    x : nodes    (GL quadrature points, positive)
    """
    H = float(hurst)

    # Optimal constants from Bayer & Breneis (2024)
    ALPHA = 1.06418
    BETA  = 0.4275

    c_H = 1.0 / (gamma(H + 0.5) * gamma(0.5 - H))

    num_intervals       = math.floor(ALPHA * math.sqrt(n_factors))
    points_per_interval = max(1, math.floor(BETA * math.sqrt(n_factors)))
    q                   = math.exp(1.0)   # fixed log-step; domain expands as M grows

    gl_nodes, gl_weights = roots_legendre(points_per_interval)

    nodes, weights = [], []
    for j in range(-num_intervals, num_intervals + 1):
        lower = q ** (j - 1)
        upper = q ** j
        mid   = 0.5 * (upper + lower)
        half  = 0.5 * (upper - lower)

        for k in range(points_per_interval):
            x_k = mid + half * gl_nodes[k]
            w_k = half * gl_weights[k] * c_H * (x_k ** (-H - 0.5))
            nodes.append(x_k)
            weights.append(w_k)

    x = np.array(nodes[:n_factors])
    w = np.array(weights[:n_factors])
    # Sort by ascending speed so the factor ordering is consistent
    order = np.argsort(x)
    return w[order], x[order]


def _nnls_weights(
    hurst: float,
    n_factors: int,
    t_min: float = 1e-4,
    t_max: float = 10.0,
    n_quad: int = 500,
) -> tuple[ArrayF, ArrayF]:
    """Original NNLS fit on a fixed geometric grid (kept for comparison)."""
    H = float(hurst)
    x = np.exp(np.linspace(np.log(1.0 / t_max), np.log(1.0 / t_min), n_factors))
    t_quad  = np.exp(np.linspace(np.log(t_min), np.log(t_max), n_quad))
    K_target = rough_heston_kernel(t_quad, H)
    A = np.exp(-x[None, :] * t_quad[:, None])
    result = lsq_linear(A, K_target, bounds=(0.0, np.inf), method="bvls")
    return np.maximum(result.x, 0.0), x


def markovian_lift_weights(
    hurst: float,
    n_factors: int = 8,
    method: str = "bayer-breneis",
    **kwargs,
) -> tuple[ArrayF, ArrayF]:
    """Weights and nodes for the Markovian lift of the rough Heston kernel.

    Approximates  K(t) = t^(H-½)/Γ(H+½)  by

        K_N(t) ≈ Σ_{m=1}^{N} w_m · exp(-x_m · t)

    Parameters
    ----------
    hurst     : Hurst parameter H ∈ (0, 0.5)
    n_factors : number of exponential factors N
    method    : ``"nnls"`` (default) — NNLS pointwise fit; best for N ≤ ~16.
                ``"bayer-breneis"`` — spectral GL quadrature; super-polynomial
                convergence in N, overtakes NNLS around N=32.

    Returns
    -------
    w : (N,) non-negative weights
    x : (N,) positive mean-reversion speeds
    """
    H = float(hurst)
    if not (0.0 < H < 0.5):
        raise ValueError("hurst must lie in (0, 0.5).")
    if n_factors < 1:
        raise ValueError("n_factors must be >= 1.")

    if method == "nnls":
        return _nnls_weights(H, n_factors, **kwargs)
    elif method == "bayer-breneis":
        return _bayer_breneis_weights(H, n_factors)
    else:
        raise ValueError(f"Unknown method {method!r}. Choose 'nnls' or 'bayer-breneis'.")
