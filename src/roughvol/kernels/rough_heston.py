from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import gamma

from roughvol.types import ArrayF


def rough_heston_kernel(t: ArrayF, hurst: float) -> ArrayF:
    """Evaluate the rough Heston Volterra kernel K(t) = t^(H-1/2) / Gamma(H+1/2).

    Parameters
    ----------
    t     : array of positive time values
    hurst : Hurst parameter H in (0, 0.5)

    Returns
    -------
    K(t) values, same shape as t. Entries with t <= 0 are set to 0.
    """
    H = float(hurst)
    if not (0.0 < H < 0.5):
        raise ValueError("hurst must lie in (0, 0.5).")

    t = np.asarray(t, dtype=float)
    result = np.where(t > 0.0, np.power(np.maximum(t, 0.0), H - 0.5) / gamma(H + 0.5), 0.0)
    return result


def markovian_lift_weights(
    hurst: float,
    n_factors: int = 8,
    t_min: float = 1e-4,
    t_max: float = 10.0,
    n_quad: int = 500,
) -> tuple[ArrayF, ArrayF]:
    """Fit a sum-of-exponentials approximation to the rough Heston kernel.

    Approximates  K(t) = t^(H-1/2) / Gamma(H+1/2)  by

        K_N(t) = sum_{m=1}^{N} w_m * exp(-x_m * t),

    where x_m are geometrically-spaced mean-reversion speeds and w_m >= 0 are
    found by non-negative least squares on a dense log-spaced quadrature grid.

    The resulting (w_m, x_m) define the Markovian lift of the rough Heston model
    (Abi Jaber & El Euch, 2019).

    Parameters
    ----------
    hurst     : Hurst parameter H in (0, 0.5)
    n_factors : number of exponential factors N; default 8
    t_min     : lower bound of the fitting range (log-spaced); default 1e-4
    t_max     : upper bound of the fitting range; default 10.0
    n_quad    : number of quadrature points for the LS fit; default 500

    Returns
    -------
    w : (n_factors,) non-negative weights
    x : (n_factors,) positive mean-reversion speeds (fixed, geometric grid)
    """
    H = float(hurst)
    if not (0.0 < H < 0.5):
        raise ValueError("hurst must lie in (0, 0.5).")
    if n_factors < 1:
        raise ValueError("n_factors must be >= 1.")

    # Geometric grid for mean-reversion speeds
    x = np.exp(np.linspace(np.log(1.0 / t_max), np.log(1.0 / t_min), n_factors))

    # Dense log-spaced quadrature points for the LS fit
    t_quad = np.exp(np.linspace(np.log(t_min), np.log(t_max), n_quad))
    K_target = rough_heston_kernel(t_quad, H)  # (n_quad,)

    # Design matrix: A[i, m] = exp(-x_m * t_quad[i])
    A = np.exp(-x[None, :] * t_quad[:, None])  # (n_quad, n_factors)

    # Non-negative least squares: min ||A w - K_target||^2, w >= 0
    result = lsq_linear(A, K_target, bounds=(0.0, np.inf), method="bvls")
    w = np.maximum(result.x, 0.0)  # enforce non-negativity

    return w, x
