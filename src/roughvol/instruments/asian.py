# Instrument: Asian Options.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from roughvol.types import ArrayF, Instrument, PathBundle


def _callput_sign(callput: Literal["call", "put"]) -> float:
    if callput == "call":
        return 1.0
    if callput == "put":
        return -1.0
    raise ValueError(f"callput must be 'call' or 'put', got {callput!r}")


def _times_to_indices(t_grid: ArrayF, obs_times: ArrayF, tol: float = 1e-12) -> np.ndarray:
    '''
    Map observation times to indices in t_grid.
    Requires obs_times to lie on t_grid within tolerance.
    '''
    t_grid = np.asarray(t_grid, dtype=float)
    obs_times = np.asarray(obs_times, dtype=float)

    if obs_times.ndim != 1:
        raise ValueError("obs_times must be 1D")
    if t_grid.ndim != 1:
        raise ValueError("t_grid must be 1D")

    # For each obs time, find nearest grid index
    idx = np.searchsorted(t_grid, obs_times, side="left")
    idx = np.clip(idx, 0, len(t_grid) - 1)

    # Check whether left neighbor is closer
    left = np.clip(idx - 1, 0, len(t_grid) - 1)
    choose_left = np.abs(t_grid[left] - obs_times) < np.abs(t_grid[idx] - obs_times)
    idx = np.where(choose_left, left, idx)

    max_err = float(np.max(np.abs(t_grid[idx] - obs_times))) if len(obs_times) else 0.0
    if max_err > tol:
        raise ValueError(
            f"obs_times must lie on simulation grid within tol={tol}. "
            f"Max |grid-obs|={max_err}."
        )

    # Ensure strictly increasing indices if times increasing
    if np.any(np.diff(idx) <= 0):
        # This catches duplicates and non-increasing schedules
        raise ValueError("obs_times must map to a strictly increasing set of grid times.")

    return idx.astype(int)


@dataclass(frozen=True)
class AsianArithmeticOption(Instrument):
    '''
    Arithmetic Asian option on spot, with pathwise payoff.

    If obs_times is None, uses the model grid from paths.t:
      - exclude t=0 by default (include_t0=False)
    '''
    maturity: float
    strike: float
    callput: Literal["call", "put"] = "call"
    obs_times: Optional[ArrayF] = None
    include_t0: bool = False
    tol: float = 1e-12  # for mapping obs_times to grid

    def payoff(self, paths: PathBundle) -> ArrayF:
        # Basic maturity consistency: require T to be last grid point (common convention)
        T = float(paths.t[-1])
        if abs(T - float(self.maturity)) > self.tol:
            raise ValueError(
                f"AsianArithmeticOption maturity={self.maturity} must match last grid time {T} "
                f"within tol={self.tol}."
            )

        spot = np.asarray(paths.spot, dtype=float)  # (n_paths, n_times)
        t_grid = np.asarray(paths.t, dtype=float)

        if self.obs_times is None:
            start = 0 if self.include_t0 else 1
            if start >= spot.shape[1]:
                raise ValueError("Simulation grid must have at least 2 points to exclude t=0.")
            spot_obs = spot[:, start:]  # (n_paths, m)
        else:
            idx = _times_to_indices(t_grid, np.asarray(self.obs_times, dtype=float), tol=self.tol)
            spot_obs = spot[:, idx]  # (n_paths, m)

        avg = np.mean(spot_obs, axis=1)  # (n_paths,)
        cp = _callput_sign(self.callput)
        payoff = np.maximum(cp * (avg - float(self.strike)), 0.0)
        return payoff
