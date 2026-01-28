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


@dataclass(frozen=True)
class AsianArithmeticOption(Instrument):
    '''
    Arithmetic Asian option on spot.

    Observation times can be arbitrary; spot is sampled via paths.spot_at(...)
    using the chosen interpolation method.
    '''
    maturity: float
    strike: float
    callput: Literal["call", "put"] = "call"

    # If None, defaults to averaging over the model grid (exclude t=0 by default)
    obs_times: Optional[ArrayF] = None
    include_t0: bool = False

    # Interpolation for off-grid sampling, default "linear".
    interp: Literal["ladder", "linear"] = "linear"
    tol: float = 1e-12

    def payoff(self, paths: PathBundle) -> ArrayF:
        # Optional strictness: ensure maturity matches last time of simulated grid
        T_grid = float(paths.t[-1])
        if abs(T_grid - float(self.maturity)) > self.tol:
            raise ValueError(
                f"AsianArithmeticOption maturity={self.maturity} must match last grid time {T_grid} "
                f"within tol={self.tol}."
            )

        if self.obs_times is None:
            # Default: use the grid points (excluding t=0 unless include_t0)
            start = 0 if self.include_t0 else 1
            if start >= len(paths.t):
                raise ValueError("Simulation grid must have at least 2 points to exclude t=0.")
            obs_times = np.asarray(paths.t[start:], dtype=float)
        else:
            obs_times = np.asarray(self.obs_times, dtype=float)
            if obs_times.ndim != 1:
                raise ValueError("obs_times must be 1D")
            # Enforce within [0, T] (with tol)
            if np.any(obs_times < float(paths.t[0]) - self.tol) or np.any(obs_times > float(self.maturity) + self.tol):
                raise ValueError("obs_times must lie within the simulated time interval [t0, maturity].")
            # Sort and enforce strictly increasing (typical Asian definition)
            if np.any(np.diff(obs_times) <= 0):
                raise ValueError("obs_times must be strictly increasing.")

        S_obs = paths.spot_at(obs_times, method=self.interp, tol=self.tol)  # (n_paths, m)
        avg = np.mean(S_obs, axis=1)  # (n_paths,)

        cp = _callput_sign(self.callput)
        return np.maximum(cp * (avg - float(self.strike)), 0.0)
