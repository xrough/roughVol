'''
A vanilla option is: European, Call or Put, Payoff depends only on the price at maturity.
'''

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from roughvol.types import ArrayF, PathBundle


@dataclass(frozen=True)
class VanillaOption:
    '''
    European vanilla option (call or put).
    '''
    strike: float
    maturity: float # maturity should belong to the derivative instead of the asset.
    is_call: bool = True

    def payoff(self, paths: PathBundle) -> ArrayF:
        '''
        PathBundle-native payoff.
        Uses terminal spot extracted from paths.spot_T.
        '''
        
        spot_T = np.asarray(paths.spot_T, dtype=float)  # PathBundle provides spot_T :contentReference[oaicite:3]{index=3}
        return self.payoff_terminal(spot_T)

    def payoff_terminal(self, spot_T: ArrayF) -> ArrayF:
        '''
        Terminal-only payoff (legacy-compatible).
        '''
        spot_T = np.asarray(spot_T, dtype=float)

        # Allow either shape (n_paths,) or (n_paths, 1) and normalize
        if spot_T.ndim == 2 and spot_T.shape[1] == 1:
            spot_T = spot_T[:, 0]
        if spot_T.ndim != 1:
            raise ValueError("spot_T must be a 1D array of terminal spot values.")

        k = float(self.strike)
        if self.is_call:
            return np.maximum(spot_T - k, 0.0)
        return np.maximum(k - spot_T, 0.0)
