'''
A vanilla option is: European, Call or Put, Payoff depends only on the price at maturity.
'''

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from roughvol.types import ArrayF


@dataclass(frozen=True)
class VanillaOption:
    '''
    European vanilla option (call or put).
    '''
    strike: float
    maturity: float
    is_call: bool = True

    def payoff(self, spot_T: ArrayF) -> ArrayF: # self: option哑变量
        spot_T = np.asarray(spot_T, dtype=float) # 向量化 -> 数值稳定
        if spot_T.ndim != 1:
            raise ValueError("spot_T must be a 1D array of terminal spot values.")
        if self.is_call:
            return np.maximum(spot_T - self.strike, 0.0)
        return np.maximum(self.strike - spot_T, 0.0)
