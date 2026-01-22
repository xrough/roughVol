"""
Type定义了class, PriceResult是定价输出, instrument是被定价的模型(定义为protocol保证灵活性), 
PathModel定义各个过程. 
"""

from __future__ import annotations #避免循环reference

from dataclasses import dataclass # 数据载体：class
from typing import Protocol # Protocol是特殊的class，只规定了必要的元素
from typing import runtime_checkable
import numpy as np

ArrayF = np.ndarray #简写


@dataclass(frozen=True)
class PriceResult:
    """Standard output container for pricing engines."""
    price: float
    stderr: float # MC标准误差
    ci95: tuple # MC标准误差
    n_paths: int
    n_steps: int
    seed: int | None = None


@runtime_checkable
class Instrument(Protocol): # Protocol规定class必须要满足的条件，但可加入更多。
    '''Contract: an instrument defines payoff from terminal spot values.'''

    maturity: float  #结算时间

    def payoff(self, spot_T: ArrayF) -> ArrayF:
        '''Return payoff for each terminal spot in `spot_T`.

        spot_T: shape (n_paths,)
        returns: shape (n_paths,)'''
        
        ...


@runtime_checkable
class PathModel(Protocol):
    '''Contract: a model can simulate spot paths on a uniform time grid.'''

    def simulate_paths(
        self,
        *,
        n_paths: int,
        n_steps: int,
        maturity: float,
        rng: np.random.Generator, # 随机生成子
    ) -> ArrayF:
        '''
        Return spot paths including S0.
        returns shape (n_paths, n_steps + 1)
        '''
        ...

@runtime_checkable
# AntitheticPathModel是PathModel，同时还多一个simulate_paths_antithetic()。
class AntitheticPathModel(PathModel, Protocol): 
    '''
    Optional capability: model can simulate antithetic paths in one call.
    The model guarantees paths are produced using paired (Z, -Z) construction.
    '''
    def simulate_paths_antithetic(
        self,
        *,
        n_paths: int,
        n_steps: int,
        maturity: float,
        rng: np.random.Generator,
    ) -> ArrayF:
        ...