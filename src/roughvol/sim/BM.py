from __future__ import annotations

import numpy as np
from roughvol.types import ArrayF

def time_grid(*, maturity: float, n_steps: int) -> ArrayF:
    '''
    Uniform grid from 0 to maturity (inclusive). Python中*之后的所有输入必须通过名称而非位置. 
    '''
    if maturity <= 0:
        raise ValueError("maturity must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    return np.linspace(0.0, float(maturity), int(n_steps) + 1)


def brownian_increments(
    *,
    n_paths: int,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> ArrayF:
    '''Generate Brownian increments dW ~ Normal(0, dt).

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_steps). Column j is the increment over step j.
    '''
    if n_paths <= 0: #MC模拟轨道次数
        raise ValueError("n_paths must be positive.")
    if n_steps <= 0: #每次模拟的grid
        raise ValueError("n_steps must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    z = rng.standard_normal(size=(int(n_paths), int(n_steps))) #随机的矩阵，每个坐标都normal取样.
    return np.sqrt(float(dt)) * z
