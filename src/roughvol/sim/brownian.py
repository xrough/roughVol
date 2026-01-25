from __future__ import annotations

import numpy as np
from roughvol.types import ArrayF

def time_grid(*, maturity: float, n_steps: int) -> ArrayF:
    '''
    Uniform grid from 0 to maturity (inclusive). Python中*之后的所有输入必须通过名称而非位置. 
    '''
    if maturity < 0:
        raise ValueError("maturity must be non-negative.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    # Allow maturity == 0: grid is just [0.0]
    if maturity == 0.0:
        return np.array([0.0], dtype=float)

    return np.linspace(0.0, float(maturity), int(n_steps) + 1)


def brownian_increments(
    *,
    n_paths: int,
    n_steps: int,
    dt: float,
    rng: np.random.Generator, #将rng作为input是为了统一seed，保持随机生成一致.
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
    if dt < 0:
        raise ValueError("dt must be non-negative.")

    z = rng.standard_normal(size=(int(n_paths), int(n_steps))) #随机的矩阵，每个坐标都normal取样.
    return np.sqrt(float(dt)) * z

# isolate antithetic in the BM increment function.
def brownian_increments_antithetic(
    *,
    n_paths: int,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> ArrayF:
    '''
    Generate antithetic Brownian increments.

    Uses half paths dW and mirrors with -dW.
    Requires n_paths even.

    Returns shape (n_paths, n_steps).
    
    这里我们不仅仅在maturity T flip，是因为一般的derivative不仅依赖于maturity的asset price。
    '''
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if dt < 0:
        raise ValueError("dt must be non-negative.")
    if n_paths % 2 != 0:
        raise ValueError("n_paths must be even for antithetic increments.")

    half = n_paths // 2
    dW_half = brownian_increments(n_paths=half, n_steps=n_steps, dt=dt, rng=rng)
    return np.vstack([dW_half, -dW_half])
