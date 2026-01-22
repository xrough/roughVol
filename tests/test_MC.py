'''
A robustness check of the MC engine.
'''

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

@dataclass
class MCResult:
    price: float
    stderr: float
    ci95: Tuple[float, float] # 95%置信区间
    payoff_mean: float
    payoff_std: float
    n_paths: int
    seed: Optional[int] 

def mc_price_european_gbm(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    payoff_fn: Callable[[np.ndarray, float], np.ndarray], # 这里的callable是代表输入是一个函数，这里相比于直接import而言很方便的一点是这个函数的接入就很自由。
    n_paths: int = 200_000,
    seed: Optional[int] = 123, # 与 seed: int | None 同含义
) -> MCResult:
    
    '''
     Risk-neutral GBM terminal simulation:
      S_T = S0 * exp((r - 0.5*sigma^2)T + sigma*sqrt(T)*Z)
    Returns discounted MC estimator + standard error and 95% CI.

    payoff_fn signature: payoff_fn(ST: np.ndarray, K: float) -> np.ndarray
    '''

    # --- Edge cases / guards ---
    if n_paths <= 1:
        raise ValueError("n_paths must be > 1")

    if T < 0:
        raise ValueError("T must be non-negative")

    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    # If T == 0: option value is intrinsic (discount factor = 1)
    if T == 0:
        ST = np.array([S0], dtype=float) # np.array([]) 向量化了这个scalar，[]把S0转化成list。
        payoff = payoff_fn(ST, K)[0] # payout should be dimension tolerant.
        price = float(payoff)
        return MCResult(
            price=price,
            stderr=0.0,
            ci95=(price, price),
            payoff_mean=price,
            payoff_std=0.0,
            n_paths=1,
            seed=seed,
        )

    # If sigma == 0: deterministic under RN: S_T = S0 * exp(rT)
    if sigma == 0:
        ST_det = S0 * np.exp(r * T)
        payoff = float(payoff_fn(np.array([ST_det]), K)[0])
        price = float(np.exp(-r * T) * payoff)
        return MCResult(
            price=price,
            stderr=0.0,
            ci95=(price, price),
            payoff_mean=float(payoff),
            payoff_std=0.0,
            n_paths=1,
            seed=seed,
        )

    # --- RNG (reproducible) ---
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)

    # --- Vectorized terminal simulation ---
    drift = (r - 0.5 * sigma * sigma) * T
    vol = sigma * np.sqrt(T)
    ST = S0 * np.exp(drift + vol * Z)

    payoff = payoff_fn(ST, K)  # vector of payoffs
    disc = np.exp(-r * T)
    disc_payoff = disc * payoff

    # --- Estimator and diagnostics ---
    price = float(disc_payoff.mean())
    payoff_std = float(disc_payoff.std(ddof=1))
    stderr = payoff_std / np.sqrt(n_paths)
    ci95 = (price - 1.96 * stderr, price + 1.96 * stderr)

    return MCResult(
        price=price,
        stderr=float(stderr),
        ci95=(float(ci95[0]), float(ci95[1])),
        payoff_mean=float(payoff.mean()),
        payoff_std=float(payoff_std),
        n_paths=int(n_paths),
        seed=seed,
    )
