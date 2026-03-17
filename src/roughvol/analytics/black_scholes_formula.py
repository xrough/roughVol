'''
The Black–Scholes formula prices a European option as the discounted expected payoff at maturity, 
assuming lognormal asset prices with constant volatility.

Benchmark implementation for European option pricing and implied volatility calculation.
'''

from __future__ import annotations

import math
from scipy.stats import norm # normal distribution.

def bs_price(*, spot: float, strike: float, maturity: float, rate: float, div: float, vol: float, is_call: bool) -> float:
    '''
    Closed-form Black–Scholes price with continuous dividend yield.
    '''
    if maturity <= 0:
        return max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0) # intrisic value，即在maturity完成了交易或作废option。
    if vol < 0:
        raise ValueError("vol must be non-negative.")
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive.")

    if vol == 0.0:
        fwd = spot * math.exp((rate - div) * maturity)
        disc = math.exp(-rate * maturity)
        return disc * max(fwd - strike, 0.0) if is_call else disc * max(strike - fwd, 0.0)

    sqrtT = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol * vol) * maturity) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT
    disc_r = math.exp(-rate * maturity)
    disc_q = math.exp(-div * maturity)

    if is_call:
        return spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2)
    return strike * disc_r * norm.cdf(-d2) - spot * disc_q * norm.cdf(-d1) #经典BS公式


def bs_delta(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div: float,
    vol: float,
    is_call: bool,
) -> float:
    """Closed-form Black-Scholes delta with continuous dividend yield."""
    if maturity <= 0:
        if is_call:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    if vol < 0:
        raise ValueError("vol must be non-negative.")
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive.")
    if vol == 0.0:
        fwd = spot * math.exp((rate - div) * maturity)
        if is_call:
            return math.exp(-div * maturity) if fwd > strike else 0.0
        return -math.exp(-div * maturity) if fwd < strike else 0.0

    sqrtT = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol * vol) * maturity) / (
        vol * sqrtT
    )
    disc_q = math.exp(-div * maturity)
    if is_call:
        return disc_q * norm.cdf(d1)
    return disc_q * (norm.cdf(d1) - 1.0)


'''
Implied volatility calculation via bisection method, solvable since BS price is monotonic in vol.
'''

def implied_vol(*, price: float, spot: float, strike: float, maturity: float, rate: float, div: float, is_call: bool) -> float:
    '''
    Implied vol by bisection (robust, no Greeks required). Monotonicity used.
    '''
    if price < 0:
        raise ValueError("price must be non-negative.")
    if maturity <= 0:
        return 0.0

    disc_r = math.exp(-rate * maturity)
    disc_q = math.exp(-div * maturity)

    # no-arbitrage bounds: upper <-> inf. vol. lower <-> zero vol.
    if is_call:
        lower = max(0.0, spot * disc_q - strike * disc_r)
        upper = spot * disc_q
    else:
        lower = max(0.0, strike * disc_r - spot * disc_q)
        upper = strike * disc_r

    if not (lower - 1e-12 <= price <= upper + 1e-12): # 1e-12 数值稳定
        raise ValueError(f"price out of bounds: price={price}, bounds=({lower},{upper})") # An f-string lets you embed variables directly into a string.

    lo, hi = 1e-8, 5.0  # 500% cap
    for _ in range(120): # _代表我们不在乎这个variable
        mid = 0.5 * (lo + hi)
        pmid = bs_price(spot=spot, strike=strike, maturity=maturity, rate=rate, div=div, vol=mid, is_call=is_call)
        if pmid > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
