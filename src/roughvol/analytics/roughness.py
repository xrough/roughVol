from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from roughvol.sim.volterra import simulate_exact
from roughvol.types import make_rng


@dataclass(frozen=True)
class RoughnessEstimate:
    """Log-log structure-function fit used as a simple roughness proxy."""

    hurst: float
    intercept: float
    r_squared: float
    lags: np.ndarray
    structure_function: np.ndarray
    fitted_structure_function: np.ndarray


def _session_keys(index: pd.Index) -> pd.Index:
    if not isinstance(index, pd.DatetimeIndex):
        return index
    session_index = index.tz_convert(None) if index.tz is not None else index
    return session_index.normalize()


def _time_of_day_keys(index: pd.Index) -> pd.Index:
    if not isinstance(index, pd.DatetimeIndex):
        return index
    clock_index = index.tz_convert(None) if index.tz is not None else index
    seconds = (
        clock_index.hour.to_numpy() * 3600
        + clock_index.minute.to_numpy() * 60
        + clock_index.second.to_numpy()
    )
    return pd.Index(seconds)


def log_returns_from_close(
    close: pd.Series,
    *,
    session_aware: bool = True,
) -> pd.Series:
    """Compute close-to-close log returns, optionally resetting at each session."""
    prices = pd.Series(close, copy=False).astype(float)
    if (prices <= 0.0).any():
        raise ValueError("close prices must be strictly positive")

    log_prices = np.log(prices)
    if session_aware and isinstance(prices.index, pd.DatetimeIndex):
        session_keys = _session_keys(prices.index)
        log_returns = log_prices.groupby(session_keys).diff()
    else:
        log_returns = log_prices.diff()
    return log_returns.dropna()


def deseasonalize_intraday_returns(log_returns: pd.Series) -> pd.Series:
    """Remove deterministic intraday time-of-day scale effects from returns."""
    returns = pd.Series(log_returns, copy=False).astype(float).dropna()
    if returns.empty or not isinstance(returns.index, pd.DatetimeIndex):
        return returns

    bucket_keys = _time_of_day_keys(returns.index)
    seasonal_scale = returns.groupby(bucket_keys).transform(lambda s: s.std(ddof=0))
    positive_scale = seasonal_scale[(seasonal_scale > 0.0) & np.isfinite(seasonal_scale)]
    fallback_scale = float(positive_scale.median()) if not positive_scale.empty else 1.0
    seasonal_scale = seasonal_scale.where((seasonal_scale > 0.0) & np.isfinite(seasonal_scale), fallback_scale)
    return returns / seasonal_scale


def realized_volatility_proxy(
    close: pd.Series,
    window: int = 21,
    annualization: float = 252.0,
    session_aware: bool = True,
) -> pd.Series:
    """Estimate annualized realized volatility from closing prices."""
    if window < 2:
        raise ValueError("window must be at least 2")
    if annualization <= 0.0:
        raise ValueError("annualization must be positive")

    prices = pd.Series(close, copy=False).astype(float)
    if (prices <= 0.0).any():
        raise ValueError("close prices must be strictly positive")

    log_prices = np.log(prices)
    if session_aware and isinstance(prices.index, pd.DatetimeIndex):
        session_keys = _session_keys(prices.index)
        log_returns = log_prices.groupby(session_keys).diff()
        realized = log_returns.groupby(session_keys).transform(
            lambda s: s.rolling(window=window, min_periods=window).std(ddof=0)
        )
    else:
        log_returns = log_prices.diff()
        realized = log_returns.rolling(window=window, min_periods=window).std(ddof=0)

    realized = realized * np.sqrt(annualization)
    return realized.dropna()


def local_volatility_proxy(
    close: pd.Series,
    *,
    window: int,
    annualization: float,
    session_aware: bool = True,
    deseasonalize_intraday: bool = True,
) -> pd.Series:
    """Estimate a high-frequency local volatility proxy from close prices."""
    if window < 2:
        raise ValueError("window must be at least 2")
    if annualization <= 0.0:
        raise ValueError("annualization must be positive")

    log_returns = log_returns_from_close(close, session_aware=session_aware)
    if deseasonalize_intraday:
        log_returns = deseasonalize_intraday_returns(log_returns)
    if log_returns.empty:
        return pd.Series(dtype=float)

    if session_aware and isinstance(log_returns.index, pd.DatetimeIndex):
        session_keys = _session_keys(log_returns.index)
        local_vol = log_returns.groupby(session_keys).transform(
            lambda s: s.rolling(window=window, min_periods=window).std(ddof=0)
        )
    else:
        local_vol = log_returns.rolling(window=window, min_periods=window).std(ddof=0)

    return (local_vol * np.sqrt(annualization)).dropna()


def realized_variance_blocks(
    close: pd.Series,
    *,
    block_size: int,
    annualization: float,
    session_aware: bool = True,
    deseasonalize_intraday: bool = True,
) -> pd.DataFrame:
    """Build non-overlapping realized-variance blocks from close prices."""
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    if annualization <= 0.0:
        raise ValueError("annualization must be positive")

    log_returns = log_returns_from_close(close, session_aware=session_aware)
    if deseasonalize_intraday:
        log_returns = deseasonalize_intraday_returns(log_returns)
    if log_returns.empty:
        return pd.DataFrame(
            columns=[
                "raw_realized_variance",
                "annualized_variance",
                "annualized_volatility",
                "n_returns",
                "session",
                "block_in_session",
            ]
        )

    rows: list[dict[str, float | int | pd.Timestamp]] = []

    if session_aware and isinstance(log_returns.index, pd.DatetimeIndex):
        grouped_returns = log_returns.groupby(_session_keys(log_returns.index))
    else:
        grouped_returns = [(None, log_returns)]

    for session_label, session_returns in grouped_returns:
        session_returns = session_returns.dropna()
        if session_returns.empty:
            continue

        n_blocks = len(session_returns) // block_size
        for block_idx in range(n_blocks):
            start = block_idx * block_size
            end = start + block_size
            block = session_returns.iloc[start:end]
            raw_rv = float(np.sum(np.square(block.to_numpy())))
            annualized_variance = raw_rv * annualization / block_size
            rows.append(
                {
                    "timestamp": block.index[-1],
                    "raw_realized_variance": raw_rv,
                    "annualized_variance": annualized_variance,
                    "annualized_volatility": float(np.sqrt(annualized_variance)),
                    "n_returns": int(block_size),
                    "session": session_label,
                    "block_in_session": int(block_idx),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "raw_realized_variance",
                "annualized_variance",
                "annualized_volatility",
                "n_returns",
                "session",
                "block_in_session",
            ]
        )

    rv_blocks = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return rv_blocks


def estimate_hurst_exponent(
    series: pd.Series | np.ndarray,
    *,
    min_lag: int = 1,
    max_lag: int = 32,
) -> RoughnessEstimate:
    """Estimate H via E|X(t+lag)-X(t)| ~ lag^H on a 1D series."""
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]

    if min_lag < 1:
        raise ValueError("min_lag must be at least 1")
    if max_lag <= min_lag:
        raise ValueError("max_lag must exceed min_lag")
    if values.size <= max_lag:
        raise ValueError("series is too short for the requested lag range")

    lags = np.arange(min_lag, max_lag + 1, dtype=int)
    structure = np.empty_like(lags, dtype=float)

    for idx, lag in enumerate(lags):
        increments = values[lag:] - values[:-lag]
        structure[idx] = float(np.mean(np.abs(increments)))

    if np.any(structure <= 0.0):
        raise ValueError("structure function must be strictly positive")

    log_lags = np.log(lags.astype(float))
    log_structure = np.log(structure)
    slope, intercept = np.polyfit(log_lags, log_structure, deg=1)
    fitted_log_structure = intercept + slope * log_lags

    ss_res = float(np.sum((log_structure - fitted_log_structure) ** 2))
    ss_tot = float(np.sum((log_structure - np.mean(log_structure)) ** 2))
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    return RoughnessEstimate(
        hurst=float(slope),
        intercept=float(intercept),
        r_squared=r_squared,
        lags=lags.astype(float),
        structure_function=structure,
        fitted_structure_function=np.exp(fitted_log_structure),
    )


def simulate_lognormal_vol_paths(
    *,
    hurst: float,
    n_steps: int,
    horizon: float,
    initial_vol: float,
    vol_of_vol: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate matched rough and Brownian lognormal volatility paths."""
    if not (0.0 < hurst < 0.5):
        raise ValueError("hurst must lie in (0, 0.5)")
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if horizon <= 0.0:
        raise ValueError("horizon must be positive")
    if initial_vol <= 0.0:
        raise ValueError("initial_vol must be positive")
    if vol_of_vol <= 0.0:
        raise ValueError("vol_of_vol must be positive")

    t = np.linspace(0.0, horizon, n_steps + 1)
    rng = make_rng(seed)

    rough_driver, _ = simulate_exact(
        t=t,
        hurst=hurst,
        n_paths=1,
        antithetic=False,
        rng=rng,
    )
    rough_driver = rough_driver[0]

    dt = np.diff(t)
    brownian_increments = rng.normal(loc=0.0, scale=np.sqrt(dt), size=n_steps)
    brownian_driver = np.concatenate([[0.0], np.cumsum(brownian_increments)])

    log_initial = np.log(initial_vol)
    rough_log_vol = (
        log_initial
        + vol_of_vol * rough_driver
        - 0.5 * vol_of_vol**2 * np.power(t, 2.0 * hurst)
    )
    brownian_log_vol = (
        log_initial
        + vol_of_vol * brownian_driver
        - 0.5 * vol_of_vol**2 * t
    )

    return t, np.exp(rough_log_vol), np.exp(brownian_log_vol)
