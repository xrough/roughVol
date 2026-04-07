"""Empirical roughness demo backed by yfinance data.

The script combines:
  1. de-seasonalized intraday returns and non-overlapping realized-variance blocks,
  2. a structure-function regression on log realized variance to estimate the Hurst exponent,
  3. the current ATM implied-vol term structure from the listed options chain,
  4. a small matched simulation comparing rough vs Brownian volatility paths.

Run with:
    pip install yfinance
    python -m roughvol.experiments.rough_estimate.run_empirical_roughness_demo
    python -m roughvol.experiments.rough_estimate.run_empirical_roughness_demo SPY AAPL MSFT
    python -m roughvol.experiments.rough_estimate.run_empirical_roughness_demo --hurst-hist-top-n 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    raise ImportError(
        "yfinance is required to run this demo.\n"
        "Install it with:  pip install yfinance"
    ) from exc

from roughvol.analytics.black_scholes_formula import implied_vol
from roughvol.analytics.roughness import (
    RoughnessEstimate,
    _session_keys,
    estimate_hurst_exponent,
    local_volatility_proxy,
    realized_variance_blocks,
    simulate_lognormal_vol_paths,
)
from roughvol.data.yfinance_loader import get_market_data, get_price_history
from roughvol.experiments._paths import output_path
from roughvol.types import MarketData

DEFAULT_TICKERS = ["SPY"]
DEFAULT_PRICE_HISTORY_INTERVAL = "1m"
DEFAULT_RV_BLOCK_SIZE = 30
MAX_LAG = 32
SIMULATION_STEPS = 252
SIMULATION_HORIZON = 1.0
SIMULATION_VOL_OF_VOL = 1.35
REALIZED_VOL_ZOOM_HOURS = 4
LOCAL_VOL_WINDOW_RETURNS = 5
CACHE_VERSION = 1
DEFAULT_CACHE_PATH = output_path("rough_estimate", "empirical_roughness_cache.json")
LARGE_CAP_CANDIDATE_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSM", "AVGO", "TSLA",
    "JPM", "LLY", "WMT", "V", "XOM", "UNH", "ORCL", "MA", "NFLX", "COST",
    "JNJ", "HD", "PG", "ABBV", "BAC", "KO", "SAP", "ASML", "CVX", "TMUS",
    "CRM", "AMD", "NVO", "CSCO", "MRK", "WFC", "PM", "IBM", "MCD", "LIN",
    "AXP", "ABT", "GS", "GE", "DIS", "NOW", "CAT", "TXN", "ISRG", "QCOM",
    "BKNG", "T", "VZ", "RTX", "INTU", "AMGN", "UBER", "PFE", "MS", "SPGI",
    "BLK", "PLTR", "SYK", "NEE", "ADBE", "ETN", "TJX", "HON", "PGR", "SCHW",
    "CMCSA", "MU", "UNP", "VRTX", "BSX", "COP", "C", "PANW", "DE", "ANET",
    "BMY", "LOW", "SBUX", "FI", "AMAT", "MDT", "LRCX", "INTC", "ADI", "MMC",
    "KKR", "GILD", "AMT", "DHR", "CB", "SO", "NKE", "MO", "BA", "ELV",
    "ICE", "MDLZ", "DUK", "CI", "UPS", "MELI", "TT", "REGN", "APH", "TTD",
]


def output_figure_name(kind: str) -> str:
    return output_path("rough_estimate", f"empirical_roughness_{kind}.png")


def stable_seed_from_ticker(ticker_symbol: str) -> int:
    """Create a reproducible but ticker-specific simulation seed."""
    digest = hashlib.blake2b(ticker_symbol.upper().encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % (2**32 - 1)


def cache_key(ticker_symbol: str, *, interval: str, period: str | None, rv_block_size: int) -> str:
    """Build a deterministic key for a cached roughness estimate."""
    resolved_period = "" if period is None else period
    return f"{ticker_symbol.upper()}|{interval.lower()}|{resolved_period}|{rv_block_size}"


def rank_tickers_by_market_cap(market_caps: dict[str, float], top_n: int) -> list[str]:
    """Return tickers ranked by descending market cap."""
    valid_items = [(ticker, cap) for ticker, cap in market_caps.items() if cap > 0.0]
    ranked = sorted(valid_items, key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in ranked[:top_n]]


def default_period_for_interval(interval: str) -> str:
    """Choose a conservative yfinance lookback period for the requested interval."""
    interval = interval.lower()
    if interval == "1m":
        return "7d"
    if interval in {"2m", "5m", "15m", "30m", "60m", "90m", "1h"}:
        return "60d"
    return "5y"


def annualization_for_interval(interval: str) -> float:
    """Approximate annualization factor for the chosen return sampling interval."""
    interval = interval.lower()
    minutes_per_bar = {
        "1m": 1.0,
        "2m": 2.0,
        "5m": 5.0,
        "15m": 15.0,
        "30m": 30.0,
        "60m": 60.0,
        "90m": 90.0,
        "1h": 60.0,
    }
    if interval in minutes_per_bar:
        bars_per_day = 390.0 / minutes_per_bar[interval]
        return 252.0 * bars_per_day
    if interval == "1d":
        return 252.0
    if interval == "5d":
        return 252.0 / 5.0
    if interval in {"1wk", "1w"}:
        return 52.0
    if interval in {"1mo", "3mo"}:
        return 12.0 if interval == "1mo" else 4.0
    return 252.0


def minutes_per_interval(interval: str) -> float | None:
    interval = interval.lower()
    minutes_map = {
        "1m": 1.0,
        "2m": 2.0,
        "5m": 5.0,
        "15m": 15.0,
        "30m": 30.0,
        "60m": 60.0,
        "90m": 90.0,
        "1h": 60.0,
    }
    return minutes_map.get(interval)


def is_intraday_interval(interval: str) -> bool:
    return interval.lower() in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


def block_label(interval: str, block_size: int) -> str:
    interval = interval.lower()
    if interval.endswith("m") or interval == "1h":
        minutes_per_bar = {
            "1m": 1,
            "2m": 2,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "60m": 60,
            "90m": 90,
            "1h": 60,
        }[interval]
        total_minutes = block_size * minutes_per_bar
        if total_minutes % 60 == 0:
            hours = total_minutes // 60
            return f"{hours}h RV blocks"
        return f"{total_minutes}min RV blocks"
    return f"{block_size}-bar RV blocks"


def plot_series_with_session_gaps(
    ax: plt.Axes,
    series: pd.Series,
    *,
    color: str = "midnightblue",
    linewidth: float = 1.2,
    gap: float = 2.0,
) -> None:
    """Plot a DateTime series on a compressed axis with small gaps between sessions."""
    if series.empty:
        return
    if not isinstance(series.index, pd.DatetimeIndex):
        ax.plot(series.index, series.values, color=color, linewidth=linewidth)
        return

    session_keys = _session_keys(series.index)
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    cursor = 0.0

    for session_label, session_series in series.groupby(session_keys):
        n = len(session_series)
        x = cursor + np.arange(n, dtype=float)
        ax.plot(x, session_series.values, color=color, linewidth=linewidth)

        tick_positions.append(float(x[len(x) // 2]))
        tick_labels.append(pd.Timestamp(session_label).strftime("%Y-%m-%d"))
        cursor = float(x[-1] + gap + 1.0)

    max_ticks = 8
    if len(tick_positions) > max_ticks:
        idx = np.linspace(0, len(tick_positions) - 1, max_ticks, dtype=int)
        tick_positions = [tick_positions[i] for i in idx]
        tick_labels = [tick_labels[i] for i in idx]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")


def make_panel_figure(n_panels: int, *, n_cols: int = 2, panel_height: float = 3.8) -> tuple[plt.Figure, np.ndarray]:
    """Create a compact subplot grid and return flattened axes."""
    n_cols = max(1, min(n_cols, n_panels))
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2 * n_cols, panel_height * n_rows))
    axes_arr = np.atleast_1d(axes).ravel()
    return fig, axes_arr


def hide_unused_axes(axes: np.ndarray, used: int) -> None:
    for ax in axes[used:]:
        ax.set_visible(False)


def recent_intraday_zoom_series(
    series: pd.Series,
    *,
    interval: str,
    zoom_hours: int = REALIZED_VOL_ZOOM_HOURS,
) -> pd.Series:
    """Take the most recent couple of trading hours from the latest session."""
    if series.empty or not isinstance(series.index, pd.DatetimeIndex):
        return series

    session_keys = _session_keys(series.index)
    latest_session = session_keys[-1]
    latest_session_series = series[session_keys == latest_session]
    minutes = minutes_per_interval(interval)
    if minutes is None:
        return latest_session_series

    zoom_count = max(10, int(math.ceil(60.0 * zoom_hours / minutes)))
    return latest_session_series.iloc[-zoom_count:]


def get_market_cap(ticker_symbol: str) -> float:
    """Fetch a best-effort market cap for ranking the cross-sectional universe."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            try:
                market_cap = fast_info.get("marketCap")
            except AttributeError:
                market_cap = getattr(fast_info, "marketCap", None)
            if market_cap:
                return float(market_cap)
        info = ticker.info
        market_cap = info.get("marketCap") if info else None
        if market_cap:
            return float(market_cap)
    except Exception:
        pass
    return 0.0


def cached_market_cap(
    ticker_symbol: str,
    *,
    cache_payload: dict,
    refresh_cache: bool = False,
) -> float:
    """Return a cached market cap when available, otherwise fetch and store it."""
    market_cap_entries = cache_payload.setdefault("market_caps", {})
    if not refresh_cache:
        cached = market_cap_entries.get(ticker_symbol)
        if isinstance(cached, dict) and "market_cap" in cached:
            return float(cached["market_cap"])

    market_cap = get_market_cap(ticker_symbol)
    market_cap_entries[ticker_symbol] = {
        "market_cap": float(market_cap),
        "cached_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    return float(market_cap)


def rank_large_cap_candidates(
    top_n: int,
    *,
    cache_payload: dict,
    refresh_cache: bool = False,
) -> list[str]:
    """Rank the large-cap universe using cached-or-live market caps."""
    market_caps = {
        ticker: cached_market_cap(
            ticker,
            cache_payload=cache_payload,
            refresh_cache=refresh_cache,
        )
        for ticker in LARGE_CAP_CANDIDATE_TICKERS
    }
    return rank_tickers_by_market_cap(market_caps, top_n)


def load_estimate_cache(cache_path: str) -> dict:
    """Load cached H-estimate summaries from disk."""
    path = Path(cache_path)
    if not path.exists():
        return {"version": CACHE_VERSION, "entries": {}, "market_caps": {}}

    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"version": CACHE_VERSION, "entries": {}, "market_caps": {}}

    if payload.get("version") != CACHE_VERSION or not isinstance(payload.get("entries"), dict):
        return {"version": CACHE_VERSION, "entries": {}, "market_caps": {}}
    if not isinstance(payload.get("market_caps"), dict):
        payload["market_caps"] = {}
    return payload


def save_estimate_cache(cache_path: str, cache_payload: dict) -> None:
    """Persist cached H-estimate summaries to disk."""
    path = Path(cache_path)
    path.write_text(json.dumps(cache_payload, indent=2, sort_keys=True))


def series_to_cache_payload(series: pd.Series) -> str:
    """Serialize a series for JSON cache storage."""
    return series.to_json(date_format="iso", date_unit="ns")


def series_from_cache_payload(payload: str) -> pd.Series:
    """Deserialize a cached series payload."""
    return pd.read_json(StringIO(payload), typ="series")


def dataframe_to_cache_payload(frame: pd.DataFrame) -> str:
    """Serialize a dataframe for JSON cache storage."""
    return frame.to_json(orient="split", date_format="iso", date_unit="ns")


def dataframe_from_cache_payload(payload: str) -> pd.DataFrame:
    """Deserialize a cached dataframe payload."""
    return pd.read_json(StringIO(payload), orient="split")


def cache_entry_from_report(report: dict) -> dict:
    """Extract the reusable summary and full plotting payload from a report."""
    roughness = report["roughness"]
    entry = {
        "ticker": report["ticker"],
        "interval": report["interval"],
        "period": report["period"],
        "rv_block_size": report["rv_block_size"],
        "hurst": roughness.hurst,
        "r_squared": roughness.r_squared,
        "latest_annualized_volatility": float(report["realized_variance_blocks"]["annualized_volatility"].iloc[-1]),
        "cached_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }

    full_report_keys = {
        "market",
        "close",
        "intraday_mode",
        "rv_block_label",
        "annualization",
        "local_volatility",
        "log_realized_variance",
        "atm_term_structure",
        "simulation_time",
        "simulation_seed",
        "rough_vol_path",
        "brownian_vol_path",
        "clipped_hurst",
    }
    roughness_keys = {"intercept", "lags", "structure_function", "fitted_structure_function"}
    if not full_report_keys.issubset(report.keys()) or not roughness_keys.issubset(vars(roughness).keys()):
        return entry

    entry["full_report"] = {
            "market": {
                "spot": float(report["market"].spot),
                "rate": float(report["market"].rate),
                "div_yield": float(report["market"].div_yield),
            },
            "close": series_to_cache_payload(report["close"]),
            "intraday_mode": bool(report["intraday_mode"]),
            "rv_block_label": report["rv_block_label"],
            "annualization": float(report["annualization"]),
            "realized_variance_blocks": dataframe_to_cache_payload(report["realized_variance_blocks"]),
            "local_volatility": series_to_cache_payload(report["local_volatility"]),
            "log_realized_variance": series_to_cache_payload(report["log_realized_variance"]),
            "roughness": {
                "hurst": float(roughness.hurst),
                "intercept": float(roughness.intercept),
                "r_squared": float(roughness.r_squared),
                "lags": roughness.lags.tolist(),
                "structure_function": roughness.structure_function.tolist(),
                "fitted_structure_function": roughness.fitted_structure_function.tolist(),
            },
            "atm_term_structure": dataframe_to_cache_payload(report["atm_term_structure"]),
            "simulation_time": report["simulation_time"].tolist(),
            "simulation_seed": int(report["simulation_seed"]),
            "rough_vol_path": report["rough_vol_path"].tolist(),
            "brownian_vol_path": report["brownian_vol_path"].tolist(),
            "clipped_hurst": float(report["clipped_hurst"]),
    }
    return entry


def histogram_report_from_cache_entry(entry: dict) -> dict:
    """Convert a cached scalar summary back into the minimal report needed for a histogram."""
    return {
        "ticker": entry["ticker"],
        "roughness": RoughnessEstimate(
            hurst=float(entry["hurst"]),
            intercept=0.0,
            r_squared=float(entry["r_squared"]),
            lags=np.array([], dtype=float),
            structure_function=np.array([], dtype=float),
            fitted_structure_function=np.array([], dtype=float),
        ),
        "from_cache": True,
    }


def full_report_from_cache_entry(entry: dict) -> dict:
    """Convert a cached full-report payload back into the report structure used by plots."""
    full_report = entry.get("full_report")
    if not isinstance(full_report, dict):
        raise KeyError("cached entry does not contain a full empirical report")

    roughness_payload = full_report["roughness"]
    return {
        "ticker": entry["ticker"],
        "market": MarketData(
            spot=float(full_report["market"]["spot"]),
            rate=float(full_report["market"]["rate"]),
            div_yield=float(full_report["market"]["div_yield"]),
        ),
        "close": series_from_cache_payload(full_report["close"]),
        "interval": entry["interval"],
        "intraday_mode": bool(full_report["intraday_mode"]),
        "period": entry["period"],
        "rv_block_size": int(entry["rv_block_size"]),
        "rv_block_label": full_report["rv_block_label"],
        "annualization": float(full_report["annualization"]),
        "realized_variance_blocks": dataframe_from_cache_payload(full_report["realized_variance_blocks"]),
        "local_volatility": series_from_cache_payload(full_report["local_volatility"]),
        "log_realized_variance": series_from_cache_payload(full_report["log_realized_variance"]),
        "roughness": RoughnessEstimate(
            hurst=float(roughness_payload["hurst"]),
            intercept=float(roughness_payload["intercept"]),
            r_squared=float(roughness_payload["r_squared"]),
            lags=np.array(roughness_payload["lags"], dtype=float),
            structure_function=np.array(roughness_payload["structure_function"], dtype=float),
            fitted_structure_function=np.array(roughness_payload["fitted_structure_function"], dtype=float),
        ),
        "atm_term_structure": dataframe_from_cache_payload(full_report["atm_term_structure"]),
        "simulation_time": np.array(full_report["simulation_time"], dtype=float),
        "simulation_seed": int(full_report["simulation_seed"]),
        "rough_vol_path": np.array(full_report["rough_vol_path"], dtype=float),
        "brownian_vol_path": np.array(full_report["brownian_vol_path"], dtype=float),
        "clipped_hurst": float(full_report["clipped_hurst"]),
        "from_cache": True,
    }


def load_or_build_empirical_roughness_report(
    ticker_symbol: str,
    *,
    interval: str,
    period: str | None,
    rv_block_size: int,
    cache_entries: dict[str, dict] | None = None,
    refresh_cache: bool = False,
) -> tuple[dict, bool]:
    """Return a full report from cache when available, otherwise rebuild it."""
    resolved_period = period or default_period_for_interval(interval)
    key = cache_key(
        ticker_symbol,
        interval=interval,
        period=resolved_period,
        rv_block_size=rv_block_size,
    )
    if not refresh_cache and cache_entries is not None and key in cache_entries:
        try:
            return full_report_from_cache_entry(cache_entries[key]), False
        except (KeyError, TypeError, ValueError, OSError):
            pass

    report = build_empirical_roughness_report(
        ticker_symbol,
        interval=interval,
        period=period,
        rv_block_size=rv_block_size,
    )
    if cache_entries is not None:
        cache_entries[key] = cache_entry_from_report(report)
    return report, True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Empirically estimate rough volatility from yfinance history and option data.",
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        default=DEFAULT_TICKERS,
        help="Ticker symbols to analyze. Defaults to SPY.",
    )
    parser.add_argument(
        "--interval",
        default=DEFAULT_PRICE_HISTORY_INTERVAL,
        help="Historical price interval passed to yfinance, e.g. 1m, 5m, 30m, 60m, 1d.",
    )
    parser.add_argument(
        "--period",
        default=None,
        help="Optional yfinance lookback period override. By default it is chosen from the interval.",
    )
    parser.add_argument(
        "--rv-block-size",
        type=int,
        default=DEFAULT_RV_BLOCK_SIZE,
        help="Non-overlapping realized-variance block size in observations.",
    )
    parser.add_argument(
        "--window",
        type=int,
        dest="rv_block_size_alias",
        default=None,
        help="Deprecated alias for --rv-block-size.",
    )
    parser.add_argument(
        "--hurst-hist-top-n",
        type=int,
        default=0,
        help="If > 0, rank a large-cap universe by market cap and plot a histogram of H estimates for the top N names.",
    )
    parser.add_argument(
        "--cache-path",
        default=DEFAULT_CACHE_PATH,
        help="Path to the JSON cache used to reuse H estimates across runs.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached H estimates and recompute them from fresh data.",
    )
    return parser.parse_args(argv)


def get_atm_iv_term_structure(
    ticker_symbol: str,
    *,
    spot: float,
    rate: float,
    div_yield: float,
    min_expiry_days: int = 14,
    max_expiry_days: int = 365,
    max_relative_moneyness: float = 0.05,
) -> pd.DataFrame:
    """Collect one near-ATM implied vol quote per listed expiry."""
    ticker = yf.Ticker(ticker_symbol)
    try:
        expiries = ticker.options
    except Exception:
        return pd.DataFrame()
    today = date.today()
    rows: list[dict[str, float | str | bool]] = []

    for expiry_str in expiries:
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        days = (expiry_date - today).days
        if not (min_expiry_days <= days <= max_expiry_days):
            continue

        maturity_years = days / 365.25

        try:
            chain = ticker.option_chain(expiry_str)
        except Exception:
            continue

        for side_df, is_call in ((chain.calls, True), (chain.puts, False)):
            if side_df is None or side_df.empty:
                continue

            candidates = side_df[side_df["lastPrice"] > 0.0].copy()
            if candidates.empty:
                continue

            candidates["relative_moneyness"] = np.abs(candidates["strike"] / spot - 1.0)
            candidates = candidates[candidates["relative_moneyness"] <= max_relative_moneyness]
            if candidates.empty:
                continue

            atm_row = candidates.sort_values("relative_moneyness", ascending=True).iloc[0]
            try:
                iv = implied_vol(
                    price=float(atm_row["lastPrice"]),
                    spot=spot,
                    strike=float(atm_row["strike"]),
                    maturity=maturity_years,
                    rate=rate,
                    div=div_yield,
                    is_call=is_call,
                )
            except ValueError:
                continue

            rows.append(
                {
                    "expiry_str": expiry_str,
                    "maturity_years": maturity_years,
                    "days_to_expiry": days,
                    "strike": float(atm_row["strike"]),
                    "relative_moneyness": float(atm_row["relative_moneyness"]),
                    "is_call": is_call,
                    "market_price": float(atm_row["lastPrice"]),
                    "implied_vol": iv,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    best_per_expiry = (
        df.sort_values(["expiry_str", "relative_moneyness"])
        .groupby("expiry_str", as_index=False)
        .first()
    )
    return best_per_expiry.sort_values("maturity_years").reset_index(drop=True)


def build_empirical_roughness_report(
    ticker_symbol: str,
    *,
    interval: str = DEFAULT_PRICE_HISTORY_INTERVAL,
    period: str | None = None,
    rv_block_size: int = DEFAULT_RV_BLOCK_SIZE,
) -> dict:
    """Fetch data, estimate roughness, and simulate a matched rough-vol path."""
    market = get_market_data(ticker_symbol)
    if market is None:
        raise RuntimeError(f"Could not fetch market data for {ticker_symbol}")

    intraday_mode = is_intraday_interval(interval)
    price_history_period = period or default_period_for_interval(interval)
    annualization = annualization_for_interval(interval)
    history = get_price_history(
        ticker_symbol,
        period=price_history_period,
        interval=interval,
    )
    if history.empty or "Close" not in history.columns:
        raise RuntimeError(f"Could not fetch closing-price history for {ticker_symbol}")

    close = history["Close"].dropna()
    rv_blocks = realized_variance_blocks(
        close,
        block_size=rv_block_size,
        annualization=annualization,
        session_aware=intraday_mode,
        deseasonalize_intraday=intraday_mode,
    )
    if rv_blocks.empty:
        raise RuntimeError(
            "realized-variance blocks are empty; try a smaller --rv-block-size or a longer --period"
        )

    local_vol = local_volatility_proxy(
        close,
        window=LOCAL_VOL_WINDOW_RETURNS,
        annualization=annualization,
        session_aware=intraday_mode,
        deseasonalize_intraday=intraday_mode,
    )

    log_realized_variance = np.log(rv_blocks["raw_realized_variance"])
    max_lag = min(MAX_LAG, max(4, len(log_realized_variance) // 4))
    if max_lag <= 1:
        raise RuntimeError("not enough volatility observations to estimate roughness")

    estimate = estimate_hurst_exponent(log_realized_variance, min_lag=1, max_lag=max_lag)
    clipped_hurst = float(np.clip(estimate.hurst, 0.03, 0.49))
    simulation_seed = stable_seed_from_ticker(ticker_symbol)

    atm_term_structure = get_atm_iv_term_structure(
        ticker_symbol,
        spot=market.spot,
        rate=market.rate,
        div_yield=market.div_yield,
    )

    t_sim, rough_vol, brownian_vol = simulate_lognormal_vol_paths(
        hurst=clipped_hurst,
        n_steps=SIMULATION_STEPS,
        horizon=SIMULATION_HORIZON,
        initial_vol=float(rv_blocks["annualized_volatility"].iloc[-1]),
        vol_of_vol=SIMULATION_VOL_OF_VOL,
        seed=simulation_seed,
    )

    return {
        "ticker": ticker_symbol,
        "market": market,
        "close": close,
        "interval": interval,
        "intraday_mode": intraday_mode,
        "period": price_history_period,
        "rv_block_size": rv_block_size,
        "rv_block_label": block_label(interval, rv_block_size),
        "annualization": annualization,
        "realized_variance_blocks": rv_blocks,
        "local_volatility": local_vol,
        "log_realized_variance": log_realized_variance,
        "roughness": estimate,
        "atm_term_structure": atm_term_structure,
        "simulation_time": t_sim,
        "simulation_seed": simulation_seed,
        "rough_vol_path": rough_vol,
        "brownian_vol_path": brownian_vol,
        "clipped_hurst": clipped_hurst,
    }


def plot_realized_vol_reports(reports: list[dict], output_path: str) -> None:
    """Create a realized-volatility overview + zoom figure across tickers."""
    n_rows = max(1, len(reports))
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4.4 * n_rows), squeeze=False)

    for row_axes, report in zip(axes, reports):
        ticker_symbol = report["ticker"]
        rv_blocks = report["realized_variance_blocks"]
        local_vol = report["local_volatility"]
        full_ax, zoom_ax = row_axes

        plot_series_with_session_gaps(
            full_ax,
            rv_blocks["annualized_volatility"],
            color="midnightblue",
            linewidth=0.85,
            gap=2.0,
        )
        full_ax.set_title(
            f"{ticker_symbol}\n"
            + (
                f"Full sample: {report['rv_block_label']} using de-seasonalized {report['interval']} returns"
                if report["intraday_mode"]
                else f"Full sample: {report['rv_block_label']} using {report['interval']} returns"
            )
        )
        full_ax.set_xlabel("Trading sessions (compressed)")
        full_ax.set_ylabel("Volatility")
        full_ax.grid(alpha=0.25)

        zoom_series = recent_intraday_zoom_series(
            local_vol if not local_vol.empty else rv_blocks["annualized_volatility"],
            interval=report["interval"],
            zoom_hours=REALIZED_VOL_ZOOM_HOURS,
        )
        plot_series_with_session_gaps(
            zoom_ax,
            zoom_series,
            color="crimson",
            linewidth=0.3,
            gap=2.0,
        )
        zoom_ax.set_title(
            f"{ticker_symbol}\n"
            + (
                f"Recent zoom: high-frequency local vol over last {REALIZED_VOL_ZOOM_HOURS}h"
                if report["intraday_mode"]
                else "Recent zoom: local volatility proxy"
            )
        )
        zoom_ax.set_xlabel("Latest session (compressed)" if report["intraday_mode"] else "Recent observations")
        zoom_ax.set_ylabel("Volatility")
        zoom_ax.grid(alpha=0.25)

    fig.suptitle(
        "Empirical volatility: block-based overview and high-frequency zoom",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=500)
    plt.close(fig)


def plot_roughness_regression_reports(reports: list[dict], output_path: str) -> None:
    """Create a multi-panel roughness-regression figure across tickers."""
    fig, axes = make_panel_figure(len(reports))
    for ax, report in zip(axes, reports):
        ticker_symbol = report["ticker"]
        roughness = report["roughness"]
        ax.loglog(
            roughness.lags,
            roughness.structure_function,
            "o",
            color="firebrick",
            label="Empirical structure function",
        )
        ax.loglog(
            roughness.lags,
            roughness.fitted_structure_function,
            "-",
            color="black",
            linewidth=1.4,
            label=f"Fit slope = H = {roughness.hurst:.2f}",
        )
        ax.set_title(f"{ticker_symbol}\nH={roughness.hurst:.2f}, R^2={roughness.r_squared:.2f}")
        ax.set_xlabel("Lag (RV blocks)")
        ax.set_ylabel(r"$E[|\log RV_{t+\Delta} - \log RV_t|]$")
        ax.legend()
        ax.grid(alpha=0.25)

    hide_unused_axes(axes, len(reports))
    fig.suptitle("Roughness regression on log realized variance", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_atm_term_structure_reports(reports: list[dict], output_path: str) -> None:
    """Create a multi-panel ATM implied-vol term-structure figure across tickers."""
    fig, axes = make_panel_figure(len(reports))
    for ax, report in zip(axes, reports):
        ticker_symbol = report["ticker"]
        atm_term_structure = report["atm_term_structure"]
        if atm_term_structure.empty:
            ax.text(0.5, 0.5, "No ATM option quotes available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        ax.plot(
            atm_term_structure["days_to_expiry"],
            atm_term_structure["implied_vol"],
            "o-",
            color="darkgreen",
            linewidth=1.5,
        )
        ax.set_title(f"{ticker_symbol}")
        ax.set_xlabel("Days to expiry")
        ax.set_ylabel("Implied volatility")
        ax.grid(alpha=0.25)

    hide_unused_axes(axes, len(reports))
    fig.suptitle("ATM implied-vol term structures", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_simulation_reports(reports: list[dict], output_path: str) -> None:
    """Create a multi-panel rough-vs-Brownian simulation figure across tickers."""
    fig, axes = make_panel_figure(len(reports))
    for ax, report in zip(axes, reports):
        ticker_symbol = report["ticker"]
        ax.plot(
            report["simulation_time"],
            report["rough_vol_path"],
            color="crimson",
            linewidth=1.5,
            label=f"Rough simulation (H={report['clipped_hurst']:.2f})",
        )
        ax.plot(
            report["simulation_time"],
            report["brownian_vol_path"],
            color="steelblue",
            linewidth=1.5,
            label="Brownian benchmark (H=0.50)",
        )
        ax.set_title(f"{ticker_symbol}")
        ax.set_xlabel("Years")
        ax.set_ylabel("Volatility")
        ax.legend()
        ax.grid(alpha=0.25)

    hide_unused_axes(axes, len(reports))
    fig.suptitle("Matched rough-vs-Brownian volatility simulations", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_hurst_histogram_reports(
    *,
    top_n: int,
    interval: str,
    period: str | None,
    rv_block_size: int,
    cached_reports: dict[str, dict] | None = None,
    cache_entries: dict[str, dict] | None = None,
    cache_payload: dict | None = None,
    refresh_cache: bool = False,
) -> tuple[list[dict], list[str]]:
    """Estimate H across a live large-cap cross-section ranked by market cap."""
    if top_n <= 0:
        return [], []

    if cache_payload is not None:
        ranked_tickers = rank_large_cap_candidates(
            top_n,
            cache_payload=cache_payload,
            refresh_cache=refresh_cache,
        )
    else:
        market_caps = {
            ticker: get_market_cap(ticker)
            for ticker in LARGE_CAP_CANDIDATE_TICKERS
        }
        ranked_tickers = rank_tickers_by_market_cap(market_caps, top_n)

    reports: list[dict] = []
    failures: list[str] = []
    in_memory_reports = cached_reports or {}
    persisted_entries = cache_entries or {}

    for idx, ticker_symbol in enumerate(ranked_tickers, start=1):
        print(f"[Histogram] {idx}/{len(ranked_tickers)} {ticker_symbol}")
        if ticker_symbol in in_memory_reports:
            reports.append(in_memory_reports[ticker_symbol])
            continue

        key = cache_key(
            ticker_symbol,
            interval=interval,
            period=period or default_period_for_interval(interval),
            rv_block_size=rv_block_size,
        )
        if not refresh_cache and key in persisted_entries:
            entry = persisted_entries[key]
            try:
                print("  using cached full report")
                reports.append(full_report_from_cache_entry(entry))
                continue
            except (KeyError, TypeError, ValueError, OSError):
                print("  using cached H estimate")
                reports.append(histogram_report_from_cache_entry(entry))
                continue

        try:
            report = build_empirical_roughness_report(
                ticker_symbol,
                interval=interval,
                period=period,
                rv_block_size=rv_block_size,
            )
        except Exception:
            failures.append(ticker_symbol)
            continue

        reports.append(report)
        in_memory_reports[ticker_symbol] = report
        persisted_entries[key] = cache_entry_from_report(report)

    return reports, failures


def plot_hurst_histogram(reports: list[dict], output_path: str, *, top_n: int) -> None:
    """Plot the cross-sectional histogram of estimated Hurst exponents."""
    hursts = np.array([report["roughness"].hurst for report in reports], dtype=float)
    tickers = [report["ticker"] for report in reports]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = min(40, max(16, int(np.sqrt(len(hursts)) * 3)))
    ax.hist(hursts, bins=bins, color="slateblue", edgecolor="white", alpha=0.85)
    mean_h = float(np.mean(hursts))
    median_h = float(np.median(hursts))
    ax.axvline(mean_h, color="crimson", linestyle="--", linewidth=1.6, label=f"Mean H = {mean_h:.2f}")
    ax.axvline(median_h, color="black", linestyle="-.", linewidth=1.4, label=f"Median H = {median_h:.2f}")
    ax.set_title(f"Estimated Hurst exponents across top {top_n} stocks by market cap")
    ax.set_xlabel("Estimated H")
    ax.set_ylabel("Number of stocks")
    ax.legend()
    ax.grid(alpha=0.2)

    summary = (
        f"n={len(hursts)} successful estimates\n"
        f"{sum(hursts < 0.5)} below 0.50\n"
        f"min={hursts.min():.2f}, max={hursts.max():.2f}"
    )
    ax.text(
        0.98,
        0.98,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )

    lowest = np.argsort(hursts)[: min(5, len(hursts))]
    lowest_text = ", ".join(f"{tickers[i]} ({hursts[i]:.2f})" for i in lowest)
    fig.text(0.5, 0.01, f"Lowest H names: {lowest_text}", ha="center", fontsize=9)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    tickers = [ticker.upper() for ticker in args.tickers if ticker.strip()]
    cached_reports: dict[str, dict] = {}
    rv_block_size = args.rv_block_size_alias or args.rv_block_size
    cache_payload = load_estimate_cache(args.cache_path)
    cache_entries = cache_payload["entries"]
    cache_dirty = False

    for ticker_symbol in tickers:
        print("=" * 70)
        print(f"Empirical roughness demo for {ticker_symbol}")
        print("=" * 70)
        try:
            report, rebuilt = load_or_build_empirical_roughness_report(
                ticker_symbol,
                interval=args.interval,
                period=args.period,
                rv_block_size=rv_block_size,
                cache_entries=cache_entries,
                refresh_cache=args.refresh_cache,
            )
        except Exception as exc:
            print(f"Failed for {ticker_symbol}: {exc}")
            continue

        cached_reports[ticker_symbol] = report
        if rebuilt:
            cache_dirty = True
        else:
            print("Using cached full report.")
        estimate = report["roughness"]
        print(f"Spot: {report['market'].spot:.2f}")
        print(
            f"History sampling: interval={report['interval']}, "
            f"period={report['period']}, rv_block_size={report['rv_block_size']} obs"
        )
        print(
            f"Annualization factor: {report['annualization']:.1f} "
            + (
                "(overnight gaps excluded and intraday returns de-seasonalized)"
                if report["intraday_mode"]
                else "(non-overlapping realized-variance blocks)"
            )
        )
        print(
            "Latest annualized realized vol block: "
            f"{report['realized_variance_blocks']['annualized_volatility'].iloc[-1]:.3f}"
        )
        print(f"Estimated Hurst exponent: {estimate.hurst:.3f}")
        print(f"Regression R^2: {estimate.r_squared:.3f}")
        print(f"Simulation seed: {report['simulation_seed']}")
        if estimate.hurst < 0.5:
            print("Interpretation: H < 0.5, so the empirical volatility proxy is rough.")
        else:
            print("Interpretation: H is not below 0.5 in this sample.")
        if report["atm_term_structure"].empty:
            print("ATM term structure: unavailable from the current option chain snapshot.")
        else:
            print(
                "ATM term structure quotes: "
                f"{len(report['atm_term_structure'])} expiries "
                f"(front ATM IV={report['atm_term_structure']['implied_vol'].iloc[0]:.3f})"
            )
        print()

    if not cached_reports:
        print("No per-ticker reports were generated because all requested ticker runs failed.")

    if args.hurst_hist_top_n > 0:
        print("=" * 70)
        print(f"Cross-sectional Hurst summary for top {args.hurst_hist_top_n} stocks")
        print("=" * 70)
        cache_size_before = len(cache_entries)
        hist_reports, failures = build_hurst_histogram_reports(
            top_n=args.hurst_hist_top_n,
            interval=args.interval,
            period=args.period,
            rv_block_size=rv_block_size,
            cached_reports=cached_reports,
            cache_entries=cache_entries,
            cache_payload=cache_payload,
            refresh_cache=args.refresh_cache,
        )
        if hist_reports:
            print(f"Histogram sample: {len(hist_reports)} successful estimates" + (f", {len(failures)} failures" if failures else ""))
            cache_dirty = cache_dirty or (len(cache_entries) > cache_size_before)
        else:
            print("No cross-sectional H estimates succeeded.")

    if cache_dirty:
        save_estimate_cache(args.cache_path, cache_payload)
        print(f"Updated cache: {args.cache_path}")


if __name__ == "__main__":
    main()
