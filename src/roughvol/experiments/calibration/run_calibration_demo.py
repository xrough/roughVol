"""Calibration workflow for live option-surface experiments.

This module owns the calibration pipeline and returns structured, readable data
that the one-plot scripts in this folder can render in different ways.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yfinance  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "yfinance is required to run this demo.\n"
        "Install it with:  pip install yfinance"
    ) from exc

from roughvol.analytics.black_scholes_formula import implied_vol
from roughvol.calibration.calibration import (
    CalibResult,
    make_gbm_calibrator,
    make_heston_calibrator,
    make_rough_bergomi_calibrator,
)
from roughvol.data.yfinance_loader import get_market_data, get_option_surface
from roughvol.engines.mc import MonteCarloEngine
from roughvol.instruments.vanilla import VanillaOption
from roughvol.models.GBM_model import GBM_Model
from roughvol.models.heston_model import HestonModel
from roughvol.models.rough_bergomi_model import RoughBergomiModel
from roughvol.experiments._paths import output_path
from roughvol.types import MarketData

TICKERS = ["SPY", "AAPL"]
CACHE_VERSION = 1
DEFAULT_CACHE_PATH = output_path("calibration", "calibration_cache.json")

CALIB_ENGINE_GBM = {
    "n_paths": 2_000,
    "n_steps": 20,
    "seed": 42,
    "antithetic": True,
}

CALIB_ENGINE_HESTON = {
    "n_paths": 2_000,
    "n_steps": 20,
    "seed": 42,
    "antithetic": True,
}

CALIB_ENGINE_RB = {
    "n_paths": 5_000,
    "n_steps": 52,
    "seed": 42,
    "antithetic": True,
}

VIZ_ENGINE = {
    "n_paths": 300,
    "n_steps": 16,
    "seed": 99,
    "antithetic": True,
}

MODEL_COLOURS = {
    "GBM": "steelblue",
    "Heston": "darkorange",
    "RoughBergomi": "crimson",
}

MODEL_LINESTYLES = {
    "GBM": "--",
    "Heston": ":",
    "RoughBergomi": "-.",
}

MODEL_LABELS = {
    "GBM": "GBM",
    "Heston": "Heston",
    "RoughBergomi": "Rough Bergomi",
}


@dataclass(frozen=True)
class TickerCalibrationReport:
    """Structured result for one ticker calibration run."""

    ticker: str
    market_data: MarketData
    surface_df: pd.DataFrame
    calib_df: pd.DataFrame
    results: dict[str, CalibResult | None]
    iv_rmse: dict[str, float]
    error: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch live option data and calibrate GBM, Heston, and Rough Bergomi.",
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        default=TICKERS,
        help="Ticker symbols to calibrate. Defaults to SPY AAPL.",
    )
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args(argv)


def load_calibration_cache(cache_path: str) -> dict:
    """Load cached calibration reports from disk."""
    path = Path(cache_path)
    if not path.exists():
        return {"version": CACHE_VERSION, "entries": {}}

    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"version": CACHE_VERSION, "entries": {}}

    if payload.get("version") != CACHE_VERSION or not isinstance(payload.get("entries"), dict):
        return {"version": CACHE_VERSION, "entries": {}}
    return payload


def save_calibration_cache(cache_path: str, cache_payload: dict) -> None:
    """Persist cached calibration reports to disk."""
    Path(cache_path).write_text(json.dumps(cache_payload, indent=2, sort_keys=True))


def calibration_cache_key(ticker_symbol: str) -> str:
    """Build a deterministic calibration cache key."""
    return ticker_symbol.upper()


def dataframe_to_cache_payload(frame: pd.DataFrame) -> str:
    """Serialize a dataframe for JSON cache storage."""
    return frame.to_json(orient="split", date_format="iso", date_unit="ns")


def dataframe_from_cache_payload(payload: str) -> pd.DataFrame:
    """Deserialize a dataframe from the JSON cache."""
    return pd.read_json(StringIO(payload), orient="split")


def calib_result_to_payload(result: CalibResult | None) -> dict | None:
    """Serialize a calibration result."""
    if result is None:
        return None
    return {
        "model_name": result.model_name,
        "params": {name: float(value) for name, value in result.params.items()},
        "mse": float(result.mse),
        "per_option_ivols": [float(value) for value in result.per_option_ivols],
        "elapsed_s": float(result.elapsed_s),
    }


def calib_result_from_payload(payload: dict | None) -> CalibResult | None:
    """Deserialize a calibration result."""
    if payload is None:
        return None
    return CalibResult(
        model_name=payload["model_name"],
        params={name: float(value) for name, value in payload["params"].items()},
        mse=float(payload["mse"]),
        per_option_ivols=[float(value) for value in payload.get("per_option_ivols", [])],
        elapsed_s=float(payload.get("elapsed_s", 0.0)),
    )


def cache_entry_from_report(report: TickerCalibrationReport) -> dict:
    """Serialize a full ticker calibration report for cache reuse."""
    return {
        "ticker": report.ticker,
        "market_data": {
            "spot": float(report.market_data.spot),
            "rate": float(report.market_data.rate),
            "div_yield": float(report.market_data.div_yield),
        },
        "surface_df": dataframe_to_cache_payload(report.surface_df),
        "calib_df": dataframe_to_cache_payload(report.calib_df),
        "results": {
            model_name: calib_result_to_payload(result)
            for model_name, result in report.results.items()
        },
        "iv_rmse": {
            model_name: float(value)
            for model_name, value in report.iv_rmse.items()
        },
        "error": report.error,
        "cached_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def report_from_cache_entry(entry: dict) -> TickerCalibrationReport:
    """Restore a ticker calibration report from cache."""
    return TickerCalibrationReport(
        ticker=entry["ticker"],
        market_data=MarketData(
            spot=float(entry["market_data"]["spot"]),
            rate=float(entry["market_data"]["rate"]),
            div_yield=float(entry["market_data"]["div_yield"]),
        ),
        surface_df=dataframe_from_cache_payload(entry["surface_df"]),
        calib_df=dataframe_from_cache_payload(entry["calib_df"]),
        results={
            model_name: calib_result_from_payload(payload)
            for model_name, payload in entry["results"].items()
        },
        iv_rmse={model_name: float(value) for model_name, value in entry["iv_rmse"].items()},
        error=entry.get("error"),
    )


def load_or_collect_ticker_report(
    ticker_symbol: str,
    *,
    cache_entries: dict[str, dict],
    refresh_cache: bool = False,
) -> tuple[TickerCalibrationReport, bool]:
    """Return a cached calibration report when available, otherwise build one."""
    key = calibration_cache_key(ticker_symbol)
    if not refresh_cache and key in cache_entries:
        try:
            return report_from_cache_entry(cache_entries[key]), False
        except (KeyError, TypeError, ValueError, OSError):
            pass

    market_data = get_market_data(ticker_symbol)
    if market_data is None:
        raise RuntimeError(f"Could not fetch market data for {ticker_symbol}")

    surface_df = get_option_surface(ticker_symbol, market_data)
    if surface_df.empty:
        raise RuntimeError(f"Could not fetch option surface for {ticker_symbol}")

    calib_df = filter_options_for_calibration(surface_df, market_data.spot)
    if len(calib_df) < 3:
        raise RuntimeError(f"Too few liquid options for calibration ({len(calib_df)})")

    report = calibrate_ticker(ticker_symbol, calib_df, surface_df, market_data)
    cache_entries[key] = cache_entry_from_report(report)
    return report, True


def filter_options_for_calibration(
    surface_df: pd.DataFrame,
    spot: float,
    min_days: int = 30,
    max_days: int = 90,
    moneyness: float = 0.20,
) -> pd.DataFrame:
    """Keep the liquid 1-3 month, near-the-money slice used for calibration."""
    lo = min_days / 365.25
    hi = max_days / 365.25
    mask = (
        (surface_df["maturity_years"] >= lo)
        & (surface_df["maturity_years"] <= hi)
        & (surface_df["strike"] >= spot * (1.0 - moneyness))
        & (surface_df["strike"] <= spot * (1.0 + moneyness))
    )
    filtered = surface_df[mask].copy()

    if len(filtered) < 5:
        relaxed_moneyness = 0.30
        mask = (
            (surface_df["maturity_years"] >= lo)
            & (surface_df["maturity_years"] <= hi)
            & (surface_df["strike"] >= spot * (1.0 - relaxed_moneyness))
            & (surface_df["strike"] <= spot * (1.0 + relaxed_moneyness))
        )
        filtered = surface_df[mask].copy()

    return filtered.reset_index(drop=True)


def build_model_from_params(model_name: str, params: dict[str, float]) -> object:
    """Rebuild a model instance from calibrated parameters."""
    if model_name == "GBM":
        return GBM_Model(sigma=params["sigma"])
    if model_name == "Heston":
        return HestonModel(
            kappa=params["kappa"],
            theta=params["theta"],
            xi=params["xi"],
            rho=params["rho"],
            v0=params["v0"],
        )
    if model_name == "RoughBergomi":
        return RoughBergomiModel(
            hurst=params["hurst"],
            eta=params["eta"],
            rho=params["rho"],
            xi0=params["xi0"],
            scheme="blp-hybrid",
        )
    raise ValueError(f"Unknown model: {model_name}")


def compute_model_iv_smile(
    model_name: str,
    params: dict[str, float],
    market_data: MarketData,
    maturity: float,
    strikes: list[float],
    engine_kwargs: dict[str, Any],
) -> list[float | None]:
    """Reprice a strip of options and invert them to implied vols."""
    model = build_model_from_params(model_name, params)
    engine = MonteCarloEngine(**engine_kwargs)
    ivs: list[float | None] = []

    for strike in strikes:
        instrument = VanillaOption(strike=strike, maturity=maturity, is_call=True)
        try:
            price_result = engine.price(model=model, instrument=instrument, market=market_data)
            iv = implied_vol(
                price=price_result.price,
                spot=market_data.spot,
                strike=strike,
                maturity=maturity,
                rate=market_data.rate,
                div=market_data.div_yield,
                is_call=True,
            )
        except (ValueError, Exception):
            iv = None
        ivs.append(iv)

    return ivs


def compute_iv_rmse(
    model_name: str,
    params: dict[str, float],
    calib_df: pd.DataFrame,
    market_data: MarketData,
    engine_kwargs: dict[str, Any],
) -> float:
    """Compute IV RMSE between a calibrated model and the filtered market surface."""
    model = build_model_from_params(model_name, params)
    engine = MonteCarloEngine(**engine_kwargs)
    errors: list[float] = []

    for _, row in calib_df.iterrows():
        instrument = VanillaOption(
            strike=float(row["strike"]),
            maturity=float(row["maturity_years"]),
            is_call=bool(row["is_call"]),
        )
        try:
            price_result = engine.price(model=model, instrument=instrument, market=market_data)
            iv_model = implied_vol(
                price=price_result.price,
                spot=market_data.spot,
                strike=float(row["strike"]),
                maturity=float(row["maturity_years"]),
                rate=market_data.rate,
                div=market_data.div_yield,
                is_call=bool(row["is_call"]),
            )
        except (ValueError, Exception):
            continue
        errors.append(iv_model - float(row["implied_vol"]))

    if not errors:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(np.asarray(errors, dtype=float)))))


def _stratified_sample(pool: pd.DataFrame, n: int) -> pd.DataFrame:
    pool = pool.sort_values(["maturity_years", "strike"]).reset_index(drop=True)
    step = max(1, len(pool) // n)
    return pool.iloc[::step].head(n).reset_index(drop=True)


def calibrate_ticker(
    ticker_symbol: str,
    calib_df: pd.DataFrame,
    surface_df: pd.DataFrame,
    market_data: MarketData,
) -> TickerCalibrationReport:
    """Calibrate the three models for one ticker and return a structured report."""
    atm_iv = float(calib_df["implied_vol"].median())
    base_cols = ["strike", "maturity_years", "is_call", "market_price"]

    opts_df_gbm = _stratified_sample(calib_df[base_cols], n=10)
    opts_df_heston = _stratified_sample(calib_df[base_cols], n=14)
    opts_df_rb = _stratified_sample(calib_df[base_cols], n=14)

    results: dict[str, CalibResult | None] = {}
    iv_rmse: dict[str, float] = {}

    calibrators = [
        ("GBM", make_gbm_calibrator(x0_sigma=atm_iv, engine_kwargs=CALIB_ENGINE_GBM), opts_df_gbm),
        ("Heston", make_heston_calibrator(x0_sigma=atm_iv, engine_kwargs=CALIB_ENGINE_HESTON), opts_df_heston),
        (
            "RoughBergomi",
            make_rough_bergomi_calibrator(
                x0_sigma=atm_iv,
                engine_kwargs=CALIB_ENGINE_RB,
                scheme="blp-hybrid",
            ),
            opts_df_rb,
        ),
    ]

    for model_name, calibrator, opts_df in calibrators:
        print(f"  [{ticker_symbol}] Calibrating {model_name} on {len(opts_df)} options...")
        try:
            calib_result = calibrator.calibrate(
                spot=market_data.spot,
                options_df=opts_df,
                rate=market_data.rate,
                div=market_data.div_yield,
            )
            results[model_name] = calib_result
            if calib_result.mse > 0.01:
                print(
                    f"  [WARN] {ticker_symbol} {model_name}: "
                    f"poor calibration (MSE={calib_result.mse:.3e})",
                )
            rmse = compute_iv_rmse(
                model_name=model_name,
                params=calib_result.params,
                calib_df=calib_df,
                market_data=market_data,
                engine_kwargs=VIZ_ENGINE,
            )
            iv_rmse[model_name] = rmse
            print(
                f"  [{ticker_symbol}] {model_name} done  "
                f"params={calib_result.params}  "
                f"IV-RMSE={rmse:.4f}",
            )
        except Exception as exc:
            print(f"  [ERROR] {ticker_symbol} {model_name} calibration failed: {exc}")
            results[model_name] = None
            iv_rmse[model_name] = float("nan")

    return TickerCalibrationReport(
        ticker=ticker_symbol,
        market_data=market_data,
        surface_df=surface_df,
        calib_df=calib_df,
        results=results,
        iv_rmse=iv_rmse,
        error=None,
    )


def print_calibration_summary(all_results: dict[str, TickerCalibrationReport]) -> None:
    """Print a readable summary table of per-model RMSEs."""
    if not all_results:
        print(
            "\nNo tickers calibrated successfully.\n"
            "Check network connection and that yfinance is installed.",
        )
        return

    print("\n" + "=" * 60)
    print("  Calibration Summary  (IV RMSE in vol units)")
    print("=" * 60)
    print(f"  {'Ticker':<8} {'GBM RMSE':>10} {'Heston RMSE':>12} {'rBergomi RMSE':>14}")
    print(f"  {'─' * 48}")
    for ticker_symbol, report in all_results.items():
        g = f"{report.iv_rmse.get('GBM', float('nan')):.4f}"
        h = f"{report.iv_rmse.get('Heston', float('nan')):.4f}"
        rb = f"{report.iv_rmse.get('RoughBergomi', float('nan')):.4f}"
        print(f"  {ticker_symbol:<8} {g:>10} {h:>12} {rb:>14}")


def collect_calibration_results(
    tickers: list[str] | None = None,
    *,
    cache_path: str = DEFAULT_CACHE_PATH,
    refresh_cache: bool = False,
) -> dict[str, TickerCalibrationReport]:
    """Fetch market data, filter the surface, and calibrate all requested tickers."""
    tickers = [ticker.upper() for ticker in (tickers or TICKERS)]
    cache_payload = load_calibration_cache(cache_path)
    cache_dirty = False
    print("=" * 60)
    print("  Rough Vol Calibration Demo")
    print(f"  Tickers: {tickers}")
    print("=" * 60)

    all_results: dict[str, TickerCalibrationReport] = {}

    for ticker_symbol in tickers:
        print(f"\n{'─' * 50}")
        print(f"  {ticker_symbol}")
        print(f"{'─' * 50}")
        try:
            report, rebuilt = load_or_collect_ticker_report(
                ticker_symbol,
                cache_entries=cache_payload["entries"],
                refresh_cache=refresh_cache,
            )
        except RuntimeError as exc:
            print(f"  Skipping {ticker_symbol}: {exc}")
            continue

        if rebuilt:
            print(
                f"  Spot={report.market_data.spot:.2f}  "
                f"Rate={report.market_data.rate:.2%}  "
                f"Div={report.market_data.div_yield:.2%}",
            )
            print(f"  Fetched {len(report.surface_df)} options across surface")
            print(
                f"  Using {len(report.calib_df)} options for calibration  "
                f"(expiries: {report.calib_df['expiry_str'].nunique()})",
            )
            cache_dirty = True
        else:
            print("  Using cached calibration report")

        all_results[ticker_symbol] = report

    print_calibration_summary(all_results)
    if cache_dirty:
        save_calibration_cache(cache_path, cache_payload)
        print(f"Updated cache: {cache_path}")
    return all_results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    collect_calibration_results(
        args.tickers,
        cache_path=args.cache_path,
        refresh_cache=args.refresh_cache,
    )


if __name__ == "__main__":
    main()
