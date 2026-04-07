"""Calibration workflow for live option-surface experiments.

This module owns the calibration pipeline and returns structured, readable data
that the one-plot scripts in this folder can render in different ways.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from roughvol.types import MarketData

TICKERS = ["SPY", "AAPL"]

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
    return parser.parse_args(argv)


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
) -> dict[str, TickerCalibrationReport]:
    """Fetch market data, filter the surface, and calibrate all requested tickers."""
    tickers = [ticker.upper() for ticker in (tickers or TICKERS)]
    print("=" * 60)
    print("  Rough Vol Calibration Demo")
    print(f"  Tickers: {tickers}")
    print("=" * 60)

    all_results: dict[str, TickerCalibrationReport] = {}

    for ticker_symbol in tickers:
        print(f"\n{'─' * 50}")
        print(f"  {ticker_symbol}")
        print(f"{'─' * 50}")

        market_data = get_market_data(ticker_symbol)
        if market_data is None:
            print(f"  Skipping {ticker_symbol}: no market data")
            continue
        print(
            f"  Spot={market_data.spot:.2f}  "
            f"Rate={market_data.rate:.2%}  "
            f"Div={market_data.div_yield:.2%}",
        )

        surface_df = get_option_surface(ticker_symbol, market_data)
        if surface_df.empty:
            print(f"  Skipping {ticker_symbol}: no options data")
            continue
        print(f"  Fetched {len(surface_df)} options across surface")

        calib_df = filter_options_for_calibration(surface_df, market_data.spot)
        if len(calib_df) < 3:
            print(
                f"  Skipping {ticker_symbol}: too few liquid options "
                f"for calibration ({len(calib_df)})",
            )
            continue
        print(
            f"  Using {len(calib_df)} options for calibration  "
            f"(expiries: {calib_df['expiry_str'].nunique()})",
        )

        all_results[ticker_symbol] = calibrate_ticker(
            ticker_symbol,
            calib_df,
            surface_df,
            market_data,
        )

    print_calibration_summary(all_results)
    return all_results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    collect_calibration_results(args.tickers)


if __name__ == "__main__":
    main()
