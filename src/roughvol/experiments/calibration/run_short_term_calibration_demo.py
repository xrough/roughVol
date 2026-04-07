"""Short-term calibration workflow with daily snapshot caching.

This module owns the short-dated calibration track used by the calibration
panel/animation scripts. It is separate from the broader calibration demo.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, date, datetime
from hashlib import sha256
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
    make_rough_heston_calibrator,
)
from roughvol.data.yfinance_loader import get_market_data, get_option_surface
from roughvol.engines.mc import MonteCarloEngine
from roughvol.experiments._paths import output_dir, output_path
from roughvol.experiments.calibration.run_calibration_demo import (
    MODEL_COLOURS,
    MODEL_LABELS,
    MODEL_LINESTYLES,
    calib_result_from_payload,
    calib_result_to_payload,
    dataframe_from_cache_payload,
    dataframe_to_cache_payload,
)
from roughvol.instruments.vanilla import VanillaOption
from roughvol.models.GBM_model import GBM_Model
from roughvol.models.heston_model import HestonModel
from roughvol.models.rough_bergomi_model import RoughBergomiModel
from roughvol.models.rough_heston_model import RoughHestonModel
from roughvol.types import MarketData

WORKFLOW_NAME = "short_term_calibration_animation"
CACHE_VERSION = 1
DEFAULT_SHORT_TERM_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AMD",
    "NFLX",
]
SHORT_TERM_MIN_DAYS = 14
SHORT_TERM_MAX_DAYS = 45
SHORT_TERM_MONEYNESS = 0.20
SHORT_TERM_RELAXED_MONEYNESS = 0.30
TARGET_SMILE_DAYS = 30
RB_SCHEME = "blp-hybrid"
RH_SCHEME = "bayer-breneis"
RH_N_FACTORS = 8
MODEL_NAMES = ("GBM", "Heston", "RoughBergomi", "RoughHeston")
DEFAULT_MONEYNESS_GRID = [round(value, 4) for value in np.linspace(0.80, 1.20, 11)]
DEFAULT_CACHE_PATH = output_path("calibration", "short_term_calibration_cache.json")
DEFAULT_SNAPSHOT_DIR = str(output_dir("calibration") / "short_term_snapshots")

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
CALIB_ENGINE_RH = {
    "n_paths": 2_000,
    "n_steps": 24,
    "seed": 42,
    "antithetic": True,
}
VIZ_ENGINE = {
    "n_paths": 300,
    "n_steps": 20,
    "seed": 99,
    "antithetic": True,
}
CALIBRATION_SAMPLE_SIZES = {
    "GBM": 10,
    "Heston": 14,
    "RoughBergomi": 14,
    "RoughHeston": 12,
}


@dataclass(frozen=True)
class ShortTermTickerReport:
    """Plot-ready calibration result for one ticker/date."""

    ticker: str
    snapshot_date: str
    market_data: MarketData | None
    short_term_df: pd.DataFrame
    available_expiries: list[dict[str, float | str]]
    selected_expiry: str | None
    selected_maturity: float | None
    market_smile_df: pd.DataFrame
    results: dict[str, CalibResult | None]
    model_settings: dict[str, dict[str, Any]]
    model_smiles: dict[str, list[float | None]]
    iv_rmse: dict[str, float]
    status: str = "ok"
    error: str | None = None


@dataclass(frozen=True)
class ShortTermSnapshot:
    """Daily snapshot used by both the static panel and the animation."""

    snapshot_date: str
    created_at: str
    basket: list[str]
    reports: dict[str, ShortTermTickerReport]
    workflow_fingerprint: str


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "tickers",
        nargs="*",
        default=DEFAULT_SHORT_TERM_TICKERS,
        help="Ticker symbols for the short-term panel. Defaults to a fixed 9-name basket.",
    )
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="Snapshot date label in YYYY-MM-DD. Defaults to the current local date.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser(
        "Fetch short-dated options, calibrate four models, and store a daily snapshot.",
    ).parse_args(argv)


def normalize_tickers(tickers: list[str] | None) -> list[str]:
    raw = tickers or DEFAULT_SHORT_TERM_TICKERS
    deduped: list[str] = []
    seen: set[str] = set()
    for ticker in raw:
        symbol = ticker.upper().strip()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    return deduped or list(DEFAULT_SHORT_TERM_TICKERS)


def resolve_snapshot_date(snapshot_date: str | None) -> str:
    if snapshot_date is None:
        return date.today().isoformat()
    return date.fromisoformat(snapshot_date).isoformat()


def workflow_settings() -> dict[str, Any]:
    return {
        "models": list(MODEL_NAMES),
        "rough_bergomi_scheme": RB_SCHEME,
        "rough_heston_scheme": RH_SCHEME,
        "rough_heston_n_factors": RH_N_FACTORS,
        "calibration_engine_kwargs": {
            "GBM": CALIB_ENGINE_GBM,
            "Heston": CALIB_ENGINE_HESTON,
            "RoughBergomi": CALIB_ENGINE_RB,
            "RoughHeston": CALIB_ENGINE_RH,
        },
        "visualization_engine_kwargs": VIZ_ENGINE,
        "short_term_filter": {
            "min_days": SHORT_TERM_MIN_DAYS,
            "max_days": SHORT_TERM_MAX_DAYS,
            "moneyness": SHORT_TERM_MONEYNESS,
            "relaxed_moneyness": SHORT_TERM_RELAXED_MONEYNESS,
        },
        "displayed_maturity_rule": f"closest_to_{TARGET_SMILE_DAYS}D",
        "moneyness_grid": DEFAULT_MONEYNESS_GRID,
        "objective": "iv_mse",
        "sample_sizes": CALIBRATION_SAMPLE_SIZES,
    }


def workflow_fingerprint() -> str:
    payload = json.dumps(workflow_settings(), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def empty_latest_cache() -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "workflow_name": WORKFLOW_NAME,
        "workflow_fingerprint": workflow_fingerprint(),
        "entries": {},
    }


def load_latest_cache(cache_path: str) -> dict[str, Any]:
    path = Path(cache_path)
    if not path.exists():
        return empty_latest_cache()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return empty_latest_cache()

    if (
        payload.get("cache_version") != CACHE_VERSION
        or payload.get("workflow_name") != WORKFLOW_NAME
        or payload.get("workflow_fingerprint") != workflow_fingerprint()
        or not isinstance(payload.get("entries"), dict)
    ):
        return empty_latest_cache()
    return payload


def save_latest_cache(cache_path: str, cache_payload: dict[str, Any]) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache_payload, indent=2, sort_keys=True))


def snapshot_path(snapshot_dir: str, snapshot_date: str) -> Path:
    path = Path(snapshot_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{snapshot_date}.json"


def _stratified_sample(pool: pd.DataFrame, n: int) -> pd.DataFrame:
    ordered = pool.sort_values(["maturity_years", "strike"]).reset_index(drop=True)
    step = max(1, len(ordered) // n)
    return ordered.iloc[::step].head(n).reset_index(drop=True)


def filter_short_term_options(
    surface_df: pd.DataFrame,
    spot: float,
    *,
    min_days: int = SHORT_TERM_MIN_DAYS,
    max_days: int = SHORT_TERM_MAX_DAYS,
    moneyness: float = SHORT_TERM_MONEYNESS,
    relaxed_moneyness: float = SHORT_TERM_RELAXED_MONEYNESS,
) -> pd.DataFrame:
    """Keep a short-dated, near-ATM slice for short-end calibration."""
    lo = min_days / 365.25
    hi = max_days / 365.25
    mask = (
        (surface_df["maturity_years"] >= lo)
        & (surface_df["maturity_years"] <= hi)
        & (surface_df["strike"] >= spot * (1.0 - moneyness))
        & (surface_df["strike"] <= spot * (1.0 + moneyness))
    )
    filtered = surface_df.loc[mask].copy()

    if len(filtered) < 5:
        relaxed_mask = (
            (surface_df["maturity_years"] >= lo)
            & (surface_df["maturity_years"] <= hi)
            & (surface_df["strike"] >= spot * (1.0 - relaxed_moneyness))
            & (surface_df["strike"] <= spot * (1.0 + relaxed_moneyness))
        )
        filtered = surface_df.loc[relaxed_mask].copy()

    return filtered.reset_index(drop=True)


def select_target_expiry(short_term_df: pd.DataFrame, *, target_days: int = TARGET_SMILE_DAYS) -> tuple[str, float]:
    """Choose the plotted expiry as the one closest to the target short maturity."""
    if short_term_df.empty:
        raise ValueError("Cannot select an expiry from an empty short-term surface.")

    expiry_table = (
        short_term_df.groupby("expiry_str", as_index=False)["maturity_years"]
        .first()
        .sort_values("maturity_years")
        .reset_index(drop=True)
    )
    target_maturity = target_days / 365.25
    expiry_table["distance"] = np.abs(expiry_table["maturity_years"] - target_maturity)
    best = expiry_table.sort_values(["distance", "maturity_years"]).iloc[0]
    return str(best["expiry_str"]), float(best["maturity_years"])


def available_expiry_payload(short_term_df: pd.DataFrame) -> list[dict[str, float | str]]:
    table = (
        short_term_df.groupby("expiry_str", as_index=False)["maturity_years"]
        .first()
        .sort_values("maturity_years")
        .reset_index(drop=True)
    )
    return [
        {
            "expiry_str": str(row.expiry_str),
            "maturity_years": float(row.maturity_years),
        }
        for row in table.itertuples(index=False)
    ]


def build_model_from_params(model_name: str, params: dict[str, float]) -> object:
    """Rebuild a model instance using the short-term workflow settings."""
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
            scheme=RB_SCHEME,
        )
    if model_name == "RoughHeston":
        return RoughHestonModel(
            hurst=params["hurst"],
            lam=params["lam"],
            theta=params["theta"],
            nu=params["nu"],
            rho=params["rho"],
            v0=params["v0"],
            scheme=RH_SCHEME,
            n_factors=RH_N_FACTORS,
        )
    raise ValueError(f"Unknown model: {model_name}")


def compute_model_iv_smile(
    model_name: str,
    params: dict[str, float],
    market_data: MarketData,
    maturity: float,
    strikes: list[float],
    *,
    engine_kwargs: dict[str, Any] | None = None,
) -> list[float | None]:
    """Reprice a strip of options and convert the prices to Black-Scholes IVs."""
    model = build_model_from_params(model_name, params)
    engine = MonteCarloEngine(**(engine_kwargs or VIZ_ENGINE))
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
    short_term_df: pd.DataFrame,
    market_data: MarketData,
    *,
    engine_kwargs: dict[str, Any] | None = None,
) -> float:
    """Compute IV RMSE on the short-term calibration bucket."""
    model = build_model_from_params(model_name, params)
    engine = MonteCarloEngine(**(engine_kwargs or VIZ_ENGINE))
    errors: list[float] = []

    for _, row in short_term_df.iterrows():
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


def model_settings_payload(model_name: str) -> dict[str, Any]:
    if model_name == "RoughBergomi":
        return {"scheme": RB_SCHEME}
    if model_name == "RoughHeston":
        return {"scheme": RH_SCHEME, "n_factors": RH_N_FACTORS}
    return {}


def empty_error_report(ticker: str, snapshot_date: str, error: str) -> ShortTermTickerReport:
    return ShortTermTickerReport(
        ticker=ticker,
        snapshot_date=snapshot_date,
        market_data=None,
        short_term_df=pd.DataFrame(),
        available_expiries=[],
        selected_expiry=None,
        selected_maturity=None,
        market_smile_df=pd.DataFrame(),
        results={model_name: None for model_name in MODEL_NAMES},
        model_settings={model_name: model_settings_payload(model_name) for model_name in MODEL_NAMES},
        model_smiles={model_name: [] for model_name in MODEL_NAMES},
        iv_rmse={model_name: float("nan") for model_name in MODEL_NAMES},
        status="error",
        error=error,
    )


def build_short_term_report(ticker: str, *, snapshot_date: str) -> ShortTermTickerReport:
    """Fetch market data, calibrate on the short bucket, and prepare plot-ready data."""
    market_data = get_market_data(ticker)
    if market_data is None:
        raise RuntimeError(f"Could not fetch market data for {ticker}")

    surface_df = get_option_surface(ticker, market_data)
    if surface_df.empty:
        raise RuntimeError(f"Could not fetch option surface for {ticker}")

    short_term_df = filter_short_term_options(surface_df, market_data.spot)
    if len(short_term_df) < 3:
        raise RuntimeError(f"Too few short-dated options for calibration ({len(short_term_df)})")

    selected_expiry, selected_maturity = select_target_expiry(short_term_df)
    market_smile_df = (
        short_term_df.loc[short_term_df["expiry_str"] == selected_expiry]
        .sort_values(["strike", "is_call"])
        .reset_index(drop=True)
    )

    atm_iv = float(short_term_df["implied_vol"].median())
    base_cols = ["strike", "maturity_years", "is_call", "market_price"]
    sample_frames = {
        model_name: _stratified_sample(short_term_df[base_cols], n=size)
        for model_name, size in CALIBRATION_SAMPLE_SIZES.items()
    }

    calibrators = {
        "GBM": make_gbm_calibrator(x0_sigma=atm_iv, engine_kwargs=CALIB_ENGINE_GBM),
        "Heston": make_heston_calibrator(x0_sigma=atm_iv, engine_kwargs=CALIB_ENGINE_HESTON),
        "RoughBergomi": make_rough_bergomi_calibrator(
            x0_sigma=atm_iv,
            engine_kwargs=CALIB_ENGINE_RB,
            scheme=RB_SCHEME,
        ),
        "RoughHeston": make_rough_heston_calibrator(
            x0_sigma=atm_iv,
            engine_kwargs=CALIB_ENGINE_RH,
            scheme=RH_SCHEME,
            n_factors=RH_N_FACTORS,
        ),
    }

    results: dict[str, CalibResult | None] = {}
    iv_rmse: dict[str, float] = {}
    model_smiles: dict[str, list[float | None]] = {}
    model_settings = {model_name: model_settings_payload(model_name) for model_name in MODEL_NAMES}

    strikes = [market_data.spot * moneyness for moneyness in DEFAULT_MONEYNESS_GRID]

    for model_name in MODEL_NAMES:
        calibrator = calibrators[model_name]
        opts_df = sample_frames[model_name]
        print(f"  [{ticker}] Calibrating {model_name} on {len(opts_df)} short-dated options...")
        try:
            calib_result = calibrator.calibrate(
                spot=market_data.spot,
                options_df=opts_df,
                rate=market_data.rate,
                div=market_data.div_yield,
            )
            results[model_name] = calib_result
            iv_rmse[model_name] = compute_iv_rmse(
                model_name=model_name,
                params=calib_result.params,
                short_term_df=short_term_df,
                market_data=market_data,
            )
            model_smiles[model_name] = compute_model_iv_smile(
                model_name=model_name,
                params=calib_result.params,
                market_data=market_data,
                maturity=selected_maturity,
                strikes=strikes,
            )
        except Exception as exc:
            print(f"  [ERROR] {ticker} {model_name} short-term calibration failed: {exc}")
            results[model_name] = None
            iv_rmse[model_name] = float("nan")
            model_smiles[model_name] = [None for _ in DEFAULT_MONEYNESS_GRID]

    return ShortTermTickerReport(
        ticker=ticker,
        snapshot_date=snapshot_date,
        market_data=market_data,
        short_term_df=short_term_df,
        available_expiries=available_expiry_payload(short_term_df),
        selected_expiry=selected_expiry,
        selected_maturity=selected_maturity,
        market_smile_df=market_smile_df,
        results=results,
        model_settings=model_settings,
        model_smiles=model_smiles,
        iv_rmse=iv_rmse,
        status="ok",
        error=None,
    )


def report_to_payload(report: ShortTermTickerReport) -> dict[str, Any]:
    return {
        "ticker": report.ticker,
        "snapshot_date": report.snapshot_date,
        "market_data": None
        if report.market_data is None
        else {
            "spot": float(report.market_data.spot),
            "rate": float(report.market_data.rate),
            "div_yield": float(report.market_data.div_yield),
        },
        "short_term_df": dataframe_to_cache_payload(report.short_term_df),
        "available_expiries": report.available_expiries,
        "selected_expiry": report.selected_expiry,
        "selected_maturity": report.selected_maturity,
        "market_smile_df": dataframe_to_cache_payload(report.market_smile_df),
        "results": {
            model_name: {
                "calibration": calib_result_to_payload(result),
                "settings": report.model_settings.get(model_name, {}),
            }
            for model_name, result in report.results.items()
        },
        "model_smiles": {
            model_name: [None if value is None else float(value) for value in values]
            for model_name, values in report.model_smiles.items()
        },
        "iv_rmse": {
            model_name: float(value) for model_name, value in report.iv_rmse.items()
        },
        "status": report.status,
        "error": report.error,
    }


def report_from_payload(payload: dict[str, Any]) -> ShortTermTickerReport:
    market_data_payload = payload.get("market_data")
    market_data = None
    if market_data_payload is not None:
        market_data = MarketData(
            spot=float(market_data_payload["spot"]),
            rate=float(market_data_payload["rate"]),
            div_yield=float(market_data_payload["div_yield"]),
        )

    results_payload = payload.get("results", {})
    return ShortTermTickerReport(
        ticker=str(payload["ticker"]),
        snapshot_date=str(payload["snapshot_date"]),
        market_data=market_data,
        short_term_df=dataframe_from_cache_payload(payload["short_term_df"]),
        available_expiries=list(payload.get("available_expiries", [])),
        selected_expiry=payload.get("selected_expiry"),
        selected_maturity=(
            None if payload.get("selected_maturity") is None else float(payload["selected_maturity"])
        ),
        market_smile_df=dataframe_from_cache_payload(payload["market_smile_df"]),
        results={
            model_name: calib_result_from_payload((result_payload or {}).get("calibration"))
            for model_name, result_payload in results_payload.items()
        },
        model_settings={
            model_name: dict((result_payload or {}).get("settings", {}))
            for model_name, result_payload in results_payload.items()
        },
        model_smiles={
            model_name: [None if value is None else float(value) for value in values]
            for model_name, values in payload.get("model_smiles", {}).items()
        },
        iv_rmse={
            model_name: float(value) for model_name, value in payload.get("iv_rmse", {}).items()
        },
        status=str(payload.get("status", "ok")),
        error=payload.get("error"),
    )


def snapshot_to_payload(snapshot: ShortTermSnapshot) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "workflow_name": WORKFLOW_NAME,
        "workflow_fingerprint": snapshot.workflow_fingerprint,
        "created_at": snapshot.created_at,
        "snapshot_date": snapshot.snapshot_date,
        "basket": snapshot.basket,
        "config": workflow_settings(),
        "reports": {
            ticker: report_to_payload(report)
            for ticker, report in snapshot.reports.items()
        },
    }


def snapshot_from_payload(payload: dict[str, Any]) -> ShortTermSnapshot:
    return ShortTermSnapshot(
        snapshot_date=str(payload["snapshot_date"]),
        created_at=str(payload["created_at"]),
        basket=[str(ticker) for ticker in payload.get("basket", [])],
        reports={
            ticker: report_from_payload(report_payload)
            for ticker, report_payload in payload.get("reports", {}).items()
        },
        workflow_fingerprint=str(payload["workflow_fingerprint"]),
    )


def load_snapshot_file(path: str | Path) -> ShortTermSnapshot | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        payload = json.loads(file_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    if (
        payload.get("cache_version") != CACHE_VERSION
        or payload.get("workflow_name") != WORKFLOW_NAME
        or payload.get("workflow_fingerprint") != workflow_fingerprint()
    ):
        return None
    return snapshot_from_payload(payload)


def save_snapshot_file(snapshot: ShortTermSnapshot, *, snapshot_dir: str) -> Path:
    path = snapshot_path(snapshot_dir, snapshot.snapshot_date)
    path.write_text(json.dumps(snapshot_to_payload(snapshot), indent=2, sort_keys=True))
    return path


def update_latest_cache(
    cache_payload: dict[str, Any],
    snapshot: ShortTermSnapshot,
) -> dict[str, Any]:
    entries = dict(cache_payload.get("entries", {}))
    for ticker, report in snapshot.reports.items():
        entries[ticker] = {
            "snapshot_date": snapshot.snapshot_date,
            "cached_at": snapshot.created_at,
            "report": report_to_payload(report),
        }
    return {
        "cache_version": CACHE_VERSION,
        "workflow_name": WORKFLOW_NAME,
        "workflow_fingerprint": workflow_fingerprint(),
        "entries": entries,
    }


def snapshot_from_latest_cache(
    cache_payload: dict[str, Any],
    tickers: list[str],
) -> ShortTermSnapshot | None:
    entries = cache_payload.get("entries", {})
    if not all(ticker in entries for ticker in tickers):
        return None

    reports = {
        ticker: report_from_payload(entries[ticker]["report"])
        for ticker in tickers
    }
    snapshot_dates = sorted({report.snapshot_date for report in reports.values() if report.snapshot_date})
    synthetic_date = snapshot_dates[0] if len(snapshot_dates) == 1 else "cached-latest"
    created_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return ShortTermSnapshot(
        snapshot_date=synthetic_date,
        created_at=created_at,
        basket=tickers,
        reports=reports,
        workflow_fingerprint=workflow_fingerprint(),
    )


def slice_snapshot(snapshot: ShortTermSnapshot, tickers: list[str]) -> ShortTermSnapshot:
    return ShortTermSnapshot(
        snapshot_date=snapshot.snapshot_date,
        created_at=snapshot.created_at,
        basket=tickers,
        reports={ticker: snapshot.reports[ticker] for ticker in tickers if ticker in snapshot.reports},
        workflow_fingerprint=snapshot.workflow_fingerprint,
    )


def load_or_build_snapshot(
    tickers: list[str] | None = None,
    *,
    cache_path: str = DEFAULT_CACHE_PATH,
    snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
    refresh_cache: bool = False,
    snapshot_date: str | None = None,
) -> tuple[ShortTermSnapshot, bool]:
    """Return a cached short-term snapshot when possible, otherwise build a new one."""
    basket = normalize_tickers(tickers)
    resolved_date = resolve_snapshot_date(snapshot_date)
    cache_payload = load_latest_cache(cache_path)

    if not refresh_cache:
        dated_snapshot = load_snapshot_file(snapshot_path(snapshot_dir, resolved_date))
        if dated_snapshot is not None and all(ticker in dated_snapshot.reports for ticker in basket):
            return slice_snapshot(dated_snapshot, basket), False

        latest_snapshot = snapshot_from_latest_cache(cache_payload, basket)
        if latest_snapshot is not None:
            return latest_snapshot, False

    created_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    reports: dict[str, ShortTermTickerReport] = {}
    for ticker in basket:
        try:
            reports[ticker] = build_short_term_report(ticker, snapshot_date=resolved_date)
        except Exception as exc:
            reports[ticker] = empty_error_report(ticker, resolved_date, str(exc))

    snapshot = ShortTermSnapshot(
        snapshot_date=resolved_date,
        created_at=created_at,
        basket=basket,
        reports=reports,
        workflow_fingerprint=workflow_fingerprint(),
    )
    save_snapshot_file(snapshot, snapshot_dir=snapshot_dir)
    save_latest_cache(cache_path, update_latest_cache(cache_payload, snapshot))
    return snapshot, True


def collect_short_term_snapshot(
    tickers: list[str] | None = None,
    *,
    cache_path: str = DEFAULT_CACHE_PATH,
    snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
    refresh_cache: bool = False,
    snapshot_date: str | None = None,
) -> ShortTermSnapshot:
    """Public workflow entry point used by plot/animation scripts."""
    basket = normalize_tickers(tickers)
    print("=" * 60)
    print("  Short-Term Calibration Snapshot")
    print(f"  Basket: {basket}")
    print(f"  Short bucket: {SHORT_TERM_MIN_DAYS}-{SHORT_TERM_MAX_DAYS}D")
    print(f"  Smile slice: expiry closest to {TARGET_SMILE_DAYS}D")
    print("=" * 60)

    snapshot, rebuilt = load_or_build_snapshot(
        basket,
        cache_path=cache_path,
        snapshot_dir=snapshot_dir,
        refresh_cache=refresh_cache,
        snapshot_date=snapshot_date,
    )

    if rebuilt:
        print(f"Built new snapshot for {snapshot.snapshot_date}")
    else:
        print(f"Using cached snapshot ({snapshot.snapshot_date})")

    for ticker in basket:
        report = snapshot.reports.get(ticker)
        if report is None or report.status != "ok":
            message = report.error if report is not None else "missing from snapshot"
            print(f"  {ticker:<6} ERROR  {message}")
            continue
        maturity_days = report.selected_maturity * 365.25 if report.selected_maturity is not None else float("nan")
        rmse_bits = "  ".join(
            f"{MODEL_LABELS[model_name]}={report.iv_rmse.get(model_name, float('nan')):.4f}"
            for model_name in MODEL_NAMES
        )
        print(
            f"  {ticker:<6} OK  "
            f"n={len(report.short_term_df):>3}  "
            f"T~{maturity_days:>5.1f}d  "
            f"{rmse_bits}",
        )
    print(f"Cache file: {cache_path}")
    print(f"Snapshot dir: {snapshot_dir}")
    return snapshot


def iter_snapshot_files(
    snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[Path]:
    """List valid dated snapshot files in ascending date order."""
    path = Path(snapshot_dir)
    if not path.exists():
        return []

    lo = resolve_snapshot_date(date_from) if date_from else None
    hi = resolve_snapshot_date(date_to) if date_to else None
    files: list[Path] = []
    for file_path in sorted(path.glob("*.json")):
        stem = file_path.stem
        try:
            snapshot_day = date.fromisoformat(stem).isoformat()
        except ValueError:
            continue
        if lo is not None and snapshot_day < lo:
            continue
        if hi is not None and snapshot_day > hi:
            continue
        if load_snapshot_file(file_path) is None:
            continue
        files.append(file_path)
    return files


def load_snapshot_series(
    snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[ShortTermSnapshot]:
    return [
        snapshot
        for path in iter_snapshot_files(snapshot_dir, date_from=date_from, date_to=date_to)
        if (snapshot := load_snapshot_file(path)) is not None
    ]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    collect_short_term_snapshot(
        args.tickers,
        cache_path=args.cache_path,
        snapshot_dir=args.snapshot_dir,
        refresh_cache=args.refresh_cache,
        snapshot_date=args.snapshot_date,
    )


if __name__ == "__main__":
    main()
