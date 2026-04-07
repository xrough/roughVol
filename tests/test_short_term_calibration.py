from __future__ import annotations

import json

import pandas as pd

from roughvol.calibration.calibration import CalibResult, make_rough_heston_calibrator
from roughvol.experiments.calibration._short_term_panel import render_short_term_panel
from roughvol.experiments.calibration.animate_short_term_panel import build_animation
from roughvol.experiments.calibration.run_short_term_calibration_demo import (
    DEFAULT_CACHE_PATH,
    MODEL_NAMES,
    ShortTermSnapshot,
    ShortTermTickerReport,
    empty_latest_cache,
    filter_short_term_options,
    load_latest_cache,
    load_or_build_snapshot,
    load_snapshot_series,
    report_from_payload,
    report_to_payload,
    resolve_snapshot_date,
    save_latest_cache,
    save_snapshot_file,
    select_target_expiry,
    snapshot_from_payload,
    snapshot_to_payload,
    update_latest_cache,
    workflow_fingerprint,
)
from roughvol.types import MarketData


def _sample_short_report(*, ticker: str = "AAPL", snapshot_date: str = "2026-04-07") -> ShortTermTickerReport:
    short_term_df = pd.DataFrame(
        {
            "strike": [95.0, 100.0, 105.0, 95.0, 100.0, 105.0],
            "maturity_years": [20 / 365.25, 20 / 365.25, 20 / 365.25, 32 / 365.25, 32 / 365.25, 32 / 365.25],
            "is_call": [True, True, True, False, False, False],
            "market_price": [8.2, 4.1, 1.5, 3.8, 5.1, 8.0],
            "implied_vol": [0.25, 0.22, 0.23, 0.26, 0.24, 0.25],
            "expiry_str": ["2026-04-27", "2026-04-27", "2026-04-27", "2026-05-09", "2026-05-09", "2026-05-09"],
        }
    )
    market_smile_df = short_term_df.loc[short_term_df["expiry_str"] == "2026-05-09"].reset_index(drop=True)
    base_result = CalibResult(
        model_name="RoughHeston",
        params={"hurst": 0.12, "lam": 1.1, "theta": 0.04, "nu": 0.3, "rho": -0.7, "v0": 0.04},
        mse=0.001,
        per_option_ivols=[0.21, 0.22],
        elapsed_s=1.5,
    )
    return ShortTermTickerReport(
        ticker=ticker,
        snapshot_date=snapshot_date,
        market_data=MarketData(spot=100.0, rate=0.03, div_yield=0.01),
        short_term_df=short_term_df,
        available_expiries=[
            {"expiry_str": "2026-04-27", "maturity_years": 20 / 365.25},
            {"expiry_str": "2026-05-09", "maturity_years": 32 / 365.25},
        ],
        selected_expiry="2026-05-09",
        selected_maturity=32 / 365.25,
        market_smile_df=market_smile_df,
        results={
            "GBM": CalibResult(model_name="GBM", params={"sigma": 0.2}, mse=0.01, elapsed_s=0.5),
            "Heston": CalibResult(model_name="Heston", params={"kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.5, "v0": 0.04}, mse=0.004, elapsed_s=0.8),
            "RoughBergomi": CalibResult(model_name="RoughBergomi", params={"hurst": 0.1, "eta": 1.8, "rho": -0.8, "xi0": 0.04}, mse=0.003, elapsed_s=1.1),
            "RoughHeston": base_result,
        },
        model_settings={
            "GBM": {},
            "Heston": {},
            "RoughBergomi": {"scheme": "blp-hybrid"},
            "RoughHeston": {"scheme": "bayer-breneis", "n_factors": 8},
        },
        model_smiles={
            "GBM": [0.24, 0.23, 0.22, 0.21, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27],
            "Heston": [0.26, 0.25, 0.24, 0.23, 0.22, 0.22, 0.23, 0.24, 0.25, 0.27, 0.28],
            "RoughBergomi": [0.27, 0.26, 0.25, 0.23, 0.22, 0.21, 0.22, 0.24, 0.26, 0.28, 0.29],
            "RoughHeston": [0.28, 0.27, 0.25, 0.23, 0.22, 0.21, 0.22, 0.24, 0.26, 0.29, 0.30],
        },
        iv_rmse={"GBM": 0.03, "Heston": 0.02, "RoughBergomi": 0.015, "RoughHeston": 0.012},
        status="ok",
        error=None,
    )


def _sample_snapshot(snapshot_date: str = "2026-04-07", *, include_missing: bool = False) -> ShortTermSnapshot:
    reports = {ticker: _sample_short_report(ticker=ticker, snapshot_date=snapshot_date) for ticker in ["AAPL", "MSFT"]}
    if include_missing:
        reports["NVDA"] = ShortTermTickerReport(
            ticker="NVDA",
            snapshot_date=snapshot_date,
            market_data=None,
            short_term_df=pd.DataFrame(),
            available_expiries=[],
            selected_expiry=None,
            selected_maturity=None,
            market_smile_df=pd.DataFrame(),
            results={model_name: None for model_name in MODEL_NAMES},
            model_settings={model_name: {} for model_name in MODEL_NAMES},
            model_smiles={model_name: [] for model_name in MODEL_NAMES},
            iv_rmse={model_name: float("nan") for model_name in MODEL_NAMES},
            status="error",
            error="missing",
        )
    return ShortTermSnapshot(
        snapshot_date=snapshot_date,
        created_at="2026-04-07T18:00:00Z",
        basket=["AAPL", "MSFT", "NVDA"] if include_missing else ["AAPL", "MSFT"],
        reports=reports,
        workflow_fingerprint=workflow_fingerprint(),
    )


def test_short_term_filter_and_target_expiry_selection():
    surface_df = pd.DataFrame(
        {
            "strike": [90.0, 100.0, 110.0, 100.0, 100.0],
            "maturity_years": [10 / 365.25, 20 / 365.25, 30 / 365.25, 50 / 365.25, 40 / 365.25],
            "is_call": [True, True, True, True, True],
            "market_price": [12.0, 6.0, 3.0, 2.0, 4.0],
            "implied_vol": [0.3, 0.25, 0.22, 0.2, 0.21],
            "expiry_str": ["2026-04-17", "2026-04-27", "2026-05-07", "2026-05-27", "2026-05-17"],
        }
    )

    filtered = filter_short_term_options(surface_df, spot=100.0)

    assert all((14 / 365.25) <= value <= (45 / 365.25) for value in filtered["maturity_years"])
    expiry, maturity = select_target_expiry(filtered)
    assert expiry == "2026-05-07"
    assert round(maturity * 365.25) == 30


def test_short_term_report_payload_round_trip_preserves_rough_heston_settings():
    report = _sample_short_report()
    restored = report_from_payload(report_to_payload(report))

    assert restored.ticker == "AAPL"
    assert restored.market_data is not None
    assert restored.results["RoughHeston"] is not None
    assert restored.results["RoughHeston"].params["hurst"] == 0.12
    assert restored.model_settings["RoughHeston"]["scheme"] == "bayer-breneis"


def test_snapshot_payload_round_trip_and_fingerprint():
    snapshot = _sample_snapshot(include_missing=True)
    restored = snapshot_from_payload(snapshot_to_payload(snapshot))

    assert restored.snapshot_date == "2026-04-07"
    assert restored.workflow_fingerprint == workflow_fingerprint()
    assert restored.reports["AAPL"].selected_expiry == "2026-05-09"
    assert restored.reports["NVDA"].status == "error"


def test_load_latest_cache_rejects_fingerprint_mismatch(tmp_path):
    bad_cache = tmp_path / "short_term_cache.json"
    bad_cache.write_text(
        json.dumps(
            {
                "cache_version": 1,
                "workflow_name": "short_term_calibration_animation",
                "workflow_fingerprint": "stale",
                "entries": {"AAPL": {"snapshot_date": "2026-04-07"}},
            }
        )
    )

    payload = load_latest_cache(str(bad_cache))
    assert payload["workflow_fingerprint"] == workflow_fingerprint()
    assert payload["entries"] == {}


def test_load_or_build_snapshot_uses_latest_cache_without_live_fetch(monkeypatch, tmp_path):
    cache_path = tmp_path / "short_term_cache.json"
    cache_payload = empty_latest_cache()
    snapshot = _sample_snapshot()
    save_latest_cache(str(cache_path), update_latest_cache(cache_payload, snapshot))

    def boom(*args, **kwargs):
        raise AssertionError("live fetch should not be called")

    monkeypatch.setattr(
        "roughvol.experiments.calibration.run_short_term_calibration_demo.build_short_term_report",
        boom,
    )

    loaded, rebuilt = load_or_build_snapshot(
        ["AAPL", "MSFT"],
        cache_path=str(cache_path),
        snapshot_dir=str(tmp_path / "snapshots"),
        refresh_cache=False,
        snapshot_date="2026-04-08",
    )

    assert rebuilt is False
    assert loaded.snapshot_date == "2026-04-07"
    assert set(loaded.reports) == {"AAPL", "MSFT"}


def test_render_panel_and_animation_from_cached_snapshots(tmp_path):
    snapshot_dir = tmp_path / "short_term_snapshots"
    save_snapshot_file(_sample_snapshot("2026-04-07", include_missing=True), snapshot_dir=str(snapshot_dir))
    save_snapshot_file(_sample_snapshot("2026-04-08", include_missing=True), snapshot_dir=str(snapshot_dir))

    png_out = tmp_path / "panel.png"
    render_short_term_panel(
        _sample_snapshot("2026-04-07", include_missing=True),
        tickers=["AAPL", "MSFT", "NVDA"],
        out=str(png_out),
    )
    assert png_out.exists()

    series = load_snapshot_series(str(snapshot_dir), date_from="2026-04-07", date_to="2026-04-08")
    assert [snapshot.snapshot_date for snapshot in series] == ["2026-04-07", "2026-04-08"]

    gif_out = tmp_path / "panel.gif"
    built = build_animation(
        tickers=["AAPL", "MSFT", "NVDA"],
        snapshot_dir=str(snapshot_dir),
        date_from="2026-04-07",
        date_to="2026-04-08",
        fps=4,
        fmt="gif",
        out=str(gif_out),
    )
    assert built == str(gif_out)
    assert gif_out.exists()


def test_rough_heston_calibrator_factory_exists():
    calibrator = make_rough_heston_calibrator(x0_sigma=0.2, scheme="bayer-breneis")
    assert calibrator.model_name == "RoughHeston"
    assert calibrator.param_names == ["hurst", "lam", "theta", "nu", "rho", "v0"]


def test_default_cache_path_targets_calibration_output():
    assert DEFAULT_CACHE_PATH.endswith("output/calibration/short_term_calibration_cache.json")
    assert resolve_snapshot_date("2026-04-07") == "2026-04-07"
