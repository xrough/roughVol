from __future__ import annotations

import pandas as pd

from roughvol.calibration.calibration import CalibResult
from roughvol.experiments.calibration.run_calibration_demo import (
    DEFAULT_CACHE_PATH,
    TickerCalibrationReport,
    cache_entry_from_report,
    calibration_cache_key,
    parse_args,
    report_from_cache_entry,
)
from roughvol.types import MarketData


def test_parse_args_accepts_cache_controls():
    args = parse_args(["SPY", "--cache-path", "custom_calibration_cache.json", "--refresh-cache"])

    assert args.tickers == ["SPY"]
    assert args.cache_path == "custom_calibration_cache.json"
    assert args.refresh_cache is True


def test_default_cache_path_targets_calibration_output():
    assert DEFAULT_CACHE_PATH.endswith("output/calibration/calibration_cache.json")


def test_calibration_cache_key_is_uppercase():
    assert calibration_cache_key("spy") == "SPY"


def test_calibration_report_cache_round_trip():
    report = TickerCalibrationReport(
        ticker="SPY",
        market_data=MarketData(spot=500.0, rate=0.04, div_yield=0.01),
        surface_df=pd.DataFrame(
            {
                "strike": [480.0, 500.0],
                "maturity_years": [0.1, 0.1],
                "is_call": [True, True],
                "implied_vol": [0.22, 0.20],
            }
        ),
        calib_df=pd.DataFrame(
            {
                "strike": [480.0, 500.0],
                "maturity_years": [0.1, 0.1],
                "is_call": [True, True],
                "market_price": [30.0, 15.0],
                "implied_vol": [0.22, 0.20],
                "expiry_str": ["2026-05-15", "2026-05-15"],
            }
        ),
        results={
            "GBM": CalibResult(model_name="GBM", params={"sigma": 0.2}, mse=0.01, per_option_ivols=[0.2], elapsed_s=1.2),
            "Heston": None,
            "RoughBergomi": CalibResult(
                model_name="RoughBergomi",
                params={"hurst": 0.12, "eta": 2.1, "rho": -0.8, "xi0": 0.04},
                mse=0.005,
                per_option_ivols=[0.21, 0.22],
                elapsed_s=3.4,
            ),
        },
        iv_rmse={"GBM": 0.03, "Heston": float("nan"), "RoughBergomi": 0.02},
        error=None,
    )

    restored = report_from_cache_entry(cache_entry_from_report(report))

    assert restored.ticker == "SPY"
    assert restored.market_data.spot == 500.0
    assert list(restored.surface_df["strike"]) == [480.0, 500.0]
    assert restored.results["GBM"] is not None
    assert restored.results["GBM"].params["sigma"] == 0.2
    assert restored.results["Heston"] is None
    assert restored.results["RoughBergomi"] is not None
    assert restored.results["RoughBergomi"].params["hurst"] == 0.12
    assert restored.iv_rmse["RoughBergomi"] == 0.02
