from __future__ import annotations

import pandas as pd

from roughvol.analytics.black_scholes_formula import bs_price
from roughvol.service.toolbox import CalibrationToolbox


def test_windowed_calibration_reuses_cached_snapshot_within_interval():
    toolbox = CalibrationToolbox()
    spot = 100.0

    timed_quotes = pd.DataFrame(
        [
            {
                "strike": 100.0,
                "maturity_years": 0.5,
                "is_call": True,
                "market_price": bs_price(
                    spot=spot,
                    strike=100.0,
                    maturity=0.5,
                    rate=0.01,
                    div=0.0,
                    vol=0.2,
                    is_call=True,
                ),
                "observed_at_ms": 1_000,
            },
            {
                "strike": 105.0,
                "maturity_years": 1.0,
                "is_call": True,
                "market_price": bs_price(
                    spot=spot,
                    strike=105.0,
                    maturity=1.0,
                    rate=0.01,
                    div=0.0,
                    vol=0.2,
                    is_call=True,
                ),
                "observed_at_ms": 1_400,
            },
        ]
    )

    first = toolbox.calibrate_windowed(
        asset_id="spx",
        model_name="BS",
        spot=spot,
        timed_quotes_df=timed_quotes,
        rate=0.01,
        div=0.0,
        as_of_ms=1_500,
        calibration_window_ms=1_000,
        update_interval_ms=500,
    )
    assert first.recalibrated is True
    assert first.quotes_in_window == 2

    second = toolbox.calibrate_windowed(
        asset_id="spx",
        model_name="BS",
        spot=spot,
        timed_quotes_df=timed_quotes,
        rate=0.01,
        div=0.0,
        as_of_ms=1_800,
        calibration_window_ms=1_000,
        update_interval_ms=500,
    )
    assert second.recalibrated is False
    assert second.calibrated_at_ms == first.calibrated_at_ms
    assert second.next_update_due_ms == first.next_update_due_ms


def test_windowed_calibration_refreshes_after_interval_and_filters_window():
    toolbox = CalibrationToolbox()
    spot = 100.0

    timed_quotes = pd.DataFrame(
        [
            {
                "strike": 95.0,
                "maturity_years": 0.25,
                "is_call": True,
                "market_price": bs_price(
                    spot=spot,
                    strike=95.0,
                    maturity=0.25,
                    rate=0.0,
                    div=0.0,
                    vol=0.18,
                    is_call=True,
                ),
                "observed_at_ms": 1_000,
            },
            {
                "strike": 100.0,
                "maturity_years": 0.5,
                "is_call": True,
                "market_price": bs_price(
                    spot=spot,
                    strike=100.0,
                    maturity=0.5,
                    rate=0.0,
                    div=0.0,
                    vol=0.22,
                    is_call=True,
                ),
                "observed_at_ms": 1_800,
            },
            {
                "strike": 105.0,
                "maturity_years": 1.0,
                "is_call": True,
                "market_price": bs_price(
                    spot=spot,
                    strike=105.0,
                    maturity=1.0,
                    rate=0.0,
                    div=0.0,
                    vol=0.24,
                    is_call=True,
                ),
                "observed_at_ms": 2_200,
            },
        ]
    )

    result = toolbox.calibrate_windowed(
        asset_id="spx",
        model_name="BS",
        spot=spot,
        timed_quotes_df=timed_quotes,
        rate=0.0,
        div=0.0,
        as_of_ms=2_400,
        calibration_window_ms=500,
        update_interval_ms=300,
    )

    assert result.recalibrated is True
    assert result.window_start_ms == 1_900
    assert result.quotes_in_window == 1
    assert result.quotes_total == 3
