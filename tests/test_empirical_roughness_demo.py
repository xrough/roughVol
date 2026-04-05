from __future__ import annotations

import pandas as pd
import pytest

yfinance = pytest.importorskip("yfinance")

from roughvol.experiments.run_empirical_roughness_demo import (  # noqa: E402
    DEFAULT_PRICE_HISTORY_INTERVAL,
    DEFAULT_RV_BLOCK_SIZE,
    block_label,
    default_period_for_interval,
    parse_args,
    recent_intraday_zoom_series,
    stable_seed_from_ticker,
)


def test_default_interval_is_one_minute():
    assert DEFAULT_PRICE_HISTORY_INTERVAL == "1m"
    assert DEFAULT_RV_BLOCK_SIZE == 30


def test_default_period_mapping_matches_interval():
    assert default_period_for_interval("1m") == "7d"
    assert default_period_for_interval("5m") == "60d"
    assert default_period_for_interval("1d") == "5y"


def test_parse_args_keeps_custom_interval_and_rv_block_size():
    args = parse_args(["SPY", "NVDA", "--interval", "30m", "--rv-block-size", "42"])

    assert args.tickers == ["SPY", "NVDA"]
    assert args.interval == "30m"
    assert args.rv_block_size == 42


def test_parse_args_accepts_legacy_window_alias():
    args = parse_args(["SPY", "--window", "18"])

    assert args.rv_block_size_alias == 18


def test_block_label_formats_intraday_blocks():
    assert block_label("1m", 30) == "30min RV blocks"
    assert block_label("30m", 2) == "1h RV blocks"


def test_stable_seed_from_ticker_is_reproducible_and_ticker_specific():
    assert stable_seed_from_ticker("SPY") == stable_seed_from_ticker("spy")
    assert stable_seed_from_ticker("SPY") != stable_seed_from_ticker("AAPL")


def test_recent_intraday_zoom_series_uses_last_couple_of_hours():
    index = pd.date_range("2024-01-02 09:30:00", periods=240, freq="1min")
    series = pd.Series(range(240), index=index, dtype=float)

    zoom = recent_intraday_zoom_series(series, interval="1m", zoom_hours=2)

    assert len(zoom) == 120
    assert zoom.index[0] == index[-120]
    assert zoom.index[-1] == index[-1]
