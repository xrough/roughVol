from __future__ import annotations

import argparse

from roughvol.experiments.calibration.run_calibration_demo import (
    DEFAULT_CACHE_PATH,
    TICKERS,
    TickerCalibrationReport,
    collect_calibration_results,
)


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "tickers",
        nargs="*",
        default=TICKERS,
        help="Ticker symbols to calibrate. Defaults to SPY AAPL.",
    )
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def build_results(args: argparse.Namespace) -> dict[str, TickerCalibrationReport]:
    return collect_calibration_results(
        args.tickers,
        cache_path=args.cache_path,
        refresh_cache=args.refresh_cache,
    )


def successful_reports(
    all_results: dict[str, TickerCalibrationReport],
) -> list[TickerCalibrationReport]:
    return [report for report in all_results.values() if report.error is None]
