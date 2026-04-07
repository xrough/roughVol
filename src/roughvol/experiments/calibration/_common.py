from __future__ import annotations

import argparse

from roughvol.experiments.calibration.run_calibration_demo import (
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
    return parser.parse_args()


def build_results(args: argparse.Namespace) -> dict[str, TickerCalibrationReport]:
    return collect_calibration_results(args.tickers)


def successful_reports(
    all_results: dict[str, TickerCalibrationReport],
) -> list[TickerCalibrationReport]:
    return [report for report in all_results.values() if report.error is None]
