from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roughvol.experiments._paths import output_path
from roughvol.experiments.calibration._common import build_results, parse_args, successful_reports
from roughvol.experiments.calibration.run_calibration_demo import (
    MODEL_COLOURS,
    MODEL_LABELS,
    TickerCalibrationReport,
)


def plot_rmse_bars(
    all_results: dict[str, TickerCalibrationReport],
    out: str | None = None,
) -> None:
    reports = successful_reports(all_results)
    if not reports:
        return

    model_names = ["GBM", "Heston", "RoughBergomi"]
    tickers = [report.ticker for report in reports]
    x = np.arange(len(tickers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(tickers)), 5))

    for i, model_name in enumerate(model_names):
        rmses = [report.iv_rmse.get(model_name, float("nan")) * 100 for report in reports]
        bars = ax.bar(
            x + i * width,
            rmses,
            width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLOURS[model_name],
            alpha=0.85,
            edgecolor="white",
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.1f}" if not np.isnan(value) else "N/A" for value in rmses],
            fontsize=8,
            padding=2,
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel("IV RMSE (vol ppts)", fontsize=10)
    ax.set_title("Calibration Quality: IV RMSE per Model and Ticker", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = out or output_path("calibration", "calibration_demo_rmse_bars.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    args = parse_args("Plot grouped IV RMSE bars for the calibrated models.")
    results = build_results(args)
    if not results:
        return
    plot_rmse_bars(results)


if __name__ == "__main__":
    main()
