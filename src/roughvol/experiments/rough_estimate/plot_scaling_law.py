"""Single-panel cross-sectional scaling-law view for empirical rough volatility.

Run with:
    python -m roughvol.experiments.rough_estimate.plot_scaling_law --top-n 50
    python -m roughvol.experiments.rough_estimate.plot_scaling_law SPY AAPL MSFT
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roughvol.experiments.rough_estimate._common import build_ranked_full_reports, parse_ranked_report_args
from roughvol.experiments.rough_estimate._style import (
    DEEP_BLUE,
    LIGHT_GREY,
    MID_BLUE,
    PALE_BLUE,
    SLATE_GREY,
    configure_libertine_style,
)
from roughvol.experiments.rough_estimate.run_empirical_roughness_demo import output_figure_name

OUTPUT_FIGURE = output_figure_name("scaling_law")


def main(argv: list[str] | None = None) -> None:
    args = parse_ranked_report_args(
        "Plot a single cross-sectional scaling-law summary.",
        allow_explicit_tickers=True,
        default_top_n=0,
    )
    configure_libertine_style()
    reports, _ = build_ranked_full_reports(args)

    if not reports:
        print("No scaling-law figure was generated because all ticker runs failed.")
        return

    common_len = min(len(report["roughness"].lags) for report in reports)
    common_lags = reports[0]["roughness"].lags[:common_len]
    structure_matrix = np.vstack(
        [report["roughness"].structure_function[:common_len] for report in reports]
    )
    median_structure = np.median(structure_matrix, axis=0)
    q25 = np.quantile(structure_matrix, 0.25, axis=0)
    q75 = np.quantile(structure_matrix, 0.75, axis=0)
    median_h = float(np.median([report["roughness"].hurst for report in reports]))
    intercept = float(np.mean(np.log(median_structure) - median_h * np.log(common_lags)))
    fitted = np.exp(intercept + median_h * np.log(common_lags))

    fig, ax = plt.subplots(figsize=(8.6, 6.1))

    for report in reports:
        roughness = report["roughness"]
        ax.loglog(
            roughness.lags[:common_len],
            roughness.structure_function[:common_len],
            color=MID_BLUE,
            linewidth=0.75,
            alpha=0.16,
        )
    ax.fill_between(common_lags, q25, q75, color=PALE_BLUE, alpha=0.65, label="Interquartile band")
    ax.loglog(common_lags, median_structure, color=DEEP_BLUE, linewidth=2.3, label="Cross-sectional median")
    ax.loglog(common_lags, fitted, color=SLATE_GREY, linestyle="--", linewidth=1.5, label=fr"Median-slope fit $H={median_h:.2f}$")

    ax.set_title(
        f"Cross-Sectional Scaling Law ({len(reports)} stocks)",
        fontsize=17,
        pad=10,
    )
    ax.set_xlabel("Lag (RV blocks)")
    ax.set_ylabel(r"$\mathrm{E}\!\left[\left|\log \mathrm{RV}_{t+\Delta}-\log \mathrm{RV}_t\right|\right]$")
    ax.grid(alpha=0.35)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=260)
    plt.close(fig)
    print(f"Saved figure: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
