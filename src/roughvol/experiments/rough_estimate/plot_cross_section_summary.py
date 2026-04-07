"""Cross-sectional roughness summary view.

Produces one figure with a single cross-sectional scaling-law panel
for the top-N stocks by market cap.

Run with:
    python -m roughvol.experiments.rough_estimate.plot_cross_section_summary --top-n 50
"""

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

OUTPUT_FIGURE = output_figure_name("cross_section_summary")


def main(argv: list[str] | None = None) -> None:
    args = parse_ranked_report_args("Plot a single cross-sectional scaling-law summary.")
    configure_libertine_style()
    reports, failures = build_ranked_full_reports(args)

    if not reports:
        print("No cross-sectional summary figure was generated because no H estimates succeeded.")
        return

    common_len = min(len(report["roughness"].lags) for report in reports)
    scaling_reports = [report for report in reports if len(report["roughness"].lags) >= common_len]
    common_lags = scaling_reports[0]["roughness"].lags[:common_len]
    structure_matrix = np.vstack(
        [report["roughness"].structure_function[:common_len] for report in scaling_reports]
    )
    median_structure = np.median(structure_matrix, axis=0)
    q25 = np.quantile(structure_matrix, 0.25, axis=0)
    q75 = np.quantile(structure_matrix, 0.75, axis=0)
    median_h = float(np.median([report["roughness"].hurst for report in scaling_reports]))
    intercept = float(np.mean(np.log(median_structure) - median_h * np.log(common_lags)))
    fitted = np.exp(intercept + median_h * np.log(common_lags))

    fig, scaling_ax = plt.subplots(figsize=(8.8, 6.2))

    for report in scaling_reports:
        roughness = report["roughness"]
        scaling_ax.loglog(
            roughness.lags[:common_len],
            roughness.structure_function[:common_len],
            color=MID_BLUE,
            linewidth=0.8,
            alpha=0.18,
        )
    scaling_ax.fill_between(common_lags, q25, q75, color=PALE_BLUE, alpha=0.65, label="Interquartile band")
    scaling_ax.loglog(common_lags, median_structure, color=DEEP_BLUE, linewidth=2.2, label="Cross-sectional median")
    scaling_ax.loglog(common_lags, fitted, color=SLATE_GREY, linewidth=1.5, linestyle="--", label=f"Median-slope fit H={median_h:.2f}")
    scaling_ax.set_title("Cross-Sectional Scaling Law", fontsize=16, pad=10)
    scaling_ax.set_xlabel("Lag (RV blocks)")
    scaling_ax.set_ylabel(r"$\mathrm{E}\!\left[\left|\log \mathrm{RV}_{t+\Delta}-\log \mathrm{RV}_t\right|\right]$")
    scaling_ax.grid(alpha=0.35)
    scaling_ax.legend(frameon=False, loc="lower right")
    summary = (
        f"{len(reports)} successful estimates\n"
        f"median H = {median_h:.2f}\n"
        f"lags used = {common_len}"
    )
    scaling_ax.text(
        0.98,
        0.98,
        summary,
        transform=scaling_ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "#f5f7fa", "edgecolor": LIGHT_GREY, "boxstyle": "round,pad=0.3"},
    )

    fig.suptitle(
        f"Empirical Roughness Summary Across Top {args.top_n} Stocks",
        fontsize=18,
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=240)
    plt.close(fig)

    print(f"Saved figure: {OUTPUT_FIGURE}")
    print(f"Successful estimates: {len(reports)}" + (f", failures: {len(failures)}" if failures else ""))
    print(f"Updated cache: {args.cache_path}")


if __name__ == "__main__":
    main()
