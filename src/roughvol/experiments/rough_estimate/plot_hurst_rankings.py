"""Rank stocks by estimated Hurst exponent and plot a sorted bar chart.

Run with:
    python -m roughvol.experiments.rough_estimate.plot_hurst_rankings --top-n 50
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roughvol.experiments.rough_estimate._common import build_hurst_reports, parse_ranked_report_args
from roughvol.experiments.rough_estimate.run_empirical_roughness_demo import output_figure_name


def main(argv: list[str] | None = None) -> None:
    args = parse_ranked_report_args("Plot ranked Hurst exponents across top-cap stocks.")
    reports, failures = build_hurst_reports(args)
    if not reports:
        print("No ranking figure was generated because no H estimates succeeded.")
        return

    sorted_reports = sorted(reports, key=lambda report: report["roughness"].hurst)
    tickers = [report["ticker"] for report in sorted_reports]
    hursts = np.array([report["roughness"].hurst for report in sorted_reports], dtype=float)
    colours = ["seagreen" if h < 0.5 else "darkorange" for h in hursts]

    fig, ax = plt.subplots(figsize=(max(12, 0.22 * len(tickers)), 6))
    ax.bar(np.arange(len(tickers)), hursts, color=colours, width=0.82)
    ax.axhline(np.median(hursts), color="black", linestyle="--", linewidth=1.3, label=f"Median H = {np.median(hursts):.2f}")
    ax.axhline(np.quantile(hursts, 0.25), color="gray", linestyle=":", linewidth=1.0, label="25/75 pct")
    ax.axhline(np.quantile(hursts, 0.75), color="gray", linestyle=":", linewidth=1.0)
    ax.set_title(f"Ranked Hurst estimates across top {args.top_n} stocks by market cap")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Estimated H")
    ax.set_xticks(np.arange(len(tickers)))
    ax.set_xticklabels(tickers, rotation=75, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

    output_path = output_figure_name(f"hurst_rankings_top{args.top_n}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    print(f"Saved figure: {output_path}")
    print(f"Successful estimates: {len(reports)}" + (f", failures: {len(failures)}" if failures else ""))
    print(f"Updated cache: {args.cache_path}")


if __name__ == "__main__":
    main()
