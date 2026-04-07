"""Plot sector-wise distributions of estimated Hurst exponents.

Run with:
    python -m roughvol.experiments.rough_estimate.plot_hurst_sector --top-n 100
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roughvol.experiments.rough_estimate._common import build_hurst_reports, get_ticker_sector, parse_ranked_report_args
from roughvol.experiments.rough_estimate.run_empirical_roughness_demo import output_figure_name


def main(argv: list[str] | None = None) -> None:
    args = parse_ranked_report_args(
        "Plot sector-wise Hurst distributions across top-cap stocks.",
        default_top_n=100,
    )
    reports, failures = build_hurst_reports(args)
    if not reports:
        print("No sector figure was generated because no H estimates succeeded.")
        return

    sector_to_values: dict[str, list[float]] = {}
    for report in reports:
        sector = get_ticker_sector(report["ticker"])
        sector_to_values.setdefault(sector, []).append(report["roughness"].hurst)

    filtered = {sector: values for sector, values in sector_to_values.items() if len(values) >= 2}
    if not filtered:
        print("No sector figure was generated because there were not enough multi-name sectors.")
        return

    ordered = sorted(filtered.items(), key=lambda item: np.median(item[1]))
    labels = [sector for sector, _ in ordered]
    values = [np.array(vals, dtype=float) for _, vals in ordered]

    fig, ax = plt.subplots(figsize=(max(12, 1.1 * len(labels)), 6.5))
    violin = ax.violinplot(values, showmeans=False, showmedians=True, widths=0.9)
    for body in violin["bodies"]:
        body.set_facecolor("lightsteelblue")
        body.set_edgecolor("navy")
        body.set_alpha(0.6)
    violin["cmedians"].set_color("black")

    rng = np.random.default_rng(7)
    for idx, vals in enumerate(values, start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), idx) + jitter, vals, color="navy", s=14, alpha=0.65)

    ax.set_title(f"Sector-wise distributions of estimated H across top {args.top_n} stocks")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Estimated H")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.2)

    output_path = output_figure_name(f"hurst_by_sector_top{args.top_n}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    print(f"Saved figure: {output_path}")
    print(f"Successful estimates: {len(reports)}" + (f", failures: {len(failures)}" if failures else ""))
    print(f"Updated cache: {args.cache_path}")


if __name__ == "__main__":
    main()
