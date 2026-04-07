from __future__ import annotations

from roughvol.experiments._paths import output_path
from roughvol.experiments.rough_estimate._common import build_hurst_reports, parse_ranked_report_args
from roughvol.experiments.rough_estimate.run_empirical_roughness_demo import (
    plot_hurst_histogram,
)


def main() -> None:
    args = parse_ranked_report_args(
        "Plot the cross-sectional histogram of estimated Hurst exponents.",
    )
    reports, failures = build_hurst_reports(args)
    if not reports:
        print("No histogram figure was generated because no H estimates succeeded.")
        return

    figure_path = output_path("rough_estimate", f"empirical_roughness_hurst_histogram_top{args.top_n}.png")
    plot_hurst_histogram(reports, figure_path, top_n=args.top_n)

    print(f"Saved figure: {figure_path}")
    print(f"Successful estimates: {len(reports)}" + (f", failures: {len(failures)}" if failures else ""))
    print(f"Updated cache: {args.cache_path}")


if __name__ == "__main__":
    main()
