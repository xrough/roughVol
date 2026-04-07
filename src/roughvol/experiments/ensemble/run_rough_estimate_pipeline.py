"""Run the empirical roughness workflow once and render the default figure set."""

from __future__ import annotations

import sys

from roughvol.experiments.rough_estimate.run_empirical_roughness_demo import (
    build_hurst_histogram_reports,
    load_estimate_cache,
    load_or_build_empirical_roughness_report,
    output_figure_name,
    parse_args,
    plot_atm_term_structure_reports,
    plot_hurst_histogram,
    plot_realized_vol_reports,
    plot_roughness_regression_reports,
    plot_simulation_reports,
    save_estimate_cache,
)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    tickers = [ticker.upper() for ticker in args.tickers if ticker.strip()]
    rv_block_size = args.rv_block_size_alias or args.rv_block_size

    reports: list[dict] = []
    cached_reports: dict[str, dict] = {}
    cache_payload = load_estimate_cache(args.cache_path)
    cache_entries = cache_payload["entries"]
    cache_dirty = False

    for ticker_symbol in tickers:
        print("=" * 70)
        print(f"Empirical roughness demo for {ticker_symbol}")
        print("=" * 70)
        try:
            report, rebuilt = load_or_build_empirical_roughness_report(
                ticker_symbol,
                interval=args.interval,
                period=args.period,
                rv_block_size=rv_block_size,
                cache_entries=cache_entries,
                refresh_cache=args.refresh_cache,
            )
        except Exception as exc:
            print(f"Failed for {ticker_symbol}: {exc}")
            continue

        reports.append(report)
        cached_reports[ticker_symbol] = report
        if rebuilt:
            cache_dirty = True
        else:
            print("Using cached full report.")

    if not reports:
        print("No per-ticker figures were generated because all requested ticker runs failed.")

    output_paths: dict[str, str] = {}
    if reports:
        output_paths = {
            "realized_vol": output_figure_name("realized_vol"),
            "roughness_regression": output_figure_name("roughness_regression"),
            "atm_term_structure": output_figure_name("atm_term_structure"),
            "simulation": output_figure_name("simulation"),
        }
        plot_realized_vol_reports(reports, output_paths["realized_vol"])
        plot_roughness_regression_reports(reports, output_paths["roughness_regression"])
        plot_atm_term_structure_reports(reports, output_paths["atm_term_structure"])
        plot_simulation_reports(reports, output_paths["simulation"])

    if args.hurst_hist_top_n > 0:
        print("=" * 70)
        print(f"Cross-sectional Hurst histogram for top {args.hurst_hist_top_n} stocks")
        print("=" * 70)
        hist_reports, failures = build_hurst_histogram_reports(
            top_n=args.hurst_hist_top_n,
            interval=args.interval,
            period=args.period,
            rv_block_size=rv_block_size,
            cached_reports=cached_reports,
            cache_entries=cache_entries,
            cache_payload=cache_payload,
            refresh_cache=args.refresh_cache,
        )
        if hist_reports:
            output_paths[f"hurst_histogram_top{args.hurst_hist_top_n}"] = output_figure_name(
                f"hurst_histogram_top{args.hurst_hist_top_n}",
            )
            plot_hurst_histogram(
                hist_reports,
                output_paths[f"hurst_histogram_top{args.hurst_hist_top_n}"],
                top_n=args.hurst_hist_top_n,
            )
            print(
                f"Histogram sample: {len(hist_reports)} successful estimates"
                + (f", {len(failures)} failures" if failures else ""),
            )
            cache_dirty = cache_dirty or bool(failures) or any(not report.get("from_cache", False) for report in hist_reports)
        else:
            print("Histogram figure was not generated because no H estimates succeeded.")

    if cache_dirty:
        save_estimate_cache(args.cache_path, cache_payload)

    if output_paths:
        print("Saved figures:")
        for path in output_paths.values():
            print(f"  {path}")
        if cache_dirty:
            print(f"Updated cache: {args.cache_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
