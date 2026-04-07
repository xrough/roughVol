"""Run the calibration workflow once and render all calibration figures."""

from __future__ import annotations

from roughvol.experiments.calibration.plot_iv_smile import plot_iv_smiles
from roughvol.experiments.calibration.plot_rmse_bars import plot_rmse_bars
from roughvol.experiments.calibration.plot_simulated_paths import plot_simulated_paths
from roughvol.experiments.calibration.plot_surface import plot_vol_surface
from roughvol.experiments.calibration.run_calibration_demo import collect_calibration_results, parse_args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = collect_calibration_results(args.tickers)
    if not results:
        return

    print("\nGenerating figures...")
    print("  Figure 1: IV smile comparison")
    plot_iv_smiles(results)
    print("  Figure 2: RMSE bar chart")
    plot_rmse_bars(results)
    print("  Figure 3: Simulated paths")
    plot_simulated_paths(results)
    print("  Figure 4: Vol surface heatmaps")
    plot_vol_surface(results)
    print("\nDone. All figures saved to output/calibration.")


if __name__ == "__main__":
    main()
