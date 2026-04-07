from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from roughvol.experiments._paths import output_path
from roughvol.experiments.calibration._common import build_results, parse_args, successful_reports
from roughvol.experiments.calibration.run_calibration_demo import (
    TickerCalibrationReport,
    build_model_from_params,
)
from roughvol.types import SimConfig, make_rng


def plot_simulated_paths(
    all_results: dict[str, TickerCalibrationReport],
    out: str | None = None,
) -> None:
    reports = successful_reports(all_results)
    if not reports:
        return

    report = reports[0]
    gbm_calibration = report.results.get("GBM")
    rb_calibration = report.results.get("RoughBergomi")
    if gbm_calibration is None or rb_calibration is None:
        print("  [WARN] Skipping path plot - GBM or RoughBergomi calibration missing")
        return

    horizon = 1.0
    n_paths = 30
    n_steps = 252
    sim = SimConfig(
        n_paths=n_paths,
        maturity=horizon,
        n_steps=n_steps,
        seed=7,
        antithetic=False,
        store_paths=True,
    )

    gbm_model = build_model_from_params("GBM", gbm_calibration.params)
    rb_model = build_model_from_params("RoughBergomi", rb_calibration.params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gbm_paths = gbm_model.simulate_paths(market=report.market_data, sim=sim, rng=make_rng(7))
        rb_paths = rb_model.simulate_paths(market=report.market_data, sim=sim, rng=make_rng(8))

    t_grid = gbm_paths.t
    gbm_spot_norm = gbm_paths.spot / report.market_data.spot
    rb_spot_norm = rb_paths.spot / report.market_data.spot

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    sigma_str = f"σ={gbm_calibration.params['sigma']:.3f}"
    rough_str = f"H={rb_calibration.params['hurst']:.3f}, η={rb_calibration.params['eta']:.2f}"

    panels = [
        (axes[0], gbm_spot_norm, "steelblue", "navy", f"GBM  ({sigma_str})"),
        (axes[1], rb_spot_norm, "lightcoral", "darkred", f"Rough Bergomi  ({rough_str})"),
    ]

    for ax, spot_norm, colour, mean_colour, title in panels:
        for path in spot_norm:
            ax.plot(t_grid, path, color=colour, alpha=0.25, linewidth=0.8)
        ax.plot(t_grid, spot_norm.mean(axis=0), color=mean_colour, linewidth=2.2, label="Mean path")
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"{report.ticker}  -  {title}", fontsize=11)
        ax.set_xlabel("Time (years)", fontsize=9)
        ax.set_ylabel("S(t) / S(0)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Simulated Spot Paths: GBM vs Rough Bergomi  ({report.ticker})", fontsize=13)
    fig.tight_layout()
    out = out or output_path("calibration", "calibration_demo_paths.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    args = parse_args("Plot simulated spot paths using the calibrated models.")
    results = build_results(args)
    if not results:
        return
    plot_simulated_paths(results)


if __name__ == "__main__":
    main()
