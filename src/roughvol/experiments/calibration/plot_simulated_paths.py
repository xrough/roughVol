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
    rh_calibration = report.results.get("RoughHeston")
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

    models_to_plot = [
        ("GBM", gbm_calibration, "steelblue", "navy"),
        ("RoughBergomi", rb_calibration, "lightcoral", "darkred"),
    ]
    if rh_calibration is not None:
        models_to_plot.append(("RoughHeston", rh_calibration, "mediumpurple", "indigo"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulated = {}
        for model_name, calib, _, _ in models_to_plot:
            model = build_model_from_params(model_name, calib.params)
            simulated[model_name] = model.simulate_paths(
                market=report.market_data, sim=sim, rng=make_rng(7 + len(simulated))
            )

    n_panels = len(models_to_plot)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (model_name, calib, colour, mean_colour) in zip(axes, models_to_plot):
        spot_norm = simulated[model_name].spot / report.market_data.spot
        t_grid = simulated[model_name].t
        if model_name == "GBM":
            subtitle = f"σ={calib.params['sigma']:.3f}"
        elif model_name == "RoughBergomi":
            subtitle = f"H={calib.params['hurst']:.3f}, η={calib.params['eta']:.2f}"
        else:
            subtitle = f"H={calib.params['hurst']:.3f}, ν={calib.params['nu']:.2f}"
        for path in spot_norm:
            ax.plot(t_grid, path, color=colour, alpha=0.25, linewidth=0.8)
        ax.plot(t_grid, spot_norm.mean(axis=0), color=mean_colour, linewidth=2.2, label="Mean path")
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"{report.ticker}  –  {model_name}\n({subtitle})", fontsize=11)
        ax.set_xlabel("Time (years)", fontsize=9)
        ax.set_ylabel("S(t) / S(0)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Simulated Spot Paths  ({report.ticker})", fontsize=13)
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
