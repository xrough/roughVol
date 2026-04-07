from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roughvol.experiments._paths import output_path
from roughvol.experiments.calibration._common import build_results, parse_args, successful_reports
from roughvol.experiments.calibration.run_calibration_demo import (
    TickerCalibrationReport,
    VIZ_ENGINE,
    compute_model_iv_smile,
)


def plot_vol_surface(
    all_results: dict[str, TickerCalibrationReport],
    out: str | None = None,
) -> None:
    reports = successful_reports(all_results)
    if not reports:
        return

    report = reports[0]
    market_data = report.market_data
    surface_df = report.surface_df
    spot = market_data.spot

    moneyness_grid = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    strikes_abs = [m * spot for m in moneyness_grid]

    maturities = sorted(surface_df["maturity_years"].unique())
    if len(maturities) > 6:
        idx = np.round(np.linspace(0, len(maturities) - 1, 6)).astype(int)
        maturities = [maturities[i] for i in idx]

    n_mat = len(maturities)
    n_mon = len(moneyness_grid)
    market_grid = np.full((n_mon, n_mat), np.nan)

    for j, maturity in enumerate(maturities):
        for i, m_mid in enumerate(moneyness_grid):
            sub = surface_df[
                (np.abs(surface_df["maturity_years"] - maturity) < 0.01)
                & (np.abs(surface_df["strike"] / spot - m_mid) < 0.03)
                & (surface_df["is_call"])
            ]
            if not sub.empty:
                market_grid[i, j] = sub["implied_vol"].mean()

    model_grids: dict[str, np.ndarray] = {}
    for model_name in ("GBM", "Heston", "RoughBergomi"):
        calib_result = report.results.get(model_name)
        if calib_result is None:
            model_grids[model_name] = np.full((n_mon, n_mat), np.nan)
            continue

        grid = np.full((n_mon, n_mat), np.nan)
        for j, maturity in enumerate(maturities):
            ivs = compute_model_iv_smile(
                model_name=model_name,
                params=calib_result.params,
                market_data=market_data,
                maturity=maturity,
                strikes=strikes_abs,
                engine_kwargs=VIZ_ENGINE,
            )
            for i, iv in enumerate(ivs):
                if iv is not None:
                    grid[i, j] = iv
        model_grids[model_name] = grid

    all_values = [
        value
        for grid in [market_grid] + list(model_grids.values())
        for value in grid.flatten()
        if not np.isnan(value)
    ]
    vmin = float(np.nanpercentile(all_values, 5)) if all_values else 0.0
    vmax = float(np.nanpercentile(all_values, 95)) if all_values else 1.0

    mat_labels = [f"{maturity:.2f}" for maturity in maturities]
    mon_labels = [f"{int(moneyness * 100)}%" for moneyness in moneyness_grid]
    titles = ["Market", "GBM", "Heston", "Rough Bergomi"]
    grids = [
        market_grid,
        model_grids["GBM"],
        model_grids["Heston"],
        model_grids["RoughBergomi"],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    image = None
    for ax, title, grid in zip(axes_flat, titles, grids):
        image = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn_r",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(range(n_mat))
        ax.set_xticklabels(mat_labels, fontsize=8, rotation=30)
        ax.set_yticks(range(n_mon))
        ax.set_yticklabels(mon_labels, fontsize=8)
        ax.set_xlabel("Maturity (yr)", fontsize=9)
        ax.set_ylabel("Moneyness", fontsize=9)
        ax.set_title(title, fontsize=11)

        for i in range(n_mon):
            for j in range(n_mat):
                value = grid[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.2%}", ha="center", va="center", fontsize=6.5, color="black")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes_flat, orientation="vertical", fraction=0.02, pad=0.04)
        cbar.set_label("Implied Volatility", fontsize=9)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.suptitle(
        f"Implied Volatility Surface  ({report.ticker})\nMarket vs Calibrated Models",
        fontsize=13,
    )
    fig.tight_layout()
    out = out or output_path("calibration", "calibration_demo_surface.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    args = parse_args("Plot market and model-implied volatility surfaces.")
    results = build_results(args)
    if not results:
        return
    plot_vol_surface(results)


if __name__ == "__main__":
    main()
