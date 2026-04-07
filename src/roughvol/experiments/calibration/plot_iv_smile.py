from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roughvol.experiments._paths import output_path
from roughvol.experiments.calibration._common import build_results, parse_args, successful_reports
from roughvol.experiments.calibration.run_calibration_demo import (
    MODEL_COLOURS,
    MODEL_LABELS,
    MODEL_LINESTYLES,
    TickerCalibrationReport,
    compute_model_iv_smile,
    VIZ_ENGINE,
)
from roughvol.types import MarketData


def plot_iv_smiles(
    all_results: dict[str, TickerCalibrationReport],
    out: str | None = None,
) -> None:
    reports = successful_reports(all_results)
    if not reports:
        return

    n = len(reports)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, report in enumerate(reports):
        ax = axes_flat[idx]
        market_data: MarketData = report.market_data
        calib_df = report.calib_df
        spot = market_data.spot

        dominant_expiry = calib_df.groupby("expiry_str").size().idxmax()
        exp_df = calib_df[calib_df["expiry_str"] == dominant_expiry]
        maturity = float(exp_df["maturity_years"].iloc[0])

        moneyness_market = exp_df["strike"].values / spot
        iv_market = exp_df["implied_vol"].values
        calls = exp_df["is_call"].values
        ax.scatter(
            moneyness_market[calls],
            iv_market[calls],
            color="black",
            marker="o",
            s=30,
            label="Market (call)",
            zorder=5,
        )
        ax.scatter(
            moneyness_market[~calls],
            iv_market[~calls],
            color="dimgray",
            marker="^",
            s=30,
            label="Market (put)",
            zorder=5,
        )

        fine_moneyness = np.linspace(0.80, 1.20, 10)
        fine_strikes = (fine_moneyness * spot).tolist()

        for model_name in ("GBM", "Heston", "RoughBergomi"):
            calib_result = report.results.get(model_name)
            if calib_result is None:
                continue
            ivs = compute_model_iv_smile(
                model_name=model_name,
                params=calib_result.params,
                market_data=market_data,
                maturity=maturity,
                strikes=fine_strikes,
                engine_kwargs=VIZ_ENGINE,
            )
            valid_x = [fine_moneyness[i] for i, value in enumerate(ivs) if value is not None]
            valid_y = [value for value in ivs if value is not None]
            if valid_x:
                ax.plot(
                    valid_x,
                    valid_y,
                    color=MODEL_COLOURS[model_name],
                    linestyle=MODEL_LINESTYLES[model_name],
                    linewidth=1.8,
                    label=MODEL_LABELS[model_name],
                )

        ax.set_title(f"{report.ticker}  T={maturity:.2f}yr  ({dominant_expiry})", fontsize=11)
        ax.set_xlabel("Moneyness  (K / S)", fontsize=9)
        ax.set_ylabel("Implied Volatility", fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("IV Smile: Market vs Calibrated Models", fontsize=14, y=1.01)
    fig.tight_layout()
    out = out or output_path("calibration", "calibration_demo_iv_smile.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


def main() -> None:
    args = parse_args("Plot market IV smiles against calibrated model smiles.")
    results = build_results(args)
    if not results:
        return
    plot_iv_smiles(results)


if __name__ == "__main__":
    main()
