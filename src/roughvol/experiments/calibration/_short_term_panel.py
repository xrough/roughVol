from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from roughvol.experiments._paths import output_path
from roughvol.experiments.calibration.run_short_term_calibration_demo import (
    DEFAULT_MONEYNESS_GRID,
    DEFAULT_SHORT_TERM_TICKERS,
    MODEL_COLOURS,
    MODEL_LABELS,
    MODEL_LINESTYLES,
    MODEL_NAMES,
    ShortTermSnapshot,
)


def _default_y_limits(snapshot: ShortTermSnapshot) -> tuple[float, float]:
    values: list[float] = []
    for report in snapshot.reports.values():
        if report.status != "ok":
            continue
        if not report.market_smile_df.empty:
            values.extend(report.market_smile_df["implied_vol"].astype(float).tolist())
        for smile in report.model_smiles.values():
            values.extend(float(value) for value in smile if value is not None)

    if not values:
        return (0.05, 1.0)
    lo = max(0.01, float(np.nanpercentile(values, 5)) * 0.9)
    hi = float(np.nanpercentile(values, 95)) * 1.1
    if hi <= lo:
        hi = lo + 0.10
    return lo, hi


def build_legend_handles() -> list[Line2D]:
    handles = [
        Line2D([], [], color="black", marker="o", linestyle="None", markersize=5, label="Market call"),
        Line2D([], [], color="dimgray", marker="^", linestyle="None", markersize=5, label="Market put"),
    ]
    for model_name in MODEL_NAMES:
        handles.append(
            Line2D(
                [],
                [],
                color=MODEL_COLOURS[model_name],
                linestyle=MODEL_LINESTYLES[model_name],
                linewidth=1.8,
                label=MODEL_LABELS[model_name],
            )
        )
    return handles


def _draw_panel(
    snapshot: ShortTermSnapshot,
    *,
    tickers: list[str],
    y_limits: tuple[float, float],
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(3, 3, figsize=(17, 14), squeeze=False)
    axes_flat = axes.flatten()

    for idx, ticker in enumerate(tickers[:9]):
        ax = axes_flat[idx]
        report = snapshot.reports.get(ticker)
        if report is None or report.status != "ok" or report.market_data is None or report.market_smile_df.empty:
            ax.text(
                0.5,
                0.55,
                f"{ticker}\ndata unavailable",
                ha="center",
                va="center",
                fontsize=12,
                color="dimgray",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            continue

        spot = report.market_data.spot
        market_df = report.market_smile_df
        moneyness_market = market_df["strike"].to_numpy(dtype=float) / spot
        iv_market = market_df["implied_vol"].to_numpy(dtype=float)
        calls = market_df["is_call"].to_numpy(dtype=bool)

        ax.scatter(moneyness_market[calls], iv_market[calls], color="black", marker="o", s=18, zorder=5)
        ax.scatter(moneyness_market[~calls], iv_market[~calls], color="dimgray", marker="^", s=18, zorder=5)

        for model_name in MODEL_NAMES:
            smile = report.model_smiles.get(model_name, [])
            if not smile:
                continue
            xs = [DEFAULT_MONEYNESS_GRID[i] for i, value in enumerate(smile) if value is not None]
            ys = [value for value in smile if value is not None]
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                color=MODEL_COLOURS[model_name],
                linestyle=MODEL_LINESTYLES[model_name],
                linewidth=1.8,
            )

        maturity_days = (
            report.selected_maturity * 365.25
            if report.selected_maturity is not None
            else float("nan")
        )
        ax.set_title(
            f"{ticker}  {report.selected_expiry}\nT≈{maturity_days:.1f}d",
            fontsize=11,
        )
        ax.set_xlim(min(DEFAULT_MONEYNESS_GRID), max(DEFAULT_MONEYNESS_GRID))
        ax.set_ylim(*y_limits)
        ax.set_xlabel("Moneyness  (K / S)", fontsize=9)
        ax.set_ylabel("Implied Volatility", fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(tickers[:9]):]:
        ax.set_visible(False)

    fig.legend(
        handles=build_legend_handles(),
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        fontsize=10,
    )
    fig.suptitle(
        f"Short-Term Calibration Panel\nSnapshot: {snapshot.snapshot_date}",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))
    return fig, axes_flat


def render_short_term_panel(
    snapshot: ShortTermSnapshot,
    *,
    tickers: list[str] | None = None,
    out: str | None = None,
    y_limits: tuple[float, float] | None = None,
) -> str:
    tickers = tickers or snapshot.basket or list(DEFAULT_SHORT_TERM_TICKERS)
    y_limits = y_limits or _default_y_limits(snapshot)
    fig, _ = _draw_panel(snapshot, tickers=tickers, y_limits=y_limits)

    out = out or output_path("calibration", "short_term_calibration_panel.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def render_panel_image(
    snapshot: ShortTermSnapshot,
    *,
    tickers: list[str] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> Image.Image:
    tickers = tickers or snapshot.basket or list(DEFAULT_SHORT_TERM_TICKERS)
    y_limits = y_limits or _default_y_limits(snapshot)
    fig, _ = _draw_panel(snapshot, tickers=tickers, y_limits=y_limits)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def panel_y_limits(snapshots: list[ShortTermSnapshot]) -> tuple[float, float]:
    if not snapshots:
        return (0.05, 1.0)
    lows: list[float] = []
    highs: list[float] = []
    for snapshot in snapshots:
        lo, hi = _default_y_limits(snapshot)
        lows.append(lo)
        highs.append(hi)
    return (min(lows), max(highs))
