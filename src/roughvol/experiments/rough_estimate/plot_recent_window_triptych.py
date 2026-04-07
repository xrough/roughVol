"""Recent-window triptych: returns, local vol, and RV blocks side by side.

Run with:
    python -m roughvol.experiments.rough_estimate.plot_recent_window_triptych SPY AAPL
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from roughvol.analytics.roughness import log_returns_from_close
from roughvol.experiments.rough_estimate._common import build_reports, parse_report_args
from roughvol.experiments.rough_estimate.run_empirical_roughness_demo import (
    REALIZED_VOL_ZOOM_HOURS,
    output_figure_name,
    plot_series_with_session_gaps,
    recent_intraday_zoom_series,
)

OUTPUT_FIGURE = output_figure_name("recent_window_triptych")

def main(argv: list[str] | None = None) -> None:
    _ = argv
    args = parse_report_args("Plot recent-window returns, local vol, and RV blocks.")
    reports = build_reports(args)

    if not reports:
        print("No triptych figure was generated because all ticker runs failed.")
        return

    fig, axes = plt.subplots(len(reports), 3, figsize=(16, 4.3 * len(reports)), squeeze=False)
    for row_axes, report in zip(axes, reports):
        returns_ax, local_vol_ax, rv_ax = row_axes
        close = report["close"]
        returns = log_returns_from_close(close, session_aware=report["intraday_mode"])
        recent_returns = recent_intraday_zoom_series(returns, interval=report["interval"], zoom_hours=REALIZED_VOL_ZOOM_HOURS)
        recent_local_vol = recent_intraday_zoom_series(
            report["local_volatility"],
            interval=report["interval"],
            zoom_hours=REALIZED_VOL_ZOOM_HOURS,
        )
        recent_rv = recent_intraday_zoom_series(
            report["realized_variance_blocks"]["annualized_volatility"],
            interval=report["interval"],
            zoom_hours=REALIZED_VOL_ZOOM_HOURS,
        )

        plot_series_with_session_gaps(returns_ax, recent_returns, color="dimgray", linewidth=0.55, gap=2.0)
        returns_ax.set_title(f"{report['ticker']}\nRecent close-to-close log returns")
        returns_ax.set_ylabel("Return")
        returns_ax.set_xlabel("Latest session (compressed)")
        returns_ax.grid(alpha=0.2)

        plot_series_with_session_gaps(local_vol_ax, recent_local_vol, color="crimson", linewidth=0.5, gap=2.0)
        local_vol_ax.set_title(f"{report['ticker']}\nLocal volatility proxy")
        local_vol_ax.set_ylabel("Volatility")
        local_vol_ax.set_xlabel("Latest session (compressed)")
        local_vol_ax.grid(alpha=0.2)

        plot_series_with_session_gaps(rv_ax, recent_rv, color="navy", linewidth=0.75, gap=2.0)
        rv_ax.set_title(f"{report['ticker']}\nRecent realized-volatility blocks")
        rv_ax.set_ylabel("Volatility")
        rv_ax.set_xlabel("Latest session (compressed)")
        rv_ax.grid(alpha=0.2)

    fig.suptitle("Recent-window triptych: returns, local vol, and realized-volatility blocks", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=240)
    plt.close(fig)
    print(f"Saved figure: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
