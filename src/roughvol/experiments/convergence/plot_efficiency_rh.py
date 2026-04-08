"""Plot: accuracy vs wall-clock time for rough Heston schemes at H=0.01, 0.05, 0.1.

Three panels (one per Hurst value).  x-axis = absolute error vs CF benchmark,
y-axis = wall-clock time (seconds).  Each scheme traces a curve as n_steps
increases; points toward the bottom-left are better.

Run:
    python -m roughvol.experiments.convergence.plot_efficiency_rh
    python -m roughvol.experiments.convergence.plot_efficiency_rh --quick
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roughvol.experiments._paths import output_path
from roughvol.experiments.convergence.run_rough_vol import run_rough_heston_convergence
from roughvol.experiments.rough_estimate._style import (
    DEEP_BLUE,
    LIGHT_GREY,
    MID_BLUE,
    PALE_BLUE,
    SLATE_GREY,
    configure_libertine_style,
)

HURST_VALUES = [0.01, 0.05, 0.1]

_STYLE = {
    "volterra-euler": (DEEP_BLUE,  "o-",  "Volterra Euler"),
    "markovian-lift": (SLATE_GREY, "D--", "Markovian lift (NNLS)"),
    "bayer-breneis":  (MID_BLUE,   "s:",  "Bayer-Breneis (BB+GH5)"),
}


def plot_efficiency(
    results_by_hurst: dict[float, dict],
    out: str | None = None,
) -> None:
    configure_libertine_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, H in zip(axes, HURST_VALUES):
        data = results_by_hurst[H]

        for scheme, sdata in data["schemes"].items():
            color, fmt, label = _STYLE[scheme]
            errors = np.array(sdata["errors"], dtype=float)
            times  = np.array(sdata["times"],  dtype=float)

            mask = errors > 0
            if not mask.any():
                continue

            ax.semilogy(errors[mask], times[mask], fmt, color=color,
                        label=label, linewidth=1.8, markersize=6)

            steps = np.array(sdata["steps"])[mask]
            for err, t, n in zip(errors[mask], times[mask], steps):
                ax.annotate(
                    str(n), (err, t),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=6, color=color, alpha=0.75,
                )

        ax.set_xlabel("Absolute error vs CF benchmark", fontsize=12)
        ax.set_ylabel("Wall-clock time (s)", fontsize=12)
        ax.set_title(f"H = {H}", fontsize=14)
        ax.grid(alpha=0.35)
        ax.invert_xaxis()
        ax.legend(frameon=False, loc="upper left", fontsize=10)

    fig.tight_layout()
    out = out or output_path("convergence", "rough_heston_efficiency.png")
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-cv", dest="use_cv", action="store_false")
    args = ap.parse_args()

    results_by_hurst: dict[float, dict] = {}
    for H in HURST_VALUES:
        print(f"\n{'='*60}")
        print(f"Running H = {H}")
        results_by_hurst[H] = run_rough_heston_convergence(
            quick=args.quick,
            use_cv=args.use_cv,
            hurst=H,
        )

    plot_efficiency(results_by_hurst)


if __name__ == "__main__":
    main()
