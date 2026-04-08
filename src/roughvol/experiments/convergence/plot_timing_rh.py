"""Plot: Rough Heston wall-clock time vs n_steps (all three schemes).

Run:
    python -m roughvol.experiments.convergence.plot_timing_rh
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from roughvol.experiments._paths import output_path
from roughvol.experiments.convergence.run_rough_vol import run_rough_heston_convergence


def plot_timing_panel_rh(rh_results: dict, out: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    style_map = {
        "volterra-euler": ("C0", "o-",  "Volterra Euler"),
        "markovian-lift": ("C3", "D--", "Markovian lift"),
        "bayer-breneis":  ("C2", "s:",  "Bayer-Breneis"),
    }
    for scheme, data in rh_results["schemes"].items():
        color, fmt, label = style_map[scheme]
        ax.loglog(data["steps"], data["times"], fmt, color=color, label=label, linewidth=1.8, markersize=7)

    ax.set_title("Rough Heston wall-clock time vs n_steps")
    ax.set_xlabel("n_steps")
    ax.set_ylabel("Wall-clock time (s)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out or output_path("convergence", "rough_heston_timing.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    plot_timing_panel_rh(run_rough_heston_convergence())


if __name__ == "__main__":
    main()
