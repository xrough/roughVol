"""Plot: rBergomi absolute price error vs n_steps.

Run:
    python -m roughvol.experiments.convergence.plot_error
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roughvol.experiments._paths import output_path
from roughvol.experiments.convergence.run_rough_vol import (
    RB_PARAMS,
    TEST_STEPS,
    run_rough_bergomi_convergence,
)


def plot_error_panel(rb_results: dict, out: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    steps = np.array(TEST_STEPS, dtype=float)
    style_map = {
        "volterra-midpoint": ("C0", "o-",  "Volterra midpoint (§2.5)"),
        "blp-hybrid":        ("C1", "s--", "BLP hybrid (§2.3)"),
        "exact-gaussian":    ("C2", "^:",  "Exact Gaussian (§2.1)"),
    }

    n_ref = np.array([steps[0], steps[-1]], dtype=float)
    for scheme, data in rb_results["schemes"].items():
        color, fmt, label = style_map[scheme]
        scheme_steps = np.array(data["steps"], dtype=float)
        errors = np.maximum(np.array(data["errors"]), 1e-7)
        ax.loglog(scheme_steps, errors, fmt, color=color, label=label, linewidth=1.8, markersize=7)

    hurst = RB_PARAMS["hurst"]
    midpoint_error = max(rb_results["schemes"]["volterra-midpoint"]["errors"][0], 1e-7)
    ax.loglog(
        n_ref,
        midpoint_error * (n_ref / n_ref[0]) ** (-(hurst + 0.5)),
        "k:", linewidth=1, label=f"O(n^{{-{hurst+0.5:.1f}}}) guide",
    )
    hybrid_error = max(rb_results["schemes"]["blp-hybrid"]["errors"][0], 1e-7)
    ax.loglog(
        n_ref,
        hybrid_error * (n_ref / n_ref[0]) ** (-1.5),
        "k--", linewidth=1, label="O(n^{-1.5}) guide",
    )

    ax.set_title("rBergomi price error vs n_steps")
    ax.set_xlabel("n_steps")
    ax.set_ylabel("Absolute pricing error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out or output_path("convergence", "rough_vol_error.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    plot_error_panel(run_rough_bergomi_convergence())


if __name__ == "__main__":
    main()
