"""Plot: Rough Heston scheme comparison — error, Richardson, cross-scheme diagnostic.

Three-panel figure:
  Top    — absolute error vs CF benchmark
  Middle — Richardson successive differences |p(2n) − p(n)|  (convergence-rate proxy)
  Bottom — cross-scheme spread vs MC noise

Run:
    python -m roughvol.experiments.convergence.plot_error_rh          # full run
    python -m roughvol.experiments.convergence.plot_error_rh --quick  # quick pass
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roughvol.experiments._paths import output_path
from roughvol.experiments.convergence.run_rough_vol import run_rough_heston_convergence

_STYLE = {
    "volterra-euler": ("C0", "o-",  "Volterra Euler (O(n²))"),
    "markovian-lift": ("C3", "D--", "Markovian lift (O(N·n))"),
    "bayer-breneis":  ("C2", "s:",  "Bayer-Breneis (order-2 weak)"),
}


def plot_error_panel_rh(rh_results: dict, out: str | None = None) -> None:
    """3-panel convergence figure for the Rough Heston schemes."""
    use_cv = rh_results.get("use_cv", False)
    cv_tag = " +CV" if use_cv else ""

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 11.0), sharex=False)
    ax_err, ax_rich, ax_diag = axes

    # ------------------------------------------------------------------ #
    # Panel 1 — absolute error vs benchmark
    # ------------------------------------------------------------------ #
    for scheme, data in rh_results["schemes"].items():
        color, fmt, label = _STYLE[scheme]
        xs  = np.array(data["steps"],  dtype=float)
        err = np.maximum(np.array(data["errors"],  dtype=float), 1e-9)
        se  = np.array(data["stderrs"], dtype=float)
        ax_err.loglog(xs, err, fmt, color=color, label=label + cv_tag,
                      linewidth=1.8, markersize=7)
        ax_err.fill_between(xs, np.maximum(err - 2*se, 1e-9), err + 2*se,
                            color=color, alpha=0.12)

    ax_err.set_ylabel("Absolute error vs CF benchmark")
    ax_err.set_title(
        f"Rough Heston: price error vs n_steps{cv_tag}\n"
        f"(n_paths VE={rh_results.get('n_paths_ve', '?'):,}, "
        f"fast={rh_results.get('n_paths_fast', '?'):,})"
    )
    ax_err.grid(True, which="both", alpha=0.25)
    ax_err.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------ #
    # Panel 2 — Richardson successive differences |p(2n) − p(n)|
    # ------------------------------------------------------------------ #
    for scheme, rich in rh_results.get("richardson", {}).items():
        if not rich["steps"]:
            continue
        color, fmt, label = _STYLE[scheme]
        xs    = np.array(rich["steps"], dtype=float)
        diffs = np.maximum(np.array(rich["diffs"], dtype=float), 1e-9)
        ax_rich.loglog(xs, diffs, fmt, color=color,
                       label=label, linewidth=1.8, markersize=7)

    ax_rich.set_ylabel("|p(2n) − p(n)|")
    ax_rich.set_title("Richardson successive differences  (convergence-rate proxy)")
    ax_rich.grid(True, which="both", alpha=0.25)
    ax_rich.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------ #
    # Panel 3 — cross-scheme spread vs MC noise
    # ------------------------------------------------------------------ #
    diag       = rh_results["diagnostics"]
    diag_steps = np.array(diag["steps"], dtype=float)

    ax_diag.loglog(
        diag_steps,
        np.maximum(np.array(diag["pairwise_spread"], dtype=float), 1e-9),
        "ko-", linewidth=1.8, markersize=6, label="Max cross-scheme spread",
    )
    ax_diag.loglog(
        diag_steps,
        np.maximum(np.array(diag["max_pairwise_noise"], dtype=float), 1e-9),
        color="0.45", linestyle="--", marker="s", linewidth=1.6, markersize=5,
        label="Max pairwise MC noise",
    )
    ax_diag2 = ax_diag.twinx()
    ax_diag2.semilogx(
        diag_steps,
        np.array(diag["max_pairwise_zscore"], dtype=float),
        color="crimson", linestyle=":", marker="^",
        linewidth=1.6, markersize=6, label="Spread / noise",
    )

    ax_diag.set_xlabel("n_steps")
    ax_diag.set_ylabel("Price spread / MC noise")
    ax_diag.set_title("Cross-scheme agreement vs MC noise")
    ax_diag.grid(True, which="both", alpha=0.25)
    ax_diag2.set_ylabel("Noise-normalised spread")

    h_l, l_l = ax_diag.get_legend_handles_labels()
    h_r, l_r = ax_diag2.get_legend_handles_labels()
    ax_diag.legend(h_l + h_r, l_l + l_r, fontsize=8, loc="best")

    # ------------------------------------------------------------------ #
    fig.suptitle(
        "Benchmark: rough Heston CF via fractional Riccati + Fourier inversion.",
        fontsize=11, y=0.995,
    )
    fig.tight_layout()
    out = out or output_path("convergence", "rough_heston_error.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-cv", dest="use_cv", action="store_false")
    args = ap.parse_args()
    plot_error_panel_rh(run_rough_heston_convergence(quick=args.quick, use_cv=args.use_cv))


if __name__ == "__main__":
    main()
