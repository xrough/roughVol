"""One-figure script: Rough Heston price error vs n_steps (all three schemes)."""

from roughvol.experiments.convergence.run_rough_vol_convergence import (
    plot_error_panel_rh,
    run_rough_heston_convergence,
)


def main() -> None:
    plot_error_panel_rh(run_rough_heston_convergence())


if __name__ == "__main__":
    main()
