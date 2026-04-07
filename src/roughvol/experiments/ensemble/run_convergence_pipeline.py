"""Run the convergence workflow once and render all convergence figures."""

from __future__ import annotations

from roughvol.experiments.convergence.run_rough_vol_convergence import (
    plot_error_panel,
    plot_timing_panel,
    run_rough_bergomi_convergence,
)


def main() -> None:
    results = run_rough_bergomi_convergence()
    print("\nGenerating figures...")
    plot_error_panel(results)
    plot_timing_panel(results)
    print("\nDone. All figures saved to output/convergence.")


if __name__ == "__main__":
    main()
