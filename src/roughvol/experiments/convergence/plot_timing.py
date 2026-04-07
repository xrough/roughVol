from roughvol.experiments.convergence._common import build_results
from roughvol.experiments.convergence.run_rough_vol_convergence import plot_timing_panel


def main() -> None:
    plot_timing_panel(build_results())


if __name__ == "__main__":
    main()
