from roughvol.experiments._paths import output_path
from roughvol.experiments.rough_estimate._common import build_reports, parse_report_args
from roughvol.experiments.rough_estimate.run_empirical_roughness_demo import plot_simulation_reports


def main() -> None:
    args = parse_report_args("Plot rough-vs-Brownian simulations for empirical roughness estimation.")
    reports = build_reports(args)
    if not reports:
        return
    plot_simulation_reports(reports, output_path("rough_estimate", "empirical_roughness_simulation.png"))


if __name__ == "__main__":
    main()
