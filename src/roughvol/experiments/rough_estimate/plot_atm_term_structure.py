from roughvol.experiments._paths import output_path
from roughvol.experiments.rough_estimate._common import build_reports, parse_report_args
from roughvol.experiments.rough_estimate.run_empirical_roughness_demo import plot_atm_term_structure_reports


def main() -> None:
    args = parse_report_args("Plot ATM term structures for empirical roughness estimation.")
    reports = build_reports(args)
    if not reports:
        return
    plot_atm_term_structure_reports(reports, output_path("rough_estimate", "empirical_roughness_atm_term_structure.png"))


if __name__ == "__main__":
    main()
