from __future__ import annotations

from roughvol.experiments.calibration._short_term_panel import render_short_term_panel
from roughvol.experiments.calibration.run_short_term_calibration_demo import (
    build_arg_parser,
    collect_short_term_snapshot,
)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser(
        "Plot a 3x3 short-term calibration panel using cached daily snapshots when available.",
    ).parse_args(argv)
    snapshot = collect_short_term_snapshot(
        args.tickers,
        cache_path=args.cache_path,
        snapshot_dir=args.snapshot_dir,
        refresh_cache=args.refresh_cache,
        snapshot_date=args.snapshot_date,
    )
    out = render_short_term_panel(snapshot, tickers=args.tickers)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
