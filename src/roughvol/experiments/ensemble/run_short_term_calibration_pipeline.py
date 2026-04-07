"""Update the short-term calibration snapshot, panel, and optional animation."""

from __future__ import annotations

import argparse

from roughvol.experiments.calibration.animate_short_term_panel import build_animation
from roughvol.experiments.calibration._short_term_panel import render_short_term_panel
from roughvol.experiments.calibration.run_short_term_calibration_demo import (
    DEFAULT_CACHE_PATH,
    DEFAULT_SHORT_TERM_TICKERS,
    DEFAULT_SNAPSHOT_DIR,
    collect_short_term_snapshot,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the short-term calibration cache and regenerate panel outputs.",
    )
    parser.add_argument("tickers", nargs="*", default=DEFAULT_SHORT_TERM_TICKERS)
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--build-animation", action="store_true")
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--format", choices=("gif", "mp4"), default="gif")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    snapshot = collect_short_term_snapshot(
        args.tickers,
        cache_path=args.cache_path,
        snapshot_dir=args.snapshot_dir,
        refresh_cache=args.refresh_cache,
        snapshot_date=args.snapshot_date,
    )
    panel_out = render_short_term_panel(snapshot, tickers=args.tickers)
    print(f"Panel: {panel_out}")

    if args.build_animation:
        out = build_animation(
            tickers=args.tickers,
            snapshot_dir=args.snapshot_dir,
            date_from=args.date_from,
            date_to=args.date_to,
            fps=args.fps,
            fmt=args.format,
        )
        print(f"Animation: {out}")


if __name__ == "__main__":
    main()
