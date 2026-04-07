from __future__ import annotations

import argparse

from roughvol.experiments._paths import output_path
from roughvol.experiments.calibration._short_term_panel import panel_y_limits, render_panel_image
from roughvol.experiments.calibration.run_short_term_calibration_demo import (
    DEFAULT_SHORT_TERM_TICKERS,
    DEFAULT_SNAPSHOT_DIR,
    load_snapshot_series,
    normalize_tickers,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a short-term calibration animation from cached daily snapshots.",
    )
    parser.add_argument("tickers", nargs="*", default=DEFAULT_SHORT_TERM_TICKERS)
    parser.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def build_animation(
    *,
    tickers: list[str] | None = None,
    snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
    date_from: str | None = None,
    date_to: str | None = None,
    fps: int = 5,
    fmt: str = "gif",
    out: str | None = None,
) -> str:
    snapshots = load_snapshot_series(snapshot_dir, date_from=date_from, date_to=date_to)
    if not snapshots:
        raise RuntimeError("No compatible short-term snapshots were found for the requested date range.")

    basket = normalize_tickers(tickers)
    y_limits = panel_y_limits(snapshots)
    frames = [render_panel_image(snapshot, tickers=basket, y_limits=y_limits) for snapshot in snapshots]
    out = out or output_path("calibration", f"short_term_calibration_panel_animation.{fmt}")

    if fmt == "gif":
        duration_ms = max(1, int(1000 / max(fps, 1)))
        frames[0].save(
            out,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        return out

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("MP4 output requires imageio to be installed.") from exc

    iio.imwrite(out, [frame for frame in frames], fps=max(fps, 1))
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out = build_animation(
        tickers=args.tickers,
        snapshot_dir=args.snapshot_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        fps=args.fps,
        fmt=args.format,
        out=args.out,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
