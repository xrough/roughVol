from __future__ import annotations

import argparse

import yfinance as yf

from roughvol.experiments.rough_estimate.run_empirical_roughness_demo import (
    DEFAULT_CACHE_PATH,
    DEFAULT_PRICE_HISTORY_INTERVAL,
    DEFAULT_RV_BLOCK_SIZE,
    LARGE_CAP_CANDIDATE_TICKERS,
    build_empirical_roughness_report,
    build_hurst_histogram_reports,
    cache_entry_from_report,
    cache_key,
    get_market_cap,
    load_or_build_empirical_roughness_report,
    load_estimate_cache,
    rank_large_cap_candidates,
    rank_tickers_by_market_cap,
    save_estimate_cache,
)


def parse_report_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("tickers", nargs="*", default=["SPY"], help="Ticker symbols to analyze.")
    parser.add_argument("--interval", default=DEFAULT_PRICE_HISTORY_INTERVAL)
    parser.add_argument("--period", default=None)
    parser.add_argument("--rv-block-size", type=int, default=DEFAULT_RV_BLOCK_SIZE)
    parser.add_argument("--window", type=int, dest="rv_block_size_alias", default=None)
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def build_reports(args: argparse.Namespace) -> list[dict]:
    rv_block_size = args.rv_block_size_alias or args.rv_block_size
    cache_payload = load_estimate_cache(args.cache_path)
    reports = []
    for ticker_symbol in [ticker.upper() for ticker in args.tickers]:
        print(f"[RoughEstimate] {ticker_symbol}")
        try:
            report, rebuilt = load_or_build_empirical_roughness_report(
                ticker_symbol,
                interval=args.interval,
                period=args.period,
                rv_block_size=rv_block_size,
                cache_entries=cache_payload["entries"],
                refresh_cache=args.refresh_cache,
            )
        except Exception as exc:
            print(f"  failed: {exc}")
            continue
        if not rebuilt:
            print("  using cached full report")
        reports.append(report)
    if reports:
        save_estimate_cache(args.cache_path, cache_payload)
    return reports


def parse_ranked_report_args(description: str, *, allow_explicit_tickers: bool = False, default_top_n: int = 50) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    if allow_explicit_tickers:
        parser.add_argument("tickers", nargs="*", help="Explicit ticker symbols to analyze.")
    parser.add_argument("--top-n", type=int, default=default_top_n)
    parser.add_argument("--interval", default=DEFAULT_PRICE_HISTORY_INTERVAL)
    parser.add_argument("--period", default=None)
    parser.add_argument("--rv-block-size", type=int, default=DEFAULT_RV_BLOCK_SIZE)
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def build_ranked_full_reports(args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    cache_payload = load_estimate_cache(args.cache_path)
    explicit = [ticker.upper() for ticker in getattr(args, "tickers", []) if ticker.strip()]
    if explicit:
        tickers = explicit
    else:
        tickers = rank_large_cap_candidates(
            args.top_n,
            cache_payload=cache_payload,
            refresh_cache=args.refresh_cache,
        )

    reports: list[dict] = []
    failures: list[str] = []
    for idx, ticker_symbol in enumerate(tickers, start=1):
        print(f"[RoughEstimate] {idx}/{len(tickers)} {ticker_symbol}")
        try:
            report, rebuilt = load_or_build_empirical_roughness_report(
                ticker_symbol,
                interval=args.interval,
                period=args.period,
                rv_block_size=args.rv_block_size,
                cache_entries=cache_payload["entries"],
                refresh_cache=args.refresh_cache,
            )
        except Exception as exc:
            print(f"  failed: {exc}")
            failures.append(ticker_symbol)
            continue

        reports.append(report)
        if not rebuilt:
            print("  using cached full report")

    if reports:
        save_estimate_cache(args.cache_path, cache_payload)
    return reports, failures


def build_hurst_reports(args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    cache_payload = load_estimate_cache(args.cache_path)
    reports, failures = build_hurst_histogram_reports(
        top_n=args.top_n,
        interval=args.interval,
        period=args.period,
        rv_block_size=args.rv_block_size,
        cache_entries=cache_payload["entries"],
        cache_payload=cache_payload,
        refresh_cache=args.refresh_cache,
    )
    if reports:
        save_estimate_cache(args.cache_path, cache_payload)
    return reports, failures


def get_ticker_sector(ticker_symbol: str) -> str:
    try:
        info = yf.Ticker(ticker_symbol).info
        sector = info.get("sector") if info else None
        if sector:
            return str(sector)
    except Exception:
        pass
    return "Unknown"
