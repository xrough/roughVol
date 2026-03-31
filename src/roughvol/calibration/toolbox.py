from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from roughvol.calibration.calibration import (
    BSCalibrator,
    CalibResult,
    make_gbm_calibrator,
    make_heston_calibrator,
    make_rough_bergomi_calibrator,
)


MODEL_TYPE_NAMES: dict[int, str] = {
    0: "BS",
    1: "GBM_MC",
    2: "HESTON",
    3: "ROUGH_BERGOMI",
}


@dataclass(frozen=True)
class CalibrationWindowResult:
    calibration: CalibResult
    recalibrated: bool
    calibrated_at_ms: int
    next_update_due_ms: int
    window_start_ms: int
    window_end_ms: int
    quotes_in_window: int
    quotes_total: int


@dataclass(frozen=True)
class _CacheKey:
    asset_id: str
    model_name: str
    calibration_window_ms: int
    update_interval_ms: int
    n_paths: int
    n_steps: int
    seed: int
    antithetic: bool


class CalibrationToolbox:
    """Stateful calibration cache for fixed-window proto updates."""

    def __init__(self) -> None:
        self._cache: dict[_CacheKey, CalibrationWindowResult] = {}

    def calibrate(
        self,
        *,
        model_name: str,
        spot: float,
        options_df: pd.DataFrame,
        rate: float = 0.05,
        div: float = 0.0,
        x0: list[float] | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> CalibResult:
        x0 = x0 or []
        if model_name == "BS":
            return BSCalibrator().calibrate(spot, options_df, rate, div)

        if model_name == "GBM_MC":
            cal = make_gbm_calibrator(
                x0_sigma=x0[0] if x0 else 0.20,
                engine_kwargs=engine_kwargs,
            )
            return cal.calibrate(spot, options_df, rate, div)

        if model_name == "HESTON":
            cal = make_heston_calibrator(
                x0_sigma=x0[0] if x0 else 0.20,
                engine_kwargs=engine_kwargs,
            )
            return cal.calibrate(spot, options_df, rate, div)

        if model_name == "ROUGH_BERGOMI":
            cal = make_rough_bergomi_calibrator(
                x0_sigma=x0[0] if x0 else 0.20,
                x0=x0 if x0 else None,
                engine_kwargs=engine_kwargs,
            )
            return cal.calibrate(spot, options_df, rate, div)

        raise ValueError(f"Unsupported model_name={model_name!r}")

    def calibrate_windowed(
        self,
        *,
        asset_id: str,
        model_name: str,
        spot: float,
        timed_quotes_df: pd.DataFrame,
        rate: float = 0.05,
        div: float = 0.0,
        x0: list[float] | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        as_of_ms: int | None = None,
        calibration_window_ms: int,
        update_interval_ms: int,
        force_refresh: bool = False,
    ) -> CalibrationWindowResult:
        if calibration_window_ms <= 0:
            raise ValueError("calibration_window_ms must be positive")
        if update_interval_ms <= 0:
            raise ValueError("update_interval_ms must be positive")
        if timed_quotes_df.empty:
            raise ValueError("timed_quotes_df must contain at least one quote")
        if "observed_at_ms" not in timed_quotes_df.columns:
            raise ValueError("timed_quotes_df must contain observed_at_ms")

        quotes_total = int(len(timed_quotes_df))
        window_end_ms = int(as_of_ms) if as_of_ms is not None and as_of_ms > 0 else int(
            timed_quotes_df["observed_at_ms"].max()
        )
        window_start_ms = int(window_end_ms - calibration_window_ms)

        window_mask = (
            (timed_quotes_df["observed_at_ms"] >= window_start_ms)
            & (timed_quotes_df["observed_at_ms"] <= window_end_ms)
        )
        window_df = timed_quotes_df.loc[window_mask].copy()
        quotes_in_window = int(len(window_df))

        if quotes_in_window == 0:
            raise ValueError("No quotes fall inside the requested calibration window")

        options_df = window_df[
            ["strike", "maturity_years", "is_call", "market_price"]
        ].reset_index(drop=True)

        engine_kwargs = dict(engine_kwargs or {})
        key = _CacheKey(
            asset_id=asset_id,
            model_name=model_name,
            calibration_window_ms=int(calibration_window_ms),
            update_interval_ms=int(update_interval_ms),
            n_paths=int(engine_kwargs.get("n_paths", 20_000)),
            n_steps=int(engine_kwargs.get("n_steps", 50)),
            seed=int(engine_kwargs.get("seed", 42)),
            antithetic=bool(engine_kwargs.get("antithetic", False)),
        )

        cached = self._cache.get(key)
        if (
            cached is not None
            and not force_refresh
            and window_end_ms < cached.next_update_due_ms
        ):
            return CalibrationWindowResult(
                calibration=cached.calibration,
                recalibrated=False,
                calibrated_at_ms=cached.calibrated_at_ms,
                next_update_due_ms=cached.next_update_due_ms,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                quotes_in_window=quotes_in_window,
                quotes_total=quotes_total,
            )

        calibration = self.calibrate(
            model_name=model_name,
            spot=spot,
            options_df=options_df,
            rate=rate,
            div=div,
            x0=x0,
            engine_kwargs=engine_kwargs,
        )

        result = CalibrationWindowResult(
            calibration=calibration,
            recalibrated=True,
            calibrated_at_ms=window_end_ms,
            next_update_due_ms=window_end_ms + int(update_interval_ms),
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            quotes_in_window=quotes_in_window,
            quotes_total=quotes_total,
        )
        self._cache[key] = result
        return result
