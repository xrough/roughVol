"""
Calibration module for the RoughPricingService.

Ported from Effective_Engine/MVP/python/calibration/model_calibrator.py
and adapted to the roughvol package API (GBM_Model, VanillaOption with is_call).

Three calibrators with a uniform interface:

  BSCalibrator       — closed-form BS implied vol, ~0 ms
  MCCalibrator       — generic MC calibrator via MonteCarloEngine
  make_gbm_calibrator    — factory for 1-param GBM calibrator
  make_heston_calibrator — factory for 5-param Heston calibrator
  make_rough_bergomi_calibrator — factory for 4-param rough Bergomi calibrator

All calibrators accept a pandas DataFrame with columns:
  strike, maturity_years, is_call, market_price

and return a CalibResult.

Calibration is offline-only. Never call from the live event loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from roughvol.analytics.black_scholes_formula import implied_vol, bs_price
from roughvol.engines.mc import MonteCarloEngine
from roughvol.instruments.vanilla import VanillaOption
from roughvol.types import MarketData


# ── Result container ───────────────────────────────────────────────────────

@dataclass
class CalibResult:
    """Uniform output for all calibrators."""
    model_name: str
    params: dict[str, float]
    mse: float
    per_option_ivols: list[float] = field(default_factory=list)
    elapsed_s: float = 0.0

    def __str__(self) -> str:
        param_str = "  ".join(f"{k}={v:.4f}" for k, v in self.params.items())
        return (f"{self.model_name:<12} │ {param_str:<40} │ "
                f"MSE={self.mse:.3e}  │ {self.elapsed_s:.2f}s")


# ── Black-Scholes calibrator ───────────────────────────────────────────────

class BSCalibrator:
    """Calibrate BS implied vol per option; report the ATM-weighted average."""

    def calibrate(
        self,
        spot: float,
        options_df: pd.DataFrame,
        rate: float = 0.05,
        div: float = 0.0,
    ) -> CalibResult:
        t0 = time.perf_counter()

        ivols: list[float] = []
        failed = 0

        for _, row in options_df.iterrows():
            try:
                iv = implied_vol(
                    price=float(row["market_price"]),
                    spot=spot,
                    strike=float(row["strike"]),
                    maturity=float(row["maturity_years"]),
                    rate=rate,
                    div=div,
                    is_call=bool(row["is_call"]),
                )
                ivols.append(iv)
            except (ValueError, ZeroDivisionError):
                failed += 1

        if not ivols:
            raise RuntimeError(
                "BSCalibrator: could not compute any implied vols "
                "— check option prices vs no-arb bounds"
            )

        if failed:
            print(f"[BS] Warning: {failed} option(s) out of no-arb bounds, skipped")

        strikes = options_df["strike"].values
        weights = 1.0 / (np.abs(strikes - spot) + 1e-4)
        weights = weights[: len(ivols)] / weights[: len(ivols)].sum()
        sigma_atm = float(np.dot(weights, ivols))

        mse = _compute_bs_mse(sigma_atm, spot, options_df, rate, div)
        elapsed = time.perf_counter() - t0

        return CalibResult(
            model_name="BS",
            params={"sigma_atm": sigma_atm},
            mse=mse,
            per_option_ivols=ivols,
            elapsed_s=elapsed,
        )


# ── Generic MC calibrator ──────────────────────────────────────────────────

class MCCalibrator:
    """Calibrate any PathModel using scipy + MonteCarloEngine.

    Parameters
    ----------
    model_name    : display name
    model_factory : maps numpy array of raw params → PathModel
    param_names   : list of param names (same order as factory input)
    bounds        : list of (lo, hi) pairs
    x0            : initial parameter values
    engine_kwargs : forwarded to MonteCarloEngine constructor
    """

    def __init__(
        self,
        model_name: str,
        model_factory: Callable[[np.ndarray], object],
        param_names: list[str],
        bounds: list[tuple[float, float]],
        x0: list[float],
        engine_kwargs: dict | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_factory = model_factory
        self.param_names = param_names
        self.bounds = bounds
        self.x0 = x0
        self._engine = MonteCarloEngine(**(engine_kwargs or {}))

    def calibrate(
        self,
        spot: float,
        options_df: pd.DataFrame,
        rate: float = 0.05,
        div: float = 0.0,
    ) -> CalibResult:
        t0 = time.perf_counter()
        market = MarketData(spot=spot, rate=rate, div_yield=div)

        instruments = [
            VanillaOption(
                strike=float(row["strike"]),
                maturity=float(row["maturity_years"]),
                is_call=bool(row["is_call"]),
            )
            for _, row in options_df.iterrows()
        ]
        market_prices = options_df["market_price"].values.astype(float)

        call_count = [0]

        def loss_fn(x: np.ndarray) -> float:
            call_count[0] += 1
            model = self.model_factory(x)
            mc_prices = np.array([
                self._engine.price(model=model, instrument=inst, market=market).price
                for inst in instruments
            ])
            return float(np.mean((mc_prices - market_prices) ** 2))

        print(f"[{self.model_name}] Starting optimisation  "
              f"(params: {self.param_names}, x0: {[f'{v:.4f}' for v in self.x0]})")

        result = minimize(
            loss_fn,
            x0=np.array(self.x0),
            method="L-BFGS-B",
            bounds=self.bounds,
            options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-8},
        )

        best_x = result.x
        best_mse = float(result.fun)
        elapsed = time.perf_counter() - t0

        print(f"[{self.model_name}] Done  iters={call_count[0]}  "
              f"MSE={best_mse:.3e}  elapsed={elapsed:.1f}s")

        return CalibResult(
            model_name=self.model_name,
            params={name: float(val) for name, val in zip(self.param_names, best_x)},
            mse=best_mse,
            elapsed_s=elapsed,
        )


# ── Factory functions ──────────────────────────────────────────────────────

def make_gbm_calibrator(
    x0_sigma: float = 0.20,
    engine_kwargs: dict | None = None,
) -> MCCalibrator:
    """MCCalibrator for GBM_Model (1-param: sigma)."""
    from roughvol.models.GBM_model import GBM_Model

    return MCCalibrator(
        model_name="GBM-MC",
        model_factory=lambda x: GBM_Model(sigma=float(x[0])),
        param_names=["sigma"],
        bounds=[(1e-4, 5.0)],
        x0=[x0_sigma],
        engine_kwargs=engine_kwargs or {"n_paths": 20_000, "n_steps": 50, "seed": 42, "antithetic": True},
    )


def make_heston_calibrator(
    x0_sigma: float = 0.20,
    engine_kwargs: dict | None = None,
) -> MCCalibrator:
    """MCCalibrator for HestonModel (5-params), warm-started from BS sigma."""
    from roughvol.models.heston_model import HestonModel

    v0_guess = x0_sigma ** 2

    return MCCalibrator(
        model_name="Heston",
        model_factory=lambda x: HestonModel(
            kappa=float(x[0]),
            theta=float(x[1]),
            xi=float(x[2]),
            rho=float(x[3]),
            v0=float(x[4]),
        ),
        param_names=["kappa", "theta", "xi", "rho", "v0"],
        bounds=[
            (0.1, 10.0),    # kappa
            (1e-4, 1.0),    # theta
            (0.01, 2.0),    # xi
            (-0.99, 0.99),  # rho
            (1e-4, 1.0),    # v0
        ],
        x0=[2.0, v0_guess, 0.3, -0.5, v0_guess],
        engine_kwargs=engine_kwargs or {"n_paths": 20_000, "n_steps": 50, "seed": 42, "antithetic": True},
    )


def make_rough_bergomi_calibrator(
    x0_sigma: float = 0.20,
    x0: list[float] | None = None,
    engine_kwargs: dict | None = None,
) -> MCCalibrator:
    """MCCalibrator for RoughBergomiModel (4 params)."""
    from roughvol.models.rough_bergomi_model import RoughBergomiModel

    default_x0 = [0.1, 1.5, -0.7, x0_sigma ** 2]
    raw_x0 = list(x0) if x0 else default_x0
    if len(raw_x0) != 4:
        raise ValueError("rough Bergomi warm start x0 must have four values")

    return MCCalibrator(
        model_name="RoughBergomi",
        model_factory=lambda x: RoughBergomiModel(
            hurst=float(x[0]),
            eta=float(x[1]),
            rho=float(x[2]),
            xi0=float(x[3]),
        ),
        param_names=["hurst", "eta", "rho", "xi0"],
        bounds=[
            (1e-3, 0.499),  # hurst
            (1e-3, 5.0),    # eta
            (-0.99, 0.99),  # rho
            (1e-5, 1.0),    # xi0
        ],
        x0=raw_x0,
        engine_kwargs=engine_kwargs or {"n_paths": 8_000, "n_steps": 64, "seed": 42, "antithetic": True},
    )


# ── Helper ─────────────────────────────────────────────────────────────────

def _compute_bs_mse(
    sigma: float,
    spot: float,
    options_df: pd.DataFrame,
    rate: float,
    div: float,
) -> float:
    """MSE of flat-sigma BS repricing vs market prices."""
    errors: list[float] = []
    for _, row in options_df.iterrows():
        model_price = bs_price(
            spot=spot,
            strike=float(row["strike"]),
            maturity=float(row["maturity_years"]),
            rate=rate,
            div=div,
            vol=sigma,
            is_call=bool(row["is_call"]),
        )
        errors.append(model_price - float(row["market_price"]))
    return float(np.mean(np.array(errors) ** 2))
