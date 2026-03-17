from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from roughvol.analytics.black_scholes_formula import bs_delta, bs_price, implied_vol
from roughvol.engines.mc import MonteCarloEngine
from roughvol.instruments.vanilla import VanillaOption
from roughvol.models.GBM_model import GBM_Model
from roughvol.models.heston_model import HestonModel
from roughvol.service.calibration import CalibResult
from roughvol.service.toolbox import CalibrationToolbox
from roughvol.types import MarketData, SimConfig, make_rng


@dataclass(frozen=True)
class HedgeBookConfig:
    strike: float
    maturity: float
    is_call: bool = True
    n_realized_paths: int = 48
    n_hedge_steps: int = 12
    realized_seed: int = 7
    hedge_pricer_paths: int = 2_000
    hedge_pricer_seed: int = 11


@dataclass(frozen=True)
class ModelComparisonResult:
    model_name: str
    calibration: CalibResult
    price_rmse: float
    iv_rmse: float
    hedge_pnl_mean: float
    hedge_pnl_std: float
    hedge_pnl_rmse: float


@dataclass(frozen=True)
class ModelComparisonReport:
    reference_model_name: str
    n_surface_quotes: int
    results: list[ModelComparisonResult]


def make_surface_dataset(
    *,
    market: MarketData,
    model_name: str,
    params: dict[str, float],
    strikes: list[float],
    maturities: list[float],
    is_call: bool = True,
    engine_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | bool]] = []
    for maturity in maturities:
        for strike in strikes:
            price = _price_vanilla(
                model_name=model_name,
                params=params,
                spot=market.spot,
                strike=strike,
                maturity=maturity,
                rate=market.rate,
                div=market.div_yield,
                is_call=is_call,
                engine_kwargs=engine_kwargs,
            )
            rows.append(
                {
                    "strike": float(strike),
                    "maturity_years": float(maturity),
                    "is_call": bool(is_call),
                    "market_price": float(price),
                }
            )
    return pd.DataFrame(rows)


def compare_models(
    *,
    market: MarketData,
    surface_df: pd.DataFrame,
    candidate_models: list[str],
    reference_model_name: str,
    reference_params: dict[str, float],
    hedge_book: HedgeBookConfig,
    calibration_engine_kwargs: dict[str, Any] | None = None,
    surface_engine_kwargs: dict[str, Any] | None = None,
) -> ModelComparisonReport:
    toolbox = CalibrationToolbox()
    calibration_engine_kwargs = calibration_engine_kwargs or {}
    surface_engine_kwargs = surface_engine_kwargs or {}

    results: list[ModelComparisonResult] = []
    for model_name in candidate_models:
        calibration = toolbox.calibrate(
            model_name=model_name,
            spot=market.spot,
            options_df=surface_df,
            rate=market.rate,
            div=market.div_yield,
            engine_kwargs=calibration_engine_kwargs,
        )

        predicted_prices = [
            _price_vanilla(
                model_name=model_name,
                params=calibration.params,
                spot=market.spot,
                strike=float(row["strike"]),
                maturity=float(row["maturity_years"]),
                rate=market.rate,
                div=market.div_yield,
                is_call=bool(row["is_call"]),
                engine_kwargs=surface_engine_kwargs,
            )
            for _, row in surface_df.iterrows()
        ]
        market_prices = surface_df["market_price"].to_numpy(dtype=float)
        price_rmse = float(np.sqrt(np.mean((np.asarray(predicted_prices) - market_prices) ** 2)))

        iv_rmse = _surface_iv_rmse(
            spot=market.spot,
            rate=market.rate,
            div=market.div_yield,
            surface_df=surface_df,
            predicted_prices=np.asarray(predicted_prices, dtype=float),
        )

        pnl = _delta_hedge_pnl(
            market=market,
            model_name=model_name,
            model_params=calibration.params,
            reference_model_name=reference_model_name,
            reference_params=reference_params,
            hedge_book=hedge_book,
        )

        results.append(
            ModelComparisonResult(
                model_name=model_name,
                calibration=calibration,
                price_rmse=price_rmse,
                iv_rmse=iv_rmse,
                hedge_pnl_mean=float(np.mean(pnl)),
                hedge_pnl_std=float(np.std(pnl, ddof=1)) if pnl.size > 1 else 0.0,
                hedge_pnl_rmse=float(np.sqrt(np.mean(pnl ** 2))),
            )
        )

    return ModelComparisonReport(
        reference_model_name=reference_model_name,
        n_surface_quotes=int(len(surface_df)),
        results=results,
    )


def _surface_iv_rmse(
    *,
    spot: float,
    rate: float,
    div: float,
    surface_df: pd.DataFrame,
    predicted_prices: np.ndarray,
) -> float:
    errors: list[float] = []
    for (_, row), predicted in zip(surface_df.iterrows(), predicted_prices):
        try:
            market_iv = implied_vol(
                price=float(row["market_price"]),
                spot=spot,
                strike=float(row["strike"]),
                maturity=float(row["maturity_years"]),
                rate=rate,
                div=div,
                is_call=bool(row["is_call"]),
            )
            model_iv = implied_vol(
                price=float(predicted),
                spot=spot,
                strike=float(row["strike"]),
                maturity=float(row["maturity_years"]),
                rate=rate,
                div=div,
                is_call=bool(row["is_call"]),
            )
        except (ValueError, ZeroDivisionError):
            continue
        errors.append(model_iv - market_iv)

    if not errors:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(errors))))


def _delta_hedge_pnl(
    *,
    market: MarketData,
    model_name: str,
    model_params: dict[str, float],
    reference_model_name: str,
    reference_params: dict[str, float],
    hedge_book: HedgeBookConfig,
) -> np.ndarray:
    option = VanillaOption(
        strike=hedge_book.strike,
        maturity=hedge_book.maturity,
        is_call=hedge_book.is_call,
    )

    realized_paths = _simulate_realized_spots(
        market=market,
        model_name=reference_model_name,
        params=reference_params,
        n_paths=hedge_book.n_realized_paths,
        n_steps=hedge_book.n_hedge_steps,
        seed=hedge_book.realized_seed,
        maturity=hedge_book.maturity,
    )

    premium = _price_vanilla(
        model_name=reference_model_name,
        params=reference_params,
        spot=market.spot,
        strike=option.strike,
        maturity=option.maturity,
        rate=market.rate,
        div=market.div_yield,
        is_call=option.is_call,
        engine_kwargs={
            "n_paths": max(hedge_book.hedge_pricer_paths, 4_000),
            "n_steps": max(hedge_book.n_hedge_steps, 16),
            "seed": hedge_book.hedge_pricer_seed,
            "antithetic": True,
        },
    )

    t = np.linspace(0.0, hedge_book.maturity, hedge_book.n_hedge_steps + 1)
    dt = np.diff(t)

    pnl = np.empty(realized_paths.shape[0], dtype=float)
    for i, path in enumerate(realized_paths):
        delta = _model_delta(
            model_name=model_name,
            params=model_params,
            spot=float(path[0]),
            strike=option.strike,
            maturity=option.maturity,
            rate=market.rate,
            div=market.div_yield,
            is_call=option.is_call,
            n_steps_remaining=hedge_book.n_hedge_steps,
            hedge_pricer_paths=hedge_book.hedge_pricer_paths,
            hedge_pricer_seed=hedge_book.hedge_pricer_seed,
        )
        cash = premium - delta * float(path[0])

        for step, dt_step in enumerate(dt):
            cash *= float(np.exp(market.rate * dt_step))
            if step == len(dt) - 1:
                break
            tau = float(option.maturity - t[step + 1])
            next_delta = _model_delta(
                model_name=model_name,
                params=model_params,
                spot=float(path[step + 1]),
                strike=option.strike,
                maturity=tau,
                rate=market.rate,
                div=market.div_yield,
                is_call=option.is_call,
                n_steps_remaining=max(hedge_book.n_hedge_steps - (step + 1), 1),
                hedge_pricer_paths=hedge_book.hedge_pricer_paths,
                hedge_pricer_seed=hedge_book.hedge_pricer_seed + step + 1,
            )
            cash -= (next_delta - delta) * float(path[step + 1])
            delta = next_delta

        payoff = float(np.maximum(path[-1] - option.strike, 0.0)) if option.is_call else float(np.maximum(option.strike - path[-1], 0.0))
        pnl[i] = cash + delta * float(path[-1]) - payoff

    return pnl


def _model_delta(
    *,
    model_name: str,
    params: dict[str, float],
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div: float,
    is_call: bool,
    n_steps_remaining: int,
    hedge_pricer_paths: int,
    hedge_pricer_seed: int,
) -> float:
    if maturity <= 0.0:
        if is_call:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0

    if model_name in {"BS", "GBM_MC"}:
        sigma = float(params["sigma_atm"] if model_name == "BS" else params["sigma"])
        return bs_delta(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            div=div,
            vol=sigma,
            is_call=is_call,
        )

    bump = max(1e-2, 1e-3 * spot)
    up = _price_vanilla(
        model_name=model_name,
        params=params,
        spot=spot + bump,
        strike=strike,
        maturity=maturity,
        rate=rate,
        div=div,
        is_call=is_call,
        engine_kwargs={
            "n_paths": hedge_pricer_paths,
            "n_steps": max(n_steps_remaining, 8),
            "seed": hedge_pricer_seed,
            "antithetic": True,
        },
    )
    dn = _price_vanilla(
        model_name=model_name,
        params=params,
        spot=max(spot - bump, 1e-6),
        strike=strike,
        maturity=maturity,
        rate=rate,
        div=div,
        is_call=is_call,
        engine_kwargs={
            "n_paths": hedge_pricer_paths,
            "n_steps": max(n_steps_remaining, 8),
            "seed": hedge_pricer_seed,
            "antithetic": True,
        },
    )
    return float((up - dn) / (2.0 * bump))


def _simulate_realized_spots(
    *,
    market: MarketData,
    model_name: str,
    params: dict[str, float],
    n_paths: int,
    n_steps: int,
    seed: int,
    maturity: float,
) -> np.ndarray:
    model = _make_path_model(model_name=model_name, params=params)
    sim = SimConfig(
        n_paths=n_paths,
        maturity=maturity,
        n_steps=n_steps,
        seed=seed,
        antithetic=False,
    )
    paths = model.simulate_paths(
        market=market,
        sim=sim,
        rng=make_rng(seed),
    )
    return np.asarray(paths.spot, dtype=float)


def _price_vanilla(
    *,
    model_name: str,
    params: dict[str, float],
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div: float,
    is_call: bool,
    engine_kwargs: dict[str, Any] | None = None,
) -> float:
    if model_name == "BS":
        return bs_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            div=div,
            vol=float(params["sigma_atm"]),
            is_call=is_call,
        )

    if model_name == "GBM_MC":
        return bs_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            div=div,
            vol=float(params["sigma"]),
            is_call=is_call,
        )

    instrument = VanillaOption(strike=strike, maturity=maturity, is_call=is_call)
    market = MarketData(spot=spot, rate=rate, div_yield=div)
    engine = MonteCarloEngine(
        n_paths=int((engine_kwargs or {}).get("n_paths", 6_000)),
        n_steps=int((engine_kwargs or {}).get("n_steps", 24)),
        seed=int((engine_kwargs or {}).get("seed", 17)),
        antithetic=bool((engine_kwargs or {}).get("antithetic", True)),
    )
    model = _make_path_model(model_name=model_name, params=params)
    return float(engine.price(model=model, instrument=instrument, market=market).price)


def _make_path_model(*, model_name: str, params: dict[str, float]) -> object:
    if model_name == "GBM_MC":
        return GBM_Model(sigma=float(params["sigma"]))
    if model_name == "HESTON":
        return HestonModel(
            kappa=float(params["kappa"]),
            theta=float(params["theta"]),
            xi=float(params["xi"]),
            rho=float(params["rho"]),
            v0=float(params["v0"]),
        )
    raise ValueError(f"Model {model_name!r} does not define path dynamics")
