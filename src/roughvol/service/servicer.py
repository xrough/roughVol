"""
RoughVolServicer — implements RoughPricingService RPC methods.

Translates proto messages → roughvol Python objects → back to proto responses.
Pricing handlers are stateless; calibration window updates keep an in-process cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the generated stubs importable
_GEN_DIR = Path(__file__).resolve().parents[3] / "generated" / "python"
if str(_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(_GEN_DIR))

import grpc
import pandas as pd

import rough_pricing_pb2 as pb2
import rough_pricing_pb2_grpc as pb2_grpc

from roughvol.analytics.black_scholes_formula import bs_price, implied_vol
from roughvol.engines.mc import MonteCarloEngine
from roughvol.models.GBM_model import GBM_Model
from roughvol.models.heston_model import HestonModel
from roughvol.instruments.vanilla import VanillaOption
from roughvol.instruments.asian import AsianArithmeticOption
from roughvol.types import MarketData

from roughvol.service.toolbox import CalibrationToolbox, MODEL_TYPE_NAMES


class RoughVolServicer(pb2_grpc.RoughPricingServiceServicer):
    def __init__(self) -> None:
        self._toolbox = CalibrationToolbox()

    # ── MCPrice ───────────────────────────────────────────────────────────────

    async def MCPrice(
        self, request: pb2.MCPriceRequest, context: grpc.aio.ServicerContext
    ) -> pb2.MCPriceResponse:
        # --- model ---
        model_field = request.WhichOneof("model")
        if model_field == "gbm":
            model = GBM_Model(sigma=request.gbm.sigma)
        elif model_field == "heston":
            h = request.heston
            model = HestonModel(
                kappa=h.kappa, theta=h.theta, xi=h.xi, rho=h.rho, v0=h.v0
            )
        else:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No model specified")
            return pb2.MCPriceResponse()

        # --- instrument ---
        inst_field = request.WhichOneof("instrument")
        if inst_field == "vanilla":
            v = request.vanilla
            instrument = VanillaOption(
                strike=v.strike, maturity=v.maturity, is_call=v.is_call
            )
        elif inst_field == "asian":
            a = request.asian
            obs = list(a.obs_times) if a.obs_times else None
            instrument = AsianArithmeticOption(
                maturity=a.maturity,
                strike=a.strike,
                callput=a.callput or "call",
                obs_times=obs,
                include_t0=a.include_t0,
                interp=a.interp or "linear",
            )
        else:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "No instrument specified"
            )
            return pb2.MCPriceResponse()

        # --- market ---
        m = request.market
        market = MarketData(spot=m.spot, rate=m.rate, div_yield=m.div_yield)

        # --- engine ---
        engine = MonteCarloEngine(
            n_paths=request.n_paths or 20_000,
            n_steps=request.n_steps or 50,
            seed=request.seed if request.seed >= 0 else 42,
            antithetic=request.antithetic,
        )

        result = engine.price(model=model, instrument=instrument, market=market)

        return pb2.MCPriceResponse(
            price=result.price,
            stderr=result.stderr,
            ci95_lo=result.ci95[0],
            ci95_hi=result.ci95[1],
            n_paths=result.n_paths,
            n_steps=result.n_steps,
        )

    # ── BSPrice ───────────────────────────────────────────────────────────────

    async def BSPrice(
        self, request: pb2.BSPriceRequest, context: grpc.aio.ServicerContext
    ) -> pb2.BSPriceResponse:
        price = bs_price(
            spot=request.spot,
            strike=request.strike,
            maturity=request.maturity,
            rate=request.rate,
            div=request.div,
            vol=request.vol,
            is_call=request.is_call,
        )
        return pb2.BSPriceResponse(price=price)

    # ── ImpliedVol ────────────────────────────────────────────────────────────

    async def ImpliedVol(
        self, request: pb2.ImpliedVolRequest, context: grpc.aio.ServicerContext
    ) -> pb2.ImpliedVolResponse:
        try:
            vol = implied_vol(
                price=request.price,
                spot=request.spot,
                strike=request.strike,
                maturity=request.maturity,
                rate=request.rate,
                div=request.div,
                is_call=request.is_call,
            )
            return pb2.ImpliedVolResponse(vol=vol)
        except (ValueError, ZeroDivisionError) as exc:
            await context.abort(grpc.StatusCode.OUT_OF_RANGE, str(exc))
            return pb2.ImpliedVolResponse()

    # ── Calibrate ─────────────────────────────────────────────────────────────

    async def Calibrate(
        self, request: pb2.CalibrateRequest, context: grpc.aio.ServicerContext
    ) -> pb2.CalibrateResponse:
        options_df = pd.DataFrame([
            {
                "strike": q.strike,
                "maturity_years": q.maturity_years,
                "is_call": q.is_call,
                "market_price": q.market_price,
            }
            for q in request.quotes
        ])

        engine_kwargs = {
            "n_paths": request.n_paths or 20_000,
            "n_steps": request.n_steps or 50,
            "seed": request.seed if request.seed > 0 else 42,
            "antithetic": request.antithetic,
        }

        spot, rate, div = request.spot, request.rate, request.div
        model_type = request.model_type
        model_name = MODEL_TYPE_NAMES.get(model_type)

        try:
            if model_name is None:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Unknown model_type={model_type}",
                )
                return pb2.CalibrateResponse()

            result = self._toolbox.calibrate(
                model_name=model_name,
                spot=spot,
                options_df=options_df,
                rate=rate,
                div=div,
                x0=list(request.x0),
                engine_kwargs=engine_kwargs,
            )

        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return pb2.CalibrateResponse()
        except RuntimeError as exc:
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            return pb2.CalibrateResponse()

        return pb2.CalibrateResponse(
            model_name=result.model_name,
            params=result.params,
            mse=result.mse,
            elapsed_s=result.elapsed_s,
            per_option_ivols=result.per_option_ivols,
        )

    async def UpdateCalibrationWindow(
        self,
        request: pb2.CalibrationWindowRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.CalibrationWindowResponse:
        timed_quotes_df = pd.DataFrame([
            {
                "strike": q.quote.strike,
                "maturity_years": q.quote.maturity_years,
                "is_call": q.quote.is_call,
                "market_price": q.quote.market_price,
                "observed_at_ms": q.observed_at_ms,
            }
            for q in request.quotes
        ])

        model_name = MODEL_TYPE_NAMES.get(request.model_type)
        if model_name is None:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Unknown model_type={request.model_type}",
            )
            return pb2.CalibrationWindowResponse()

        engine_kwargs = {
            "n_paths": request.n_paths or 20_000,
            "n_steps": request.n_steps or 50,
            "seed": request.seed if request.seed > 0 else 42,
            "antithetic": request.antithetic,
        }

        try:
            result = self._toolbox.calibrate_windowed(
                asset_id=request.asset_id or "default",
                model_name=model_name,
                spot=request.spot,
                timed_quotes_df=timed_quotes_df,
                rate=request.rate,
                div=request.div,
                x0=list(request.x0),
                engine_kwargs=engine_kwargs,
                as_of_ms=request.as_of_ms if request.as_of_ms > 0 else None,
                calibration_window_ms=request.calibration_window_ms,
                update_interval_ms=request.update_interval_ms,
                force_refresh=request.force_refresh,
            )
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return pb2.CalibrationWindowResponse()
        except RuntimeError as exc:
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            return pb2.CalibrationWindowResponse()

        calibration = pb2.CalibrateResponse(
            model_name=result.calibration.model_name,
            params=result.calibration.params,
            mse=result.calibration.mse,
            elapsed_s=result.calibration.elapsed_s,
            per_option_ivols=result.calibration.per_option_ivols,
        )

        return pb2.CalibrationWindowResponse(
            calibration=calibration,
            recalibrated=result.recalibrated,
            calibrated_at_ms=result.calibrated_at_ms,
            next_update_due_ms=result.next_update_due_ms,
            window_start_ms=result.window_start_ms,
            window_end_ms=result.window_end_ms,
            quotes_in_window=result.quotes_in_window,
            quotes_total=result.quotes_total,
        )
