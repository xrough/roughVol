"""Console-based model comparison workflow."""

from __future__ import annotations

from roughvol.experiments.model_comparison.model_comparison import HedgeBookConfig, compare_models, make_surface_dataset
from roughvol.types import MarketData


def main() -> None:
    market = MarketData(spot=100.0, rate=0.01, div_yield=0.0)
    reference_model_name = "HESTON"
    reference_params = {
        "kappa": 2.4,
        "theta": 0.04,
        "xi": 0.55,
        "rho": -0.7,
        "v0": 0.045,
    }

    surface_df = make_surface_dataset(
        market=market,
        model_name=reference_model_name,
        params=reference_params,
        strikes=[80.0, 90.0, 100.0, 110.0, 120.0],
        maturities=[0.25, 0.5, 1.0],
        engine_kwargs={"n_paths": 8_000, "n_steps": 48, "seed": 21, "antithetic": True},
    )

    report = compare_models(
        market=market,
        surface_df=surface_df,
        candidate_models=["BS", "GBM_MC", "HESTON"],
        reference_model_name=reference_model_name,
        reference_params=reference_params,
        hedge_book=HedgeBookConfig(
            strike=100.0,
            maturity=1.0,
            n_realized_paths=32,
            n_hedge_steps=12,
            hedge_pricer_paths=2_000,
        ),
        calibration_engine_kwargs={"n_paths": 3_000, "n_steps": 24, "seed": 4, "antithetic": True},
        surface_engine_kwargs={"n_paths": 4_000, "n_steps": 24, "seed": 9, "antithetic": True},
    )

    print(
        f"Reference model={report.reference_model_name} | "
        f"surface_quotes={report.n_surface_quotes}",
    )
    for result in report.results:
        print(
            f"{result.model_name:<8} "
            f"price_rmse={result.price_rmse:.6f} "
            f"iv_rmse={result.iv_rmse:.6f} "
            f"hedge_mean={result.hedge_pnl_mean:.6f} "
            f"hedge_std={result.hedge_pnl_std:.6f} "
            f"hedge_rmse={result.hedge_pnl_rmse:.6f} "
            f"params={result.calibration.params}",
        )


if __name__ == "__main__":
    main()
