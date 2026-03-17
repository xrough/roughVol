from __future__ import annotations

from roughvol.lab import HedgeBookConfig, compare_models, make_surface_dataset
from roughvol.types import MarketData


def test_model_lab_produces_surface_and_hedge_metrics():
    market = MarketData(spot=100.0, rate=0.01, div_yield=0.0)
    reference_model_name = "GBM_MC"
    reference_params = {"sigma": 0.2}

    surface_df = make_surface_dataset(
        market=market,
        model_name=reference_model_name,
        params=reference_params,
        strikes=[90.0, 100.0, 110.0],
        maturities=[0.5, 1.0],
    )

    report = compare_models(
        market=market,
        surface_df=surface_df,
        candidate_models=["BS"],
        reference_model_name=reference_model_name,
        reference_params=reference_params,
        hedge_book=HedgeBookConfig(
            strike=100.0,
            maturity=1.0,
            n_realized_paths=8,
            n_hedge_steps=4,
            hedge_pricer_paths=500,
        ),
    )

    assert report.reference_model_name == "GBM_MC"
    assert report.n_surface_quotes == 6
    assert len(report.results) == 1

    result = report.results[0]
    assert result.model_name == "BS"
    assert result.price_rmse >= 0.0
    assert result.iv_rmse >= 0.0
    assert result.hedge_pnl_rmse >= 0.0
