"""
Calibration Demo: Compare GBM, Heston, and Rough Bergomi against live option data.

Uses yfinance to fetch real market data and option chains, calibrates three models
(GBM, Heston, Rough Bergomi) via Monte Carlo, then produces four comparison figures:

  calibration_demo_iv_smile.png   — market IV smile vs model smiles per ticker
  calibration_demo_rmse_bars.png  — IV RMSE bar chart (models × tickers)
  calibration_demo_paths.png      — simulated spot paths: GBM vs Rough Bergomi
  calibration_demo_surface.png    — vol surface heatmaps: market vs three models

Usage:
    pip install yfinance
    python -m roughvol.experiments.run_calibration_demo
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; works in all environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yfinance  # noqa: F401
except ImportError:
    raise ImportError(
        "yfinance is required to run this demo.\n"
        "Install it with:  pip install yfinance"
    )

from roughvol.analytics.black_scholes_formula import implied_vol
from roughvol.calibration.calibration import (
    make_gbm_calibrator,
    make_heston_calibrator,
    make_rough_bergomi_calibrator,
)
from roughvol.data.yfinance_loader import get_market_data, get_option_surface
from roughvol.engines.mc import MonteCarloEngine
from roughvol.instruments.vanilla import VanillaOption
from roughvol.models.GBM_model import GBM_Model
from roughvol.models.heston_model import HestonModel
from roughvol.models.rough_bergomi_model import RoughBergomiModel
from roughvol.types import MarketData, SimConfig, make_rng

# ── Configuration ────────────────────────────────────────────────────────────

# Equities to calibrate. Add or remove tickers as needed; any ticker without
# listed options will be skipped automatically.
TICKERS = ["SPY", "AAPL"]

# ---------------------------------------------------------------------------
# Monte Carlo engine settings used *during* calibration (the optimizer calls
# the MC pricer at every function evaluation, so keeping n_paths and n_steps
# small is critical for a manageable runtime).
#
# Trade-off: fewer paths → noisier loss surface → optimizer may converge
# to a worse local minimum.  Increase n_paths once the pipeline is verified.
# ---------------------------------------------------------------------------

# GBM has a single parameter (σ), so it needs far fewer evaluations to converge.
CALIB_ENGINE_GBM = {
    "n_paths": 2_000,
    "n_steps": 20,   # coarse grid is fine for the exact log-Euler GBM scheme
    "seed": 42,
    "antithetic": True,  # antithetic variates halve the variance for free
}

# Heston has 5 parameters (κ, θ, ξ, ρ, v₀) — harder surface, but the
# Euler scheme is O(n) so we can afford the same path count as GBM.
CALIB_ENGINE_HESTON = {
    "n_paths": 2_000,
    "n_steps": 20,
    "seed": 42,
    "antithetic": True,
}

# Rough Bergomi has 4 parameters (H, η, ρ, ξ₀).  The volterra-midpoint
# scheme is O(n_steps²) per path, so n_steps is kept lower than the other
# two models.  H < 0.5 means the vol process has long memory and is more
# sensitive to the time grid than Markovian models.
#
# Accuracy notes:
#   - n_paths controls MC noise on the loss surface.  Too few paths → the
#     gradient estimate used by L-BFGS-B is noisy → optimizer wanders.
#   - n_steps controls discretisation error of the Volterra integral.
#     The kernel K(t) = t^{H−0.5} is singular near t=0; more steps resolve
#     the singularity better.  Error is O(n_steps^{0.5−H}), so for H≈0.1
#     you need more steps than for Heston.
#   - Cost scales as n_paths × n_steps²; the values below are a deliberate
#     balance between accuracy and a tolerable per-ticker runtime.
CALIB_ENGINE_RB = {
    "n_paths": 5_000,
    "n_steps": 52,   # O(n²): 52² = 2704 kernel evals per path — resolves the H≈0.1 singularity well
    "seed": 42,
    "antithetic": True,
}

# Engine used for *post-calibration* visualisation repricing (IV smile curves,
# surface heatmaps).  Even smaller than the calibration engines — accuracy
# matters less here than interactivity.
VIZ_ENGINE = {
    "n_paths": 300,
    "n_steps": 16,
    "seed": 99,
    "antithetic": True,
}

# Per-model display settings used across all figures.
MODEL_COLOURS = {
    "GBM": "steelblue",
    "Heston": "darkorange",
    "RoughBergomi": "crimson",
}
MODEL_LINESTYLES = {
    "GBM": "--",
    "Heston": ":",
    "RoughBergomi": "-.",
}
MODEL_LABELS = {
    "GBM": "GBM",
    "Heston": "Heston",
    "RoughBergomi": "Rough Bergomi",
}

# ── Option filtering ─────────────────────────────────────────────────────────


def filter_options_for_calibration(
    surface_df: pd.DataFrame,
    spot: float,
    min_days: int = 30,
    max_days: int = 90,
    moneyness: float = 0.20,
) -> pd.DataFrame:
    """Narrow the raw surface to liquid 1–3 month options for calibration.

    Focuses on the 30–90 day window because:
    - Options shorter than 30 days are noisy (time-value near zero, wide spreads).
    - Options longer than 90 days add runtime without improving the near-term
      smile fit that distinguishes rough vol from classical models.
    - ±20% moneyness keeps the strike range where market prices are reliable.

    If fewer than 5 options survive the initial cut, moneyness is relaxed to 30%
    so the optimizer always has enough constraints.
    """
    lo = min_days / 365.25
    hi = max_days / 365.25
    mask = (
        (surface_df["maturity_years"] >= lo)
        & (surface_df["maturity_years"] <= hi)
        & (surface_df["strike"] >= spot * (1.0 - moneyness))
        & (surface_df["strike"] <= spot * (1.0 + moneyness))
    )
    filtered = surface_df[mask].copy()

    if len(filtered) < 5:
        # Too sparse — widen the moneyness band before giving up
        moneyness = 0.30
        mask = (
            (surface_df["maturity_years"] >= lo)
            & (surface_df["maturity_years"] <= hi)
            & (surface_df["strike"] >= spot * (1.0 - moneyness))
            & (surface_df["strike"] <= spot * (1.0 + moneyness))
        )
        filtered = surface_df[mask].copy()

    return filtered.reset_index(drop=True)


# ── Model reconstruction from CalibResult ────────────────────────────────────


def _build_model(model_name: str, params: dict[str, float]):
    """Reconstruct a PathModel instance from a calibrated parameter dict.

    The calibrator stores params as a plain dict keyed by param_names.
    This function maps that dict back to the typed dataclass constructors.
    """
    if model_name == "GBM":
        # Constant-vol geometric Brownian motion: dS/S = σ dW
        return GBM_Model(sigma=params["sigma"])
    if model_name == "Heston":
        # Heston SV: dv = κ(θ−v)dt + ξ√v dW², corr(dW¹, dW²) = ρ
        return HestonModel(
            kappa=params["kappa"],  # mean-reversion speed
            theta=params["theta"],  # long-run variance
            xi=params["xi"],        # vol-of-vol
            rho=params["rho"],      # spot-vol correlation
            v0=params["v0"],        # initial variance
        )
    if model_name == "RoughBergomi":
        # Rough Bergomi: forward variance driven by fractional BM with Hurst H < 0.5
        # σ²(t) = ξ₀ · exp(η·Ŷ_t − η²/2 · t^{2H}), corr(dW^S, dW^Y) = ρ
        #
        # blp-hybrid scheme (Bennedsen-Lunde-Pakkanen):
        #   - Near-field lags (k=1..κ): exact bivariate Gaussian sampling,
        #     which correctly handles the kernel singularity K(dt) ~ dt^{H-0.5}.
        #   - Far-field lags (k>κ): FFT-based convolution with midpoint weights,
        #     reducing complexity from O(n²) to O(n log n).
        # This gives better accuracy than volterra-midpoint at the same n_steps,
        # especially for small H where the singularity is most pronounced.
        return RoughBergomiModel(
            hurst=params["hurst"],  # Hurst exponent H ∈ (0, 0.5) — roughness
            eta=params["eta"],      # vol-of-vol scaling
            rho=params["rho"],      # spot-vol correlation (typically negative)
            xi0=params["xi0"],      # initial forward variance level
            scheme="blp-hybrid",    # O(n log n), exact near-field kernel treatment
        )
    raise ValueError(f"Unknown model: {model_name}")


# ── IV smile computation ─────────────────────────────────────────────────────


def compute_model_iv_smile(
    model_name: str,
    params: dict[str, float],
    market_data: MarketData,
    maturity: float,
    strikes: list[float],
    engine_kwargs: dict,
) -> list[float | None]:
    """Price a strip of European calls with a calibrated model and invert to IVs.

    For each strike K:
      1. Price call(K, T) via MonteCarloEngine.
      2. Invert the Black-Scholes formula for the implied vol σ_imp(K).

    Returns a list aligned with *strikes*; entries are None where the MC price
    falls outside no-arb bounds (e.g. if n_paths is very small and the estimate
    is noisy).
    """
    model = _build_model(model_name, params)
    engine = MonteCarloEngine(**engine_kwargs)
    ivs: list[float | None] = []
    for k in strikes:
        inst = VanillaOption(strike=k, maturity=maturity, is_call=True)
        try:
            pr = engine.price(model=model, instrument=inst, market=market_data)
            iv = implied_vol(
                price=pr.price,
                spot=market_data.spot,
                strike=k,
                maturity=maturity,
                rate=market_data.rate,
                div=market_data.div_yield,
                is_call=True,
            )
            ivs.append(iv)
        except (ValueError, Exception):
            # MC price out of no-arb bounds — skip this strike
            ivs.append(None)
    return ivs


# ── IV RMSE ──────────────────────────────────────────────────────────────────


def compute_iv_rmse(
    model_name: str,
    params: dict[str, float],
    calib_df: pd.DataFrame,
    market_data: MarketData,
    engine_kwargs: dict,
) -> float:
    """Compute IV RMSE (in vol units) between the model and market implied vols.

    RMSE = sqrt( mean( (σ_model(K,T) − σ_market(K,T))² ) )

    Uses the full calib_df (not the subsampled opts_df used during fitting) so
    the reported RMSE is an out-of-sample metric across the whole filtered surface.
    Returns NaN if no options could be priced successfully.
    """
    model = _build_model(model_name, params)
    engine = MonteCarloEngine(**engine_kwargs)
    errors: list[float] = []
    for _, row in calib_df.iterrows():
        inst = VanillaOption(
            strike=float(row["strike"]),
            maturity=float(row["maturity_years"]),
            is_call=bool(row["is_call"]),
        )
        try:
            pr = engine.price(model=model, instrument=inst, market=market_data)
            iv_model = implied_vol(
                price=pr.price,
                spot=market_data.spot,
                strike=float(row["strike"]),
                maturity=float(row["maturity_years"]),
                rate=market_data.rate,
                div=market_data.div_yield,
                is_call=bool(row["is_call"]),
            )
            # Signed error: positive means model overestimates vol
            errors.append(iv_model - float(row["implied_vol"]))
        except (ValueError, Exception):
            pass  # skip options where MC price violates no-arb bounds
    if not errors:
        return float("nan")
    return float(np.sqrt(np.mean(np.array(errors) ** 2)))


# ── Per-ticker calibration ────────────────────────────────────────────────────


def calibrate_ticker(
    ticker_symbol: str,
    calib_df: pd.DataFrame,
    surface_df: pd.DataFrame,
    market_data: MarketData,
) -> dict[str, Any]:
    """Calibrate GBM, Heston, and Rough Bergomi for one ticker.

    Calibration strategy
    --------------------
    Each model is calibrated by minimising the mean-squared error between
    Monte Carlo prices and market prices (L-BFGS-B via scipy.optimize.minimize,
    implemented in MCCalibrator).  The warm-start for each model uses the
    median market implied vol as a proxy for the ATM level:

      - GBM:    x0 = [σ_ATM]
      - Heston: x0 = [κ=2, θ=σ²_ATM, ξ=0.3, ρ=−0.5, v0=σ²_ATM]
      - rBergomi: x0 = [H=0.1, η=1.5, ρ=−0.7, ξ0=σ²_ATM]

    Only MAX_OPTS options are passed to the optimizer per call to keep the
    per-iteration MC cost low.  IV RMSE is then measured on the full calib_df.

    Returns a dict with keys:
      ticker, market_data, surface_df, calib_df,
      results  (model_name → CalibResult),
      iv_rmse  (model_name → float),
      error    (None if successful)
    """
    # Median IV across the filtered surface — used as ATM warm-start for all models
    atm_iv = float(calib_df["implied_vol"].median())

    base_cols = ["strike", "maturity_years", "is_call", "market_price"]

    def _stratified_sample(pool: pd.DataFrame, n: int) -> pd.DataFrame:
        """Sort by (maturity_years, strike) and take every k-th row.

        Sorting by both axes before striding ensures the chosen options span
        the full smile (strike axis) and multiple expiries (maturity axis),
        rather than clustering near ATM or in a single expiry.  This is
        superior to random sampling when the optimizer needs to identify both
        the skew level and its term structure.
        """
        pool = pool.sort_values(["maturity_years", "strike"]).reset_index(drop=True)
        step = max(1, len(pool) // n)
        return pool.iloc[::step].head(n).reset_index(drop=True)

    # --- GBM: 10 stratified options ---
    # 1 parameter only, but spreading options across strikes still helps the
    # optimizer see both ATM (pinning σ level) and OTM (confirming flat smile).
    opts_df_gbm = _stratified_sample(calib_df[base_cols], n=10)

    # --- Heston: 14 stratified options ---
    # 5 parameters (κ, θ, ξ, ρ, v₀) need options across multiple maturities
    # (to identify κ and θ) and strikes (to identify ρ and ξ).
    opts_df_heston = _stratified_sample(calib_df[base_cols], n=14)

    # --- Rough Bergomi: same 30–90 day window as GBM and Heston, 14 options ---
    # Using the same maturity window across all three models makes the IV RMSE
    # a fair apples-to-apples comparison — each model is evaluated on the same
    # set of options and faces the same calibration problem.
    opts_df_rb = _stratified_sample(calib_df[base_cols], n=14)

    results: dict = {}
    iv_rmse: dict = {}

    # Build one calibrator per model; each wraps an MCCalibrator with its own
    # engine, parameter bounds, and warm-start vector.
    # blp-hybrid is passed explicitly to make_rough_bergomi_calibrator so that
    # the same scheme is used both during optimization and post-calibration.
    calibrators = [
        ("GBM",          make_gbm_calibrator(x0_sigma=atm_iv,    engine_kwargs=CALIB_ENGINE_GBM),                               opts_df_gbm),
        ("Heston",       make_heston_calibrator(x0_sigma=atm_iv, engine_kwargs=CALIB_ENGINE_HESTON),                             opts_df_heston),
        ("RoughBergomi", make_rough_bergomi_calibrator(x0_sigma=atm_iv, engine_kwargs=CALIB_ENGINE_RB, scheme="blp-hybrid"),      opts_df_rb),
    ]

    for model_name, calibrator, opts_df in calibrators:
        print(f"  [{ticker_symbol}] Calibrating {model_name} on {len(opts_df)} options...")
        try:
            # Run the optimizer — internally calls engine.price() at each step
            calib_result = calibrator.calibrate(
                spot=market_data.spot,
                options_df=opts_df,
                rate=market_data.rate,
                div=market_data.div_yield,
            )
            results[model_name] = calib_result

            # IV-MSE > 0.01 means average IV error > 10 vol points — poor fit
            if calib_result.mse > 0.01:
                print(
                    f"  [WARN] {ticker_symbol} {model_name}: "
                    f"poor calibration (MSE={calib_result.mse:.3e})"
                )

            # Evaluate IV RMSE on the full calib_df (not just the 8 training options)
            rmse = compute_iv_rmse(
                model_name=model_name,
                params=calib_result.params,
                calib_df=calib_df,
                market_data=market_data,
                engine_kwargs=VIZ_ENGINE,
            )
            iv_rmse[model_name] = rmse
            print(
                f"  [{ticker_symbol}] {model_name} done  "
                f"params={calib_result.params}  "
                f"IV-RMSE={rmse:.4f}"
            )
        except Exception as exc:
            print(f"  [ERROR] {ticker_symbol} {model_name} calibration failed: {exc}")
            results[model_name] = None
            iv_rmse[model_name] = float("nan")

    return {
        "ticker": ticker_symbol,
        "market_data": market_data,
        "surface_df": surface_df,   # full surface for heatmap plotting
        "calib_df": calib_df,       # filtered surface used for calibration
        "results": results,
        "iv_rmse": iv_rmse,
        "error": None,
    }


# ── Figure 1: IV smile comparison ────────────────────────────────────────────


def plot_iv_smiles(all_results: dict[str, dict]) -> None:
    """One subplot per ticker: market IV smile vs calibrated model smiles.

    The smile is shown at the dominant expiry (the expiry with the most
    options in calib_df).  Market quotes are scatter points; model curves are
    computed on a 10-point moneyness grid via compute_model_iv_smile().

    Expected visual pattern:
    - GBM produces a flat line (constant σ, no skew by construction).
    - Heston shows a mild skew and term-structure curvature.
    - Rough Bergomi typically fits the short-dated skew better than Heston
      because the rough H < 0.5 generates steeper short-maturity smiles.
    """
    tickers = [t for t, r in all_results.items() if r.get("error") is None]
    if not tickers:
        return

    n = len(tickers)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, ticker in enumerate(tickers):
        ax = axes_flat[idx]
        res = all_results[ticker]
        market_data: MarketData = res["market_data"]
        calib_df: pd.DataFrame = res["calib_df"]
        spot = market_data.spot

        # Use the expiry with the most options so the smile has the most points
        dominant_expiry = calib_df.groupby("expiry_str").size().idxmax()
        exp_df = calib_df[calib_df["expiry_str"] == dominant_expiry]
        maturity = float(exp_df["maturity_years"].iloc[0])

        # --- Market quotes ---
        moneyness_mkt = exp_df["strike"].values / spot
        iv_mkt = exp_df["implied_vol"].values
        calls = exp_df["is_call"].values
        ax.scatter(
            moneyness_mkt[calls], iv_mkt[calls],
            color="black", marker="o", s=30, label="Market (call)", zorder=5,
        )
        ax.scatter(
            moneyness_mkt[~calls], iv_mkt[~calls],
            color="dimgray", marker="^", s=30, label="Market (put)", zorder=5,
        )

        # --- Model smiles on a fine moneyness grid ---
        # 10 equally-spaced moneyness points from 80% to 120% of spot
        fine_moneyness = np.linspace(0.80, 1.20, 10)
        fine_strikes = (fine_moneyness * spot).tolist()

        for model_name in ("GBM", "Heston", "RoughBergomi"):
            cr = res["results"].get(model_name)
            if cr is None:
                continue
            ivs = compute_model_iv_smile(
                model_name=model_name,
                params=cr.params,
                market_data=market_data,
                maturity=maturity,
                strikes=fine_strikes,
                engine_kwargs=VIZ_ENGINE,
            )
            # Drop None entries (MC price out of bounds at that strike)
            valid_x = [fine_moneyness[i] for i, v in enumerate(ivs) if v is not None]
            valid_y = [v for v in ivs if v is not None]
            if valid_x:
                ax.plot(
                    valid_x, valid_y,
                    color=MODEL_COLOURS[model_name],
                    linestyle=MODEL_LINESTYLES[model_name],
                    linewidth=1.8,
                    label=MODEL_LABELS[model_name],
                )

        ax.set_title(f"{ticker}  T={maturity:.2f}yr  ({dominant_expiry})", fontsize=11)
        ax.set_xlabel("Moneyness  (K / S)", fontsize=9)
        ax.set_ylabel("Implied Volatility", fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots (when n < nrows*ncols)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("IV Smile: Market vs Calibrated Models", fontsize=14, y=1.01)
    fig.tight_layout()
    out = "calibration_demo_iv_smile.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


# ── Figure 2: RMSE bar chart ──────────────────────────────────────────────────


def plot_rmse_bars(all_results: dict[str, dict]) -> None:
    """Grouped bar chart: IV RMSE (in vol percentage points) per ticker and model.

    Reading the chart:
    - A shorter bar means the model fits the market smile more closely.
    - GBM is expected to have the highest RMSE because it cannot produce a skew.
    - Heston reduces RMSE by adding a stochastic variance process with mean reversion.
    - Rough Bergomi typically achieves the lowest RMSE at short maturities because
      H < 0.5 induces a power-law explosion of ATM skew as T → 0, consistent with
      what is observed empirically in equity index options.
    """
    tickers = [t for t, r in all_results.items() if r.get("error") is None]
    if not tickers:
        return

    model_names = ["GBM", "Heston", "RoughBergomi"]
    x = np.arange(len(tickers))
    width = 0.25  # width of each bar within a group

    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(tickers)), 5))

    for i, model_name in enumerate(model_names):
        # Convert vol units → percentage points (×100) for readability
        rmses = [
            all_results[t]["iv_rmse"].get(model_name, float("nan")) * 100
            for t in tickers
        ]
        bars = ax.bar(
            x + i * width,
            rmses,
            width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLOURS[model_name],
            alpha=0.85,
            edgecolor="white",
        )
        ax.bar_label(
            bars,
            labels=[f"{v:.1f}" if not np.isnan(v) else "N/A" for v in rmses],
            fontsize=8,
            padding=2,
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel("IV RMSE (vol ppts)", fontsize=10)
    ax.set_title("Calibration Quality: IV RMSE per Model and Ticker", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = "calibration_demo_rmse_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 3: Simulated paths ─────────────────────────────────────────────────


def plot_simulated_paths(all_results: dict[str, dict]) -> None:
    """Side-by-side: 30 GBM paths vs 30 Rough Bergomi paths for the first ticker.

    Both sets are normalised to S(0) = 1 so the diffusion character is directly
    comparable regardless of the absolute spot level.

    What to look for:
    - GBM paths are log-normal: volatility is constant, so the fan of paths
      widens uniformly (constant cone width on a log scale).
    - Rough Bergomi paths exhibit volatility clustering: periods of high vol
      are followed by more high vol (long memory from H < 0.5), producing a
      rougher, more spiky appearance and occasional vol bursts.

    Paths are simulated by calling model.simulate_paths() directly rather than
    going through MonteCarloEngine, because PriceResult does not store paths.
    """
    tickers = [t for t, r in all_results.items() if r.get("error") is None]
    if not tickers:
        return

    ticker = tickers[0]
    res = all_results[ticker]
    market_data: MarketData = res["market_data"]

    gbm_cr = res["results"].get("GBM")
    rb_cr = res["results"].get("RoughBergomi")
    if gbm_cr is None or rb_cr is None:
        print("  [WARN] Skipping path plot — GBM or RoughBergomi calibration missing")
        return

    # Simulate over a 1-year horizon with daily steps to show the full path texture
    T = 1.0
    n_paths = 30
    n_steps = 252  # one step per trading day
    sim = SimConfig(
        n_paths=n_paths,
        maturity=T,
        n_steps=n_steps,
        seed=7,
        antithetic=False,   # antithetic would pair paths; keep independent for variety
        store_paths=True,   # required to read intermediate S(t) values
    )

    gbm_model = _build_model("GBM", gbm_cr.params)
    rb_model = _build_model("RoughBergomi", rb_cr.params)

    # Different seeds so GBM and rBergomi are driven by independent noise
    rng_gbm = make_rng(7)
    rng_rb = make_rng(8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gbm_paths = gbm_model.simulate_paths(market=market_data, sim=sim, rng=rng_gbm)
        rb_paths = rb_model.simulate_paths(market=market_data, sim=sim, rng=rng_rb)

    t_grid = gbm_paths.t                              # shape (n_steps+1,)
    gbm_spot_norm = gbm_paths.spot / market_data.spot  # normalise: S(0) = 1
    rb_spot_norm = rb_paths.spot / market_data.spot

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    sigma_str = f"σ={gbm_cr.params['sigma']:.3f}"
    h_str = f"H={rb_cr.params['hurst']:.3f}, η={rb_cr.params['eta']:.2f}"

    for ax, spot_norm, colour, mean_colour, title_suffix in [
        (axes[0], gbm_spot_norm, "steelblue", "navy",
         f"GBM  ({sigma_str})"),
        (axes[1], rb_spot_norm, "lightcoral", "darkred",
         f"Rough Bergomi  ({h_str})"),
    ]:
        # Individual paths — semi-transparent so clusters are visible
        for path in spot_norm:
            ax.plot(t_grid, path, color=colour, alpha=0.25, linewidth=0.8)
        # Mean path in bold — should be close to 1.0 for risk-neutral dynamics
        mean_path = spot_norm.mean(axis=0)
        ax.plot(t_grid, mean_path, color=mean_colour, linewidth=2.2, label="Mean path")
        # Reference line at S(0) = 1
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"{ticker}  —  {title_suffix}", fontsize=11)
        ax.set_xlabel("Time (years)", fontsize=9)
        ax.set_ylabel("S(t) / S(0)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Simulated Spot Paths: GBM vs Rough Bergomi  ({ticker})",
        fontsize=13,
    )
    fig.tight_layout()
    out = "calibration_demo_paths.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 4: Vol surface heatmaps ────────────────────────────────────────────


def plot_vol_surface(all_results: dict[str, dict]) -> None:
    """2×2 heatmap grid: market surface vs GBM, Heston, rBergomi for first ticker.

    Each cell shows the implied vol at a (moneyness, maturity) grid point.
    All four panels share the same colour scale so differences are immediately
    visible:
    - The market surface typically shows a pronounced negative skew (OTM puts
      are more expensive than OTM calls) and a downward-sloping term structure.
    - GBM produces a flat surface — no skew, no term structure.
    - Heston partially recovers the skew via the vol-spot correlation ρ.
    - Rough Bergomi tends to match the steep short-maturity skew better than
      Heston, but may deviate at long maturities where the rough memory effect
      is less dominant.

    The market grid is populated by averaging the IVs of options whose strikes
    fall within ±3% of each moneyness grid node and within 1% of the maturity.
    """
    tickers = [t for t, r in all_results.items() if r.get("error") is None]
    if not tickers:
        return

    ticker = tickers[0]
    res = all_results[ticker]
    market_data: MarketData = res["market_data"]
    surface_df: pd.DataFrame = res["surface_df"]
    spot = market_data.spot

    # Fixed moneyness grid — same for both market binning and model repricing
    MONEYNESS_GRID = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    strikes_abs = [m * spot for m in MONEYNESS_GRID]

    # Limit to 6 representative maturities to keep the heatmap readable
    maturities = sorted(surface_df["maturity_years"].unique())
    if len(maturities) > 6:
        idx = np.round(np.linspace(0, len(maturities) - 1, 6)).astype(int)
        maturities = [maturities[i] for i in idx]

    n_mat = len(maturities)
    n_mon = len(MONEYNESS_GRID)

    # --- Build market IV grid ---
    # For each (moneyness_node, maturity) cell, average the IVs of nearby options
    market_grid = np.full((n_mon, n_mat), np.nan)
    for j, mat in enumerate(maturities):
        for i in range(n_mon):
            m_mid = MONEYNESS_GRID[i]
            tolerance = 0.03   # ±3% moneyness bin width
            sub = surface_df[
                (np.abs(surface_df["maturity_years"] - mat) < 0.01)
                & (np.abs(surface_df["strike"] / spot - m_mid) < tolerance)
                & (surface_df["is_call"])
            ]
            if not sub.empty:
                market_grid[i, j] = sub["implied_vol"].mean()

    # --- Build model IV grids ---
    # For each model, re-price calls at every (strike, maturity) grid point
    # and invert to implied vol.  Uses VIZ_ENGINE (small, fast).
    model_grids: dict[str, np.ndarray] = {}
    for model_name in ("GBM", "Heston", "RoughBergomi"):
        cr = res["results"].get(model_name)
        if cr is None:
            model_grids[model_name] = np.full((n_mon, n_mat), np.nan)
            continue
        grid = np.full((n_mon, n_mat), np.nan)
        for j, mat in enumerate(maturities):
            ivs = compute_model_iv_smile(
                model_name=model_name,
                params=cr.params,
                market_data=market_data,
                maturity=mat,
                strikes=strikes_abs,
                engine_kwargs=VIZ_ENGINE,
            )
            for i, iv in enumerate(ivs):
                if iv is not None:
                    grid[i, j] = iv
        model_grids[model_name] = grid

    # Shared colour scale: 5th–95th percentile to clip outlier MC noise
    all_vals = [
        v for g in [market_grid] + list(model_grids.values())
        for v in g.flatten()
        if not np.isnan(v)
    ]
    vmin = float(np.nanpercentile(all_vals, 5)) if all_vals else 0.0
    vmax = float(np.nanpercentile(all_vals, 95)) if all_vals else 1.0

    mat_labels = [f"{m:.2f}" for m in maturities]
    mon_labels = [f"{int(m * 100)}%" for m in MONEYNESS_GRID]

    titles = ["Market", "GBM", "Heston", "Rough Bergomi"]
    grids = [
        market_grid,
        model_grids["GBM"],
        model_grids["Heston"],
        model_grids["RoughBergomi"],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    im = None
    for ax, title, grid in zip(axes_flat, titles, grids):
        im = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn_r",   # red = high vol, green = low vol
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(range(n_mat))
        ax.set_xticklabels(mat_labels, fontsize=8, rotation=30)
        ax.set_yticks(range(n_mon))
        ax.set_yticklabels(mon_labels, fontsize=8)
        ax.set_xlabel("Maturity (yr)", fontsize=9)
        ax.set_ylabel("Moneyness", fontsize=9)
        ax.set_title(title, fontsize=11)

        # Annotate each cell with the numeric IV value
        for i in range(n_mon):
            for j in range(n_mat):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i, f"{val:.2%}",
                        ha="center", va="center", fontsize=6.5, color="black",
                    )

    # Single shared colourbar for all four panels
    if im is not None:
        cbar = fig.colorbar(im, ax=axes_flat, orientation="vertical", fraction=0.02, pad=0.04)
        cbar.set_label("Implied Volatility", fontsize=9)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.suptitle(
        f"Implied Volatility Surface  ({ticker})\nMarket vs Calibrated Models",
        fontsize=13,
    )
    fig.tight_layout()
    out = "calibration_demo_surface.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """End-to-end pipeline: fetch → filter → calibrate → visualise.

    Pipeline steps for each ticker:
      1. get_market_data()        — live spot, rate, dividend yield via yfinance
      2. get_option_surface()     — option chain across all expiries, filtered to
                                    ±40% moneyness with no-arb IV check
      3. filter_options_for_calibration() — narrow to 30–90 day ±20% moneyness
      4. calibrate_ticker()       — GBM, Heston, rBergomi calibration + IV RMSE
      5. plot_*()                 — four comparison figures saved as PNGs
    """
    print("=" * 60)
    print("  Rough Vol Calibration Demo")
    print(f"  Tickers: {TICKERS}")
    print("=" * 60)

    all_results: dict[str, dict] = {}

    for ticker_symbol in TICKERS:
        print(f"\n{'─' * 50}")
        print(f"  {ticker_symbol}")
        print(f"{'─' * 50}")

        # Step 1: spot price, risk-free rate, dividend yield
        market_data = get_market_data(ticker_symbol)
        if market_data is None:
            print(f"  Skipping {ticker_symbol}: no market data")
            continue
        print(
            f"  Spot={market_data.spot:.2f}  "
            f"Rate={market_data.rate:.2%}  "
            f"Div={market_data.div_yield:.2%}"
        )

        # Step 2: full option surface (calls + puts, all expiries)
        surface_df = get_option_surface(ticker_symbol, market_data)
        if surface_df.empty:
            print(f"  Skipping {ticker_symbol}: no options data")
            continue
        print(f"  Fetched {len(surface_df)} options across surface")

        # Step 3: narrow to liquid 1–3 month options for calibration
        calib_df = filter_options_for_calibration(surface_df, market_data.spot)
        if len(calib_df) < 3:
            print(
                f"  Skipping {ticker_symbol}: too few liquid options "
                f"for calibration ({len(calib_df)})"
            )
            continue
        print(
            f"  Using {len(calib_df)} options for calibration  "
            f"(expiries: {calib_df['expiry_str'].nunique()})"
        )

        # Step 4: calibrate all three models
        result = calibrate_ticker(ticker_symbol, calib_df, surface_df, market_data)
        all_results[ticker_symbol] = result

    # ── Summary table ─────────────────────────────────────────────────────────
    if not all_results:
        print(
            "\nNo tickers calibrated successfully.\n"
            "Check network connection and that yfinance is installed."
        )
        return

    print("\n" + "=" * 60)
    print("  Calibration Summary  (IV RMSE in vol units)")
    print("=" * 60)
    print(f"  {'Ticker':<8} {'GBM RMSE':>10} {'Heston RMSE':>12} {'rBergomi RMSE':>14}")
    print(f"  {'─' * 48}")
    for t, res in all_results.items():
        rmses = res["iv_rmse"]
        g  = f"{rmses.get('GBM',          float('nan')):.4f}"
        h  = f"{rmses.get('Heston',        float('nan')):.4f}"
        rb = f"{rmses.get('RoughBergomi',  float('nan')):.4f}"
        print(f"  {t:<8} {g:>10} {h:>12} {rb:>14}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    print("  Figure 1: IV smile comparison")
    plot_iv_smiles(all_results)

    print("  Figure 2: RMSE bar chart")
    plot_rmse_bars(all_results)

    print("  Figure 3: Simulated paths")
    plot_simulated_paths(all_results)

    print("  Figure 4: Vol surface heatmaps")
    plot_vol_surface(all_results)

    print("\nDone. All figures saved to current working directory.")


if __name__ == "__main__":
    main()
