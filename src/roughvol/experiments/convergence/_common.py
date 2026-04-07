from __future__ import annotations

from roughvol.experiments.convergence.run_rough_vol_convergence import (
    run_rough_bergomi_convergence,
    run_rough_heston_convergence,
)


def build_results() -> dict:
    return run_rough_bergomi_convergence()


def build_results_rh() -> dict:
    return run_rough_heston_convergence()
