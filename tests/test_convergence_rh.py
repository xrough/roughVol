from __future__ import annotations

import pytest

from roughvol.experiments.convergence.run_rough_vol import compute_rh_scheme_diagnostics


def test_compute_rh_scheme_diagnostics_uses_cross_scheme_spread_not_reference():
    schemes = {
        "volterra-euler": {
            "steps": [8, 16],
            "prices": [10.0, 9.8],
            "stderrs": [0.05, 0.04],
        },
        "markovian-lift": {
            "steps": [8, 16],
            "prices": [9.9, 9.7],
            "stderrs": [0.04, 0.03],
        },
        "bayer-breneis": {
            "steps": [8, 16],
            "prices": [10.2, 9.6],
            "stderrs": [0.06, 0.02],
        },
    }

    diagnostics = compute_rh_scheme_diagnostics(schemes)

    assert diagnostics["steps"] == [8, 16]
    assert diagnostics["pairwise_spread"] == pytest.approx([0.3, 0.2])
    assert diagnostics["median_price"] == pytest.approx([10.0, 9.7])
    assert diagnostics["max_pairwise_noise"][0] > 0.0
    assert diagnostics["max_pairwise_zscore"][0] > 1.0
