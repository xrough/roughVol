import numpy as np
import pytest

from roughvol.types import PathBundle
from roughvol.instruments.asian import AsianArithmeticOption


def make_paths_linear():
    # Spot is linear in time on each interval so linear interpolation is exact
    t = np.array([0.0, 1.0, 2.0])
    spot = np.array([
        [100.0, 110.0, 120.0],  # increases by 10 per unit time
        [100.0,  90.0,  80.0],  # decreases by 10 per unit time
    ])
    return PathBundle(t=t, state={"spot": spot})


def test_asian_offgrid_linear_interp():
    paths = make_paths_linear()
    # Observe at 0.5 and 1.5 (off-grid)
    obs = np.array([0.5, 1.5])
    inst = AsianArithmeticOption(maturity=2.0, strike=100.0, callput="call", obs_times=obs, interp="linear")
    payoff = inst.payoff(paths)

    # Path1: S(0.5)=105, S(1.5)=115, avg=110 -> payoff=10
    # Path2: S(0.5)=95,  S(1.5)=85,  avg=90  -> payoff=0
    assert np.allclose(payoff, np.array([10.0, 0.0]))


def test_asian_offgrid_previous_interp():
    paths = make_paths_linear()
    obs = np.array([0.5, 1.5])
    inst = AsianArithmeticOption(maturity=2.0, strike=100.0, callput="call", obs_times=obs, interp="previous")
    payoff = inst.payoff(paths)

    # "previous" uses left endpoint values:
    # at 0.5 -> t=0 spot; at 1.5 -> t=1 spot
    # Path1: avg=(100+110)/2=105 -> payoff=5
    # Path2: avg=(100+ 90)/2=95  -> payoff=0
    assert np.allclose(payoff, np.array([5.0, 0.0]))


def test_spot_at_rejects_out_of_range():
    paths = make_paths_linear()
    with pytest.raises(ValueError):
        _ = paths.spot_at(np.array([-0.1]), method="linear")
    with pytest.raises(ValueError):
        _ = paths.spot_at(np.array([2.1]), method="linear")
