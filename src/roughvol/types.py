from __future__ import annotations

from dataclasses import dataclass, field 

from typing import Protocol, runtime_checkable, Mapping, Any, Callable, Literal
import numpy as np

ArrayF = np.ndarray # 简写ndarray

# ============================================================================
# Input containers
# ============================================================================

@dataclass(frozen=True)
class SimConfig:
    '''
    Numerical/simulation configuration. 
    Input includes n_paths, maturity, n_steps/time_grid, seed, antithetic, scheme, store_paths and metadata.'''
    n_paths: int
    maturity: float

    n_steps: int | None = None
    time_grid: ArrayF | None = None

    seed: int | None = None
    
    # Extensions
    
    antithetic: bool = False
    scheme: str = "euler"  # e.g. "euler", "qe", "hybrid"
    store_paths: bool = True  # allow terminal-only mode

    metadata: Mapping[str, Any] = field(default_factory=dict) # dict储存额外信息。

    def grid(self) -> ArrayF:
        if self.time_grid is not None:
            t = np.asarray(self.time_grid, dtype=float)
            if t.ndim != 1:
                raise ValueError("time_grid must be 1D")
            return t

        if self.n_steps is None:
            raise ValueError("Provide either time_grid or n_steps")
        if self.maturity == 0.0:
            return np.array([0.0], dtype=float)

        return np.linspace(0.0, self.maturity, self.n_steps + 1)

@dataclass(frozen=True)
class MarketData:
    '''Economic environment.'''
    spot: float
    rate: float = 0.0
    div_yield: float = 0.0

    # general models
    # callable是确定对象是否是一个函数，而Callable是static typing，在此处定义一些method为Callable。
    discount_curve: Any | None = None
    forward_variance_curve: Callable[[ArrayF], ArrayF] | None = None # default to be None. 
    metadata: Mapping[str, Any] = field(default_factory=dict) # 我们允许通过dict记录一些额外的信息。


# ============================================================================
# Model output container
# ============================================================================

@dataclass(frozen=True)
class PathBundle:
    '''
    Standardized model output container.
    - spot shape: (n_paths, n_times), economic state variable along paths (spot S_t, volatility v_t etc.).
      - therefore should align with time grid t.
    - t: time shape: (n_times,)
    - variance shape (if present): (n_paths, n_times)
    '''
    t: ArrayF
    state: Mapping[str, ArrayF]
    
    '''
    This object may optionally contain a dictionary of extra arrays, indexed by string names.
    '''
    
    extras: Mapping[str, ArrayF] = field(default_factory=dict) # numerical arrays produced by the simulation (path-level data).
    metadata: Mapping[str, Any] = field(default_factory=dict) # descriptive context about how/why the simulation was produced.
    
    def __post_init__(self) -> None: # __post_init__ runs after dataclass is initialized.
        # Validate time grid and state arrays alignment
        if "spot" not in self.state:
            raise ValueError("PathBundle.state must contain 'spot'")

        spot = np.asarray(self.state["spot"])
        t = np.asarray(self.t)

        if spot.ndim != 2:
            raise ValueError("state['spot'] must be 2D (n_paths, n_times)")
        if spot.shape[1] != t.shape[0]:
            raise ValueError("spot arrays must align with time grid")

        for k, v in self.state.items(): # items() returns key-value pairs.
            arr = np.asarray(v)
            if arr.shape != spot.shape:
                raise ValueError(f"state['{k}'] shape mismatch")

    # properties are snapshots of the object state, read-only attributes computed on-the-fly.
    @property
    def spot(self) -> ArrayF:
        return self.state["spot"]

    @property
    def spot_T(self) -> ArrayF:
        return self.spot[:, -1]

    @property
    def n_paths(self) -> int:
        return int(self.spot.shape[0])

    @property
    def n_times(self) -> int:
        return int(self.spot.shape[1])
    
    # for extension: get state by name. 
    def get(self, name: str) -> ArrayF:
        return self.state[name]
    
    # interpolate spot at arbitrary times in case continuum evaluation is needed.
    def spot_at(
        self,
        times: ArrayF,
        *,
        method: Literal["ladder", "linear"] = "linear",
        tol: float = 1e-12,
    ) -> ArrayF:
        '''
        Evaluate spot S(t) at arbitrary times by interpreting the discrete path
        as a continuous-time function via interpolation.

        Parameters
        ----------
        times:
            1D array of query times in [t[0], t[-1]].
        method:
            "ladder": left-continuous piecewise-constant (ladder) interpolation
            "linear": piecewise-linear interpolation
        tol:
            tolerance for boundary handling (e.g., t[-1]).

        Returns
        -------
        ArrayF with shape (n_paths, n_times_query)
        '''
        t_grid = np.asarray(self.t, dtype=float)
        S = np.asarray(self.spot, dtype=float)  # (n_paths, n_times_grid)
        q = np.asarray(times, dtype=float)

        if q.ndim != 1:
            raise ValueError("times must be a 1D array")
        if t_grid.ndim != 1:
            raise ValueError("PathBundle.t must be a 1D array")
        if S.ndim != 2 or S.shape[1] != t_grid.shape[0]:
            raise ValueError("PathBundle.spot must be (n_paths, n_times) aligned with t")

        t0 = float(t_grid[0])
        t1 = float(t_grid[-1])

        # Allow tiny tolerance at boundaries
        if np.any(q < t0 - tol) or np.any(q > t1 + tol):
            raise ValueError(f"Query times must lie within [{t0}, {t1}] (tol={tol}).")

        # Clip within bounds for stable indexing (esp. q == t1)
        q = np.clip(q, t0, t1)

        # For each q, find right interval index i such that t[i] <= q <= t[i+1]
        # searchsorted returns insertion point; subtract 1 gives left index.
        idx = np.searchsorted(t_grid, q, side="right") - 1
        idx = np.clip(idx, 0, len(t_grid) - 2)  # last interval is [n-2, n-1]

        if method == "ladder":
            # left-continuous: S(q) = S(t[idx])
            return S[:, idx]

        if method == "linear":
            tL = t_grid[idx]          # (m,)
            tR = t_grid[idx + 1]      # (m,)
            SL = S[:, idx]            # (n_paths, m)
            SR = S[:, idx + 1]        # (n_paths, m)

            denom = (tR - tL)
            # denom should be > 0 if grid is strictly increasing, but guard anyway
            if np.any(denom <= 0):
                raise ValueError("Time grid must be strictly increasing for linear interpolation.")

            w = (q - tL) / denom       # (m,)
            return SL + (SR - SL) * w  # broadcasts w across paths

        raise ValueError(f"Unknown method: {method!r}. Use 'ladder' or 'linear'.")

# ============================================================================
# Pricing output
# ============================================================================

@dataclass(frozen=True)
class PriceResult:
    '''Standard output container for pricing engines.'''
    price: float
    stderr: float # MC standard error
    ci95: tuple # MC 95% confidence interval
    n_paths: int
    n_steps: int
    seed: int | None = None
    
    metadata: Mapping[str, Any] = field(default_factory=dict)

# ============================================================================
# Protocols (capability boundaries)
# ============================================================================

@runtime_checkable
class Instrument(Protocol): # Protocol规定class至少要满足的条件。
    '''Contract: an instrument defines payoff, may depend on the full path.'''

    maturity: float  #结算时间

    def payoff(self, path: PathBundle) -> ArrayF:
        ...

@runtime_checkable
class TerminalInstrument(Protocol):
    '''
    Legacy / terminal-only payoff.
    '''
    maturity: float

    def payoff_terminal(self, spot_T: ArrayF) -> ArrayF:
        ...

@runtime_checkable
class PathModel(Protocol):
    '''
    Model that can simulate paths.
    '''
    def simulate_paths(
        self,
        *,
        market: MarketData,
        sim: SimConfig,
        rng: np.random.Generator,
    ) -> PathBundle:
        ...

# ============================================================================
# Adapter helpers
# ============================================================================
# In this version, we have different interfaces, and adpater serves as temporary glue. 

def compute_payoff(
    instrument: Instrument | TerminalInstrument,
    paths: PathBundle,
) -> ArrayF:
    '''
    Backward-compatible payoff resolution.
    '''
    
    ''' payoff_paths means:
        Does instrument have (path) payoff?
        ├─ Yes → try path payoff
        │   ├─ Works → return result
        │   └─ TypeError → ignore, continue
        └─ No → try terminal payoff...
    '''
    payoff_paths = getattr(instrument, "payoff", None)
    if callable(payoff_paths):
        return np.asarray(payoff_paths(paths))

    payoff_terminal = getattr(instrument, "payoff_terminal", None)
    if callable(payoff_terminal):
        return np.asarray(payoff_terminal(paths.spot_T))

    raise TypeError(
        "Instrument must implement payoff(paths) or payoff_terminal(spot_T)"
    )


def make_rng(seed: int | None) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    # Use SeedSequence for robust, reproducible streams
    return np.random.default_rng(np.random.SeedSequence(seed))


def flat_discount_factor(rate: float, t: float) -> float:
    return float(np.exp(-rate * t))