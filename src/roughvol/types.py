'''
Type定义了class,其中最基本的包括
- PriceResult：定价输出, 
- instrument：衍生品及其payoff(定义为protocol保证灵活性), 
- PathModel定义各个随机过程。

types.py
└── Core architecture
    ├── Data containers (dataclasses: concrete schemas)
    │   ├── MarketData
    │   │   ├── spot: float
    │   │   ├── rate: float
    │   │   ├── div_yield: float
    │   │   ├── discount_curve: Any | None
    │   │   ├── forward_variance_curve: Callable[[ArrayF], ArrayF] | None
    │   │   └── metadata: Mapping[str, Any]
    │   │
    │   ├── SimConfig
    │   │   ├── n_paths: int
    │   │   ├── maturity: float
    │   │   ├── n_steps: int | None
    │   │   ├── time_grid: ArrayF | None
    │   │   ├── seed: int | None
    │   │   ├── antithetic: bool
    │   │   ├── scheme: str
    │   │   ├── store_paths: bool
    │   │   ├── metadata: Mapping[str, Any]
    │   │   └── method: grid() -> ArrayF
    │   │
    │   ├── PathBundle
    │   │   ├── t: ArrayF
    │   │   ├── state: Mapping[str, ArrayF]
    │   │   │   └── required key: "spot" -> ArrayF (n_paths, n_times)
    │   │   ├── extras: Mapping[str, ArrayF]
    │   │   ├── metadata: Mapping[str, Any]
    │   │   ├── method: __post_init__()  (validates invariants)
    │   │   └── properties (derived facts)
    │   │       ├── spot -> ArrayF
    │   │       ├── spot_T -> ArrayF
    │   │       ├── n_paths -> int
    │   │       ├── n_times -> int
    │   │       └── get(name: str) -> ArrayF
    │   │
    │   └── PriceResult
    │       ├── price: float
    │       ├── stderr: float
    │       ├── ci95: tuple[float, float]
    │       ├── n_paths: int
    │       ├── n_steps: int
    │       ├── seed: int | None
    │       └── metadata: Mapping[str, Any]
    │
    ├── Capability boundaries (Protocols: behavioral contracts)
    │   ├── Instrument
    │   │   ├── maturity: float
    │   │   └── payoff(paths: PathBundle) -> ArrayF
    │   │
    │   ├── TerminalInstrument (legacy support)
    │   │   ├── maturity: float
    │   │   └── payoff_terminal(spot_T: ArrayF) -> ArrayF
    │   │
    │   └── PathModel
    │       └── simulate_paths(market: MarketData, sim: SimConfig, rng: Generator) -> PathBundle
    │
    └── Adapter / utility functions (glue)
        ├── compute_payoff(instrument, paths) -> ArrayF
        │   ├── tries Instrument.payoff(paths)
        │   ├── else tries TerminalInstrument.payoff_terminal(spot_T)
        │   └── else tries legacy payoff(spot_T)
        ├── make_rng(seed) -> Generator
        └── flat_discount_factor(rate, t) -> float

'''

from __future__ import annotations #避免循环reference

from dataclasses import dataclass, field # 数据载体：class, field指代属性。

# Protocol是特殊的class，只规定了必要的元素。
# Mapping提示type是dictionary-like，但比dict安全。
# Any: disable type checking here.

from typing import Protocol, runtime_checkable, Mapping, Any, Callable
import numpy as np

ArrayF = np.ndarray # 简写ndarray

# ============================================================================
# Input containers
# ============================================================================

@dataclass(frozen=True)
class SimConfig:
    '''Numerical/simulation configuration (input how to simulate).'''
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

    Conventions:
    - spot shape: (n_paths, n_times)
    - t: time shape: (n_times,)
    - variance shape (if present): (n_paths, n_times)
    '''
    t: ArrayF
    state: Mapping[str, ArrayF]
    
    '''This object may optionally contain a dictionary of extra arrays, indexed by string names.
    If the user does not provide any extras, create an empty dictionary for this instance.'''
    
    # 这里不能写extra={}，不然所有class中的对象会共有这个dict.
    # algorithms should not depend on extras!
    extras: Mapping[str, ArrayF] = field(default_factory=dict) # numerical arrays produced by the simulation (path-level data).
    metadata: Mapping[str, Any] = field(default_factory=dict) # descriptive context about how/why the simulation was produced.
    
    def __post_init__(self) -> None: # __post_init__运行在obj创建之后。
        if "spot" not in self.state:
            raise ValueError("PathBundle.state must contain 'spot'")

        spot = np.asarray(self.state["spot"])
        t = np.asarray(self.t)

        if spot.ndim != 2:
            raise ValueError("state['spot'] must be 2D (n_paths, n_times)")
        if spot.shape[1] != t.shape[0]:
            raise ValueError("state arrays must align with time grid")

        for k, v in self.state.items():
            arr = np.asarray(v)
            if arr.shape != spot.shape:
                raise ValueError(f"state['{k}'] shape mismatch")

    #Property是class中包含的数据，而非behavior，但是又不希望class存储这些data（snapshot）。
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

    def get(self, name: str) -> ArrayF:
        return self.state[name]

# ============================================================================
# Pricing output
# ============================================================================

@dataclass(frozen=True)
class PriceResult:
    '''Standard output container for pricing engines.'''
    price: float
    stderr: float # MC标准误差
    ci95: tuple # MC标准误差
    n_paths: int
    n_steps: int
    seed: int | None = None
    
    metadata: Mapping[str, Any] = field(default_factory=dict)

# ============================================================================
# Protocols (capability boundaries)
# 即规定了几个Components最少需要满足哪些条件。
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
        Does instrument have payoff?
        ├─ Yes → try payoff(paths)
        │   ├─ Works → return result
        │   └─ TypeError → ignore, continue
        └─ No → continue
    '''
    payoff_paths = getattr(instrument, "payoff", None)
    if callable(payoff_paths):
        try:
            return np.asarray(payoff_paths(paths))
        except TypeError: # except的代码执行若try部分报错。
            pass # 即若是TypeError则忽略并继续。

    payoff_terminal = getattr(instrument, "payoff_terminal", None)
    if callable(payoff_terminal):
        return np.asarray(payoff_terminal(paths.spot_T))

    legacy = getattr(instrument, "payoff", None)
    if callable(legacy):
        return np.asarray(legacy(paths.spot_T))

    raise TypeError(
        "Instrument must implement payoff(paths) or payoff_terminal(spot_T)"
    )


def make_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def flat_discount_factor(rate: float, t: float) -> float:
    return float(np.exp(-rate * t))