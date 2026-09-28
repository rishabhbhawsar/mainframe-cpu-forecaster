"""Centralized configuration registry for the forecasting pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Dict, Iterable, Tuple


def _normalize_windows(name: str, values: Iterable[int], minimum: int) -> Tuple[int, ...]:
    """Validate a window collection and return it as a sorted, de-duplicated tuple."""
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of integers.") from exc

    if not raw:
        raise ValueError(f"{name} must contain at least one window.")
    if any(isinstance(w, bool) or not isinstance(w, Integral) for w in raw):
        raise TypeError(f"{name} must contain only integers.")

    windows = tuple(sorted({int(w) for w in raw}))
    if windows[0] < minimum:
        raise ValueError(f"{name} values must be >= {minimum}; got {windows[0]}.")
    return windows


def _require_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}.")
    return int(value)


def _require_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    return float(value)


@dataclass(frozen=True)
class ForecastConfig:
    """Immutable container for time-series, validation, and drift parameters.

    Attributes:
        lag_windows: Lookback offsets (in steps) used to build auto-regressive features.
        rolling_windows: Window sizes (in steps) for rolling statistics.
        forecast_horizon: Steps ahead to predict.
        min_history_points: Minimum observations required to train a series.
        drift_alpha_threshold: Significance level for the Kolmogorov-Smirnov test.
        psi_threshold: Population Stability Index level flagged as significant drift.
    """

    lag_windows: Tuple[int, ...] = (1, 2, 3, 4, 24)
    rolling_windows: Tuple[int, ...] = (3, 6, 12, 24)
    forecast_horizon: int = 1
    min_history_points: int = 168
    drift_alpha_threshold: float = 0.05
    psi_threshold: float = 0.2

    def __post_init__(self) -> None:
        lags = _normalize_windows("lag_windows", self.lag_windows, minimum=1)
        rolls = _normalize_windows("rolling_windows", self.rolling_windows, minimum=2)
        horizon = _require_int("forecast_horizon", self.forecast_horizon, minimum=1)
        min_history = _require_int("min_history_points", self.min_history_points, minimum=1)
        alpha = _require_float("drift_alpha_threshold", self.drift_alpha_threshold)
        psi = _require_float("psi_threshold", self.psi_threshold)

        if not 0.0 < alpha < 1.0:
            raise ValueError(f"drift_alpha_threshold must be in (0, 1); got {alpha}.")
        if psi <= 0.0:
            raise ValueError(f"psi_threshold must be > 0; got {psi}.")

        # Frozen dataclass: normalized values are written back via object.__setattr__.
        object.__setattr__(self, "lag_windows", lags)
        object.__setattr__(self, "rolling_windows", rolls)
        object.__setattr__(self, "forecast_horizon", horizon)
        object.__setattr__(self, "min_history_points", min_history)
        object.__setattr__(self, "drift_alpha_threshold", alpha)
        object.__setattr__(self, "psi_threshold", psi)

        required = self.max_lookback + self.forecast_horizon
        if self.min_history_points <= required:
            raise ValueError(
                f"min_history_points ({self.min_history_points}) must exceed "
                f"max_lookback + forecast_horizon ({required})."
            )

    @property
    def max_lookback(self) -> int:
        """Longest history span (in steps) any feature depends on."""
        return max(max(self.lag_windows), max(self.rolling_windows))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary for artifact metadata."""
        payload = asdict(self)
        payload["lag_windows"] = list(self.lag_windows)
        payload["rolling_windows"] = list(self.rolling_windows)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ForecastConfig":
        """Rebuild a configuration from a dictionary produced by `to_dict`."""
        return cls(**payload)


DEFAULT_CONFIG = ForecastConfig()