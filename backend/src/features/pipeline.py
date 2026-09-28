"""Leakage-safe feature engineering for hierarchical CPU telemetry."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from src.core.config import DEFAULT_CONFIG, ForecastConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelemetrySchema:
    """Column contract for incoming telemetry."""

    key_columns: Tuple[str, ...] = ("box", "system", "service_class")
    timestamp_column: str = "timestamp"
    target_column: str = "cpu_usage"

    def __post_init__(self) -> None:
        keys = tuple(self.key_columns)
        if not keys:
            raise ValueError("key_columns must contain at least one column.")
        object.__setattr__(self, "key_columns", keys)

    @property
    def required_columns(self) -> List[str]:
        return [*self.key_columns, self.timestamp_column, self.target_column]


class CyclicSpec(NamedTuple):
    """Cyclic calendar component: output prefix, DatetimeIndex attribute, period."""

    prefix: str
    attribute: str
    period: int


DEFAULT_CYCLES: Tuple[CyclicSpec, ...] = (
    CyclicSpec("hour", "hour", 24),
    CyclicSpec("dow", "dayofweek", 7),
)


class BaseFeatureTransformer(ABC):
    """Contract for per-series feature transformers.

    Implementations receive a single, time-sorted, regular-grid series and
    return only the new feature columns, indexed identically to the input.
    """

    def fit(self, frame: pd.DataFrame) -> "BaseFeatureTransformer":
        return self

    @abstractmethod
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return feature columns aligned to `frame.index`."""

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Ordered names of the columns produced by `transform`."""

    @property
    def warmup(self) -> int:
        """Leading rows per series that cannot be computed."""
        return 0

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


class LagFeatures(BaseFeatureTransformer):
    """Auto-regressive lags anchored to the forecast origin.

    Lag `k` is the k-th most recent observation available at the forecast
    origin `t - horizon`, i.e. `y.shift(k + horizon - 1)`. Columns are named by
    effective shift, so `lag_1` at horizon 1 is `y[t-1]`.
    """

    def __init__(self, target_column: str, lags: Iterable[int], horizon: int) -> None:
        self.target_column = target_column
        self.horizon = int(horizon)
        self._shifts = [int(k) + self.horizon - 1 for k in lags]

    @property
    def feature_names(self) -> List[str]:
        return [f"lag_{s}" for s in self._shifts]

    @property
    def warmup(self) -> int:
        return max(self._shifts)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        y = frame[self.target_column]
        return pd.DataFrame({f"lag_{s}": y.shift(s) for s in self._shifts}, index=frame.index)


class RollingFeatures(BaseFeatureTransformer):
    """Rolling mean and standard deviation over a window ending at the forecast origin.

    The series is shifted by `horizon` before windowing, so `y[t]` (and anything
    after `t - horizon`) never enters its own summary. Windows are strict:
    partial windows yield NaN.
    """

    def __init__(self, target_column: str, windows: Iterable[int], horizon: int) -> None:
        self.target_column = target_column
        self.horizon = int(horizon)
        self.windows = [int(w) for w in windows]

    @property
    def feature_names(self) -> List[str]:
        return [f"roll_{stat}_{w}" for w in self.windows for stat in ("mean", "std")]

    @property
    def warmup(self) -> int:
        return self.horizon + max(self.windows) - 1

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        shifted = frame[self.target_column].shift(self.horizon)
        cols = {}
        for w in self.windows:
            window = shifted.rolling(window=w, min_periods=w)
            cols[f"roll_mean_{w}"] = window.mean()
            cols[f"roll_std_{w}"] = window.std()
        return pd.DataFrame(cols, index=frame.index)[self.feature_names]


class CyclicTimeFeatures(BaseFeatureTransformer):
    """Sine/cosine encoding of periodic calendar components.

    Calendar terms derive from the target timestamp, which is known in advance
    and carries no target information.
    """

    def __init__(
        self,
        timestamp_column: str,
        specs: Sequence[CyclicSpec] = DEFAULT_CYCLES,
    ) -> None:
        self.timestamp_column = timestamp_column
        self.specs = tuple(specs)

    @property
    def feature_names(self) -> List[str]:
        return [f"{s.prefix}_{fn}" for s in self.specs for fn in ("sin", "cos")]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        index = pd.DatetimeIndex(frame[self.timestamp_column])
        cols = {}
        for spec in self.specs:
            values = np.asarray(getattr(index, spec.attribute), dtype="float64")
            angle = 2.0 * np.pi * values / spec.period
            cols[f"{spec.prefix}_sin"] = np.sin(angle)
            cols[f"{spec.prefix}_cos"] = np.cos(angle)
        return pd.DataFrame(cols, index=frame.index)[self.feature_names]


class FeaturePipeline:
    """Partitions telemetry by series key and applies feature transformers per series.

    Output contains the schema columns plus engineered features. Columns outside
    the schema are dropped. Rows without an observed target (including grid
    gap-fill rows) are excluded from the output; their effect persists as NaN
    in downstream lag/rolling features, which XGBoost handles natively.
    """

    def __init__(
        self,
        config: ForecastConfig = DEFAULT_CONFIG,
        schema: Optional[TelemetrySchema] = None,
        freq: str = "h",
        transformers: Optional[Sequence[BaseFeatureTransformer]] = None,
    ) -> None:
        self.config = config
        self.schema = schema or TelemetrySchema()
        self.freq = freq
        self.transformers: List[BaseFeatureTransformer] = (
            list(transformers) if transformers is not None else self._default_transformers()
        )
        self.feature_names_: Optional[List[str]] = None

    def _default_transformers(self) -> List[BaseFeatureTransformer]:
        target = self.schema.target_column
        horizon = self.config.forecast_horizon
        return [
            LagFeatures(target, self.config.lag_windows, horizon),
            RollingFeatures(target, self.config.rolling_windows, horizon),
            CyclicTimeFeatures(self.schema.timestamp_column),
        ]

    @property
    def warmup_rows(self) -> int:
        return max((t.warmup for t in self.transformers), default=0)

    def fit(self, frame: pd.DataFrame) -> "FeaturePipeline":
        prepared = self._prepare(frame)
        for transformer in self.transformers:
            transformer.fit(prepared)
        names = [n for t in self.transformers for n in t.feature_names]
        if len(names) != len(set(names)):
            raise ValueError("Feature names must be unique across transformers.")
        self.feature_names_ = names
        return self

    def transform(self, frame: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise NotFittedError("FeaturePipeline must be fitted before transform.")

        schema = self.schema
        prepared = self._prepare(frame)
        parts: List[pd.DataFrame] = []
        filled_total = 0

        for _, part in prepared.groupby(list(schema.key_columns), sort=False, observed=True):
            series, filled = self._regularize(part)
            filled_total += filled
            features = pd.concat([t.transform(series) for t in self.transformers], axis=1)
            out = pd.concat([series, features], axis=1)
            if drop_warmup:
                out = out.iloc[self.warmup_rows:]
            parts.append(out.dropna(subset=[schema.target_column]))

        if filled_total:
            logger.warning("Gap-filled %d missing timestamps across series.", filled_total)
        if not parts:
            return pd.DataFrame(columns=[*schema.required_columns, *self.feature_names_])
        return pd.concat(parts, ignore_index=True)

    def fit_transform(self, frame: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
        return self.fit(frame).transform(frame, drop_warmup=drop_warmup)

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate schema, snap timestamps to the grid, reject duplicates, sort."""
        s = self.schema
        missing = [c for c in s.required_columns if c not in frame.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        data = frame[s.required_columns].copy()
        if not pd.api.types.is_datetime64_any_dtype(data[s.timestamp_column]):
            raise TypeError(f"'{s.timestamp_column}' must be a datetime column.")
        if not pd.api.types.is_numeric_dtype(data[s.target_column]):
            raise TypeError(f"'{s.target_column}' must be numeric.")
        if data[[*s.key_columns, s.timestamp_column]].isna().any().any():
            raise ValueError("Series keys and timestamps must not contain nulls.")

        data[s.timestamp_column] = data[s.timestamp_column].dt.floor(self.freq)
        data[s.target_column] = data[s.target_column].astype("float64")

        key_and_time = [*s.key_columns, s.timestamp_column]
        duplicates = int(data.duplicated(subset=key_and_time).sum())
        if duplicates:
            raise ValueError(
                f"{duplicates} duplicate (key, timestamp) rows found at freq='{self.freq}'."
            )
        return data.sort_values(key_and_time).reset_index(drop=True)

    def _regularize(self, part: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Reindex one series to a gap-free grid so row offsets equal time offsets."""
        s = self.schema
        ts = part[s.timestamp_column]
        grid = pd.date_range(ts.min(), ts.max(), freq=self.freq)
        series = (
            part.set_index(s.timestamp_column)
            .drop(columns=list(s.key_columns))
            .reindex(grid)
            .rename_axis(s.timestamp_column)
            .reset_index()
        )
        for column in s.key_columns:
            series[column] = part[column].iloc[0]
        return series[s.required_columns], len(grid) - len(part)


def build_feature_pipeline(
    config: ForecastConfig = DEFAULT_CONFIG,
    schema: Optional[TelemetrySchema] = None,
    freq: str = "h",
) -> FeaturePipeline:
    """Construct the standard lag + rolling + cyclic pipeline."""
    return FeaturePipeline(config=config, schema=schema, freq=freq)