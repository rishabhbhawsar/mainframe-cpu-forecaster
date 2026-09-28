"""Forward-chaining validation and parallel per-series XGBoost training."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas.tseries.frequencies import to_offset
from xgboost import XGBRegressor

from src.core.config import DEFAULT_CONFIG, ForecastConfig
from src.features.pipeline import TelemetrySchema

logger = logging.getLogger(__name__)

SeriesKey = Tuple[Any, ...]

DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": 1,
}


class InsufficientHistoryError(ValueError):
    """Raised when a series cannot support the requested validation layout."""


class ForwardChainingSplit:
    """Expanding-window splitter with horizon-based purging.

    Validation blocks are contiguous, disjoint, and anchored to the end of the
    series. Training rows for a fold are those whose target was observable at
    the forecast origin of the first validation row:
    `timestamp <= val_start - horizon * step`. Purging is timestamp-based, so
    it stays correct when rows are missing from the grid.
    """

    def __init__(
        self,
        n_splits: int,
        min_train_size: int,
        horizon: int,
        freq: str = "h",
        test_size: Optional[int] = None,
    ) -> None:
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.horizon = horizon
        self.test_size = test_size
        self._step = pd.Timedelta(to_offset(freq))

    def split(self, timestamps: Sequence[pd.Timestamp]) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        ts = pd.DatetimeIndex(timestamps)
        if ts.has_duplicates or not ts.is_monotonic_increasing:
            raise ValueError("Timestamps must be strictly increasing within a series.")

        n = len(ts)
        purge = self.horizon - 1
        available = n - self.min_train_size - purge
        test_size = self.test_size if self.test_size is not None else available // self.n_splits
        first_val = n - self.n_splits * test_size
        if test_size < 1 or first_val < self.min_train_size + purge:
            raise InsufficientHistoryError(
                f"{n} rows cannot support {self.n_splits} folds with "
                f"min_train_size={self.min_train_size}, horizon={self.horizon}."
            )

        for k in range(self.n_splits):
            start = first_val + k * test_size
            cutoff = ts[start] - self.horizon * self._step
            train_end = int(ts.searchsorted(cutoff, side="right"))
            yield np.arange(train_end), np.arange(start, start + test_size)


@dataclass(frozen=True)
class ValidationSummary:
    """Aggregate out-of-fold metrics for one series."""

    n_folds: int
    n_oof_rows: int
    mae: float
    rmse: float
    fold_mae: Tuple[float, ...]
    fold_rmse: Tuple[float, ...]
    naive_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "n_oof_rows": self.n_oof_rows,
            "mae": self.mae,
            "rmse": self.rmse,
            "fold_mae": list(self.fold_mae),
            "fold_rmse": list(self.fold_rmse),
            "naive_ratio": self.naive_ratio,
        }


@dataclass(frozen=True)
class QualityGate:
    """Export criteria applied to pooled out-of-fold metrics.

    `max_mae` / `max_rmse` are in target units. `max_naive_ratio` bounds model MAE
    relative to the persistence forecast; 1.0 rejects models that do not beat it.
    Set any threshold to None to disable it.
    """

    max_mae: Optional[float] = 10.0
    max_rmse: Optional[float] = 15.0
    max_naive_ratio: Optional[float] = 1.0

    def __post_init__(self) -> None:
        for name in ("max_mae", "max_rmse", "max_naive_ratio"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 or None; got {value}.")

    def evaluate(self, summary: ValidationSummary) -> List[str]:
        """Return failure reasons; an empty list means the gate is passed."""
        if not (np.isfinite(summary.mae) and np.isfinite(summary.rmse)):
            return ["non-finite validation metrics"]

        failures: List[str] = []
        if self.max_mae is not None and summary.mae > self.max_mae:
            failures.append(f"mae {summary.mae:.4f} > {self.max_mae:.4f}")
        if self.max_rmse is not None and summary.rmse > self.max_rmse:
            failures.append(f"rmse {summary.rmse:.4f} > {self.max_rmse:.4f}")
        if (
            self.max_naive_ratio is not None
            and summary.naive_ratio is not None
            and summary.naive_ratio > self.max_naive_ratio
        ):
            failures.append(
                f"naive_ratio {summary.naive_ratio:.4f} > {self.max_naive_ratio:.4f}"
            )
        return failures


@dataclass(frozen=True)
class TrainingConfig:
    """Validation, model, and baseline-capture parameters."""

    n_splits: int = 5
    min_train_size: Optional[int] = None
    test_size: Optional[int] = None
    freq: str = "h"
    xgb_params: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_XGB_PARAMS))
    gate: QualityGate = field(default_factory=QualityGate)
    baseline_bins: int = 10
    baseline_sample_size: int = 2000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2.")
        if self.min_train_size is not None and self.min_train_size < 1:
            raise ValueError("min_train_size must be >= 1.")
        if self.test_size is not None and self.test_size < 1:
            raise ValueError("test_size must be >= 1.")
        if self.baseline_bins < 2:
            raise ValueError("baseline_bins must be >= 2.")
        if self.baseline_sample_size < 1:
            raise ValueError("baseline_sample_size must be >= 1.")
        to_offset(self.freq)


@dataclass(frozen=True, eq=False)
class FeatureBaseline:
    """Training-time reference distribution of one feature.

    `bin_edges` are interior quantile edges. A value `x` belongs to bin
    `np.searchsorted(bin_edges, x, side="right")`, so `bin_proportions` has
    `len(bin_edges) + 1` entries. `sample` is a bounded reference set for KS.
    """

    name: str
    n_obs: int
    null_fraction: float
    bin_edges: np.ndarray
    bin_proportions: np.ndarray
    sample: np.ndarray


def build_baselines(
    features: pd.DataFrame,
    n_bins: int = 10,
    sample_size: int = 2000,
    seed: int = 42,
) -> Dict[str, FeatureBaseline]:
    """Capture per-feature reference distributions from training features."""
    rng = np.random.default_rng(seed)
    baselines: Dict[str, FeatureBaseline] = {}

    for name in features.columns:
        raw = features[name].to_numpy(dtype="float64")
        values = raw[np.isfinite(raw)]
        if values.size == 0:
            continue

        interior = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(values, interior))
        counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=edges.size + 1)
        sample = (
            rng.choice(values, size=sample_size, replace=False)
            if values.size > sample_size
            else values.copy()
        )
        baselines[name] = FeatureBaseline(
            name=name,
            n_obs=int(values.size),
            null_fraction=float(1.0 - values.size / raw.size),
            bin_edges=edges,
            bin_proportions=counts / counts.sum(),
            sample=sample,
        )
    return baselines


class TrainingStatus(str, Enum):
    PASSED = "passed"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, eq=False)
class SeriesResult:
    """Training outcome for one series. `model` is set only if the gate passed."""

    key: SeriesKey
    status: TrainingStatus
    n_rows: int
    summary: Optional[ValidationSummary] = None
    model: Optional[XGBRegressor] = None
    baselines: Mapping[str, FeatureBaseline] = field(default_factory=dict)
    feature_names: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()

    @property
    def exportable(self) -> bool:
        return self.status is TrainingStatus.PASSED and self.model is not None


@dataclass(frozen=True, eq=False)
class TrainingReport:
    """Collection of per-series results with export and audit helpers."""

    results: Tuple[SeriesResult, ...]
    config: ForecastConfig
    key_columns: Tuple[str, ...]

    @property
    def exportable(self) -> List[SeriesResult]:
        return [r for r in self.results if r.exportable]

    @property
    def models(self) -> Dict[SeriesKey, XGBRegressor]:
        return {r.key: r.model for r in self.exportable}

    @property
    def baselines(self) -> Dict[SeriesKey, Mapping[str, FeatureBaseline]]:
        return {r.key: r.baselines for r in self.exportable}

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            s = r.summary
            rows.append(
                {
                    **dict(zip(self.key_columns, r.key)),
                    "status": r.status.value,
                    "n_rows": r.n_rows,
                    "n_folds": s.n_folds if s else None,
                    "mae": s.mae if s else np.nan,
                    "rmse": s.rmse if s else np.nan,
                    "naive_ratio": s.naive_ratio if s else None,
                    "reasons": "; ".join(r.reasons),
                }
            )
        return pd.DataFrame(rows)

    def export(self, directory: str | Path) -> List[Path]:
        """Persist gate-passing artifacts and a full audit summary."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)

        written: List[Path] = []
        for r in self.exportable:
            path = target / _artifact_name(r.key)
            joblib.dump(
                {
                    "key": r.key,
                    "model": r.model,
                    "baselines": dict(r.baselines),
                    "feature_names": r.feature_names,
                    "summary": r.summary.to_dict() if r.summary else None,
                    "config": self.config.to_dict(),
                },
                path,
            )
            written.append(path)

        self.summary_frame().to_csv(target / "training_summary.csv", index=False)
        return written


def _artifact_name(key: SeriesKey) -> str:
    slug = re.sub(r"[^\w.-]+", "_", "__".join(map(str, key)))
    digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}.joblib"


def _as_key(key: Any) -> SeriesKey:
    return key if isinstance(key, tuple) else (key,)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _naive_ratio(y_true: np.ndarray, y_pred: np.ndarray, naive: np.ndarray) -> Optional[float]:
    """Model MAE divided by persistence MAE on rows where persistence exists."""
    mask = np.isfinite(naive)
    if not mask.any():
        return None
    baseline = _mae(y_true[mask], naive[mask])
    if baseline <= 0.0:
        return None
    return _mae(y_true[mask], y_pred[mask]) / baseline


def _make_model(training: TrainingConfig) -> XGBRegressor:
    params = dict(training.xgb_params)
    params.setdefault("random_state", training.seed)
    return XGBRegressor(**params)


def _fit_series(
    key: SeriesKey,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    schema: TelemetrySchema,
    config: ForecastConfig,
    training: TrainingConfig,
) -> SeriesResult:
    data = frame.sort_values(schema.timestamp_column)
    data = data[np.isfinite(data[schema.target_column].to_numpy(dtype="float64"))]
    data = data.reset_index(drop=True)

    X = data[list(feature_columns)]
    y = data[schema.target_column].to_numpy(dtype="float64")
    timestamps = data[schema.timestamp_column]

    splitter = ForwardChainingSplit(
        n_splits=training.n_splits,
        min_train_size=training.min_train_size or config.min_history_points,
        horizon=config.forecast_horizon,
        freq=training.freq,
        test_size=training.test_size,
    )

    naive_column = f"lag_{config.forecast_horizon}"
    naive_values = X[naive_column].to_numpy(dtype="float64") if naive_column in X.columns else None

    oof_true: List[np.ndarray] = []
    oof_pred: List[np.ndarray] = []
    oof_naive: List[np.ndarray] = []
    fold_mae: List[float] = []
    fold_rmse: List[float] = []

    for train_idx, val_idx in splitter.split(timestamps):
        model = _make_model(training)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = np.asarray(model.predict(X.iloc[val_idx]), dtype="float64")

        oof_true.append(y[val_idx])
        oof_pred.append(pred)
        if naive_values is not None:
            oof_naive.append(naive_values[val_idx])
        fold_mae.append(_mae(y[val_idx], pred))
        fold_rmse.append(_rmse(y[val_idx], pred))

    y_true = np.concatenate(oof_true)
    y_pred = np.concatenate(oof_pred)
    summary = ValidationSummary(
        n_folds=len(fold_mae),
        n_oof_rows=int(y_true.size),
        mae=_mae(y_true, y_pred),
        rmse=_rmse(y_true, y_pred),
        fold_mae=tuple(fold_mae),
        fold_rmse=tuple(fold_rmse),
        naive_ratio=(
            _naive_ratio(y_true, y_pred, np.concatenate(oof_naive)) if oof_naive else None
        ),
    )

    failures = training.gate.evaluate(summary)
    if failures:
        return SeriesResult(
            key=key,
            status=TrainingStatus.REJECTED,
            n_rows=len(data),
            summary=summary,
            feature_names=tuple(feature_columns),
            reasons=tuple(failures),
        )

    final_model = _make_model(training)
    final_model.fit(X, y)
    return SeriesResult(
        key=key,
        status=TrainingStatus.PASSED,
        n_rows=len(data),
        summary=summary,
        model=final_model,
        baselines=build_baselines(
            X, training.baseline_bins, training.baseline_sample_size, training.seed
        ),
        feature_names=tuple(feature_columns),
    )


def _train_series(
    key: SeriesKey,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    schema: TelemetrySchema,
    config: ForecastConfig,
    training: TrainingConfig,
) -> SeriesResult:
    """Worker entry point. Exceptions are captured so one series cannot abort the batch."""
    try:
        return _fit_series(key, frame, feature_columns, schema, config, training)
    except InsufficientHistoryError as exc:
        return SeriesResult(key, TrainingStatus.SKIPPED, len(frame), reasons=(str(exc),))
    except Exception as exc:  # noqa: BLE001
        return SeriesResult(
            key, TrainingStatus.FAILED, len(frame), reasons=(f"{type(exc).__name__}: {exc}",)
        )


def train_all_series(
    frame: pd.DataFrame,
    config: ForecastConfig = DEFAULT_CONFIG,
    schema: Optional[TelemetrySchema] = None,
    training: Optional[TrainingConfig] = None,
    feature_columns: Optional[Sequence[str]] = None,
    n_jobs: int = -1,
    verbose: int = 0,
) -> TrainingReport:
    """Train, validate, and gate one XGBoost model per series key in parallel.

    `frame` is the output of `FeaturePipeline.transform`. Any column outside the
    schema is treated as a feature unless `feature_columns` is given.
    """
    schema = schema or TelemetrySchema()
    training = training or TrainingConfig()

    missing = [c for c in schema.required_columns if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    features = (
        list(feature_columns)
        if feature_columns is not None
        else [c for c in frame.columns if c not in schema.required_columns]
    )
    if not features:
        raise ValueError("No feature columns found.")

    columns = [schema.timestamp_column, schema.target_column, *features]
    partitions = [
        (_as_key(key), part[columns])
        for key, part in frame.groupby(list(schema.key_columns), sort=False, observed=True)
    ]

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_train_series)(key, part, features, schema, config, training)
        for key, part in partitions
    )

    for r in results:
        if r.status is not TrainingStatus.PASSED:
            logger.warning("Series %s %s: %s", r.key, r.status.value, "; ".join(r.reasons))
    passed = sum(r.exportable for r in results)
    logger.info("Training complete: %d/%d series exportable.", passed, len(results))

    return TrainingReport(
        results=tuple(results), config=config, key_columns=tuple(schema.key_columns)
    )