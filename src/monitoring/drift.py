"""Online per-series drift detection against training-time baselines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from src.core.config import DEFAULT_CONFIG, ForecastConfig
from src.features.pipeline import TelemetrySchema

if TYPE_CHECKING:
    from src.training.train import FeatureBaseline

logger = logging.getLogger(__name__)

SeriesKey = Tuple[Any, ...]


class DriftRecommendation(str, Enum):
    NOMINAL = "NOMINAL"
    WATCH = "WATCH"
    RETRAIN_TRIGGERED = "RETRAIN_TRIGGERED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class DriftPolicy:
    """Decision rules layered on top of the statistical tests.

    `require_both` flags a feature only when KS and PSI agree. A series triggers
    when the drifted share of evaluated features reaches `retrain_feature_fraction`
    for `persistence_windows` consecutive evaluations. Features matching
    `excluded_prefixes` or `excluded_features` are not monitored.
    """

    min_samples: int = 30
    require_both: bool = True
    max_null_shift: float = 0.10
    retrain_feature_fraction: float = 0.25
    persistence_windows: int = 1
    psi_epsilon: float = 1e-4
    excluded_prefixes: Tuple[str, ...] = ("hour_", "dow_")
    excluded_features: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.min_samples < 2:
            raise ValueError("min_samples must be >= 2.")
        if not 0.0 < self.max_null_shift <= 1.0:
            raise ValueError("max_null_shift must be in (0, 1].")
        if not 0.0 < self.retrain_feature_fraction <= 1.0:
            raise ValueError("retrain_feature_fraction must be in (0, 1].")
        if self.persistence_windows < 1:
            raise ValueError("persistence_windows must be >= 1.")
        if not 0.0 < self.psi_epsilon < 1.0:
            raise ValueError("psi_epsilon must be in (0, 1).")
        object.__setattr__(self, "excluded_prefixes", tuple(self.excluded_prefixes))
        object.__setattr__(self, "excluded_features", tuple(self.excluded_features))

    def is_excluded(self, name: str) -> bool:
        return name in self.excluded_features or name.startswith(self.excluded_prefixes)


def bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin proportions using the baseline convention: `searchsorted(edges, x, side="right")`."""
    idx = np.searchsorted(edges, values, side="right")
    counts = np.bincount(idx, minlength=edges.size + 1)
    return counts / counts.sum()


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, epsilon: float = 1e-4
) -> float:
    """PSI = sum((actual - expected) * ln(actual / expected)) over aligned bins.

    Proportions are floored at `epsilon` and renormalized so empty bins yield a
    finite, bounded contribution.
    """
    e = np.asarray(expected, dtype="float64")
    a = np.asarray(actual, dtype="float64")
    if e.shape != a.shape or e.ndim != 1:
        raise ValueError("expected and actual must be 1-D arrays of equal length.")
    e = np.clip(e, epsilon, None)
    a = np.clip(a, epsilon, None)
    e /= e.sum()
    a /= a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


@dataclass(frozen=True)
class FeatureDrift:
    """Drift statistics for one feature of one series."""

    feature: str
    n_obs: int
    evaluated: bool
    null_fraction: float
    baseline_null_fraction: float
    null_alert: bool
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    ks_drift: bool = False
    psi: Optional[float] = None
    psi_drift: bool = False
    drifted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "n_obs": self.n_obs,
            "evaluated": self.evaluated,
            "null_fraction": self.null_fraction,
            "baseline_null_fraction": self.baseline_null_fraction,
            "null_alert": self.null_alert,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
            "ks_drift": self.ks_drift,
            "psi": self.psi,
            "psi_drift": self.psi_drift,
            "drifted": self.drifted,
        }


@dataclass(frozen=True)
class SeriesDrift:
    """Drift assessment for one series."""

    key: SeriesKey
    recommendation: DriftRecommendation
    n_rows: int
    features: Tuple[FeatureDrift, ...] = ()
    drifted_fraction: float = 0.0
    consecutive_windows: int = 0
    missing_features: Tuple[str, ...] = ()

    @property
    def drifted_features(self) -> List[str]:
        return [f.feature for f in self.features if f.drifted]

    @property
    def null_alert_features(self) -> List[str]:
        return [f.feature for f in self.features if f.null_alert]


def _native(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


@dataclass(frozen=True, eq=False)
class DriftReport:
    """Structured monitoring payload across all evaluated series."""

    series: Tuple[SeriesDrift, ...]
    unmonitored_keys: Tuple[SeriesKey, ...]
    missing_keys: Tuple[SeriesKey, ...]
    key_columns: Tuple[str, ...]
    alpha: float
    psi_threshold: float
    generated_at: str

    @property
    def recommendation(self) -> DriftRecommendation:
        states = {s.recommendation for s in self.series}
        if DriftRecommendation.RETRAIN_TRIGGERED in states:
            return DriftRecommendation.RETRAIN_TRIGGERED
        if DriftRecommendation.WATCH in states:
            return DriftRecommendation.WATCH
        if DriftRecommendation.NOMINAL in states:
            return DriftRecommendation.NOMINAL
        return DriftRecommendation.INSUFFICIENT_DATA

    @property
    def retrain_keys(self) -> List[SeriesKey]:
        return [
            s.key
            for s in self.series
            if s.recommendation is DriftRecommendation.RETRAIN_TRIGGERED
        ]

    def _key_dict(self, key: SeriesKey) -> Dict[str, Any]:
        return {c: _native(v) for c, v in zip(self.key_columns, key)}

    def summary_frame(self) -> pd.DataFrame:
        """One row per series."""
        rows = [
            {
                **self._key_dict(s.key),
                "recommendation": s.recommendation.value,
                "n_rows": s.n_rows,
                "n_features": sum(f.evaluated for f in s.features),
                "drifted_fraction": s.drifted_fraction,
                "consecutive_windows": s.consecutive_windows,
                "drifted_features": ", ".join(s.drifted_features),
                "null_alerts": ", ".join(s.null_alert_features),
                "missing_features": ", ".join(s.missing_features),
            }
            for s in self.series
        ]
        return pd.DataFrame(rows)

    def feature_frame(self) -> pd.DataFrame:
        """One row per series and feature."""
        rows = [
            {**self._key_dict(s.key), **f.to_dict()} for s in self.series for f in s.features
        ]
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-compatible payload."""
        return {
            "recommendation": self.recommendation.value,
            "generated_at": self.generated_at,
            "thresholds": {"ks_alpha": self.alpha, "psi": self.psi_threshold},
            "n_series_evaluated": len(self.series),
            "retrain_keys": [self._key_dict(k) for k in self.retrain_keys],
            "unmonitored_keys": [self._key_dict(k) for k in self.unmonitored_keys],
            "missing_keys": [self._key_dict(k) for k in self.missing_keys],
            "series": [
                {
                    "key": self._key_dict(s.key),
                    "recommendation": s.recommendation.value,
                    "n_rows": s.n_rows,
                    "drifted_fraction": s.drifted_fraction,
                    "consecutive_windows": s.consecutive_windows,
                    "drifted_features": s.drifted_features,
                    "null_alert_features": s.null_alert_features,
                    "missing_features": list(s.missing_features),
                    "features": [f.to_dict() for f in s.features],
                }
                for s in self.series
            ],
        }


def _as_key(key: Any) -> SeriesKey:
    return key if isinstance(key, tuple) else (key,)


def _validate_baseline(key: SeriesKey, name: str, baseline: "FeatureBaseline") -> None:
    if baseline.bin_proportions.size != baseline.bin_edges.size + 1:
        raise ValueError(f"Inconsistent baseline bins for series {key}, feature '{name}'.")
    if baseline.sample.size == 0:
        raise ValueError(f"Empty baseline sample for series {key}, feature '{name}'.")


class DriftMonitor:
    """Compares inference windows to per-series training baselines.

    Holds a per-series count of consecutive triggering windows for persistence
    gating. Instances are not thread-safe.
    """

    def __init__(
        self,
        baselines: Mapping[SeriesKey, Mapping[str, "FeatureBaseline"]],
        config: ForecastConfig = DEFAULT_CONFIG,
        policy: Optional[DriftPolicy] = None,
        key_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.config = config
        self.policy = policy or DriftPolicy()
        self.key_columns: Tuple[str, ...] = tuple(
            key_columns if key_columns is not None else TelemetrySchema().key_columns
        )
        self._baselines: Dict[SeriesKey, Mapping[str, "FeatureBaseline"]] = {}
        for key, features in baselines.items():
            norm = _as_key(key)
            for name, baseline in features.items():
                _validate_baseline(norm, name, baseline)
            self._baselines[norm] = features
        if not self._baselines:
            raise ValueError("At least one series baseline is required.")
        self._streaks: Dict[SeriesKey, int] = {}

    @classmethod
    def from_training_report(
        cls, report: Any, policy: Optional[DriftPolicy] = None
    ) -> "DriftMonitor":
        """Build from a `TrainingReport`; only gate-passing series are monitored."""
        return cls(report.baselines, report.config, policy, report.key_columns)

    @classmethod
    def from_artifacts(
        cls,
        directory: str | Path,
        policy: Optional[DriftPolicy] = None,
        key_columns: Optional[Sequence[str]] = None,
    ) -> "DriftMonitor":
        """Build from exported `.joblib` artifacts. Load trusted files only."""
        paths = sorted(Path(directory).glob("*.joblib"))
        if not paths:
            raise FileNotFoundError(f"No .joblib artifacts found in {directory}.")

        baselines: Dict[SeriesKey, Mapping[str, "FeatureBaseline"]] = {}
        config: Optional[ForecastConfig] = None
        for path in paths:
            payload = joblib.load(path)
            baselines[_as_key(payload["key"])] = payload["baselines"]
            if config is None:
                config = ForecastConfig.from_dict(payload["config"])
        return cls(baselines, config or DEFAULT_CONFIG, policy, key_columns)

    def reset(self, key: Optional[SeriesKey] = None) -> None:
        """Clear persistence state for one series, or for all series."""
        if key is None:
            self._streaks.clear()
        else:
            self._streaks.pop(_as_key(key), None)

    def evaluate(self, frame: pd.DataFrame, update_state: bool = True) -> DriftReport:
        """Evaluate one inference window; `frame` holds key columns and feature columns."""
        missing = [c for c in self.key_columns if c not in frame.columns]
        if missing:
            raise KeyError(f"Missing key columns: {missing}")

        results: List[SeriesDrift] = []
        unmonitored: List[SeriesKey] = []
        seen: set = set()

        for raw_key, part in frame.groupby(list(self.key_columns), sort=False, observed=True):
            key = _as_key(raw_key)
            seen.add(key)
            baselines = self._baselines.get(key)
            if baselines is None:
                unmonitored.append(key)
                continue
            result = self._evaluate_series(key, part, baselines, update_state)
            results.append(result)
            if result.recommendation is DriftRecommendation.RETRAIN_TRIGGERED:
                logger.warning(
                    "Series %s retrain triggered: %d/%d features drifted (%s).",
                    key,
                    len(result.drifted_features),
                    sum(f.evaluated for f in result.features),
                    ", ".join(result.drifted_features),
                )

        report = DriftReport(
            series=tuple(results),
            unmonitored_keys=tuple(unmonitored),
            missing_keys=tuple(k for k in self._baselines if k not in seen),
            key_columns=self.key_columns,
            alpha=self.config.drift_alpha_threshold,
            psi_threshold=self.config.psi_threshold,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        logger.info(
            "Drift evaluation: %s (%d series, %d retrain).",
            report.recommendation.value,
            len(results),
            len(report.retrain_keys),
        )
        return report

    def _evaluate_series(
        self,
        key: SeriesKey,
        part: pd.DataFrame,
        baselines: Mapping[str, "FeatureBaseline"],
        update_state: bool,
    ) -> SeriesDrift:
        policy = self.policy
        n_rows = len(part)
        prior = self._streaks.get(key, 0)

        if n_rows < policy.min_samples:
            return SeriesDrift(
                key, DriftRecommendation.INSUFFICIENT_DATA, n_rows, consecutive_windows=prior
            )

        features: List[FeatureDrift] = []
        missing_features: List[str] = []
        for name, baseline in baselines.items():
            if policy.is_excluded(name):
                continue
            if name not in part.columns:
                missing_features.append(name)
                continue
            values = part[name].to_numpy(dtype="float64")
            features.append(self._evaluate_feature(name, baseline, values))

        evaluated = [f for f in features if f.evaluated]
        drifted = [f for f in evaluated if f.drifted]
        fraction = len(drifted) / len(evaluated) if evaluated else 0.0
        trigger = bool(evaluated) and fraction >= policy.retrain_feature_fraction

        streak = (prior + 1 if trigger else 0) if evaluated else prior
        if update_state:
            self._streaks[key] = streak

        attention = bool(
            drifted or missing_features or any(f.null_alert for f in features)
        )
        if streak >= policy.persistence_windows and trigger:
            recommendation = DriftRecommendation.RETRAIN_TRIGGERED
        elif attention or trigger:
            recommendation = DriftRecommendation.WATCH
        elif not evaluated:
            recommendation = DriftRecommendation.INSUFFICIENT_DATA
        else:
            recommendation = DriftRecommendation.NOMINAL

        return SeriesDrift(
            key=key,
            recommendation=recommendation,
            n_rows=n_rows,
            features=tuple(features),
            drifted_fraction=fraction,
            consecutive_windows=streak,
            missing_features=tuple(missing_features),
        )

    def _evaluate_feature(
        self, name: str, baseline: "FeatureBaseline", raw: np.ndarray
    ) -> FeatureDrift:
        policy = self.policy
        finite = raw[np.isfinite(raw)]
        null_fraction = float(1.0 - finite.size / raw.size)
        null_alert = abs(null_fraction - baseline.null_fraction) > policy.max_null_shift

        if finite.size < policy.min_samples:
            return FeatureDrift(
                feature=name,
                n_obs=int(finite.size),
                evaluated=False,
                null_fraction=null_fraction,
                baseline_null_fraction=baseline.null_fraction,
                null_alert=null_alert,
            )

        ks = stats.ks_2samp(baseline.sample, finite)
        ks_pvalue = float(ks.pvalue)
        ks_drift = ks_pvalue < self.config.drift_alpha_threshold

        psi = population_stability_index(
            baseline.bin_proportions,
            bin_proportions(finite, baseline.bin_edges),
            policy.psi_epsilon,
        )
        psi_drift = psi > self.config.psi_threshold
        drifted = (ks_drift and psi_drift) if policy.require_both else (ks_drift or psi_drift)

        return FeatureDrift(
            feature=name,
            n_obs=int(finite.size),
            evaluated=True,
            null_fraction=null_fraction,
            baseline_null_fraction=baseline.null_fraction,
            null_alert=null_alert,
            ks_statistic=float(ks.statistic),
            ks_pvalue=ks_pvalue,
            ks_drift=bool(ks_drift),
            psi=psi,
            psi_drift=bool(psi_drift),
            drifted=bool(drifted),
        )