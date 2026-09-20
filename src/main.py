"""FastAPI gateway: telemetry ingest, per-series inference, and drift reporting."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pandas.tseries.frequencies import to_offset
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.config import ForecastConfig
from src.features.pipeline import TelemetrySchema, build_feature_pipeline
from src.monitoring.drift import DriftMonitor, DriftPolicy, DriftRecommendation, SeriesDrift

if TYPE_CHECKING:
    from xgboost import XGBRegressor

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("forecast.gateway")

SeriesKey = Tuple[Any, ...]

SCHEMA = TelemetrySchema()
_AUTOREGRESSIVE_PREFIXES = ("lag_", "roll_")
_MAX_ERROR_DETAILS = 50
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")
_SEVERITY = {
    DriftRecommendation.INSUFFICIENT_DATA: 0,
    DriftRecommendation.NOMINAL: 1,
    DriftRecommendation.WATCH: 2,
    DriftRecommendation.RETRAIN_TRIGGERED: 3,
}


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    """Runtime parameters, sourced from environment variables."""

    artifact_dir: Path = Path("artifacts/models")
    freq: str = "h"
    max_series: int = 100
    max_points: int = 5000
    max_grid_steps: int = 10000
    max_future_skew_seconds: int = 300

    def __post_init__(self) -> None:
        to_offset(self.freq)
        for name in ("max_series", "max_points", "max_grid_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1.")
        if self.max_future_skew_seconds < 0:
            raise ValueError("max_future_skew_seconds must be >= 0.")

    @property
    def step(self) -> pd.Timedelta:
        return pd.Timedelta(to_offset(self.freq))

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            artifact_dir=Path(os.getenv("MODEL_ARTIFACT_DIR", "artifacts/models")),
            freq=os.getenv("FORECAST_FREQ", "h"),
            max_series=int(os.getenv("MAX_SERIES_PER_REQUEST", "100")),
            max_points=int(os.getenv("MAX_POINTS_PER_SERIES", "5000")),
            max_grid_steps=int(os.getenv("MAX_GRID_STEPS", "10000")),
            max_future_skew_seconds=int(os.getenv("MAX_FUTURE_SKEW_SECONDS", "300")),
        )


SETTINGS = Settings.from_env()


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class GatewayError(Exception):
    """Base class for errors mapped to structured client responses."""

    status_code = 500
    code = "GATEWAY_ERROR"

    def __init__(self, message: str, details: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or []


class ModelNotFoundError(GatewayError):
    status_code = 404
    code = "MODEL_NOT_FOUND"


class TemporalIntegrityError(GatewayError):
    status_code = 422
    code = "TEMPORAL_INTEGRITY_VIOLATION"


class InsufficientDataError(GatewayError):
    status_code = 422
    code = "INSUFFICIENT_DATA"


class ArtifactsUnavailableError(GatewayError):
    status_code = 503
    code = "MODEL_ARTIFACTS_UNAVAILABLE"


def _as_key(key: Any) -> SeriesKey:
    return key if isinstance(key, tuple) else (key,)


def _key_dict(key: SeriesKey) -> Dict[str, Any]:
    return dict(zip(SCHEMA.key_columns, key))


# --------------------------------------------------------------------------- #
# Artifact registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class ServedModel:
    model: "XGBRegressor"
    feature_names: Tuple[str, ...]
    validation: Optional[Mapping[str, Any]]


class ModelRegistry:
    """In-memory artifact store with a shared drift monitor."""

    def __init__(
        self,
        models: Mapping[SeriesKey, ServedModel],
        config: Optional[ForecastConfig],
        monitor: Optional[DriftMonitor],
    ) -> None:
        self._models = dict(models)
        self.config = config
        self.monitor = monitor
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._models)

    @property
    def ready(self) -> bool:
        return self.size > 0 and self.config is not None and self.monitor is not None

    def resolve(self, keys: Sequence[SeriesKey]) -> Dict[SeriesKey, ServedModel]:
        missing = [k for k in keys if k not in self._models]
        if missing:
            raise ModelNotFoundError(
                "No gate-passing model artifact exists for the requested series.",
                [{**_key_dict(k), "reason": "no model artifact"} for k in missing],
            )
        return {k: self._models[k] for k in keys}

    def predict(self, key: SeriesKey, row: pd.DataFrame) -> float:
        # Booster access is serialized; single-row prediction is millisecond-scale.
        with self._lock:
            return float(self._models[key].model.predict(row)[0])

    @classmethod
    def load(cls, settings: Settings) -> "ModelRegistry":
        """Load exported `.joblib` artifacts. Load trusted files only."""
        directory = settings.artifact_dir
        paths = sorted(directory.glob("*.joblib")) if directory.is_dir() else []
        if not paths:
            logger.warning("No model artifacts found in %s.", directory)
            return cls({}, None, None)

        models: Dict[SeriesKey, ServedModel] = {}
        baselines: Dict[SeriesKey, Mapping[str, Any]] = {}
        config: Optional[ForecastConfig] = None

        for path in paths:
            payload = joblib.load(path)
            artifact_config = ForecastConfig.from_dict(payload["config"])
            if config is None:
                config = artifact_config
            elif artifact_config != config:
                raise RuntimeError(f"{path.name} was trained with a different ForecastConfig.")

            key = _as_key(payload["key"])
            if key in models:
                raise RuntimeError(f"Duplicate artifact for series {key}.")
            models[key] = ServedModel(
                model=payload["model"],
                feature_names=tuple(payload["feature_names"]),
                validation=payload.get("summary"),
            )
            baselines[key] = payload["baselines"]

        assert config is not None
        expected = {
            name
            for t in build_feature_pipeline(config, SCHEMA, settings.freq).transformers
            for name in t.feature_names
        }
        for key, served in models.items():
            unknown = set(served.feature_names) - expected
            if unknown:
                raise RuntimeError(f"Series {key} expects features the pipeline cannot build: {sorted(unknown)}.")

        monitor = DriftMonitor(baselines, config, DriftPolicy(), SCHEMA.key_columns)
        return cls(models, config, monitor)


# --------------------------------------------------------------------------- #
# Request / response contracts
# --------------------------------------------------------------------------- #
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
CpuValue = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


def _to_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    if not 1970 <= value.year <= 2100:
        raise ValueError("timestamp year must be within 1970-2100.")
    return value


class SeriesPayload(BaseModel):
    """Telemetry for one Box / System / Service Class series."""

    model_config = ConfigDict(extra="forbid")

    box: NonBlank
    system: NonBlank
    service_class: NonBlank
    timestamps: List[datetime] = Field(
        min_length=1, max_length=SETTINGS.max_points, description="ISO-8601; naive values are UTC."
    )
    cpu_usage: List[Optional[CpuValue]] = Field(
        min_length=1, max_length=SETTINGS.max_points, description="Aligned to timestamps; null = missing."
    )

    @property
    def key(self) -> SeriesKey:
        return (self.box, self.system, self.service_class)

    @field_validator("timestamps")
    @classmethod
    def _normalize_timestamps(cls, values: List[datetime]) -> List[datetime]:
        return [_to_naive_utc(v) for v in values]

    @model_validator(mode="after")
    def _check_alignment(self) -> "SeriesPayload":
        if len(self.timestamps) != len(self.cpu_usage):
            raise ValueError("timestamps and cpu_usage must have equal length.")
        if len(set(self.timestamps)) != len(self.timestamps):
            raise ValueError("timestamps must be unique within a series.")
        return self


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series: List[SeriesPayload] = Field(min_length=1, max_length=SETTINGS.max_series)
    require_drift_assessment: bool = Field(
        default=True,
        description="Reject series whose window is too short for drift tests; if false, return INSUFFICIENT_DATA status.",
    )
    include_feature_detail: bool = Field(default=True, description="Include per-feature KS/PSI statistics.")

    @model_validator(mode="after")
    def _check_unique_keys(self) -> "ForecastRequest":
        keys = [s.key for s in self.series]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate series keys in request.")
        return self


class FeatureDriftMetrics(BaseModel):
    feature: str
    n_obs: int
    evaluated: bool
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    ks_drift: bool
    psi: Optional[float] = None
    psi_drift: bool
    drifted: bool
    null_fraction: float
    baseline_null_fraction: float
    null_alert: bool


class DriftStatus(BaseModel):
    status: DriftRecommendation
    n_window_rows: int
    drifted_fraction: float
    max_psi: Optional[float] = None
    min_ks_pvalue: Optional[float] = None
    drifted_features: List[str]
    null_alert_features: List[str]
    missing_features: List[str]
    features: Optional[List[FeatureDriftMetrics]] = None


class ValidationInfo(BaseModel):
    """Out-of-fold error recorded at training time."""

    mae: float
    rmse: float
    n_folds: int
    naive_ratio: Optional[float] = None


class SeriesForecast(BaseModel):
    box: str
    system: str
    service_class: str
    forecast_timestamp: datetime
    predicted_cpu_usage: float
    n_observations: int
    drift: DriftStatus
    validation: Optional[ValidationInfo] = None


class DriftThresholds(BaseModel):
    ks_alpha: float
    psi: float


class ForecastResponse(BaseModel):
    request_id: str
    generated_at: datetime
    frequency: str
    horizon_steps: int
    drift_status: DriftRecommendation
    thresholds: DriftThresholds
    forecasts: List[SeriesForecast]


# --------------------------------------------------------------------------- #
# Inference service
# --------------------------------------------------------------------------- #
def _guard_temporal_integrity(series: Sequence[SeriesPayload], settings: Settings) -> None:
    """Reject future-dated, grid-colliding, or pathologically sparse series."""
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
        seconds=settings.max_future_skew_seconds
    )
    violations: List[Dict[str, Any]] = []

    for s in series:
        slots = pd.DatetimeIndex(s.timestamps).floor(settings.freq)
        if max(s.timestamps) > cutoff:
            reason = "future-dated observations"
        elif slots.has_duplicates:
            reason = f"multiple observations within one '{settings.freq}' grid slot"
        elif (slots.max() - slots.min()) / settings.step > settings.max_grid_steps:
            reason = f"timestamp span exceeds {settings.max_grid_steps} grid steps"
        else:
            continue
        violations.append({**_key_dict(s.key), "reason": reason})

    if violations:
        raise TemporalIntegrityError(
            "Telemetry violates temporal integrity constraints.", violations
        )


def _build_features(
    series: Sequence[SeriesPayload],
    targets: Mapping[SeriesKey, pd.Timestamp],
    config: ForecastConfig,
    settings: Settings,
) -> Tuple[pd.DataFrame, int]:
    """Run the training-time pipeline over history plus one placeholder row per series.

    The placeholder carries the forecast timestamp. Every feature is shifted by at
    least `forecast_horizon`, so its value never enters any feature.
    """
    ts_col, y_col = SCHEMA.timestamp_column, SCHEMA.target_column
    frames: List[pd.DataFrame] = []

    for s in series:
        timestamps = [*s.timestamps, targets[s.key].to_pydatetime()]
        values = [np.nan if v is None else v for v in s.cpu_usage] + [0.0]
        frame = pd.DataFrame(
            {ts_col: pd.DatetimeIndex(timestamps), y_col: np.asarray(values, dtype="float64")}
        )
        for column, value in zip(SCHEMA.key_columns, s.key):
            frame[column] = value
        frames.append(frame)

    pipeline = build_feature_pipeline(config=config, schema=SCHEMA, freq=settings.freq)
    features = pipeline.fit_transform(pd.concat(frames, ignore_index=True))
    return features, pipeline.warmup_rows


def _has_autoregressive_signal(row: pd.DataFrame) -> bool:
    columns = [c for c in row.columns if c.startswith(_AUTOREGRESSIVE_PREFIXES)]
    return not columns or bool(row[columns].notna().any(axis=1).iloc[0])


def _drift_status(drift: Optional[SeriesDrift], detail: bool) -> DriftStatus:
    if drift is None:
        return DriftStatus(
            status=DriftRecommendation.INSUFFICIENT_DATA,
            n_window_rows=0,
            drifted_fraction=0.0,
            drifted_features=[],
            null_alert_features=[],
            missing_features=[],
        )
    evaluated = [f for f in drift.features if f.evaluated]
    return DriftStatus(
        status=drift.recommendation,
        n_window_rows=drift.n_rows,
        drifted_fraction=drift.drifted_fraction,
        max_psi=max((f.psi for f in evaluated), default=None),
        min_ks_pvalue=min((f.ks_pvalue for f in evaluated), default=None),
        drifted_features=drift.drifted_features,
        null_alert_features=drift.null_alert_features,
        missing_features=list(drift.missing_features),
        features=(
            [FeatureDriftMetrics.model_validate(f.to_dict()) for f in drift.features]
            if detail
            else None
        ),
    )


def _validation_info(summary: Optional[Mapping[str, Any]]) -> Optional[ValidationInfo]:
    if not summary:
        return None
    return ValidationInfo(
        mae=summary["mae"],
        rmse=summary["rmse"],
        n_folds=summary["n_folds"],
        naive_ratio=summary.get("naive_ratio"),
    )


def _execute_forecast(
    payload: ForecastRequest, registry: ModelRegistry, settings: Settings, request_id: str
) -> ForecastResponse:
    """Synchronous inference path; executed off the event loop."""
    config, monitor = registry.config, registry.monitor
    if config is None or monitor is None:
        raise ArtifactsUnavailableError("No model artifacts are loaded.")

    series = payload.series
    served = registry.resolve([s.key for s in series])
    _guard_temporal_integrity(series, settings)

    horizon = config.forecast_horizon
    targets = {
        s.key: pd.Timestamp(max(s.timestamps)).floor(settings.freq) + horizon * settings.step
        for s in series
    }
    features, warmup = _build_features(series, targets, config, settings)
    min_steps = max(warmup - horizon + 1, 1)

    ts_col = SCHEMA.timestamp_column
    groups = {
        _as_key(k): part
        for k, part in features.groupby(list(SCHEMA.key_columns), sort=False, observed=True)
    }

    rows: Dict[SeriesKey, pd.DataFrame] = {}
    windows: List[pd.DataFrame] = []
    short: List[Dict[str, Any]] = []

    for s in series:
        part = groups.get(s.key)
        is_target = part[ts_col] == targets[s.key] if part is not None else None
        if part is None or not is_target.any():
            reason = f"history spans too few grid steps; at least {min_steps} required"
        else:
            row = part.loc[is_target, list(served[s.key].feature_names)].astype("float64")
            if _has_autoregressive_signal(row):
                rows[s.key] = row
                windows.append(part.loc[~is_target])
                continue
            reason = "no usable autoregressive history within the lookback span"
        short.append(
            {**_key_dict(s.key), "reason": reason, "observed_points": len(s.timestamps)}
        )

    if short:
        raise InsufficientDataError("Telemetry history is too short to build forecast features.", short)

    report = monitor.evaluate(pd.concat(windows, ignore_index=True), update_state=False)
    drift_by_key = {d.key: d for d in report.series}
    statuses = {
        s.key: _drift_status(drift_by_key.get(s.key), payload.include_feature_detail)
        for s in series
    }

    if payload.require_drift_assessment:
        blind = [
            {
                **_key_dict(s.key),
                "reason": (
                    f"drift window has {statuses[s.key].n_window_rows} rows "
                    f"(min_samples={monitor.policy.min_samples}); supply a longer history "
                    "or set require_drift_assessment=false"
                ),
            }
            for s in series
            if statuses[s.key].status is DriftRecommendation.INSUFFICIENT_DATA
        ]
        if blind:
            raise InsufficientDataError("Drift cannot be assessed on the supplied window.", blind)

    forecasts = [
        SeriesForecast(
            box=s.box,
            system=s.system,
            service_class=s.service_class,
            forecast_timestamp=targets[s.key].to_pydatetime().replace(tzinfo=timezone.utc),
            predicted_cpu_usage=max(registry.predict(s.key, rows[s.key]), 0.0),
            n_observations=sum(v is not None for v in s.cpu_usage),
            drift=statuses[s.key],
            validation=_validation_info(served[s.key].validation),
        )
        for s in series
    ]

    return ForecastResponse(
        request_id=request_id,
        generated_at=datetime.now(timezone.utc),
        frequency=settings.freq,
        horizon_steps=horizon,
        drift_status=max((f.drift.status for f in forecasts), key=_SEVERITY.__getitem__),
        thresholds=DriftThresholds(ks_alpha=report.alpha, psi=report.psi_threshold),
        forecasts=forecasts,
    )


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.registry = await run_in_threadpool(ModelRegistry.load, SETTINGS)
    logger.info("Gateway started: %d model artifact(s) loaded.", app.state.registry.size)
    yield


app = FastAPI(
    title="Mainframe CPU Forecasting Gateway",
    version="1.0.0",
    description="Per-series CPU forecasts with live drift diagnostics.",
    lifespan=lifespan,
)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Optional[List[Dict[str, Any]]] = None,
) -> JSONResponse:
    rid = _request_id(request)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {"code": code, "message": message, "details": details or [], "request_id": rid}
        },
        headers={"X-Request-ID": rid},
    )


def get_registry(request: Request) -> ModelRegistry:
    registry: Optional[ModelRegistry] = getattr(request.app.state, "registry", None)
    if registry is None or not registry.ready:
        raise ArtifactsUnavailableError("No model artifacts are loaded.")
    return registry


@app.middleware("http")
async def request_context(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "")
    rid = supplied if _REQUEST_ID_PATTERN.fullmatch(supplied) else uuid.uuid4().hex
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(GatewayError)
async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
    log = logger.error if exc.status_code >= 500 else logger.warning
    log("request_id=%s %s: %s", _request_id(request), exc.code, exc.message)
    return _error_response(request, exc.status_code, exc.code, exc.message, exc.details)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
        for e in exc.errors()[:_MAX_ERROR_DETAILS]
    ]
    return _error_response(request, 422, "INVALID_REQUEST", "Request schema validation failed.", details)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _error_response(request, exc.status_code, "HTTP_ERROR", str(exc.detail))


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("request_id=%s unhandled error", _request_id(request))
    return _error_response(request, 500, "INTERNAL_ERROR", "Unexpected server error.")


@app.get("/healthz", include_in_schema=False)
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
async def readyz(request: Request) -> JSONResponse:
    registry: Optional[ModelRegistry] = getattr(request.app.state, "registry", None)
    ready = bool(registry and registry.ready)
    return JSONResponse(
        {"status": "ready" if ready else "unavailable", "models_loaded": registry.size if registry else 0},
        status_code=200 if ready else 503,
    )


@app.post(
    "/api/v1/forecast",
    response_model=ForecastResponse,
    tags=["forecast"],
    summary="Forecast CPU utilization and report drift per series",
)
async def forecast(
    payload: ForecastRequest,
    request: Request,
    registry: ModelRegistry = Depends(get_registry),
) -> ForecastResponse:
    started = time.perf_counter()
    response = await run_in_threadpool(
        _execute_forecast, payload, registry, SETTINGS, request.state.request_id
    )
    logger.info(
        "request_id=%s series=%d drift=%s duration_ms=%.1f",
        response.request_id,
        len(response.forecasts),
        response.drift_status.value,
        (time.perf_counter() - started) * 1000.0,
    )
    return response