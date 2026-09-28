"""End-to-end smoke test: synthetic telemetry -> features -> training -> drift.

Usage: python scripts/run_pipeline.py [--output-dir artifacts/models] [--seed 42]
Exit status is non-zero when a blocking check fails.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Worker processes resolve `src.*` through the inherited environment.
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p
)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pandas.tseries.frequencies import to_offset  # noqa: E402

from src.core.config import DEFAULT_CONFIG, ForecastConfig  # noqa: E402
from src.features.pipeline import (  # noqa: E402
    FeaturePipeline,
    TelemetrySchema,
    build_feature_pipeline,
)
from src.monitoring.drift import DriftMonitor, DriftRecommendation, DriftReport  # noqa: E402
from src.training.train import (  # noqa: E402
    ForwardChainingSplit,
    InsufficientHistoryError,
    QualityGate,
    TrainingConfig,
    TrainingReport,
    TrainingStatus,
    train_all_series,
)

SeriesKey = Tuple[Any, ...]

SCHEMA = TelemetrySchema()
TS, Y = SCHEMA.timestamp_column, SCHEMA.target_column
FREQ = "h"
STEP = pd.Timedelta(to_offset(FREQ))
START = pd.Timestamp("2026-01-05")  # Monday 00:00
TRAIN_DAYS = 14
LIVE_DAYS = 7
LIVE_START = START + pd.Timedelta(days=TRAIN_DAYS)
WIDTH = 104


# --------------------------------------------------------------------------- #
# Synthetic workload grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorkloadProfile:
    """Diurnal CPU profile in percentage points."""

    base: float
    amplitude: float
    peak_hour: float
    noise_sd: float
    ar_phi: float = 0.5
    weekend_factor: float = 0.9


BOXES = ("BOX01", "BOX02")
SYSTEMS = ("SYSA", "SYSB")
PROFILES = {
    "CICS_PRD": WorkloadProfile(base=45.0, amplitude=24.0, peak_hour=14.0, noise_sd=1.5),
    "DB2_HIGH": WorkloadProfile(base=40.0, amplitude=20.0, peak_hour=13.0, noise_sd=1.5),
    "BATCH_LOW": WorkloadProfile(base=34.0, amplitude=22.0, peak_hour=2.0, noise_sd=2.0),
}

# Structurally unpredictable series; the quality gate must reject it.
ERRATIC_KEY: SeriesKey = ("BOX02", "SYSB", "BATCH_LOW")
ERRATIC_PROFILE = WorkloadProfile(
    base=50.0, amplitude=5.0, peak_hour=12.0, noise_sd=28.0, ar_phi=0.2
)
# Series receiving the regime shift in the live window.
SPIKED_KEYS = frozenset({("BOX01", "SYSA", "BATCH_LOW"), ("BOX02", "SYSA", "CICS_PRD")})


def series_keys() -> List[SeriesKey]:
    return list(itertools.product(BOXES, SYSTEMS, PROFILES))


def _series_mask(frame: pd.DataFrame, key: SeriesKey) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for column, value in zip(SCHEMA.key_columns, key):
        mask &= (frame[column] == value).to_numpy()
    return mask


def _ar1(rng: np.random.Generator, n: int, sd: float, phi: float) -> np.ndarray:
    """Stationary AR(1) noise with marginal standard deviation `sd`."""
    innovations = rng.normal(0.0, sd * math.sqrt(1.0 - phi**2), n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sd)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + innovations[i]
    return out


def synthesize_telemetry(days: int, seed: int) -> pd.DataFrame:
    """Hourly CPU telemetry for every Box / System / Service Class series."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range(START, periods=days * 24, freq=FREQ)
    hour = stamps.hour.to_numpy(dtype="float64")
    weekend = stamps.dayofweek.to_numpy() >= 5

    frames: List[pd.DataFrame] = []
    for key in series_keys():
        profile = ERRATIC_PROFILE if key == ERRATIC_KEY else PROFILES[key[2]]
        scale = rng.uniform(0.85, 1.15)
        jitter = rng.uniform(-1.0, 1.0)

        phase = 2.0 * np.pi * (hour - profile.peak_hour - jitter) / 24.0
        seasonal = profile.amplitude * (np.cos(phase) + 0.25 * np.cos(2.0 * phase + 0.6))
        level = np.where(weekend, profile.weekend_factor, 1.0)
        noise = _ar1(rng, len(stamps), profile.noise_sd, profile.ar_phi)
        cpu = np.clip((profile.base * scale + seasonal) * level + noise, 0.0, 100.0)

        frame = pd.DataFrame({TS: stamps, Y: cpu.round(3)})
        for column, value in zip(SCHEMA.key_columns, key):
            frame[column] = value
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)[SCHEMA.required_columns]


def inject_batch_spike(
    frame: pd.DataFrame,
    keys: frozenset,
    live_start: pd.Timestamp,
    seed: int,
    level_shift: float = 25.0,
    nightly_burst: float = 20.0,
    extra_sd: float = 5.0,
) -> pd.DataFrame:
    """Regime shift: sustained level increase, nightly batch bursts, inflated variance."""
    rng = np.random.default_rng(seed + 1)
    out = frame.copy()
    live = (out[TS] >= live_start).to_numpy()

    for key in sorted(keys):
        mask = _series_mask(out, key) & live
        hours = out.loc[mask, TS].dt.hour.to_numpy()
        burst = np.where(hours < 6, nightly_burst, 0.0)
        noise = rng.normal(0.0, extra_sd, size=int(mask.sum()))
        shifted = out.loc[mask, Y].to_numpy() + level_shift + burst + noise
        out.loc[mask, Y] = np.clip(shifted, 0.0, 100.0).round(3)
    return out


# --------------------------------------------------------------------------- #
# Console rendering
# --------------------------------------------------------------------------- #
def section(title: str) -> None:
    print(f"\n{'=' * WIDTH}\n{title}\n{'=' * WIDTH}")


def render_table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]], align: Optional[Sequence[str]] = None
) -> str:
    cells = [[str(c) for c in row] for row in rows]
    align = align or ["l"] * len(headers)
    widths = [max([len(h)] + [len(r[i]) for r in cells]) for i, h in enumerate(headers)]

    def fmt(row: Sequence[str]) -> str:
        parts = [
            f" {c.rjust(w) if a == 'r' else c.ljust(w)} "
            for c, w, a in zip(row, widths, align)
        ]
        return "|" + "|".join(parts) + "|"

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    return "\n".join([sep, fmt(list(headers)), sep, *(fmt(r) for r in cells), sep])


def _fmt(value: Optional[float], spec: str) -> str:
    return "-" if value is None else spec.format(value)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    blocking: bool = True


class Checklist:
    """Accumulates verification outcomes; only blocking failures affect exit status."""

    def __init__(self) -> None:
        self.items: List[Check] = []

    def record(self, name: str, passed: bool, detail: str = "", blocking: bool = True) -> bool:
        self.items.append(Check(name, bool(passed), detail, blocking))
        return bool(passed)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.items if c.blocking)

    def render(self) -> str:
        rows = [
            ["PASS" if c.passed else ("FAIL" if c.blocking else "WARN"), c.name, c.detail]
            for c in self.items
        ]
        return render_table(["Status", "Check", "Detail"], rows)


# --------------------------------------------------------------------------- #
# Stage 2: features
# --------------------------------------------------------------------------- #
def run_feature_stage(
    history: pd.DataFrame, config: ForecastConfig, checks: Checklist
) -> Tuple[FeaturePipeline, pd.DataFrame]:
    pipeline = build_feature_pipeline(config=config, schema=SCHEMA, freq=FREQ)
    features = pipeline.fit_transform(history)
    names = list(pipeline.feature_names_ or [])
    key_columns = list(SCHEMA.key_columns)
    warmup = pipeline.warmup_rows
    expected_rows = TRAIN_DAYS * 24 - warmup

    print(f"features        : {len(names)} ({', '.join(names)})")
    print(f"warm-up rows    : {warmup} per series")
    print(f"rows in -> out  : {len(history)} -> {len(features)}")

    sizes = features.groupby(key_columns, observed=True).size()
    checks.record(
        "Warm-up rows dropped per series",
        sizes.size == len(series_keys()) and bool((sizes == expected_rows).all()),
        f"{warmup} dropped; {expected_rows} rows/series retained",
    )
    checks.record(
        "First feature row = series start + warm-up",
        features[TS].min() == START + warmup * STEP,
        str(features[TS].min()),
    )
    checks.record(
        "No missing values on contiguous grid",
        int(features[names].isna().sum().sum()) == 0,
        f"{int(features[names].isna().sum().sum())} NaN cells",
    )

    # Leakage probe: perturbing y[t] must leave every feature at row t unchanged.
    probe_key = series_keys()[0]
    probe_ts = START + 200 * STEP
    mutated = history.copy()
    hit = _series_mask(mutated, probe_key) & (mutated[TS] == probe_ts).to_numpy()
    mutated.loc[hit, Y] += 25.0
    features_mut = pipeline.transform(mutated)

    def row(df: pd.DataFrame, t: pd.Timestamp) -> np.ndarray:
        sel = _series_mask(df, probe_key) & (df[TS] == t).to_numpy()
        return df.loc[sel, names].to_numpy(dtype="float64")

    at_t, at_t_mut = row(features, probe_ts), row(features_mut, probe_ts)
    at_next, at_next_mut = row(features, probe_ts + STEP), row(features_mut, probe_ts + STEP)
    checks.record(
        "Leakage probe: features[t] independent of y[t]",
        at_t.shape[0] == 1 and np.allclose(at_t, at_t_mut, equal_nan=True),
        f"y[{probe_ts}] += 25.0",
    )
    checks.record(
        "Leakage probe is live: features[t+1] respond to y[t]",
        at_next.shape[0] == 1 and not np.allclose(at_next, at_next_mut, equal_nan=True),
        "lag/rolling features propagate the perturbation forward",
    )

    def others(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[~_series_mask(df, probe_key)].reset_index(drop=True)

    checks.record(
        "Series isolation: perturbation does not cross keys",
        others(features).equals(others(features_mut)),
        f"{len(series_keys()) - 1} untouched series identical",
    )
    return pipeline, features


def verify_forward_chaining(
    features: pd.DataFrame, config: ForecastConfig, training: TrainingConfig, checks: Checklist
) -> None:
    _, sample = next(iter(features.groupby(list(SCHEMA.key_columns), sort=False, observed=True)))
    stamps = sample[TS].sort_values().reset_index(drop=True)
    splitter = ForwardChainingSplit(
        n_splits=training.n_splits,
        min_train_size=training.min_train_size or config.min_history_points,
        horizon=config.forecast_horizon,
        freq=training.freq,
        test_size=training.test_size,
    )
    try:
        folds = list(splitter.split(stamps))
    except InsufficientHistoryError as exc:
        checks.record("Forward-chaining layout feasible", False, str(exc))
        return

    gap = config.forecast_horizon * STEP
    pairs = list(zip(folds, folds[1:]))
    checks.record(
        "Forward chaining: train precedes validation by >= horizon",
        all(stamps.iloc[tr[-1]] + gap <= stamps.iloc[va[0]] for tr, va in folds),
        f"{len(folds)} folds, horizon={config.forecast_horizon}",
    )
    checks.record(
        "Forward chaining: expanding train, disjoint validation blocks",
        len(folds) == training.n_splits
        and all(len(a[0]) < len(b[0]) and a[1][-1] < b[1][0] for a, b in pairs)
        and folds[-1][1][-1] == len(stamps) - 1,
        f"train sizes {[len(tr) for tr, _ in folds]}, val block {len(folds[0][1])}",
    )


# --------------------------------------------------------------------------- #
# Stage 3-4: training reporting and artifact verification
# --------------------------------------------------------------------------- #
def render_accuracy(report: TrainingReport) -> str:
    rows = []
    for r in report.results:
        s = r.summary
        rows.append(
            [
                *r.key,
                r.status.value.upper(),
                r.n_rows,
                s.n_folds if s else "-",
                _fmt(s.mae if s else None, "{:.3f}"),
                _fmt(s.rmse if s else None, "{:.3f}"),
                _fmt(s.naive_ratio if s else None, "{:.3f}"),
            ]
        )
    return render_table(
        ["Box", "System", "Service Class", "Status", "Rows", "Folds", "MAE", "RMSE", "MAE/Naive"],
        rows,
        ["l", "l", "l", "l", "r", "r", "r", "r", "r"],
    )


def pooled_metrics(report: TrainingReport) -> Optional[Tuple[float, float, int]]:
    summaries = [r.summary for r in report.exportable if r.summary]
    n = sum(s.n_oof_rows for s in summaries)
    if n == 0:
        return None
    mae = sum(s.mae * s.n_oof_rows for s in summaries) / n
    rmse = math.sqrt(sum(s.rmse**2 * s.n_oof_rows for s in summaries) / n)
    return mae, rmse, n


def prepare_output_dir(directory: Path, keep_existing: bool) -> List[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    removed: List[Path] = []
    if not keep_existing:
        for stale in directory.glob("*.joblib"):
            stale.unlink()
            removed.append(stale)
    return removed


def verify_artifacts(
    paths: Sequence[Path],
    report: TrainingReport,
    features: pd.DataFrame,
    directory: Path,
    checks: Checklist,
) -> None:
    expected = {r.key: r for r in report.exportable}
    contract = {"key", "model", "baselines", "feature_names", "summary", "config"}
    seen: set = set()
    contract_ok = config_ok = predict_ok = True

    for path in paths:
        payload = joblib.load(path)
        contract_ok &= contract <= set(payload)
        key = tuple(payload["key"])
        seen.add(key)
        result = expected.get(key)
        if result is None:
            continue
        config_ok &= ForecastConfig.from_dict(payload["config"]) == report.config
        sample = features.loc[_series_mask(features, key), list(payload["feature_names"])].tail(16)
        predict_ok &= bool(
            np.allclose(payload["model"].predict(sample), result.model.predict(sample), atol=1e-6)
        )

    checks.record(
        "Artifacts: one file per gate-passing series",
        seen == set(expected) and len(paths) == len(expected),
        f"{len(paths)} written, {len(expected)} exportable",
    )
    checks.record("Artifacts: payload contract complete", contract_ok, ", ".join(sorted(contract)))
    checks.record("Artifacts: ForecastConfig round-trips", config_ok)
    checks.record("Artifacts: reloaded model predictions match in-memory", predict_ok)
    checks.record(
        "Artifacts: audit summary written",
        (directory / "training_summary.csv").is_file(),
        "training_summary.csv",
    )


# --------------------------------------------------------------------------- #
# Stage 5: drift
# --------------------------------------------------------------------------- #
def build_live_window(pipeline: FeaturePipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Features for the live window, computed with preceding history for lag continuity."""
    features = pipeline.transform(frame)
    return features[features[TS] >= LIVE_START].reset_index(drop=True)


def live_mean(frame: pd.DataFrame, key: SeriesKey) -> float:
    mask = _series_mask(frame, key) & (frame[TS] >= LIVE_START).to_numpy()
    return float(frame.loc[mask, Y].mean())


def render_drift(report: DriftReport) -> str:
    rows = []
    for s in sorted(report.series, key=lambda x: x.key):
        evaluated = [f for f in s.features if f.evaluated]
        drivers = sorted((f for f in evaluated if f.drifted), key=lambda f: f.psi, reverse=True)[:3]
        rows.append(
            [
                *s.key,
                s.recommendation.value,
                f"{len(s.drifted_features)}/{len(evaluated)}",
                _fmt(max((f.psi for f in evaluated), default=None), "{:.3f}"),
                _fmt(min((f.ks_pvalue for f in evaluated), default=None), "{:.2e}"),
                ", ".join(f.feature for f in drivers) or "-",
            ]
        )
    return render_table(
        ["Box", "System", "Service Class", "Verdict", "Drifted", "Max PSI", "Min KS p", "Top drivers"],
        rows,
        ["l", "l", "l", "l", "r", "r", "r", "l"],
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end pipeline smoke test.")
    parser.add_argument("--output-dir", default="artifacts/models", help="Artifact directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--keep-existing", action="store_true", help="Do not remove existing *.joblib files."
    )
    parser.add_argument(
        "--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    started = time.perf_counter()
    config = DEFAULT_CONFIG
    checks = Checklist()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    # 1/6 Telemetry
    section("1/6  Synthetic telemetry")
    stable_frame = synthesize_telemetry(TRAIN_DAYS + LIVE_DAYS, args.seed)
    shifted_frame = inject_batch_spike(stable_frame, SPIKED_KEYS, LIVE_START, args.seed)
    history = stable_frame[stable_frame[TS] < LIVE_START].reset_index(drop=True)

    n_series = len(series_keys())
    print(f"series          : {n_series} ({len(BOXES)} boxes x {len(SYSTEMS)} systems x {len(PROFILES)} service classes)")
    print(f"training window : {TRAIN_DAYS} d hourly, {len(history)} rows [{START.date()} .. {LIVE_START.date()})")
    print(f"live window     : {LIVE_DAYS} d hourly from {LIVE_START.date()}")
    print(f"regime shift    : {len(SPIKED_KEYS)} series | unpredictable series: {'/'.join(ERRATIC_KEY)}")
    checks.record(
        "Synthetic telemetry well-formed",
        len(history) == n_series * TRAIN_DAYS * 24
        and not history.isna().any().any()
        and bool(history[Y].between(0.0, 100.0).all())
        and SPIKED_KEYS <= set(series_keys()),
        f"{len(history)} rows, cpu_usage in [0, 100]",
    )

    # 2/6 Features
    section("2/6  Feature pipeline")
    pipeline, features = run_feature_stage(history, config, checks)

    # 3/6 Training
    section("3/6  Parallel training with forward-chaining validation")
    training = TrainingConfig(n_splits=args.n_splits, gate=QualityGate(), seed=args.seed)
    gate = training.gate
    print(
        f"gate            : MAE <= {gate.max_mae}, RMSE <= {gate.max_rmse}, "
        f"MAE/naive <= {gate.max_naive_ratio} (cpu_usage in % points)"
    )
    verify_forward_chaining(features, config, training, checks)

    fit_started = time.perf_counter()
    report = train_all_series(
        features, config=config, schema=SCHEMA, training=training, n_jobs=args.n_jobs
    )
    print(f"trained {len(report.results)} series in {time.perf_counter() - fit_started:.1f}s "
          f"({len(report.exportable)} exportable)\n")
    print(render_accuracy(report))

    pooled = pooled_metrics(report)
    if pooled:
        print(f"\npooled out-of-fold (exportable series): MAE={pooled[0]:.3f}  RMSE={pooled[1]:.3f}  rows={pooled[2]}")
    rejected = [r for r in report.results if r.status is not TrainingStatus.PASSED]
    if rejected:
        print("\nnot exported:")
        for r in rejected:
            print(f"  {'/'.join(map(str, r.key))} [{r.status.value}] {'; '.join(r.reasons)}")

    by_key = {r.key: r for r in report.results}
    healthy = [r for k, r in by_key.items() if k != ERRATIC_KEY]
    erratic = by_key.get(ERRATIC_KEY)
    checks.record(
        "Training completed without worker failures",
        all(r.status in (TrainingStatus.PASSED, TrainingStatus.REJECTED) for r in report.results),
        ", ".join(f"{s.value}={sum(r.status is s for r in report.results)}" for s in TrainingStatus),
    )
    checks.record(
        "Quality gate passes learnable series",
        all(r.exportable for r in healthy),
        f"{sum(r.exportable for r in healthy)}/{len(healthy)} passed",
    )
    checks.record(
        "Quality gate blocks unpredictable series",
        erratic is not None
        and erratic.status is TrainingStatus.REJECTED
        and erratic.model is None,
        "; ".join(erratic.reasons) if erratic else "series missing",
    )

    # 4/6 Artifacts
    section("4/6  Artifact export and round-trip verification")
    removed = prepare_output_dir(output_dir, args.keep_existing)
    if removed:
        print(f"removed {len(removed)} stale artifact(s) from {output_dir}")
    paths = report.export(output_dir)
    print(f"exported {len(paths)} model artifact(s) + training_summary.csv -> {output_dir}")
    verify_artifacts(paths, report, features, output_dir, checks)
    if not paths:
        print("\nNo exportable models; drift stage skipped.")
        checks.record("Drift stage executed", False, "no artifacts to monitor")
        section("6/6  Verdict")
        print(checks.render())
        return 1

    # 5/6 Drift
    section("5/6  Drift monitoring on live window")
    monitor = DriftMonitor.from_artifacts(output_dir)
    policy = monitor.policy
    print(
        f"thresholds      : KS alpha={monitor.config.drift_alpha_threshold}, PSI={monitor.config.psi_threshold} | "
        f"require_both={policy.require_both}, retrain_fraction={policy.retrain_feature_fraction}, "
        f"persistence={policy.persistence_windows}"
    )
    for key in sorted(SPIKED_KEYS):
        print(
            f"injected shift  : {'/'.join(key)} live mean CPU "
            f"{live_mean(stable_frame, key):.1f}% -> {live_mean(shifted_frame, key):.1f}%"
        )

    stable_window = build_live_window(pipeline, stable_frame)
    shifted_window = build_live_window(pipeline, shifted_frame)
    control = monitor.evaluate(stable_window, update_state=False)
    alert = monitor.evaluate(shifted_window)

    print(
        f"\ncontrol window (no shift): {control.recommendation.value}, "
        f"{len(control.retrain_keys)} retrain trigger(s) across {len(control.series)} series"
    )
    print("\nshifted window:")
    print(render_drift(alert))
    print(f"\nsystem verdict  : {alert.recommendation.value}")
    if alert.retrain_keys:
        print(f"retrain keys    : {', '.join('/'.join(map(str, k)) for k in alert.retrain_keys)}")
    if alert.unmonitored_keys:
        print(f"unmonitored     : {', '.join('/'.join(map(str, k)) for k in alert.unmonitored_keys)} (no gate-passing model)")
    if alert.missing_keys:
        print(f"missing         : {', '.join('/'.join(map(str, k)) for k in alert.missing_keys)}")

    alert_by_key = {s.key: s for s in alert.series}
    missed = [
        k
        for k in sorted(SPIKED_KEYS)
        if k not in alert_by_key
        or alert_by_key[k].recommendation is not DriftRecommendation.RETRAIN_TRIGGERED
    ]
    checks.record(
        "Drift: injected regime shift triggers retrain",
        not missed,
        "all injected series flagged" if not missed else f"missed: {missed}",
    )
    checks.record(
        "Drift: aggregate recommendation is RETRAIN_TRIGGERED",
        alert.recommendation is DriftRecommendation.RETRAIN_TRIGGERED,
        alert.recommendation.value,
    )
    checks.record(
        "Drift: rejected series reported as unmonitored",
        ERRATIC_KEY in alert.unmonitored_keys,
        "/".join(ERRATIC_KEY),
    )
    finite = all(
        f.psi is not None
        and f.ks_pvalue is not None
        and math.isfinite(f.psi)
        and math.isfinite(f.ks_pvalue)
        for s in alert.series
        for f in s.features
        if f.evaluated
    )
    checks.record("Drift: PSI and KS statistics finite", finite)
    false_alarms = [*control.retrain_keys, *(k for k in alert.retrain_keys if k not in SPIKED_KEYS)]
    checks.record(
        "Drift: no retrain triggers on stable series",
        not false_alarms,
        "none" if not false_alarms else f"false alarms: {false_alarms}",
        blocking=False,
    )

    # 6/6 Verdict
    section("6/6  Verdict")
    print(checks.render())
    print(f"\nelapsed: {time.perf_counter() - started:.1f}s")
    print(f"RESULT: {'PASS' if checks.ok else 'FAIL'}")
    return 0 if checks.ok else 1


if __name__ == "__main__":
    sys.exit(main())