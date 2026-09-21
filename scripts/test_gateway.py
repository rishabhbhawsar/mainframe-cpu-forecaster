"""Gateway smoke test: response contract, error mapping, and drift behavior.

Usage:
    python scripts/test_gateway.py                                   # in-process ASGI client
    python scripts/test_gateway.py --base-url http://127.0.0.1:8000  # running Uvicorn server

Requires artifacts from `python scripts/run_pipeline.py`. Exit status: 0 pass, 1 failed
blocking checks, 2 environment not ready.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

try:
    import httpx  # noqa: E402
except ImportError:  # pragma: no cover
    sys.exit("httpx is required: pip install httpx")

from scripts.run_pipeline import (  # noqa: E402
    ERRATIC_KEY,
    FREQ,
    LIVE_DAYS,
    LIVE_START,
    SCHEMA,
    SPIKED_KEYS,
    STEP,
    TRAIN_DAYS,
    TS,
    Y,
    Checklist,
    inject_batch_spike,
    render_table,
    section,
    series_keys,
    synthesize_telemetry,
)
from src.core.config import ForecastConfig  # noqa: E402
from src.features.pipeline import build_feature_pipeline  # noqa: E402

SeriesKey = Tuple[str, ...]

FORECAST_PATH = "/api/v1/forecast"
LIVE_END = LIVE_START + pd.Timedelta(days=LIVE_DAYS)
WINDOW = LIVE_DAYS * 24
UNMONITORED = ("roll_mean_24", "roll_std_24")
CALENDAR_PREFIXES = ("hour_", "dow_")


@dataclass
class Context:
    client: httpx.Client
    checks: Checklist
    stable: pd.DataFrame
    shifted: pd.DataFrame
    artifacts: Dict[SeriesKey, Dict[str, Any]]
    config: ForecastConfig
    warmup: int
    stable_keys: List[SeriesKey]
    spiked_key: SeriesKey
    report: Optional[str] = field(default=None)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def fmt(value: Optional[float], spec: str) -> str:
    return "-" if value is None else spec.format(value)


def clip(text: str, limit: int = 96) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def window_frame(
    frame: pd.DataFrame, key: SeriesKey, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    mask = (frame[TS] >= start) & (frame[TS] < end)
    for column, value in zip(SCHEMA.key_columns, key):
        mask &= frame[column] == value
    return frame.loc[mask].sort_values(TS).reset_index(drop=True)


def to_payload(rows: pd.DataFrame, key: SeriesKey) -> Dict[str, Any]:
    box, system, service_class = key
    return {
        "box": box,
        "system": system,
        "service_class": service_class,
        "timestamps": [t.isoformat() for t in rows[TS]],
        "cpu_usage": [float(v) for v in rows[Y]],
    }


def live_payload(frame: pd.DataFrame, key: SeriesKey, hours: int = WINDOW) -> Dict[str, Any]:
    return to_payload(window_frame(frame, key, LIVE_END - hours * STEP, LIVE_END), key)


def post(ctx: Context, body: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> httpx.Response:
    return ctx.client.post(FORECAST_PATH, json=body, headers=headers)


def body_of(resp: httpx.Response) -> Dict[str, Any]:
    try:
        data = resp.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def error_of(resp: httpx.Response) -> Dict[str, Any]:
    err = body_of(resp).get("error")
    return err if isinstance(err, dict) else {}


def brief(resp: httpx.Response) -> str:
    err = error_of(resp)
    tag = f"HTTP {resp.status_code}"
    return clip(f"{tag} {err['code']}: {err.get('message', '')}") if err.get("code") else tag


def error_keys(err: Dict[str, Any]) -> List[SeriesKey]:
    return [(d.get("box"), d.get("system"), d.get("service_class")) for d in err.get("details", [])]


def check_error(
    ctx: Context,
    name: str,
    resp: httpx.Response,
    status: int,
    code: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    error = error_of(resp)
    ok = resp.status_code == status and error.get("code") == code
    if ok and reason is not None:
        ok = any(reason in str(d.get("reason", "")) for d in error.get("details", []))
    first = (error.get("details") or [{}])[0]
    extra = first.get("reason") or first.get("msg")
    ctx.checks.record(name, ok, clip(f"HTTP {resp.status_code} {error.get('code')}" + (f" | {extra}" if extra else "")))
    return error


def drift_summary(drift: Dict[str, Any]) -> str:
    evaluated = [x for x in drift.get("features") or [] if x["evaluated"]]
    return (
        f"{drift.get('status')}, {len(drift.get('drifted_features', []))}/{len(evaluated)} drifted, "
        f"max PSI {fmt(drift.get('max_psi'), '{:.3f}')}, min KS p {fmt(drift.get('min_ks_pvalue'), '{:.2e}')}"
    )


def offline_forecast(artifact: Dict[str, Any], history: pd.DataFrame) -> float:
    """Reference prediction from the offline pipeline and the serialized model."""
    config = ForecastConfig.from_dict(artifact["config"])
    target_ts = history[TS].max() + config.forecast_horizon * STEP
    placeholder = history.iloc[[-1]].copy()
    placeholder[TS] = target_ts
    placeholder[Y] = 0.0
    frame = pd.concat([history, placeholder], ignore_index=True)
    features = build_feature_pipeline(config=config, schema=SCHEMA, freq=FREQ).fit_transform(frame)
    row = features.loc[features[TS] == target_ts, list(artifact["feature_names"])].astype("float64")
    return max(float(artifact["model"].predict(row)[0]), 0.0)


def render_forecasts(forecasts: Sequence[Dict[str, Any]]) -> str:
    rows = []
    for f in forecasts:
        d = f["drift"]
        evaluated = sum(1 for x in d.get("features") or [] if x["evaluated"])
        v = f.get("validation") or {}
        rows.append(
            [
                f["box"],
                f["system"],
                f["service_class"],
                f["forecast_timestamp"],
                f"{f['predicted_cpu_usage']:.2f}",
                d["status"],
                f"{len(d['drifted_features'])}/{evaluated}",
                fmt(d.get("max_psi"), "{:.3f}"),
                fmt(d.get("min_ks_pvalue"), "{:.2e}"),
                fmt(v.get("mae"), "{:.3f}"),
            ]
        )
    return render_table(
        ["Box", "System", "Service Class", "Forecast (UTC)", "CPU %", "Drift", "Drifted", "Max PSI", "Min KS p", "OOF MAE"],
        rows,
        ["l", "l", "l", "l", "r", "l", "r", "r", "r", "r"],
    )


# --------------------------------------------------------------------------- #
# Suites
# --------------------------------------------------------------------------- #
def test_ops(ctx: Context) -> None:
    c = ctx.checks
    r = ctx.client.get("/healthz")
    c.record("GET /healthz -> 200", r.status_code == 200 and body_of(r).get("status") == "ok", f"HTTP {r.status_code}")

    r = ctx.client.get("/readyz")
    loaded = body_of(r).get("models_loaded")
    c.record(
        "GET /readyz -> 200, all artifacts loaded",
        r.status_code == 200 and loaded == len(ctx.artifacts),
        f"models_loaded={loaded}, artifacts on disk={len(ctx.artifacts)}",
    )

    r = ctx.client.get(FORECAST_PATH)
    c.record(
        "GET on forecast route -> 405 structured error",
        r.status_code == 405 and error_of(r).get("code") == "HTTP_ERROR",
        brief(r),
    )

    r = ctx.client.get("/healthz", headers={"X-Request-ID": "bad id with spaces!"})
    rid = r.headers.get("X-Request-ID", "")
    c.record(
        "Malformed X-Request-ID replaced by generated ID",
        re.fullmatch(r"[0-9a-f]{32}", rid) is not None,
        rid,
    )


def test_happy_path(ctx: Context) -> None:
    c = ctx.checks
    key = ctx.stable_keys[0]
    history = window_frame(ctx.stable, key, LIVE_END - WINDOW * STEP, LIVE_END)
    payload = to_payload(history, key)
    rid = "smoke-happy-001"

    resp = post(ctx, {"series": [payload]}, {"X-Request-ID": rid})
    if not c.record("Forecast: 200 on valid stable window", resp.status_code == 200, brief(resp)):
        return
    body = resp.json()
    forecasts = body["forecasts"]

    top = {"request_id", "generated_at", "frequency", "horizon_steps", "drift_status", "thresholds", "forecasts"}
    c.record("Forecast: response contract complete", top <= set(body) and len(forecasts) == 1, f"{len(body)} top-level keys")
    c.record(
        "Forecast: request ID echoed (header + body)",
        resp.headers.get("X-Request-ID") == rid and body["request_id"] == rid,
        rid,
    )

    f = forecasts[0]
    c.record("Forecast: keyed to requested series", (f["box"], f["system"], f["service_class"]) == key, "/".join(key))

    expected_ts = history[TS].max() + body["horizon_steps"] * STEP
    got_ts = pd.Timestamp(f["forecast_timestamp"]).tz_convert("UTC").tz_localize(None)
    c.record("Forecast: timestamp = last observation + horizon", got_ts == expected_ts, f"{got_ts} (expected {expected_ts})")

    pred = f["predicted_cpu_usage"]
    c.record("Forecast: prediction finite, within [0, 100]", finite(pred) and 0.0 <= pred <= 100.0, f"{pred}")

    artifact = ctx.artifacts[key]
    expected = offline_forecast(artifact, history)
    c.record(
        "Parity: API prediction == offline pipeline + model",
        abs(pred - expected) < 1e-4,
        f"api={pred:.6f} offline={expected:.6f}",
    )

    drift = f["drift"]
    feats = drift.get("features") or []
    names = [x["feature"] for x in feats]
    n_points = len(payload["timestamps"])
    c.record(
        "Drift: placeholder row excluded from window",
        drift["n_window_rows"] == n_points - ctx.warmup,
        f"{drift['n_window_rows']} rows = {n_points} points - warmup {ctx.warmup}",
    )
    c.record(
        "Drift: KS p-value and PSI reported for every feature",
        bool(feats) and all(x["evaluated"] and finite(x["psi"]) and finite(x["ks_pvalue"]) for x in feats),
        f"{len(feats)} features",
    )
    leaked = [n for n in names if n in UNMONITORED or n.startswith(CALENDAR_PREFIXES)]
    c.record(
        "Drift: roll_*_24 and calendar terms not monitored",
        bool(names) and not leaked,
        f"{len(names)} monitored" if not leaked else f"still monitored: {leaked}",
    )
    c.record(
        "Drift: stable window is not a retrain trigger",
        drift["status"] in ("NOMINAL", "WATCH") and finite(drift.get("max_psi")) and finite(drift.get("min_ks_pvalue")),
        drift_summary(drift),
    )

    val = f.get("validation") or {}
    c.record(
        "Validation: out-of-fold MAE/RMSE attached",
        all(k in val for k in ("mae", "rmse", "n_folds")) and val["mae"] > 0,
        f"mae={val.get('mae')}, rmse={val.get('rmse')}, folds={val.get('n_folds')}",
    )

    again = post(ctx, {"series": [payload], "include_feature_detail": False})
    again_f = body_of(again).get("forecasts", [{}])[0]
    c.record(
        "Forecast: deterministic on repeat request",
        again.status_code == 200 and again_f.get("predicted_cpu_usage") == pred,
        brief(again),
    )
    c.record(
        "include_feature_detail=false omits feature block",
        again.status_code == 200 and again_f.get("drift", {}).get("features") is None,
        "features=null",
    )


def test_missing_data(ctx: Context) -> None:
    c = ctx.checks
    payload = live_payload(ctx.stable, ctx.stable_keys[0])
    total = len(payload["timestamps"])
    for i in (50, 51, 100):
        payload["cpu_usage"][i] = None
    for i in (71, 70):
        del payload["timestamps"][i]
        del payload["cpu_usage"][i]

    resp = post(ctx, {"series": [payload]})
    if not c.record("Missing data: 200 with nulls and dropped rows", resp.status_code == 200, brief(resp)):
        return
    f = resp.json()["forecasts"][0]
    expected = total - 2 - 3
    c.record(
        "Missing data: n_observations counts non-null points",
        f["n_observations"] == expected and finite(f["predicted_cpu_usage"]),
        f"n_observations={f['n_observations']} (expected {expected})",
    )


def test_drift_regimes(ctx: Context) -> None:
    c = ctx.checks
    stable_a, stable_b = ctx.stable_keys[:2]
    body = {
        "series": [
            live_payload(ctx.stable, stable_a),
            live_payload(ctx.stable, stable_b),
            live_payload(ctx.shifted, ctx.spiked_key),
        ]
    }
    resp = post(ctx, body)
    if not c.record("Batch: 200 for stable + shifted series", resp.status_code == 200, brief(resp)):
        return
    data = resp.json()
    forecasts = data["forecasts"]
    ctx.report = render_forecasts(forecasts)

    order = [(f["box"], f["system"], f["service_class"]) for f in forecasts]
    by_key = dict(zip(order, forecasts))
    c.record(
        "Batch: one forecast per series, order preserved",
        order == [stable_a, stable_b, ctx.spiked_key],
        f"{len(order)} forecasts",
    )
    c.record(
        "Batch: thresholds mirror ForecastConfig",
        abs(data["thresholds"]["ks_alpha"] - ctx.config.drift_alpha_threshold) < 1e-12
        and abs(data["thresholds"]["psi"] - ctx.config.psi_threshold) < 1e-12,
        f"ks_alpha={data['thresholds']['ks_alpha']}, psi={data['thresholds']['psi']}",
    )

    spiked = by_key.get(ctx.spiked_key, {}).get("drift", {})
    c.record(
        "Drift: injected regime shift -> RETRAIN_TRIGGERED",
        spiked.get("status") == "RETRAIN_TRIGGERED",
        clip(drift_summary(spiked)),
    )
    for key in (stable_a, stable_b):
        drift = by_key.get(key, {}).get("drift", {})
        c.record(
            f"Drift: {key[0]}/{key[1]}/{key[2]} not RETRAIN_TRIGGERED",
            bool(drift) and drift["status"] != "RETRAIN_TRIGGERED",
            clip(drift_summary(drift)),
        )
        c.record(
            f"Drift: {key[0]}/{key[1]}/{key[2]} reads NOMINAL",
            drift.get("status") == "NOMINAL",
            clip(drift_summary(drift)),
            blocking=False,
        )
    c.record(
        "Drift: aggregate status = RETRAIN_TRIGGERED",
        data["drift_status"] == "RETRAIN_TRIGGERED",
        data["drift_status"],
    )


def test_not_found(ctx: Context) -> None:
    c = ctx.checks
    healthy = live_payload(ctx.stable, ctx.stable_keys[0])
    rejected = live_payload(ctx.stable, ERRATIC_KEY)
    rid = "smoke-404-001"

    resp = post(ctx, {"series": [rejected]}, {"X-Request-ID": rid})
    err = check_error(ctx, "Gate-rejected series -> 404 MODEL_NOT_FOUND", resp, 404, "MODEL_NOT_FOUND")
    c.record("404 details identify the offending series", error_keys(err) == [ERRATIC_KEY], str(error_keys(err)))
    c.record(
        "Request ID propagates on error responses",
        err.get("request_id") == rid and resp.headers.get("X-Request-ID") == rid,
        rid,
    )

    resp = post(ctx, {"series": [healthy, rejected]})
    err = check_error(ctx, "Mixed batch -> 404 listing only the missing series", resp, 404, "MODEL_NOT_FOUND")
    c.record("Mixed batch: healthy series not reported missing", error_keys(err) == [ERRATIC_KEY], str(error_keys(err)))

    unknown = {**healthy, "box": "BOX99", "system": "SYSZ", "service_class": "NO_SUCH_CLASS"}
    check_error(ctx, "Unknown series key -> 404 MODEL_NOT_FOUND", post(ctx, {"series": [unknown]}), 404, "MODEL_NOT_FOUND")


def test_insufficient_history(ctx: Context) -> None:
    c = ctx.checks
    key = ctx.stable_keys[0]

    tiny = live_payload(ctx.stable, key, hours=10)
    err = check_error(
        ctx, "Short history (10 pts) -> 422 INSUFFICIENT_DATA", post(ctx, {"series": [tiny]}),
        422, "INSUFFICIENT_DATA", "too few grid steps",
    )
    observed = [d.get("observed_points") for d in err.get("details", [])]
    c.record("Short-history details report observed points", observed == [10], str(observed))

    mid = live_payload(ctx.stable, key, hours=40)
    check_error(
        ctx, "Drift-blind window (40 pts) -> 422 INSUFFICIENT_DATA", post(ctx, {"series": [mid]}),
        422, "INSUFFICIENT_DATA", "drift window",
    )

    resp = post(ctx, {"series": [mid], "require_drift_assessment": False})
    f = body_of(resp).get("forecasts", [{}])[0]
    c.record(
        "require_drift_assessment=false -> 200, drift INSUFFICIENT_DATA",
        resp.status_code == 200
        and f.get("drift", {}).get("status") == "INSUFFICIENT_DATA"
        and finite(f.get("predicted_cpu_usage")),
        brief(resp) if resp.status_code != 200 else f"n_window_rows={f['drift']['n_window_rows']}",
    )


def test_temporal_integrity(ctx: Context) -> None:
    key = ctx.stable_keys[0]
    base = window_frame(ctx.stable, key, LIVE_END - WINDOW * STEP, LIVE_END)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor(FREQ)

    future = base.copy()
    future[TS] = pd.date_range(end=now + pd.Timedelta(days=2), periods=len(base), freq=FREQ)
    check_error(
        ctx, "Lookahead: future-dated series -> 422", post(ctx, {"series": [to_payload(future, key)]}),
        422, "TEMPORAL_INTEGRITY_VIOLATION", "future-dated",
    )

    leaked = to_payload(base, key)
    leaked["timestamps"].append((now + pd.Timedelta(days=3)).isoformat())
    leaked["cpu_usage"].append(99.0)
    check_error(
        ctx, "Lookahead: single future point appended -> 422", post(ctx, {"series": [leaked]}),
        422, "TEMPORAL_INTEGRITY_VIOLATION", "future-dated",
    )

    collision = to_payload(base, key)
    collision["timestamps"].append((base[TS].max() + pd.Timedelta(minutes=30)).isoformat())
    collision["cpu_usage"].append(50.0)
    check_error(
        ctx, "Grid collision: two points in one slot -> 422", post(ctx, {"series": [collision]}),
        422, "TEMPORAL_INTEGRITY_VIOLATION", "grid slot",
    )

    span = {
        **to_payload(base.head(0), key),
        "timestamps": ["2000-01-01T00:00:00", "2026-01-25T23:00:00"],
        "cpu_usage": [10.0, 20.0],
    }
    check_error(
        ctx, "Absurd timestamp span -> 422", post(ctx, {"series": [span]}),
        422, "TEMPORAL_INTEGRITY_VIOLATION", "span exceeds",
    )


def test_schema(ctx: Context) -> None:
    base = live_payload(ctx.stable, ctx.stable_keys[0], hours=48)

    short = copy.deepcopy(base)
    short["cpu_usage"].pop()
    negative = copy.deepcopy(base)
    negative["cpu_usage"][3] = -5.0
    bad_ts = copy.deepcopy(base)
    bad_ts["timestamps"][0] = "not-a-date"

    cases = [
        ("mismatched array lengths", {"series": [short]}),
        ("negative cpu_usage", {"series": [negative]}),
        ("blank system name", {"series": [{**base, "system": "   "}]}),
        ("unknown field", {"series": [{**base, "unexpected": 1}]}),
        ("unparseable timestamp", {"series": [bad_ts]}),
        ("duplicate series keys", {"series": [base, base]}),
        ("empty series list", {"series": []}),
    ]
    for label, body in cases:
        check_error(ctx, f"Schema: {label} -> 422", post(ctx, body), 422, "INVALID_REQUEST")

    resp = ctx.client.post(FORECAST_PATH, content=b"{not json", headers={"Content-Type": "application/json"})
    check_error(ctx, "Schema: malformed JSON body -> 422", resp, 422, "INVALID_REQUEST")


SUITES: List[Tuple[str, Callable[[Context], None]]] = [
    ("Ops endpoints", test_ops),
    ("Forecast contract (happy path)", test_happy_path),
    ("Missing-data tolerance", test_missing_data),
    ("Drift regimes (batch)", test_drift_regimes),
    ("404 model resolution", test_not_found),
    ("422 insufficient history", test_insufficient_history),
    ("422 temporal integrity", test_temporal_integrity),
    ("422 schema validation", test_schema),
]


def run_suite(ctx: Context, title: str, fn: Callable[[Context], None]) -> None:
    before = len(ctx.checks.items)
    try:
        fn(ctx)
    except Exception as exc:  # noqa: BLE001
        ctx.checks.record(f"{title}: unexpected {type(exc).__name__}", False, clip(str(exc)))
    new = ctx.checks.items[before:]
    print(f"  {title:<34} {sum(x.passed for x in new):>2}/{len(new)} checks passed")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gateway smoke test.")
    parser.add_argument("--base-url", default=None, help="Target a running server instead of the in-process app.")
    parser.add_argument("--artifact-dir", default="artifacts/models")
    parser.add_argument("--seed", type=int, default=42, help="Must match the run_pipeline.py training seed.")
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--log-level", default="CRITICAL", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args(argv)


def open_client(args: argparse.Namespace) -> httpx.Client:
    if args.base_url:
        return httpx.Client(base_url=args.base_url, timeout=60.0)
    from fastapi.testclient import TestClient

    from src.main import app

    return TestClient(app, raise_server_exceptions=False)


def wait_for_server(client: httpx.Client, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        try:
            if client.get("/healthz").status_code == 200:
                return True
        except httpx.TransportError:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = ROOT / artifact_dir

    section("Environment")
    paths = sorted(artifact_dir.glob("*.joblib"))
    if not paths:
        print(f"No artifacts in {artifact_dir}. Run: python scripts/run_pipeline.py")
        return 2
    artifacts: Dict[SeriesKey, Dict[str, Any]] = {}
    for path in paths:
        payload = joblib.load(path)
        artifacts[tuple(payload["key"])] = payload

    config = ForecastConfig.from_dict(next(iter(artifacts.values()))["config"])
    warmup = build_feature_pipeline(config=config, schema=SCHEMA, freq=FREQ).warmup_rows
    stable_keys = [k for k in series_keys() if k in artifacts and k not in SPIKED_KEYS][:2]
    spiked = [k for k in sorted(SPIKED_KEYS) if k in artifacts]
    if len(stable_keys) < 2 or not spiked:
        print("Artifacts do not cover the synthetic series grid. Re-run: python scripts/run_pipeline.py")
        return 2

    # Must be set before `src.main` is imported (lazily, in open_client).
    os.environ["MODEL_ARTIFACT_DIR"] = str(artifact_dir)
    os.environ["LOG_LEVEL"] = args.log_level

    stable = synthesize_telemetry(TRAIN_DAYS + LIVE_DAYS, args.seed)
    shifted = inject_batch_spike(stable, SPIKED_KEYS, LIVE_START, args.seed)

    print(f"target          : {args.base_url or 'in-process ASGI app (lifespan enabled)'}")
    print(f"artifacts       : {len(paths)} in {artifact_dir}")
    print(f"stable series   : {', '.join('/'.join(k) for k in stable_keys)}")
    print(f"shifted series  : {'/'.join(spiked[0])}")
    print(f"rejected series : {'/'.join(ERRATIC_KEY)}")
    print(f"window          : {WINDOW} hourly points ending {LIVE_END}")

    checks = Checklist()
    checks.record(
        "Precondition: rejected series has no artifact",
        ERRATIC_KEY not in artifacts,
        "/".join(ERRATIC_KEY),
    )

    with open_client(args) as client:
        if args.base_url and not wait_for_server(client, args.startup_timeout):
            print(f"\nServer unreachable at {args.base_url}. Start it with: uvicorn src.main:app --port 8000")
            return 2
        ctx = Context(
            client=client,
            checks=checks,
            stable=stable,
            shifted=shifted,
            artifacts=artifacts,
            config=config,
            warmup=warmup,
            stable_keys=stable_keys,
            spiked_key=spiked[0],
        )
        section("Gateway checks")
        for title, fn in SUITES:
            run_suite(ctx, title, fn)

    if ctx.report:
        section("Forecast batch: stable + regime-shifted series")
        print(ctx.report)

    section("Verdict")
    print(checks.render())
    print(f"\nelapsed: {time.perf_counter() - started:.1f}s")
    print(f"RESULT: {'PASS' if checks.ok else 'FAIL'}")
    return 0 if checks.ok else 1


if __name__ == "__main__":
    sys.exit(main())