# Mainframe CPU Usage Forecasting & MLOps Pipeline

> An end-to-end, high-performance distributed time-series MLOps pipeline engineered to predict hourly mainframe utilization across a multi-tenant corporate infrastructure matrix with zero train-serve skew and online statistical drift monitoring telemetry.

---

## 1. Executive Abstract & Problem Statement

Mainframe capacity planning is a partitioned problem, not a single-series one. Every Box, System, and Service Class combination carries its own workload signature, and a forecast built at the wrong granularity either masks a hot partition inside a healthy average or over-provisions an entire Box to cover one noisy Service Class.

Two failure modes make naive forecasting dangerous in this environment:

- **Lookahead leakage in validation.** Random K-Fold splits place future observations in the training set while scoring against the past. A model validated this way reports strong offline accuracy and then degrades in production, because it was never actually tested on the one thing that matters: predicting data it has not seen yet.
- **Silent regime shifts.** A batch workload gets rescheduled, a Service Class migrates Boxes, a nightly job changes shape — and a static model keeps producing confident, wrong forecasts with no signal that anything changed.

This architecture treats both as first-class engineering problems rather than afterthoughts. A purged, forward-chaining validation gate enforces strict temporal ordering before any model is allowed to export. An online, dual-signal drift monitor (Kolmogorov–Smirnov plus Population Stability Index) continuously compares live inference windows against each model's training-time baseline, so a decaying series is flagged at the partition level — not discovered after an incident.

---

## 2. Core Technical Stack

| Layer | Tools |
|---|---|
| Predictive Engine | XGBoost (gradient-boosted decision trees) |
| Math, Vectors & Core ML | NumPy, Pandas (>=2.2.0, required for `"h"` frequency regularization), Scikit-Learn, SciPy (`stats`) |
| Concurrency & Serialization | joblib (parallel per-series model fitting and artifact persistence) |
| Web Ingress / ASGI Layer | FastAPI, Uvicorn, Pydantic v2 |
| Cloud Hosting Infrastructure | Render (Singapore regional container node) |

---

## 3. System Architecture & Data Flow

```text
[Raw Historical Telemetry Logs]
      (Box, System, Service Class, timestamp, cpu_usage)
              │
              ▼
[Deterministic Feature Pipeline Grid]
      per-series partitioning → gap-free hourly grid
      → lags → shifted rolling stats → cyclic sin/cos encodings
              │
              ▼
[Forward-Chaining Parallel XGBoost Training]  (joblib)
      purged expanding-window CV → quality gate → final refit
              │
              ├──────────────────────────────┐
              ▼                              ▼
   [Artifact Serialization]        [Baseline Feature Distributions]
      (.joblib per series)          (quantile bins + KS reference sample)
              │                              │
              ▼                              ▼
        [FastAPI Serving Ingress]  ◄──  [Online Dual-Signal Drift Monitor]
         POST /api/v1/forecast             (KS two-sample test + vectorized PSI)
              │
              ▼
   [Predicted CPU Utilization + Drift Status Payload]
```

---

## 4. Core Technical Implementation

- **Feature Pipeline Layer.** Auto-regressive lags at offsets `1, 2, 3, 4, 24` capture short-term momentum and the daily batch cycle. Rolling mean and standard deviation are computed over `3h, 6h, 12h, 24h` windows, each shifted by the forecast horizon so the current observation never enters its own summary statistic. Hour-of-day and day-of-week are encoded as `sin(2π·v/period)` / `cos(2π·v/period)` pairs, which preserve calendar wrap-around (hour 23 sits next to hour 0) in a way integer encodings cannot.
- **Forward-Chaining Validation Cross-Splitter.** Five expanding-window folds, each purged so training data strictly precedes its validation block by at least one forecast horizon. Training set size grows monotonically fold-to-fold; validation blocks are contiguous, disjoint, and anchored to the end of the series.
- **In-Memory Model Registry.** Artifacts load once during FastAPI's `lifespan` startup hook into a process-local registry. Inference reads are served from memory under a lightweight lock around booster access, so concurrent ASGI requests never trigger disk I/O on the hot path.

---

## 5. Key Features & Engineering Decisions

- **Automated Quality Gate Enforcement.** Pooled out-of-fold MAE, RMSE, and a persistence-ratio check (model error versus a naive "same as last period" forecast) are evaluated before any model is exported. The adversarial synthetic channel `BOX02/SYSB/BATCH_LOW` — engineered with dominant noise variance — is correctly rejected at the gate and never reaches production artifacts.
- **Train-Serve Skew Elimination.** The serving path appends one placeholder row per series at `last_timestamp + horizon` and runs it through the identical `FeaturePipeline` used in training. Every lag and rolling feature is shifted by at least the forecast horizon, so the placeholder value structurally cannot leak into its own feature row. Verified parity between in-process offline reconstruction and live API output: **0.000000% deviation** to six decimal places.
- **Calibrated False-Alarm Isolation.** The 24-hour rolling mean and standard deviation carry roughly one effective independent sample per day. Against a 7-day inference window, both KS and PSI treat their natural low-frequency variance as a false drift signal. These two features are explicitly excluded from the drift policy's monitored set, which eliminated spurious `WATCH` verdicts on stable, unshifted series while preserving full sensitivity to genuine regime shifts.

---

## 6. How to Run & Reproduce

**PowerShell (Windows)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m scripts.run_pipeline
python scripts/test_gateway.py
```

**Bash (macOS / Linux)**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m scripts.run_pipeline
python scripts/test_gateway.py
```

Running against a live Uvicorn instance instead of the in-process ASGI client:
```bash
uvicorn src.main:app --host 127.0.0.1 --port 8000
python scripts/test_gateway.py --base-url http://127.0.0.1:8000
```

---

## 7. Testing & Validation

The core leakage guarantee is verified by direct perturbation rather than by inspection. `scripts/run_pipeline.py` injects an arbitrary **+25.0 point spike** into `y[t]` for a target series, reruns the fitted `FeaturePipeline`, and asserts two properties simultaneously:

- `features[t]` is bit-for-bit unchanged — the perturbation has zero lookahead effect on its own row.
- `features[t+1]` **does** change — proving the lag and rolling-window mechanics are live and correctly propagate history forward, so the first assertion cannot pass vacuously.

A companion isolation check confirms every other series' feature frame is untouched by the perturbation, verifying strict per-key partitioning across the Box → System → Service Class hierarchy.

This structural probe is layered under 22 automated checks in the training/drift runner and 52 automated checks in the gateway smoke harness (`scripts/test_gateway.py`), covering forward-chaining fold geometry, quality-gate accept/reject behavior, artifact round-trip fidelity, HTTP error-contract mapping, and drift-injection response.

---

## 8. Performance & Measured Results

| Metric | Result |
|---|---|
| Pooled out-of-fold MAE (learnable channels) | **1.564%** CPU utilization |
| Pooled out-of-fold RMSE (learnable channels) | **1.968%** CPU utilization |
| Exportable series | 11 / 12 (1 adversarial channel correctly gate-rejected) |
| Injected regime-shift drift capture (`BOX01/SYSA/BATCH_LOW`) | **11 / 11** monitored features flagged drifting |
| Max PSI on injected shift (`BOX01/SYSA/BATCH_LOW`) | **6.455** (threshold: 0.2) |
| Min KS p-value on injected shift (`BOX01/SYSA/BATCH_LOW`) | **1.06e-63** (threshold: 0.05) |
| Aggregate verdict on injected regime shift | `RETRAIN_TRIGGERED` |
| Aggregate verdict on stable, unshifted series | `NOMINAL`, 0 monitored features flagged |

All figures are measured against synthetic telemetry with a controlled diurnal signal-to-noise ratio and should be read as a validation of pipeline correctness, not as a claim about real-world mainframe forecast accuracy. Figures are drawn from the post-calibration gateway smoke test (`scripts/test_gateway.py`), which exercises one of the two injected-shift series (`BOX01/SYSA/BATCH_LOW`); the second (`BOX02/SYSA/CICS_PRD`) has not been re-measured since the `roll_mean_24`/`roll_std_24` exclusion and is omitted here rather than reported from stale pre-calibration numbers.

---

## 9. Known Limitations

- **Autoregressive History Floor.** The system requires a strict minimum lookback history anchor of 24 regularized grid steps to warm up features and 54 steps to clear the drift sample threshold floor, throwing structured HTTP 422 `INSUFFICIENT_DATA` error codes otherwise.
- **Hardware / Scaling Constraints.** Synthetic data bounds are currently scaled to 25 unique infrastructure series keys and 2000 points per request to fit within free-tier container memory bounds (512MB RAM limits).
- **Synthetic data only.** All models are trained on procedurally generated telemetry with an engineered diurnal cycle. No real mainframe SMF/RMF data has been used or validated against.
- **No authentication layer yet.** The forecast endpoint is currently open to any caller with the URL; see Roadmap.

---

## 10. Roadmap & Future Improvements

- Migration from standard Python pickles (`joblib`) to native, lightweight XGBoost JSON serialization models.
- Implementation of secure, token-based API key validation guardrails on ingress endpoints to prevent open cluster consumption.
- Integration of windowed list virtualization and debounced input anchors via a dedicated React/Next.js monitoring UI dashboard utilizing Advanced Frontend System Design paradigms.

---

## Author

Rishabh Bhawsar — AI Systems & MLOps Engineer
GitHub: [github.com/rishabhbhawsar](https://github.com/rishabhbhawsar)
LinkedIn: [linkedin.com/in/rishabh-bhawsar-409098262](https://linkedin.com/in/rishabh-bhawsar-409098262)

## License

MIT License