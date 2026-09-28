"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { FC } from "react";
import { ForecastChart, type ForecastPoint } from "@/components/ForecastChart";
import { TopologySelector, type TopologyCoordinate } from "@/components/TopologySelector";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

type DriftScenario = "stable" | "shift";
type LogLevel = "INFO" | "OK" | "WARN" | "ERROR";
type Tone = "emerald" | "cyan" | "amber" | "zinc";
type RequestState = "idle" | "loading" | "ok" | "error";
type DriftStatusName = "NOMINAL" | "WATCH" | "RETRAIN_TRIGGERED" | "INSUFFICIENT_DATA";

interface LogEntry {
  id: number;
  time: string;
  level: LogLevel;
  message: string;
}

interface DriftStatus {
  status: DriftStatusName;
  n_window_rows: number;
  drifted_fraction: number;
  max_psi: number | null;
  min_ks_pvalue: number | null;
  drifted_features: string[];
  null_alert_features: string[];
  missing_features: string[];
}

interface ValidationInfo {
  mae: number;
  rmse: number;
  n_folds: number;
  naive_ratio: number | null;
}

interface SeriesForecast {
  box: string;
  system: string;
  service_class: string;
  forecast_timestamp: string;
  predicted_cpu_usage: number;
  n_observations: number;
  drift: DriftStatus;
  validation: ValidationInfo | null;
}

interface ForecastResponse {
  request_id: string;
  generated_at: string;
  frequency: string;
  horizon_steps: number;
  drift_status: DriftStatusName;
  thresholds: { ks_alpha: number; psi: number };
  forecasts: SeriesForecast[];
}

interface ApiErrorBody {
  error?: {
    code?: string;
    message?: string;
    details?: unknown[];
    request_id?: string;
  };
}

interface ForecastRequestBody {
  series: Array<{
    box: string;
    system: string;
    service_class: string;
    timestamps: string[];
    cpu_usage: number[];
  }>;
  require_drift_assessment: boolean;
  include_feature_detail: boolean;
}

interface WorkloadProfile {
  base: number;
  amplitude: number;
  peakHour: number;
  noiseSd: number;
}

interface SimulatedSeries {
  timestamps: string[];
  values: number[];
}

interface ChartState {
  points: ForecastPoint[];
  origin: string;
}

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */

const API_BASE = (
  process.env.NEXT_PUBLIC_MAINFRAME_API_URL || "https://mainframe-cpu-forecaster.onrender.com"
).replace(/\/+$/, "");
const FORECAST_ENDPOINT = `${API_BASE}/api/v1/forecast`;

/** Keys must match the series the gateway was trained on. */
const BOXES = ["BOX01", "BOX02"] as const;
const SYSTEMS = ["SYSA", "SYSB"] as const;
const SERVICE_CLASSES = ["CICS_PRD", "DB2_HIGH", "BATCH_LOW"] as const;

const HOUR_MS = 3_600_000;
const WINDOW_HOURS = 168;
const CHART_HISTORY_HOURS = 72;
const REQUEST_TIMEOUT_MS = 90_000;
const SLOW_NOTICE_MS = 8_000;
const MAX_LOG_ENTRIES = 200;
const MAX_ERROR_CHARS = 600;

const DEFAULT_PROFILE: WorkloadProfile = { base: 45, amplitude: 24, peakHour: 14, noiseSd: 1.5 };
const PROFILES: Record<string, WorkloadProfile> = {
  CICS_PRD: DEFAULT_PROFILE,
  DB2_HIGH: { base: 40, amplitude: 20, peakHour: 13, noiseSd: 1.5 },
  BATCH_LOW: { base: 34, amplitude: 22, peakHour: 2, noiseSd: 2.0 },
};
const WEEKEND_FACTOR = 0.9;
const AR_PHI = 0.5;
const SHIFT_LEVEL = 25;
const SHIFT_BURST = 20;
const SHIFT_NOISE_SD = 5;

const TONE_CLASSES: Record<Tone, string> = {
  emerald: "border-emerald-800/60 bg-emerald-950/30 text-emerald-400",
  cyan: "border-cyan-800/60 bg-cyan-950/30 text-cyan-400",
  amber: "border-amber-800/60 bg-amber-950/30 text-amber-400",
  zinc: "border-[#27272a] bg-[#0c0c0e] text-zinc-400",
};

const LEVEL_CLASSES: Record<LogLevel, string> = {
  INFO: "text-cyan-400",
  OK: "text-emerald-400",
  WARN: "text-amber-400",
  ERROR: "text-red-400",
};

const REQUEST_BADGE: Record<RequestState, { label: string; className: string }> = {
  idle: { label: "idle", className: "border-[#27272a] text-zinc-500" },
  loading: { label: "requesting", className: "border-amber-800/60 bg-amber-950/40 text-amber-400" },
  ok: { label: "200 ok", className: "border-emerald-800/60 bg-emerald-950/40 text-emerald-400" },
  error: { label: "error", className: "border-red-900/70 bg-red-950/40 text-red-400" },
};

/* ------------------------------------------------------------------ */
/* Simulated telemetry (client-side; submitted to the live gateway)    */
/* ------------------------------------------------------------------ */

function hashSeed(text: string): number {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function mulberry32(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand: () => number): number {
  const u = Math.max(rand(), 1e-12);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Hourly series ending at the current UTC hour; the last observation is never future-dated. */
function buildHistory(coordinate: TopologyCoordinate, scenario: DriftScenario): SimulatedSeries {
  const profile = PROFILES[coordinate.serviceClass] ?? DEFAULT_PROFILE;
  const rand = mulberry32(hashSeed(`${coordinate.box}/${coordinate.system}/${coordinate.serviceClass}`));
  const endMs = Math.floor(Date.now() / HOUR_MS) * HOUR_MS;
  const innovationSd = profile.noiseSd * Math.sqrt(1 - AR_PHI ** 2);

  const timestamps: string[] = [];
  const values: number[] = [];
  let noise = gaussian(rand) * profile.noiseSd;

  for (let i = 0; i < WINDOW_HOURS; i += 1) {
    const date = new Date(endMs - (WINDOW_HOURS - 1 - i) * HOUR_MS);
    const hour = date.getUTCHours();
    const day = date.getUTCDay();
    const phase = (2 * Math.PI * (hour - profile.peakHour)) / 24;
    const seasonal = profile.amplitude * (Math.cos(phase) + 0.25 * Math.cos(2 * phase + 0.6));
    const weekend = day === 0 || day === 6 ? WEEKEND_FACTOR : 1;

    noise = AR_PHI * noise + innovationSd * gaussian(rand);
    let value = (profile.base + seasonal) * weekend + noise;
    if (scenario === "shift") {
      value += SHIFT_LEVEL + (hour < 6 ? SHIFT_BURST : 0) + gaussian(rand) * SHIFT_NOISE_SD;
    }

    timestamps.push(date.toISOString());
    values.push(Number(Math.min(100, Math.max(0, value)).toFixed(3)));
  }
  return { timestamps, values };
}

function toChartHistory(series: SimulatedSeries): ForecastPoint[] {
  return series.timestamps.slice(-CHART_HISTORY_HOURS).map((timestamp, index) => ({
    timestamp,
    actual: series.values[series.values.length - CHART_HISTORY_HOURS + index] ?? null,
    predicted: null,
  }));
}

/* ------------------------------------------------------------------ */
/* Network helpers                                                     */
/* ------------------------------------------------------------------ */

function truncate(text: string, limit: number): string {
  return text.length <= limit ? text : `${text.slice(0, limit - 3)}...`;
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

async function describeHttpError(response: Response): Promise<string> {
  const raw = await response.text().catch(() => "");
  let message = raw;
  try {
    const parsed = JSON.parse(raw) as ApiErrorBody;
    if (parsed.error) {
      const { code = "ERROR", message: text = "", details, request_id: requestId } = parsed.error;
      const detailText = details && details.length > 0 ? ` | details=${JSON.stringify(details)}` : "";
      const idText = requestId ? ` | request_id=${requestId}` : "";
      message = `${code}: ${text}${detailText}${idText}`;
    }
  } catch {
    /* non-JSON body (proxy or gateway error page) */
  }
  return `HTTP ${response.status} ${truncate(message || response.statusText, MAX_ERROR_CHARS)}`;
}

/* ------------------------------------------------------------------ */
/* Formatting                                                          */
/* ------------------------------------------------------------------ */

function fixed(value: number | null | undefined, digits: number, suffix = ""): string {
  return value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(digits)}${suffix}`;
}

function exponent(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "—" : value.toExponential(2);
}

function toneForStatus(status: DriftStatusName | null): Tone {
  if (status === "NOMINAL") return "cyan";
  if (status === "WATCH" || status === "RETRAIN_TRIGGERED") return "amber";
  return "zinc";
}

function verdictClasses(status: DriftStatusName | null): string {
  if (status === "NOMINAL") return "border-emerald-800 bg-emerald-950/40 text-emerald-400";
  if (status === "WATCH") return "border-amber-800 bg-amber-950/30 text-amber-400";
  if (status === "RETRAIN_TRIGGERED") return "border-amber-700 bg-amber-950/60 text-amber-300";
  return "border-[#27272a] bg-[#09090b] text-zinc-500";
}

/* ------------------------------------------------------------------ */
/* Presentational components                                           */
/* ------------------------------------------------------------------ */

const StatBlock: FC<{ label: string; value: string; detail: string; tone: Tone }> = ({
  label,
  value,
  detail,
  tone,
}) => (
  <div className={`rounded-sm border p-3 ${TONE_CLASSES[tone]}`}>
    <div className="font-mono text-[10px] uppercase tracking-widest opacity-70">{label}</div>
    <div className="mt-1 font-mono text-xl">{value}</div>
    <div className="mt-1 truncate font-mono text-[10px] opacity-60">{detail}</div>
  </div>
);

const ScenarioToggle: FC<{ value: DriftScenario; onChange: (next: DriftScenario) => void }> = ({
  value,
  onChange,
}) => {
  const options: ReadonlyArray<{ id: DriftScenario; label: string }> = [
    { id: "stable", label: "Stable window" },
    { id: "shift", label: "Injected regime shift" },
  ];
  return (
    <div className="inline-flex overflow-hidden rounded-sm border border-[#27272a] font-mono text-[11px]">
      {options.map((option) => {
        const active = option.id === value;
        const activeClass = option.id === "shift" ? "bg-amber-950/60 text-amber-400" : "bg-emerald-950/50 text-emerald-400";
        return (
          <button
            key={option.id}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(option.id)}
            className={`px-3 py-1.5 transition-colors ${
              active ? activeClass : "bg-[#09090b] text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
};

const LogMonitor: FC<{ logs: readonly LogEntry[]; state: RequestState }> = ({ logs, state }) => {
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const element = scrollRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [logs]);

  const badge = REQUEST_BADGE[state];

  return (
    <section className="flex h-96 flex-col rounded-md border border-[#27272a] bg-[#0c0c0e] p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="font-mono text-xs uppercase tracking-widest text-zinc-400">Operations Log</h2>
        <span className={`rounded-sm border px-2 py-0.5 font-mono text-[10px] ${badge.className}`}>
          {badge.label}
        </span>
      </div>
      <div
        ref={scrollRef}
        className="min-h-0 flex-1 space-y-1 overflow-y-auto rounded-sm border border-[#27272a] bg-[#09090b] p-3 font-mono text-[11px] leading-relaxed"
      >
        {logs.length === 0 ? (
          <div className="text-zinc-600">awaiting first request...</div>
        ) : (
          logs.map((entry) => (
            <div key={entry.id} className="break-words">
              <span className="text-zinc-600">{entry.time} </span>
              <span className={LEVEL_CLASSES[entry.level]}>[{entry.level}]</span>{" "}
              <span className={entry.level === "ERROR" ? "text-red-300" : "text-zinc-300"}>{entry.message}</span>
            </div>
          ))
        )}
      </div>
    </section>
  );
};

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function DashboardPage() {
  const [coordinate, setCoordinate] = useState<TopologyCoordinate>({
    box: "BOX01",
    system: "SYSA",
    serviceClass: "CICS_PRD",
  });
  const [scenario, setScenario] = useState<DriftScenario>("stable");
  const [requestState, setRequestState] = useState<RequestState>("idle");
  const [chart, setChart] = useState<ChartState>({ points: [], origin: "" });
  const [result, setResult] = useState<{ forecast: SeriesForecast; response: ForecastResponse } | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const abortRef = useRef<AbortController | null>(null);
  const logIdRef = useRef<number>(0);

  const pushLog = useCallback((level: LogLevel, message: string) => {
    logIdRef.current += 1;
    const entry: LogEntry = {
      id: logIdRef.current,
      time: new Date().toLocaleTimeString("en-GB"),
      level,
      message,
    };
    setLogs((previous) => [...previous, entry].slice(-MAX_LOG_ENTRIES));
  }, []);

  const runForecast = useCallback(async (): Promise<void> => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const isCurrent = (): boolean => abortRef.current === controller;

    const series = buildHistory(coordinate, scenario);
    const history = toChartHistory(series);
    const origin = history[history.length - 1]?.timestamp ?? "";
    setChart({ points: history, origin });
    setResult(null);
    setRequestState("loading");

    const key = `${coordinate.box}/${coordinate.system}/${coordinate.serviceClass}`;
    pushLog("INFO", "Intercepting infrastructure change context parameters...");
    pushLog("INFO", `Context: ${key} | scenario=${scenario} | ${WINDOW_HOURS} hourly points`);
    pushLog("INFO", "Egressing POST request packet to production XGBoost forecast cluster...");

    const body: ForecastRequestBody = {
      series: [
        {
          box: coordinate.box,
          system: coordinate.system,
          service_class: coordinate.serviceClass,
          timestamps: series.timestamps,
          cpu_usage: series.values,
        },
      ],
      require_drift_assessment: true,
      include_feature_detail: false,
    };

    let timedOut = false;
    const timeoutId = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, REQUEST_TIMEOUT_MS);
    const slowId = window.setTimeout(() => {
      if (isCurrent()) {
        pushLog("WARN", "No response yet; a free-tier instance can take up to ~60s to wake from idle.");
      }
    }, SLOW_NOTICE_MS);

    try {
      const response = await fetch(FORECAST_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response.ok) {
        const detail = await describeHttpError(response);
        if (!isCurrent()) return;
        pushLog("ERROR", detail);
        setRequestState("error");
        return;
      }

      const data = (await response.json()) as ForecastResponse;
      if (!isCurrent()) return;

      const forecast = data.forecasts?.[0];
      if (!forecast) {
        pushLog("ERROR", "Malformed response: forecasts array is empty.");
        setRequestState("error");
        return;
      }

      const forecastPoint: ForecastPoint = {
        timestamp: new Date(forecast.forecast_timestamp).toISOString(),
        actual: null,
        predicted: forecast.predicted_cpu_usage,
      };
      setChart({ points: [...history, forecastPoint], origin });
      setResult({ forecast, response: data });
      setRequestState("ok");

      const drift = forecast.drift;
      pushLog(
        "OK",
        `HTTP 200 | request_id=${data.request_id} | +${data.horizon_steps}h forecast ${fixed(
          forecast.predicted_cpu_usage,
          2,
          "%",
        )} at ${forecast.forecast_timestamp}`,
      );
      pushLog(
        "OK",
        `Drift ${drift.status} | ${drift.drifted_features.length} feature(s) drifted | max PSI ${fixed(
          drift.max_psi,
          3,
        )} | min KS p ${exponent(drift.min_ks_pvalue)}`,
      );
      if (drift.status === "RETRAIN_TRIGGERED") {
        pushLog("WARN", `Retrain recommended for ${key}: ${drift.drifted_features.join(", ")}`);
      }
    } catch (error) {
      if (!isCurrent()) return;
      if (isAbortError(error)) {
        if (!timedOut) return;
        pushLog("ERROR", `Request timed out after ${REQUEST_TIMEOUT_MS / 1000}s.`);
      } else {
        const reason = error instanceof Error ? error.message : String(error);
        pushLog(
          "ERROR",
          `Network failure: ${reason}. Check backend availability and that its CORS policy allows this origin.`,
        );
      }
      setRequestState("error");
    } finally {
      window.clearTimeout(timeoutId);
      window.clearTimeout(slowId);
    }
  }, [coordinate, scenario, pushLog]);

  useEffect(() => {
    void runForecast();
    return () => abortRef.current?.abort();
  }, [runForecast]);

  const drift = result?.forecast.drift ?? null;
  const validation = result?.forecast.validation ?? null;
  const thresholds = result?.response.thresholds ?? null;
  const status = drift?.status ?? null;
  const driftTone = toneForStatus(status);
  const loading = requestState === "loading";

  return (
    <main className="min-h-screen bg-[#09090b] px-6 py-8 text-zinc-100">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <header className="flex flex-col gap-1 border-b border-[#27272a] pb-4">
          <h1 className="font-mono text-lg font-semibold tracking-tight">
            Mainframe CPU Forecasting — Telemetry Console
          </h1>
          <p className="font-mono text-xs text-zinc-500">
            Box → System → Service Class forecasting with forward-chaining validation and dual-signal drift
            monitoring.{" "}
            <span className="text-zinc-600">
              The observed series is simulated in-browser and submitted to the live gateway; the forecast and
              drift statistics are computed by the deployed service.
            </span>
          </p>
        </header>

        <TopologySelector
          value={coordinate}
          onChange={setCoordinate}
          boxes={BOXES}
          systems={SYSTEMS}
          serviceClasses={SERVICE_CLASSES}
        />

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <ForecastChart data={chart.points} forecastOriginTimestamp={chart.origin} />
          </div>
          <LogMonitor logs={logs} state={requestState} />
        </div>

        <section className="rounded-md border border-[#27272a] bg-[#0c0c0e] p-4">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <h2 className="font-mono text-xs uppercase tracking-widest text-zinc-400">
              Statistical Drift Monitor
            </h2>
            <div className="flex items-center gap-3">
              <ScenarioToggle value={scenario} onChange={setScenario} />
              <button
                type="button"
                disabled={loading}
                onClick={() => void runForecast()}
                className="rounded-sm border border-[#27272a] bg-[#09090b] px-3 py-1.5 font-mono text-[11px] text-zinc-400 transition-colors hover:text-zinc-200 disabled:opacity-40"
              >
                Re-run
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
            <StatBlock
              label="Forecast +1h"
              value={fixed(result?.forecast.predicted_cpu_usage, 2, "%")}
              detail={result ? result.forecast.forecast_timestamp.slice(0, 16).replace("T", " ") + " UTC" : "awaiting response"}
              tone={result ? "emerald" : "zinc"}
            />
            <StatBlock
              label="Series OOF MAE"
              value={fixed(validation?.mae, 3, "%")}
              detail={validation ? `${validation.n_folds} forward-chaining folds` : "awaiting response"}
              tone={validation ? "emerald" : "zinc"}
            />
            <StatBlock
              label="Series OOF RMSE"
              value={fixed(validation?.rmse, 3, "%")}
              detail={validation ? `MAE/naive ${fixed(validation.naive_ratio, 3)}` : "awaiting response"}
              tone={validation ? "emerald" : "zinc"}
            />
            <StatBlock
              label="PSI (max)"
              value={fixed(drift?.max_psi, 3)}
              detail={
                drift && thresholds
                  ? `${drift.drifted_features.length} drifted · threshold ${thresholds.psi}`
                  : "awaiting response"
              }
              tone={driftTone}
            />
            <StatBlock
              label="KS p-value (min)"
              value={exponent(drift?.min_ks_pvalue)}
              detail={thresholds ? `alpha ${thresholds.ks_alpha}` : "awaiting response"}
              tone={driftTone}
            />
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-[#27272a] pt-3">
            <span className="font-mono text-[10px] text-zinc-600">
              Reference: pooled OOF across 11 exportable series — MAE 1.564% · RMSE 1.968%
            </span>
            <div className="flex items-center gap-3">
              <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
                aggregate verdict
              </span>
              <span className={`rounded-sm border px-2 py-0.5 font-mono text-xs ${verdictClasses(status)}`}>
                {status ?? (loading ? "REQUESTING" : "—")}
              </span>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}