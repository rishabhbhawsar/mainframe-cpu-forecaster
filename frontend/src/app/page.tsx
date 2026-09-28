"use client";

import { useMemo, useState } from "react";
import type { FC } from "react";
import { ForecastChart, type ForecastPoint } from "@/components/ForecastChart";
import { TopologySelector, type TopologyCoordinate } from "@/components/TopologySelector";

type DriftScenario = "stable" | "shift";
type Tone = "emerald" | "cyan" | "amber";

interface StatBlockProps {
  label: string;
  value: string;
  detail: string;
  tone: Tone;
}

interface DriftMetrics {
  psi: string;
  psiDetail: string;
  ksP: string;
  ksDetail: string;
  verdict: "NOMINAL" | "RETRAIN_TRIGGERED";
  tone: Tone;
}

const HISTORY_HOURS = 48;
const FORECAST_HOURS = 24;
const SHIFT_WINDOW_HOURS = 24;
const SHIFT_LEVEL = 25;
const ANCHOR_UTC_MS = Date.UTC(2026, 0, 26, 0, 0, 0);
const HOUR_MS = 3_600_000;

/** Stable values: measured on BOX01/SYSA/DB2_HIGH. Shift values: BOX01/SYSA/BATCH_LOW, injected regime shift. */
const DRIFT_METRICS: Record<DriftScenario, DriftMetrics> = {
  stable: {
    psi: "0.123",
    psiDetail: "0/11 drifted · threshold 0.2",
    ksP: "2.04e-01",
    ksDetail: "p ≥ 0.05 · no divergence",
    verdict: "NOMINAL",
    tone: "cyan",
  },
  shift: {
    psi: "6.46",
    psiDetail: "11/11 drifted · threshold 0.2",
    ksP: "1.06e-63",
    ksDetail: "p < 0.05 · divergence detected",
    verdict: "RETRAIN_TRIGGERED",
    tone: "amber",
  },
};

const TONE_CLASSES: Record<Tone, string> = {
  emerald: "border-emerald-800/60 bg-emerald-950/30 text-emerald-400",
  cyan: "border-cyan-800/60 bg-cyan-950/30 text-cyan-400",
  amber: "border-amber-800/60 bg-amber-950/30 text-amber-400",
};

function buildSeries(
  coordinate: TopologyCoordinate,
  scenario: DriftScenario,
): { points: ForecastPoint[]; forecastOrigin: string } {
  const seed = [...`${coordinate.box}${coordinate.system}${coordinate.serviceClass}`].reduce(
    (sum, char) => sum + char.charCodeAt(0),
    0,
  );
  const isBatch = coordinate.serviceClass === "BATCH_LOW";
  const base = isBatch ? 34 : 52;
  const amplitude = isBatch ? 22 : 18;
  const peakHour = isBatch ? 2 : 14;

  const points: ForecastPoint[] = Array.from(
    { length: HISTORY_HOURS + FORECAST_HOURS },
    (_, i) => {
      const date = new Date(ANCHOR_UTC_MS + (i - HISTORY_HOURS) * HOUR_MS);
      const phase = (2 * Math.PI * (date.getUTCHours() - peakHour)) / 24;
      const shift =
        scenario === "shift" && i >= HISTORY_HOURS - SHIFT_WINDOW_HOURS ? SHIFT_LEVEL : 0;
      const level = base + amplitude * Math.cos(phase) + Math.sin(seed + i * 0.7) * 3 + shift;
      const clamped = Math.min(100, Math.max(0, level));
      const isHistory = i < HISTORY_HOURS;

      return {
        timestamp: date.toISOString(),
        actual: isHistory ? Number(clamped.toFixed(2)) : null,
        predicted: isHistory
          ? null
          : Number(Math.min(100, Math.max(0, clamped + Math.sin(seed + i) * 1.2)).toFixed(2)),
      };
    },
  );

  return { points, forecastOrigin: points[HISTORY_HOURS].timestamp };
}

const StatBlock: FC<StatBlockProps> = ({ label, value, detail, tone }) => (
  <div className={`rounded-sm border p-3 ${TONE_CLASSES[tone]}`}>
    <div className="font-mono text-[10px] uppercase tracking-widest opacity-70">{label}</div>
    <div className="mt-1 font-mono text-xl">{value}</div>
    <div className="mt-1 font-mono text-[10px] opacity-60">{detail}</div>
  </div>
);

const ScenarioToggle: FC<{
  value: DriftScenario;
  onChange: (next: DriftScenario) => void;
}> = ({ value, onChange }) => {
  const options: ReadonlyArray<{ id: DriftScenario; label: string }> = [
    { id: "stable", label: "Stable window" },
    { id: "shift", label: "Injected regime shift" },
  ];
  return (
    <div className="inline-flex overflow-hidden rounded-sm border border-[#27272a] font-mono text-[11px]">
      {options.map((option) => {
        const active = option.id === value;
        return (
          <button
            key={option.id}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(option.id)}
            className={`px-3 py-1.5 transition-colors ${
              active
                ? option.id === "shift"
                  ? "bg-amber-950/60 text-amber-400"
                  : "bg-emerald-950/50 text-emerald-400"
                : "bg-[#09090b] text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
};

export default function DashboardPage() {
  const [coordinate, setCoordinate] = useState<TopologyCoordinate>({
    box: "BOX-01",
    system: "SYS-A",
    serviceClass: "ONLINE_HIGH",
  });
  const [scenario, setScenario] = useState<DriftScenario>("stable");

  const { points, forecastOrigin } = useMemo(
    () => buildSeries(coordinate, scenario),
    [coordinate, scenario],
  );
  const drift = DRIFT_METRICS[scenario];

  return (
    <main className="min-h-screen bg-[#09090b] px-6 py-8 text-zinc-100">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <header className="flex flex-col gap-1 border-b border-[#27272a] pb-4">
          <h1 className="font-mono text-lg font-semibold tracking-tight">
            Mainframe CPU Forecasting — Telemetry Console
          </h1>
          <p className="font-mono text-xs text-zinc-500">
            Box → System → Service Class forecasting with forward-chaining validation and dual-signal
            drift monitoring.{" "}
            <span className="text-zinc-600">
              Chart data is simulated; the deployed gateway serves one-step-ahead forecasts via POST
              /api/v1/forecast.
            </span>
          </p>
        </header>

        <TopologySelector value={coordinate} onChange={setCoordinate} />
        <ForecastChart data={points} forecastOriginTimestamp={forecastOrigin} />

        <section className="rounded-md border border-[#27272a] bg-[#0c0c0e] p-4">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <h2 className="font-mono text-xs uppercase tracking-widest text-zinc-400">
              Statistical Drift Monitor
            </h2>
            <ScenarioToggle value={scenario} onChange={setScenario} />
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatBlock label="Pooled MAE" value="1.56%" detail="1.564% · expanding-window OOF" tone="emerald" />
            <StatBlock label="Pooled RMSE" value="1.97%" detail="1.968% · expanding-window OOF" tone="emerald" />
            <StatBlock label="PSI" value={drift.psi} detail={drift.psiDetail} tone={drift.tone} />
            <StatBlock label="KS p-value" value={drift.ksP} detail={drift.ksDetail} tone={drift.tone} />
          </div>

          <div className="mt-4 flex items-center justify-between border-t border-[#27272a] pt-3">
            <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
              aggregate verdict
            </span>
            <span
              className={`rounded-sm border px-2 py-0.5 font-mono text-xs ${
                drift.verdict === "RETRAIN_TRIGGERED"
                  ? "border-amber-700 bg-amber-950/50 text-amber-400"
                  : "border-emerald-800 bg-emerald-950/40 text-emerald-400"
              }`}
            >
              {drift.verdict}
            </span>
          </div>
        </section>
      </div>
    </main>
  );
}