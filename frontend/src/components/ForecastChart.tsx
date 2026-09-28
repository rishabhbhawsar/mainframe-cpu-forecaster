"use client";

import type { FC } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface ForecastPoint {
  /** ISO-8601 hourly timestamp (UTC). */
  timestamp: string;
  /** Observed CPU utilization (%); null after the forecast origin. */
  actual: number | null;
  /** Forecast CPU utilization (%); null before the forecast origin. */
  predicted: number | null;
}

interface ForecastChartProps {
  data: readonly ForecastPoint[];
  forecastOriginTimestamp: string;
}

interface TooltipEntry {
  dataKey?: string | number;
  value?: number | string | null;
}

interface ChronoTooltipProps {
  active?: boolean;
  payload?: readonly TooltipEntry[];
  label?: string | number;
}

const COLORS = {
  grid: "#27272a",
  axis: "#52525b",
  actual: "#10b981",
  predicted: "#22d3ee",
  origin: "#f59e0b",
} as const;

const AXIS_TICK = { fontFamily: "monospace", fontSize: 10, fill: COLORS.axis } as const;

function formatHour(iso: string): string {
  return `${new Date(iso).getUTCHours().toString().padStart(2, "0")}h`;
}

function formatStamp(iso: string): string {
  return `${iso.slice(0, 16).replace("T", " ")} UTC`;
}

const ChronoTooltip: FC<ChronoTooltipProps> = ({ active, payload, label }) => {
  if (!active || !payload || payload.length === 0) return null;

  const valueOf = (key: string): number | null => {
    const entry = payload.find((item) => item.dataKey === key);
    return entry?.value != null ? Number(entry.value) : null;
  };
  const actual = valueOf("actual");
  const predicted = valueOf("predicted");

  return (
    <div className="rounded-sm border border-[#27272a] bg-[#0c0c0e] px-3 py-2 font-mono text-xs shadow-lg shadow-black/40">
      <div className="mb-1 text-zinc-500">{typeof label === "string" ? formatStamp(label) : ""}</div>
      {actual !== null && <div className="text-emerald-400">observed {actual.toFixed(2)}%</div>}
      {predicted !== null && <div className="text-cyan-400">forecast {predicted.toFixed(2)}%</div>}
    </div>
  );
};

export const ForecastChart: FC<ForecastChartProps> = ({ data, forecastOriginTimestamp }) => (
  <section className="rounded-md border border-[#27272a] bg-[#0c0c0e] p-4">
    <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
      <h2 className="font-mono text-xs uppercase tracking-widest text-zinc-400">
        CPU Utilization — Chrono-Timeline
      </h2>
      <div className="flex items-center gap-4 font-mono text-[10px] text-zinc-500">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-3 bg-emerald-500" /> observed
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-3 bg-cyan-400" /> forecast (simulated)
        </span>
      </div>
    </div>

    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={[...data]} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
          <defs>
            <linearGradient id="observedFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={COLORS.actual} stopOpacity={0.25} />
              <stop offset="100%" stopColor={COLORS.actual} stopOpacity={0} />
            </linearGradient>
          </defs>

          <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="timestamp"
            tickFormatter={formatHour}
            interval={5}
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={{ stroke: COLORS.grid }}
          />
          <YAxis
            domain={[0, 100]}
            unit="%"
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={{ stroke: COLORS.grid }}
            width={44}
          />
          <Tooltip
            content={<ChronoTooltip />}
            cursor={{ stroke: COLORS.axis, strokeDasharray: "3 3" }}
          />
          <ReferenceLine
            x={forecastOriginTimestamp}
            stroke={COLORS.origin}
            strokeDasharray="4 4"
            label={{
              value: "forecast origin",
              position: "insideTopRight",
              fill: COLORS.origin,
              fontSize: 10,
              fontFamily: "monospace",
            }}
          />

          <Area
            type="monotone"
            dataKey="actual"
            stroke="none"
            fill="url(#observedFill)"
            isAnimationActive={false}
            connectNulls={false}
            activeDot={false}
          />
          <Line
            type="monotone"
            dataKey="actual"
            stroke={COLORS.actual}
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 3, fill: COLORS.actual, stroke: "#09090b" }}
            isAnimationActive={false}
            connectNulls={false}
          />
          <Line
            type="monotone"
            dataKey="predicted"
            stroke={COLORS.predicted}
            strokeWidth={2}
            strokeDasharray="5 3"
            dot={{ r: 4, fill: COLORS.predicted, stroke: "#09090b" }}
            activeDot={{ r: 6 }}
            isAnimationActive={false}
            connectNulls={false}
            />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  </section>
);

export default ForecastChart;