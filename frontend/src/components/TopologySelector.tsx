"use client";

import type { FC } from "react";

export interface TopologyCoordinate {
  box: string;
  system: string;
  serviceClass: string;
}

interface TopologySelectorProps {
  value: TopologyCoordinate;
  onChange: (next: TopologyCoordinate) => void;
  boxes?: readonly string[];
  systems?: readonly string[];
  serviceClasses?: readonly string[];
}

interface TopologyFieldProps {
  label: string;
  options: readonly string[];
  value: string;
  onChange: (value: string) => void;
}

const DEFAULT_BOXES = ["BOX-01", "BOX-02"] as const;
const DEFAULT_SYSTEMS = ["SYS-A", "SYS-B"] as const;
const DEFAULT_SERVICE_CLASSES = ["ONLINE_HIGH", "BATCH_LOW"] as const;

const LABEL_CLASS = "font-mono text-[10px] uppercase tracking-widest text-zinc-500";
const SELECT_CLASS =
  "w-full appearance-none rounded-sm border border-[#27272a] bg-[#09090b] px-3 py-2 " +
  "font-mono text-sm text-zinc-200 outline-none transition-colors " +
  "hover:border-zinc-600 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/40";

const TopologyField: FC<TopologyFieldProps> = ({ label, options, value, onChange }) => (
  <label className="flex flex-col gap-1.5">
    <span className={LABEL_CLASS}>{label}</span>
    <div className="relative">
      <select
        className={SELECT_CLASS}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
      <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs text-zinc-600">
        ▾
      </span>
    </div>
  </label>
);

export const TopologySelector: FC<TopologySelectorProps> = ({
  value,
  onChange,
  boxes = DEFAULT_BOXES,
  systems = DEFAULT_SYSTEMS,
  serviceClasses = DEFAULT_SERVICE_CLASSES,
}) => (
  <section className="rounded-md border border-[#27272a] bg-[#0c0c0e] p-4">
    <div className="mb-4 flex items-center justify-between">
      <h2 className="font-mono text-xs uppercase tracking-widest text-zinc-400">
        Topology Coordinate
      </h2>
      <span className="rounded-sm border border-emerald-800/60 bg-emerald-950/40 px-2 py-0.5 font-mono text-[10px] text-emerald-400">
        series resolved
      </span>
    </div>

    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      <TopologyField
        label="Box"
        options={boxes}
        value={value.box}
        onChange={(box) => onChange({ ...value, box })}
      />
      <TopologyField
        label="System"
        options={systems}
        value={value.system}
        onChange={(system) => onChange({ ...value, system })}
      />
      <TopologyField
        label="Service Class"
        options={serviceClasses}
        value={value.serviceClass}
        onChange={(serviceClass) => onChange({ ...value, serviceClass })}
      />
    </div>

    <div className="mt-4 border-t border-[#27272a] pt-3 font-mono text-xs text-zinc-500">
      key{" "}
      <span className="text-cyan-400">
        {value.box} / {value.system} / {value.serviceClass}
      </span>
    </div>
  </section>
);

export default TopologySelector;