// Detail-panel signal chart: a door cycle waveform, or a time series for the other subsystems.
import LineChart from "./LineChart.jsx";
import { fmtDayHm, fmtNum, toMs } from "../lib/format.js";

// cycle: {t:[s], pos:[], pos_ref:[], current:[A], pwm:[]}
export function CycleWaveform({
  cycle,
  signal = "current",
  width = 438,
  height = 96,
  empty = "no telemetry stored for this cycle",
}) {
  const t = (cycle && cycle.t) || [];
  const y = (cycle && cycle[signal]) || [];
  const pts = t.map((tv, i) => [Number(tv), Number(y[i])]);
  const pos = (cycle && cycle.pos) || [];
  const posRef = (cycle && cycle.pos_ref) || [];
  const extra = [];
  if (pos.length === t.length && pos.length)
    extra.push({
      points: t.map((tv, i) => [Number(tv), Number(pos[i])]),
      color: "#4b5361",
      normalise: true,
      opacity: 0.85,
    });
  if (posRef.length === t.length && posRef.length)
    extra.push({
      points: t.map((tv, i) => [Number(tv), Number(posRef[i])]),
      color: "#1f2374",
      dash: "2 3",
      normalise: true,
      opacity: 0.5,
    });

  return (
    <LineChart
      points={pts}
      series={extra}
      width={width}
      height={height}
      color="#008f95"
      markLast="#006d73"
      xFormat={(v) => `${fmtNum(v, 1)} s`}
      yFormat={(v) => fmtNum(v, Math.abs(v) >= 10 ? 1 : 2)}
      empty={empty}
    />
  );
}

// series: {signal, unit, points:[[ts, value], ...]}
export function TimeSeriesChart({
  series,
  width = 438,
  height = 96,
  empty = "no telemetry in this window",
}) {
  const pts = ((series && series.points) || [])
    .map((p) => [toMs(p[0]), Number(p[1])])
    .filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  return (
    <LineChart
      points={pts}
      width={width}
      height={height}
      color="#008f95"
      markLast="#006d73"
      xFormat={(v) => fmtDayHm(v)}
      yFormat={(v) => fmtNum(v, Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2)}
      empty={empty}
    />
  );
}

export default TimeSeriesChart;
