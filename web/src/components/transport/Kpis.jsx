// Fleet KPI strip: frame.kpis, in mono numerals with a label above each figure.
// Contract section 4 (/api/kpis) and section 5 (frame.kpis).

import { fmtScore } from "../../lib/format.js";

const DASH = "–";

// A KPI computed at a MetroPT-3 timestamp divides by an almost-empty count of scored sim days
// and comes back as 1.5e9 false alarms per train-day; printed in full it blows the strip apart.
function fmt(value, digits) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return DASH;
  const v = Number(value);
  return Math.abs(v) >= 1e5 ? fmtScore(v) : v.toFixed(digits);
}

function Cell({ label, value, unit, tone }) {
  return (
    <div className="tb-kpi">
      <span className="tb-label">{label}</span>
      <span className={"tb-kpi-value" + (tone ? ` is-${tone}` : "")}>
        {value}
        {unit ? <span className="tb-kpi-unit"> {unit}</span> : null}
      </span>
    </div>
  );
}

export default function Kpis({ kpis, nTrains = 0, faBudget = 0.1, systemStats }) {
  if (systemStats) return (
    <div className="tb-kpis">
      <Cell label="Active systems" value={String(systemStats.active)} unit="/ 6" />
      <Cell label="Alert components" value={String(systemStats.alerts)} tone={systemStats.alerts ? "crit" : null} />
      <Cell label="Predicted examples" value={String(systemStats.examples)} unit="LTA PS3" />
      <Cell label="Replay set" value={systemStats.trainId || DASH} />
    </div>
  );
  const k = kpis || {};
  const open = Number.isFinite(Number(k.open_alerts)) ? Number(k.open_alerts) : null;
  const fa = Number.isFinite(Number(k.fa_per_train_day)) ? Number(k.fa_per_train_day) : null;
  const det = Number.isFinite(Number(k.events_detected)) ? Number(k.events_detected) : null;
  const tot = Number.isFinite(Number(k.events_total)) ? Number(k.events_total) : null;

  return (
    <div className="tb-kpis">
      <Cell
        label="Open alerts"
        value={open === null ? DASH : String(open)}
        unit={nTrains ? `/ ${nTrains} train${nTrains === 1 ? "" : "s"}` : ""}
        tone={open ? "crit" : null}
      />
      <Cell label="Median lead time" value={fmt(k.median_lead_h, 1)} unit="h to failure" />
      <Cell
        label="False alarms"
        value={fmt(k.fa_per_train_day, 2)}
        unit="/ train-day"
        tone={fa === null ? null : fa > faBudget ? "warn" : "ok"}
      />
      <Cell
        label="Events detected"
        value={det === null ? DASH : String(det)}
        unit={tot === null ? "" : `/ ${tot}`}
        tone={det !== null && tot ? (det >= tot ? "ok" : null) : null}
      />
    </div>
  );
}
