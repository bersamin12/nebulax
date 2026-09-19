// Three subsystem lanes (Doors / Pneumatics / Bearings), one cell per car.
// A cell is coloured by the worst health among that car's components of that subsystem.
import {
  C,
  countHealth,
  displayCar,
  fmtScore,
  fmtThreshold,
  health,
  shortComponent,
  worstComponent,
} from "../lib/format.js";

const CARS = [1, 2, 3, 4, 5, 6];

function DoorIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={C.accent} strokeWidth="1.7">
      <rect x="3" y="3" width="8" height="18" />
      <rect x="13" y="3" width="8" height="18" />
      <path d="M8 12h.01" />
      <path d="M16 12h.01" />
    </svg>
  );
}
function PneuIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={C.accent} strokeWidth="1.7">
      <circle cx="12" cy="13" r="7" />
      <path d="M12 13V8" />
      <path d="M9 3h6" />
      <path d="M12 3v2" />
    </svg>
  );
}
function BearIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={C.accent} strokeWidth="1.7">
      <circle cx="12" cy="12" r="8" />
      <circle cx="12" cy="12" r="3" />
      <path d="M12 4v2" />
      <path d="M12 18v2" />
      <path d="M4 12h2" />
      <path d="M18 12h2" />
    </svg>
  );
}

function OkGlyph() {
  return (
    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
      <path d="M4 12l6 6L20 5" />
    </svg>
  );
}
function WarnGlyph() {
  return (
    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6">
      <path d="M12 3L1.5 21h21z" />
      <path d="M12 9.5v5" />
      <path d="M12 18v.5" />
    </svg>
  );
}
function CritGlyph() {
  return (
    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6">
      <path d="M8 1.5h8L22.5 8v8L16 22.5H8L1.5 16V8z" />
      <path d="M12 7v6" />
      <path d="M12 16.5v.5" />
    </svg>
  );
}
function NoDataGlyph() {
  return (
    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4">
      <path d="M4 12h16" />
    </svg>
  );
}
const GLYPH = { ok: OkGlyph, warn: WarnGlyph, crit: CritGlyph, nodata: NoDataGlyph };

const LANES = [
  {
    subsystem: "door",
    title: "DOORS",
    note: "worst leaf per car · scored per open/close cycle",
    signals: "motor current · closing time · position error · retries",
    Icon: DoorIcon,
    label: (car) => `CAR ${car}`,
  },
  {
    subsystem: "pneumatic",
    title: "PNEUMATICS · BRAKE AIR SUPPLY",
    note: "air production unit · unit level, drawn under car 3",
    signals: "TP2 · TP3 · MR pressure · oil temp · duty cycle",
    Icon: PneuIcon,
    label: (car) => `APU ${car}`,
  },
  {
    subsystem: "bearing",
    title: "BOGIE AXLE BEARINGS",
    note: "worst of 8 axle boxes per car · 5-min windows",
    signals: "axle-box temp · Δ vs peers · vib RMS · kurtosis",
    Icon: BearIcon,
    label: (car) => `CAR ${car}`,
  },
];

function Cell({ lane, car, comps, selected, onSelect, trainId }) {
  const worst = worstComponent(comps);
  const n = countHealth(comps);
  const key = worst ? worst.health || "ok" : "nodata";
  const h = health(key);
  const Glyph = GLYPH[key] || NoDataGlyph;
  const isSel = Boolean(
    selected &&
      selected.subsystem === lane.subsystem &&
      comps.some((c) => c.component_id === selected.component_id && c.car === selected.car)
  );

  const click = () => {
    if (!worst || !onSelect) return;
    onSelect({
      train_id: trainId,
      car: worst.car,
      subsystem: worst.subsystem,
      component_id: worst.component_id,
    });
  };

  return (
    <button
      type="button"
      className={
        "nx-cell" + (worst ? "" : " nx-cell--nodata") + (isSel ? " nx-cell--sel" : "")
      }
      style={{
        background: h.bg,
        borderColor: h.border,
        borderTopColor: h.color,
        opacity: worst ? 1 : 0.72,
      }}
      onClick={click}
      disabled={!worst}
      title={
        worst
          ? `${lane.label(car)} · worst ${shortComponent(worst.component_id)} · ${
              worst.model || ""
            }`
          : `${lane.label(car)} · no telemetry on this ${lane.subsystem}`
      }
    >
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
        <span className="mono" style={{ fontSize: 9.5, color: worst ? h.ink : C.dim2 }}>
          {lane.label(car)}
        </span>
        <span
          className="mono"
          style={{ fontSize: 21, fontWeight: 600, color: worst ? h.color : C.line2 }}
        >
          {worst ? fmtScore(worst.score) : "—"}
        </span>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 4,
          fontSize: 9,
          letterSpacing: "0.1em",
          color: h.color,
          fontWeight: 600,
        }}
      >
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <Glyph />
          {h.word}
          {worst && worst.stale ? (
            <span style={{ color: C.dim2, fontWeight: 400 }}>&middot; STALE</span>
          ) : null}
        </span>
        <span
          className="mono"
          title="alert / watch / scored components in this cell"
          style={{ fontSize: 8.5, color: C.dim2, fontWeight: 400 }}
        >
          {worst ? `${n.crit} / ${n.warn} / ${comps.length}` : ""}
        </span>
      </div>

      <div
        className="mono"
        style={{
          marginTop: "auto",
          fontSize: 9.5,
          color: worst ? h.ink : C.dim2,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {worst
          ? `${shortComponent(worst.component_id)} · ${fmtScore(worst.score)} / ${fmtThreshold(
              worst.threshold
            )}`
          : "not instrumented"}
      </div>
    </button>
  );
}

export default function SubsystemLanes({ train, trainId, selected, onSelect }) {
  const comps = (train && train.components) || [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, minWidth: 0 }}>
      {LANES.map((lane) => {
        const mine = comps.filter((c) => c.subsystem === lane.subsystem);
        const model = (worstComponent(mine) || mine[0] || {}).model;
        const byCar = new Map(CARS.map((c) => [c, []]));
        for (const c of mine) {
          const dc = displayCar(c);
          if (byCar.has(dc)) byCar.get(dc).push(c);
        }
        return (
          <div className="nx-lane" key={lane.subsystem}>
            <div
              style={{
                height: 16,
                flex: "none",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 10,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 9, minWidth: 0 }}>
                <lane.Icon />
                <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.13em" }}>
                  {lane.title}
                </span>
                <span
                  style={{
                    fontSize: 9.5,
                    color: C.dim,
                    letterSpacing: "0.06em",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {lane.note}
                </span>
              </div>
              <div
                className="mono"
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  fontSize: 9.5,
                  color: C.dim,
                  flex: "none",
                }}
              >
                <span className="nx-tag" style={{ color: model ? C.violet : C.dim2 }}>
                  {model || "no model"}
                </span>
                <span>{lane.signals}</span>
              </div>
            </div>

            <div className="nx-lane-grid">
              {CARS.map((car) => (
                <Cell
                  key={car}
                  lane={lane}
                  car={car}
                  comps={byCar.get(car) || []}
                  selected={selected}
                  onSelect={onSelect}
                  trainId={trainId}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
