// Right column of the predict page: why *this* prediction, for the selected results row.
//
// It renders exactly what `explanation.as_dict()` carries (docs/ps3_contract.md section 2) and
// invents nothing: the two or three `numbers`, the `trace` through the console's LineChart, its
// `marks`, and the `viewport` target the 3D twin is pointed at. Each variable carries an (i)
// icon whose hover / focus tooltip defines it (numberDefs.js), for engineers new to the task.
import { C, MONO, fmtNum, fmtScore, health as healthOf } from "../lib/format.js";
import LineChart from "../charts/LineChart.jsx";
import { defineNumber } from "./numberDefs.js";
import { COLUMN_LABELS, FIELD_LABELS, TASK_META } from "./taskMeta.js";

const CHART_W = 356;

/** Pretty-print a snake_case key: `side_i_score` -> `side I score`. */
function keyLabel(k) {
  return String(k)
    .split("_")
    .map((w) => (/^(i|ii|iii|iv|v)$/i.test(w) ? w.toUpperCase() : w))
    .join(" ");
}

function numberCell(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "n/a";
  if (Math.abs(n) >= 1e5 || (n !== 0 && Math.abs(n) < 1e-3)) return fmtScore(n);
  return fmtNum(n, Math.abs(n) >= 100 ? 1 : 3);
}

/**
 * The trace as plottable points. `trace.x` is the task's own axis and may be strings (door sends
 * native timestamps, ACV ISO ones): those are drawn against the sample index and the axis is
 * labelled with the string itself, so nothing is silently reinterpreted as a number.
 */
function traceModel(trace) {
  const xs = (trace && trace.x) || [];
  const ys = (trace && trace.y) || [];
  const n = Math.min(xs.length, ys.length);
  if (!n) return null;
  const numeric = xs.every((v) => Number.isFinite(Number(v)) && v !== "" && v !== null);
  const points = [];
  const index = new Map();
  for (let i = 0; i < n; i += 1) {
    const x = numeric ? Number(xs[i]) : i;
    index.set(String(xs[i]), x);
    points.push([x, Number(ys[i])]);
  }
  const at = (v) => {
    if (v === null || v === undefined) return null;
    if (index.has(String(v))) return index.get(String(v));
    const num = Number(v);
    return numeric && Number.isFinite(num) ? num : null;
  };
  const marks = ((trace && trace.marks) || []).map((m) => ({
    ...m,
    _x: at(m.x),
    _x1: m.x1 !== undefined ? at(m.x1) : null,
  }));
  const label = (trace && trace.label) || "";
  const xFormat = numeric
    ? (v) => fmtScore(v)
    : (v) => {
        const i = Math.round(v);
        const raw = xs[Math.max(0, Math.min(xs.length - 1, i))];
        return String(raw ?? i);
      };
  return { points, marks, label, numeric, xFormat, xs };
}

/** A variable name with its (i) definition, shown on hover or keyboard focus. */
function InfoLabel({ label, name, def }) {
  if (!def) return <span style={{ color: C.dim, fontSize: 10.5, letterSpacing: "0.04em" }}>{label}</span>;
  return (
    <span className="nx-info">
      <span style={{ color: C.dim, fontSize: 10.5, letterSpacing: "0.04em" }}>{label}</span>
      <button type="button" className="nx-info-btn" aria-label={`What is ${label}?`}>i</button>
      <span className="nx-info-tip" role="tooltip">
        <strong>{name}</strong>
        {def}
      </span>
    </span>
  );
}

function Row({ k, v, name, def }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 10, padding: "3px 0", borderBottom: `1px solid ${C.grid}` }}>
      <InfoLabel label={k} name={name || k} def={def} />
      <span className="mono" style={{ fontSize: 11.5, color: C.text }}>{v}</span>
    </div>
  );
}

export default function ExplanationPanel({ task, row, explanation, selectedCar = null, columns = [], height = 540 }) {
  const meta = TASK_META[task] || {};
  const model = traceModel(explanation && explanation.trace);
  const vp = (explanation && explanation.viewport) || null;
  const numbers = Object.entries((explanation && explanation.numbers) || {});
  const carRank = task === "acv" && selectedCar && row
    ? String(row.ranked_cars || "").split("|").findIndex((id) => +id === +selectedCar) + 1
    : 0;
  const h = healthOf(carRank ? (carRank === 1 ? "crit" : carRank <= 3 ? "warn" : "ok") : vp && vp.health);
  const carDelta = selectedCar && explanation?.numbers
    ? explanation.numbers[`car_${selectedCar}_peer_delta_hot_k`]
    : null;
  const bands = model
    ? model.marks
        .filter((m) => Number.isFinite(m._x))
        .map((m) => [m._x, Number.isFinite(m._x1) ? m._x1 : m._x])
    : [];

  return (
    <div
      style={{
        height,
        background: C.panel,
        border: `1px solid ${C.line}`,
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
      }}
    >
      <div
        style={{
          height: 26,
          flex: "none",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
          padding: "0 10px",
          borderBottom: `1px solid ${C.line}`,
          background: C.panel2,
        }}
      >
        <span style={{ fontSize: 9.5, letterSpacing: "0.16em", color: C.dim }}>EXPLANATION</span>
        {vp && (
          <span className="nx-tag" style={{ color: h.color, borderColor: h.border }}>
            {task === "acv" ? `RANK ${carRank || 1}` : h.word}
          </span>
        )}
      </div>

      {!row ? (
        <div style={{ flex: 1, display: "grid", placeItems: "center", color: C.dim2, fontSize: 11 }}>
          Run a prediction, then select a result.
        </div>
      ) : (
        <div className="nx-scroll" style={{ flex: 1, minHeight: 0, padding: "9px 10px 12px 10px" }}>
          <div style={{ fontSize: 12.5, fontWeight: 600, color: C.text, wordBreak: "break-all" }}>
            {row.file_id ?? String(row.start_time ?? "segment")}
          </div>
          <div style={{ fontSize: 10, color: C.dim, marginTop: 2, marginBottom: 8 }}>
            {meta.title || task}
          </div>

          {Array.isArray(explanation?.dropped_fields) && explanation.dropped_fields.length > 0 && (
            <div className="nx-dropped-note">
              <strong>SIMULATED MISSING COLUMNS</strong>
              <span>{explanation.dropped_fields.map((f) => FIELD_LABELS[f] || f).join(", ")} removed before the model ran; the numbers below are what it produced without them.</span>
              {Array.isArray(explanation.dropped_columns) && explanation.dropped_columns.length === 0 && (
                <span style={{ color: C.dim }}>(the file did not carry those columns anyway)</span>
              )}
            </div>
          )}

          {carRank > 0 && (
            <div className="nx-car-detail">
              <strong>CAR {selectedCar} · RANK {carRank}</strong>
              <span>of {String(row.ranked_cars || "").split("|").length} cars · {carRank === 1 ? "highest leak likelihood" : "ranking context"}</span>
              {carDelta !== null && carDelta !== undefined && Number.isFinite(Number(carDelta)) && <span>peer hot-temperature difference: {numberCell(carDelta)} K</span>}
              {carRank > 1 && <span>Numbers below include all cars; fields marked “top car” describe rank 1.</span>}
            </div>
          )}

          {/* the prediction itself, in the organiser's own column names */}
          <div style={{ marginBottom: 10 }}>
            {columns.map((c) => (
              <Row key={c} k={COLUMN_LABELS[c] || c} name={c} def={defineNumber(task, c)} v={String(row[c] ?? "n/a")} />
            ))}
          </div>

          {numbers.length > 0 && (
            <>
              <div style={{ fontSize: 9.5, letterSpacing: "0.16em", color: C.dim, margin: "2px 0 4px" }}>
                NUMBERS
              </div>
              <div style={{ marginBottom: 10 }}>
                {numbers.map(([k, v]) => (
                  <Row key={k} k={keyLabel(k)} name={k} def={defineNumber(task, k)} v={numberCell(v)} />
                ))}
              </div>
            </>
          )}

          {model && (
            <>
              <div style={{ fontSize: 9.5, letterSpacing: "0.16em", color: C.dim, margin: "2px 0 4px" }}>
                {(model.label || "TRACE").toUpperCase()}
              </div>
              <LineChart
                points={model.points}
                width={CHART_W}
                height={104}
                color={C.accent}
                bands={bands}
                xFormat={model.xFormat}
                yFormat={(v) => fmtScore(v)}
                empty="No trace available"
              />
              <div style={{ display: "flex", justifyContent: "space-between", fontFamily: MONO, fontSize: 8.5, color: C.dim2, marginTop: 2 }}>
                <span>{model.xFormat(model.points[0][0])}</span>
                <span>{model.xFormat(model.points[model.points.length - 1][0])}</span>
              </div>
              {model.marks.length > 0 && (
                <div style={{ marginTop: 6 }}>
                  {model.marks.slice(0, 6).map((m, i) => (
                    <div key={i} style={{ display: "flex", gap: 6, alignItems: "baseline", padding: "1.5px 0" }}>
                      <span style={{ width: 5, height: 5, flex: "none", background: C.crit, display: "block" }} />
                      <span style={{ fontSize: 10.5, color: C.text2 }}>{m.label || m.kind || "mark"}</span>
                      <span className="mono" style={{ fontSize: 9.5, color: C.dim2, marginLeft: "auto" }}>
                        {model.numeric ? fmtScore(Number(m.x)) : String(m.x)}
                      </span>
                    </div>
                  ))}
                  {model.marks.length > 6 && (
                    <div style={{ fontSize: 9.5, color: C.dim2, paddingTop: 2 }}>
                      +{model.marks.length - 6} more marks
                    </div>
                  )}
                </div>
              )}
            </>
          )}

          {vp && (
            <>
              <div style={{ fontSize: 9.5, letterSpacing: "0.16em", color: C.dim, margin: "10px 0 4px" }}>
                VIEWPORT TARGET
              </div>
              <Row k="car" def={defineNumber(task, "car")} v={selectedCar || (vp.car === null || vp.car === undefined ? "n/a" : String(vp.car))} />
              <Row k="side" def={defineNumber(task, "side")} v={vp.side || "n/a"} />
              <Row k="component" def={defineNumber(task, "component")} v={carRank ? `car${Number(selectedCar)}_ac1` : vp.component || "n/a"} />
            </>
          )}

          {!model && !numbers.length && (
            <div style={{ color: C.dim2, fontSize: 11, paddingTop: 6 }}>
              No explanation is available for this result.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
