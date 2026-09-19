// Centre column: the rows that will be written to the organiser CSV, in the contract's column
// order, one row per predicted item. Clicking a row selects it (the 3D twin and the explanation
// panel follow): every row ends in a chevron and the header says so, since a
// plain table does not look clickable. Nothing is reformatted - a cell is the CSV cell.
import { C, health as healthOf } from "../lib/format.js";
import { COLUMN_LABELS, rowHealth } from "./taskMeta.js";

/** The grey "what to do here" line under the table, per system and state. */
function guidance(task, n, running, selectedCar) {
  if (running && !n) return "Processing files. Results will appear as each file finishes.";
  if (!n) return "Choose a system, add its Test files, and select Run. Results will appear here; select a row to inspect its details.";
  if (task === "acv") {
    return selectedCar
      ? `Car ${selectedCar} is selected. The train view shows its roof unit, and the details panel shows its rank. Select the row or SHOW ALL CARS to return to the full ranking.`
      : "Each row ranks the cars in one workbook by leak likelihood. Select a car to view its position and rank; the red car is ranked first.";
  }
  if (task === "door") return "Each row is one door cycle. Select a result to highlight the door and review the current trace.";
  if (task === "rail") return "Each row is one run. Select a result to highlight the predicted rail side and review its wavelength spectrum.";
  if (task === "shm") return "Each row is one stress record. Select a result to review the predicted damage and rainflow cycles.";
  return "Select a result to update the train view and details.";
}

/**
 * ACV's `ranked_cars` is the whole ranking, `|`-joined, most likely first: the rank-1 car is the
 * call being scored (rank decay). Every car is clickable for inspection; rank 1 is emphasized.
 */
function Ranked({ value, selectedCar, onSelectCar }) {
  const ids = String(value ?? "").split("|");
  return (
    <span className="nx-ranked-cars">
      {ids.map((id, i) => (
        <button
          key={`${id}-${i}`}
          type="button"
          className={`nx-car-rank${selectedCar === id ? " is-selected" : ""}`}
          style={{ color: i === 0 ? C.crit : C.text2 }}
          title={`Car ${id} · rank ${i + 1} of ${ids.length}`}
          aria-label={`Focus car ${id}, rank ${i + 1} of ${ids.length}`}
          aria-pressed={selectedCar === id}
          onClick={(e) => { e.stopPropagation(); onSelectCar?.(id); }}
        >
          <small>{i + 1}</small> {id}
        </button>
      ))}
    </span>
  );
}

export default function ResultsTable({ task, rows = [], columns = [], selected = -1, selectedCar = null, onSelect, onSelectCar, onShowAll, height = 300, running = false, progress = null }) {
  const n = rows.length;
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
        <span style={{ fontSize: 9.5, letterSpacing: "0.16em", color: C.dim }}>PREDICTIONS</span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      {n > 1 && <span className="nx-row-hint">SELECT A ROW FOR DETAILS</span>}
          {task === "acv" && n > 0 && <button type="button" className="nx-inline-link" onClick={onShowAll}>SHOW ALL CARS</button>}
          <span className="mono" style={{ fontSize: 10, color: C.dim2 }}>
            {n} row{n === 1 ? "" : "s"}{progress ? ` · ${progress}` : ""}
          </span>
        </div>
      </div>

      {!n ? (
        <div style={{ flex: 1, display: "grid", placeItems: "center", color: C.dim2, fontSize: 11, textAlign: "center", padding: 16 }}>
          {running ? "Processing files…" : "No predictions yet. Add files on the left, then select Run."}
        </div>
      ) : (
        <div className="nx-scroll" style={{ flex: 1, minHeight: 0 }}>
          <table className="nx-predict-table" style={{ width: "100%", borderCollapse: "collapse", tableLayout: "fixed" }}>
            <thead>
              <tr>
                {columns.map((c, i) => (
                  <th
                    key={c}
                    style={{
                      position: "sticky",
                      top: 0,
                      zIndex: 1,
                      textAlign: "left",
                      fontSize: 9,
                      letterSpacing: "0.12em",
                      textTransform: "uppercase",
                      color: C.dim2,
                      fontWeight: 600,
                      background: C.panel,
                      borderBottom: `1px solid ${C.line}`,
                      padding: "5px 8px",
                      width: i === columns.length - 1 ? undefined : c === "file_id" ? 150 : undefined,
                    }}
                  >
                    {COLUMN_LABELS[c] || c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => {
                const h = healthOf(rowHealth(task, r));
                const sel = i === selected;
                return (
                  <tr
                    key={i}
                    className={sel ? "is-selected" : undefined}
                    onClick={() => onSelect && onSelect(i)}
                    title={sel ? undefined : "Select this row to update the train view and details."}
                    style={{
                      cursor: "pointer",
                      background: sel ? C.accentBg : "transparent",
                      outline: sel ? `1px solid ${C.accentBright}` : "none",
                      outlineOffset: -1,
                    }}
                  >
                    {columns.map((c, j) => (
                      <td
                        key={c}
                        className="mono"
                        style={{
                          fontSize: 10.5,
                          padding: "4px 8px",
                          borderBottom: `1px solid ${C.grid}`,
                          color: c === "prediction" ? h.color : C.text2,
                          fontWeight: c === "prediction" ? 600 : 400,
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          borderLeft: j === 0 ? `3px solid ${sel ? C.accent : h.color}` : "none",
                        }}
                        title={String(r[c] ?? "")}
                      >
                        {c === "ranked_cars" ? <Ranked value={r[c]} selectedCar={sel ? selectedCar : null} onSelectCar={(id) => onSelectCar?.(id, i)} /> : String(r[c] ?? "")}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="nx-guidance">{guidance(task, n, running, selectedCar)}</div>
    </div>
  );
}
