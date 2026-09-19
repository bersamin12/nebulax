// The confirmation dialog after a file pick: what the chosen model expects, what the picked
// file(s) actually carry (columns read from the file head, or the server's ACV workbook report),
// whether the two agree, and a button that queues only the files that match.
import { useEffect } from "react";
import { C } from "../lib/format.js";
import { INPUT_SPEC } from "./fileCheck.js";
import { fmtBytes, TASK_META } from "./taskMeta.js";

const MAX_ROWS = 10;
const MAX_CHIPS = 40;

function Verdict({ ok }) {
  return (
    <span className="nx-tag" style={{ color: ok ? C.ok : C.crit, borderColor: ok ? C.ok : C.crit, flex: "none" }}>
      {ok ? "MATCHES" : "DOES NOT MATCH"}
    </span>
  );
}

/** One picked file in full: its columns as chips, the recognised fields, and any problem. */
function SingleFile({ task, item }) {
  const r = item.report;
  const requiredNames = new Set(Object.entries(r.found).filter(([k]) => (INPUT_SPEC[task]?.required || []).includes(k)).map(([, v]) => v));
  const recognised = new Set(Object.values(r.found));
  const chips = r.columns.slice(0, MAX_CHIPS);
  return (
    <div className="nx-confirm-file">
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        <span className="mono" style={{ fontSize: 11.5, color: C.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.name}</span>
        <span className="mono" style={{ fontSize: 9.5, color: C.dim2, flex: "none" }}>{fmtBytes(r.size)}</span>
        <span style={{ marginLeft: "auto" }} />
        <Verdict ok={r.ok} />
      </div>
      {r.detail && <div style={{ fontSize: 10.5, color: C.dim }}>{r.detail}</div>}
      {chips.length > 0 && (
        <div className="nx-confirm-chips">
          {chips.map((c, i) => (
            <span
              key={`${c}-${i}`}
              className={"nx-confirm-chip" + (requiredNames.has(c) ? " is-required" : recognised.has(c) ? " is-known" : "")}
              title={requiredNames.has(c) ? "required by the model" : recognised.has(c) ? "recognised field" : "not used by the model"}
            >
              {c}
            </span>
          ))}
          {r.columns.length > MAX_CHIPS && <span className="nx-confirm-chip">+{r.columns.length - MAX_CHIPS} more</span>}
        </div>
      )}
      {Object.keys(r.found).length > 0 && task !== "door" && (
        <div style={{ fontSize: 10.5, color: C.text2 }}>
          {Object.entries(r.found).map(([k, v]) => (
            <span key={k} style={{ marginRight: 12 }}>
              <span style={{ color: C.dim }}>{k}:</span> {String(v)}
            </span>
          ))}
        </div>
      )}
      {r.missing.length > 0 && (
        <div style={{ fontSize: 10.5, color: C.crit }}>missing: {r.missing.join(", ")}</div>
      )}
      {r.problems.map((p, i) => (
        <div key={i} style={{ fontSize: 10.5, color: C.crit }}>{p}</div>
      ))}
    </div>
  );
}

/** Many picked files: one line each with its verdict and first problem. */
function FileList({ items }) {
  const shown = items.slice(0, MAX_ROWS);
  return (
    <div className="nx-confirm-list">
      {shown.map((it) => (
        <div key={it.report.name} className="nx-confirm-row">
          <span className="mono" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: C.text }}>{it.report.name}</span>
          <span className="mono" style={{ color: C.dim2, flex: "none" }}>{fmtBytes(it.report.size)}</span>
          <span style={{ color: it.report.ok ? C.ok : C.crit, flex: "none", fontWeight: 600, letterSpacing: "0.06em", fontSize: 9.5 }}>
            {it.report.ok ? "MATCHES" : "NO MATCH"}
          </span>
          <span style={{ color: it.report.ok ? C.dim : C.crit, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {it.report.ok ? it.report.detail : it.report.problems[0] || ""}
          </span>
        </div>
      ))}
      {items.length > MAX_ROWS && (
        <div style={{ fontSize: 10, color: C.dim2, padding: "4px 0 0" }}>+{items.length - MAX_ROWS} more files, checked the same way</div>
      )}
    </div>
  );
}

export default function FileConfirm({ task, items, checking = false, onConfirm, onCancel }) {
  const meta = TASK_META[task] || {};
  const spec = INPUT_SPEC[task] || {};
  const okItems = items.filter((it) => it.report.ok);
  const nOk = okItems.length;

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onCancel?.();
      else if (e.key === "Enter" && !checking && nOk) onConfirm?.(okItems.map((it) => it.file));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel, onConfirm, okItems, nOk, checking]);

  return (
    <div className="nx-modal" role="dialog" aria-modal="true" aria-label="Confirm the picked files">
      <div className="nx-modal-card" style={{ width: 640 }}>
        <div className="nx-tour-eyebrow">CHECK THE FILES · {(meta.tab || task).toUpperCase()}</div>
        <div className="nx-tour-title">
          {checking ? "Reading the files" : items.length === 1 ? `Is this the right input for ${meta.title || task}?` : `Are these the right inputs for ${meta.title || task}?`}
        </div>

        <div className="nx-confirm-spec">
          <div className="nx-eyebrow">THIS MODEL EXPECTS</div>
          <div><span style={{ color: C.dim }}>file:</span> {spec.file}</div>
          <div><span style={{ color: C.dim }}>columns:</span> {spec.columns}</div>
        </div>

        <div className="nx-eyebrow">YOU PICKED {items.length} FILE{items.length === 1 ? "" : "S"}{checking ? " · checking" : ` · ${nOk} matching`}</div>
        <div className="nx-scroll nx-confirm-body">
          {checking ? (
            <div style={{ fontSize: 11, color: C.dim2, padding: "12px 0" }}>reading the header of each file{task === "acv" ? " and checking the workbook fields on the server" : ""}…</div>
          ) : items.length === 1 ? (
            <SingleFile task={task} item={items[0]} />
          ) : (
            <FileList items={items} />
          )}
        </div>

        <div className="nx-tour-actions" style={{ marginTop: 6 }}>
          <button type="button" className="nx-btn nx-btn--primary" disabled={checking || !nOk} onClick={() => onConfirm?.(okItems.map((it) => it.file))}>
            {!nOk ? "NOTHING TO QUEUE" : nOk < items.length ? `QUEUE THE ${nOk} THAT MATCH` : `QUEUE ${nOk} FILE${nOk === 1 ? "" : "S"}`}
          </button>
          <button type="button" className="nx-btn" onClick={onCancel}>CANCEL</button>
          <span style={{ marginLeft: "auto", fontSize: 10, color: C.dim2 }}>
            {nOk < items.length && !checking ? "files that do not match are left out" : "the server re-checks every file on upload"}
          </span>
        </div>
      </div>
    </div>
  );
}
