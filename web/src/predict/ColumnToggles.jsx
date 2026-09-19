// Simulated missing columns (Door / ACV). The server lists the optional fields a task can run
// without (`droppable_fields` on GET /api/ps3/tasks); switching one off here removes that column
// from every file of the next RUN before it is loaded, so the model's own fallback for an
// absent field is what gets exercised. Rail and SHM have nothing optional and render nothing.
import { useState } from "react";
import { C } from "../lib/format.js";
import { FIELD_LABELS } from "./taskMeta.js";

/**
 * @param {object} p
 * @param {string[]} p.fields       droppable canonical fields (from the task entry)
 * @param {string[]} p.dropped      the ones currently switched off
 * @param {(field:string)=>void} p.onToggle
 * @param {object} [p.found]        canonical -> raw column name from the last file check; a
 *                                  droppable field the file does not carry is shown as absent
 * @param {boolean} [p.compact]     the left-column strip: one summary line, chips fold out on click
 */
export default function ColumnToggles({ fields, dropped, onToggle, found = null, compact = false, disabled = false }) {
  const [open, setOpen] = useState(!compact);
  if (!fields || !fields.length) return null;
  const off = new Set(dropped || []);
  const summary = off.size ? `${off.size} OFF · ${[...off].map((f) => FIELD_LABELS[f] || f).join(", ")}` : "ALL ON";
  return (
    <div className={"nx-coltoggles" + (compact ? " is-compact" : "")} data-tour="columns">
      {compact ? (
        <button type="button" className="nx-coltoggles-head" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
          <span className="nx-eyebrow">SIMULATE MISSING COLUMNS</span>
          <span className="mono nx-coltoggles-sum" style={{ color: off.size ? C.warn : C.dim2 }}>{summary}</span>
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ transform: open ? "rotate(180deg)" : "none", flex: "none" }}><path d="M6 9l6 6 6-6" /></svg>
        </button>
      ) : (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6 }}>
          <span className="nx-eyebrow">SIMULATE MISSING COLUMNS</span>
          <span className="mono" style={{ fontSize: 8.5, color: off.size ? C.warn : C.dim2 }}>
            {off.size ? `${off.size} OFF` : "ALL ON"}
          </span>
        </div>
      )}
      {!open ? null : !compact && (
        <div style={{ fontSize: 10, color: C.dim, lineHeight: 1.4 }}>
          Switch an optional field off to run the model as if the file never carried it. The required columns cannot be switched off.
        </div>
      )}
      {open && (
      <div className="nx-coltoggles-chips">
        {fields.map((f) => {
          const absent = found && !(f in found);
          const isOff = off.has(f);
          return (
            <button
              key={f}
              type="button"
              className={"nx-coltoggle" + (isOff ? " is-off" : "") + (absent ? " is-absent" : "")}
              aria-pressed={!isOff}
              disabled={disabled || absent}
              title={absent ? `${FIELD_LABELS[f] || f}: not in the file` : isOff ? `${FIELD_LABELS[f] || f}: switched off (removed before the model runs)` : `${FIELD_LABELS[f] || f}: on`}
              onClick={() => onToggle?.(f)}
            >
              <span className="nx-coltoggle-dot" aria-hidden="true" />
              {FIELD_LABELS[f] || f}
            </button>
          );
        })}
      </div>
      )}
    </div>
  );
}
