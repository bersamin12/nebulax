// 56 px header: brand, the page nav, and - on the fleet twin - the line/train selector, the
// model in use per subsystem and the replay clock.
//
// Three pages share this bar (App.jsx): `overview` (the landing page), `predict` (the PS3
// upload-and-predict page, which the nav calls the Digital Twin) and `twin` (the replay
// console). The twin's controls are meaningless on the other two pages - there is no train set,
// no replay clock and no per-subsystem model chip there - so they are replaced by the page nav,
// the tutorial button and one PS3 chip; everything else is untouched.
import { useEffect, useState } from "react";
import { getSelection } from "../api.js";
import { C, fmtStamp } from "../lib/format.js";

const SUBSYSTEMS = ["pneumatic", "bearing"];

const SUB_LABEL = { door: "DOOR", pneumatic: "PNEU", bearing: "BEAR" };

// Header space is fixed: show a short display name, keep the registry name in the tooltip.
const SHORT_MODEL = {
  cusum_cycle_scalar: "CUSUM cycle",
  sparse_autoencoder: "Sparse AE",
  lgbm_residual: "LGBM resid",
  minirocket: "MiniRocket",
};
const shortModel = (m) => SHORT_MODEL[m] || (m.length > 14 ? m.slice(0, 13) + "…" : m);

function rowsOf(sel) {
  if (!sel) return [];
  if (Array.isArray(sel)) return sel;
  for (const k of ["rows", "selection", "selected", "leaderboard", "winners"]) {
    if (Array.isArray(sel[k])) return sel[k];
  }
  return [];
}

// The chips say which model is *in use* on the train on screen, so the replay frame wins: it is
// the model that actually produced the scores being drawn. The leaderboard row is only a
// fallback for a subsystem this frame has not scored yet, and is never read across datasets -
// MetroPT-3's own selection row reads "deferred: no validation-slice metric" (the pick is
// hand-made, contract section 2), and the Cranfield / Ottawa rows are a classification story
// that nothing in the app scores.
function modelsFor(sel, train, dataset) {
  const rows = rowsOf(sel);
  const out = {};
  for (const s of SUBSYSTEMS) {
    const comp = ((train && train.components) || []).find((c) => c.subsystem === s && c.model);
    if (comp) {
      out[s] = { model: comp.model, source: "frame" };
      continue;
    }
    const hit = rows.find(
      (r) => r.subsystem === s && r.dataset === dataset && r.model && !/^deferred\b/.test(r.model)
    );
    if (hit) out[s] = { model: hit.model, source: "selection" };
  }
  return out;
}

function PageNav({ page, onPage }) {
  if (page === "twin") {
    return <button type="button" className="nx-btn" style={{ flex: "none", whiteSpace: "nowrap" }} onClick={() => onPage?.("predict")}>← DIGITAL TWIN</button>;
  }
  const link = (id, label) => (
    <button
      key={id}
      type="button"
      className={"nx-nav" + (page === id ? " is-active" : "")}
      aria-current={page === id ? "page" : undefined}
      onClick={() => page !== id && onPage?.(id)}
    >
      {label}
    </button>
  );
  return (
    <nav style={{ display: "flex", gap: 4, flex: "none" }} aria-label="Pages">
      {link("overview", "OVERVIEW")}
      {link("predict", "DIGITAL TWIN")}
    </nav>
  );
}

/** Predict-page tutorial: restarts the guided tour (predict/Tutorial.jsx). */
function TutorialButton({ onPage }) {
  return (
    <button type="button" className="nx-btn" style={{ height: 26, flex: "none", whiteSpace: "nowrap" }} onClick={() => onPage?.("predict", { tour: true })} title="Guided tour of the Digital Twin page">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10" /><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3" /><line x1="12" y1="17" x2="12" y2="17" /></svg>
      TUTORIAL
    </button>
  );
}

export default function Header({ frame, train, trainId, trainIds, onSelectTrain, connected, page = "twin", onPage }) {
  const [sel, setSel] = useState(null);

  useEffect(() => {
    if (page !== "twin") return undefined; // the chips are twin-only; do not poke /api on predict
    let live = true;
    getSelection()
      .then((d) => live && setSel(d))
      .catch(() => live && setSel(null));
    return () => {
      live = false;
    };
  }, [page]);

  const dataset = trainId === "MP3" ? "metropt3" : "sim";
  const models = modelsFor(sel, train, dataset);

  return (
    <div
      style={{
        height: 56,
        flex: "none",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 16,
        padding: "0 20px",
        background: C.panel,
        borderBottom: `1px solid ${C.line}`,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 14, flex: "none" }}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke={C.accent} strokeWidth="1.6" strokeLinecap="round">
          <path d="M5 4h14v11a3 3 0 0 1-3 3H8a3 3 0 0 1-3-3z" />
          <path d="M5 9h14" />
          <path d="M8 18l-2.5 3" />
          <path d="M16 18l2.5 3" />
          <path d="M9 13h.01" />
          <path d="M15 13h.01" />
        </svg>
        <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: "0.13em" }}>
            <span style={{ color: C.accentBright }}>TEAM BUS MRT WALK</span> &middot; TRAIN DIGITAL TWIN
          </div>
          <div style={{ fontSize: 9.5, letterSpacing: "0.16em", color: C.dim }}>
            TRACK 3 &middot; PROBLEM STATEMENT 3 &middot; CONDITION MONITORING OF ROLLING STOCK
          </div>
        </div>
      </div>

      <PageNav page={page} onPage={onPage} />

      {page === "twin" && (
      <div style={{ display: "flex", alignItems: "center", gap: 8, flex: "none" }}>
        <div className="nx-chip">
          NORTH SOUTH LINE
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke={C.dim} strokeWidth="2">
            <path d="M6 9l6 6 6-6" />
          </svg>
        </div>
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 9, letterSpacing: "0.14em", color: C.dim2 }}>SET</span>
          <select
            className="nx-select"
            value={trainId || ""}
            onChange={(e) => onSelectTrain && onSelectTrain(e.target.value)}
            aria-label="Train set"
          >
            {(trainIds || []).map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </label>
      </div>
      )}

      {page === "twin" && (
      <div style={{ display: "flex", alignItems: "center", gap: 5, flex: "none" }}>
        {SUBSYSTEMS.map((s) => {
          const m = models[s];
          return (
            <span
              key={s}
              className="nx-tag"
              style={{ color: m ? C.violet : C.dim2, borderColor: m ? C.line2 : C.grid }}
              title={
                m
                  ? `${s}: ${m.model} (${
                      m.source === "selection" ? "/api/bench/selection" : "replay frame"
                    })`
                  : `${s}: no model in this frame`
              }
            >
              <span style={{ color: C.dim }}>{SUB_LABEL[s]}</span> {m ? shortModel(m.model) : "—"}
            </span>
          );
        })}
      </div>
      )}

      {page === "twin" ? (
      <div style={{ display: "flex", alignItems: "center", gap: 12, flex: "none" }}>
        <div
          className="mono"
          style={{ fontSize: 15, fontWeight: 500, letterSpacing: "0.04em", whiteSpace: "nowrap" }}
          title="Sim clock (UTC)"
        >
          {fmtStamp(frame && frame.ts)} <span style={{ color: C.dim }}>UTC</span>
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            height: 22,
            padding: "0 9px",
            border: `1px solid ${C.warn}`,
            background: "#fff5df",
            color: C.warn,
            fontSize: 9.5,
            fontWeight: 600,
            letterSpacing: "0.12em",
            whiteSpace: "nowrap",
          }}
          title={
            "Replay of stored sample data. " +
            (connected ? "Replay websocket connected." : "Replay websocket not connected.")
          }
        >
          <span
            style={{
              width: 6,
              height: 6,
              flex: "none",
              display: "block",
              borderRadius: "50%",
              background: connected ? C.accent : C.line2,
            }}
          />
          SAMPLE &middot; REPLAY
        </div>
      </div>
      ) : (
        <div style={{ display: "flex", alignItems: "center", gap: 10, flex: "none" }}>
          <span className="nx-tag" style={{ color: C.dim, borderColor: C.line2 }}>
            OFFLINE &middot; 4 / 4 MODELS
          </span>
          {page === "predict" ? (
            <TutorialButton onPage={onPage} />
          ) : (
            <>
              <button type="button" className="nx-btn" style={{ height: 26, flex: "none" }} onClick={() => onPage?.("predict", { tour: true })}>
                START TUTORIAL
              </button>
              <button type="button" className="nx-btn nx-btn--primary" style={{ height: 26, flex: "none" }} onClick={() => onPage?.("predict")}>
                OPEN DIGITAL TWIN
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
