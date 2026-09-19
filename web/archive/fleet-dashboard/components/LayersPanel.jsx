import React from "react";
import {
  usePs3StreamStore,
  setLayerActive,
  setAllLayersActive,
  getLayerCurrent,
} from "../state/ps3StreamStore.js";
import { C } from "../lib/format.js";

const LAYER_CONFIG = [
  { key: "brake", label: "Brake air supply", source: "sim · MetroPT-3 calibrated", isPs3: false },
  { key: "bearing", label: "Axle bearing (sim)", source: "sim · Ottawa calibrated", isPs3: false },
  { key: "door", label: "Door", source: "LTA PS3", isPs3: true, task: "door", unit: "s" },
  { key: "acv", label: "ACV refrigerant leak", source: "LTA PS3 upload", isPs3: true, task: "acv", unit: "h" },
  { key: "rail", label: "Rail corrugation", source: "LTA PS3 upload", isPs3: true, task: "rail", unit: "s" },
  { key: "shm", label: "SHM fatigue damage", source: "LTA PS3 upload", isPs3: true, task: "shm", unit: "s" },
];

export default function LayersPanel({ style, width = 310, height = 464 }) {
  const store = usePs3StreamStore();
  const { layers } = store;

  const allActive = Object.values(layers).every((l) => l.active);

  return (
    <div
      className="nx-layers-panel"
      style={{
        width,
        height,
        display: "flex",
        flexDirection: "column",
        background: C.panel,
        border: `1px solid ${C.line}`,
        padding: "6px 8px",
        overflowY: "auto",
        boxSizing: "border-box",
        ...style,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 6,
          paddingBottom: 4,
          borderBottom: `1px solid ${C.line}`,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: "0.12em", color: C.text }}>
            SIX-SYSTEM LAYERS
          </span>
          <span className="nx-tag" style={{ fontSize: 8.5, color: C.accent, padding: "0 4px" }}>
            TWIN + PS3
          </span>
        </div>
        <button
          type="button"
          className="nx-btn"
          style={{ fontSize: 8.5, padding: "1px 6px", height: 20 }}
          onClick={() => setAllLayersActive(!allActive)}
        >
          {allActive ? "ALL OFF" : "ALL ON"}
        </button>
      </div>

      {/* Systems List */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {LAYER_CONFIG.map(({ key, label, source, isPs3, task, unit }) => {
          const lState = layers[key] || {};
          const active = !!lState.active;
          const files = lState.files || [];
          const hasFiles = files.length > 0;
          const current = isPs3 ? getLayerCurrent(key, lState) : null;

          return (
            <div
              key={key}
              style={{
                display: "flex",
                flexDirection: "column",
                padding: "6px 8px",
                background: active ? "#f8fafb" : "#f4f7f8",
                border: `1px solid ${active ? (isPs3 ? "#b9dede" : C.line2) : C.line}`,
                borderRadius: 2,
                opacity: active ? 1 : 0.6,
                transition: "all 0.15s ease",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6 }}>
                <label
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    cursor: "pointer",
                    userSelect: "none",
                    minWidth: 0,
                  }}
                >
                  <input
                    type="checkbox"
                    checked={active}
                    onChange={(e) => setLayerActive(key, e.target.checked)}
                    style={{ accentColor: C.accent, cursor: "pointer" }}
                  />
                  <span
                    style={{
                      fontSize: 10.5,
                      fontWeight: 600,
                      color: active ? C.text : C.dim2,
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {label}
                  </span>
                </label>
                <span
                  className="mono"
                  style={{
                    fontSize: 8,
                    letterSpacing: "0.04em",
                    padding: "1px 5px",
                    borderRadius: 2,
                    background: isPs3 ? "rgba(0, 143, 149, 0.12)" : "rgba(130, 141, 158, 0.15)",
                    color: isPs3 ? C.accent : C.dim,
                    border: `1px solid ${isPs3 ? "rgba(0, 143, 149, 0.25)" : "transparent"}`,
                    whiteSpace: "nowrap",
                    flex: "none",
                  }}
                >
                  {lState.example ? "predicted example" : source}
                </span>
              </div>

              {/* Status / file note */}
              {isPs3 && (
                <div style={{ marginTop: 4, paddingLeft: 22, fontSize: 9.5 }}>
                  {!hasFiles ? (
                    <span style={{ color: "#a66500", fontStyle: "italic" }}>
                      upload on Predict or use Fault / Example
                    </span>
                  ) : current ? (
                    <div className="mono" style={{ color: C.dim, fontSize: 9, display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ color: C.text2 }}>{current.file.name}</span>
                      <span>·</span>
                      <span>step {current.frameIdx + 1}/{current.totalFrames}</span>
                      <span>·</span>
                      <span style={{ color: C.accent }}>
                        {Number(current.frame.t || 0).toFixed(1)} {current.frame.t_unit || unit}
                      </span>
                    </div>
                  ) : (
                    <span style={{ color: C.dim2 }}>{files.length} file(s) loaded</span>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Progress Lines Summary under the panel */}
      <div
        style={{
          marginTop: "auto",
          paddingTop: 8,
          borderTop: `1px solid ${C.line}`,
          display: "flex",
          flexDirection: "column",
          gap: 3,
        }}
      >
        <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: "0.08em", color: C.dim }}>
          STREAM PROGRESS
        </div>
        {["door", "acv", "rail", "shm"].map((k) => {
          const l = layers[k];
          if (!l || !l.files || !l.files.length) return null;
          const cur = getLayerCurrent(k, l);
          if (!cur) return null;
          const unit = k === "acv" ? "h" : "s";
          const taskName = k;
          return (
            <div
              key={k}
              className="mono"
              style={{
                fontSize: 8.5,
                color: C.dim,
                display: "flex",
                justifyContent: "space-between",
                padding: "1px 0",
              }}
            >
              <span style={{ color: C.text2, textTransform: "uppercase" }}>{taskName}</span>
              <span>
                {cur.file.name} · {cur.frameIdx + 1}/{cur.totalFrames} ·{" "}
                <b style={{ color: C.accent }}>{Number(cur.frame.t || 0).toFixed(1)} {cur.frame.t_unit || unit}</b>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
