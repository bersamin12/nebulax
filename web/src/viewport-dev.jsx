// Scratch entry point for the 3D viewport only: `npx vite` then open
//   http://localhost:5173/viewport-dev.html            (3D, the replay frame of the mock)
//   http://localhost:5173/viewport-dev.html?fallback=1 (2D elevation fallback)
//   http://localhost:5173/viewport-dev.html?mode=rail  (PS3 healthMap: rail corrugation, Side II)
//   http://localhost:5173/viewport-dev.html?mode=acv   (PS3 healthMap: ACV ranking)
//   http://localhost:5173/viewport-dev.html?mode=door  (PS3 healthMap: door segments)
//   http://localhost:5173/viewport-dev.html?mode=shm   (PS3 healthMap: SHM damage)
//   http://localhost:5173/viewport-dev.html?probe=91,172  (hover + click at those page pixels,
//                                                          for headless screenshot checks)
// It is NOT part of the app build (vite only builds index.html); it exists so the viewport can be
// developed and screenshotted without touching App.jsx. Left in place for the integrator.
import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import TrainViewport from "./components/viewport/TrainViewport.jsx";
import { ps3HealthMap } from "./components/viewport/componentMap.js";
import mockFrame from "./mock/frame.json";

/** Sample PS3 predictions, in the shapes the predict page will hand to ps3HealthMap(). */
const PS3_SAMPLES = {
  rail: {
    task: "rail",
    rows: [{ file_id: "Test12.csv", prediction: "Side II" }],
    explanations: [
      {
        file_id: "Test12.csv",
        top_boxes: [
          { car: 3, position: 4 },
          { car: 3, position: 6 },
          { car: 4, position: 2 },
        ],
      },
    ],
  },
  acv: {
    task: "acv",
    rows: [{ file_id: "acv_test_case.xlsx", ranked_cars: "03|01|05|02|04|06|07|08" }],
  },
  door: {
    task: "door",
    rows: [
      { start_time: "2023-7-5-0-0-0-0", end_time: "2023-7-5-0-0-4-500", prediction: "Normal" },
      { start_time: "2023-7-5-0-1-10-0", end_time: "2023-7-5-0-1-13-200", prediction: "Abnormal resistance" },
    ],
  },
  shm: {
    task: "shm",
    rows: [{ file_id: "Test3.csv", prediction: 0.3345 }],
  },
};

/** ?probe=x,y - fire a real hover + click at those page pixels once the model has settled. */
function useProbe() {
  useEffect(() => {
    const q = new URLSearchParams(window.location.search).get("probe");
    if (!q) return;
    const [x, y] = q.split(",").map(Number);
    const t = setTimeout(() => {
      const el = document.elementFromPoint(x, y);
      if (!el) return;
      const opts = { clientX: x, clientY: y, bubbles: true, pointerId: 1, pointerType: "mouse", isPrimary: true, button: 0, buttons: 1 };
      el.dispatchEvent(new PointerEvent("pointermove", { ...opts, buttons: 0 }));
      el.dispatchEvent(new PointerEvent("pointerdown", opts));
      el.dispatchEvent(new PointerEvent("pointerup", { ...opts, buttons: 0 }));
      el.dispatchEvent(new MouseEvent("click", { clientX: x, clientY: y, bubbles: true }));
      el.dispatchEvent(new PointerEvent("pointermove", { ...opts, buttons: 0 }));
    }, 4000);
    return () => clearTimeout(t);
  }, []);
}

const btn = (active) => ({
  font: "500 11px/1 'IBM Plex Mono', monospace",
  padding: "6px 10px",
  color: active ? "#0d1114" : "#d7dce4",
  background: active ? "#4bb8c9" : "#21262e",
  border: "1px solid #3a424f",
  cursor: "pointer",
});

function Harness() {
  const [selected, setSelected] = useState(null);
  const [trainId, setTrainId] = useState(mockFrame.trains[0].train_id);
  const [mode, setMode] = useState(
    () => new URLSearchParams(window.location.search).get("mode") ?? "twin",
  );
  useProbe();
  const healthMap = useMemo(
    () => (PS3_SAMPLES[mode] ? ps3HealthMap(PS3_SAMPLES[mode]) : null),
    [mode],
  );
  return (
    <div style={{ width: 1440, margin: "0 auto", background: "#16191f", minHeight: "100vh" }}>
      <div
        style={{
          height: 56,
          display: "flex",
          alignItems: "center",
          gap: 16,
          padding: "0 20px",
          background: "#1a1e25",
          borderBottom: "1px solid #2c323c",
          color: "#d7dce4",
          font: "13px/1 'IBM Plex Sans', sans-serif",
          letterSpacing: "0.13em",
        }}
      >
        <b>NEBULA X · TRAINVIEWPORT DEV HARNESS</b>
        {mockFrame.trains.map((t) => (
          <button
            key={t.train_id}
            onClick={() => {
              setMode("twin");
              setTrainId(t.train_id);
              setSelected(null);
            }}
            style={btn(mode === "twin" && t.train_id === trainId)}
          >
            {t.train_id}
          </button>
        ))}
        <span style={{ color: "#59616f" }}>|</span>
        {Object.keys(PS3_SAMPLES).map((m) => (
          <button
            key={m}
            onClick={() => {
              setMode(m);
              setSelected(null);
            }}
            style={btn(mode === m)}
          >
            PS3 {m.toUpperCase()}
          </button>
        ))}
      </div>
      <TrainViewport
        frame={mockFrame}
        trainId={mode === "twin" ? trainId : null}
        selected={selected}
        onSelect={setSelected}
        height={206}
        healthMap={healthMap}
      />
      <pre
        style={{
          margin: 0,
          padding: "10px 20px",
          color: "#828d9e",
          font: "11px/1.6 'IBM Plex Mono', monospace",
        }}
      >
        mode = {mode} · selected = {JSON.stringify(selected)}
        {healthMap ? `\nhealthMap = ${JSON.stringify(healthMap).slice(0, 600)}` : ""}
      </pre>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<Harness />);
