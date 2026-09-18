// Scratch entry for the transport agent: renders TransportBar on its own so the bar can be
// driven and screenshotted without App.jsx (owned by the shell agent).
//
//   node web/dev/mockServer.mjs &      # mock API + WS on 127.0.0.1:8000
//   npx vite                           # http://localhost:5173/transport-dev.html
import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import TransportBar from "./components/transport/TransportBar.jsx";
import { useReplay } from "./state/replayStore.js";

// Collect React warnings and uncaught errors so a headless run can assert a clean console.
if (typeof window !== "undefined" && !window.__errs) {
  window.__errs = [];
  const orig = console.error;
  console.error = (...a) => {
    window.__errs.push(a.map(String).join(" ").slice(0, 300));
    orig(...a);
  };
  window.addEventListener("error", (e) => window.__errs.push("onerror: " + e.message));
  window.addEventListener("unhandledrejection", (e) => window.__errs.push("rejection: " + e.reason));
}

function Harness() {
  const replay = useReplay();
  const [selected, setSelected] = useState({
    train_id: "T01",
    car: 1,
    subsystem: "door",
    component_id: "door_L1",
  });
  const [log, setLog] = useState([]);
  const seen = useRef(new Set());
  const replayRef = useRef(replay);
  replayRef.current = replay;

  // Sample the replay clock so a headless DOM dump can prove that it advances.
  useEffect(() => {
    const id = setInterval(() => {
      const ts = replayRef.current.ts;
      if (seen.current.has(ts)) return;
      seen.current.add(ts);
      setLog((l) => [...l.slice(-7), ts]);
    }, 400);
    return () => clearInterval(id);
  }, []);

  const box = {
    width: 1440,
    margin: "0 auto",
    background: "#16191f",
    border: "1px solid #2c323c",
  };

  return (
    <div style={{ padding: 16 }}>
      <div style={{ ...box, padding: "10px 20px", display: "flex", gap: 18, alignItems: "center", fontSize: 11 }}>
        <strong style={{ letterSpacing: "0.12em" }}>TRANSPORT DEV HARNESS</strong>
        <span style={{ color: "#828d9e" }}>
          mode <b style={{ color: "#7fd3e0" }}>{replay.mode}</b> · connected{" "}
          <b style={{ color: replay.connected ? "#2e9e6b" : "#d99a1e" }}>{String(replay.connected)}</b> · source{" "}
          {replay.source} · train {replay.trainId} · speed {replay.speed}
        </span>
        <button
          type="button"
          onClick={() => setSelected(selected ? null : { train_id: "T01", car: 1, subsystem: "door", component_id: "door_L1" })}
          style={{ background: "#21262e", color: "#d7dce4", border: "1px solid #3a424f", padding: "3px 8px", cursor: "pointer" }}
        >
          toggle selection
        </button>
        <span style={{ color: "#6c778a" }}>selected: {selected ? selected.component_id : "none"}</span>
      </div>

      <div style={box}>
        <TransportBar
          replay={replay}
          frame={replay.frame}
          selected={selected}
          onInjected={() => {}}
          height={144}
        />
      </div>

      <pre
        id="ts-log"
        style={{ ...box, fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#7fd3e0", padding: 10, whiteSpace: "pre-wrap" }}
      >
        {log.join("\n")}
      </pre>
      <div id="ts-count" style={{ ...box, padding: 10, fontSize: 11, color: "#828d9e" }}>
        distinct ts samples: {log.length}
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Harness />
  </React.StrictMode>,
);
