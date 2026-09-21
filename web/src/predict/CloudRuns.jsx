import { useEffect, useState } from "react";
import { getSavedRuns } from "../api.js";
import { C } from "../lib/format.js";

export default function CloudRuns({ p, task }) {
  const [open, setOpen] = useState(false);
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [enabled, setEnabled] = useState(null);
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let live = true;
    setBusy(true); setError("");
    getSavedRuns(task, open ? start : "", open ? end : "").then((res) => {
      if (!live) return;
      setEnabled(res.enabled); setRuns(res.runs); setSelected(res.runs[0]?.id || "");
    }).catch((e) => live && setError(e.message)).finally(() => live && setBusy(false));
    return () => { live = false; };
  }, [task, open, start, end, refresh]);
  const action = async (fn, close = true) => {
    setBusy(true); setError("");
    try { await fn(); if (close) setOpen(false); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };
  const field = { background: C.panel, color: C.text, border: `1px solid ${C.line2}`, padding: 5, minWidth: 0 };
  const entry = runs.find((r) => r.id === selected);
  return <div style={{ flex: "none", border: `1px solid ${C.line}`, padding: 8, fontSize: 10, background: C.panel }}>
    <div style={{ display: "flex", gap: 5, alignItems: "center" }}>
      <input aria-label="Run name" maxLength={80} placeholder={`${task}-run (optional name)`}
        value={p.runName} onChange={(e) => p.setRunName(e.target.value)}
        disabled={p.running || !!p.session} style={{ ...field, flex: 1, width: 100 }} />
      <button className="nx-btn" disabled={p.running} onClick={() => setOpen(true)}>CLOUD RUNS</button>
    </div>
    <div aria-live="polite" style={{ marginTop: 5, color: p.storage?.status === "error" ? C.crit : C.dim }}>
      {p.storage?.status === "saved" ? `Saved: ${p.storage.run.name}` :
        p.storage?.status === "error" ? p.storage.message :
        enabled ? "Run saves inputs and results to the shared cloud library." :
        enabled === false ? "Cloud saving is not configured on this server." : "Checking cloud storage…"}
      {p.storage?.status === "error" && <button className="nx-btn" disabled={busy || p.running} onClick={() => action(p.retrySave, false)}>RETRY SAVE</button>}
    </div>
    {!open && error && <div role="alert" style={{ color: C.crit }}>{error}</div>}
    {open && <div role="presentation" style={{ position: "fixed", inset: 0, zIndex: 2000, background: "#0009", display: "grid", placeItems: "center" }}>
      <section role="dialog" aria-modal="true" aria-label="Shared cloud runs" style={{ background: C.panel, color: C.text, width: 660, maxHeight: "85%", overflow: "auto", padding: 22, border: `1px solid ${C.line2}`, fontSize: 13 }}>
        <div style={{ display: "flex", justifyContent: "space-between" }}><strong>SHARED CLOUD RUNS · {task.toUpperCase()}</strong><button autoFocus className="nx-btn" disabled={busy} onClick={() => setOpen(false)}>CLOSE</button></div>
        <p>Newest first. Open saved results instantly, or queue their inputs for a new prediction.</p>
        <div style={{ display: "flex", gap: 12 }}>
          <label>From (UTC) <input aria-label="From date" type="date" value={start} onChange={(e) => setStart(e.target.value)} style={field} /></label>
          <label>To (UTC) <input aria-label="To date" type="date" value={end} onChange={(e) => setEnd(e.target.value)} style={field} /></label>
          <button className="nx-btn" disabled={busy} onClick={() => setRefresh((v) => v + 1)}>REFRESH</button>
        </div>
        {error && <p role="alert" style={{ color: C.crit }}>{error}</p>}
        {busy && <p role="status">Loading…</p>}
        {enabled === false && <p>Cloud storage is not configured on this server.</p>}
        {!busy && enabled && !runs.length && <p>No saved runs in this date range.</p>}
        {!!runs.length && <>
          <select aria-label="Saved run" value={selected} disabled={busy} onChange={(e) => setSelected(e.target.value)} style={{ ...field, width: "100%", marginTop: 16 }}>
            {runs.map((r, i) => <option key={r.id} value={r.id}>{i === 0 ? "Latest · " : ""}{r.created_at.replace("T", " ").slice(0, 19)} UTC · {r.name} · {r.n_done} predicted</option>)}
          </select>
          <ul style={{ maxHeight: 150, overflow: "auto" }}>{entry?.files.map((f) => <li key={f.name}>{f.name} · {(f.size / 1024).toFixed(1)} KB</li>)}</ul>
          {!!entry?.errors?.length && <p style={{ color: C.crit }}>This run includes file errors; only successful predictions are shown.</p>}
          <div style={{ display: "flex", gap: 10 }}>
            <button className="nx-btn" disabled={busy || !entry?.n_done} onClick={() => action(() => p.openSavedRun(selected))}>OPEN SAVED RESULTS</button>
            <button className="nx-btn" disabled={busy || !entry} onClick={() => action(() => p.queueSavedRun(selected))}>USE INPUTS FOR NEW RUN</button>
          </div>
        </>}
      </section>
    </div>}
  </div>;
}
