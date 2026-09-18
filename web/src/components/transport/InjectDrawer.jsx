// Inject drawer - POST /api/sim/inject for the selected component (contract section 4).
// Targets the current schematic selection; the selects below double as a picker when
// nothing is selected. Fault types are the injectable subset of nebulax.schema.FAULT_TYPES.
import { useEffect, useMemo, useState } from "react";
import { postInject, clearInject, getPs3ExampleStream } from "../../api.js";
import { startExampleReplay, stopExampleReplay, uploadedExample, usePs3StreamStore } from "../../state/ps3StreamStore.js";

export const FAULT_TYPES = {
  pneumatic: ["air_leak", "oil_leak", "compressor_wear", "dryer_valve_stuck", "clogged_filter"],
  bearing: ["bearing_degradation", "hot_axle_box", "outer_race", "inner_race", "sensor_stuck", "sensor_offset"],
};

export const COMPONENT_IDS = {
  pneumatic: ["apu_1"],
  bearing: ["axlebox_1L", "axlebox_1R", "axlebox_2L", "axlebox_2R", "axlebox_3L", "axlebox_3R", "axlebox_4L", "axlebox_4R"],
};

const SUBSYSTEMS = ["pneumatic", "bearing", "door", "acv", "rail", "shm"];
const LTA = new Set(["door", "acv", "rail", "shm"]);
/** Real recorded units: they have no simulator to re-run, so POST /sim/inject cannot target them. */
const RECORDED = new Set(["MP3"]);
const CARS = [1, 2, 3, 4, 5, 6];
const DASH = "–";

const pretty = (s) => String(s || "").replace(/_/g, " ");

function targetLabel(t) {
  if (!t.train_id || !t.component_id) return "no target";
  const car = t.car === 0 || t.car === "0" ? "UNIT" : `C${t.car}`;
  return `${t.train_id} / ${car} / ${String(t.component_id).toUpperCase()}`;
}

function summarise(res) {
  if (!res) return "";
  if (res.message) return res.message;
  const eps = Array.isArray(res.episodes) ? res.episodes : [];
  const peak = eps.reduce((m, e) => Math.max(m, Number(e?.peak_score) || 0), 0);
  const leads = eps.map((e) => Number(e?.lead_to_failure_h)).filter((x) => Number.isFinite(x));
  const bits = [
    `${res.run_id ?? "run"} ok`,
    `${res.n_rows ?? "?"} rows`,
    `${eps.length} episode${eps.length === 1 ? "" : "s"}`,
  ];
  if (peak > 0) bits.push(`peak ${peak.toFixed(2)}`);
  if (leads.length) bits.push(`lead ${Math.max(...leads).toFixed(0)} h`);
  if (Number.isFinite(Number(res.elapsed_s))) bits.push(`${Number(res.elapsed_s).toFixed(1)} s`);
  return bits.join(" · ");
}

export default function InjectDrawer({ replay, frame, selected, onInjected, onExampleSelected, open = true, onToggle }) {
  const streamStore = usePs3StreamStore();
  const [pick, setPick] = useState({ train_id: null, car: 1, subsystem: "bearing", component_id: null });
  const [faultType, setFaultType] = useState(FAULT_TYPES.bearing[0]);
  const [ramp, setRamp] = useState(7);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // The whole fleet (GET /api/trains via the store), not just the trains in this frame. MP3 is
  // real recorded data, not a simulator run, so it cannot be injected into.
  // The whole fleet (GET /api/trains via the store), not just the trains in this frame. MP3 is
  // recorded Porto-metro data rather than a simulator run, so it can be named but not injected
  // into - it only appears in the list while it is the current target, so the select is never
  // blank under an "MP3 / ..." label.
  const trains = useMemo(() => {
    const ids = (replay?.trainIds?.length ? replay.trainIds : (frame?.trains ?? []).map((t) => t.train_id))
      .filter(Boolean)
      .filter((id) => !RECORDED.has(id));
    const cur = replay?.trainId;
    if (cur && !ids.includes(cur)) ids.unshift(cur);
    return ids.length ? ids : ["T01"];
  }, [frame, replay?.trainId, replay?.trainIds]);

  // The schematic selection wins whenever it changes.
  const selKey = selected
    ? `${selected.train_id}|${selected.car}|${selected.subsystem}|${selected.component_id}`
    : "";
  useEffect(() => {
    if (!selected || !selected.component_id) {
      // Deselecting drops the target, so the button disables with a reason; the picker
      // below still lets an operator aim it by hand.
      setPick((p) => ({ ...p, component_id: null }));
      setResult(null);
      setError(null);
      return;
    }
    const sub = SUBSYSTEMS.includes(selected.subsystem) ? selected.subsystem : "bearing";
    setPick({
      train_id: selected.train_id ?? replay?.trainId ?? trains[0],
      car: sub === "pneumatic" ? 0 : Number(selected.car ?? 1),
      subsystem: sub,
      component_id: LTA.has(sub) ? null : selected.component_id,
    });
    if (FAULT_TYPES[sub]) setFaultType((ft) => (FAULT_TYPES[sub].includes(ft) ? ft : FAULT_TYPES[sub][0]));
    setResult(null);
    setError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selKey]);

  const target = {
    train_id: pick.train_id ?? replay?.trainId ?? trains[0],
    car: pick.subsystem === "pneumatic" ? 0 : Number(pick.car ?? 1),
    component_id: pick.component_id,
    subsystem: pick.subsystem,
  };
  const recorded = RECORDED.has(target.train_id);
  const isLta = LTA.has(pick.subsystem);
  const ready = Boolean(target.train_id && target.component_id && faultType) && !recorded && !isLta;
  const reason = recorded
    ? `${target.train_id} is recorded data, not a simulator run: pick a T01-T10 set to inject.`
    : !target.component_id
      ? "Select a component on the schematic, or pick one here."
      : !faultType
        ? "Pick a fault type."
        : "";

  function setSubsystem(sub) {
    setPick((p) => ({
      ...p,
      subsystem: sub,
      car: sub === "pneumatic" ? 0 : p.car || 1,
      component_id: COMPONENT_IDS[sub]?.includes(p.component_id) ? p.component_id : null,
    }));
    if (FAULT_TYPES[sub]) setFaultType((ft) => (FAULT_TYPES[sub].includes(ft) ? ft : FAULT_TYPES[sub][0]));
  }

  async function doInject() {
    if (!ready || busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const res = await postInject({
        train_id: target.train_id,
        car: target.car,
        component_id: target.component_id,
        fault_type: faultType,
        severity_ramp_days: Number(ramp),
        // the onset is the current replay position (contract section 4: the API defaults
        // t_onset to this ts), so the drift starts where the operator is looking
        ts: replay?.ts ?? frame?.ts ?? undefined,
      });
      setResult(res);
      replay?.refresh?.();
      onInjected?.(res);
    } catch (err) {
      setError(String(err?.message || err));
    } finally {
      setBusy(false);
    }
  }

  async function doClear() {
    if (busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      await clearInject();
      setResult({ message: "Injected faults cleared \u00b7 overlay removed." });
      replay?.refresh?.();
      onInjected?.(null);
    } catch (err) {
      setError(String(err?.message || err));
    } finally {
      setBusy(false);
    }
  }

  async function doExample() {
    if (busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      let res;
      try {
        res = await getPs3ExampleStream(pick.subsystem);
      } catch (err) {
        if (err?.status !== 404) throw err;
        res = uploadedExample(pick.subsystem);
        if (!res) throw new Error(`Local ${pick.subsystem.toUpperCase()} Test data is unavailable. Upload a qualifying file on Predict, then return here.`);
      }
      if (!startExampleReplay(pick.subsystem, res)) throw new Error("The predicted example has no preview frames.");
      onExampleSelected?.(pick.subsystem, res);
      setResult({ message: `${res.example?.file || res.files?.[0]} · ${res.example?.criterion || "predicted example"} · prediction only` });
    } catch (err) {
      setError(String(err?.message || err));
    } finally {
      setBusy(false);
    }
  }

  function doStopExample() {
    stopExampleReplay(pick.subsystem);
    onExampleSelected?.(pick.subsystem, null);
    setResult({ message: "Example stopped; previous stream and cursor restored." });
    setError(null);
  }

  if (!open) {
    return (
      <div className="tb-inject is-closed">
        <button
          type="button"
          className="tb-toggle"
          onClick={onToggle}
          title="Open the inject drawer"
          aria-label="Open the inject drawer"
          aria-expanded="false"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1f2374" strokeWidth="1.8">
            <path d="M4 7h16" /><path d="M4 17h16" />
            <circle cx="10" cy="7" r="2.5" /><circle cx="16" cy="17" r="2.5" />
          </svg>
        </button>
        <span className="tb-inject-spine">FAULT / EXAMPLE</span>
      </div>
    );
  }

  if (isLta) {
    const activeExample = Boolean(streamStore.layers[pick.subsystem]?.example);
    return (
      <div className="tb-inject">
        <div className="tb-inject-head">
          <div className="tb-inject-title">SIX-SYSTEM FAULT CONTROL</div>
          <span className="tb-target">LTA PS3 · PREDICTED EXAMPLE</span>
          <button type="button" className="tb-toggle" onClick={onToggle} aria-label="Close the inject drawer" aria-expanded="true">›</button>
        </div>
        <div className="tb-inject-grid">
          <label className="tb-field is-wide">
            <span className="tb-field-label">System</span>
            <select className="tb-select" value={pick.subsystem} onChange={(e) => setSubsystem(e.target.value)}>
              {SUBSYSTEMS.map((s) => <option key={s} value={s}>{s === "pneumatic" ? "brake air supply · sim" : s === "bearing" ? "axle bearing · sim" : `${s.toUpperCase()} · LTA PS3`}</option>)}
            </select>
          </label>
          <div className="tb-field is-wide" style={{ fontSize: 10, color: "#a66500" }}>
            Replays a model prediction from the LTA Test set. This is not a ground-truth fault or a change to recorded data.
          </div>
        </div>
        <div className="tb-inject-foot">
          <span className={"tb-note" + (error ? " is-err" : result ? " is-ok" : "")} role={error ? "alert" : undefined} title={error || result?.message || ""}>
            {error || result?.message || "Inspect a predicted example in the twin."}
          </span>
          {activeExample && <button type="button" className="tb-btn is-ghost" onClick={doStopExample} disabled={busy}>STOP</button>}
          <button type="button" className="tb-btn" onClick={doExample} disabled={busy}>{busy ? "LOADING" : "REPLAY EXAMPLE"}</button>
        </div>
      </div>
    );
  }

  const cids = COMPONENT_IDS[pick.subsystem] ?? [];
  const note = error
    ? error
    : busy
      ? "Re-simulating the component and re-scoring with the fitted winner…"
      : result
        ? summarise(result)
        : ready
          ? `Drift ramped over ${ramp} simulated day${ramp === 1 ? "" : "s"} from the replay cursor.`
          : reason;

  return (
    <div className="tb-inject">
      <div className="tb-inject-head">
        <div className="tb-inject-title">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1f2374" strokeWidth="1.8">
            <path d="M4 7h16" /><path d="M4 17h16" />
            <circle cx="10" cy="7" r="2.5" /><circle cx="16" cy="17" r="2.5" />
          </svg>
          SIX-SYSTEM FAULT CONTROL &middot; SIMULATOR
        </div>
        <span className="tb-target" title={targetLabel(target)}>{targetLabel(target)}</span>
        <button
          type="button"
          className="tb-toggle"
          onClick={onToggle}
          title="Close the inject drawer"
          aria-label="Close the inject drawer"
          aria-expanded="true"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M9 6l6 6-6 6" />
          </svg>
        </button>
      </div>

      <div className="tb-inject-grid">
        <label className="tb-field">
          <span className="tb-field-label"><span>Train</span></span>
          <select
            className="tb-select"
            value={target.train_id ?? ""}
            onChange={(e) => setPick((p) => ({ ...p, train_id: e.target.value }))}
          >
            {trains.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>

        <label className="tb-field">
          <span className="tb-field-label"><span>Car</span></span>
          <select
            className="tb-select"
            value={String(target.car)}
            disabled={pick.subsystem === "pneumatic"}
            onChange={(e) => setPick((p) => ({ ...p, car: Number(e.target.value) }))}
          >
            {pick.subsystem === "pneumatic"
              ? <option value="0">unit (car 0)</option>
              : CARS.map((c) => <option key={c} value={String(c)}>car {c}</option>)}
          </select>
        </label>

        <label className="tb-field">
          <span className="tb-field-label"><span>Subsystem</span></span>
          <select className="tb-select" value={pick.subsystem} onChange={(e) => setSubsystem(e.target.value)}>
            {SUBSYSTEMS.map((s) => <option key={s} value={s}>{s === "pneumatic" ? "brake air supply · sim" : s === "bearing" ? "axle bearing · sim" : `${s.toUpperCase()} · LTA PS3`}</option>)}
          </select>
        </label>

        <label className="tb-field">
          <span className="tb-field-label"><span>Component</span></span>
          <select
            className="tb-select"
            value={pick.component_id ?? ""}
            onChange={(e) => setPick((p) => ({ ...p, component_id: e.target.value || null }))}
          >
            <option value="">{DASH} pick {DASH}</option>
            {cids.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>

        <label className="tb-field is-wide">
          <span className="tb-field-label"><span>Fault type</span></span>
          <select className="tb-select" value={faultType} onChange={(e) => setFaultType(e.target.value)}>
            {(FAULT_TYPES[pick.subsystem] ?? []).map((f) => (
              <option key={f} value={f}>{pretty(f)}</option>
            ))}
          </select>
        </label>

        <label className="tb-field is-wide">
          <span className="tb-field-label">
            <span>Severity ramp</span>
            <span className="tb-field-value">{ramp} d</span>
          </span>
          <input
            className="tb-range"
            type="range"
            min="1"
            max="12"
            step="1"
            value={ramp}
            onChange={(e) => setRamp(Number(e.target.value))}
            aria-label="Severity ramp, days"
          />
        </label>
      </div>

      <div className="tb-inject-foot">
        <span
          className={"tb-note" + (error ? " is-err" : result && !busy ? " is-ok" : "")}
          title={note}
          role={error ? "alert" : undefined}
        >
          {note}
        </span>
        <button type="button" className="tb-btn is-ghost" onClick={doClear} disabled={busy}>
          CLEAR
        </button>
        <button
          type="button"
          className="tb-btn"
          onClick={doInject}
          disabled={!ready || busy}
          title={ready ? `Inject ${faultType} on ${targetLabel(target)}` : reason}
        >
          {busy ? "WORKING" : "INJECT"}
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4">
            <path d="M5 12h14" /><path d="M13 6l6 6-6 6" />
          </svg>
        </button>
      </div>
    </div>
  );
}
