// NEBULA X console shell: fixed 1440x900, scaled (never reflowed).
//
// Two pages share the shell and the 56 px header (`?page=` in the URL, the nav in Header.jsx):
//   predict  (default)  the Problem Statement 3 upload-and-predict page - web/src/predict/
//   twin     `?page=twin`  the replay operations console, unchanged:
//                          header 56 / ticker 30 / schematic 206 / body 464 / transport 144
// Only the chosen page is mounted, so the twin's replay websocket is never opened on the
// predict page (and the predict page's /api/ps3/tasks is never fetched on the twin).
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import "./styles.css";

import { useReplay } from "./state/replayStore.js";
import TrainViewport from "./components/viewport/TrainViewport.jsx";
import TransportBar from "./components/transport/TransportBar.jsx";

import PredictPage from "./predict/PredictPage.jsx";
import Header from "./components/Header.jsx";
import AlertTicker from "./components/AlertTicker.jsx";
import SubsystemLanes from "./components/SubsystemLanes.jsx";
import DetailPanel from "./components/DetailPanel.jsx";
import LayersPanel from "./components/LayersPanel.jsx";
import { usePs3StreamStore, tickPlayheads, getLayerCurrent } from "./state/ps3StreamStore.js";
import { componentKey, indexFrame, mergeHealthMaps, ps3HealthMap } from "./components/viewport/componentMap.js";
import { C, worstComponent } from "./lib/format.js";

const W = 1440;
const H = 900;

/**
 * Deep link, read once on load: `?train=T04&car=5&component=door_L3` (`subsystem` optional - it
 * is inferred from the component id). `train` alone just opens that set; a `component` also
 * preselects it, which suppresses the "worst component" auto-selection for that train.
 */
function readDeepLink() {
  if (typeof window === "undefined") return null;
  const q = new URLSearchParams(window.location.search);
  const train_id = q.get("train");
  const component_id = q.get("component");
  if (!train_id && !component_id) return null;
  const car = q.get("car");
  const inferred = /^door_/.test(component_id || "")
    ? "door"
    : /^axlebox_/.test(component_id || "")
      ? "bearing"
      : /^apu/.test(component_id || "")
        ? "pneumatic"
        : null;
  return {
    train_id: train_id || null,
    component_id: component_id || null,
    car: car === null || car === "" ? null : Number(car),
    subsystem: q.get("subsystem") || inferred,
  };
}

/** `?page=twin` selects the replay console; anything else (and no query at all) = Predict. */
function readPage() {
  if (typeof window === "undefined") return "predict";
  return new URLSearchParams(window.location.search).get("page") === "twin" ? "twin" : "predict";
}

function useConsoleScale() {
  const [scale, setScale] = useState(1);
  useLayoutEffect(() => {
    const compute = () => {
      const w = window.innerWidth || W;
      const h = window.innerHeight || H;
      setScale(Math.max(0.2, Math.min(1, w / W, h / H)));
    };
    compute();
    let ro = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(compute);
      ro.observe(document.documentElement);
    }
    window.addEventListener("resize", compute);
    return () => {
      if (ro) ro.disconnect();
      window.removeEventListener("resize", compute);
    };
  }, []);
  return scale;
}

function TwinConsole({ page, onPage }) {
  const replay = useReplay();
  const frame = replay.frame || null;
  const streamStore = usePs3StreamStore();

  const deepLink = useRef(readDeepLink());
  const [trainOverride, setTrainOverride] = useState(null);
  const [selected, setSelected] = useState(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [viewMode, setViewMode] = useState("split"); // "split" | "layers" | "lanes"
  const autoFor = useRef(null);
  const linkApplied = useRef(false);
  const lastTick = useRef(typeof performance !== "undefined" ? performance.now() : Date.now());
  const exampleSelection = useRef({});

  // Drive playhead ticks from transport bar
  useEffect(() => {
    let rafId;
    const loop = (now) => {
      const dt = Math.min(0.5, (now - lastTick.current) / 1000);
      lastTick.current = now;
      const speedRatio = (replay.speed || 8640) / 8640;
      if (replay.playing && dt > 0) {
        tickPlayheads(dt, true, speedRatio);
      }
      rafId = requestAnimationFrame(loop);
    };
    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [replay.playing, replay.speed]);

  // The selector lists the whole fleet (GET /api/trains, MP3 included), not just the trains the
  // current frame happens to carry: the REST /state route answers with one train only.
  const frameIds = ((frame && frame.trains) || []).map((t) => t.train_id);
  const trainIds = (replay.trainIds || []).length ? replay.trainIds : frameIds;
  const trainId = trainOverride || replay.trainId || trainIds[0] || null;
  // Exact match only. `trainOf` falls back to the frame's first train, which would paint T01's
  // lanes under an "MP3" header for the second or two it takes MP3's own frame to arrive.
  const train = ((frame && frame.trains) || []).find((t) => t.train_id === trainId) || null;

  useEffect(() => {
    if (trainOverride && replay.trainId === trainOverride) setTrainOverride(null);
  }, [replay.trainId, trainOverride]);

  // deep link: apply once, as soon as the fleet list is known
  useEffect(() => {
    const link = deepLink.current;
    // wait for the real fleet list, or a ?train= id would be checked against the mock frame
    if (!link || linkApplied.current || !replay.fleetLoaded || !trainIds.length) return;
    linkApplied.current = true;
    const id = link.train_id && trainIds.includes(link.train_id) ? link.train_id : trainId;
    if (id && id !== replay.trainId) {
      setTrainOverride(id);
      replay.selectTrain(id);
    }
    if (link.component_id) {
      autoFor.current = id; // a deep-linked component wins over the worst-component default
      setSelected({
        train_id: id,
        car: link.car,
        subsystem: link.subsystem,
        component_id: link.component_id,
      });
    }
  }, [trainIds, trainId, replay]);

  // first meaningful selection per train: the worst component the frame reports
  useEffect(() => {
    if (!train || !trainId) return;
    if (autoFor.current === trainId) return;
    autoFor.current = trainId;
    const w = worstComponent((train.components || []).filter((c) => c.subsystem !== "door"));
    setSelected(
      w
        ? {
            train_id: trainId,
            car: w.car,
            subsystem: w.subsystem,
            component_id: w.component_id,
          }
        : null
    );
  }, [trainId, train]);

  const onSelectTrain = useCallback(
    (id) => {
      setTrainOverride(id);
      if (replay.selectTrain) replay.selectTrain(id);
    },
    [replay]
  );

  const onSelect = useCallback((sel) => {
    setSelected(sel ? { ...sel } : null);
  }, []);

  const onInjected = useCallback(() => setReloadKey((k) => k + 1), []);

  const onExampleSelected = useCallback((task, res) => {
    if (!res) {
      setSelected(exampleSelection.current[task] || null);
      delete exampleSelection.current[task];
      return;
    }
    exampleSelection.current[task] = selected;
    const frameIndex = Number(res.example?.frame_index ?? res.frames.length - 1);
    const frame = res.frames[Math.max(0, Math.min(res.frames.length - 1, frameIndex))];
    const row = frame?.rows?.at(-1) || res.rows?.at(-1) || {};
    const car = task === "acv" ? Number(String(row.ranked_cars || "01").split("|")[0]) : Number(frame?.viewport?.car) || 1;
    const component_id = task === "door" ? "door_L1"
      : task === "acv" ? "ac_1"
        : task === "rail" ? (String(row.prediction).toLowerCase() === "side ii" ? "rail_II" : "rail_I")
          : "bogie_1";
    setSelected({ train_id: trainId, car: task === "rail" ? 0 : car, subsystem: task, component_id });
    replay.play?.();
  }, [selected, trainId, replay]);

  // PS3 current frames
  const doorCur = streamStore.layers.door?.active ? getLayerCurrent("door", streamStore.layers.door) : null;
  const acvCur = streamStore.layers.acv?.active ? getLayerCurrent("acv", streamStore.layers.acv) : null;
  const railCur = streamStore.layers.rail?.active ? getLayerCurrent("rail", streamStore.layers.rail) : null;
  const shmCur = streamStore.layers.shm?.active ? getLayerCurrent("shm", streamStore.layers.shm) : null;

  const doorMap = doorCur
    ? ps3HealthMap({
        task: "door",
        rows: doorCur.frame.rows.slice(-1),
        explanations: [{ file_id: doorCur.file.name, viewport: doorCur.frame.viewport, numbers: doorCur.frame.numbers }],
        fileId: doorCur.file.name,
      })
    : null;

  const acvMap = acvCur
    ? ps3HealthMap({
        task: "acv",
        rows: acvCur.frame.rows,
        explanations: [{ file_id: acvCur.file.name, viewport: acvCur.frame.viewport, numbers: acvCur.frame.numbers }],
        fileId: acvCur.file.name,
      })
    : null;

  const railMap = railCur
    ? ps3HealthMap({
        task: "rail",
        rows: railCur.frame.rows,
        explanations: [{ file_id: railCur.file.name, viewport: railCur.frame.viewport, numbers: railCur.frame.numbers }],
        fileId: railCur.file.name,
      })
    : null;

  const shmMap = shmCur
    ? ps3HealthMap({
        task: "shm",
        rows: shmCur.frame.rows,
        explanations: [{ file_id: shmCur.file.name, viewport: shmCur.frame.viewport, numbers: shmCur.frame.numbers }],
        fileId: shmCur.file.name,
      })
    : null;

  const rawTwin = useMemo(() => indexFrame(train), [train]);
  const filteredTwin = useMemo(() => {
    const m = new Map();
    for (const [k, v] of rawTwin.entries()) {
      if (k.includes("|pneumatic|") && !streamStore.layers.brake?.active) continue;
      if (k.includes("|bearing|") && !streamStore.layers.bearing?.active) continue;
      if (k.includes("|door|")) continue;
      m.set(k, v);
    }
    return m;
  }, [rawTwin, streamStore.layers.brake?.active, streamStore.layers.bearing?.active]);

  const overlay = useMemo(() => {
    return mergeHealthMaps(
      {
        twin: filteredTwin,
        door: doorMap,
        acv: acvMap,
        rail: railMap,
        shm: shmMap,
      }
    );
  }, [filteredTwin, doorMap, acvMap, railMap, shmMap]);

  const showSim = useCallback((c) =>
    c.subsystem !== "door" &&
    (c.subsystem !== "pneumatic" || streamStore.layers.brake?.active) &&
    (c.subsystem !== "bearing" || streamStore.layers.bearing?.active),
  [streamStore.layers.brake?.active, streamStore.layers.bearing?.active]);
  const visibleTrain = useMemo(() => train ? {
    ...train, components: (train.components || []).filter(showSim),
  } : null, [train, showSim]);
  const visibleFrame = useMemo(() => frame ? {
    ...frame,
    trains: (frame.trains || []).map((t) => ({ ...t, components: (t.components || []).filter(showSim) })),
    alerts: (frame.alerts || []).filter((a) => !/^door_/i.test(String(a.component_id || ""))),
    ticker: (frame.ticker || []).filter((line) => !/\bdoor\b/i.test(String(line))),
  } : null, [frame, showSim]);
  const lanesTrain = useMemo(() => {
    if (!visibleTrain) return null;
    const entries = [...overlay.entries()].filter(([k]) => /\|door\|/.test(k));
    const doors = entries.map(([k, value]) => {
      const [car, subsystem, component_id] = k.split("|");
      return { car: Number(car), subsystem, component_id, health: value.health,
        score: value.health === "crit" ? 1 : 0, threshold: 0.5, model: "LTA PS3" };
    });
    return { ...visibleTrain, components: [...visibleTrain.components, ...doors] };
  }, [visibleTrain, overlay]);
  const systemStats = useMemo(() => ({
    active: Object.values(streamStore.layers).filter((layer) => layer.active).length,
    alerts: [...overlay.values()].filter((entry) => entry.health === "crit").length,
    examples: ["door", "acv", "rail", "shm"].filter((task) => streamStore.layers[task]?.example).length,
    trainId,
  }), [streamStore.layers, overlay, trainId]);

  const captions = useMemo(() => {
    const out = [];
    if (doorCur) {
      const step = doorCur.frameIdx;
      const n_steps = doorCur.totalFrames;
      out.push(`DOOR · cycle ${step + 1}/${n_steps} · ${streamStore.layers.door?.example ? "predicted example · " : ""}causal preview · final frame = submitted row`);
    }
    if (railCur) {
      const file_id = railCur.file.name;
      const t = Number(railCur.frame.t || 0);
      const speed = Number(railCur.frame.numbers?.speed_kmh_so_far ?? railCur.frame.numbers?.speed_kmh ?? 0);
      const speedText = Number(railCur.frame.numbers?.speed_ready) ? `${Math.round(speed)} km/h` : "estimating speed";
      out.push(`RAIL · ${file_id} · ${t.toFixed(1)} s · ${speedText} · ${streamStore.layers.rail?.example ? "predicted example · " : ""}demo ordering (files are unordered)`);
    }
    if (shmCur) {
      const file_id = shmCur.file.name;
      const d_running = Number(shmCur.frame.numbers?.damage_running ?? 0);
      const d_final = Number(shmCur.frame.numbers?.damage_final ?? 0);
      out.push(`SHM · ${file_id} · ${Number(shmCur.frame.t || 0).toFixed(0)} ${shmCur.frame.t_unit || "sample"} · D = ${d_running.toFixed(3)} → ${d_final.toFixed(3)} · ${streamStore.layers.shm?.example ? "predicted example · " : ""}Miner accumulation, ends on submitted value · demo ordering (files are unordered)`);
    }
    if (acvCur) {
      const t = Number(acvCur.frame.t || 0);
      const top_car = String(acvCur.frame.numbers?.top_car ?? "01").padStart(2, "0");
      out.push(`ACV · ${t.toFixed(1)} h · car ${top_car} rank 1 · ${streamStore.layers.acv?.example ? "predicted example · " : ""}prefix ranking`);
    }
    return out;
  }, [doorCur, railCur, shmCur, acvCur, streamStore.layers]);

  const ps3Details = useMemo(() => {
    if (!selected) return null;
    const key = componentKey(selected.car, selected.subsystem, selected.component_id);
    const entry = overlay ? overlay.get(key) : null;
    const task = selected.subsystem;
    if (!["door", "acv", "rail", "shm"].includes(task) || !entry) return null;
    const layerKey = task;
    const layerState = streamStore.layers[layerKey];
    const current = layerState?.active ? getLayerCurrent(layerKey, layerState) : null;
    if (!current) return null;

    return {
      task,
      selected,
      key,
      entry,
      layerState,
      current,
      fileId: entry?.fileId || current?.file?.name || (current?.file?.frames?.[0]?.file_id) || "—",
      numbers: entry?.numbers || current?.frame?.numbers || null,
      frame: current?.frame || null,
      frameIdx: current?.frameIdx ?? 0,
      totalFrames: current?.totalFrames ?? 1,
      health: entry?.health || "ok",
      label: entry?.label || null,
    };
  }, [selected, overlay, streamStore]);

  return (
    <>
      <Header
        frame={visibleFrame}
        train={visibleTrain}
        trainId={trainId}
        trainIds={trainIds}
        onSelectTrain={onSelectTrain}
        connected={replay.connected}
        page={page}
        onPage={onPage}
      />

      <AlertTicker frame={visibleFrame} />

      <div
        style={{
          height: 206,
          flex: "none",
          borderBottom: `1px solid ${C.line}`,
          overflow: "hidden",
        }}
      >
        <TrainViewport
          frame={visibleFrame}
          trainId={trainId}
          selected={selected}
          onSelect={onSelect}
          height={206}
          overlay={overlay}
          captions={captions}
        />
      </div>

      <div
        style={{
          height: 464,
          flex: "none",
          display: "flex",
          flexDirection: "column",
          padding: "8px 16px 0 16px",
          gap: 6,
        }}
      >
        {/* View mode toggle bar */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: 20 }}>
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <span style={{ fontSize: 9.5, fontWeight: 700, letterSpacing: "0.1em", color: C.dim }}>VIEW:</span>
            <button
              type="button"
              className="nx-btn"
              style={{
                fontSize: 9,
                padding: "2px 8px",
                height: 20,
                background: viewMode === "split" ? C.accent : "transparent",
                color: viewMode === "split" ? "#000" : C.dim,
                fontWeight: viewMode === "split" ? 700 : 400,
              }}
              onClick={() => setViewMode("split")}
            >
              SIX-SYSTEM LAYERS + LANES
            </button>
            <button
              type="button"
              className="nx-btn"
              style={{
                fontSize: 9,
                padding: "2px 8px",
                height: 20,
                background: viewMode === "layers" ? C.accent : "transparent",
                color: viewMode === "layers" ? "#000" : C.dim,
                fontWeight: viewMode === "layers" ? 700 : 400,
              }}
              onClick={() => setViewMode("layers")}
            >
              LAYERS ONLY
            </button>
            <button
              type="button"
              className="nx-btn"
              style={{
                fontSize: 9,
                padding: "2px 8px",
                height: 20,
                background: viewMode === "lanes" ? C.accent : "transparent",
                color: viewMode === "lanes" ? "#000" : C.dim,
                fontWeight: viewMode === "lanes" ? 700 : 400,
              }}
              onClick={() => setViewMode("lanes")}
            >
              LANES ONLY
            </button>
          </div>
          {captions.length > 0 ? (
            <span className="mono" style={{ fontSize: 9, color: C.accent }}>
              {captions.length} PS3 STREAM{captions.length > 1 ? "S" : ""} ACTIVE
            </span>
          ) : null}
        </div>

        {/* Middle panels grid */}
        <div
          style={{
            flex: 1,
            display: "grid",
            gridTemplateColumns:
              viewMode === "split"
                ? "290px minmax(0, 1fr) 470px"
                : viewMode === "layers"
                  ? "minmax(0, 1fr) 470px"
                  : "minmax(0, 1fr) 470px",
            gap: 10,
            minHeight: 0,
          }}
        >
          {viewMode === "split" ? (
            <>
              <LayersPanel width={290} height={430} />
              <SubsystemLanes
                train={lanesTrain}
                trainId={trainId}
                selected={selected}
                onSelect={onSelect}
              />
            </>
          ) : viewMode === "layers" ? (
            <LayersPanel width="100%" height={430} />
          ) : (
            <SubsystemLanes
              train={lanesTrain}
              trainId={trainId}
              selected={selected}
              onSelect={onSelect}
            />
          )}
          <DetailPanel
            frame={visibleFrame}
            train={visibleTrain}
            selected={selected}
            playing={replay.playing}
            reloadKey={reloadKey}
            ps3Details={ps3Details}
          />
        </div>
      </div>

      <div style={{ height: 144, flex: "none", overflow: "hidden" }}>
        <TransportBar
          replay={replay}
          frame={visibleFrame}
          selected={selected}
          onInjected={onInjected}
          onExampleSelected={onExampleSelected}
          systemStats={systemStats}
          height={144}
        />
      </div>
    </>
  );
}

export default function App() {
  const scale = useConsoleScale();
  const [page, setPage] = useState(readPage);

  // the nav writes `?page=` so a link can be pasted, and the Back button switches pages back
  const onPage = useCallback((next) => {
    setPage(next);
    if (typeof window === "undefined") return;
    const q = new URLSearchParams(window.location.search);
    q.set("page", next);
    if (next === "twin") q.delete("task");
    window.history.pushState(null, "", `${window.location.pathname}?${q}`);
  }, []);

  useEffect(() => {
    const onPop = () => setPage(readPage());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  return (
    <div className="nx-stage">
      <div style={{ width: W * scale, height: H * scale, flex: "none" }}>
        <div className="nx-console" style={{ transform: `scale(${scale})` }}>
          {page === "twin" ? (
            <TwinConsole page={page} onPage={onPage} />
          ) : (
            <>
              <Header page={page} onPage={onPage} />
              <PredictPage height={H - 56} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
