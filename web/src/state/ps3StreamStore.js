// PS3 stream store: module-level store shared between Predict and Twin pages.
//
// Keeps per task:
//   - files streamed in this browser session (each with its frames, rows, explanation)
//   - per-layer playhead and clock
//   - layer on/off toggles and reversible predicted-example replay
import { useSyncExternalStore } from "react";

const TASK_NAMES = ["door", "acv", "rail", "shm"];

const FILE_SECONDS = {
  door: 30.0, // the whole stream over 30 s
  acv: 6.0,   // 6 s per file
  rail: 6.0,
  shm: 6.0,
};

let state = {
  layers: {
    brake: { active: true, source: "sim · MetroPT-3 calibrated", label: "Brake air supply" },
    bearing: { active: true, source: "sim · Ottawa calibrated", label: "Axle bearing (sim)" },
    door: { active: true, source: "LTA PS3", label: "Door", files: [], elapsed: 0 },
    acv: { active: true, source: "LTA PS3 upload", label: "ACV refrigerant leak", files: [], elapsed: 0 },
    rail: { active: true, source: "LTA PS3 upload", label: "Rail corrugation", files: [], elapsed: 0 },
    shm: { active: true, source: "LTA PS3 upload", label: "SHM fatigue damage", files: [], elapsed: 0 },
  },
};

const listeners = new Set();

function emit() {
  for (const l of Array.from(listeners)) l();
}

function setState(patch) {
  state = typeof patch === "function" ? patch(state) : { ...state, ...patch };
  emit();
}

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getSnapshot() {
  return state;
}

export function usePs3StreamStore() {
  return useSyncExternalStore(subscribe, getSnapshot);
}

/** Push a stream result from POST /api/ps3/{task}/stream into the store. */
export function addStreamResult(task, res) {
  if (!task || !res) return;
  const key = task;
  setState((s) => {
    const layer = s.layers[key];
    if (!layer || !Array.isArray(layer.files)) return s;
    const frames = Array.isArray(res.frames) ? res.frames : [];
    if (!frames.length) return s;
    const name = res.files?.[res.files.length - 1] || frames[0]?.file_id || (task === "door" ? "Test.csv" : "file");
    const original = layer.exampleSnapshot || layer;
    const existingIdx = original.files.findIndex((f) => f.name === name);
    const item = {
      name,
      frames,
      rows: res.rows || [],
      explanation: res.explanations?.[res.explanations.length - 1] || null,
      fileId: frames[0]?.file_id || name,
    };
    const files = [...original.files];
    if (existingIdx >= 0) {
      files[existingIdx] = item;
    } else {
      files.push(item);
    }
    return {
      ...s,
      layers: {
        ...s.layers,
        [key]: {
          ...layer,
          ...(layer.exampleSnapshot ? { exampleSnapshot: { ...original, files } } : { files }),
        },
      },
    };
  });
}

/** Set whether a layer is active on the twin. */
export function setLayerActive(key, active) {
  setState((s) => {
    const layer = s.layers[key];
    if (!layer) return s;
    return {
      ...s,
      layers: {
        ...s.layers,
        [key]: { ...layer, active: Boolean(active) },
      },
    };
  });
}

/** Set all layers active or inactive. */
export function setAllLayersActive(active = true) {
  setState((s) => {
    const newLayers = {};
    for (const [k, v] of Object.entries(s.layers)) {
      newLayers[k] = { ...v, active: Boolean(active) };
    }
    return { ...s, layers: newLayers };
  });
}

/** Replace one LTA layer with a predicted example; stop restores its uploads and cursor. */
export function startExampleReplay(task, res) {
  const frames = Array.isArray(res?.frames) ? res.frames : [];
  if (!TASK_NAMES.includes(task) || !frames.length) return false;
  setState((s) => {
    const layer = s.layers[task];
    const snapshot = layer.exampleSnapshot || {
      files: layer.files, elapsed: layer.elapsed, active: layer.active, source: layer.source,
    };
    const name = res.example?.file || res.files?.[0] || frames[0].file_id || "example";
    const index = Math.max(0, Math.min(frames.length - 1, Number(res.example?.frame_index) || 0));
    const seconds = FILE_SECONDS[task] || 6;
    const target = ((index + 0.5) / frames.length) * seconds;
    return { ...s, layers: { ...s.layers, [task]: {
      ...layer,
      files: [{ name, frames, rows: res.rows || [], explanation: res.explanations?.[0] || null }],
      elapsed: Math.max(0, target - 2),
      exampleTarget: target,
      active: true,
      source: res.source || "LTA PS3 predicted example",
      example: res.example || { file: name, frame_index: index, prediction_only: true },
      exampleSnapshot: snapshot,
    } } };
  });
  return true;
}

export function stopExampleReplay(task) {
  setState((s) => {
    const layer = s.layers[task];
    if (!layer?.exampleSnapshot) return s;
    const { exampleSnapshot, example, exampleTarget, ...rest } = layer;
    return { ...s, layers: { ...s.layers, [task]: { ...rest, ...exampleSnapshot } } };
  });
}

/** Browser upload fallback when the server has no local Test data. */
export function uploadedExample(task) {
  const layer = state.layers[task];
  const files = layer?.exampleSnapshot?.files || layer?.files || [];
  const candidates = files.flatMap((file) => (file.frames || []).map((frame, frame_index) => ({ file, frame, frame_index })));
  let picked = null;
  if (task === "door") picked = candidates.find((c) => c.frame.rows?.some((r) => /abnormal/i.test(String(r.prediction)))) || null;
  if (task === "rail") picked = candidates.find((c) => c.frame.rows?.some((r) => /^side (i|ii)$/i.test(String(r.prediction)))) || null;
  if (task === "acv") picked = candidates.findLast((c) => c.frame.rows?.[0]?.ranked_cars) || null;
  if (task === "shm") picked = candidates.filter((c) => c.frame.final).sort((a, b) => Number(b.frame.rows?.[0]?.prediction) - Number(a.frame.rows?.[0]?.prediction))[0] || null;
  if (!picked) return null;
  return {
    task, source: "browser uploaded LTA PS3 stream", files: [picked.file.name],
    frames: picked.file.frames, rows: picked.file.rows,
    explanations: picked.file.explanation ? [picked.file.explanation] : [],
    example: { file: picked.file.name, frame_index: picked.frame_index, criterion: "predicted uploaded example", prediction_only: true },
  };
}

/** Advance playhead for each PS3 layer by dt wall-clock seconds. */
export function tickPlayheads(dt, playing = true, speedRatio = 1.0) {
  if (!playing || dt <= 0) return;
  setState((s) => {
    let changed = false;
    const newLayers = { ...s.layers };
    const ps3Keys = TASK_NAMES;
    for (const k of ps3Keys) {
      const l = newLayers[k];
      if (!l || !l.files || !l.files.length) continue;
      changed = true;
      newLayers[k] = {
        ...l,
        elapsed: l.example ? Math.min(l.exampleTarget, l.elapsed + dt * speedRatio) : l.elapsed + dt * speedRatio,
      };
    }
    return changed ? { ...s, layers: newLayers } : s;
  });
}

/** Compute current frame information for a PS3 layer. */
export function getLayerCurrent(layerKey, layerState) {
  if (!layerState || !layerState.files || !layerState.files.length) return null;
  const task = layerKey;
  const files = layerState.files;
  const elapsed = layerState.elapsed || 0;

  if (task === "door") {
    const file = files[0];
    const frames = file.frames;
    if (!frames || !frames.length) return null;
    const totalSec = FILE_SECONDS.door;
    const progress = (layerState.example ? Math.min(elapsed, totalSec - 1e-6) : elapsed % totalSec) / totalSec;
    const frameIdx = Math.min(frames.length - 1, Math.floor(progress * frames.length));
    return {
      task: "door",
      file,
      fileIdx: 0,
      totalFiles: 1,
      frame: frames[frameIdx],
      frameIdx,
      totalFrames: frames.length,
    };
  }

  const perFileSec = FILE_SECONDS[task] || 6.0;
  const totalSec = files.length * perFileSec;
  const timeInLoop = layerState.example ? Math.min(elapsed, totalSec - 1e-6) : elapsed % totalSec;
  const fileIdx = Math.min(files.length - 1, Math.floor(timeInLoop / perFileSec));
  const file = files[fileIdx];
  const frames = file.frames;
  if (!frames || !frames.length) return null;
  const timeInFile = timeInLoop - fileIdx * perFileSec;
  const progressInFile = timeInFile / perFileSec;
  const frameIdx = Math.min(frames.length - 1, Math.floor(progressInFile * frames.length));

  return {
    task,
    file,
    fileIdx,
    totalFiles: files.length,
    frame: frames[frameIdx],
    frameIdx,
    totalFrames: frames.length,
  };
}
