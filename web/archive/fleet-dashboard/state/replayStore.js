// Replay store: one module-level store shared by the whole console.
//
//   useReplay() -> {
//     frame,        // the latest replay frame (contract section 5), all trains
//     train,        // the selected train's entry of frame.trains, or null
//     components,   // train?.components ?? []   (convenience, always an array)
//     connected,    // true while a live source (WS or REST polling) is answering
//     playing, speed, trainId, ts,
//     trains,       // frame.trains (objects); NOT the fleet list - see trainIds
//     trainIds,     // every train GET /api/trains knows, MP3 included (selector list)
//     fleetLoaded,  // true once that list has been fetched (or the fetch has failed)
//     source,       // "ws" | "rest" | "aux" | "mock"
//     mode,         // "connecting" | "ws" | "rest" | "aux" | "offline"
//     error,        // last transport error message, or null
//     play(), pause(), toggle(), seek(tsIso), setSpeed(n), selectTrain(id), refresh()
//   }
//
// Transport, in order of preference:
//   1. WS  ws://<host>/api/replay  - sends {cmd:"play"|"pause"|"seek"|"speed", ts?, speed?},
//      applies pushed frames, reconnects with exponential backoff.
//   2. REST GET /api/train/{id}/state?ts=  once a second while playing, driven by a local
//      clock, so the console also works against a plain REST backend.
//   3. web/src/mock/frame.json with a locally advancing ts ("offline: mock frame").
import { useMemo, useSyncExternalStore } from "react";
import mockFrame from "../mock/frame.json";
import { getState, getTrains } from "../api.js";

/** ?offline=1 pins the console to the bundled mock frame (screenshot / no-backend check). */
const FORCE_OFFLINE =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("offline") === "1";

/**
 * MetroPT-3 is one real Porto-metro unit on its own 2020 clock. `WS /api/replay` only ever
 * pushes the sim fleet (`frame_at` with no train filter = `sim_train_ids()`), so MP3 is reached
 * through `GET /api/train/MP3/state`, which defaults to the end of *its* clock. While it is
 * selected the store publishes that frame instead of the live one and ignores WS pushes: the
 * contract is explicit that the replay never mixes the two clocks.
 */
const AUX_TRAINS = new Set(["MP3"]);
export const isAuxTrain = (id) => AUX_TRAINS.has(id);

// ---------------------------------------------------------------- sim clock
export const CLOCK_START = "2026-09-01T00:00:00Z";
export const CLOCK_END = "2026-09-30T23:59:00Z";
export const CLOCK_START_MS = Date.parse(CLOCK_START);
export const CLOCK_END_MS = Date.parse(CLOCK_END);

/** speed = sim seconds per wall second. */
export const DEFAULT_SPEED = 8640; // 1 day / 10 s
export const SPEED_PRESETS = [
  { value: 3600, label: "1 h/s", title: "3600x - one simulated hour per second" },
  { value: 8640, label: "1 DAY / 10 s", title: "8640x - default" },
  { value: 21600, label: "6 h/s", title: "21600x - six simulated hours per second" },
  { value: 86400, label: "1 DAY/s", title: "86400x - one simulated day per second" },
];

export const clampMs = (ms) => Math.min(CLOCK_END_MS, Math.max(CLOCK_START_MS, ms));
/** Whole-second ISO-8601 UTC, the form every /api route and the WS protocol expect. */
export const isoOf = (ms) => new Date(Math.floor(clampMs(ms) / 1000) * 1000).toISOString().replace(".000Z", "Z");

// ---------------------------------------------------------------- state
const MOCK_TS_MS = Date.parse(mockFrame.ts);

let simMs = Number.isFinite(MOCK_TS_MS) ? clampMs(MOCK_TS_MS) : CLOCK_START_MS;

let state = {
  frame: { ...mockFrame, ts: isoOf(simMs) },
  connected: false,
  // The console opens running: the replay is the demo. A WS backend overrides this with
  // whatever the server reports on its first frame.
  playing: mockFrame.playing !== false,
  speed: DEFAULT_SPEED,
  trainId: mockFrame.trains?.[0]?.train_id ?? "T01",
  ts: isoOf(simMs),
  source: "mock",
  mode: "connecting",
  error: null,
  // Every train the backend knows about, MP3 included. `frame.trains` is not this list: the
  // REST /state route answers with one train and the WS with the sim fleet only. Empty until
  // GET /api/trains answers, so nothing resolves a train id against the mock fleet by mistake.
  trainIds: [],
  fleetLoaded: false,
};

const listeners = new Set();

function emit() {
  for (const l of Array.from(listeners)) l();
}

function setState(patch) {
  let changed = false;
  for (const k of Object.keys(patch)) {
    if (!Object.is(state[k], patch[k])) {
      changed = true;
      break;
    }
  }
  if (!changed) return;
  state = { ...state, ...patch };
  emit();
}

function subscribe(listener) {
  listeners.add(listener);
  start();
  return () => listeners.delete(listener);
}

const getSnapshot = () => state;

// ---------------------------------------------------------------- frames
function mockAt(ms) {
  return { ...mockFrame, ts: isoOf(ms), playing: state.playing, speed: state.speed };
}

function applyFrame(frame, source) {
  if (!frame || typeof frame !== "object") return;
  // While an aux train is selected its own frame owns the display; the live fleet keeps
  // streaming underneath but must not overwrite it.
  if (isAuxTrain(state.trainId) && source !== "aux") return;
  const tsMs = Date.parse(frame.ts);
  // MetroPT (MP3) runs on its own 2020 clock: keep the frame's own ts for display and only
  // drive the local sim clock from timestamps that fall inside the sim month.
  if (Number.isFinite(tsMs) && tsMs >= CLOCK_START_MS && tsMs <= CLOCK_END_MS) simMs = tsMs;
  const patch = { frame, ts: frame.ts ?? isoOf(simMs), source, error: null };
  if (source === "ws") {
    // In WS mode the server owns the clock: mirror what it reports.
    if (typeof frame.playing === "boolean") patch.playing = frame.playing;
    if (Number.isFinite(frame.speed)) patch.speed = frame.speed;
    patch.connected = true;
    patch.mode = "ws";
  } else if (source === "rest") {
    patch.connected = true;
    patch.mode = "rest";
  } else if (source === "aux") {
    patch.connected = true;
    patch.mode = "aux";
    patch.playing = false; // MP3 is a still, not a replay: it has no sim clock to run
  }
  setState(patch);
}

function goOffline(err) {
  setState({
    frame: mockAt(simMs),
    ts: isoOf(simMs),
    source: "mock",
    connected: false,
    mode: "offline",
    error: err ? String(err.message || err) : state.error,
  });
}

// ---------------------------------------------------------------- websocket
let socket = null;
let wsAttempts = 0;
let reconnectTimer = null;
let started = false;

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/api/replay`;
}

function wsSend(msg) {
  if (socket && socket.readyState === 1) {
    try {
      socket.send(JSON.stringify(msg));
      return true;
    } catch {
      /* fall through to the REST path */
    }
  }
  return false;
}

function scheduleReconnect() {
  if (reconnectTimer || !started) return;
  const delay = Math.min(15000, 1000 * 2 ** Math.min(wsAttempts, 4));
  wsAttempts += 1;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connectWs();
  }, delay);
}

function connectWs() {
  if (!started || typeof WebSocket === "undefined") {
    startRest();
    return;
  }
  if (socket && (socket.readyState === 0 || socket.readyState === 1)) return;
  let ws;
  try {
    ws = new WebSocket(wsUrl());
  } catch (err) {
    startRest();
    scheduleReconnect();
    return;
  }
  socket = ws;
  const openTimer = setTimeout(() => {
    if (ws.readyState !== 1) {
      try {
        ws.close();
      } catch {
        /* ignore */
      }
    }
  }, 4000);

  ws.onopen = () => {
    clearTimeout(openTimer);
    wsAttempts = 0;
    stopRest();
    setState({ source: "ws", mode: "ws", connected: true, error: null });
    // Re-assert the client's intent on (re)connect.
    wsSend({ cmd: "speed", speed: state.speed });
    wsSend({ cmd: "seek", ts: isoOf(simMs) });
    wsSend({ cmd: state.playing ? "play" : "pause" });
  };
  ws.onmessage = (ev) => {
    let frame;
    try {
      frame = JSON.parse(ev.data);
    } catch {
      return;
    }
    applyFrame(frame, "ws");
  };
  ws.onerror = () => {
    /* a close event always follows */
  };
  ws.onclose = () => {
    clearTimeout(openTimer);
    if (socket === ws) socket = null;
    if (state.source === "ws") setState({ connected: false, mode: "connecting" });
    startRest();
    scheduleReconnect();
  };
}

// ---------------------------------------------------------------- REST fallback
let restTimer = null;
let restBusy = false;
let restFails = 0;
let pollOnce = true;

function startRest() {
  if (restTimer || !started) return;
  restTimer = setInterval(pollRest, 1000);
  pollOnce = true;
  pollRest();
}

function stopRest() {
  if (restTimer) clearInterval(restTimer);
  restTimer = null;
  restBusy = false;
}

async function pollRest() {
  if (restBusy || socket?.readyState === 1) return;
  if (!state.playing && !pollOnce) return;
  restBusy = true;
  pollOnce = false;
  try {
    const frame = await getState(state.trainId, isoOf(simMs));
    restFails = 0;
    applyFrame(frame, "rest");
  } catch (err) {
    restFails += 1;
    if (restFails >= 2) goOffline(err);
  } finally {
    restBusy = false;
  }
}

// ---------------------------------------------------------------- aux trains (MP3)
let auxTimer = null;
let auxBusy = false;

async function pollAux() {
  const id = state.trainId;
  if (auxBusy || !isAuxTrain(id)) return;
  auxBusy = true;
  try {
    // no ts: the server answers at the end of MP3's own clock
    applyFrame(await getState(id), "aux");
  } catch (err) {
    setState({ error: String(err.message || err), connected: false, mode: "offline" });
  } finally {
    auxBusy = false;
  }
}

function startAux() {
  if (auxTimer || !started) return;
  auxTimer = setInterval(pollAux, 5000);
  pollAux();
}

function stopAux() {
  if (auxTimer) clearInterval(auxTimer);
  auxTimer = null;
  auxBusy = false;
}

// ---------------------------------------------------------------- fleet list
async function loadFleet() {
  try {
    const rows = await getTrains();
    const ids = (Array.isArray(rows) ? rows : []).map((t) => t && t.train_id).filter(Boolean);
    if (ids.length) setState({ trainIds: ids, fleetLoaded: true });
  } catch {
    /* the selector falls back to whatever the current frame carries */
    setState({ fleetLoaded: true });
  }
}

// ---------------------------------------------------------------- local clock
let tickTimer = null;
let lastTickWall = 0;

function tick() {
  const now = Date.now();
  const dt = lastTickWall ? (now - lastTickWall) / 1000 : 0;
  lastTickWall = now;
  if (isAuxTrain(state.trainId)) return; // MP3 does not run on the sim clock
  // The WS server owns the clock while it is connected.
  if (socket && socket.readyState === 1) return;
  if (!state.playing || dt <= 0) return;
  const next = simMs + dt * state.speed * 1000;
  const atEnd = next >= CLOCK_END_MS;
  simMs = atEnd ? CLOCK_END_MS : next;
  const patch = { ts: isoOf(simMs) };
  if (atEnd) patch.playing = false;
  if (state.source === "mock") patch.frame = mockAt(simMs);
  setState(patch);
}

function start() {
  if (started || typeof window === "undefined") return;
  started = true;
  lastTickWall = Date.now();
  tickTimer = setInterval(tick, 250);
  if (FORCE_OFFLINE) {
    goOffline(null);
    setState({
      mode: "offline",
      trainIds: mockFrame.trains.map((t) => t.train_id),
      fleetLoaded: true,
    });
    return;
  }
  loadFleet();
  connectWs();
}

/** Tear the transport down (tests / hot reload); the next useReplay() restarts it. */
export function stopReplay() {
  started = false;
  if (tickTimer) clearInterval(tickTimer);
  tickTimer = null;
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectTimer = null;
  stopRest();
  stopAux();
  if (socket) {
    try {
      socket.close();
    } catch {
      /* ignore */
    }
    socket = null;
  }
}

// ---------------------------------------------------------------- commands
export function play() {
  if (simMs >= CLOCK_END_MS) simMs = CLOCK_START_MS;
  lastTickWall = Date.now();
  setState({ playing: true, ts: isoOf(simMs) });
  wsSend({ cmd: "play" });
  pollOnce = true;
}

export function pause() {
  setState({ playing: false });
  wsSend({ cmd: "pause" });
}

export function toggle() {
  if (state.playing) pause();
  else play();
}

export function seek(ts) {
  const ms = typeof ts === "number" ? ts : Date.parse(ts);
  if (!Number.isFinite(ms)) return;
  simMs = clampMs(ms);
  lastTickWall = Date.now();
  setState({ ts: isoOf(simMs), frame: state.source === "mock" ? mockAt(simMs) : state.frame });
  wsSend({ cmd: "seek", ts: isoOf(simMs) });
  pollOnce = true;
}

export function setSpeed(speed) {
  const n = Number(speed);
  if (!Number.isFinite(n) || n <= 0) return;
  lastTickWall = Date.now();
  setState({ speed: n });
  wsSend({ cmd: "speed", speed: n });
}

/** Switch which train's components the store exposes. No reconnect: frames carry all trains. */
export function selectTrain(id) {
  if (!id || id === state.trainId) return;
  setState({ trainId: id });
  if (isAuxTrain(id)) {
    // MP3 is not in the live frame at all: fetch its own state and hold the sim clock.
    startAux();
    return;
  }
  stopAux();
  // Coming back from MP3 the displayed frame is the 2020 still; ask for a live one at once.
  refresh();
  pollOnce = true;
  if (restTimer) pollRest();
}

/** Force one fetch of the current frame (used after POST /sim/inject). */
export function refresh() {
  if (isAuxTrain(state.trainId)) {
    pollAux();
    return;
  }
  pollOnce = true;
  if (socket && socket.readyState === 1) wsSend({ cmd: "seek", ts: isoOf(simMs) });
  else if (restTimer) pollRest();
  else startRest();
}

const ACTIONS = { play, pause, toggle, seek, setSpeed, selectTrain, refresh, stopReplay };

// ---------------------------------------------------------------- hook
export function useReplay() {
  const snap = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  return useMemo(() => {
    const trains = Array.isArray(snap.frame?.trains) ? snap.frame.trains : [];
    const train = trains.find((t) => t.train_id === snap.trainId) ?? trains[0] ?? null;
    return {
      ...snap,
      trains,
      train,
      components: Array.isArray(train?.components) ? train.components : [],
      clockStart: CLOCK_START,
      clockEnd: CLOCK_END,
      ...ACTIONS,
    };
  }, [snap]);
}

/** Non-hook access, for code outside React. */
export const replayStore = { getSnapshot, subscribe, ...ACTIONS };
