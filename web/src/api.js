// REST client for the NEBULA X demo API (docs/app_contract.md section 4).
// Every route lives under /api; in dev the Vite proxy forwards it to 127.0.0.1:8000.
// All helpers return parsed JSON and throw an Error (with .status when the server answered)
// on a non-2xx response, a timeout or a transport failure.

export const BASE = "/api";

/** Default per-request timeout (ms). POST /sim/inject gets a much longer one. */
const DEFAULT_TIMEOUT_MS = 15000;

/** Build a query string, dropping null / undefined / "" values. */
export function qs(params = {}) {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

async function request(path, { method = "GET", body, timeout = DEFAULT_TIMEOUT_MS, signal } = {}) {
  const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timer = ctrl && timeout ? setTimeout(() => ctrl.abort(), timeout) : null;
  if (signal && ctrl) signal.addEventListener("abort", () => ctrl.abort(), { once: true });
  const init = { method, signal: ctrl ? ctrl.signal : signal };
  if (body !== undefined) {
    init.headers = { "content-type": "application/json" };
    init.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(BASE + path, init);
  } catch (err) {
    const e = new Error(`${method} ${BASE}${path}: ${err && err.name === "AbortError" ? (signal && signal.aborted ? "cancelled" : "timeout") : err}`);
    e.cause = err;
    e.cancelled = !!(signal && signal.aborted);
    throw e;
  } finally {
    if (timer) clearTimeout(timer);
  }
  if (!res.ok) {
    let detail = "";
    try {
      const txt = await res.text();
      try {
        const j = JSON.parse(txt);
        detail = j.detail ? ` - ${typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)}` : "";
      } catch {
        detail = txt ? ` - ${txt.slice(0, 200)}` : "";
      }
    } catch {
      /* body already consumed / unreadable */
    }
    const e = new Error(`${method} ${BASE}${path}: ${res.status}${detail}`);
    e.status = res.status;
    throw e;
  }
  if (res.status === 204) return null;
  return res.json();
}

/**
 * POST /api/ps3/{task}/stream - predict and stream exactly one file, returning preview frames.
 *
 *   postPs3Stream("rail", File, session|null)
 *     -> {session, task, output_filename, rows, explanations, csv_url, n_done, files, errors,
 *         frames, expires_at}
 */
export async function postPs3Stream(task, file, session = null, { timeout = 600000, signal, onUploadProgress } = {}) {
  const fd = new FormData();
  if (file) {
    const f = Array.isArray(file) ? file[0] : file;
    fd.append("files", f, f.name);
  }
  if (session) fd.append("session", session);
  const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timer = ctrl && timeout ? setTimeout(() => ctrl.abort(), timeout) : null;
  if (signal && ctrl) signal.addEventListener("abort", () => ctrl.abort(), { once: true });
  const path = `/ps3/${enc(task)}/stream`;
  if (onUploadProgress) onUploadProgress(0);
  let res;
  try {
    res = await fetch(BASE + path, { method: "POST", body: fd, signal: ctrl ? ctrl.signal : signal });
  } catch (err) {
    const e = new Error(`POST ${BASE}${path}: ${err && err.name === "AbortError" ? (signal && signal.aborted ? "cancelled" : "timeout") : err}`);
    e.cause = err;
    e.cancelled = !!(signal && signal.aborted);
    throw e;
  } finally {
    if (timer) clearTimeout(timer);
  }
  if (!res.ok) {
    let detail = "";
    try {
      const txt = await res.text();
      try {
        const j = JSON.parse(txt);
        detail = j.detail ? (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)) : "";
      } catch {
        detail = txt ? txt.slice(0, 300) : "";
      }
    } catch {
      /* unreadable body */
    }
    const e = new Error(detail || `POST ${BASE}${path}: ${res.status}`);
    e.status = res.status;
    e.detail = detail;
    throw e;
  }
  return res.json();
}


const enc = encodeURIComponent;

/** GET /api/health -> {status, scores_loaded, n_trains, clock:{start,end}} */
export const getHealth = () => request("/health");

/** GET /api/trains -> [{train_id, line, cars, instrumented, faults}] */
export const getTrains = () => request("/trains");

/** GET /api/train/{id}/state?ts= -> one replay frame (contract section 5). */
export const getState = (id, ts) => request(`/train/${enc(id)}/state${qs({ ts })}`);

/** GET .../series?signal=&from=&to=&car=&max_points= -> {signal, unit, points:[[ts,value]]} */
export const getSeries = (id, cid, q = {}) =>
  request(`/train/${enc(id)}/component/${enc(cid)}/series${qs(q)}`);

/** GET .../scores?from=&to=&car= -> {model, threshold, points:[[ts,score,alert]], top_signals} */
export const getScores = (id, cid, q = {}) =>
  request(`/train/${enc(id)}/component/${enc(cid)}/scores${qs(q)}`);

/** GET .../cycle?ts=&car= -> {t, pos, pos_ref, current, pwm} (doors only, 404 otherwise). */
export const getCycle = (id, cid, q = {}) =>
  request(`/train/${enc(id)}/component/${enc(cid)}/cycle${qs(q)}`);

/** GET /api/alerts?ts=&train= -> episodes open or closed at ts, newest first. */
export const getAlerts = (q = {}) => request(`/alerts${qs(q)}`);

/** GET /api/kpis?ts= -> {open_alerts, median_lead_h, fa_per_train_day, events_detected, events_total} */
export const getKpis = (ts) => request(`/kpis${qs({ ts })}`);

/** GET /api/bench/selection -> manifest + the leaderboard selection rows. */
export const getSelection = () => request("/bench/selection");

/**
 * POST /api/sim/inject
 * body {train_id, car, component_id, fault_type, severity_ramp_days, seed?, t_onset?}
 * -> {run_id, n_rows, episodes:[...], elapsed_s}. The contract budgets < 60 s for a door.
 */
export const postInject = (body, opts = {}) =>
  request("/sim/inject", { method: "POST", body, timeout: 120000, ...opts });

/** DELETE /api/sim/inject -> clears the in-memory overlay. */
export const clearInject = () => request("/sim/inject", { method: "DELETE", timeout: 30000 });

/** POST /api/advisory/{episode_id} -> Advisory (contract section 6). */
export const postAdvisory = (episodeId) =>
  request(`/advisory/${enc(episodeId)}`, { method: "POST", timeout: 40000 });

// ------------------------------------------------------------------ PS3 predict (nebulax/api/ps3.py)

/**
 * GET /api/ps3/tasks -> [{name, label, accepts, output_filename, cv, model_loaded, available,
 * detail, max_file_bytes, max_files_per_request}] in door / acv / rail / shm order.
 */
export const getPs3Tasks = () => request("/ps3/tasks");

/** Desktop-only fallback for browsers whose native picker returns no File objects. */
export const inspectPs3LocalPath = (task, path, { signal } = {}) =>
  request(`/ps3/${enc(task)}/local/inspect`, { method: "POST", body: { path }, timeout: 30000, signal });

export const postPs3LocalStream = (task, path, session = null, { signal } = {}) =>
  request(`/ps3/${enc(task)}/local/stream`, {
    method: "POST", body: { path, session }, timeout: 600000, signal,
  });

/** Check ACV workbook structure before adding it to the prediction queue. */
export async function validateAcvWorkbook(file, { signal } = {}) {
  const body = new FormData();
  body.append("file", file, file.name);
  const res = await fetch(`${BASE}/ps3/acv/validate`, { method: "POST", body, signal });
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json()).detail || ""; } catch { /* keep HTTP status */ }
    throw new Error(detail || `workbook check failed (${res.status})`);
  }
  return res.json();
}

/** Deterministic model-predicted example from local organiser Test data. */
export const getPs3ExampleStream = (task) =>
  request(`/ps3/${enc(task)}/example/stream`, { timeout: 180000 });

/** GET /api/ps3/{task}/cv -> the whole results/ps3/<task>_cv.json (404 when it is not written). */
export const getPs3Cv = (task) => request(`/ps3/${enc(task)}/cv`);

/**
 * POST /api/ps3/{task}/predict - one **batch** of uploads (at most `max_files_per_request`,
 * which is 32; the 68 rail files are ~1.1 GB, so the page repeats the call with the session it
 * gets back). Multipart, and deliberately *not* through `request`: setting a content-type by
 * hand would drop the boundary the browser computes for the FormData.
 *
 *   postPs3Predict("rail", [File, File], session|null)
 *     -> {session, task, output_filename, rows, explanations, csv_url, n_done, files, errors,
 *         expires_at}
 *
 * Throws with `.status` (the FastAPI `detail` appended) on 4xx/5xx, exactly like `request`.
 * The timeout is per batch and generous: a rail file is ~17 MB and is predicted server-side
 * before the response comes back.
 */
export async function postPs3Predict(task, files, session = null, { timeout = 600000, signal, onUploadProgress } = {}) {
  const fd = new FormData();
  for (const f of files || []) fd.append("files", f, f.name);
  if (session) fd.append("session", session);
  const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timer = ctrl && timeout ? setTimeout(() => ctrl.abort(), timeout) : null;
  if (signal && ctrl) signal.addEventListener("abort", () => ctrl.abort(), { once: true });
  const path = `/ps3/${enc(task)}/predict`;
  if (onUploadProgress) onUploadProgress(0);
  let res;
  try {
    // no `headers`: the browser writes multipart/form-data; boundary=... itself
    res = await fetch(BASE + path, { method: "POST", body: fd, signal: ctrl ? ctrl.signal : signal });
  } catch (err) {
    const e = new Error(`POST ${BASE}${path}: ${err && err.name === "AbortError" ? (signal && signal.aborted ? "cancelled" : "timeout") : err}`);
    e.cause = err;
    e.cancelled = !!(signal && signal.aborted);
    throw e;
  } finally {
    if (timer) clearTimeout(timer);
  }
  if (!res.ok) {
    let detail = "";
    try {
      const txt = await res.text();
      try {
        const j = JSON.parse(txt);
        detail = j.detail ? (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)) : "";
      } catch {
        detail = txt ? txt.slice(0, 300) : "";
      }
    } catch {
      /* unreadable body */
    }
    const e = new Error(detail || `POST ${BASE}${path}: ${res.status}`);
    e.status = res.status;
    e.detail = detail;
    throw e;
  }
  return res.json();
}

/** The download link for a session (a plain URL: the browser fetches it, not us). */
export const ps3CsvUrl = (session) => `${BASE}/ps3/results/${enc(session)}.csv`;

/** DELETE /api/ps3/results/{session} -> {session, deleted, n_files, n_rows}. */
export const deletePs3Session = (session) =>
  request(`/ps3/results/${enc(session)}`, { method: "DELETE", timeout: 30000 });
