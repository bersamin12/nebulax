// Dev-only mock of the NEBULA X API (docs/app_contract.md sections 4 and 5).
// No npm packages: node's own `http` plus a minimal RFC 6455 handshake/framing, so the
// transport bar can be driven end to end (WS push, REST polling, inject) without the
// FastAPI backend. Not a substitute for it: the data is web/src/mock/frame.json with a
// moving clock.
//
//   node web/dev/mockServer.mjs [--port 8000] [--no-ws] [--speed 8640] [--paused]
//
// Then `npx vite` in web/ proxies /api (and the ws upgrade) to it.
import http from "node:http";
import crypto from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BASE_FRAME = JSON.parse(readFileSync(path.join(HERE, "..", "src", "mock", "frame.json"), "utf8"));

const argv = process.argv.slice(2);
const argOf = (name, dflt) => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt;
};
const PORT = Number(argOf("--port", process.env.PORT || 8000));
const WS_ENABLED = !argv.includes("--no-ws");

const CLOCK_START = Date.parse("2026-09-01T00:00:00Z");
const CLOCK_END = Date.parse("2026-09-30T23:59:00Z");
const clamp = (ms) => Math.min(CLOCK_END, Math.max(CLOCK_START, ms));
const iso = (ms) => new Date(clamp(ms)).toISOString().replace(".000Z", "Z");

// ------------------------------------------------------------------ clock
let simMs = clamp(Date.parse(BASE_FRAME.ts));
let speed = Number(argOf("--speed", 8640));
let playing = !argv.includes("--paused");
let lastWall = Date.now();

setInterval(() => {
  const now = Date.now();
  const dt = (now - lastWall) / 1000;
  lastWall = now;
  if (!playing) return;
  simMs += dt * speed * 1000;
  if (simMs >= CLOCK_END) {
    simMs = CLOCK_END;
    playing = false;
  }
}, 100);

// ----------------------------------------------------------------- overlay
/** POST /api/sim/inject writes here; every GET reads it first, as the real API does. */
let overlay = null;
let injectSeq = 0;


function frameAt(ms, trainId) {
  const wobble = 0.35 * Math.sin(ms / (6 * 3600 * 1000));
  const trains = BASE_FRAME.trains.map((t) => ({
    ...t,
    components: t.components.map((c) => {
      const hit =
        overlay &&
        overlay.train_id === t.train_id &&
        overlay.component_id === c.component_id &&
        Number(overlay.car) === Number(c.car);
      const score = hit
        ? Number((c.threshold * 1.9).toFixed(3))
        : Number(Math.max(0, c.score + wobble * (c.score || 0.2)).toFixed(3));
      // health and alert must agree with the wobbled score, or the mock emits "warn" rows that
      // still carry alert=true - which the real API never does (contract section 5).
      const alert = hit ? true : Boolean(c.alert) && score > c.threshold;
      const health = alert ? "crit" : score > c.threshold ? "warn" : score > 0.8 * c.threshold ? "warn" : "ok";
      return { ...c, score, health, alert };
    }),
  }));
  const alerts = [...(BASE_FRAME.alerts ?? [])];
  if (overlay) alerts.unshift(overlay.alert);
  // no debug line: the ticker is the console's advisory feed, and a "mock replay <iso>" row
  // reads as fleet content on screen. The offline badge in the transport bar says it is mock.
  const ticker = [...(BASE_FRAME.ticker ?? [])];
  const kpis = { ...BASE_FRAME.kpis };
  if (overlay) kpis.open_alerts = (kpis.open_alerts ?? 0) + 1;
  const frame = { ts: iso(ms), speed, playing, trains, alerts, kpis, ticker };
  if (trainId) frame.trains = trains.filter((t) => t.train_id === trainId).concat(
    trains.filter((t) => t.train_id !== trainId),
  );
  return frame;
}

// -------------------------------------------------------------------- REST
const send = (res, code, body) => {
  const txt = JSON.stringify(body);
  res.writeHead(code, {
    "content-type": "application/json",
    "access-control-allow-origin": "*",
    "access-control-allow-headers": "content-type",
    "access-control-allow-methods": "GET,POST,DELETE,OPTIONS",
    "cache-control": "no-store",
  });
  res.end(txt);
};

function readBody(req) {
  return new Promise((resolve) => {
    let raw = "";
    req.on("data", (c) => {
      raw += c;
    });
    req.on("end", () => {
      try {
        resolve(raw ? JSON.parse(raw) : {});
      } catch {
        resolve({});
      }
    });
  });
}

// ------------------------------------------------------------------ PS3 mock state
/** GET /api/ps3/tasks - the same entry shape the real router publishes. */
const PS3_TASKS = [
  {
    name: "door",
    label: "Door",
    accepts: [".csv"],
    output_filename: "door_predictions.csv",
    cv: { task: "door", scheme: "5 contiguous time blocks of the raw stream, end to end", n_blocks: 5, seeds: [0], metric: "iou_f1", git_rev: "mock", headline: { model: "multirocket_ridge", iou_f1_mean: 0.94, iou_f1_sd: 0.03 } },
    model_loaded: true,
  },
  {
    name: "acv",
    label: "ACV",
    accepts: [".xlsx", ".xls"],
    output_filename: "acv_predictions.csv",
    cv: { task: "acv", scheme: "leave-one-case-out (6 cases, exploratory)", seeds: [0], metric: "rank_decay (ACV Info Kit section 4)", score: 0.979, git_rev: "mock" },
    model_loaded: true,
  },
  {
    name: "rail",
    label: "Rail Corrugation",
    accepts: [".csv"],
    output_filename: "rail_predictions.csv",
    cv: { task: "rail", scheme: "grouped 5-fold over runs", seeds: [0], metric: "macro_f1", git_rev: "mock", headline: { macro_f1_mean: 0.87 } },
    model_loaded: true,
  },
  {
    name: "shm",
    label: "SHM",
    accepts: [".csv"],
    output_filename: "shm_predictions.csv",
    cv: { task: "shm", scheme: "loo + repeated 5x10-fold", seeds: [0, 1, 2, 3, 4], metric: "mape_score", git_rev: "mock", headline: { mape_score_mean: 0.82 } },
    model_loaded: true,
  },
].map((t) => ({ ...t, available: true, detail: null, max_file_bytes: 64 * 1024 * 1024, max_files_per_request: 32 }));

const PS3_COLUMNS = {
  door: ["start_time", "end_time", "prediction"],
  acv: ["file_id", "ranked_cars"],
  rail: ["file_id", "prediction"],
  shm: ["file_id", "prediction"],
};

const ps3Sessions = new Map();
let ps3Seq = 0;

/** The whole request body (uploads are small in the mock; the real API streams to disk). */
function readRaw(req) {
  return new Promise((resolve) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
  });
}

/**
 * Enough of multipart/form-data for the mock: the `filename="..."` of every part and the
 * `session` field. Latin-1 so a byte offset is a character offset; the payloads are ignored.
 */
function parseMultipart(buf) {
  const text = buf.toString("latin1");
  const names = [];
  const re = /name="(?:files|files\[\]|file)"(?:;\s*filename="([^"]*)")?/g;
  let m;
  while ((m = re.exec(text))) if (m[1]) names.push(m[1].split(/[\\/]/).pop());
  const sess = /name="session"\r\n\r\n([^\r\n]*)/.exec(text);
  return { names, session: sess ? sess[1].trim() : null };
}

/** A believable prediction + Explanation payload (docs/ps3_contract.md section 2) for one file. */
function ps3Predict(task, name, index) {
  const seed = [...name].reduce((a, c) => a + c.charCodeAt(0), index * 17);
  const rnd = (k) => ((Math.sin(seed * 12.9898 + k * 78.233) * 43758.5453) % 1 + 1) % 1;
  if (task === "door") {
    const n = 6 + (seed % 5);
    const rows = [];
    const marks = [];
    for (let i = 0; i < n; i += 1) {
      const t0 = 11 + i * 37;
      const abnormal = rnd(i) > 0.68;
      rows.push({
        start_time: `2023-7-5-0-${Math.floor(t0 / 60)}-${t0 % 60}-664`,
        end_time: `2023-7-5-0-${Math.floor((t0 + 4) / 60)}-${(t0 + 4) % 60}-112`,
        prediction: abnormal ? "Abnormal resistance" : "Normal",
        confidence: Number((0.62 + 0.36 * rnd(i + 99)).toFixed(3)),
      });
      marks.push({ x: t0, x1: t0 + 4, label: `cycle ${i + 1}`, kind: "segment" });
    }
    const x = Array.from({ length: 220 }, (_, i) => 10 + i * 1.1);
    return {
      rows,
      explanation: {
        file_id: name,
        numbers: { n_segments: rows.length, n_abnormal: rows.filter((r) => /Abnormal/.test(r.prediction)).length, median_cycle_s: 4.12 },
        trace: { x, y: x.map((v, i) => 1.2 + 0.9 * Math.sin(i / 6) + 0.5 * rnd(i)), marks: marks.slice(0, 4), label: "motor current, A" },
        viewport: { car: 1, side: "L", health: rows.some((r) => /Abnormal/.test(r.prediction)) ? "crit" : "ok", component: "door_L1" },
      },
    };
  }
  if (task === "acv") {
    const cars = ["01", "02", "03", "04", "05", "06", "07", "08"];
    const ranked = cars.slice().sort((a, b) => rnd(+a) - rnd(+b));
    const x = Array.from({ length: 160 }, (_, i) => i);
    return {
      rows: [{ file_id: name, ranked_cars: ranked.join("|") }],
      explanation: {
        file_id: name,
        numbers: { indoor_minus_peer_c: -2.41, sustained_minutes: 74, compressor_duty: 0.81 },
        trace: { x, y: x.map((i) => -2.4 + 1.1 * Math.sin(i / 11) - 0.5 * rnd(i)), marks: [{ x: 96, label: "worst sustained excursion", kind: "peak" }], label: "indoor − fleet median, °C" },
        viewport: { car: Number(ranked[0]), side: null, health: "crit", component: `car${Number(ranked[0])}_ac1` },
      },
    };
  }
  if (task === "rail") {
    const label = ["Normal", "Side I", "Side II"][Math.floor(rnd(1) * 3) % 3];
    const car = 1 + (seed % 8);
    const x = Array.from({ length: 140 }, (_, i) => 2 + i * 0.14);
    return {
      rows: [{ file_id: name, prediction: label }],
      explanation: {
        file_id: name,
        numbers: { side_i_score: Number((0.3 + 0.6 * rnd(2)).toFixed(3)), side_ii_score: Number((0.3 + 0.6 * rnd(3)).toFixed(3)), speed_kmh: 48 + Math.round(20 * rnd(4)) },
        trace: { x, y: x.map((v) => 0.2 + Math.exp(-((v - 6.3) ** 2) / 1.6) + 0.15 * rnd(v)), marks: [{ x: 6.3, label: "6.3 cm peak", kind: "peak" }], label: "wavelength PSD, axlebox accelerations" },
        viewport: { car, side: label === "Side II" ? "II" : "I", health: label === "Normal" ? "ok" : "crit", component: `axlebox_c${car}_p${label === "Side II" ? 4 : 3}` },
      },
    };
  }
  const damage = Number((0.05 + 0.75 * rnd(5)).toFixed(4));
  const x = Array.from({ length: 200 }, (_, i) => i * 40);
  return {
    rows: [{ file_id: name, prediction: damage }],
    explanation: {
      file_id: name,
      numbers: { cumulative_damage: damage, n_cycles: 18422, max_range_mpa: 61.3 },
      trace: { x, y: x.map((i) => 30 * Math.sin(i / 260) + 12 * rnd(i)), marks: [{ x: 3200, label: "largest block", kind: "peak" }], label: "dynamic stress, MPa" },
      viewport: { car: 1, side: null, health: damage >= 0.5 ? "crit" : damage >= 0.25 ? "warn" : "ok", component: "car1_bogie1" },
    },
  };
}

function mockStreamFrames(task, name, rows, explanation) {
  if (task === "door") {
    const n = rows.length;
    return rows.map((r, i) => ({
      task: "door",
      file_id: "",
      step: i,
      n_steps: n,
      t: Number((i * 4.2).toFixed(2)),
      t_unit: "s",
      progress: Number(((i + 1) / n).toFixed(4)),
      rows: rows.slice(0, i + 1),
      numbers: { p_abnormal: /abnormal/i.test(String(r.prediction)) ? 0.92 : 0.08 },
      viewport: { car: 1, side: "L", health: /abnormal/i.test(String(r.prediction)) ? "crit" : "ok", component: "door_L1" },
      final: i === n - 1,
    }));
  }
  const n = 10;
  const tUnit = task === "acv" ? "h" : task === "rail" ? "s" : "sample";
  const tStep = task === "acv" ? 3.0 : task === "rail" ? 0.1 : 5000;
  return Array.from({ length: n }, (_, i) => {
    const isFinal = i === n - 1;
    let stepRows = isFinal ? rows : (task === "rail" ? [] : rows);
    let numbers = { ...explanation.numbers };
    if (task === "rail") {
      numbers.speed_kmh_so_far = 45 + i;
    } else if (task === "shm") {
      const finalVal = Number(rows[0]?.prediction || 0.5);
      numbers.damage_final = finalVal;
      numbers.damage_running = Number((finalVal * (i + 1) / n).toFixed(4));
      numbers.cycles_so_far = (i + 1) * 1800;
    }
    const vp = { ...explanation.viewport };
    if (task === "rail" && !isFinal) vp.health = "ok";
    return {
      task,
      file_id: name,
      step: i,
      n_steps: n,
      t: Number(((i + 1) * tStep).toFixed(2)),
      t_unit: tUnit,
      progress: Number(((i + 1) / n).toFixed(4)),
      rows: stepRows,
      numbers,
      viewport: vp,
      final: isFinal,
    };
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  const p = url.pathname;
  if (req.method === "OPTIONS") return send(res, 204, {});

  if (p === "/api/health") {
    return send(res, 200, {
      status: "ok",
      scores_loaded: true,
      n_trains: BASE_FRAME.trains.length,
      clock: { start: iso(CLOCK_START), end: iso(CLOCK_END) },
      mock: true,
    });
  }

  if (p === "/api/trains") {
    return send(
      res,
      200,
      BASE_FRAME.trains.map((t) => ({
        train_id: t.train_id,
        line: "NSL",
        cars: 6,
        instrumented: t.components.map((c) => ({
          car: c.car,
          subsystem: c.subsystem,
          component_id: c.component_id,
          run_id: `mock-${t.train_id}-${c.component_id}`,
        })),
        faults: [],
      })),
    );
  }

  let m = p.match(/^\/api\/train\/([^/]+)\/state$/);
  if (m) {
    const tsParam = url.searchParams.get("ts");
    const ms = tsParam ? clamp(Date.parse(tsParam)) : simMs;
    return send(res, 200, frameAt(Number.isFinite(ms) ? ms : simMs, decodeURIComponent(m[1])));
  }

  m = p.match(/^\/api\/train\/([^/]+)\/component\/([^/]+)\/(series|scores|cycle)$/);
  if (m) {
    const kind = m[3];
    const n = 120;
    const step = (6 * 3600 * 1000) / n;
    const pts = Array.from({ length: n }, (_, i) => {
      const t = simMs - (n - i) * step;
      const v = 1 + 0.4 * Math.sin(i / 7) + (i / n) * 1.5;
      return kind === "scores" ? [iso(t), Number(v.toFixed(3)), v > 2.4] : [iso(t), Number(v.toFixed(3))];
    });
    if (kind === "cycle") {
      const t = Array.from({ length: 60 }, (_, i) => i * 0.05);
      return send(res, 200, {
        t,
        pos: t.map((x) => Math.min(1, x / 2.4)),
        pos_ref: t.map((x) => Math.min(1, x / 2.2)),
        current: t.map((x) => 1.4 + 0.5 * Math.sin(x * 3)),
        pwm: t.map(() => 0.6),
      });
    }
    if (kind === "scores") {
      return send(res, 200, { model: "cusum_cycle_scalar", threshold: 2.4, points: pts, top_signals: [] });
    }
    return send(res, 200, { signal: url.searchParams.get("signal") || "mock", unit: "-", points: pts });
  }

  if (p === "/api/alerts") {
    const f = frameAt(simMs);
    const train = url.searchParams.get("train");
    return send(res, 200, train ? f.alerts.filter((a) => a.train_id === train) : f.alerts);
  }

  if (p === "/api/kpis") return send(res, 200, frameAt(simMs).kpis);

  // ------------------------------------------------------------ PS3 predict (nebulax/api/ps3.py)
  // Same route shapes and the same session semantics as the real router, with made-up rows:
  // the page can be built and screenshotted with no python process and no model artefacts.
  if (p === "/api/ps3/tasks") return send(res, 200, PS3_TASKS);

  m = p.match(/^\/api\/ps3\/([a-z]+)\/cv$/);
  if (m) {
    const t = PS3_TASKS.find((x) => x.name === m[1]);
    if (!t) return send(res, 404, { detail: `unknown PS3 task '${m[1]}'` });
    if (!t.cv) return send(res, 404, { detail: `no CV results for '${m[1]}' yet` });
    return send(res, 200, { ...t.cv, mock: true });
  }

  m = p.match(/^\/api\/ps3\/([a-z]+)\/predict$/);
  if (m && req.method === "POST") {
    const task = m[1];
    const meta = PS3_TASKS.find((x) => x.name === task);
    if (!meta) return send(res, 404, { detail: `unknown PS3 task '${task}'` });
    const body = await readRaw(req);
    const { names, session: posted } = parseMultipart(body);
    if (!names.length) return send(res, 400, { detail: "no files uploaded (send them as multipart `files`)" });
    if (names.length > meta.max_files_per_request) {
      return send(res, 413, {
        detail: `${names.length} files in one request; send at most ${meta.max_files_per_request} per request and pass the returned \`session\` back with the next batch`,
      });
    }
    const token = posted || `mock${(ps3Seq += 1).toString().padStart(4, "0")}${Math.random().toString(36).slice(2, 8)}`;
    let sess = ps3Sessions.get(token);
    if (posted && !sess) return send(res, 404, { detail: `unknown or expired session '${posted}'` });
    if (sess && sess.task !== task) {
      return send(res, 400, { detail: `session '${token}' belongs to task '${sess.task}', not '${task}'` });
    }
    if (!sess) {
      sess = { token, task, rows: [], files: [] };
      ps3Sessions.set(token, sess);
    }
    const errors = [];
    const explanations = [];
    for (const name of names) {
      const suffix = name.slice(name.lastIndexOf(".")).toLowerCase();
      if (!meta.accepts.includes(suffix)) {
        errors.push({ file: name, message: `${meta.label} accepts ${meta.accepts.join(" or ")}, not ${suffix}` });
        continue;
      }
      if (sess.files.includes(name)) {
        errors.push({ file: name, message: "already predicted in this session; delete the session to start over" });
        continue;
      }
      const { rows, explanation } = ps3Predict(task, name, sess.files.length);
      sess.rows.push(...rows);
      sess.files.push(name);
      explanations.push(explanation);
    }
    if (!sess.files.length) {
      return send(res, 400, {
        detail: "no file in this batch could be predicted: " + errors.map((e) => `${e.file}: ${e.message}`).join("; "),
      });
    }
    return send(res, 200, {
      session: token,
      task,
      output_filename: meta.output_filename,
      rows: sess.rows,
      explanations,
      csv_url: `/api/ps3/results/${token}.csv`,
      n_done: sess.files.length,
      files: sess.files,
      errors,
      expires_at: Date.now() / 1000 + 7200,
    });
  }

  m = p.match(/^\/api\/ps3\/([a-z]+)\/stream$/);
  if (m && req.method === "POST") {
    const task = m[1];
    const meta = PS3_TASKS.find((x) => x.name === task);
    if (!meta) return send(res, 404, { detail: `unknown PS3 task '${task}'` });
    const body = await readRaw(req);
    const { names, session: posted } = parseMultipart(body);
    if (names.length !== 1) {
      return send(res, 400, { detail: `stream expects exactly one file, got ${names.length}; send as multipart \`files\`` });
    }
    const name = names[0];
    const suffix = name.slice(name.lastIndexOf(".")).toLowerCase();
    if (!meta.accepts.includes(suffix)) {
      return send(res, 400, { detail: `no file in this batch could be predicted: ${name}: ${meta.label} accepts ${meta.accepts.join(" or ")}, not ${suffix}` });
    }
    const token = posted || `mock${(ps3Seq += 1).toString().padStart(4, "0")}${Math.random().toString(36).slice(2, 8)}`;
    let sess = ps3Sessions.get(token);
    if (posted && !sess) return send(res, 404, { detail: `unknown or expired session '${posted}'` });
    if (sess && sess.task !== task) {
      return send(res, 400, { detail: `session '${token}' belongs to task '${sess.task}', not '${task}'` });
    }
    if (!sess) {
      sess = { token, task, rows: [], files: [] };
      ps3Sessions.set(token, sess);
    }
    if (sess.files.includes(name)) {
      return send(res, 400, { detail: `no file in this batch could be predicted: ${name}: already predicted in this session; delete the session to start over` });
    }
    const { rows, explanation } = ps3Predict(task, name, sess.files.length);
    sess.rows.push(...rows);
    sess.files.push(name);
    const frames = mockStreamFrames(task, name, rows, explanation);
    return send(res, 200, {
      session: token,
      task,
      output_filename: meta.output_filename,
      rows: sess.rows,
      explanations: [explanation],
      csv_url: `/api/ps3/results/${token}.csv`,
      n_done: sess.files.length,
      files: sess.files,
      errors: [],
      frames,
      expires_at: Date.now() / 1000 + 7200,
    });
  }


  m = p.match(/^\/api\/ps3\/results\/([A-Za-z0-9_-]+)\.csv$/);
  if (m) {
    const sess = ps3Sessions.get(m[1]);
    if (!sess) return send(res, 404, { detail: `unknown or expired session '${m[1]}'` });
    if (!sess.rows.length) return send(res, 404, { detail: `session '${m[1]}' has no predictions yet` });
    const cols = PS3_COLUMNS[sess.task];
    const text = [cols.join(","), ...sess.rows.map((r) => cols.map((c) => String(r[c] ?? "")).join(","))].join("\n") + "\n";
    const meta = PS3_TASKS.find((x) => x.name === sess.task);
    res.writeHead(200, {
      "content-type": "text/csv; charset=utf-8",
      "content-disposition": `attachment; filename="${meta.output_filename}"`,
      "access-control-allow-origin": "*",
      "x-ps3-rows": String(sess.rows.length),
    });
    return res.end(text);
  }

  m = p.match(/^\/api\/ps3\/results\/([A-Za-z0-9_-]+)$/);
  if (m && req.method === "DELETE") {
    const sess = ps3Sessions.get(m[1]);
    if (!sess) return send(res, 404, { detail: `unknown or expired session '${m[1]}'` });
    ps3Sessions.delete(m[1]);
    return send(res, 200, { session: m[1], deleted: true, n_files: sess.files.length, n_rows: sess.rows.length });
  }

  if (p === "/api/bench/selection") {
    return send(res, 200, {
      manifest: { mock: true, generated_at: new Date().toISOString() },
      rows: [
        { dataset: "sim", subsystem: "door", model: "cusum_cycle_scalar", val_lift: 4.24, recall: 0.33, precision: null, fa: 0.1 },
        { dataset: "sim", subsystem: "pneumatic", model: "sparse_autoencoder", val_lift: null, recall: 1.0, precision: 0.19, fa: 0.1 },
        { dataset: "sim", subsystem: "bearing", model: "cusum_cycle_scalar", val_lift: 50.8, recall: 0.83, precision: null, fa: 0.1 },
        { dataset: "metropt3", subsystem: "pneumatic", model: "lgbm_residual", val_lift: null, recall: 1.0, precision: null, fa: 0.105 },
      ],
    });
  }

  if (p === "/api/sim/inject" && req.method === "POST") {
    const body = await readBody(req);
    const t0 = Date.now();
    if (!body.train_id || !body.component_id || !body.fault_type) {
      return send(res, 400, { detail: "train_id, component_id and fault_type are required" });
    }
    injectSeq += 1;
    const episodeId = `${body.train_id}-${body.component_id}-inj${injectSeq}`;
    overlay = {
      train_id: body.train_id,
      car: Number(body.car ?? 0),
      component_id: body.component_id,
      fault_type: body.fault_type,
      alert: {
        episode_id: episodeId,
        train_id: body.train_id,
        car: Number(body.car ?? 0),
        subsystem: body.component_id.startsWith("door")
          ? "door"
          : body.component_id.startsWith("axlebox")
            ? "bearing"
            : "pneumatic",
        component_id: body.component_id,
        t_start: iso(simMs),
        t_end: null,
        peak_score: 5.4,
        fault_type: body.fault_type,
        lead_to_failure_h: Number(body.severity_ramp_days ?? 7) * 24,
        advisory: null,
      },
    };
    // Pretend the re-simulation takes a moment, as the real route does.
    await new Promise((r) => setTimeout(r, 400));
    return send(res, 200, {
      run_id: `inj-${injectSeq}`,
      n_rows: 1880,
      episodes: [
        {
          episode_id: episodeId,
          t_start: iso(simMs),
          t_end: null,
          peak_score: 5.4,
          n_rows: 42,
          fault_type: body.fault_type,
          lead_to_failure_h: Number(body.severity_ramp_days ?? 7) * 24,
          matched: true,
        },
      ],
      elapsed_s: (Date.now() - t0) / 1000,
    });
  }

  if (p === "/api/sim/inject" && req.method === "DELETE") {
    overlay = null;
    return send(res, 200, { cleared: true });
  }

  m = p.match(/^\/api\/advisory\/(.+)$/);
  if (m && req.method === "POST") {
    return send(res, 200, {
      summary: "Mock advisory: the component drifts away from its fleet peers.",
      evidence: ["mock server - no model was called"],
      likely_component: decodeURIComponent(m[1]),
      likely_fault: "unknown",
      recommended_action: "Inspect at the next stabling.",
      urgency: "medium",
      confidence: 0.4,
      cbm_steps: {
        state_detection: "mock",
        health_assessment: "mock",
        prognostic_assessment: "mock",
        advisory: "mock",
      },
      source: "template",
      model: "template-v1",
    });
  }

  return send(res, 404, { detail: `no mock route for ${req.method} ${p}` });
});

// ---------------------------------------------------------------- WebSocket
const GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
const clients = new Set();

function encodeText(str) {
  const payload = Buffer.from(str, "utf8");
  const len = payload.length;
  let header;
  if (len < 126) {
    header = Buffer.alloc(2);
    header[1] = len;
  } else if (len < 65536) {
    header = Buffer.alloc(4);
    header[1] = 126;
    header.writeUInt16BE(len, 2);
  } else {
    header = Buffer.alloc(10);
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(len), 2);
  }
  header[0] = 0x81; // FIN + text
  return Buffer.concat([header, payload]);
}

/** Pull as many complete frames as `buf` holds; returns [messages, rest]. */
function decodeFrames(buf) {
  const msgs = [];
  let off = 0;
  for (;;) {
    if (buf.length - off < 2) break;
    const b0 = buf[off];
    const b1 = buf[off + 1];
    const opcode = b0 & 0x0f;
    const masked = (b1 & 0x80) !== 0;
    let len = b1 & 0x7f;
    let p = off + 2;
    if (len === 126) {
      if (buf.length < p + 2) break;
      len = buf.readUInt16BE(p);
      p += 2;
    } else if (len === 127) {
      if (buf.length < p + 8) break;
      len = Number(buf.readBigUInt64BE(p));
      p += 8;
    }
    let mask = null;
    if (masked) {
      if (buf.length < p + 4) break;
      mask = buf.subarray(p, p + 4);
      p += 4;
    }
    if (buf.length < p + len) break;
    const payload = Buffer.from(buf.subarray(p, p + len));
    if (mask) for (let i = 0; i < payload.length; i += 1) payload[i] ^= mask[i % 4];
    off = p + len;
    msgs.push({ opcode, payload });
  }
  return [msgs, buf.subarray(off)];
}

if (WS_ENABLED) {
  server.on("upgrade", (req, socket) => {
    const url = new URL(req.url, "http://localhost");
    const key = req.headers["sec-websocket-key"];
    if (url.pathname !== "/api/replay" || !key) {
      socket.write("HTTP/1.1 404 Not Found\r\n\r\n");
      socket.destroy();
      return;
    }
    const accept = crypto.createHash("sha1").update(key + GUID).digest("base64");
    socket.write(
      "HTTP/1.1 101 Switching Protocols\r\n" +
        "Upgrade: websocket\r\n" +
        "Connection: Upgrade\r\n" +
        `Sec-WebSocket-Accept: ${accept}\r\n\r\n`,
    );
    socket.setNoDelay(true);
    clients.add(socket);
    let buf = Buffer.alloc(0);

    socket.on("data", (chunk) => {
      buf = Buffer.concat([buf, chunk]);
      const [msgs, rest] = decodeFrames(buf);
      buf = rest;
      for (const { opcode, payload } of msgs) {
        if (opcode === 0x8) {
          socket.end();
          return;
        }
        if (opcode === 0x9) {
          const pong = encodeText("");
          pong[0] = 0x8a;
          socket.write(pong);
          continue;
        }
        if (opcode !== 0x1) continue;
        let cmd;
        try {
          cmd = JSON.parse(payload.toString("utf8"));
        } catch {
          continue;
        }
        if (cmd.cmd === "play") playing = true;
        else if (cmd.cmd === "pause") playing = false;
        else if (cmd.cmd === "seek" && cmd.ts) {
          const ms = Date.parse(cmd.ts);
          if (Number.isFinite(ms)) simMs = clamp(ms);
        } else if (cmd.cmd === "speed" && Number(cmd.speed) > 0) speed = Number(cmd.speed);
        lastWall = Date.now();
        socket.write(encodeText(JSON.stringify(frameAt(simMs))));
      }
    });

    const cleanup = () => clients.delete(socket);
    socket.on("close", cleanup);
    socket.on("error", cleanup);
    socket.write(encodeText(JSON.stringify(frameAt(simMs))));
  });

  // 10 Hz push, as the contract specifies.
  setInterval(() => {
    if (!clients.size) return;
    const payload = encodeText(JSON.stringify(frameAt(simMs)));
    for (const s of clients) {
      if (s.writable) s.write(payload);
      else clients.delete(s);
    }
  }, 100);
}

server.listen(PORT, "127.0.0.1", () => {
  process.stdout.write(
    `mock API on http://127.0.0.1:${PORT}  ws=${WS_ENABLED ? "on /api/replay" : "off"}  ` +
      `clock ${iso(simMs)} speed ${speed} ${playing ? "playing" : "paused"}\n`,
  );
});
