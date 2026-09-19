// Static copy for the four PS3 subsystems: what the task is, what it eats, and the sentence the
// organisers' Info Kit uses for the score. Everything *dynamic* (accepted suffixes, the CV score,
// whether the task can run at all) comes from GET /api/ps3/tasks - this file only holds prose.
//
// The metric names are the ones in docs/ps3_contract.md section 5 (`nebulax/ps3/scoring.py`),
// which are exact transcriptions of the Info Kits.

export const TASK_ORDER = ["door", "acv", "rail", "shm"];
// Match the server contract while GET /api/ps3/tasks is still loading. A user can pick a file
// before that request finishes, especially after a page reload.
export const TASK_SUFFIXES = { door: [".csv"], acv: [".xlsx", ".xls"], rail: [".csv"], shm: [".csv"] };
export const INFO_ORDER = ["pneumatic", "bearing"];
export const SYSTEM_ORDER = [...TASK_ORDER, ...INFO_ORDER];

// Research systems have no PS3 upload or submission output. Keep them separate from TASK_ORDER
// and /api/ps3/tasks so a dataset profile can never be mistaken for a runnable predictor.
export const INFO_META = {
  pneumatic: {
    tab: "Brake air supply",
    title: "Brake air supply research",
    blurb: "Air-compressor condition monitoring using MetroPT-3 and a calibrated fleet simulation.",
    dataset: "MetroPT-3 (Air Compressor)",
    source: "UCI Machine Learning Repository",
    sourceUrl: "https://doi.org/10.24432/C5VW3R",
    size: "1,516,948 timestamped records",
    signals: "7 analogue and 8 digital compressor channels",
    evaluation: "Temporal anomaly-detection benchmark; validation selection deferred",
    model: "LGBM residual benchmark and simulator calibration",
    location: "Air production unit beneath Car 3",
  },
  bearing: {
    tab: "Axle bearing",
    title: "Axle bearing research",
    blurb: "Bearing-condition research using vibration recordings from the Ottawa rolling-element dataset.",
    dataset: "UORED-VAFCLS",
    source: "University of Ottawa / Mendeley Data",
    sourceUrl: "https://doi.org/10.17632/y2px5tg92h.5",
    size: "60 recordings from 20 bearings",
    signals: "5 channels sampled at 42 kHz for 10 seconds per recording",
    evaluation: "Bearing-grouped anomaly-detection benchmark",
    model: "PCA SPE/T² benchmark",
    location: "Axle box beneath Car 1",
  },
};

export const TASK_META = {
  door: {
    tab: "Door",
    title: "Door cycle classification",
    blurb: "Identify each door cycle in the motor-current stream and classify it as Normal or Abnormal resistance.",
    input: "one Test.csv stream (datetime, motor current, …)",
    scored: "IoU-weighted F1 over the predicted segments, same-label matches only.",
    metric: "IoU-weighted F1",
    multiple: false,
    directory: false,
  },
  acv: {
    tab: "ACV",
    title: "Refrigerant-leak car ranking",
    blurb: "Rank the cars in each workbook from most to least likely to have a refrigerant leak.",
    input: "one .xlsx workbook per case",
    scored: "Rank decay (n − (r − 1)) / n on the true car, averaged over cases.",
    metric: "rank decay",
    multiple: true,
    directory: false,
  },
  rail: {
    tab: "Rail corrugation",
    title: "Rail corrugation classification",
    blurb: "Classify each run as Normal, Side I, or Side II using the 64 axle-box accelerometers.",
    input: "one .csv per run (speed + 64 boxes × 2 axes); drop the whole folder",
    scored: "Macro F1 over the fixed three labels; an absent class scores F1 = 0.",
    metric: "macro F1",
    multiple: true,
    directory: true,
  },
  shm: {
    tab: "SHM",
    title: "Fatigue damage prediction",
    blurb: "Estimate cumulative fatigue damage from a dynamic stress record.",
    input: "one headerless single-column .csv per record; drop the whole folder",
    scored: "score = max(0, 1 − MAPE) on the cumulative-damage number.",
    metric: "MAPE score",
    multiple: true,
    directory: true,
  },
};

/** Organiser CSV columns per task (`nebulax.ps3.submission.CSV_HEADERS`; order is the contract). */
export const CSV_COLUMNS = {
  door: ["start_time", "end_time", "prediction"],
  acv: ["file_id", "ranked_cars"],
  rail: ["file_id", "prediction"],
  shm: ["file_id", "prediction"],
};

/** Door's optional trailing column: shown only when every row carries it, as the API writes it. */
export const DOOR_EXTRA = "confidence";

/** Column headings for the results table (the CSV names, spelled for a human). */
export const COLUMN_LABELS = {
  start_time: "segment start",
  end_time: "segment end",
  prediction: "prediction",
  file_id: "file",
  ranked_cars: "ranked cars (most → least likely)",
  confidence: "conf.",
};

/**
 * The columns to draw for a task, plus door's `confidence` when the API attached one to every
 * row (it renders that column in the CSV under the same condition).
 */
export function columnsFor(task, rows) {
  const cols = [...(CSV_COLUMNS[task] || [])];
  if (task === "door" && rows.length && rows.every((r) => r && r[DOOR_EXTRA] !== undefined)) {
    cols.push(DOOR_EXTRA);
  }
  return cols;
}

/**
 * The CV headline of one task entry of `/api/ps3/tasks`, as `{label, value}` or null.
 *
 * `cv_summary` keeps the scalar entries of the json: some tasks publish a top-level `score`
 * (acv), others only the per-metric mean inside `headline` (door: `iou_f1_mean`). Nothing is
 * invented here - when neither is present the tab shows the scheme alone.
 */
export function cvScore(cv) {
  if (!cv) return null;
  const metric = String(cv.metric || "").replace(/\s*\(.*\)\s*$/, "").trim();
  if (Number.isFinite(cv.score)) return { label: metric || "score", value: cv.score };
  for (const box of [cv.headline, cv.summary, cv.winner]) {
    if (!box || typeof box !== "object") continue;
    const named = [`${metric}_mean`, metric, `${metric}_score`, "score_mean", "score"];
    for (const k of named) if (k && Number.isFinite(box[k])) return { label: metric || k, value: box[k] };
    for (const [k, v] of Object.entries(box)) {
      if (/_mean$/.test(k) && Number.isFinite(v)) return { label: k.replace(/_mean$/, ""), value: v };
    }
  }
  return null;
}

/** "3 files · 41.2 MB" style byte size. */
export function fmtBytes(n) {
  const b = Number(n);
  if (!Number.isFinite(b)) return "n/a";
  if (b >= 1024 * 1024 * 1024) return `${(b / 1024 ** 3).toFixed(1)} GB`;
  if (b >= 1024 * 1024) return `${(b / 1024 ** 2).toFixed(1)} MB`;
  if (b >= 1024) return `${(b / 1024).toFixed(0)} kB`;
  return `${b} B`;
}

/** Health word for one prediction cell, so the table and the 3D twin agree on the colour. */
export function rowHealth(task, row) {
  if (!row) return "nodata";
  if (task === "door") return /abnormal/i.test(String(row.prediction ?? "")) ? "crit" : "ok";
  if (task === "rail") return /^\s*normal\s*$/i.test(String(row.prediction ?? "")) ? "ok" : "crit";
  if (task === "shm") {
    const d = Number(row.prediction);
    if (!Number.isFinite(d)) return "nodata";
    return d >= 0.5 ? "crit" : d >= 0.25 ? "warn" : "ok";
  }
  if (task === "acv") return String(row.ranked_cars || "").trim() ? "crit" : "nodata";
  return "nodata";
}
