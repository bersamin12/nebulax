// Preflight for a picked file: what columns it carries and whether they are what the chosen
// model eats, shown in the confirmation dialog before anything is queued (FileConfirm.jsx).
//
// The rules are transcriptions of the server-side readers, so the dialog and the API agree:
//   door  nebulax/ps3/door_features.py  CANONICAL_COLUMNS + _REQUIRED (datetime, current, position)
//   rail  nebulax/ps3/rail_features.py  read_rail_csv: header + 129 columns (speed + 64 x 2)
//   shm   nebulax/ps3/shm_features.py   headerless, exactly one column of stress samples
//   acv   nebulax/api/ps3.py            inspect_acv_workbook via POST /api/ps3/acv/validate
// A CSV is judged from its first 64 kB (the header and the first samples); the server still
// re-checks everything when the file is uploaded, so a pass here is a preview, not a promise.
import { validateAcvWorkbook } from "../api.js";

const HEAD_BYTES = 64 * 1024;
const RAIL_COLUMNS = 129;

/** Door column aliases, as door_features.CANONICAL_COLUMNS (keys normalised by normKey). */
const DOOR_COLUMNS = {
  datetime: ["datetime", "time", "timestamp"],
  current: ["motorcurrentma", "motorcurrent", "current"],
  voltage: ["motorvoltage10mv", "motorvoltage", "voltage"],
  emf: ["motorelectrodynamicforce", "motorbackelectromotiveforce", "backemf", "emf"],
  open_time: ["dooropeningtime1s", "dooropeningtime01s", "dooropeningtime"],
  close_time: ["doorclosingtime1s", "doorclosingtime01s", "doorclosingtime"],
  close_cmd: ["closecommand", "closecmd"],
  open_cmd: ["opencommand", "opencmd"],
  dcsr: ["dcsr"],
  dcsl: ["dcsl"],
  dlsr: ["dlsr"],
  dlsl: ["dlsl"],
  door_opened: ["dooropened"],
  door_locked: ["doorlocked"],
  is_opening: ["doorisopening", "isopening", "opening"],
  is_closing: ["doorisclosing", "isclosing", "closing"],
  position: ["doorleafposition", "doorposition", "position"],
};
const DOOR_REQUIRED = ["datetime", "current", "position"];

/** What each model expects, in the words the dialog shows. */
export const INPUT_SPEC = {
  door: {
    file: "one motor-current stream (.csv) with a header row, 50 Hz samples",
    columns: "Datetime, Motor current, Door leaf position are required; voltage, EMF, commands and door switches are used when present",
    required: DOOR_REQUIRED,
  },
  acv: {
    file: "one .xlsx case workbook per train-day",
    columns: "a Time column plus the sensor fields of all 8 cars, headed 'Car NN - <field>'",
  },
  rail: {
    file: "one .csv per run, 1 s at 10 kHz with a header row",
    columns: "129 columns: Rotating speed, then vibration and shock for 64 axle-box positions",
  },
  shm: {
    file: "one headerless .csv per stress record",
    columns: "a single column of stress samples, no header line",
  },
};

/** door_features._norm_key: lower-case alphanumerics only. */
function normKey(name) {
  return String(name ?? "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function splitCsvLine(line) {
  return String(line ?? "")
    .replace(/\r$/, "")
    .split(",")
    .map((c) => c.trim().replace(/^"(.*)"$/, "$1"));
}

async function headLines(file, n = 3) {
  const text = await file.slice(0, HEAD_BYTES).text();
  return text.split("\n").slice(0, n);
}

const isNumber = (s) => s !== "" && Number.isFinite(Number(s));

function checkDoor(lines) {
  const header = splitCsvLine(lines[0] || "");
  const byKey = new Map(header.map((c) => [normKey(c), c]));
  const found = {};
  for (const [canon, aliases] of Object.entries(DOOR_COLUMNS)) {
    const alias = aliases.find((a) => byKey.has(a));
    if (alias) found[canon] = byKey.get(alias);
  }
  const missing = DOOR_REQUIRED.filter((c) => !(c in found));
  const problems = [];
  if (!header.length || header.every((c) => c === "")) problems.push("the file is empty");
  else if (header.every(isNumber)) problems.push("no header row: the first line is numbers");
  if (missing.length) problems.push(`missing required column${missing.length > 1 ? "s" : ""}: ${missing.join(", ")}`);
  return {
    columns: header,
    found,
    missing,
    problems,
    detail: `${Object.keys(found).length} of ${Object.keys(DOOR_COLUMNS).length} known fields recognised`,
  };
}

function checkRail(lines) {
  const header = splitCsvLine(lines[0] || "");
  const problems = [];
  if (!header.length || header.every((c) => c === "")) problems.push("the file is empty");
  else if (header.every(isNumber)) problems.push("no header row: the first line is numbers");
  if (header.length !== RAIL_COLUMNS) problems.push(`${header.length} columns, rail runs have ${RAIL_COLUMNS} (speed + 64 axle boxes x vibration, shock)`);
  const speed = header[0] || "";
  if (header.length === RAIL_COLUMNS && !/speed/i.test(speed)) problems.push(`first column is '${speed}', expected the rotating speed pulse`);
  return {
    columns: header,
    found: header.length === RAIL_COLUMNS ? { speed, boxes: "64 x (vibration, shock)" } : {},
    missing: [],
    problems,
    detail: `${header.length} columns${header.length === RAIL_COLUMNS ? "" : ""}`,
  };
}

function checkShm(lines) {
  const first = splitCsvLine(lines[0] || "");
  const second = splitCsvLine(lines[1] || "");
  const problems = [];
  if (!first.length || first.every((c) => c === "")) problems.push("the file is empty");
  else {
    if (first.length !== 1) problems.push(`${first.length} columns, expected exactly one column of stress samples`);
    if (!isNumber(first[0])) problems.push(`the first line is '${first[0].slice(0, 30)}', but SHM records have no header: line 1 is the first sample`);
    else if (second.length && second[0] !== "" && !isNumber(second[0])) problems.push("line 2 is not a number");
  }
  return {
    columns: [],
    found: problems.length ? {} : { samples: `starts ${first[0]}${isNumber(second[0]) ? `, ${second[0]}` : ""}` },
    missing: [],
    problems,
    detail: problems.length ? "" : "headerless single column",
  };
}

async function checkAcv(file, signal) {
  if (!/\.xlsx$/i.test(file.name)) {
    return { columns: [], found: {}, missing: [], problems: ["choose a .xlsx workbook"], detail: "" };
  }
  const report = await validateAcvWorkbook(file, { signal });
  const s = report.summary || {};
  const cars = Array.isArray(s.cars) ? s.cars : [];
  const fields = Array.isArray(s.recognized_fields) ? s.recognized_fields : [];
  return {
    columns: fields,
    found: {
      cars: cars.length ? `${cars.length} (${cars.join(", ")})` : "none",
      fields: fields.length ? `${fields.length} recognised` : "none",
    },
    missing: [],
    problems: report.valid ? [] : report.errors || ["the workbook did not pass the format check"],
    detail: `${cars.length} cars, ${fields.length} sensor fields, ${s.sampled_rows ?? 0} rows sampled`,
  };
}

/**
 * Check one picked File for `task`. Never throws: an unreadable file is reported as a problem.
 * @returns {{name:string,size:number,ok:boolean,columns:string[],found:object,missing:string[],problems:string[],detail:string}}
 */
export async function checkFile(task, file, { signal } = {}) {
  const base = { name: file.name, size: file.size };
  try {
    let r;
    if (task === "acv") r = await checkAcv(file, signal);
    else {
      const lines = await headLines(file);
      r = task === "door" ? checkDoor(lines) : task === "rail" ? checkRail(lines) : task === "shm" ? checkShm(lines) : { columns: [], found: {}, missing: [], problems: [], detail: "" };
    }
    return { ...base, ...r, ok: r.problems.length === 0 };
  } catch (err) {
    if (signal && signal.aborted) throw err;
    return { ...base, columns: [], found: {}, missing: [], problems: [String(err.message || err)], detail: "", ok: false };
  }
}

/** Check every picked file, in order; the ACV files each round-trip to the server. */
export async function checkFiles(task, files, { signal } = {}) {
  const out = [];
  for (const f of files) out.push(await checkFile(task, f, { signal }));
  return out;
}
