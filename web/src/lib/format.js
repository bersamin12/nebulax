// Formatting + palette helpers shared by the console shell.
// Light corporate-blue / teal palette, adapted for screen contrast.

export const C = {
  ground: "#f4f7f8",
  panel: "#ffffff",
  panel2: "#f8fafb",
  panel3: "#eef3f5",
  line: "#dce5e9",
  line2: "#bdcbd2",
  grid: "#eaf0f2",
  text: "#1d2633",
  text2: "#344454",
  dim: "#526273",
  dim2: "#64748b",
  accent: "#008f95",
  accentBright: "#006d73",
  accentBg: "#e4f4f3",
  violet: "#1f2374",
  violetBright: "#1f2374",
  ok: "#1f8059",
  warn: "#a66500",
  crit: "#c43d36",
  critBg: "#fff0ef",
};

export const MONO = "'IBM Plex Mono', 'SFMono-Regular', Menlo, Consolas, monospace";
export const SANS = "'IBM Plex Sans', 'Segoe UI', Tahoma, sans-serif";

// health values come straight from the replay frame (app_contract section 5).
export const HEALTH = {
  crit: { word: "ALERT", rank: 3, color: "#c43d36", bg: "#fff0ef", border: "#e7aca8", ink: "#9d2822" },
  warn: { word: "WATCH", rank: 2, color: "#a66500", bg: "#fff5df", border: "#dfc18d", ink: "#835100" },
  ok: { word: "OK", rank: 1, color: "#1f8059", bg: "#eaf7f0", border: "#a9d4bb", ink: "#176646" },
  nodata: { word: "NO DATA", rank: 0, color: "#64748b", bg: "#f4f7f8", border: "#dce5e9", ink: "#526273" },
};
export const health = (h) => HEALTH[h] || HEALTH.nodata;

export const SUBSYSTEMS = ["door", "pneumatic", "bearing"];

// ---------------------------------------------------------------- numbers

// Anomaly scores and their thresholds span eleven decades in this fleet: a CUSUM statistic
// runs from 0 to ~1e11 while the sparse autoencoder sits at 1e-3. Anything that will not fit a
// console cell in full is written as a compact mantissa+exponent ("2.1e10"), never as 12 digits.
export function fmtScore(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1e5 || (a > 0 && a < 1e-3)) {
    const e = Math.floor(Math.log10(a));
    const m = v / 10 ** e;
    return `${m.toFixed(1)}e${e}`;
  }
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100) return v.toFixed(1);
  if (a >= 10) return v.toFixed(2);
  return v.toFixed(3).replace(/0$/, "");
}

/** A threshold is read next to its score, so it gets the same compact treatment. */
export const fmtThreshold = fmtScore;

export function fmtNum(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

export function fmtZ(z) {
  if (z === null || z === undefined || Number.isNaN(z)) return "—";
  return (z >= 0 ? "+" : "−") + Math.abs(z).toFixed(1);
}

// hours -> "18 h" / "4.3 d"
export function fmtHours(h) {
  if (h === null || h === undefined || Number.isNaN(h)) return "—";
  if (Math.abs(h) >= 48) return (h / 24).toFixed(1) + " d";
  return h.toFixed(h < 10 ? 1 : 0) + " h";
}

// ---------------------------------------------------------------- time

const MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
const p2 = (n) => String(n).padStart(2, "0");

export const toMs = (v) => {
  if (v === null || v === undefined) return NaN;
  if (typeof v === "number") return v;
  const t = Date.parse(v);
  return Number.isNaN(t) ? NaN : t;
};
export const toIso = (msv) =>
  Number.isNaN(msv) ? "" : new Date(msv).toISOString().replace(/\.\d{3}Z$/, "Z");

export function fmtDay(v) {
  const t = toMs(v);
  if (Number.isNaN(t)) return "—";
  const d = new Date(t);
  return `${p2(d.getUTCDate())} ${MON[d.getUTCMonth()]}`;
}

export function fmtHm(v) {
  const t = toMs(v);
  if (Number.isNaN(t)) return "—";
  const d = new Date(t);
  return `${p2(d.getUTCHours())}:${p2(d.getUTCMinutes())}`;
}

export function fmtDayHm(v) {
  const t = toMs(v);
  if (Number.isNaN(t)) return "—";
  return `${fmtDay(t)} ${fmtHm(t)}`;
}

// "14 SEP 2026  08:30:00"
export function fmtStamp(v) {
  const t = toMs(v);
  if (Number.isNaN(t)) return "—— ——— ————  --:--:--";
  const d = new Date(t);
  return (
    `${p2(d.getUTCDate())} ${MON[d.getUTCMonth()]} ${d.getUTCFullYear()}` +
    `  ${p2(d.getUTCHours())}:${p2(d.getUTCMinutes())}:${p2(d.getUTCSeconds())}`
  );
}

export const HOUR = 3600 * 1000;
export const DAY = 24 * HOUR;

// ---------------------------------------------------------------- components

// The sim carries the APU at car=0 (unit level); the 6-car schematic draws it under car 3.
export function displayCar(c) {
  if (!c) return null;
  if (c.subsystem === "pneumatic" && (c.car === 0 || c.car === null || c.car === undefined)) return 3;
  return c.car;
}

// door_L1 -> "L1", axlebox_3R -> "3R", apu_1 -> "APU"
export function shortComponent(cid) {
  if (!cid) return "—";
  let m = /^door_([LR])(\d)$/.exec(cid);
  if (m) return m[1] + m[2];
  m = /^axlebox_(\d)([LR])$/.exec(cid);
  if (m) return m[1] + m[2];
  if (/^apu_/.test(cid)) return "APU";
  return cid.toUpperCase();
}

export function componentLabel(cid) {
  if (!cid) return "—";
  let m = /^door_([LR])(\d)$/.exec(cid);
  if (m) return `DOOR ${m[1]}${m[2]}`;
  m = /^axlebox_(\d)([LR])$/.exec(cid);
  if (m) return `AXLE BOX ${m[1]}${m[2]}`;
  m = /^apu_(\d+)$/.exec(cid);
  if (m) return `APU ${m[1]}`;
  return cid.toUpperCase().replace(/_/g, " ");
}

export function componentKind(c) {
  if (!c) return "";
  const cid = c.component_id || "";
  if (c.subsystem === "door") {
    const m = /^door_([LR])/.exec(cid);
    const side = m ? (m[1] === "L" ? "NEAR SIDE" : "FAR SIDE") : "";
    return ["ELECTRIC SLIDING DOOR", side].filter(Boolean).join(" · ");
  }
  if (c.subsystem === "pneumatic") return "AIR PRODUCTION UNIT · UNIT LEVEL";
  if (c.subsystem === "bearing") {
    const m = /^axlebox_(\d)/.exec(cid);
    const bogie = m ? `BOGIE ${Number(m[1]) <= 2 ? 1 : 2}` : "";
    return ["AXLE BOX BEARING", bogie].filter(Boolean).join(" · ");
  }
  return c.subsystem ? String(c.subsystem).toUpperCase() : "";
}

// score / threshold, the only cross-model comparable severity we have
export function severity(c) {
  if (!c || c.score === null || c.score === undefined) return -Infinity;
  if (c.threshold) return c.score / c.threshold;
  return c.score;
}

export function worstComponent(list) {
  let best = null;
  for (const c of list || []) {
    if (!best) { best = c; continue; }
    const hr = health(c.health).rank - health(best.health).rank;
    if (hr > 0 || (hr === 0 && severity(c) > severity(best))) best = c;
  }
  return best;
}

export function worstHealth(list) {
  const w = worstComponent(list);
  return w ? w.health || "ok" : "nodata";
}

export function countHealth(list) {
  const n = { crit: 0, warn: 0, ok: 0, nodata: 0 };
  for (const c of list || []) n[c.health in n ? c.health : "nodata"] += 1;
  return n;
}

export function trainOf(frame, trainId) {
  const trains = (frame && frame.trains) || [];
  return trains.find((t) => t.train_id === trainId) || trains[0] || null;
}

export function findComponent(train, sel) {
  if (!train || !sel) return null;
  return (
    (train.components || []).find(
      (c) => c.component_id === sel.component_id && (sel.car == null || c.car === sel.car)
    ) || null
  );
}

// the open (or latest) episode for a selection
export function findAlert(frame, sel) {
  if (!frame || !sel) return null;
  const hits = (frame.alerts || []).filter(
    (a) => a.train_id === sel.train_id && a.component_id === sel.component_id
  );
  if (!hits.length) return null;
  const open = hits.filter((a) => !a.t_end);
  const pool = open.length ? open : hits;
  return pool.slice().sort((a, b) => toMs(b.t_start) - toMs(a.t_start))[0];
}

// per-subsystem default detail signal (w3_web_shell brief)
export const DEFAULT_SIGNAL = {
  door: { signal: "current", label: "MOTOR CURRENT", unit: "A", span: null },
  pneumatic: { signal: "TP2", label: "TP2 COMPRESSOR PRESSURE", unit: "bar", span: 6 * HOUR },
  bearing: { signal: "T_box", label: "AXLE-BOX TEMPERATURE", unit: "°C", span: 12 * HOUR },
};
