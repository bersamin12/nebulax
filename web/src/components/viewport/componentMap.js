// Mapping between app-contract component ids and the node names inside web/public/r151.glb,
// plus the PS3 health-map helpers (ps3HealthMap) the predict page feeds to <TrainViewport>.
//
// glb node inventory (dumped with three's GLTFLoader after `build_r151.py --n-cars 8`; 1530 nodes /
// 1169 meshes, scene units = metres, train along +X, up +Y, the "L" side of the car faces +Z,
// cars 1..8 from x = -1.16 to x = 190.17, rails from x = -5 to 194.01, roof top y = 3.985):
//
//   car{n}_body                       Group   -> meshes car{n}_body_0 .. _6      (body_white / roof_grey /
//                                                 window_band / accent_stripe / gangway)
//   car{n}_ac1, car{n}_ac2            Mesh    (roof air-conditioning units, ac_default) - NEW, 8 cars x 2
//   car{n}_door_{L|R}{k}              Object3D at the door centre (k = 1..4)
//     car{n}_door_{L|R}{k}_leafA      Group   -> meshes ..._leafA_1 (door_default panel),
//                                                 ..._leafA_2 (glass_dark), ..._leafA_3 (sensor_edge)
//     car{n}_door_{L|R}{k}_leafB      Group   -> meshes ..._leafB_1 .. _3   (mirror leaf)
//     car{n}_door_{L|R}{k}_drive      Group   -> ..._drive_1 .. _2
//     car{n}_door_{L|R}{k}_pulley     Group   -> ..._pulley_1 .. _2
//     car{n}_door_{L|R}{k}_lsOpen     Mesh    (limit switch lamp)
//     car{n}_door_{L|R}{k}_lsClosed   Mesh
//   car{n}_bogie{1,2}                 Group   -> car{n}_bogie{b}_1 .. _5 (frame, springs, motors)
//   car{n}_wheel_{a}{L|R}             Mesh    (a = 1..4)
//   car{n}_brake_{a}{L|R}             Mesh
//   car{n}_axlebox_{a}{L|R}           Mesh    (a = 1..4; bogie 1 = axles 1,2, bogie 2 = axles 3,4)
//   car{n}_apu                        Object3D
//     car{n}_apu_block                Group   -> car{n}_apu_block_1 (apu_default), _2 (undercarriage)
//     car{n}_apu_fan / _pipes / _tank / _tower1 / _tower2   Meshes
//   rail, rail001                     Meshes  (the two running rails: `rail` is the L / near side at
//                                              z = +0.7175, `rail001` the R / far side at z = -0.7175)
//
// The number of cars is never assumed here: every regex is car(\d+) and the counts shown in the
// viewport header are computed from the mesh index (modelStats).
//
// NOTE: the vanilla template `web/twin_views.template.html` calls setMat("car4_door_L3_leafA", ...),
// but `..._leafA` is a Group, not a Mesh, so that tint was a no-op there. We tint the leaf's child
// meshes instead and move the Group (which is what the template's position code did).
//
// Contract ids (docs/app_contract.md sections 1 and 5) -> nodes:
//   door      car c, `door_L3`      -> car{c}_door_L3_leaf{A,B}_{1,3}   (leaf panel + sensor edge)
//   bearing   car c, `axlebox_2R`   -> car{c}_axlebox_2R
//   pneumatic car c, `apu_1`        -> car{c}_apu_block_1 / _tank / _tower1 / _tower2 / _fan,
//             with car 0 (the sim's unit-level APU) mapped to car 3 as the contract's schematic
//             says; the other cars' APUs exist on the model and render as `nodata`
// PS3-only ids (tintable by a healthMap, never selectable from a replay frame):
//   acv       car c, `ac_1`         -> car{c}_ac1 + car{c}_ac2   (one air-con unit per car)
//   shm       car c, `bogie_1|2`    -> car{c}_bogie{b}_1 .. _5
//   rail      car 0, `rail_I|II`    -> rail (Side I = L / near) and rail001 (Side II = R / far)
//
// Rail-corrugation axle boxes: the organisers number 8 positions per car, odd = Side I, even =
// Side II. Position p maps to axle ceil(p/2) and side L (odd) / R (even): railBoxComponent().

export const APU_CAR = 3; // car = 0 (unit level) is drawn on car 3

/** Subsystems a replay frame can carry; only these are pickable in the twin. */
export const TWIN_SUBSYSTEMS = new Set(["bearing", "pneumatic"]);

/** Canonical key for a frame component / a picked mesh. */
export function componentKey(car, subsystem, componentId) {
  // the sim reports the APU at unit level with car = 0; the schematic draws it under car 3
  const c = subsystem === "pneumatic" && (car === 0 || car == null) ? APU_CAR : car;
  return `${c}|${subsystem}|${componentId}`;
}

const DOOR_RE = /^car(\d+)_door_([LR][1-4])_(leafA|leafB)_([13])$/;
const AXLEBOX_RE = /^car(\d+)_axlebox_([1-4][LR])$/;
const APU_RE = /^car(\d+)_apu_(block_1|tank|tower1|tower2|fan)$/;
export const AC_RE = /^car(\d+)_ac([12])$/;
const BOGIE_RE = /^car(\d+)_bogie([12])_\d+$/;
const RAIL_RE = /^rail(001)?$/;
export const CAR_RE = /^car(\d+)_/;

/**
 * Which contract component (if any) a glb mesh belongs to.
 * Returns {car, subsystem, component_id} or null.
 */
export function componentForMesh(name) {
  let m = DOOR_RE.exec(name);
  if (m) return { car: +m[1], subsystem: "door", component_id: `door_${m[2]}` };
  m = AXLEBOX_RE.exec(name);
  if (m) return { car: +m[1], subsystem: "bearing", component_id: `axlebox_${m[2]}` };
  m = APU_RE.exec(name);
  if (m) return { car: +m[1], subsystem: "pneumatic", component_id: "apu_1" };
  m = AC_RE.exec(name);
  if (m) return { car: +m[1], subsystem: "acv", component_id: "ac_1" };
  m = BOGIE_RE.exec(name);
  if (m) return { car: +m[1], subsystem: "shm", component_id: `bogie_${m[2]}` };
  m = RAIL_RE.exec(name);
  if (m) return { car: 0, subsystem: "rail", component_id: m[1] ? "rail_II" : "rail_I" };
  return null;
}

/** The leaf Groups of a door component id, used for the open/close animation. */
export function doorLeafGroups(car, componentId) {
  const p = `car${car}_${componentId}`; // door_L3 -> car4_door_L3
  return [`${p}_leafA`, `${p}_leafB`];
}

/** Every mesh that should be tinted for one contract component. */
export function meshesForComponent(car, subsystem, componentId) {
  if (subsystem === "door") {
    const p = `car${car}_${componentId}`;
    return [`${p}_leafA_1`, `${p}_leafA_3`, `${p}_leafB_1`, `${p}_leafB_3`];
  }
  if (subsystem === "bearing") return [`car${car}_${componentId}`];
  if (subsystem === "pneumatic") {
    const c = car === 0 || car == null ? APU_CAR : car;
    return [`car${c}_apu_block_1`, `car${c}_apu_tank`, `car${c}_apu_tower1`, `car${c}_apu_tower2`, `car${c}_apu_fan`];
  }
  if (subsystem === "acv") return [`car${car}_ac1`, `car${car}_ac2`];
  if (subsystem === "shm") {
    const b = componentId.endsWith("2") ? 2 : 1;
    return [1, 2, 3, 4, 5].map((i) => `car${car}_bogie${b}_${i}`);
  }
  if (subsystem === "rail") return [componentId === "rail_II" ? "rail001" : "rail"];
  return [];
}

/** `door_L3` -> `L`, `axlebox_2R` -> `R`, `rail_II` -> `R`, apu / ac / bogie -> `L` (near side). */
export function sideOf(subsystem, componentId) {
  if (subsystem === "door") return componentId[5] === "R" ? "R" : "L";
  if (subsystem === "bearing") return componentId.endsWith("R") ? "R" : "L";
  if (subsystem === "rail") return componentId === "rail_II" ? "R" : "L";
  return "L";
}

/** Rail-corrugation axle-box position (1..8, odd = Side I) -> the bearing component of the model. */
export function railBoxComponent(car, position) {
  const p = Math.max(1, Math.min(8, Math.round(+position)));
  const axle = Math.ceil(p / 2);
  const side = p % 2 ? "L" : "R";
  return { car: +car, subsystem: "bearing", component_id: `axlebox_${axle}${side}` };
}

/** Index the components of one train of a replay frame by componentKey(). */
export function indexFrame(train) {
  const byKey = new Map();
  for (const c of train?.components ?? []) {
    byKey.set(componentKey(c.car, c.subsystem, c.component_id), c);
  }
  return byKey;
}

export const HEALTHS = ["ok", "warn", "crit", "nodata"];

function normHealth(h) {
  return HEALTHS.includes(h) ? h : "nodata";
}

/**
 * Index a healthMap prop (see the TrainViewport header comment) by componentKey():
 * `components` entries as given, `railHealth` {I, II} -> `0|rail|rail_I` / `0|rail|rail_II`,
 * `acHealth` {car: health} -> `${car}|acv|ac_1`. Every value becomes {health, label?, ...extras}.
 */
export function indexHealthMap(healthMap) {
  const byKey = new Map();
  if (!healthMap) return byKey;
  for (const [key, v] of Object.entries(healthMap.components ?? {})) {
    const entry = typeof v === "string" ? { health: v } : { ...v };
    entry.health = normHealth(entry.health);
    byKey.set(key, entry);
  }
  for (const [side, h] of Object.entries(healthMap.railHealth ?? {})) {
    const s = String(side).toUpperCase() === "II" ? "II" : "I";
    byKey.set(componentKey(0, "rail", `rail_${s}`), { health: normHealth(h), label: `Side ${s} rail` });
  }
  for (const [car, h] of Object.entries(healthMap.acHealth ?? {})) {
    const c = parseInt(car, 10);
    if (!Number.isFinite(c)) continue;
    const v = typeof h === "string" ? { health: h } : { ...h };
    v.health = normHealth(v.health);
    byKey.set(componentKey(c, "acv", "ac_1"), v);
  }
  return byKey;
}

/* ------------------------------------------------------------------ PS3 predictions -> map */

const PS3_CAPTION = {
  door: "DOOR SEGMENTS",
  rail: "RAIL CORRUGATION",
  acv: "ACV REFRIGERANT LEAK",
  shm: "SHM FATIGUE DAMAGE",
};

/** Damage -> health for the SHM task (warn from 0.25, crit from 0.5). */
export function shmHealth(damage) {
  const d = Number(damage);
  if (!Number.isFinite(d)) return "nodata";
  return d >= 0.5 ? "crit" : d >= 0.25 ? "warn" : "ok";
}

function railSide(prediction) {
  const p = String(prediction ?? "").trim().toLowerCase().replace(/\s+/g, " ");
  if (p === "side ii" || p === "side 2" || p === "ii") return "II";
  if (p === "side i" || p === "side 1" || p === "i") return "I";
  return null;
}

/**
 * Ranked box list of a rail explanation, in any of the shapes the API may use: a list under
 * `top_boxes | boxes | viewport.top_boxes | viewport.boxes`, else the single
 * `viewport.component` "axlebox_c{car}_p{position}" of docs/ps3_contract.md section 2.
 */
function railBoxes(explanation) {
  if (!explanation) return [];
  const raw =
    explanation.top_boxes ??
    explanation.boxes ??
    explanation.viewport?.top_boxes ??
    explanation.viewport?.boxes ??
    null;
  const out = [];
  const single = /^axlebox_c(\d+)_p(\d+)$/.exec(String(explanation.viewport?.component ?? ""));
  if (single && !Array.isArray(raw)) out.push({ car: +single[1], position: +single[2] });
  if (Array.isArray(raw)) {
    for (const b of raw) {
      if (Array.isArray(b)) out.push({ car: +b[0], position: +b[1] });
      else if (b && typeof b === "object") out.push({ car: +b.car, position: +(b.position ?? b.pos) });
      else if (typeof b === "string") {
        const m = /car\s*(\d+)\D+(\d+)/i.exec(b);
        if (m) out.push({ car: +m[1], position: +m[2] });
      }
    }
  }
  return out.filter((b) => Number.isFinite(b.car) && Number.isFinite(b.position));
}

/**
 * Turn the rows (+ optional explanations) of one PS3 prediction into a TrainViewport healthMap.
 *
 *   ps3HealthMap({ task, rows, explanations, fileId })
 *     task          "door" | "rail" | "acv" | "shm"
 *     rows          the prediction CSV rows as objects (door: {start_time, end_time, prediction};
 *                   rail / shm: {file_id, prediction}; acv: {file_id, ranked_cars})
 *     explanations  optional, the API's explanations[] ({file_id, numbers, trace, viewport, ...})
 *     fileId        optional; rail / acv / shm summarise ONE row: the row with this file_id, else
 *                   rows[0]. Door uses every row (the stream is one door).
 *
 *   door  car 1 `door_L1` (or explanation.viewport.car): crit if any row is "Abnormal resistance", else ok
 *   rail  rail meshes by side (predicted side crit, other side ok) + all 64 axle boxes ok, with the
 *         top-3 contributing boxes crit. Boxes come from the explanation as
 *         `top_boxes | boxes | viewport.top_boxes | viewport.boxes`: [{car, position}] (1-based,
 *         position 1..8, odd = Side I), [[car, position]] or "car3 pos5" strings, already ranked.
 *         A single `viewport.component` "axlebox_c3_p4" (docs/ps3_contract.md) is one box; without
 *         any box, `viewport: {car, side, health}` marks the four boxes of that car on that side.
 *   acv   per-car air-con unit: rank 1 crit, ranks 2-3 warn, the rest ok (ids "03" -> car 3)
 *   shm   car 1 (or explanation.viewport.car) bogies 1 and 2: warn if damage >= 0.25, crit if >= 0.5, else ok
 *
 * Returns { task, caption, components: {key: {health, label}}, railHealth: {I, II}, acHealth: {car: health} }
 */
export function ps3HealthMap({ task, rows = [], explanations = [], fileId = null } = {}) {
  const components = {};
  const railHealth = {};
  const acHealth = {};
  const set = (car, subsystem, id, health, label) => {
    components[componentKey(car, subsystem, id)] = { health, label };
  };
  const list = Array.isArray(rows) ? rows : [];
  const row = fileId ? (list.find((r) => String(r?.file_id) === String(fileId)) ?? null) : (list[0] ?? null);
  const expl =
    (row && (explanations ?? []).find((e) => String(e?.file_id) === String(row.file_id))) ??
    (fileId ? null : (explanations?.[0] ?? null));
  let caption = PS3_CAPTION[task] ?? String(task ?? "PS3").toUpperCase();
  if (row?.file_id) caption += ` · ${row.file_id}`;

  if (task === "door") {
    const abnormal = list.filter((r) => /abnormal/i.test(String(r?.prediction ?? ""))).length;
    const health = !list.length ? "nodata" : abnormal ? "crit" : "ok";
    const doorCar = Number(expl?.viewport?.car) || 1; // the stream is one door: car 1 unless told
    const label = !list.length
      ? "no segments"
      : abnormal
        ? `${abnormal} abnormal of ${list.length} segments`
        : `${list.length} normal segments`;
    set(doorCar, "door", "door_L1", health, label);
    caption = `${PS3_CAPTION.door} · ${list.length} segments`;
  } else if (task === "rail") {
    const side = railSide(row?.prediction);
    if (row) {
      railHealth.I = side === "I" ? "crit" : "ok";
      railHealth.II = side === "II" ? "crit" : "ok";
    }
    const cars = 8;
    if (row) {
      for (let c = 1; c <= cars; c++) {
        for (let p = 1; p <= 8; p++) {
          const b = railBoxComponent(c, p);
          set(b.car, b.subsystem, b.component_id, "ok", `car ${c} position ${p} · Side ${p % 2 ? "I" : "II"}`);
        }
      }
    }
    let boxes = railBoxes(expl).slice(0, 3);
    if (!boxes.length && expl?.viewport?.car != null && expl.viewport.side) {
      const s = String(expl.viewport.side).toUpperCase() === "II" ? 0 : 1;
      boxes = [1, 3, 5, 7].map((p) => ({ car: +expl.viewport.car, position: p + (s ? 0 : 1) }));
    }
    const bh = normHealth(expl?.viewport?.health ?? "crit");
    boxes.forEach((b, i) => {
      const c = railBoxComponent(b.car, b.position);
      set(c.car, c.subsystem, c.component_id, bh === "nodata" ? "crit" : bh,
        `car ${b.car} position ${b.position} · contributing #${i + 1}`);
    });
    if (row?.prediction) caption += ` · ${row.prediction}`;
  } else if (task === "acv") {
    const ranked = String(row?.ranked_cars ?? "")
      .split("|")
      .map((s) => s.trim())
      .filter(Boolean);
    ranked.forEach((id, i) => {
      const car = parseInt(id, 10);
      if (!Number.isFinite(car)) return;
      acHealth[car] = { health: i === 0 ? "crit" : i < 3 ? "warn" : "ok", label: `rank ${i + 1} of ${ranked.length} · car ${id}` };
    });
    if (ranked.length) caption += ` · car ${ranked[0]} most likely`;
  } else if (task === "shm") {
    const d = Number(row?.prediction);
    const health = shmHealth(d);
    const label = Number.isFinite(d) ? `cumulative damage ${d.toFixed(4)}` : "no prediction";
    const bogieCar = Number(expl?.viewport?.car) || 1; // car 1 unless the explanation says
    set(bogieCar, "shm", "bogie_1", health, label);
    set(bogieCar, "shm", "bogie_2", health, label);
    if (Number.isFinite(d)) caption += ` · D = ${d.toFixed(3)}`;
  }
  return { task, caption, components, railHealth, acHealth };
}

export const HEALTH_COLORS = {
  ok: "#1f8059",
  warn: "#a66500",
  crit: "#c43d36",
  nodata: "#4a5260",
};

export const HEALTH_WORDS = {
  ok: "OK",
  warn: "WATCH",
  crit: "ALERT",
  nodata: "NO DATA",
};

/** Header line: `8 cars · 64 doors · 16 bogies / 128 axle boxes · 8 air-con units`. */
export function statsLine(stats) {
  if (!stats) return "loading model";
  return `${stats.cars} cars · ${stats.doors} doors · ${stats.bogies} bogies / ${stats.axleboxes} axle boxes · ${stats.ac} air-con units`;
}

/**
 * Merge twin replay components and PS3 prediction health maps into a single Map.
 * The six systems own distinct meshes: LTA doors, simulated axle boxes, and LTA rails.
 *
 * @param {Object} layers { twin: Map|null, door, acv, rail, shm }
 * @returns {Map<string, Object>}
 */
export function mergeHealthMaps(layers = {}) {
  const merged = new Map();
  if (!layers) return merged;

  const toMap = (l) => {
    if (!l) return null;
    if (l instanceof Map) return l;
    return indexHealthMap(l);
  };

  const twin = layers.twin instanceof Map ? layers.twin : null;
  const doorMap = toMap(layers.door);
  const acvMap = toMap(layers.acv);
  const railMap = toMap(layers.rail);
  const shmMap = toMap(layers.shm);

  const sources = [
    { name: "twin", map: twin },
    { name: "door", map: doorMap },
    { name: "acv", map: acvMap },
    { name: "rail", map: railMap },
    { name: "shm", map: shmMap },
  ];

  for (const { name, map } of sources) {
    if (!map) continue;
    for (const [k, v] of map.entries()) {
      if (name === "twin" && /\|door\|/.test(k)) continue;
      if (name === "rail" && /\|bearing\|/.test(k)) continue;
      merged.set(k, v);
    }
  }

  return merged;
}
