// Material bookkeeping for the 3D train: one cloned material per tintable mesh, health tint +
// emissive, a crit pulse and the door leaf open/close offsets.
import * as THREE from "three";
import {
  CAR_RE,
  componentForMesh,
  componentKey,
  doorLeafGroups,
  HEALTH_COLORS,
} from "./componentMap.js";

const COLOR = {
  ok: new THREE.Color(HEALTH_COLORS.ok),
  warn: new THREE.Color(HEALTH_COLORS.warn),
  crit: new THREE.Color(HEALTH_COLORS.crit),
};
const ACCENT = new THREE.Color("#008f95");

// Emissive strength per health; crit is animated between CRIT_LO and CRIT_HI.
const GLOW = { ok: 0.18, warn: 0.45 };
const CRIT_LO = 0.3;
const CRIT_HI = 1.0;
const LEAF_TRAVEL = 0.44; // m, the template's HALF_LEAF logic (full travel is 0.725)

const NO_RAYCAST = () => {};

/** Make the pick meshes of a component hit-testable or not (the tint is unaffected). */
export function setPickable(entry, pickable) {
  if (entry.pickable === pickable) return;
  entry.pickable = pickable;
  for (const m of entry.pickMeshes) {
    if (pickable) delete m.raycast; // back to Mesh.prototype.raycast
    else m.raycast = NO_RAYCAST;
  }
}

/**
 * Walk a cloned glb scene once: clone the material of every mesh that belongs to a contract
 * component, remember its base look, disable raycasting on everything else, and group the meshes
 * per component id. Also measures the set (`bounds`, rails excluded) and counts what the model
 * contains (`stats`: cars, doors, bogies, axleboxes, ac, apu) so nothing downstream assumes a
 * number of cars.
 *
 * @returns {{components: Map<string, object>, order: object[], bounds: object, stats: object}}
 */
export function buildComponentIndex(scene) {
  scene.updateMatrixWorld(true);
  const components = new Map();
  const bounds = new THREE.Box3();
  const cars = new Set();
  const tmp = new THREE.Box3();
  scene.traverse((o) => {
    if (!o.isMesh) return;
    const carMatch = CAR_RE.exec(o.name);
    if (carMatch) {
      cars.add(+carMatch[1]);
      tmp.setFromObject(o);
      bounds.union(tmp);
    }
    const c = componentForMesh(o.name);
    if (!c) {
      o.raycast = NO_RAYCAST; // body, wheels, brakes, drives: not selectable, and cheap to skip
      return;
    }
    const key = componentKey(c.car, c.subsystem, c.component_id);
    let entry = components.get(key);
    if (!entry) {
      entry = { ...c, key, meshes: [], pickMeshes: [], leaves: [], health: "nodata", comp: null, pickable: true };
      components.set(key, entry);
    }
    const mat = o.material.clone();
    o.material = mat;
    o.userData.baseColor = mat.color.clone();
    o.userData.baseEmissive = mat.emissive ? mat.emissive.clone() : new THREE.Color(0, 0, 0);
    o.userData.baseEmissiveIntensity = mat.emissiveIntensity ?? 1;
    o.userData.componentKey = key;
    entry.meshes.push(o);
    entry.pickMeshes.push(o);
  });
  const stats = { cars: cars.size, doors: 0, bogies: 0, axleboxes: 0, ac: 0, apu: 0 };
  for (const e of components.values()) {
    if (e.subsystem === "door") stats.doors += 1;
    else if (e.subsystem === "shm") stats.bogies += 1;
    else if (e.subsystem === "bearing") stats.axleboxes += 1;
    else if (e.subsystem === "acv") stats.ac += 1;
    else if (e.subsystem === "pneumatic") stats.apu += 1;
  }

  for (const entry of components.values()) {
    if (entry.subsystem === "door") {
      doorLeafGroups(entry.car, entry.component_id).forEach((name, i) => {
        const g = scene.getObjectByName(name);
        if (g) {
          g.userData.homeX = g.position.x;
          entry.leaves.push({ group: g, dir: i === 0 ? -1 : 1 });
        }
      });
    }
    const box = new THREE.Box3();
    for (const m of entry.meshes) box.expandByObject(m);
    entry.box = box;
    entry.center = box.getCenter(new THREE.Vector3());
    // a stable per-component phase so the animated doors do not move in lockstep
    let h = 0;
    for (let i = 0; i < entry.key.length; i++) h = (h * 31 + entry.key.charCodeAt(i)) % 997;
    entry.phase = h / 997;
  }
  const size = bounds.getSize(new THREE.Vector3());
  const centre = bounds.getCenter(new THREE.Vector3());
  return {
    components,
    order: [...components.values()],
    stats,
    bounds: { minX: bounds.min.x, maxX: bounds.max.x, maxY: bounds.max.y, centreX: centre.x, length: size.x },
  };
}

/** Restore every tintable mesh of a component to the material it had in the glb. */
export function paintBase(entry) {
  for (const m of entry.meshes) {
    m.material.color.copy(m.userData.baseColor);
    if (m.material.emissive) {
      m.material.emissive.copy(m.userData.baseEmissive);
      m.material.emissiveIntensity = m.userData.baseEmissiveIntensity;
    }
  }
}

/**
 * Tint one component for its health.
 * @param entry      component entry from buildComponentIndex
 * @param health     ok | warn | crit | nodata
 * @param selected   draw it as the selected component (brighter, accent-shifted emissive)
 * @param t          seconds, for the crit pulse
 */
export function paintComponent(entry, health, selected, t) {
  if (health === "nodata" || !COLOR[health]) {
    if (!selected) {
      paintBase(entry);
      return;
    }
    for (const m of entry.meshes) {
      m.material.color.copy(m.userData.baseColor);
      if (m.material.emissive) {
        m.material.emissive.copy(ACCENT);
        m.material.emissiveIntensity = 0.35;
      }
    }
    return;
  }
  const c = COLOR[health];
  let glow = GLOW[health] ?? 0.2;
  if (health === "crit") {
    const pulse = 0.5 + 0.5 * Math.sin(t * 3.4 + entry.phase * 6.283);
    glow = CRIT_LO + (CRIT_HI - CRIT_LO) * pulse;
  }
  if (selected) glow = Math.max(glow, 0.55) + 0.25;
  for (const m of entry.meshes) {
    m.material.color.copy(c);
    if (m.material.emissive) {
      m.material.emissive.copy(c);
      if (selected) m.material.emissive.lerp(ACCENT, 0.45);
      m.material.emissiveIntensity = glow;
    }
  }
}

/**
 * Slow open/close of the leaves of a door that is warn or crit, so the eye is drawn to it.
 * Healthy / not-instrumented doors stay shut.
 */
export function animateDoor(entry, health, t) {
  if (!entry.leaves.length) return;
  let open = 0;
  if (health === "crit" || health === "warn") {
    const period = health === "crit" ? 7 : 11;
    const u = ((t / period + entry.phase) % 1 + 1) % 1;
    const ramp = (x) => x * x * (3 - 2 * x); // smoothstep
    if (u < 0.22) open = ramp(u / 0.22);
    else if (u < 0.5) open = 1;
    else if (u < 0.72) open = 1 - ramp((u - 0.5) / 0.22);
    else open = 0;
  }
  const dx = open * LEAF_TRAVEL;
  for (const { group, dir } of entry.leaves) {
    const x = group.userData.homeX + dir * dx;
    if (group.position.x === x) continue;
    group.position.x = x;
    // the glb subtree is taken out of the renderer's matrix walk (see TrainViewport), so a leaf
    // that moves has to re-compose its own branch
    group.updateMatrixWorld(true);
  }
}
