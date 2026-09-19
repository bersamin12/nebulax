// TrainViewport - the 3D train schematic shared by Predict and the direct-link replay console.
//
// Contract: <TrainViewport frame trainId selected onSelect height={268} healthMap={null} viewMode="door" />
//
//   frame      a replay frame (docs/app_contract.md section 5). The twin's input: the components
//              of the chosen train are tinted by their `health`; doors, axle boxes and the APU are
//              pickable; every mesh the frame does not mention renders `nodata` (with the 8-car
//              model the sim's cars 1-6 paint exactly as before and cars 7-8 stay grey).
//   trainId    which train of frame.trains to draw (falls back to the first one)
//   selected   {train_id, car, subsystem, component_id} | null
//   onSelect   called with the same shape (or null when the empty stage is clicked)
//   height     stage height in px (268 on Predict)
//   viewMode   task camera preset: one car, all cars, or overhead rails
//
//   healthMap  OPTIONAL second input mode, for the PS3 predict page. When given it wins over
//              `frame` (the frame is ignored entirely). Shape:
//                {
//                  components: { [componentKey]: { health, label? } },  // "car|subsystem|component_id"
//                  railHealth: { I: health, II: health },              // tints `rail` / `rail001`
//                  acHealth:   { [car]: health | {health, label} },    // tints car{car}_ac1 + _ac2
//                  caption?:   string                                  // header, e.g. "RAIL CORRUGATION · Test12.csv"
//                }
//              health is "ok" | "warn" | "crit" | "nodata". componentKey is
//              `${car}|${subsystem}|${component_id}` (componentMap.componentKey), with the ids
//                door      car c   door_L1..L4 / door_R1..R4
//                bearing   car c   axlebox_{1-4}{L|R}     (rail position p -> railBoxComponent(car, p))
//                pneumatic car c   apu_1
//                acv       car c   ac_1                   (both roof units of the car)
//                shm       car c   bogie_1 / bogie_2
//                rail      car 0   rail_I / rail_II
//              Only components present in the map are pickable; onSelect then receives
//              {train_id: trainId ?? "PS3", car, subsystem, component_id}. Build the map with
//              componentMap.ps3HealthMap({task, rows, explanations, fileId}) - see there.
//
// The mesh-name mapping and the full glb node inventory are documented in componentMap.js.
// Nothing here assumes six cars: the set's extent, the camera framing and the header counts come
// from the loaded glb (buildComponentIndex -> bounds / stats).
// ?fallback=1 in the URL forces the 2D fallback (used for the screenshot check).
import React, { Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, OrbitControls, useGLTF } from "@react-three/drei";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import {
  CAR_RE,
  componentForMesh,
  componentKey,
  HEALTH_COLORS,
  HEALTH_WORDS,
  indexFrame,
  indexHealthMap,
  sideOf,
  statsLine,
} from "./componentMap.js";
import { animateDoor, buildComponentIndex, paintComponent, setPickable } from "./useHealthMaterials.js";
import TrainElevationFallback from "./TrainElevationFallback.jsx";
import { fmtScore } from "../../lib/format.js";
import { useConsoleScaleValue } from "../../lib/consoleScale.js";
import "./viewport.css";

const GLB_URL = "/r151.glb";

// Framing constants. The set's extent along X is measured from the glb (index.bounds); the
// default preset fits `length * SPAN_MARGIN` world metres across the canvas width.
const SPAN_MARGIN = 1.052; // 152 / 144.4, the ratio the six-car preset used
const TARGET_Y = 1.95; // the window band: the car keeps its full height in a short strip
const HOME_OFFSET = new THREE.Vector3(0, 11, 100); // elevation, a few degrees above the horizon
// The strip is ~9:1 while an 8-car set is ~48:1, so the elevation preset stretches the vertical
// axis of the orthographic projection (the design mock does the same). 1 = true scale; the
// stretch eases out to 1 as the camera is orbited away from the elevation.
const VERTICAL_EXAGGERATION = 1.85;
const ACCENT = "#008f95";
// before the glb resolves: an 8-car set, so the first frame is already about right
const GUESS_BOUNDS = { minX: -1.16, maxX: 190.17, centreX: 94.5, length: 191.33, maxY: 3.985 };

function hasWebGL() {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

class GLErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  componentDidCatch(error) {
    // eslint-disable-next-line no-console
    console.error("TrainViewport: 3D view failed, falling back to the 2D elevation", error);
    this.props.onError?.(error);
  }
  render() {
    return this.state.error ? null : this.props.children;
  }
}

/* ------------------------------------------------------------------ camera */

/**
 * r3f keeps an orthographic camera's frustum at +-size/2, so `zoom` is pixels per metre. The
 * horizontal axis is then squashed by `exaggerationRef` inside updateProjectionMatrix, which is
 * what lets the whole set fit the width of a very wide, short strip while a car still reads at a
 * usable height. The frustum stays symmetric, so only element 0 has to change.
 */
function ElevationCamera({ exaggerationRef }) {
  const camera = useThree((s) => s.camera);
  useLayoutEffect(() => {
    const proto = Object.getPrototypeOf(camera).updateProjectionMatrix;
    camera.updateProjectionMatrix = function patched() {
      proto.call(this);
      const e = exaggerationRef.current;
      if (e && e !== 1) {
        this.projectionMatrix.elements[0] /= e;
        this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
      }
    };
    camera.updateProjectionMatrix();
    return () => {
      delete camera.updateProjectionMatrix;
      camera.updateProjectionMatrix();
    };
  }, [camera, exaggerationRef]);
  return null;
}

function CameraRig({ focus, resetToken, exaggerationRef, baseZoom, bounds, cars, viewMode }) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls);
  const invalidate = useThree((s) => s.invalidate);
  const width = useThree((s) => s.size.width);
  const height = useThree((s) => s.size.height);
  const preset = useMemo(() => {
    const carSpan = bounds.length / Math.max(cars, 1);
    const carX = (car) => bounds.minX + carSpan * (car - 0.5);
    const carZoom = Math.max(1, Math.min(width / 30, height / 4.8));
    const common = { offset: HOME_OFFSET.clone(), up: new THREE.Vector3(0, 1, 0), exaggeration: 1 };
    if (viewMode === "rail") return {
      target: new THREE.Vector3(bounds.centreX, 0.2, 0),
      zoom: Math.max(1, Math.min(width / (carSpan * 2.3), height / 5.5)),
      offset: new THREE.Vector3(0, 100, 0.01),
      up: new THREE.Vector3(0, 0, -1),
      exaggeration: 1,
      carZoom,
    };
    if (viewMode === "acv" || !viewMode) return {
      ...common, target: new THREE.Vector3(bounds.centreX, TARGET_Y, 0),
      zoom: baseZoom, exaggeration: VERTICAL_EXAGGERATION, carZoom,
    };
    const car = viewMode === "pneumatic" ? 3 : 1;
    return { ...common, target: new THREE.Vector3(carX(car), TARGET_Y, 0), zoom: carZoom, carZoom };
  }, [bounds, cars, width, height, viewMode, baseZoom]);
  const want = useRef({ target: preset.target.clone(), zoom: preset.zoom });

  // Changing systems and double-clicking reset both land on the current task's home view.
  useLayoutEffect(() => {
    if (!controls) return;
    want.current.target.copy(preset.target);
    want.current.zoom = preset.zoom;
    exaggerationRef.current = preset.exaggeration;
    camera.up.copy(preset.up);
    controls.target.copy(preset.target);
    camera.position.copy(preset.target).add(preset.offset);
    camera.zoom = preset.zoom;
    camera.lookAt(preset.target);
    camera.updateProjectionMatrix();
    controls.update();
    invalidate();
  }, [controls, camera, preset, resetToken, exaggerationRef, invalidate]);

  // ACV starts on all eight cars and moves to a car when a rank is clicked. Door and SHM keep
  // their one-car framing while tracking the selected component. Rail stays overhead.
  useEffect(() => {
    if (!focus || viewMode === "rail" || viewMode === "pneumatic" || viewMode === "bearing") {
      want.current.target.copy(preset.target);
      want.current.zoom = preset.zoom;
      invalidate();
      return;
    }
    want.current.target.set(
      THREE.MathUtils.clamp(focus.x, bounds.minX, bounds.maxX),
      TARGET_Y,
      0,
    );
    want.current.zoom = preset.carZoom;
    invalidate();
  }, [focus, viewMode, preset, bounds, invalidate]);

  useFrame((_, dt) => {
    if (!controls) return;
    const k = 1 - Math.pow(0.002, Math.min(dt, 0.1)); // frame-rate independent easing
    let moving = false;

    // ease the vertical stretch out as the camera is orbited away from the elevation
    const az = Math.abs(controls.getAzimuthalAngle ? controls.getAzimuthalAngle() : 0);
    const framingExag = focus && (viewMode === "acv" || !viewMode) ? 1 : preset.exaggeration;
    const wantExag = THREE.MathUtils.lerp(framingExag, 1, Math.min(1, az / 0.52));
    if (Math.abs(wantExag - exaggerationRef.current) > 2e-3) {
      exaggerationRef.current = THREE.MathUtils.lerp(exaggerationRef.current, wantExag, k);
      camera.updateProjectionMatrix();
      moving = true;
    }

    const off = camera.position.clone().sub(controls.target);
    if (want.current.target.distanceTo(controls.target) > 0.01) {
      controls.target.lerp(want.current.target, k);
      moving = true;
    }
    if (moving) camera.position.copy(controls.target).add(off);

    const dz = want.current.zoom - camera.zoom;
    if (Math.abs(dz) > 0.02) {
      camera.zoom += dz * k;
      camera.updateProjectionMatrix();
      moving = true;
    }
    if (moving) invalidate();
  });
  return null;
}

/* ------------------------------------------------------------------- model */

function plate(entry, geometry, position) {
  const mesh = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({ color: 0x2b313b, roughness: 0.85, metalness: 0.05 }),
  );
  mesh.name = `${entry.key}__farside`;
  mesh.position.copy(position);
  mesh.userData.homePosition = position.clone();
  mesh.userData.componentKey = entry.key;
  mesh.userData.baseColor = mesh.material.color.clone();
  mesh.userData.baseEmissive = mesh.material.emissive.clone();
  mesh.userData.baseEmissiveIntensity = mesh.material.emissiveIntensity;
  return mesh;
}

/**
 * Far-side and rail markers. R1..R4 doors and the R axle boxes sit behind the car body in a side
 * elevation, so their health is shown on a small plate on the near side: doors in the roof band
 * (as in the design mock's "FAR SIDE R1-R4" legend), axle boxes just above their near-side twin.
 * The two rails are 15 cm tall, a couple of pixels in this strip, so each gets a long plate under
 * the near rail (Side I first, Side II below it) that is only shown while the rail is tinted.
 * Plates are registered as ordinary meshes of their component, so tint, pulse and picking need no
 * special case; the real far-side / rail meshes are still tinted but only the plate is pickable.
 * Axle-box and rail plates are hidden while that component is not instrumented.
 */
function addFarSideMarkers(scene, index) {
  const group = new THREE.Group();
  group.name = "farSideMarkers";
  const doorGeo = new THREE.BoxGeometry(1.3, 0.11, 0.85);
  const boxGeo = new THREE.BoxGeometry(0.52, 0.14, 0.34);
  const railLen = index.bounds.length + 10;
  const railGeo = new THREE.BoxGeometry(railLen, 0.2, 0.3);
  for (const entry of index.order) {
    const isRail = entry.subsystem === "rail";
    if (!isRail && sideOf(entry.subsystem, entry.component_id) !== "R") continue;
    if (!isRail && entry.subsystem !== "door" && entry.subsystem !== "bearing") continue;
    let mesh;
    if (isRail) {
      const y = entry.component_id === "rail_I" ? -0.32 : -0.62;
      mesh = plate(entry, railGeo, new THREE.Vector3(index.bounds.centreX, y, 1.0));
    } else if (entry.subsystem === "door") {
      mesh = plate(entry, doorGeo, new THREE.Vector3(entry.center.x, 4.16, 0));
    } else {
      // z = 1.5: clear of the underframe skirt (+-1.35) so the plate is never hidden behind it
      mesh = plate(entry, boxGeo, new THREE.Vector3(entry.center.x, 1.02, 1.5));
    }
    // the real meshes sit behind the body (or are a few pixels tall): they are still tinted, but
    // only the plate is pickable, otherwise a click on a near-side door would fall through to
    // its far-side twin
    for (const m of entry.meshes) m.raycast = () => {};
    entry.meshes.push(mesh);
    entry.pickMeshes = [mesh];
    entry.ghost = mesh;
    group.add(mesh);
  }
  scene.add(group);
}

/**
 * The glb is ~1170 separate meshes, which is a lot of draw calls for a strip that may render every
 * frame while something pulses. Everything that is neither tintable nor animated (bodies, wheels,
 * brakes, drives, limit lamps) is baked into one mesh per car per material, which leaves per-car
 * frustum culling. Door leaves are skipped: they slide. Anything componentForMesh() knows (doors,
 * axle boxes, APU parts, the roof AC units, bogie frames, the rails) is left alone.
 */
function mergeStatic(scene) {
  const buckets = new Map();
  const doomed = [];
  scene.traverse((o) => {
    if (!o.isMesh) return;
    if (o.userData.componentKey || componentForMesh(o.name)) return; // tinted per health
    if (/_leaf[AB]_/.test(o.name)) return; // moves with the door leaf
    const car = CAR_RE.exec(o.name)?.[1] ?? "x";
    const key = `${car}|${o.material.uuid}`;
    let b = buckets.get(key);
    if (!b) buckets.set(key, (b = { material: o.material, geometries: [] }));
    b.geometries.push(o.geometry.clone().applyMatrix4(o.matrixWorld));
    doomed.push(o);
  });
  for (const o of doomed) o.removeFromParent();
  const merged = new THREE.Group();
  merged.name = "staticMerged";
  for (const [key, b] of buckets) {
    const geometry = mergeGeometries(b.geometries, false);
    for (const g of b.geometries) g.dispose();
    if (!geometry) continue;
    const mesh = new THREE.Mesh(geometry, b.material);
    mesh.name = `merged_${key}`;
    mesh.raycast = () => {};
    merged.add(mesh);
  }
  scene.add(merged);
}

/** Build the (cached) scene index once per glb; shared by SceneContents and TrainModel. */
function useModel() {
  const { scene: gltfScene } = useGLTF(GLB_URL);
  // clone so the drei cache (and any other viewport) keeps the untouched materials/transforms
  const scene = useMemo(() => gltfScene.clone(true), [gltfScene]);
  const index = useMemo(() => {
    const idx = buildComponentIndex(scene);
    addFarSideMarkers(scene, idx);
    mergeStatic(scene);
    // Static meshes otherwise share the glTF cache material. The rail view needs to fade this
    // train body while keeping the two actual rails opaque, so give each merged mesh its own mat.
    scene.traverse((o) => {
      if (o.isMesh && o.name.startsWith("merged_")) o.material = o.material.clone();
    });
    // ~1000 static meshes: compose their world matrices once and take the whole subtree out of the
    // renderer's per-frame matrix walk. animateDoor() re-composes the leaves it moves.
    scene.updateMatrixWorld(true);
    scene.matrixWorldAutoUpdate = false;
    return idx;
  }, [scene]);
  return { scene, index };
}

function TrainModel({ scene, index, lookup, mapMode, selectedKey, onPick, onHover, onFocus, viewMode }) {
  const invalidate = useThree((s) => s.invalidate);
  const live = useRef([]);
  const railMode = viewMode === "rail";

  useEffect(() => {
    scene.traverse((o) => {
      if (!o.isMesh || !o.name.startsWith("merged_")) return;
      o.material.transparent = railMode;
      o.material.opacity = railMode ? 0.12 : 1;
      o.material.depthWrite = !railMode;
      o.material.needsUpdate = true;
    });
    invalidate();
  }, [scene, railMode, invalidate]);

  // health of the current input -> tint, pickability, plus the short list that needs per-frame work
  useEffect(() => {
    for (const entry of index.order) {
      const comp = lookup.get(entry.key) ?? null;
      entry.comp = comp;
      entry.health = comp?.health ?? "nodata";
      if (entry.ghost) {
        entry.ghost.visible = entry.subsystem === "door" || (railMode && entry.subsystem === "rail") || entry.health !== "nodata";
        if (entry.subsystem === "rail") {
          entry.ghost.position.copy(entry.ghost.userData.homePosition);
          if (railMode) entry.ghost.position.set(index.bounds.centreX, 0.25, entry.component_id === "rail_I" ? 0.7175 : -0.7175);
          entry.ghost.updateMatrixWorld(true);
        }
      }
      // twin: the frame's subsystems are pickable; ps3 map: exactly what the map names
      setPickable(entry, comp != null);
      paintComponent(entry, entry.health, entry.key === selectedKey, 0);
      if (entry.subsystem !== "rail") {
        for (const m of entry.meshes) {
          m.material.transparent = railMode;
          m.material.opacity = railMode ? (entry.health === "crit" ? 0.9 : 0.16) : 1;
          m.material.depthWrite = !railMode;
          m.material.needsUpdate = true;
        }
      }
      animateDoor(entry, entry.health, 0);
    }
    live.current = index.order.filter(
      (e) => e.health === "crit" || e.health === "warn" || e.key === selectedKey,
    );
    invalidate();
  }, [lookup, mapMode, index, selectedKey, railMode, invalidate]);

  useEffect(() => {
    const entry = selectedKey ? index.components.get(selectedKey) : null;
    onFocus(entry ? entry.center.clone() : null);
  }, [selectedKey, index, onFocus]);

  // only the crit/warn/selected components need per-frame work; the canvas runs its loop only
  // while something is crit or warn (frameloop below), otherwise it renders on demand
  useFrame((state) => {
    if (!live.current.length) return;
    const t = state.clock.elapsedTime;
    for (const entry of live.current) {
      paintComponent(entry, entry.health, entry.key === selectedKey, t);
      animateDoor(entry, entry.health, t);
    }
  });

  const handle = (fn) => (e) => {
    const key = e.object?.userData?.componentKey;
    if (!key) return;
    const entry = index.components.get(key);
    if (!entry) return;
    e.stopPropagation();
    fn(entry);
  };

  const selectedEntry = selectedKey ? index.components.get(selectedKey) : null;

  return (
    <group>
      <primitive
        object={scene}
        onClick={handle(onPick)}
        onPointerMove={handle(onHover)}
        onPointerOut={() => onHover(null)}
      />
      {selectedEntry ? <box3Helper args={[selectedEntry.box, new THREE.Color(ACCENT)]} /> : null}
    </group>
  );
}

/** Everything that may only mount once the glb has resolved. */
function SceneContents(props) {
  const { resetToken, exaggerationRef, onStats, viewMode } = props;
  const { scene, index } = useModel();
  const [focus, setFocus] = useState(null);
  const size = useThree((s) => s.size);
  const bounds = index.bounds;
  const baseZoom = Math.max(1, (size.width * VERTICAL_EXAGGERATION) / (bounds.length * SPAN_MARGIN));
  const onFocus = useCallback((p) => setFocus(p), []);
  useEffect(() => {
    onStats?.(index.stats);
  }, [index, onStats]);
  return (
    <>
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        enablePan={false}
        minAzimuthAngle={-Math.PI / 6}
        maxAzimuthAngle={Math.PI / 6}
        minPolarAngle={viewMode === "rail" ? 0 : Math.PI / 2 - 0.5}
        maxPolarAngle={viewMode === "rail" ? Math.PI / 2 : Math.PI / 2 + 0.12}
        minZoom={baseZoom * 0.85}
        maxZoom={baseZoom * 12}
        zoomSpeed={0.7}
      />
      <CameraRig
        focus={focus}
        resetToken={resetToken}
        exaggerationRef={exaggerationRef}
        baseZoom={baseZoom}
        bounds={bounds}
        cars={index.stats.cars}
        viewMode={viewMode}
      />
      <TrainModel {...props} scene={scene} index={index} onFocus={onFocus} />
    </>
  );
}

/* --------------------------------------------------------------- component */

export default function TrainViewport({
  frame,
  trainId,
  selected,
  onSelect,
  height = 206,
  healthMap = null,
  overlay = null,
  captions = null,
  viewMode = null,
}) {
  const [glFailed, setGlFailed] = useState(false);
  // the console is CSS-scaled: draw at (device pixel ratio x console scale) so a 4K screen is sharp
  const consoleScale = useConsoleScaleValue();
  const dpr = Math.min(4, Math.max(1, (typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1) * consoleScale));
  const [hover, setHover] = useState(null);
  const [resetToken, setResetToken] = useState(0);
  const [stats, setStats] = useState(null);
  const exaggerationRef = useRef(VERTICAL_EXAGGERATION);

  const forced = useMemo(
    () =>
      typeof window !== "undefined" &&
      new URLSearchParams(window.location.search).get("fallback") === "1",
    [],
  );
  const webgl = useMemo(() => typeof window !== "undefined" && hasWebGL(), []);

  const mapMode = !!healthMap;
  // With a trainId the match is exact (a frame that does not carry it draws an empty set, not
  // another train's health); without one - the dev entry - the first train in the frame wins.
  const train = mapMode
    ? null
    : trainId
      ? (frame?.trains?.find((t) => t.train_id === trainId) ?? null)
      : (frame?.trains?.[0] ?? null);
  const id = mapMode ? (trainId ?? "PS3") : (train?.train_id ?? trainId ?? "-");
  const selectedKey =
    selected && (!selected.train_id || selected.train_id === id)
      ? componentKey(selected.car, selected.subsystem, selected.component_id)
      : null;
  // one lookup for both inputs: componentKey -> {health, ...}; an overlay or healthMap wins over the frame
  const lookup = useMemo(
    () => overlay || (mapMode ? indexHealthMap(healthMap) : indexFrame(train)),
    [overlay, mapMode, healthMap, train],
  );
  const counts = useMemo(() => {
    const c = { crit: 0, warn: 0, ok: 0 };
    for (const comp of lookup.values()) if (c[comp.health] != null) c[comp.health] += 1;
    return c;
  }, [lookup]);
  const animated = counts.crit > 0 || counts.warn > 0;

  const pick = useCallback(
    (entry) =>
      onSelect?.({
        train_id: id,
        car: entry.comp && entry.comp.car != null ? entry.comp.car : entry.car,
        subsystem: entry.subsystem,
        component_id: entry.component_id,
      }),
    [onSelect, id],
  );
  const hoverCb = useCallback((entry) => setHover(entry ?? null), []);
  const reset = useCallback(() => {
    setResetToken((n) => n + 1);
    onSelect?.(null);
  }, [onSelect]);
  const onStats = useCallback((s) => setStats(s), []);

  const title = mapMode ? (healthMap.caption ?? "PS3 PREDICTION") : `SET ${id}`;

  if (forced || !webgl || glFailed) {
    return (
      <TrainElevationFallback
        frame={frame}
        trainId={trainId}
        selected={selected}
        onSelect={onSelect}
        height={height}
        healthMap={healthMap}
        overlay={overlay}
        captions={captions}
        viewMode={viewMode}
        reason={
          forced
            ? "forced with ?fallback=1"
            : !webgl
              ? "no WebGL in this browser"
              : "the 3D model could not be loaded"
        }
      />
    );
  }

  const tip = hover
    ? {
        pos: [hover.center.x, hover.box.max.y + 0.18, hover.center.z],
        title:
          hover.subsystem === "rail"
            ? `RAIL · SIDE ${hover.component_id === "rail_II" ? "II" : "I"}`
            : `CAR ${hover.car} · ${hover.component_id.replace("_", " ").toUpperCase()}`,
        health: hover.health,
        score:
          hover.comp && hover.comp.score != null
            ? `${fmtScore(+hover.comp.score)} / ${fmtScore(+hover.comp.threshold)}`
            : null,
        label: hover.comp?.label ?? null,
        model: hover.comp?.model ?? null,
        stale: !!hover.comp?.stale,
      }
    : null;

  return (
    <div className="nx-vp" style={{ height }}>
      <div className="nx-vp-head">
        <div className="nx-vp-title">
          <b>{title} &middot; {viewMode === "rail" ? "TOP VIEW" : "SIDE ELEVATION"}</b>
          <span>
            {statsLine(stats)}
            {counts.crit || counts.warn ? ` · ${counts.crit} alert / ${counts.warn} watch` : ""}
          </span>
        </div>
        <div className="nx-vp-legend">
          <div><i style={{ background: HEALTH_COLORS.ok }} />OK</div>
          <div><i style={{ background: HEALTH_COLORS.warn }} />WATCH</div>
          <div><i style={{ background: HEALTH_COLORS.crit }} />ALERT</div>
          {viewMode !== "rail" && <div><i style={{ background: "#8c9aa5" }} />FAR SIDE R1&ndash;R4</div>}
        </div>
      </div>
      <div className="nx-vp-stage" onDoubleClick={reset}>
        <GLErrorBoundary onError={() => setGlFailed(true)}>
          <Canvas
            orthographic
            frameloop={animated ? "always" : "demand"}
            dpr={dpr}
            // the console is a fixed 1440x900 board scaled with a CSS transform: measure the
            // layout box, not the transformed bounding rect, or the canvas shrinks with the scale
            resize={{ offsetSize: true }}
            gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
            camera={{
              position: [GUESS_BOUNDS.centreX, TARGET_Y + HOME_OFFSET.y, HOME_OFFSET.z],
              zoom: 12,
              near: 1,
              far: 400,
            }}
            onPointerMissed={() => onSelect?.(null)}
          >
            <hemisphereLight args={[0xffffff, 0xb5c5ce, 1.65]} />
            <directionalLight position={[60, 90, 120]} intensity={1.8} />
            <directionalLight position={[-90, 40, -60]} intensity={0.55} color={0xd5e0e7} />
            <ElevationCamera exaggerationRef={exaggerationRef} />
            <Suspense fallback={null}>
              <SceneContents
                lookup={lookup}
                mapMode={mapMode}
                selectedKey={selectedKey}
                onPick={pick}
                onHover={hoverCb}
                animated={animated}
                resetToken={resetToken}
                exaggerationRef={exaggerationRef}
                onStats={onStats}
                viewMode={viewMode}
              />
            </Suspense>
            {tip ? (
              <Html position={tip.pos} zIndexRange={[20, 0]} style={{ pointerEvents: "none" }}>
                <div className="nx-vp-tip">
                  <b>{tip.title}</b>
                  <div className="nx-vp-sub">
                    <span style={{ color: HEALTH_COLORS[tip.health] }}>{HEALTH_WORDS[tip.health]}</span>
                    {tip.score ? ` · ${tip.score}` : tip.label ? ` · ${tip.label}` : " · not instrumented"}
                    {tip.stale ? " · stale" : ""}
                  </div>
                  {tip.model ? <div className="nx-vp-sub">{tip.model}</div> : null}
                </div>
              </Html>
            ) : null}
          </Canvas>
        </GLErrorBoundary>
        {viewMode === "rail" && (
          <div className="nx-rail-orientation" aria-label="Rail orientation: Side II above Side I in the top view">
            <span>SIDE II · FAR RAIL</span><span>SIDE I · NEAR RAIL</span>
          </div>
        )}
        {captions && captions.length > 0 ? (
          <div
            className="nx-vp-captions"
            style={{
              position: "absolute",
              left: 10,
              bottom: 4,
              display: "flex",
              flexDirection: "column",
              gap: 2,
              zIndex: 10,
              pointerEvents: "none",
            }}
          >
            {captions.map((cap, i) => (
              <div
                key={i}
                className="mono"
                style={{
                  fontSize: 9.5,
                  letterSpacing: "0.06em",
                  color: "#006d73",
                  background: "rgba(255, 255, 255, 0.92)",
                  padding: "1px 6px",
                  borderLeft: "2px solid #008f95",
                }}
              >
                {cap}
              </div>
            ))}
          </div>
        ) : null}
        <div className="nx-vp-hint">
          DRAG TO ROTATE &middot; SCROLL TO ZOOM &middot; DOUBLE-CLICK TO RESET
        </div>
      </div>
    </div>
  );
}

useGLTF.preload(GLB_URL);
