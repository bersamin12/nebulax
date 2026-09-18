// `explanation.viewport` (docs/ps3_contract.md section 2) -> the {car, subsystem, component_id}
// the 3D viewport understands, so clicking a results row moves the twin's selection with it.
//
// The contract's `component` is a free-form mesh/agent-readable name - `car3_ac1`, `door_L1`,
// `axlebox_c3_p1`, `car5_bogie1` - so each task reads the shape it knows and falls back to the
// component ps3HealthMap() tinted for that task, never to a guess the viewport cannot pick.

import { railBoxComponent } from "../components/viewport/componentMap.js";

const AXLEBOX = /^axlebox_c(\d+)_p(\d+)$/i;
const BOGIE = /^car(\d+)_bogie([12])/i;
const AC = /^car(\d+)_ac([12])?/i;
const DOOR = /^(?:car(\d+)_)?(door_[LR][1-4])$/i;

export function selectionFor(task, explanation, trainId = "PS3") {
  const vp = (explanation && explanation.viewport) || {};
  const comp = String(vp.component || "");
  const car = Number(vp.car);
  const hit = (c, subsystem, component_id) => ({
    train_id: trainId,
    car: c,
    subsystem,
    component_id,
  });

  if (task === "rail") {
    const m = AXLEBOX.exec(comp);
    if (m) {
      const b = railBoxComponent(+m[1], +m[2]);
      return hit(b.car, b.subsystem, b.component_id);
    }
    const side = String(vp.side || "").toUpperCase();
    return hit(0, "rail", side === "II" ? "rail_II" : "rail_I");
  }
  if (task === "acv") {
    const m = AC.exec(comp);
    const c = m ? +m[1] : Number.isFinite(car) ? car : null;
    return c ? hit(c, "acv", "ac_1") : null;
  }
  if (task === "shm") {
    const m = BOGIE.exec(comp);
    const c = m ? +m[1] : Number.isFinite(car) ? car : 1;
    return hit(c, "shm", `bogie_${m ? m[2] : 1}`);
  }
  if (task === "door") {
    const m = DOOR.exec(comp);
    const c = m && m[1] ? +m[1] : Number.isFinite(car) ? car : 1;
    return hit(c, "door", m ? m[2] : "door_L1");
  }
  return null;
}

/** The explanation that belongs to one results row (door's stream-level one for every segment). */
export function explanationFor(task, explanations, row) {
  const list = explanations || [];
  if (!list.length) return null;
  if (task === "door" || !row || row.file_id === undefined) return list[0];
  return list.find((e) => String(e.file_id) === String(row.file_id)) || null;
}
