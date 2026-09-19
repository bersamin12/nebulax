// The console's current CSS scale (App.jsx writes it), for parts that render pixels themselves:
// the 3D viewport raises its device pixel ratio with it, so a console scaled up to fill a 2.5K
// or 4K screen is drawn sharp instead of being upscaled from a 1440-wide canvas.
import { useSyncExternalStore } from "react";

let scale = 1;
const listeners = new Set();

export function setConsoleScale(next) {
  if (next === scale) return;
  scale = next;
  for (const fn of listeners) fn();
}

export function getConsoleScale() {
  return scale;
}

export function useConsoleScaleValue() {
  return useSyncExternalStore(
    (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    getConsoleScale,
    getConsoleScale
  );
}
