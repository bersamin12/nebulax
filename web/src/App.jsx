// Two-page application shell: Overview and the prediction workspace.
//
// The former fleet replay console is archived under web/archive/fleet-dashboard. Old replay
// links are redirected to Predict and their dashboard-only query parameters are removed.
import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import "./styles.css";

import PredictPage from "./predict/PredictPage.jsx";
import { setConsoleScale } from "./lib/consoleScale.js";
import OverviewPage from "./overview/OverviewPage.jsx";
import Header from "./components/Header.jsx";

const W = 1440;
const H = 900;
const PAGES = ["overview", "predict"];

function readPage() {
  if (typeof window === "undefined") return "overview";
  const q = new URLSearchParams(window.location.search);
  const page = q.get("page");
  if (PAGES.includes(page)) return page;
  // Preserve useful old links by taking them to the active workflow.
  if (page === "twin" || q.get("task") || q.get("systems") || q.get("train") || q.get("component")) {
    return "predict";
  }
  return "overview";
}

function readTour() {
  if (typeof window === "undefined") return 0;
  const n = Number(new URLSearchParams(window.location.search).get("tour"));
  return Number.isInteger(n) && n > 0 ? n : 0;
}

// Below this scale, desktop windows scroll the fixed console instead of shrinking it further.
const MIN_SCALE = 0.6;
const MAX_SCALE = 3;
const NOTICE_H = 44;

function useConsoleScale(handheld) {
  const [scale, setScale] = useState(1);
  useLayoutEffect(() => {
    const compute = () => {
      const w = window.innerWidth || W;
      const h = (window.innerHeight || H) - (handheld ? NOTICE_H : 0);
      const fit = Math.min(MAX_SCALE, w / W, h / H);
      const next = handheld ? Math.max(0.2, fit) : Math.max(MIN_SCALE, fit);
      setScale(next);
      setConsoleScale(next);
    };
    compute();
    let ro = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(compute);
      ro.observe(document.documentElement);
    }
    window.addEventListener("resize", compute);
    return () => {
      if (ro) ro.disconnect();
      window.removeEventListener("resize", compute);
    };
  }, [handheld]);
  return scale;
}

function isHandheld() {
  if (typeof navigator === "undefined" || typeof window === "undefined") return false;
  const ua = navigator.userAgent || "";
  if (navigator.userAgentData && navigator.userAgentData.mobile) return true;
  if (/Android|iPhone|iPad|iPod|Mobile|Tablet|Silk|Kindle/i.test(ua)) return true;
  if (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1) return true;
  const mq = (query) => (window.matchMedia ? window.matchMedia(query).matches : false);
  return mq("(any-pointer: coarse)") && !mq("(any-pointer: fine)");
}

function HandheldNotice() {
  return (
    <div className="nx-handheld-notice" role="status">
      This workspace is designed for desktop. On a phone or tablet, pinch to zoom or open it on a larger screen.
    </div>
  );
}

export default function App() {
  const [handheld] = useState(isHandheld);
  const scale = useConsoleScale(handheld);
  const [page, setPage] = useState(readPage);
  const [tourKey, setTourKey] = useState(() => (readPage() === "predict" ? readTour() : 0));

  const onPage = useCallback((requested, opts = {}) => {
    const next = PAGES.includes(requested) ? requested : "predict";
    setPage(next);
    if (opts.tour) setTourKey((key) => key + 1);
    if (typeof window === "undefined") return;
    const q = new URLSearchParams(window.location.search);
    q.set("page", next);
    q.delete("tour");
    for (const key of ["systems", "train", "component", "car", "subsystem"]) q.delete(key);
    if (next !== "predict") q.delete("task");
    window.history.pushState(null, "", `${window.location.pathname}?${q}`);
  }, []);

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    let changed = false;
    const legacyKeys = ["systems", "train", "component", "car", "subsystem"];
    const legacyLink = q.get("page") === "twin" || legacyKeys.some((key) => q.has(key));
    if (legacyLink) {
      q.set("page", "predict");
      changed = true;
    }
    for (const key of legacyKeys) {
      if (q.has(key)) {
        q.delete(key);
        changed = true;
      }
    }
    if (changed) window.history.replaceState(null, "", `${window.location.pathname}?${q}`);

    const onPop = () => setPage(readPage());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  return (
    <div className="nx-stage" style={handheld ? { paddingTop: NOTICE_H } : undefined}>
      {handheld && <HandheldNotice />}
      <div style={{ width: W * scale, height: H * scale, flex: "none" }}>
        <div className="nx-console" style={{ transform: `scale(${scale})` }}>
          <Header page={page} onPage={onPage} />
          {page === "predict" ? (
            <PredictPage height={H - 56} tourKey={tourKey} />
          ) : (
            <OverviewPage height={H - 56} onPage={onPage} />
          )}
        </div>
      </div>
    </div>
  );
}
