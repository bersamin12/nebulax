// Two-page application shell: Overview and the prediction workspace.
//
// The overview is an ordinary responsive page: full width, the document scrolls, it reflows on a
// phone. The prediction workspace is a fixed 1440x900 console that is CSS-scaled to fit the window
// (never reflowed); a phone or tablet gets a notice with an escape hatch to the fitted console.
//
// The former fleet replay console is archived under web/archive/fleet-dashboard. Old replay
// links are redirected to Predict and their dashboard-only query parameters are removed.
import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import "./styles.css";

import { usePs3Predict } from "./predict/usePs3Predict.js";
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
// A window narrower than this cannot show the console usefully even at MIN_SCALE.
const SMALL_W = 700;

function useConsoleScale(handheld, active) {
  const [scale, setScale] = useState(1);
  useLayoutEffect(() => {
    if (!active) return undefined;
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
  }, [handheld, active]);
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

function useSmallWindow() {
  const [small, setSmall] = useState(() => typeof window !== "undefined" && window.innerWidth < SMALL_W);
  useEffect(() => {
    const on = () => setSmall(window.innerWidth < SMALL_W);
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return small;
}

function HandheldNotice() {
  return (
    <div className="nx-handheld-notice" role="status">
      Limited support on this screen: the workspace is laid out for desktop. Pinch to zoom if needed.
    </div>
  );
}

/** Shown to phones, tablets and very narrow windows instead of the fixed console. */
function SmallScreenGate({ onBack, onContinue }) {
  return (
    <div className="nx-gate">
      <div className="nx-gate-card" role="dialog" aria-labelledby="nx-gate-title">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <rect x="2" y="4" width="20" height="13" rx="2" />
          <path d="M8 21h8M12 17v4" />
        </svg>
        <h1 id="nx-gate-title">The prediction workspace needs a larger screen</h1>
        <p>
          It is a fixed desktop console with a 3D train view, file uploads and a results table side by side.
          Open this link on a laptop or desktop browser at least 900 px wide.
        </p>
        <div className="nx-gate-actions">
          <button type="button" className="nx-gate-btn nx-gate-btn--primary" onClick={onBack}>
            Back to overview
          </button>
          <button type="button" className="nx-gate-btn" onClick={onContinue}>
            Open anyway (limited)
          </button>
        </div>
      </div>
    </div>
  );
}

/** The fixed 1440x900 console, scaled to the window. */
function Console({ handheld, page, onPage, tourKey, predictor }) {
  const scale = useConsoleScale(handheld, true);
  return (
    <div className="nx-stage" style={handheld ? { paddingTop: NOTICE_H } : undefined}>
      {handheld && <HandheldNotice />}
      <div style={{ width: W * scale, height: H * scale, flex: "none" }}>
        <div className="nx-console" style={{ transform: `scale(${scale})` }}>
          <Header page={page} onPage={onPage} />
          <PredictPage height={H - 56} tourKey={tourKey} predictor={predictor} />
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const predictor = usePs3Predict(new URLSearchParams(window.location.search).get("task"));
  const [handheld] = useState(isHandheld);
  const small = useSmallWindow();
  const [page, setPage] = useState(readPage);
  const [tourKey, setTourKey] = useState(() => (readPage() === "predict" ? readTour() : 0));
  // "Open anyway" on the small-screen gate, remembered for the session only
  const [forceConsole, setForceConsole] = useState(false);

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
    window.scrollTo(0, 0);
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

  // styles.css keys the document's scrolling and background on this
  useEffect(() => {
    document.body.dataset.page = page;
  }, [page]);

  if (page === "overview") {
    return (
      <div className="nx-page">
        <Header page={page} onPage={onPage} fluid />
        <OverviewPage onPage={onPage} />
      </div>
    );
  }
  if ((handheld || small) && !forceConsole) {
    return <SmallScreenGate onBack={() => onPage("overview")} onContinue={() => setForceConsole(true)} />;
  }
  return <Console predictor={predictor} handheld={handheld || small} page={page} onPage={onPage} tourKey={tourKey} />;
}
