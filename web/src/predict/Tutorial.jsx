// Tutorial mode for the Digital Twin (predict) page: a guided tour that spotlights one control
// at a time, greys out everything else and explains what to do there.
//
// Targets are DOM nodes tagged `data-tour="<id>"` (TaskColumn, PredictPage). The overlay sits
// inside the scaled 1440x900 console, so every measurement is converted from screen pixels back
// to console pixels with the console's own scale; it re-measures on resize and while the page
// loads, so a spotlight follows its control. Keyboard: → / Enter next, ← back, Esc skip.
//
// The tour starts by itself on a first visit (localStorage `nx.tour.seen`), from the header's
// TUTORIAL button, and from the overview page's "Take the 2-minute tour" (`?tour=1`).
import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { C } from "../lib/format.js";

const SEEN_KEY = "nx.tour.seen";
const CARD_W = 330;
const PAD = 6;

export const TOUR_STEPS = [
  {
    target: null,
    title: "Welcome to the Train Digital Twin",
    body:
      "This page runs the four Problem Statement 3 models on a released Test file and shows every prediction on the train model with an explanation. The tour takes about two minutes and walks the page from left to right. Everything inside a highlight stays clickable, so you can follow along as you read.",
  },
  {
    target: "systems",
    title: "Pick a subsystem",
    body:
      "Door, ACV, Rail corrugation and SHM are the four scored predictors, each with the honest cross-validation score of its shipped model. Brake air supply and Axle bearing are exploratory dataset profiles with no upload or prediction. Click a tile to select it; click it again to hide or show its description.",
  },
  {
    target: "files",
    title: "Queue the Test files",
    body:
      "Drop the file (Door: one CSV stream, ACV: one workbook per case) or the whole folder (Rail: 68 runs, SHM: 16 records). A dialog then shows the columns each file carries against what the model expects, and queues the ones that match. Nothing uploads yet: the queue lists what RUN will send.",
  },
  {
    target: "run",
    title: "Press RUN",
    body:
      "The server loads the same frozen model that predict.py uses, featurises each file with training-only baselines and answers with every row accumulated so far. The table fills as batches return; INCLUDE REPLAY FRAMES also streams preview frames to the train.",
  },
  {
    target: "table",
    title: "Read the predictions",
    body:
      "One row per door cycle, case, run or record, in the organiser's own column names. Click a row to select it: the train model and the explanation panel follow. For ACV, click a car chip to see where that car ranks and why.",
  },
  {
    target: "stage",
    title: "The train lights up",
    body:
      "The component behind the selected row turns green for Normal and red for a fault: a door leaf, a roof unit, a bogie side or the underframe member. Drag to orbit, scroll to zoom, click a part to select the row that painted it.",
  },
  {
    target: "explanation",
    title: "Why this prediction",
    body:
      "The numbers the model actually used for that row, its trace with the marks that mattered, and the target the train is pointed at. Hover the (i) next to a variable for its definition. Nothing here is invented: it is the explanation payload the API sent.",
  },
  {
    target: "download",
    title: "Download the submission CSV",
    body:
      "The download is the validated organiser CSV for this session, exactly the rows in the table. CLEAR empties the queue and the results. You can restart this tour any time from TUTORIAL in the header.",
  },
];

function measure(target) {
  if (typeof document === "undefined") return null;
  const el = document.querySelector(`[data-tour="${target}"]`);
  const console_ = el && el.closest(".nx-console");
  if (!el || !console_) return null;
  const cr = console_.getBoundingClientRect();
  const scale = cr.width / 1440 || 1;
  const r = el.getBoundingClientRect();
  return {
    x: (r.left - cr.left) / scale - PAD,
    y: (r.top - cr.top) / scale - PAD,
    w: r.width / scale + PAD * 2,
    h: r.height / scale + PAD * 2,
  };
}

/** The dimmed area around a spotlight: top, left, right and bottom panels (they catch clicks; the hole does not). */
function dimPanels(spot) {
  const x0 = Math.max(0, spot.x);
  const y0 = Math.max(0, spot.y);
  const x1 = Math.min(1440, spot.x + spot.w);
  const y1 = Math.min(900, spot.y + spot.h);
  return [
    { x: 0, y: 0, w: 1440, h: y0 },
    { x: 0, y: y0, w: x0, h: y1 - y0 },
    { x: x1, y: y0, w: 1440 - x1, h: y1 - y0 },
    { x: 0, y: y1, w: 1440, h: 900 - y1 },
  ];
}

function cardPosition(spot) {
  if (!spot) return { left: 720 - CARD_W / 2, top: 300 };
  const gap = 16;
  if (spot.x + spot.w + gap + CARD_W <= 1440 - 12) {
    return { left: spot.x + spot.w + gap, top: Math.max(12, Math.min(spot.y, 900 - 260)) };
  }
  if (spot.x - gap - CARD_W >= 12) {
    return { left: spot.x - gap - CARD_W, top: Math.max(12, Math.min(spot.y, 900 - 260)) };
  }
  const top = spot.y + spot.h + gap;
  return { left: Math.max(12, Math.min(spot.x, 1440 - CARD_W - 12)), top: top + 240 <= 900 ? top : Math.max(12, spot.y - gap - 240) };
}

export function tourSeen() {
  try {
    return typeof localStorage !== "undefined" && localStorage.getItem(SEEN_KEY) === "1";
  } catch {
    return false;
  }
}
function markSeen() {
  try {
    if (typeof localStorage !== "undefined") localStorage.setItem(SEEN_KEY, "1");
  } catch {
    /* private mode: the tour simply offers itself again next time */
  }
}

export default function Tutorial({ open, startStep = 0, onClose }) {
  const [step, setStep] = useState(0);
  const [spot, setSpot] = useState(null);
  const [toast, setToast] = useState(false);
  const last = TOUR_STEPS.length - 1;
  const s = TOUR_STEPS[Math.min(step, last)];

  useEffect(() => {
    if (open) {
      // a tourKey of 1 (or a first visit) starts at the welcome dialog; `?tour=N` jumps to step N
      setStep(startStep > 1 && startStep <= TOUR_STEPS.length - 1 ? startStep : 0);
      setToast(false);
    }
  }, [open, startStep]);

  const finish = useCallback(
    (done) => {
      markSeen();
      setToast(!!done);
      onClose?.();
    },
    [onClose]
  );

  // measure the current target; keep measuring while open so the spot tracks layout changes
  useLayoutEffect(() => {
    if (!open) return undefined;
    const update = () => setSpot(s.target ? measure(s.target) : null);
    update();
    const id = window.setInterval(update, 400);
    window.addEventListener("resize", update);
    return () => {
      window.clearInterval(id);
      window.removeEventListener("resize", update);
    };
  }, [open, s.target]);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      // the spotlighted control is live: keys typed into a field or pressed on a button are its own
      const t = e.target;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
      if (typing && e.key !== "Escape") return;
      if (t && t.tagName === "BUTTON" && e.key === "Enter" && !t.closest(".nx-tour-card")) return;
      if (e.key === "Escape") finish(false);
      else if (e.key === "ArrowRight" || e.key === "Enter") setStep((i) => (i >= last ? (finish(true), i) : i + 1));
      else if (e.key === "ArrowLeft") setStep((i) => Math.max(0, i - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, last, finish]);

  useEffect(() => {
    if (!toast) return undefined;
    const id = window.setTimeout(() => setToast(false), 6000);
    return () => window.clearTimeout(id);
  }, [toast]);

  if (!open) {
    return toast ? (
      <div className="nx-tour-toast" role="status">
        <span style={{ width: 7, height: 7, background: C.ok, display: "block", flex: "none", borderRadius: "50%" }} />
        Tour finished. Restart it any time from TUTORIAL in the header.
        <button type="button" className="nx-tour-link" onClick={() => setToast(false)}>dismiss</button>
      </div>
    ) : null;
  }

  const pos = cardPosition(spot);
  const dialog = step === 0;
  return (
    <div className="nx-tour" role="dialog" aria-modal="true" aria-label="Tutorial">
      {spot ? (
        <>
          {dimPanels(spot).map((r, i) => (
            <div key={i} className="nx-tour-dim" style={{ left: r.x, top: r.y, width: r.w, height: r.h }} />
          ))}
          <div className="nx-tour-spot" style={{ left: spot.x, top: spot.y, width: spot.w, height: spot.h }} />
        </>
      ) : (
        <div className="nx-tour-dim" style={{ inset: 0 }} />
      )}
      <div className="nx-tour-card" style={dialog ? { left: 720 - 210, top: 300, width: 420 } : { left: pos.left, top: pos.top, width: CARD_W }}>
        <div className="nx-tour-eyebrow">{dialog ? "TUTORIAL MODE" : `STEP ${step} OF ${last}`}</div>
        <div className="nx-tour-title">{s.title}</div>
        <div className="nx-tour-body">{s.body}</div>
        <div className="nx-tour-actions">
          {step > 0 && (
            <button type="button" className="nx-btn" onClick={() => setStep((i) => Math.max(0, i - 1))}>
              BACK
            </button>
          )}
          <button type="button" className="nx-btn nx-btn--primary" onClick={() => (step >= last ? finish(true) : setStep((i) => i + 1))}>
            {dialog ? "START THE TOUR" : step >= last ? "FINISH" : "NEXT"}
          </button>
          <button type="button" className="nx-tour-link" style={{ marginLeft: "auto" }} onClick={() => finish(false)}>
            {dialog ? "skip for now" : "skip"}
          </button>
        </div>
        {!dialog && (
          <div className="nx-tour-dots" aria-hidden="true">
            {TOUR_STEPS.slice(1).map((_, i) => (
              <span key={i} className={i + 1 === step ? "on" : ""} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
