// Transport bar: the console's bottom 144 px zone (docs/design/Main.dc.html).
// play/pause + step, a scrub bar over the sim clock with episode marks, the replay clock,
// speed presets, the fleet KPI strip and the inject drawer.
//
//   <TransportBar replay={useReplay()} frame={frame} selected={sel} onInjected={fn} height={144} />
import { useCallback, useMemo, useRef, useState } from "react";
import Kpis from "./Kpis.jsx";
import InjectDrawer from "./InjectDrawer.jsx";
import {
  CLOCK_END_MS,
  CLOCK_START_MS,
  SPEED_PRESETS,
  isoOf,
  useReplay,
} from "../../state/replayStore.js";
import "./transport.css";

const HOUR = 3600 * 1000;
const DAY = 24 * HOUR;
const SPAN = CLOCK_END_MS - CLOCK_START_MS;
const MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

const pad = (n) => String(n).padStart(2, "0");

function shortDate(ms) {
  const d = new Date(ms);
  return `${pad(d.getUTCDate())} ${MON[d.getUTCMonth()]}`;
}

function fullClock(ms) {
  const d = new Date(ms);
  return `${pad(d.getUTCDate())} ${MON[d.getUTCMonth()]} ${d.getUTCFullYear()} ${pad(d.getUTCHours())}:${pad(
    d.getUTCMinutes(),
  )}:${pad(d.getUTCSeconds())}`;
}

const pct = (ms) => Math.min(100, Math.max(0, ((ms - CLOCK_START_MS) / SPAN) * 100));

/** Episode marks: one tick per alert, crit while the episode is still open. */
function marksOf(alerts) {
  const out = [];
  for (const a of Array.isArray(alerts) ? alerts : []) {
    const ms = Date.parse(a?.t_start);
    if (!Number.isFinite(ms)) continue;
    // The frame's alert list also carries MetroPT-3's 22 episodes, which are on the 2020 clock.
    // They would all pile up on the first pixel of the scrub, so only episodes inside the
    // simulated month get a mark (contract section 1: the replay never mixes the two clocks).
    if (ms < CLOCK_START_MS || ms > CLOCK_END_MS) continue;
    out.push({
      id: a.episode_id ?? `${a.train_id}-${a.component_id}-${ms}`,
      ms,
      crit: !a.t_end,
      label: `${shortDate(ms)} ${a.train_id ?? ""}`.trim(),
      title: `${a.train_id ?? ""} ${a.component_id ?? ""} · ${a.fault_type ?? "episode"} · ${
        a.t_end ? "closed" : "open"
      } · peak ${Number(a.peak_score ?? 0).toFixed(2)}`,
    });
  }
  out.sort((x, y) => x.ms - y.ms);
  // Label at most three marks and never two that would overlap. A "03 SEP T09" label is ~54 px
  // on a ~280 px track (19 %) and is centred on its tick, so neighbours need ~20 % of clear
  // track and the outer 15 % is left to the fixed start / end labels.
  let lastLabelled = -Infinity;
  let labelled = 0;
  for (const m of out) {
    const p = pct(m.ms);
    m.showLabel = labelled < 3 && p - lastLabelled > 20 && p > 15 && p < 85;
    if (m.showLabel) {
      lastLabelled = p;
      labelled += 1;
    }
  }
  return out;
}

const CONN = {
  ws: { cls: "is-live", text: "LIVE · WS" },
  rest: { cls: "is-rest", text: "REST POLL" },
  aux: { cls: "is-rest", text: "MP3 · OWN CLOCK" },
  offline: { cls: "is-off", text: "OFFLINE · MOCK FRAME" },
  connecting: { cls: "", text: "CONNECTING…" },
};

export default function TransportBar({ replay, frame, selected, onInjected, onExampleSelected, systemStats, height = 144 }) {
  // The shell passes the store down, but the bar also works standalone (dev entry).
  const own = useReplay();
  const rp = replay ?? own;
  const f = frame ?? rp.frame;

  const [drawerOpen, setDrawerOpen] = useState(true);
  const scrubRef = useRef(null);
  const dragging = useRef(false);

  // rawMs is what the frame says (MP3 carries its own 2020 clock); tsMs is the scrub position,
  // which is always inside the sim month the bar draws.
  const rawMs = useMemo(() => {
    const ms = Date.parse(rp.ts ?? f?.ts);
    return Number.isFinite(ms) ? ms : CLOCK_START_MS;
  }, [rp.ts, f?.ts]);
  const tsMs = Math.min(CLOCK_END_MS, Math.max(CLOCK_START_MS, rawMs));

  const marks = useMemo(() => marksOf(f?.alerts), [f?.alerts]);
  const conn = CONN[rp.mode] ?? CONN.connecting;

  const seekAt = useCallback(
    (clientX) => {
      const el = scrubRef.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      if (r.width <= 0) return;
      const frac = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
      rp.seek(isoOf(CLOCK_START_MS + frac * SPAN));
    },
    [rp],
  );

  const onPointerDown = (e) => {
    dragging.current = true;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    seekAt(e.clientX);
  };
  const onPointerMove = (e) => {
    if (dragging.current) seekAt(e.clientX);
  };
  const onPointerUp = (e) => {
    dragging.current = false;
    e.currentTarget.releasePointerCapture?.(e.pointerId);
  };

  const onKeyDown = (e) => {
    const step = e.shiftKey ? DAY : HOUR;
    if (e.key === "ArrowLeft") rp.seek(isoOf(tsMs - step));
    else if (e.key === "ArrowRight") rp.seek(isoOf(tsMs + step));
    else if (e.key === "Home") rp.seek(isoOf(CLOCK_START_MS));
    else if (e.key === "End") rp.seek(isoOf(CLOCK_END_MS));
    else if (e.key === " " || e.key === "Enter") rp.toggle();
    else return;
    e.preventDefault();
  };

  return (
    <div className="tb" style={{ height }}>
      <div className="tb-main">
        <div className="tb-deck">
          <div className="tb-keys">
            <button
              type="button"
              className="tb-key"
              onClick={() => rp.seek(isoOf(tsMs - DAY))}
              title="Back one simulated day"
              aria-label="Back one simulated day"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9">
                <path d="M18 5v14" /><path d="M16 12L7 5v14z" />
              </svg>
            </button>
            <button
              type="button"
              className={"tb-key is-wide" + (rp.playing ? " is-on" : "")}
              onClick={() => rp.toggle()}
              title={rp.playing ? "Pause replay" : "Play replay"}
              aria-label={rp.playing ? "Pause replay" : "Play replay"}
              aria-pressed={rp.playing}
            >
              {rp.playing ? (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M9 5v14" /><path d="M15 5v14" />
                </svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                  <path d="M8 5l11 7-11 7z" />
                </svg>
              )}
            </button>
            <button
              type="button"
              className="tb-key"
              onClick={() => rp.seek(isoOf(tsMs + DAY))}
              title="Forward one simulated day"
              aria-label="Forward one simulated day"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9">
                <path d="M6 5v14" /><path d="M8 12l9-7v14z" />
              </svg>
            </button>
          </div>

          <div className="tb-scrubwrap">
            <div
              ref={scrubRef}
              className="tb-scrub"
              role="slider"
              tabIndex={0}
              aria-label="Replay position over the simulated month"
              aria-valuemin={CLOCK_START_MS}
              aria-valuemax={CLOCK_END_MS}
              aria-valuenow={tsMs}
              aria-valuetext={`${fullClock(tsMs)} UTC`}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
              onKeyDown={onKeyDown}
            >
              <div className="tb-scrub-track" />
              <div className="tb-scrub-fill" style={{ width: `${pct(tsMs)}%` }} />
              {marks.map((m) => (
                <div
                  key={m.id}
                  className={"tb-tick" + (m.crit ? " is-crit" : "")}
                  style={{ left: `${pct(m.ms)}%` }}
                  title={m.title}
                />
              ))}
              <div className="tb-head" style={{ left: `${pct(tsMs)}%` }} />
            </div>
            <div className="tb-scrub-labels">
              <span className="tb-lab-start">{shortDate(CLOCK_START_MS)}</span>
              {marks
                .filter((m) => m.showLabel)
                .map((m) => (
                  <span
                    key={m.id}
                    className={"tb-lab-tick" + (m.crit ? " is-crit" : "")}
                    style={{ left: `${pct(m.ms)}%` }}
                  >
                    {m.label}
                  </span>
                ))}
              <span className="tb-lab-end">{shortDate(CLOCK_END_MS)}</span>
            </div>
          </div>

          <div className="tb-stack">
            <span className="tb-label">Replay clock</span>
            <span className="tb-clock">
              {fullClock(rawMs)} <span className="tb-clock-zone">UTC</span>
            </span>
            <span className={"tb-conn " + conn.cls} title={rp.error || conn.text}>
              <span className="tb-dot" />
              {conn.text}
            </span>
          </div>

          <div className="tb-stack">
            <span className="tb-label">Replay speed</span>
            <div className="tb-chips">
              {SPEED_PRESETS.map((p) => (
                <button
                  key={p.value}
                  type="button"
                  className={"tb-chip" + (Number(rp.speed) === p.value ? " is-on" : "")}
                  onClick={() => rp.setSpeed(p.value)}
                  title={p.title}
                  aria-pressed={Number(rp.speed) === p.value}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <Kpis systemStats={systemStats} />
      </div>

      <InjectDrawer
        replay={rp}
        frame={f}
        selected={selected}
        onInjected={onInjected}
        onExampleSelected={onExampleSelected}
        open={drawerOpen}
        onToggle={() => setDrawerOpen((v) => !v)}
      />
    </div>
  );
}
