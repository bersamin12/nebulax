// 470 px detail panel: title, default signal chart, score trace vs threshold, top signals,
// the four CBM steps and the recommended action. Loads are sequenced (stale replies dropped)
// and throttled to at most one request burst per 2 s of wall clock while the replay plays.
import { useEffect, useRef, useState } from "react";
import { getCycle, getScores, getSeries, postAdvisory } from "../api.js";
import ScoreTrace from "../charts/ScoreTrace.jsx";
import { CycleWaveform, TimeSeriesChart } from "../charts/SignalChart.jsx";
import {
  C,
  DAY,
  DEFAULT_SIGNAL,
  componentKind,
  componentLabel,
  fmtDayHm,
  fmtHours,
  fmtNum,
  fmtScore,
  fmtThreshold,
  fmtZ,
  findAlert,
  findComponent,
  health,
  toIso,
  toMs,
} from "../lib/format.js";

const CHART_W = 448;
// The panel is a fixed 452 px and the four CBM steps are the point of it: when step 3 wraps to
// three lines the block needed 22 px more than it had and pushed the ASK ADVISORY button below
// the fold. The two charts give that back (each keeps ~46 px of plot above its axis labels).
const SIG_H = 73;
const SCORE_H = 67;
const SCORE_SPAN = 3 * DAY;

function useThrottled(value, waitMs) {
  const [out, setOut] = useState(value);
  const last = useRef(0);
  useEffect(() => {
    if (value === undefined || value === null) return undefined;
    const now = Date.now();
    const wait = waitMs - (now - last.current);
    if (wait <= 0) {
      last.current = now;
      setOut(value);
      return undefined;
    }
    const id = setTimeout(() => {
      last.current = Date.now();
      setOut(value);
    }, wait);
    return () => clearTimeout(id);
  }, [value, waitMs]);
  return out;
}

function Caption({ left, right, rightColor }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        justifyContent: "space-between",
        gap: 8,
        height: 11,
        fontSize: 8.5,
        letterSpacing: "0.1em",
        color: C.dim,
        whiteSpace: "nowrap",
      }}
    >
      <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{left}</span>
      <span className="mono" style={{ color: rightColor || C.dim2, flex: "none" }}>
        {right}
      </span>
    </div>
  );
}

function Step({ n, label, children }) {
  return (
    <div style={{ display: "flex", gap: 8 }}>
      <span
        className="mono"
        style={{
          flex: "none",
          width: 84,
          fontSize: 8.5,
          letterSpacing: "0.08em",
          color: C.accent,
          paddingTop: 1,
        }}
      >
        {n} {label}
      </span>
      <span style={{ fontSize: 10.5, lineHeight: 1.32, color: C.text2 }}>{children}</span>
    </div>
  );
}

function EmptyPanel({ hasFrame }) {
  return (
    <div
      style={{
        flex: "1 1 auto",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 10,
        color: C.dim2,
        textAlign: "center",
        padding: "0 30px",
      }}
    >
      <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke={C.line2} strokeWidth="1.4">
        <path d="M3 8h18" />
        <rect x="3" y="4" width="18" height="16" />
        <path d="M8 13h.01" />
        <path d="M16 13h.01" />
      </svg>
      <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.14em", color: C.dim }}>
        SELECT A COMPONENT
      </div>
      <div style={{ fontSize: 10, lineHeight: 1.45 }}>
        {hasFrame
          ? "Click a car cell in a subsystem lane, or a door / axle box / APU on the schematic above. Grey cells have no telemetry in this fleet."
          : "Waiting for the first replay frame."}
      </div>
    </div>
  );
}

function Ps3DetailView({ detail, selected, playing }) {
  const { task, fileId, numbers, frame, frameIdx, totalFrames, health: hKey, label } = detail;
  const h = health(hKey || "ok");
  const taskTitle = {
    door: "DOOR OPERATION · LTA PS3",
    rail: "RAIL CORRUGATION · LTA PS3",
    acv: "ACV REFRIGERANT LEAK · LTA PS3",
    shm: "SHM FATIGUE DAMAGE · LTA PS3",
  }[task] || "PS3 PREDICTION";

  const compTitle =
    selected.subsystem === "rail"
      ? `RAIL · ${selected.component_id === "rail_II" ? "SIDE II" : "SIDE I"}`
      : selected.subsystem === "acv"
        ? `CAR ${selected.car} · AIR CON UNIT`
        : selected.subsystem === "shm"
          ? `CAR ${selected.car} · ${selected.component_id.replace("_", " ").toUpperCase()}`
          : `CAR ${selected.car} · ${selected.component_id.replace("_", " ").toUpperCase()}`;

  const unit = frame?.t_unit || (task === "acv" ? "h" : "s");

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 8,
        background: C.panel,
        border: `1px solid ${C.line}`,
        borderTop: `3px solid ${h.color}`,
        padding: "8px 10px",
        minWidth: 0,
        height: 452,
        overflowY: "auto",
        boxSizing: "border-box",
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 8 }}>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: "0.08em" }}>{compTitle}</div>
          <div style={{ fontSize: 9, letterSpacing: "0.12em", color: C.dim, marginTop: 2 }}>{taskTitle}</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6, flex: "none" }}>
          <span
            className="nx-tag"
            style={{ background: h.bg, color: h.color, border: `1px solid ${h.border}`, fontWeight: 600 }}
          >
            {h.word}
          </span>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "6px 8px",
          background: C.panel2,
          border: `1px solid ${C.line}`,
          fontSize: 10,
        }}
      >
        <span style={{ color: C.dim }}>
          FILE: <b className="mono" style={{ color: C.text }}>{fileId}</b>
        </span>
        <span className="mono" style={{ color: C.accent }}>
          FRAME {frameIdx + 1} / {totalFrames} {frame?.t != null ? `· ${Number(frame.t).toFixed(1)} ${unit}` : ""}
        </span>
      </div>

      {label ? (
        <div style={{ fontSize: 9.5, color: C.text2, fontStyle: "italic", padding: "0 2px" }}>
          {label}
        </div>
      ) : null}

      <div style={{ fontSize: 9.5, fontWeight: 600, letterSpacing: "0.1em", color: C.accent, marginTop: 4 }}>
        CURRENT FRAME METRICS
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 6,
          background: C.panel2,
          border: `1px solid ${C.line}`,
          padding: 8,
        }}
      >
        {numbers && Object.keys(numbers).length > 0 ? (
          Object.entries(numbers).map(([k, v]) => (
            <div key={k} style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <span style={{ fontSize: 8.5, color: C.dim, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                {k.replace(/_/g, " ")}
              </span>
              <span className="mono" style={{ fontSize: 13, fontWeight: 600, color: C.text }}>
                {typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(k.includes("damage") ? 3 : 2)) : String(v)}
              </span>
            </div>
          ))
        ) : (
          <div style={{ color: C.dim2, fontSize: 10, gridColumn: "span 2" }}>No intermediate metrics for this frame</div>
        )}
      </div>

      {frame?.rows?.[0] ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 3, marginTop: 2 }}>
          <span style={{ fontSize: 8.5, color: C.dim, letterSpacing: "0.08em" }}>SUBMITTED PREDICTION (CURRENT FRAME)</span>
          <div className="mono" style={{ fontSize: 10.5, color: h.color, background: C.panel2, padding: "5px 8px", border: `1px solid ${C.line}`, overflowX: "auto" }}>
            {JSON.stringify(frame.rows[0])}
          </div>
        </div>
      ) : null}

      <div style={{ marginTop: "auto", fontSize: 9, color: C.dim2, fontStyle: "italic", borderTop: `1px solid ${C.line}`, paddingTop: 6 }}>
        {task === "rail" || task === "shm"
          ? "demo ordering (files are unordered) · Miner / spectral accumulation ends on submitted value"
          : task === "door"
            ? "causal preview · final frame = submitted row"
            : "prefix ranking · compares each car with its 7 peers"}
      </div>
    </div>
  );
}

export default function DetailPanel({ frame, train, selected, playing, reloadKey, ps3Details = null }) {
  if (ps3Details) {
    return <Ps3DetailView detail={ps3Details} selected={selected} playing={playing} />;
  }
  const comp = findComponent(train, selected);
  const alert = findAlert(frame, selected);
  const ts = frame && frame.ts;
  const loadTs = useThrottled(ts, 2000);

  const [sig, setSig] = useState({ state: "idle" });
  const [scores, setScores] = useState({ state: "idle" });
  const [adv, setAdv] = useState({ state: "idle" });
  const seq = useRef(0);
  const lastSel = useRef("");

  const selKey = selected
    ? `${selected.train_id}|${selected.component_id}|${selected.car}|${selected.subsystem}`
    : "";

  useEffect(() => {
    setAdv({ state: "idle" });
  }, [alert && alert.episode_id]);

  useEffect(() => {
    if (!selected || !loadTs) {
      setSig({ state: "idle" });
      setScores({ state: "idle" });
      return undefined;
    }
    const my = ++seq.current;
    let live = true;
    const { train_id, component_id, car, subsystem } = selected;
    const spec = DEFAULT_SIGNAL[subsystem] || DEFAULT_SIGNAL.pneumatic;
    const end = toMs(loadTs);

    // keep the current traces on screen while refreshing the same component, so the
    // charts do not blink every 2 s; drop them as soon as the selection changes.
    const keep = lastSel.current === selKey;
    lastSel.current = selKey;
    setSig((s) => ({ state: "loading", data: keep ? s.data : undefined }));
    setScores((s) => ({ state: "loading", data: keep ? s.data : undefined }));

    const sigReq =
      subsystem === "door"
        ? getCycle(train_id, component_id, { ts: loadTs, car })
        : getSeries(train_id, component_id, {
            signal: spec.signal,
            from: toIso(end - spec.span),
            to: loadTs,
            car,
            max_points: 2000,
          });

    sigReq
      .then((d) => live && seq.current === my && setSig({ state: "ok", data: d }))
      .catch((e) => live && seq.current === my && setSig({ state: "error", error: String(e && e.message ? e.message : e) }));

    getScores(train_id, component_id, { from: toIso(end - SCORE_SPAN), to: loadTs, car })
      .then((d) => live && seq.current === my && setScores({ state: "ok", data: d }))
      .catch((e) => live && seq.current === my && setScores({ state: "error", error: String(e && e.message ? e.message : e) }));

    return () => {
      live = false;
    };
  }, [selKey, loadTs, reloadKey]);

  if (!selected || !comp) {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          background: C.panel,
          border: `1px solid ${C.line}`,
          borderTop: `3px solid ${C.line2}`,
          padding: "8px 10px",
          minWidth: 0,
          overflow: "hidden",
        }}
      >
        <EmptyPanel hasFrame={Boolean(frame)} />
      </div>
    );
  }

  const h = health(comp.health);
  const spec = DEFAULT_SIGNAL[comp.subsystem] || DEFAULT_SIGNAL.pneumatic;
  const advisory = (adv.state === "ok" && adv.data) || (alert && alert.advisory) || null;
  const signals = (comp.top_signals || []).slice(0, 4);
  const openEpisode = alert && !alert.t_end;
  const durH = alert ? (toMs(alert.t_end || ts) - toMs(alert.t_start)) / 3600000 : null;

  const scoreData = scores.data || null;
  const scorePoints = (scoreData && scoreData.points) || [];
  const scoreThreshold = scoreData && Number.isFinite(scoreData.threshold)
    ? scoreData.threshold
    : comp.threshold;

  const signalNote =
    sig.state === "error"
      ? "API unavailable"
      : sig.state === "loading"
      ? (sig.data ? "refreshing…" : "loading…")
      : comp.subsystem === "door"
      ? "nearest stored cycle"
      : `last ${Math.round(spec.span / 3600000)} h`;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 5,
        background: C.panel,
        border: `1px solid ${C.line}`,
        borderTop: `3px solid ${h.color}`,
        padding: "8px 10px",
        minWidth: 0,
        overflow: "hidden",
      }}
    >
      {/* ---- title ---- */}
      <div
        style={{
          height: 28,
          flex: "none",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 1, minWidth: 0 }}>
          <div
            className="mono"
            style={{ fontSize: 12.5, fontWeight: 600, letterSpacing: "0.05em", whiteSpace: "nowrap" }}
          >
            {selected.train_id} / CAR {comp.car === 0 ? "3 (UNIT)" : comp.car} /{" "}
            {componentLabel(comp.component_id)}
          </div>
          <div
            style={{
              fontSize: 9,
              letterSpacing: "0.1em",
              color: C.dim,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {componentKind(comp)} &middot; {comp.model || "no model"}
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flex: "none" }}>
          {comp.stale && (
            <span
              className="mono"
              style={{ fontSize: 8.5, color: C.dim2, letterSpacing: "0.08em" }}
              title="latest scored row is older than the freshness window"
            >
              STALE
            </span>
          )}
          <div
            style={{
              padding: "3px 8px",
              background: h.color,
              color: "#ffffff",
              fontSize: 9.5,
              fontWeight: 700,
              letterSpacing: "0.11em",
            }}
          >
            {h.word}
          </div>
          <div
            className="mono"
            style={{ fontSize: 22, fontWeight: 600, color: h.color, lineHeight: 1 }}
          >
            {fmtScore(comp.score)}
            <span style={{ fontSize: 10, color: C.dim }}>/{fmtThreshold(comp.threshold)}</span>
          </div>
        </div>
      </div>

      {/* ---- default signal ---- */}
      <div style={{ flex: "none", display: "flex", flexDirection: "column" }}>
        <Caption
          left={
            `${spec.label}${spec.unit ? ` · ${spec.unit}` : ""}` +
            (comp.subsystem === "door" ? "  ·  grey position, violet reference" : "")
          }
          right={signalNote}
          rightColor={sig.state === "error" ? "#a66500" : C.dim2}
        />
        {comp.subsystem === "door" ? (
          <CycleWaveform
            cycle={sig.data || null}
            signal={spec.signal}
            width={CHART_W}
            height={SIG_H}
            empty={
              sig.state === "error"
                ? "cycle waveform unavailable (API not reachable)"
                : sig.state === "loading"
                ? "loading cycle…"
                : "no telemetry stored for this cycle"
            }
          />
        ) : (
          <TimeSeriesChart
            series={sig.data || null}
            width={CHART_W}
            height={SIG_H}
            empty={
              sig.state === "error"
                ? `${spec.signal} unavailable (API not reachable)`
                : sig.state === "loading"
                ? "loading series…"
                : "no telemetry in this window"
            }
          />
        )}
      </div>

      {/* ---- score trace ---- */}
      <div style={{ flex: "none", display: "flex", flexDirection: "column" }}>
        <Caption
          left="ANOMALY SCORE · LAST 3 DAYS · THRESHOLD DASHED"
          right={`${fmtScore(comp.score)} vs ${fmtThreshold(comp.threshold)}`}
          rightColor={h.color}
        />
        <ScoreTrace
          points={scorePoints}
          threshold={scoreThreshold}
          width={CHART_W}
          height={SCORE_H}
          empty={
            scores.state === "error"
              ? "score trace unavailable (API not reachable)"
              : scores.state === "loading"
              ? "loading scores…"
              : "no scored rows in this window"
          }
        />
      </div>

      {/* ---- top signals ---- */}
      <div style={{ flex: "none", display: "flex", flexDirection: "column", gap: 2 }}>
        <div
          className="nx-kv"
          style={{
            fontSize: 8.5,
            letterSpacing: "0.1em",
            color: C.dim2,
            borderBottom: `1px solid ${C.line}`,
            paddingBottom: 2,
          }}
        >
          <span>EVIDENCE SIGNAL</span>
          <span style={{ textAlign: "right" }}>VALUE</span>
          <span style={{ textAlign: "right" }}>ROBUST z</span>
        </div>
        {signals.length === 0 ? (
          <div className="mono" style={{ fontSize: 9.5, color: C.dim2, padding: "3px 0" }}>
            no signal attribution on this row
          </div>
        ) : (
          signals.map((s, i) => {
            const zColor = Math.abs(s.z) >= 3 ? C.crit : Math.abs(s.z) >= 2 ? C.warn : C.text;
            return (
              <div
                key={i}
                className="nx-kv mono"
                style={{ fontSize: 10, padding: "1px 0", lineHeight: "12px" }}
              >
                <span
                  style={{ color: C.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                >
                  {s.signal}
                </span>
                {/* a feature value can be as wide as the score it feeds: keep it compact */}
                <span style={{ textAlign: "right", color: C.dim }}>{fmtScore(s.value)}</span>
                <span style={{ textAlign: "right", color: zColor }}>
                  {fmtZ(s.z)}
                </span>
              </div>
            );
          })
        )}
      </div>

      {/* ---- CBM steps ---- */}
      <div
        className="nx-scroll"
        style={{
          flex: "1 1 auto",
          minHeight: 0,
          display: "flex",
          flexDirection: "column",
          gap: 5,
          borderTop: `1px solid ${C.line}`,
          paddingTop: 5,
        }}
      >
        <Step n="1" label="STATE DETECT">
          {comp.model || "model"}{" "}
          <span className="mono" style={{ color: h.color }}>
            {fmtScore(comp.score)}
          </span>{" "}
          vs FA-budget threshold <span className="mono">{fmtThreshold(comp.threshold)}</span> at{" "}
          {fmtDayHm(ts)} UTC.{" "}
          {comp.alert
            ? "Inside an alarm episode (3+ rows above)."
            : comp.score > comp.threshold
            ? "Above threshold but under 3 rows: a watch."
            : "Below threshold."}
          {comp.stale ? " Latest row is stale." : ""}
        </Step>

        <Step n="2" label="HEALTH">
          Health <span style={{ color: h.color, fontWeight: 600 }}>{h.word}</span>.{" "}
          {alert
            ? `Episode ${openEpisode ? "open since" : "ran from"} ${fmtDayHm(alert.t_start)}${
                openEpisode ? "" : ` to ${fmtDayHm(alert.t_end)}`
              }, ${fmtHours(durH)}, peak ${fmtScore(alert.peak_score)}.`
            : "No alarm episode on this component in the replay window."}
        </Step>

        <Step n="3" label="PROGNOSIS">
          {alert ? (
            <>
              Fault log labels it{" "}
              <span style={{ color: C.violetBright }}>{alert.fault_type || "unlabelled"}</span>.{" "}
              {Number.isFinite(alert.lead_to_failure_h) ? (
                <>
                  Lead to the logged failure{" "}
                  <span className="mono" style={{ color: h.color, fontWeight: 600 }}>
                    {fmtHours(alert.lead_to_failure_h)}
                  </span>
                  .
                </>
              ) : (
                "No logged failure joined, so no lead time is claimed."
              )}
            </>
          ) : (
            "Nothing to project: the detector has not opened an episode here."
          )}
        </Step>

        <Step n="4" label="ADVISORY">
          {advisory ? (
            <>
              <span style={{ color: C.text }}>{advisory.summary}</span>
              {Array.isArray(advisory.evidence) && advisory.evidence.length > 0 && (
                <span style={{ display: "block", color: C.dim, marginTop: 2 }}>
                  {advisory.evidence.slice(0, 3).map((e, i) => (
                    <span key={i} style={{ display: "block" }}>
                      &middot; {e}
                    </span>
                  ))}
                </span>
              )}
              <span
                className="mono"
                style={{ display: "block", marginTop: 2, fontSize: 8.5, color: C.dim2 }}
              >
                urgency {String(advisory.urgency || "?").toUpperCase()} &middot; confidence{" "}
                {fmtNum(advisory.confidence, 2)} &middot; source {advisory.source} (
                {advisory.model})
              </span>
            </>
          ) : adv.state === "loading" ? (
            "Asking the advisory model…"
          ) : adv.state === "error" ? (
            <span style={{ color: "#a66500" }}>Advisory unavailable: {adv.error}</span>
          ) : alert ? (
            <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ color: C.dim }}>Not requested for this episode yet.</span>
              <button
                type="button"
                className="nx-btn"
                onClick={() => {
                  setAdv({ state: "loading" });
                  postAdvisory(alert.episode_id)
                    .then((d) => setAdv({ state: "ok", data: d }))
                    .catch((e) =>
                      setAdv({ state: "error", error: String(e && e.message ? e.message : e) })
                    );
                }}
              >
                ASK ADVISORY
              </button>
            </span>
          ) : (
            "No open episode, so no advisory is requested."
          )}
        </Step>
      </div>

      {/* ---- recommended action ---- */}
      <div
        style={{
          flex: "none",
          height: 28,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
          background: alert ? h.bg : C.panel2,
          border: `1px solid ${alert ? h.border : C.line}`,
          padding: "0 9px",
        }}
      >
        <span
          style={{
            fontSize: 10,
            fontWeight: 600,
            letterSpacing: "0.08em",
            color: alert ? h.ink : C.dim2,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {advisory
            ? advisory.recommended_action
            : alert
            ? "RAISE WORK ORDER · ASK ADVISORY FOR THE RECOMMENDED ACTION"
            : "NO ACTION · COMPONENT WITHIN BUDGET"}
        </span>
        <span
          className="mono"
          style={{ fontSize: 9.5, color: alert ? h.ink : C.dim2, flex: "none" }}
        >
          {advisory
            ? String(advisory.urgency || "").toUpperCase()
            : alert
            ? fmtHours(alert.lead_to_failure_h)
            : playing
            ? "replaying"
            : "paused"}
        </span>
      </div>
    </div>
  );
}
