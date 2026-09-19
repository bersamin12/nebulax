// 30 px scrolling advisory ticker. Lines come verbatim from frame.ticker; the severity glyph
// is resolved against frame.alerts (open episode -> crit, closed -> warn, no match -> neutral).
import { C, shortComponent, toMs } from "../lib/format.js";

function CritGlyph() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#c43d36" strokeWidth="2.4">
      <path d="M8 1.5h8L22.5 8v8L16 22.5H8L1.5 16V8z" />
      <path d="M12 7v6" />
      <path d="M12 16.5v.5" />
    </svg>
  );
}
function WarnGlyph() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#a66500" strokeWidth="2.4">
      <path d="M12 3L1.5 21h21z" />
      <path d="M12 9.5v5" />
      <path d="M12 18v.5" />
    </svg>
  );
}
function DotGlyph() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth="2.4">
      <circle cx="12" cy="12" r="4" />
    </svg>
  );
}

function classify(line, alerts) {
  const hay = String(line).toLowerCase();
  let match = null;
  for (const a of alerts || []) {
    if (!a.train_id || !hay.includes(String(a.train_id).toLowerCase())) continue;
    const short = shortComponent(a.component_id).toLowerCase();
    const cid = String(a.component_id || "").toLowerCase();
    if (hay.includes(cid) || hay.includes(short) || hay.includes(cid.replace(/_/g, " "))) {
      if (!match || toMs(a.t_start) > toMs(match.t_start)) match = a;
    }
  }
  if (match) return match.t_end ? "warn" : "crit";
  if (/\bclosed\b/.test(hay)) return "warn";
  return "none";
}

const INK = { crit: "#c43d36", warn: "#a66500", none: C.dim };
const TAG = { crit: ["P1", "#c43d36"], warn: ["P2", "#a66500"], none: ["··", C.dim2] };

function Item({ line, kind }) {
  const [tag, tagColor] = TAG[kind];
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 7, color: INK[kind] }}>
      {kind === "crit" ? <CritGlyph /> : kind === "warn" ? <WarnGlyph /> : <DotGlyph />}
      <span style={{ color: tagColor, fontWeight: 600 }}>{tag}</span>
      <span>{line}</span>
    </div>
  );
}

export default function AlertTicker({ frame }) {
  const items = (frame && frame.ticker) || [];
  const alerts = (frame && frame.alerts) || [];
  const kinds = items.map((l) => classify(l, alerts));
  const chars = items.join("").length || 1;
  const duration = Math.max(26, Math.round(chars / 6));

  const run = (key) => (
    <div className="nx-ticker-run" key={key} aria-hidden={key === "b" ? "true" : undefined}>
      {items.map((line, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 26 }}>
          <Item line={line} kind={kinds[i]} />
          <span style={{ color: C.line2 }}>|</span>
        </div>
      ))}
    </div>
  );

  return (
    <div
      style={{
        height: 30,
        flex: "none",
        display: "flex",
        alignItems: "center",
        padding: "0 20px",
        background: "#f8fafb",
        borderBottom: `1px solid ${C.line}`,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          flex: "none",
          display: "flex",
          alignItems: "center",
          gap: 6,
          paddingRight: 16,
          marginRight: 16,
          borderRight: `1px solid ${C.line}`,
          fontSize: 9.5,
          fontWeight: 600,
          letterSpacing: "0.16em",
          color: C.dim,
        }}
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 3v4" />
          <path d="M6 21h12" />
          <path d="M5 21v-7a7 7 0 0 1 14 0v7" />
        </svg>
        LIVE ADVISORIES
      </div>

      {items.length === 0 ? (
        <div className="mono" style={{ fontSize: 11, color: C.dim2 }}>
          no advisories at this point in the replay
        </div>
      ) : (
        <div className="nx-ticker-window">
          <div
            className="nx-ticker-rail mono"
            style={{ animationDuration: `${duration}s`, fontSize: 11, whiteSpace: "nowrap" }}
          >
            {run("a")}
            {run("b")}
          </div>
        </div>
      )}
    </div>
  );
}
