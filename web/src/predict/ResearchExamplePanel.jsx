import { C } from "../lib/format.js";

const W = 500;
const H = 150;

function downloadExcerpt(example) {
  const header = "position,value,alert_or_condition";
  const rows = example.points.map((p) => {
    const label = `"${String(p.label).replaceAll('"', '""')}"`;
    return `${label},${p.value},${example.condition || (p.alert ? "alert" : "normal")}`;
  });
  const url = URL.createObjectURL(new Blob([[header, ...rows].join("\n") + "\n"], { type: "text/csv" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${example.id}_research_excerpt.csv`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function ExampleChart({ example, index }) {
  const values = example.points.map((p) => Number(p.value));
  const threshold = Number.isFinite(example.threshold) ? example.threshold : null;
  const lo = Math.min(...values, threshold ?? Infinity);
  const hi = Math.max(...values, threshold ?? -Infinity);
  const pad = Math.max((hi - lo) * 0.1, 0.1);
  const min = lo - pad;
  const max = hi + pad;
  const x = (i) => 12 + (i / Math.max(values.length - 1, 1)) * (W - 24);
  const y = (v) => 10 + (1 - (v - min) / (max - min)) * (H - 30);
  const path = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const point = values[Math.min(index, values.length - 1)];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${example.measure} over the built-in excerpt`} style={{ width: "100%", height: 166, background: C.panel2, border: `1px solid ${C.line}` }}>
      {[0.25, 0.5, 0.75].map((p) => <line key={p} x1="12" x2={W - 12} y1={12 + p * (H - 34)} y2={12 + p * (H - 34)} stroke={C.grid} />)}
      {threshold !== null && <line x1="12" x2={W - 12} y1={y(threshold)} y2={y(threshold)} stroke={C.warn} strokeDasharray="5 4" />}
      <path d={path} fill="none" stroke={C.accent} strokeWidth="2.1" />
      <line x1={x(index)} x2={x(index)} y1="10" y2={H - 20} stroke={C.violet} strokeWidth="1" opacity="0.55" />
      <circle cx={x(index)} cy={y(point)} r="5" fill={C.violet} stroke="#fff" strokeWidth="2" />
      <text x="12" y={H - 5} fill={C.dim} fontSize="10">{example.points[0].label}</text>
      <text x={W - 12} y={H - 5} textAnchor="end" fill={C.dim} fontSize="10">{example.points.at(-1).label}</text>
    </svg>
  );
}

export default function ResearchExamplePanel({ info, examples, example, index, playing, onSelect, onToggle, onSeek, height }) {
  const point = example?.points[index] || null;
  const alert = example && (example.condition === "faulty" || !!point?.alert);
  return (
    <section className="nx-info-card nx-scroll" style={{ height, minHeight: 0, overflowY: "auto", padding: 18 }}>
      <span className="nx-eyebrow">BUILT-IN DATASET EXCERPT · 3D ANIMATION</span>
      <h2>{info.dataset}</h2>
      <p>Choose a short recorded excerpt. The train model highlights the corresponding component as the timeline plays.</p>
      <div style={{ display: "flex", gap: 7, flexWrap: "wrap", marginBottom: 14 }}>
        {(examples || []).map((item) => <button key={item.id} type="button" className="nx-btn" aria-pressed={item.id === example?.id} style={{ height: 27, background: item.id === example?.id ? C.accentBg : C.panel2, borderColor: item.id === example?.id ? C.accent : C.line2, color: item.id === example?.id ? C.accentBright : C.text2 }} onClick={() => onSelect(item.id)}>{item.label.toUpperCase()}</button>)}
      </div>
      {example && point && <>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, marginBottom: 7 }}>
          <span style={{ color: C.text2, fontSize: 11 }}>{example.measure}</span>
          <strong className="mono" style={{ color: alert ? C.crit : C.ok, fontSize: 11 }}>{alert ? "ALERT" : "NORMAL"} · {point.value} {example.unit}</strong>
        </div>
        <ExampleChart example={example} index={index} />
        <div style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 12 }}>
          <button type="button" className="nx-btn" style={{ height: 28 }} onClick={onToggle}>{playing ? "PAUSE" : "PLAY"}</button>
          <button type="button" className="nx-btn" style={{ height: 28 }} onClick={() => downloadExcerpt(example)}>DOWNLOAD EXCERPT CSV</button>
          <input aria-label="Example timeline" type="range" min="0" max={example.points.length - 1} value={index} onChange={(e) => onSeek(Number(e.target.value))} style={{ flex: 1, accentColor: C.accent }} />
          <span className="mono" style={{ fontSize: 10, color: C.dim }}>{index + 1}/{example.points.length}</span>
        </div>
        <div className="nx-dataset-fact" style={{ marginTop: 13 }}><span>Source excerpt</span><strong>{example.record}</strong></div>
        <div className="nx-dataset-fact"><span>Position on train</span><strong>{info.location}</strong></div>
        <p style={{ marginBottom: 0, color: C.dim }}>This short excerpt is bundled for display. Its colours follow the recorded condition or the existing MetroPT-3 score; it does not create a PS3 submission prediction.</p>
      </>}
    </section>
  );
}
