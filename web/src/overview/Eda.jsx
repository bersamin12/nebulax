// EDA carousel: one page per track.
// Numbers, dataset counts and reference IDs come from results/ps3, docs/ps3_*.md and
// docs/research/references.md; the markup mirrors the design canvas (docs/design/Main.dc.html).
import { useState } from "react";

export default function Eda({  }) {
  const [e, setE] = useState(0);
  return (
    <>
{/* EDA carousel */}
  <section style={{ flex: "none", padding: "8px 120px 40px", display: "flex", flexDirection: "column", gap: "20px" }}>
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      <div className="eyebrow">Exploratory data analysis</div>
      <h2 style={{ margin: "0", fontSize: "28px", fontWeight: "600" }}>What the released data look like, one track at a time</h2>
    </div>
    <div style={{ display: "flex", alignItems: "flex-end", gap: "12px" }}>
      <button type="button" className="arrow" onClick={() => setE((e + 3) % 4)} aria-label="Previous track"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M15 18l-6-6 6-6"></path></svg></button>
      <div style={{ flexGrow: "1", display: "flex", gap: "8px" }}>
        <button type="button" className={"tab" + (e === 0 ? " on" : "")} onClick={() => setE(0)}><span className="mono" style={{ fontSize: "11px", letterSpacing: ".12em", color: "#1a4fa3" }}>01 · DOOR</span><span style={{ fontSize: "14px", fontWeight: "600" }}>110 labelled cycles</span></button>
        <button type="button" className={"tab" + (e === 1 ? " on" : "")} onClick={() => setE(1)}><span className="mono" style={{ fontSize: "11px", letterSpacing: ".12em", color: "#1a4fa3" }}>02 · ACV</span><span style={{ fontSize: "14px", fontWeight: "600" }}>6 cases × 8 cars</span></button>
        <button type="button" className={"tab" + (e === 2 ? " on" : "")} onClick={() => setE(2)}><span className="mono" style={{ fontSize: "11px", letterSpacing: ".12em", color: "#1a4fa3" }}>03 · RAIL</span><span style={{ fontSize: "14px", fontWeight: "600" }}>272 one-second files</span></button>
        <button type="button" className={"tab" + (e === 3 ? " on" : "")} onClick={() => setE(3)}><span className="mono" style={{ fontSize: "11px", letterSpacing: ".12em", color: "#1a4fa3" }}>04 · SHM</span><span style={{ fontSize: "14px", fontWeight: "600" }}>64 stress records</span></button>
      </div>
      <button type="button" className="arrow" onClick={() => setE((e + 1) % 4)} aria-label="Next track"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 6l6 6-6 6"></path></svg></button>
    </div>

    {e === 0 && (<>
      <div className="card" style={{ padding: "28px 32px 32px", display: "grid", gridTemplateColumns: "1fr 440px", gap: "36px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}><div style={{ fontSize: "18px", fontWeight: "600" }}>Door · labelled cycles by class</div><div className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>n = 110</div></div>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div className="bar" style={{ gridTemplateColumns: "170px 1fr 56px" }}><div>Normal</div><div className="track" style={{ height: "16px" }}><div className="fill sel" style={{ width: "72.7%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>80</div></div>
            <div className="bar" style={{ gridTemplateColumns: "170px 1fr 56px" }}><div>Abnormal resistance</div><div className="track" style={{ height: "16px" }}><div className="fill sel" style={{ width: "27.3%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>30</div></div>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: "8px" }}><div style={{ fontSize: "18px", fontWeight: "600" }}>Middle-travel current separates the classes</div><div className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>illustrative distribution</div></div>
          <svg width="640" height="150" viewBox="0 0 640 150" style={{ width: "100%", height: "150px", background: "#f1f3f5", borderRadius: "6px" }} aria-hidden="true">
            <line x1="40" y1="120" x2="620" y2="120" stroke="#b8c4cc"></line>
            <path d="M60 120 C 120 118, 150 20, 200 20 C 250 20, 280 118, 340 120" fill="rgba(31,128,89,0.18)" stroke="#1f8059" strokeWidth="2"></path>
            <path d="M330 120 C 380 118, 410 40, 460 40 C 510 40, 540 118, 600 120" fill="rgba(196,61,54,0.16)" stroke="#c43d36" strokeWidth="2"></path>
            <line x1="345" y1="26" x2="345" y2="120" stroke="#1d2633" strokeDasharray="4 3"></line>
            <text x="351" y="34" fontFamily="IBM Plex Mono, monospace" fontSize="10" fill="#1d2633">stump on i_mid_rel → 1.0000 selection CV</text>
            <text x="200" y="138" fontFamily="IBM Plex Mono, monospace" fontSize="10" fill="#1f8059" textAnchor="middle">Normal (80)</text>
            <text x="460" y="138" fontFamily="IBM Plex Mono, monospace" fontSize="10" fill="#c43d36" textAnchor="middle">Abnormal resistance (30)</text>
            <text x="620" y="112" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#5b6673" textAnchor="end">cruise-phase current ratio →</text>
          </svg>
          <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>Abnormal cycles draw visibly more current in the cruise phase; a single decision stump on the relative middle-travel current already scores 1.0000 on selection CV, and the all-Normal chance floor is 0.7273. The 24-feature model exists for robustness and explanation, not because the boundary is hard.</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
          <div className="eyebrow">Dataset facts</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "10px" }}>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>18,036</span><span style={{ fontSize: "12px", color: "#5b6673" }}>rows in one training stream</span></div>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>50 Hz</span><span style={{ fontSize: "12px", color: "#5b6673" }}>17 released columns, ms timestamps</span></div>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>110</span><span style={{ fontSize: "12px", color: "#5b6673" }}>labelled cycles, boundary-exact segmentation</span></div>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>38</span><span style={{ fontSize: "12px", color: "#5b6673" }}>cycles inferred in the unlabelled Test stream</span></div>
          </div>
          <div className="eyebrow" style={{ marginTop: "6px" }}>What the EDA changed in the model</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "8px", fontSize: "13px", lineHeight: "1.5" }}>
            <div className="step" style={{ padding: "0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>a</div><div>Gaps &gt; 0.5 s between samples mark every training cycle, so the segmenter is a rule, not a model.</div></div>
            <div className="step" style={{ padding: "0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>b</div><div>Open and Close strokes have different current profiles: two baselines and two classifiers.</div></div>
            <div className="step" style={{ padding: "0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>c</div><div>Switch and lock columns (DCSR, DCSL, DLSR, DLSL) carry no class signal and are not selected.</div></div>
          </div>
        </div>
      </div>
    </>)}

    {e === 1 && (<>
      <div className="card" style={{ padding: "28px 32px 32px", display: "grid", gridTemplateColumns: "1fr 440px", gap: "36px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}><div style={{ fontSize: "18px", fontWeight: "600" }}>ACV · case × car coverage</div><div className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>6 cases × 8 cars · 30 s telemetry each</div></div>
          <div style={{ display: "grid", gridTemplateColumns: "80px repeat(8, minmax(0, 1fr))", gap: "6px", fontSize: "12px" }}>
            <div></div><div className="mono" style={{ textAlign: "center", color: "#5b6673" }}>01</div><div className="mono" style={{ textAlign: "center", color: "#5b6673" }}>02</div><div className="mono" style={{ textAlign: "center", color: "#5b6673" }}>03</div><div className="mono" style={{ textAlign: "center", color: "#5b6673" }}>04</div><div className="mono" style={{ textAlign: "center", color: "#5b6673" }}>05</div><div className="mono" style={{ textAlign: "center", color: "#5b6673" }}>06</div><div className="mono" style={{ textAlign: "center", color: "#5b6673" }}>07</div><div className="mono" style={{ textAlign: "center", color: "#5b6673" }}>08</div>
            <div className="mono" style={{ color: "#5b6673" }}>case 01</div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div>
            <div className="mono" style={{ color: "#5b6673" }}>case 02</div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div>
            <div className="mono" style={{ color: "#5b6673" }}>case 03</div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div>
            <div className="mono" style={{ color: "#5b6673" }}>case 04</div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", border: "1px dashed #a66500", boxSizing: "border-box", borderRadius: "4px" }}></div><div style={{ height: "30px", border: "1px dashed #a66500", boxSizing: "border-box", borderRadius: "4px" }}></div><div style={{ height: "30px", border: "1px dashed #a66500", boxSizing: "border-box", borderRadius: "4px" }}></div><div style={{ height: "30px", border: "1px dashed #a66500", boxSizing: "border-box", borderRadius: "4px" }}></div>
            <div className="mono" style={{ color: "#5b6673" }}>case 05</div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div>
            <div className="mono" style={{ color: "#5b6673" }}>case 06</div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div><div style={{ height: "30px", background: "#b3c7ea", borderRadius: "4px" }}></div>
          </div>
          <div style={{ display: "flex", gap: "16px", fontSize: "12px", color: "#5b6673" }}><span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}><span style={{ width: "12px", height: "12px", background: "#b3c7ea", borderRadius: "2px" }}></span>telemetry present</span><span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}><span style={{ width: "12px", height: "12px", border: "1px dashed #a66500", boxSizing: "border-box", borderRadius: "2px" }}></span>empty car (sorted last)</span></div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: "8px" }}><div style={{ fontSize: "18px", fontWeight: "600" }}>The leaking car runs warm against its peers</div><div className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>illustrative traces</div></div>
          <svg width="640" height="140" viewBox="0 0 640 140" style={{ width: "100%", height: "140px", background: "#f1f3f5", borderRadius: "6px" }} aria-hidden="true">
            <line x1="40" y1="110" x2="620" y2="110" stroke="#b8c4cc"></line>
            <rect x="330" y="14" width="270" height="96" fill="rgba(47,98,196,0.08)"></rect>
            <text x="465" y="26" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#1a4fa3" textAnchor="middle">hot rows (outdoor T ≥ case median)</text>
            <polyline fill="none" stroke="#8a97a3" strokeWidth="1.5" points="40,80 120,78 200,82 280,80 360,79 440,81 520,80 600,79"></polyline>
            <polyline fill="none" stroke="#8a97a3" strokeWidth="1.5" points="40,86 120,84 200,88 280,86 360,85 440,87 520,86 600,85"></polyline>
            <polyline fill="none" stroke="#c43d36" strokeWidth="2" points="40,78 120,74 200,72 280,66 360,56 440,50 520,46 600,44"></polyline>
            <text x="604" y="48" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#c43d36">car 03</text>
            <text x="604" y="83" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#5b6673">peers</text>
            <text x="40" y="128" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#5b6673">indoor temperature over the 30 s window →</text>
          </svg>
          <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>Absolute cabin temperature varies with weather and load, but all eight cars share both at every timestamp, so the peer residual removes them. One labelled leaking car per case; too few cases for any learned prior.</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
          <div className="eyebrow">Dataset facts</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "10px" }}>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>6</span><span style={{ fontSize: "12px", color: "#5b6673" }}>labelled train-day cases, one leaking car each</span></div>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>8</span><span style={{ fontSize: "12px", color: "#5b6673" }}>cars per case, four of them empty in case 04</span></div>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>3</span><span style={{ fontSize: "12px", color: "#5b6673" }}>spellings of indoor temperature across workbooks</span></div>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>1</span><span style={{ fontSize: "12px", color: "#5b6673" }}>Test workbook, ranked in about 5 s</span></div>
          </div>
          <div className="eyebrow" style={{ marginTop: "6px" }}>What the EDA changed in the model</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "8px", fontSize: "13px", lineHeight: "1.5" }}>
            <div className="step" style={{ padding: "0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>a</div><div>Column discovery by pattern, because workbook width and parameter names differ per case.</div></div>
            <div className="step" style={{ padding: "0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>b</div><div>Cooling-mode and validity filters, because heating or invalid rows invert the temperature sign.</div></div>
            <div className="step" style={{ padding: "0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>c</div><div>A pre-registered rule instead of a classifier, and post-hoc 1.0000 variants left unpromoted.</div></div>
          </div>
        </div>
      </div>
    </>)}

    {e === 2 && (<>
      <div className="card" style={{ padding: "28px 32px 32px", display: "grid", gridTemplateColumns: "1fr 440px", gap: "36px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}><div style={{ fontSize: "18px", fontWeight: "600" }}>Rail · training files by class</div><div className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>n = 272 · 1 s @ 10 kHz · 129 columns</div></div>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div className="bar" style={{ gridTemplateColumns: "170px 1fr 56px" }}><div>Normal</div><div className="track" style={{ height: "16px" }}><div className="fill sel" style={{ width: "86%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>234</div></div>
            <div className="bar" style={{ gridTemplateColumns: "170px 1fr 56px" }}><div>Side I</div><div className="track" style={{ height: "16px" }}><div className="fill sel" style={{ width: "5.1%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>14</div></div>
            <div className="bar" style={{ gridTemplateColumns: "170px 1fr 56px" }}><div>Side II</div><div className="track" style={{ height: "16px" }}><div className="fill sel" style={{ width: "8.8%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>24</div></div>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: "8px" }}><div style={{ fontSize: "18px", fontWeight: "600" }}>Speed confounds the label</div><div className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>illustrative scatter</div></div>
          <svg width="640" height="150" viewBox="0 0 640 150" style={{ width: "100%", height: "150px", background: "#f1f3f5", borderRadius: "6px" }} aria-hidden="true">
            <line x1="40" y1="120" x2="620" y2="120" stroke="#b8c4cc"></line>
            <rect x="40" y="16" width="120" height="104" fill="rgba(166,101,0,0.10)"></rect>
            <text x="100" y="30" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#a66500" textAnchor="middle">&lt; 20 km/h: all Normal</text>
            <g fill="#8a97a3"><circle cx="60" cy="100" r="4"></circle><circle cx="80" cy="92" r="4"></circle><circle cx="95" cy="104" r="4"></circle><circle cx="120" cy="96" r="4"></circle><circle cx="140" cy="88" r="4"></circle><circle cx="200" cy="90" r="4"></circle><circle cx="240" cy="84" r="4"></circle><circle cx="280" cy="92" r="4"></circle><circle cx="320" cy="80" r="4"></circle><circle cx="360" cy="86" r="4"></circle><circle cx="400" cy="78" r="4"></circle><circle cx="440" cy="84" r="4"></circle><circle cx="480" cy="74" r="4"></circle><circle cx="520" cy="80" r="4"></circle><circle cx="560" cy="70" r="4"></circle><circle cx="590" cy="76" r="4"></circle></g>
            <g fill="#c43d36"><circle cx="300" cy="50" r="4"></circle><circle cx="380" cy="42" r="4"></circle><circle cx="450" cy="46" r="4"></circle><circle cx="500" cy="36" r="4"></circle><circle cx="540" cy="40" r="4"></circle><circle cx="600" cy="30" r="4"></circle></g>
            <g fill="#1f2374"><circle cx="260" cy="58" r="4"></circle><circle cx="340" cy="52" r="4"></circle><circle cx="420" cy="44" r="4"></circle><circle cx="470" cy="50" r="4"></circle><circle cx="580" cy="38" r="4"></circle></g>
            <text x="40" y="138" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#5b6673">speed from tachometer →</text>
            <text x="620" y="138" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#5b6673" textAnchor="end">axle-box RMS ↑</text>
            <g fontFamily="IBM Plex Mono, monospace" fontSize="9"><circle cx="420" cy="132" r="4" fill="#8a97a3"></circle><text x="428" y="135" fill="#5b6673">Normal</text><circle cx="490" cy="132" r="4" fill="#c43d36"></circle><text x="498" y="135" fill="#5b6673">Side I</text><circle cx="550" cy="132" r="4" fill="#1f2374"></circle><text x="558" y="135" fill="#5b6673">Side II</text></g>
          </svg>
          <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>Severe imbalance (14 Side I) drives the mirror augmentation and the ± 0.13 fold SD. Every low-speed recording is Normal, hence the flagged <span className="mono" style={{ fontSize: "12px" }}>speed &lt; 20 km/h</span> shortcut; a leave-speed-range-out split scores 0.7409 and a speed-matched subset cannot prove the confound is gone. Duplicate candidates exist and are grouped before splitting.</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
          <div className="eyebrow">Dataset facts</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "10px" }}>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>272</span><span style={{ fontSize: "12px", color: "#5b6673" }}>training files, 234 / 14 / 24 by class</span></div>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>10 kHz</span><span style={{ fontSize: "12px", color: "#5b6673" }}>about one second per file, 10,000 rows</span></div>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>129</span><span style={{ fontSize: "12px", color: "#5b6673" }}>columns: 1 tachometer + 64 vibration + 64 shock</span></div>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>68</span><span style={{ fontSize: "12px", color: "#5b6673" }}>Test files, all predicted in about 30 s</span></div>
          </div>
          <div className="eyebrow" style={{ marginTop: "6px" }}>What the EDA changed in the model</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "8px", fontSize: "13px", lineHeight: "1.5" }}>
            <div className="step" style={{ padding: "0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>a</div><div>Left / right mirroring with labels swapped, to double the 14-file Side I class honestly.</div></div>
            <div className="step" style={{ padding: "0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>b</div><div>Shock channels dropped: they add impulses, not the sustained side-wide pattern of corrugation.</div></div>
            <div className="step" style={{ padding: "0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>c</div><div>Extra stress splits (contiguous filename, held-out speed range) reported next to the headline.</div></div>
          </div>
        </div>
      </div>
    </>)}

    {e === 3 && (<>
      <div className="card" style={{ padding: "28px 32px 32px", display: "grid", gridTemplateColumns: "1fr 440px", gap: "36px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}><div style={{ fontSize: "18px", fontWeight: "600" }}>SHM · label range across the 64 records</div><div className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>cumulative damage, 0 to 1</div></div>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <div style={{ position: "relative", height: "16px", background: "#eceff2", borderRadius: "4px" }}><div style={{ position: "absolute", left: "2.9%", width: "89.9%", height: "100%", background: "#2f62c4", borderRadius: "4px" }}></div></div>
            <div style={{ position: "relative", height: "16px" }} className="mono"><span style={{ position: "absolute", left: "2.9%", fontSize: "11px" }}>0.029</span><span style={{ position: "absolute", left: "92.8%", transform: "translateX(-100%)", fontSize: "11px" }}>0.928</span></div>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: "8px" }}><div style={{ fontSize: "18px", fontWeight: "600" }}>Rainflow with m = 5 reproduces the labels</div><div className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>in-sample diagnostic, MAPE 0.0254</div></div>
          <svg width="640" height="160" viewBox="0 0 640 160" style={{ width: "100%", height: "160px", background: "#f1f3f5", borderRadius: "6px" }} aria-hidden="true">
            <line x1="60" y1="130" x2="600" y2="130" stroke="#b8c4cc"></line><line x1="60" y1="130" x2="60" y2="20" stroke="#b8c4cc"></line>
            <line x1="60" y1="130" x2="600" y2="22" stroke="#1d2633" strokeDasharray="4 3"></line>
            <g fill="#2f62c4"><circle cx="80" cy="127" r="4"></circle><circle cx="110" cy="120" r="4"></circle><circle cx="140" cy="115" r="4"></circle><circle cx="175" cy="107" r="4"></circle><circle cx="200" cy="103" r="4"></circle><circle cx="230" cy="95" r="4"></circle><circle cx="260" cy="91" r="4"></circle><circle cx="290" cy="84" r="4"></circle><circle cx="320" cy="80" r="4"></circle><circle cx="350" cy="71" r="4"></circle><circle cx="380" cy="68" r="4"></circle><circle cx="410" cy="60" r="4"></circle><circle cx="440" cy="56" r="4"></circle><circle cx="470" cy="47" r="4"></circle><circle cx="500" cy="44" r="4"></circle><circle cx="530" cy="36" r="4"></circle><circle cx="560" cy="32" r="4"></circle><circle cx="590" cy="25" r="4"></circle></g>
            <text x="64" y="16" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#5b6673">scaled rainflow Miner sum, m = 5 ↑</text>
            <text x="600" y="146" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#5b6673" textAnchor="end">released damage label →</text>
            <text x="380" y="118" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#1d2633">y = x after one global scale</text>
          </svg>
          <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>Labels span almost the whole unit interval, so a relative-error metric punishes small-damage files most; that is why the model works in log space. An all-data diagnostic shows 4-point rainflow with half-cycle residue and m = 5 reproduces the labels after one global scale, i.e. the labels behave like Miner sums. This diagnostic is in-sample and is not the shipped estimator.</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
          <div className="eyebrow">Dataset facts</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "10px" }}>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>64</span><span style={{ fontSize: "12px", color: "#5b6673" }}>single-channel stress records with one label each</span></div>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>581,120</span><span style={{ fontSize: "12px", color: "#5b6673" }}>samples per record, no header, no timestamp</span></div>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>none</span><span style={{ fontSize: "12px", color: "#5b6673" }}>sample rate supplied: spectra in cycles per sample</span></div>
            <div className="fact"><span className="mono" style={{ fontSize: "24px", fontWeight: "500" }}>16</span><span style={{ fontSize: "12px", color: "#5b6673" }}>Test files, all predicted in about 6 s</span></div>
          </div>
          <div className="eyebrow" style={{ marginTop: "6px" }}>What the EDA changed in the model</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "8px", fontSize: "13px", lineHeight: "1.5" }}>
            <div className="step" style={{ padding: "0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>a</div><div>Target modelled as log-damage, with a multiplicative bias correction fitted per fold.</div></div>
            <div className="step" style={{ padding: "0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>b</div><div>Rainflow features carry almost all the signal (0.9771 alone); stats, spectral and FDS add 0.003.</div></div>
            <div className="step" style={{ padding: "0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>c</div><div>Physics blend gated on positive stress skew, chosen in 62 of 64 nested outer folds.</div></div>
          </div>
        </div>
      </div>
    </>)}
  </section>
    </>
  );
}
