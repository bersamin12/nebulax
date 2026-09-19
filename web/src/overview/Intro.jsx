// Hero, the four headline results, the four-step workflow and where each subsystem sits on the
// train. Numbers come from results/ps3 and docs/ps3_*.md. Kept short on purpose: the detail lives
// in the folds below (OverviewPage.jsx).

const METRICS = [
  { name: "Door", metric: "IoU-weighted F1", value: "0.9818", note: "± 0.0364 · 5 contiguous blocks", model: "Open / Close logistic, 24 physics features" },
  { name: "ACV", metric: "rank decay", value: "0.9792", note: "± 0.0510 · leave-one-case-out", model: "Hot-cooling peer-delta rule, no learned weights" },
  { name: "Rail", metric: "macro F1", value: "0.8051", note: "± 0.1291 · 15 grouped folds · portal 0.8310", model: "3-seed LightGBM with same-side coherence" },
  { name: "SHM", metric: "1 − MAPE", value: "0.9813", note: "leave-one-file-out · portal 0.9728", model: "Log-damage Lasso + rainflow blend" },
];

const PLACES = [
  { id: "01 · DOOR", img: "/overview/view_door.png", alt: "Close-up render of a door", text: "Door leaves on each car side: motor current per open / close cycle" },
  { id: "02 · ACV", img: "/overview/cab_roof.png", alt: "Close-up render of the roof units", text: "Roof-mounted air-conditioning unit: cabin temperature against the sibling cars" },
  { id: "03 · RAIL", img: "/overview/view_bogie.png", alt: "Close-up render of a bogie", text: "Axle boxes on every bogie: 64 vibration channels for corrugation" },
  { id: "04 · SHM", img: "/overview/cab_underframe.png", alt: "Close-up render of the underframe", text: "Structural member under the floor: fatigue damage from its stress record" },
];

export default function Intro({ openTwin, openTour }) {
  return (
    <>
  {/* hero */}
  <section id="top" className="ov-sec ov-hero">
    <div className="ov-hero-text">
      <div className="eyebrow">Team Bus MRT Walk · Problem Statement 3</div>
      <h1>Train condition monitoring in one workspace</h1>
      <p>Upload a dataset for doors, air conditioning, rail corrugation, and structural fatigue. Review each prediction to spot affected components</p>
      <div className="ov-hero-actions">
        <a href="?page=predict" onClick={openTwin} className="btn primary" style={{ textDecoration: "none" }}>Open the prediction workspace</a>
        <a href="?page=predict&tour=1" onClick={openTour} className="btn" style={{ textDecoration: "none" }}>Take the tour</a>
      </div>
    </div>
    <div className="ov-hero-art" aria-hidden="true">
      <img src="/overview/cab_iso.png" alt="" />
    </div>
  </section>

  {/* headline results */}
  <section className="ov-sec" style={{ paddingTop: "32px", paddingBottom: "8px", display: "flex", flexDirection: "column", gap: "12px" }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: "4px 16px" }}>
      <div className="eyebrow">Cross-validation on training data · nothing fitted on Test</div>
      <div style={{ fontSize: "12px", color: "#5b6673" }}>Full ladder: <span className="mono">results/ps3/leaderboard.md</span></div>
    </div>
    <div className="ov-g4" style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "12px" }}>
      {METRICS.map((m) => (
        <div key={m.name} className="card" style={{ padding: "18px 20px", display: "flex", flexDirection: "column", gap: "6px" }}>
          <div className="eyebrow">{m.name} · {m.metric}</div>
          <div className="mono" style={{ fontSize: "34px", fontWeight: "500", lineHeight: "1" }}>{m.value}</div>
          <div style={{ fontSize: "13px", color: "#5b6673" }}>{m.note}</div>
          <div style={{ fontSize: "12px", marginTop: "4px" }}>{m.model}</div>
        </div>
      ))}
    </div>
  </section>

  {/* the workflow in four steps */}
  <section className="ov-sec" style={{ paddingTop: "40px", paddingBottom: "32px", display: "flex", flexDirection: "column", gap: "24px" }}>
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      <div className="eyebrow">Prediction workflow</div>
      <h2 style={{ margin: "0", fontSize: "28px", fontWeight: "600" }}>From Test file to prediction in four steps</h2>
      <p style={{ margin: "0", fontSize: "15px", color: "#5b6673", maxWidth: "820px", lineHeight: "1.55" }}>The browser and <span className="mono" style={{ fontSize: "13px" }}>predict.py</span> call the same functions, so the workspace and the CLI always agree.</p>
    </div>

    <div className="ov-g4" style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "20px" }}>
      <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <svg width="285" height="150" viewBox="0 0 285 150" style={{ width: "100%", height: "150px", background: "#f1f3f5" }} aria-hidden="true">
          <rect x="34" y="30" width="72" height="90" rx="6" fill="#ffffff" stroke="#b8c4cc"></rect>
          <rect x="52" y="22" width="72" height="90" rx="6" fill="#ffffff" stroke="#b8c4cc"></rect>
          <rect x="70" y="14" width="72" height="90" rx="6" fill="#ffffff" stroke="#8a97a3"></rect>
          <path d="M120 14v20h22" fill="none" stroke="#8a97a3"></path>
          <text x="82" y="56" fontFamily="IBM Plex Mono, monospace" fontSize="11" fill="#1d2633">.csv</text>
          <text x="82" y="72" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#5b6673">10 kHz</text>
          <text x="82" y="86" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#5b6673">129 col</text>
          <path d="M160 64h44" stroke="var(--accent)" strokeWidth="2"></path><path d="M198 58l8 6-8 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
          <rect x="214" y="40" width="52" height="48" rx="6" fill="#ffffff" stroke="var(--accent)" strokeWidth="1.5"></rect>
          <text x="240" y="60" fontFamily="IBM Plex Mono, monospace" fontSize="10" fill="var(--accent)" textAnchor="middle">QUEUE</text>
          <text x="240" y="78" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#1d2633" textAnchor="middle">68 / 68</text>
          <rect x="34" y="130" width="232" height="6" rx="3" fill="var(--accent)"></rect>
        </svg>
        <div style={{ padding: "16px 18px 18px", display: "flex", flexDirection: "column", gap: "6px" }}>
          <div className="mono" style={{ fontSize: "11px", color: "var(--accent)" }}>01 · UPLOAD</div>
          <div style={{ fontSize: "16px", fontWeight: "600" }}>Choose a system and add files</div>
          <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>A Door CSV, an ACV workbook, or a folder of Rail or SHM files. Large folders are sent in batches.</div>
        </div>
      </div>
      <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <svg width="285" height="150" viewBox="0 0 285 150" style={{ width: "100%", height: "150px", background: "#f1f3f5" }} aria-hidden="true">
          <rect x="20" y="22" width="150" height="106" rx="4" fill="#ffffff" stroke="#b8c4cc"></rect>
          <rect x="20" y="22" width="150" height="20" rx="4" fill="#eef1f3"></rect>
          <text x="28" y="36" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#5b6673">Time</text>
          <text x="62" y="36" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="var(--accent)">Car 3 - Indoor Avg T</text>
          <line x1="20" y1="62" x2="170" y2="62" stroke="#eef1f3"></line><line x1="20" y1="82" x2="170" y2="82" stroke="#eef1f3"></line><line x1="20" y1="102" x2="170" y2="102" stroke="#eef1f3"></line>
          <text x="28" y="56" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633">14:02:10</text><text x="96" y="56" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633">27.9</text>
          <text x="28" y="76" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633">14:02:11</text><text x="96" y="76" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633">28.1</text>
          <text x="28" y="96" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633">14:02:12</text><text x="96" y="96" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#a66500">NaN</text>
          <text x="28" y="116" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633">14:02:13</text><text x="96" y="116" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633">28.0</text>
          <path d="M178 40h18" stroke="var(--accent)" strokeWidth="2"></path><path d="M191 34l6 6-6 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
          <rect x="204" y="26" width="70" height="28" rx="4" fill="#ffffff" stroke="var(--accent)"></rect>
          <text x="239" y="38" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="var(--accent)" textAnchor="middle">canonical</text>
          <text x="239" y="48" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633" textAnchor="middle">indoor_temp</text>
          <text x="204" y="78" fontFamily="IBM Plex Sans, sans-serif" fontSize="9" fill="#5b6673">ms timestamps</text>
          <text x="204" y="94" fontFamily="IBM Plex Sans, sans-serif" fontSize="9" fill="#5b6673">alias mapping</text>
          <text x="204" y="110" fontFamily="IBM Plex Sans, sans-serif" fontSize="9" fill="#5b6673">unknowns reported</text>
        </svg>
        <div style={{ padding: "16px 18px 18px", display: "flex", flexDirection: "column", gap: "6px" }}>
          <div className="mono" style={{ fontSize: "11px", color: "var(--accent)" }}>02 · CHECK</div>
          <div style={{ fontSize: "16px", fontWeight: "600" }}>Confirm the file format</div>
          <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>Columns and timestamps are checked against what the model expects before anything is queued.</div>
        </div>
      </div>
      <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <svg width="285" height="150" viewBox="0 0 285 150" style={{ width: "100%", height: "150px", background: "#f1f3f5" }} aria-hidden="true">
          <polyline fill="none" stroke="#1d2633" strokeWidth="1.5" points="18,80 26,60 34,96 42,52 50,88 58,70 66,98 74,44 82,86 90,64 98,92 106,56 114,84 122,72 130,90 138,50 146,82"></polyline>
          <text x="18" y="124" fontFamily="IBM Plex Sans, sans-serif" fontSize="9" fill="#5b6673">raw signal (10 kHz)</text>
          <path d="M156 74h18" stroke="var(--accent)" strokeWidth="2"></path><path d="M169 68l6 6-6 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
          <rect x="184" y="34" width="84" height="8" rx="4" fill="var(--accent-soft)"></rect><rect x="184" y="34" width="60" height="8" rx="4" fill="var(--accent)"></rect>
          <rect x="184" y="50" width="84" height="8" rx="4" fill="var(--accent-soft)"></rect><rect x="184" y="50" width="30" height="8" rx="4" fill="var(--accent)"></rect>
          <rect x="184" y="66" width="84" height="8" rx="4" fill="var(--accent-soft)"></rect><rect x="184" y="66" width="72" height="8" rx="4" fill="var(--accent)"></rect>
          <rect x="184" y="82" width="84" height="8" rx="4" fill="var(--accent-soft)"></rect><rect x="184" y="82" width="44" height="8" rx="4" fill="var(--accent)"></rect>
          <text x="184" y="108" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#5b6673">RMS  band  coherence</text>
          <text x="184" y="124" fontFamily="IBM Plex Sans, sans-serif" fontSize="9" fill="#5b6673">201 features per file</text>
        </svg>
        <div style={{ padding: "16px 18px 18px", display: "flex", flexDirection: "column", gap: "6px" }}>
          <div className="mono" style={{ fontSize: "11px", color: "var(--accent)" }}>03 · FEATURISE</div>
          <div style={{ fontSize: "16px", fontWeight: "600" }}>Build the model inputs</div>
          <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>Door-cycle currents, peer temperature deltas, axle-box spectra and coherence, rainflow damage sums.</div>
        </div>
      </div>
      <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <svg width="285" height="150" viewBox="0 0 285 150" style={{ width: "100%", height: "150px", background: "#f1f3f5" }} aria-hidden="true">
          <rect x="18" y="52" width="70" height="46" rx="6" fill="#ffffff" stroke="#8a97a3"></rect>
          <text x="53" y="72" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#5b6673" textAnchor="middle">features</text>
          <text x="53" y="86" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#1d2633" textAnchor="middle">x₁ … xₙ</text>
          <path d="M92 75h18" stroke="var(--accent)" strokeWidth="2"></path><path d="M105 69l6 6-6 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
          <rect x="116" y="40" width="84" height="70" rx="6" fill="#ffffff" stroke="var(--accent)" strokeWidth="1.5"></rect>
          <text x="158" y="60" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="var(--accent)" textAnchor="middle">models/ps3/</text>
          <text x="158" y="74" fontFamily="IBM Plex Mono, monospace" fontSize="10" fill="#1d2633" textAnchor="middle">door.pkl</text>
          <text x="158" y="92" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#5b6673" textAnchor="middle">frozen artifact</text>
          <path d="M204 75h18" stroke="var(--accent)" strokeWidth="2"></path><path d="M217 69l6 6-6 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
          <rect x="228" y="46" width="44" height="22" rx="11" fill="#e6f1ea" stroke="#1f8059"></rect>
          <text x="250" y="61" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1f8059" textAnchor="middle">Normal</text>
          <rect x="228" y="82" width="44" height="22" rx="11" fill="#fbe9e8" stroke="#c43d36"></rect>
          <text x="250" y="97" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#c43d36" textAnchor="middle">Abnorm.</text>
          <text x="18" y="128" fontFamily="IBM Plex Sans, sans-serif" fontSize="9" fill="#5b6673">one row per cycle / car / file, then the organiser CSV</text>
        </svg>
        <div style={{ padding: "16px 18px 18px", display: "flex", flexDirection: "column", gap: "6px" }}>
          <div className="mono" style={{ fontSize: "11px", color: "var(--accent)" }}>04 · PREDICT</div>
          <div style={{ fontSize: "16px", fontWeight: "600" }}>Run the frozen model</div>
          <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>One row per cycle, car or file; the CSV is validated against the organiser schema before download.</div>
        </div>
      </div>
    </div>

    {/* where each subsystem lives: animated cab elevation on desktop, four tiles on a phone */}
    <div className="card" style={{ overflow: "hidden" }}>
      <div style={{ padding: "20px 24px 0", display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: "4px 16px" }}>
        <div className="eyebrow">Subsystem locations on the train</div>
        <div style={{ fontSize: "12px", color: "#5b6673" }}>Selecting a result highlights its component: green for normal, red for a detected fault.</div>
      </div>
      <div style={{ padding: "12px 24px 20px" }}>
        <div className="ov-cab">
          <img src="/overview/cab_elev.png" alt="Side elevation of the first cab of the train model" />
          <div className="spot"></div>
          {PLACES.map((p, i) => (
            <div key={p.id} className={`cap cap${i + 1}`}>
              <span className="chip" style={{ background: "var(--accent)", color: "#ffffff" }}>{p.id.slice(5)}</span>
              <span style={{ fontSize: "13px" }}>{p.text}</span>
            </div>
          ))}
        </div>
        <div className="ov-cab-tiles">
          {PLACES.map((p) => (
            <div key={p.id} style={{ position: "relative", borderRadius: "6px", overflow: "hidden", background: "#f1f3f5" }}>
              <img src={p.img} alt={p.alt} style={{ width: "100%", height: "120px", objectFit: "cover", display: "block" }} />
              <div className="mono" style={{ position: "absolute", left: "10px", bottom: "8px", fontSize: "11px", background: "rgba(255,255,255,.9)", padding: "3px 8px", borderRadius: "4px" }}>{p.id}</div>
            </div>
          ))}
        </div>
      </div>
    </div>

    <div className="callout" style={{ fontSize: "13px", color: "#5b6673" }}>
      <strong style={{ color: "#1d2633" }}>Validation rule.</strong> Every scaler, threshold and template is fitted on the training partition of each fold only; the organiser Test inputs are unlabelled and never touched. Portal scores and CSV hashes: <span className="mono" style={{ fontSize: "12px" }}>results/ps3/portal_scores.md</span>.
    </div>
  </section>
    </>
  );
}
