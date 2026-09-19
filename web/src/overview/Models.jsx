// Main models carousel: one page per PS3 task, diagram + plain-terms + architecture + literature.
// Numbers, dataset counts and reference IDs come from results/ps3, docs/ps3_*.md and
// docs/research/references.md; the markup mirrors the design canvas (docs/design/Main.dc.html).
import { useState } from "react";

export default function Models({  }) {
  const [m, setM] = useState(0);
  return (
    <>
{/* MODELS carousel */}
  <section className="ov-sec" style={{ paddingTop: "4px", paddingBottom: "40px", display: "flex", flexDirection: "column", gap: "20px" }}>

    <div style={{ display: "flex", alignItems: "flex-end", gap: "12px" }}>
      <button type="button" className="arrow" onClick={() => setM((m + 3) % 4)} aria-label="Previous model"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M15 18l-6-6 6-6"></path></svg></button>
      <div className="ov-tabs">
        <button type="button" className={"tab" + (m === 0 ? " on" : "")} onClick={() => setM(0)}><span className="mono" style={{ fontSize: "11px", letterSpacing: ".12em", color: "var(--accent)" }}>01 · DOOR</span><span style={{ fontSize: "14px", fontWeight: "600" }}>Segment + classify cycles</span><span className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>0.9818 IoU-F1 · logistic</span></button>
        <button type="button" className={"tab" + (m === 1 ? " on" : "")} onClick={() => setM(1)}><span className="mono" style={{ fontSize: "11px", letterSpacing: ".12em", color: "var(--accent)" }}>02 · ACV</span><span style={{ fontSize: "14px", fontWeight: "600" }}>Rank the leaking car</span><span className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>0.9792 rank decay · fixed rule</span></button>
        <button type="button" className={"tab" + (m === 2 ? " on" : "")} onClick={() => setM(2)}><span className="mono" style={{ fontSize: "11px", letterSpacing: ".12em", color: "var(--accent)" }}>03 · RAIL</span><span style={{ fontSize: "14px", fontWeight: "600" }}>Normal / Side I / Side II</span><span className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>0.8051 macro F1 · LightGBM × 3</span></button>
        <button type="button" className={"tab" + (m === 3 ? " on" : "")} onClick={() => setM(3)}><span className="mono" style={{ fontSize: "11px", letterSpacing: ".12em", color: "var(--accent)" }}>04 · SHM</span><span style={{ fontSize: "14px", fontWeight: "600" }}>Fatigue-damage regression</span><span className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>0.9813 (1 − MAPE) · sparse linear</span></button>
      </div>
      <button type="button" className="arrow" onClick={() => setM((m + 1) % 4)} aria-label="Next model"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 6l6 6-6 6"></path></svg></button>
    </div>

    {/* page 1 DOOR */}
    {m === 0 && (<>
      <div className="card ov-card-lg" style={{ padding: "28px 32px 32px", display: "flex", flexDirection: "column", gap: "18px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          <div className="ov-row" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}><span></span><span className="chip">0.9818 IoU-F1 · logistic regression · models/ps3/door.pkl</span></div>
          <h3 style={{ margin: "0", fontSize: "24px", fontWeight: "600", lineHeight: "1.2" }}>Door segmentation + resistance classification</h3>
        </div>
        <div className="ov-diagram-box" style={{ border: "1px solid #dfe3e7", borderRadius: "6px", padding: "16px 20px 10px", display: "flex", flexDirection: "column", gap: "6px", background: "#ffffff" }}>
          <div className="eyebrow">Model diagram · linear classifier, one per direction</div>
          <svg className="ov-diagram" width="1094" height="210" viewBox="0 0 1094 210" style={{ width: "100%", height: "auto", display: "block" }} aria-hidden="true">
              <g fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#5b6673"><text x="20" y="44">i_cruise_rel</text><text x="20" y="68">i_peak_rel</text><text x="20" y="92">profile_dist</text><text x="20" y="116">emf_resid</text><text x="20" y="140">dur_ratio</text><text x="20" y="164">… 24 features</text></g>
              <g stroke="#b8c4cc" strokeWidth="1"><line x1="130" y1="40" x2="258" y2="100"></line><line x1="130" y1="64" x2="258" y2="100"></line><line x1="130" y1="88" x2="258" y2="100"></line><line x1="130" y1="112" x2="258" y2="100"></line><line x1="130" y1="136" x2="258" y2="100"></line><line x1="130" y1="160" x2="258" y2="100"></line></g>
              <g fontFamily="IBM Plex Mono, monospace" fontSize="11" fill="var(--accent)"><text x="180" y="58">w₁</text><text x="180" y="78">w₂</text><text x="180" y="96">w₃</text><text x="180" y="122">w₄</text><text x="180" y="142">w₅</text></g>
              <circle cx="300" cy="100" r="42" fill="#ffffff" stroke="var(--accent)" strokeWidth="1.5"></circle>
              <text x="300" y="105" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#1d2633" textAnchor="middle">Σ w·x + b</text>
              <path d="M346 100h36" stroke="var(--accent)" strokeWidth="2"></path><path d="M374 94l8 6-8 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
              <circle cx="424" cy="100" r="34" fill="#ffffff" stroke="var(--accent)" strokeWidth="1.5"></circle>
              <path d="M402 118 C 418 118, 418 82, 446 82" fill="none" stroke="#1d2633" strokeWidth="1.5"></path>
              <text x="424" y="154" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#5b6673" textAnchor="middle">sigmoid → p</text>
              <path d="M462 100h44" stroke="var(--accent)" strokeWidth="2"></path><path d="M498 94l8 6-8 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
              <rect x="514" y="76" width="120" height="48" rx="6" fill="#ffffff" stroke="var(--accent)" strokeWidth="1.5"></rect>
              <text x="574" y="105" fontFamily="IBM Plex Mono, monospace" fontSize="14" fill="#1d2633" textAnchor="middle">p ≥ τ ?</text>
              <path d="M634 88 L 690 58" stroke="#8a97a3"></path><path d="M634 112 L 690 142" stroke="#8a97a3"></path>
              <text x="650" y="64" fontFamily="IBM Plex Mono, monospace" fontSize="11" fill="#5b6673">no</text><text x="650" y="142" fontFamily="IBM Plex Mono, monospace" fontSize="11" fill="#5b6673">yes</text>
              <rect x="694" y="40" width="130" height="32" rx="16" fill="#e6f1ea" stroke="#1f8059"></rect><text x="759" y="61" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#1f8059" textAnchor="middle">Normal</text>
              <rect x="694" y="128" width="190" height="32" rx="16" fill="#fbe9e8" stroke="#c43d36"></rect><text x="789" y="149" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#c43d36" textAnchor="middle">Abnormal resistance</text>
              <line x1="916" y1="40" x2="916" y2="160" stroke="#dfe3e7"></line>
              <g fontFamily="IBM Plex Sans, sans-serif" fontSize="12" fill="#5b6673"><text x="932" y="60">× 2 identical models:</text><text x="932" y="80">one for Open cycles,</text><text x="932" y="100">one for Close cycles.</text><text x="932" y="128" fill="#1d2633">25 parameters each,</text><text x="932" y="148" fill="#1d2633">fits in about 5 s.</text></g>
              <text x="20" y="198" fontFamily="IBM Plex Sans, sans-serif" fontSize="12" fill="#5b6673">Before this step the raw stream is cut into single open or close movements; each movement becomes one feature row.</text>
            </svg>
        </div>
        <div className="ov-side" style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: "32px", alignItems: "start" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            <div className="callout"><span className="mono" style={{ fontSize: "10px", letterSpacing: ".12em", color: "#006d73" }}>SUMMARY</span><br />A door that is harder to move draws more current in the middle of its travel. The 50 Hz stream is divided into individual opening and closing movements, then compared with a normal door.</div>
            <div className="kv">
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Input</div><div>One <span className="mono" style={{ fontSize: "12px" }}>Test.csv</span> stream: datetime, motor current, leaf position, voltage, back-EMF, commands</div>
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Model</div><div>Separate Open / Close logistic regression on 24 physics features, class-weighted, saved threshold</div>
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Output</div><div><span className="mono" style={{ fontSize: "12px" }}>start_time, end_time, prediction</span> · Normal / Abnormal resistance</div>
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Selection rationale</div><div>Ties the deep rows (MultiRocket, LITETime = 1.0000) with a 5 s fit and readable coefficients; segmentation is boundary-exact on all 110 training cycles</div>
          </div>
            <div>
              <div className="eyebrow" style={{ marginBottom: "4px" }}>Architecture &middot; 5 pipeline steps</div>
              <div className="step"><div className="num">1</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Segment.</strong> A gap &gt; 0.5 s between samples starts a new cycle; with no gaps, a command-edge + position-hysteresis state machine takes over. Source timestamps are kept to the millisecond; cycles never overlap.</div></div>
            <div className="step"><div className="num">2</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Featurise.</strong> Travel fraction splits a stroke into opening (0 to 15 %), cruise (15 to 85 %) and closing (85 to 100 %). 24 features: phase currents, peak and its position, charge / energy, voltage and back-EMF, resistance and speed proxies, duration vs command time, two wavelet bands, a 64-bin current-vs-position profile.</div></div>
            <div className="step"><div className="num">3</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Baseline (train only).</strong> Medians, robust scales, a current-profile template and an EMF-vs-speed fit from <em>normal</em> training cycles, separately for Open and Close, give ratios (clipped 0 to 10), robust z-scores (± 25), profile distance and EMF residual.</div></div>
            <div className="step"><div className="num">4</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Classify.</strong> One logistic model per direction yields an abnormal probability; the saved threshold assigns the label. Penalty and threshold were tuned on inner contiguous splits.</div></div>
            <div className="step"><div className="num">5</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Emit.</strong> Native sample boundaries per cycle; no post-smoothing. Validation: five contiguous raw-stream blocks, segmentation + classification rerun end to end.</div></div>
            </div>
          </div>
          <div style={{ background: "#f7f8fa", borderRadius: "6px", padding: "14px 16px", display: "flex", flexDirection: "column" }}>
            <div className="eyebrow" style={{ marginBottom: "4px" }}>Supporting research</div>
            <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R64</span><span>Ham et al., <em>Sensors</em> 2019. Traditional vs deep-learning fault diagnosis for train door systems: motor-current features with a linear model match deep nets on small labelled sets, which set our baseline.</span></div>
            <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R66</span><span>Shiao et al., <em>Sensors</em> 2026. Wavelet-based health monitoring of door actuation from motor current: source of the two coarse wavelet-band features.</span></div>
            <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R67</span><span>Shiao et al., <em>Applied Sciences</em> 2025. Motor-current analysis for door obstacles: why middle-travel current, not the start peak, separates resistance from obstruction.</span></div>
            <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R68</span><span>Song et al., <em>Scientific Reports</em> 2026. Stacking ensemble for subway door fault prediction: reproduced as the RF + XGB + calibrated logistic ladder row (0.9909).</span></div>
            <div className="ref" style={{ borderBottom: "none", fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R43</span><span>Middlehurst et al., <em>DMKD</em> 2024, bake-off redux. Chose the raw-series comparison rows: MultiRocket (R47), QUANT (R49, <a href="https://doi.org/10.1007/s10618-024-01036-9" target="_blank" rel="noreferrer" className="mono" style={{ fontSize: "10px" }}>doi:10.1007/s10618-024-01036-9</a>) and LITETime (R53, <a href="https://doi.org/10.1109/DSAA60987.2023.10302569" target="_blank" rel="noreferrer" className="mono" style={{ fontSize: "10px" }}>doi:10.1109/DSAA60987.2023.10302569</a>).</span></div>
          </div>
        </div>
      </div>
    </>)}

    {/* page 2 ACV */}
    {m === 1 && (<>
      <div className="card ov-card-lg" style={{ padding: "28px 32px 32px", display: "flex", flexDirection: "column", gap: "18px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          <div className="ov-row" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}><span></span><span className="chip">0.9792 rank decay · fixed rule · models/ps3/acv.json</span></div>
          <h3 style={{ margin: "0", fontSize: "24px", fontWeight: "600", lineHeight: "1.2" }}>Refrigerant-leak car ranking</h3>
        </div>
        <div className="ov-diagram-box" style={{ border: "1px solid #dfe3e7", borderRadius: "6px", padding: "16px 20px 10px", display: "flex", flexDirection: "column", gap: "6px", background: "#ffffff" }}>
          <div className="eyebrow">Model diagram · peer-residual rule, zero learned parameters</div>
          <svg className="ov-diagram" width="1094" height="210" viewBox="0 0 1094 210" style={{ width: "100%", height: "auto", display: "block" }} aria-hidden="true">
              <g fontFamily="IBM Plex Mono, monospace" fontSize="10" fill="#5b6673" textAnchor="middle"><text x="49" y="16">01</text><text x="117" y="16">02</text><text x="185" y="16">03</text><text x="253" y="16">04</text><text x="321" y="16">05</text><text x="389" y="16">06</text><text x="457" y="16">07</text><text x="525" y="16">08</text></g>
              <g fontFamily="IBM Plex Mono, monospace" fontSize="13" textAnchor="middle">
                <rect x="20" y="24" width="58" height="40" rx="4" fill="#ffffff" stroke="#8a97a3"></rect><text x="49" y="49" fill="#1d2633">27.4</text>
                <rect x="88" y="24" width="58" height="40" rx="4" fill="#ffffff" stroke="#8a97a3"></rect><text x="117" y="49" fill="#1d2633">27.1</text>
                <rect x="156" y="24" width="58" height="40" rx="4" fill="#fbe9e8" stroke="#c43d36"></rect><text x="185" y="49" fill="#c43d36">29.8</text>
                <rect x="224" y="24" width="58" height="40" rx="4" fill="#ffffff" stroke="#8a97a3"></rect><text x="253" y="49" fill="#1d2633">27.6</text>
                <rect x="292" y="24" width="58" height="40" rx="4" fill="#ffffff" stroke="#8a97a3"></rect><text x="321" y="49" fill="#1d2633">27.0</text>
                <rect x="360" y="24" width="58" height="40" rx="4" fill="#ffffff" stroke="#8a97a3"></rect><text x="389" y="49" fill="#1d2633">27.3</text>
                <rect x="428" y="24" width="58" height="40" rx="4" fill="#ffffff" stroke="#8a97a3"></rect><text x="457" y="49" fill="#1d2633">27.5</text>
                <rect x="496" y="24" width="58" height="40" rx="4" fill="#ffffff" stroke="#8a97a3"></rect><text x="525" y="49" fill="#1d2633">27.2</text>
              </g>
              <text x="287" y="84" fontFamily="IBM Plex Sans, sans-serif" fontSize="12" fill="#5b6673" textAnchor="middle">indoor °C for cars 01 to 08 at one hot, cooling-mode timestamp</text>
              <g fontFamily="IBM Plex Sans, sans-serif" fontSize="12" fill="#5b6673"><text x="610" y="36">usable rows = finite T, valid flag, cooling mode</text><text x="610" y="56">hot rows = outdoor T ≥ the case median</text><text x="610" y="76">fewer than 30 usable or 15 hot rows: car sorts last</text></g>
              <path d="M287 90v22" stroke="var(--accent)" strokeWidth="2"></path><path d="M280 106l7 7 7-7" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
              <rect x="140" y="118" width="294" height="44" rx="6" fill="#ffffff" stroke="var(--accent)" strokeWidth="1.5"></rect>
              <text x="287" y="145" fontFamily="IBM Plex Mono, monospace" fontSize="14" fill="#1d2633" textAnchor="middle">Δᵢ = Tᵢ − median(peers)</text>
              <path d="M440 140h44" stroke="var(--accent)" strokeWidth="2"></path><path d="M476 134l8 6-8 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
              <rect x="490" y="118" width="200" height="44" rx="6" fill="#ffffff" stroke="var(--accent)" strokeWidth="1.5"></rect>
              <text x="590" y="145" fontFamily="IBM Plex Mono, monospace" fontSize="13" fill="#1d2633" textAnchor="middle">mean over hot rows</text>
              <path d="M696 140h44" stroke="var(--accent)" strokeWidth="2"></path><path d="M732 134l8 6-8 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
              <rect x="746" y="110" width="170" height="60" rx="6" fill="#ffffff" stroke="var(--accent)" strokeWidth="1.5"></rect>
              <text x="831" y="134" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#5b6673" textAnchor="middle">sort descending</text>
              <text x="831" y="156" fontFamily="IBM Plex Mono, monospace" fontSize="13" fill="#c43d36" textAnchor="middle">03 | 04 | 01 | …</text>
              <path d="M922 140h44" stroke="var(--accent)" strokeWidth="2"></path><path d="M958 134l8 6-8 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
              <rect x="972" y="118" width="104" height="44" rx="22" fill="var(--accent-bg)" stroke="var(--accent)" strokeWidth="1.5"></rect>
              <text x="1024" y="145" fontFamily="IBM Plex Mono, monospace" fontSize="13" fill="var(--accent)" textAnchor="middle">ranking</text>
              <text x="20" y="198" fontFamily="IBM Plex Sans, sans-serif" fontSize="12" fill="#5b6673">Ties: indoor minus cooling setpoint, then car ID. Zero learned parameters: the rule was fixed before any case was scored.</text>
            </svg>
        </div>
        <div className="ov-side" style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: "32px", alignItems: "start" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            <div className="callout"><span className="mono" style={{ fontSize: "10px", letterSpacing: ".12em", color: "#006d73" }}>SUMMARY</span><br />In warm weather, while the air-conditioning is cooling, a leaking car tends to stay warmer than the other cars on the same train. Comparing cars reduces the effect of shared weather conditions.</div>
            <div className="kv">
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Input</div><div>One <span className="mono" style={{ fontSize: "12px" }}>.xlsx</span> per case: 30 s telemetry for up to 8 cars, headers <span className="mono" style={{ fontSize: "12px" }}>Car N - parameter</span>; parameter set varies by workbook</div>
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Model</div><div>Fixed, pre-registered rule <span className="mono" style={{ fontSize: "12px" }}>peer_delta_hot</span>: hot, cooling-mode indoor temperature minus contemporaneous peer median</div>
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Output</div><div><span className="mono" style={{ fontSize: "12px" }}>file_id, ranked_cars</span> · e.g. <span className="mono" style={{ fontSize: "12px" }}>01|03|04|08|07|06|02|05</span></div>
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Selection rationale</div><div>Six labelled cases cannot support a credible supervised classifier; the rule was fixed before the cases were inspected and uses no car-ID prior</div>
          </div>
            <div>
              <div className="eyebrow" style={{ marginBottom: "4px" }}>Architecture &middot; 5 pipeline steps</div>
              <div className="step"><div className="num">1</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Load.</strong> Discover car columns by pattern (one or two digit IDs, spelling preserved). Map aliases: indoor / outdoor temperature, running mode, validity flag, cooling setpoint. Unmapped parameters are listed, never silently used.</div></div>
            <div className="step"><div className="num">2</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Filter.</strong> A reading is usable when indoor temperature is finite, validity is not explicitly false and the mode is a cooling mode. “Hot” means fleet-median outdoor temperature at or above its within-case median (fallbacks: indoor median, then all rows).</div></div>
            <div className="step"><div className="num">3</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Score.</strong> At each timestamp subtract the median indoor temperature of usable peers; <span className="mono" style={{ fontSize: "12px" }}>peer_delta_hot</span> is the mean of that difference over hot usable rows. Needs ≥ 30 usable and ≥ 15 hot rows, else the car gets no score.</div></div>
            <div className="step"><div className="num">4</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Rank.</strong> Descending score. Ties: within-case control residual (indoor minus cooling setpoint), then car ID for determinism. Cars without a score sort last (four cars in training case 04 are empty).</div></div>
            <div className="step"><div className="num">5</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Emit.</strong> Every car once, <span className="mono" style={{ fontSize: "12px" }}>|</span>-separated, beside the workbook's <span className="mono" style={{ fontSize: "12px" }}>file_id</span>. Runtime, recovery and persistence features exist for the ladder only.</div></div>
            </div>
          </div>
          <div style={{ background: "#f7f8fa", borderRadius: "6px", padding: "14px 16px", display: "flex", flexDirection: "column" }}>
            <div className="eyebrow" style={{ marginBottom: "4px" }}>Supporting research</div>
            <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R252</span><span>Rossi &amp; Braun, <em>HVAC&amp;R Research</em> 1997. The founding temperature-only residual + directional-rule method for vapour-compression faults: a refrigerant leak shows as a warm cabin with the unit in cooling.</span></div>
            <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R251</span><span>Guo et al., <em>Energy and AI</em> 2024. Electric-bus air-conditioning fault diagnosis with domain knowledge: peer-fleet residuals as the reference instead of a physics model, the basis of our same-train comparison.</span></div>
            <div className="ref" style={{ borderBottom: "none", fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R253</span><span>Yoon et al., ORNL report 2024. Field HVAC fault data and manufacturer interviews: why runtime, recovery and persistence rules were added as alternatives, and why they were not promoted on six cases.</span></div>
          </div>
        </div>
      </div>
    </>)}

    {/* page 3 RAIL */}
    {m === 2 && (<>
      <div className="card ov-card-lg" style={{ padding: "28px 32px 32px", display: "flex", flexDirection: "column", gap: "18px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          <div className="ov-row" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}><span></span><span className="chip">0.8051 macro F1 · LightGBM × 3 · models/ps3/rail.pkl</span></div>
          <h3 style={{ margin: "0", fontSize: "24px", fontWeight: "600", lineHeight: "1.2" }}>Normal / Side I / Side II from 64 axle-box accelerometers</h3>
        </div>
        <div className="ov-diagram-box" style={{ border: "1px solid #dfe3e7", borderRadius: "6px", padding: "16px 20px 10px", display: "flex", flexDirection: "column", gap: "6px", background: "#ffffff" }}>
          <div className="eyebrow">Model diagram · gradient-boosted tree ensemble with mirror averaging</div>
          <svg className="ov-diagram" width="1094" height="232" viewBox="0 0 1094 232" style={{ width: "100%", height: "auto", display: "block" }} aria-hidden="true">
              <rect x="20" y="40" width="140" height="56" rx="6" fill="#ffffff" stroke="#8a97a3"></rect>
              <text x="90" y="62" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#5b6673" textAnchor="middle">one file</text><text x="90" y="82" fontFamily="IBM Plex Mono, monospace" fontSize="13" fill="#1d2633" textAnchor="middle">201 features</text>
              <rect x="20" y="140" width="140" height="56" rx="6" fill="#ffffff" stroke="#8a97a3" strokeDasharray="4 3"></rect>
              <text x="90" y="162" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#5b6673" textAnchor="middle">mirrored file</text><text x="90" y="182" fontFamily="IBM Plex Mono, monospace" fontSize="13" fill="#1d2633" textAnchor="middle">L ↔ R swapped</text>
              <path d="M164 68h40" stroke="var(--accent)" strokeWidth="2"></path><path d="M196 62l8 6-8 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
              <path d="M164 168h40" stroke="var(--accent)" strokeWidth="2"></path><path d="M196 162l8 6-8 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
              <g stroke="var(--accent)" strokeWidth="1.2" fill="#ffffff">
                <g transform="translate(212,26) scale(1.7)"><circle cx="20" cy="6" r="5"></circle><line x1="20" y1="11" x2="9" y2="22"></line><line x1="20" y1="11" x2="31" y2="22"></line><circle cx="9" cy="26" r="4"></circle><circle cx="31" cy="26" r="4"></circle><line x1="9" y1="30" x2="3" y2="40"></line><line x1="9" y1="30" x2="15" y2="40"></line><line x1="31" y1="30" x2="25" y2="40"></line><line x1="31" y1="30" x2="37" y2="40"></line></g>
                <g transform="translate(302,26) scale(1.7)"><circle cx="20" cy="6" r="5"></circle><line x1="20" y1="11" x2="9" y2="22"></line><line x1="20" y1="11" x2="31" y2="22"></line><circle cx="9" cy="26" r="4"></circle><circle cx="31" cy="26" r="4"></circle><line x1="9" y1="30" x2="3" y2="40"></line><line x1="9" y1="30" x2="15" y2="40"></line><line x1="31" y1="30" x2="25" y2="40"></line><line x1="31" y1="30" x2="37" y2="40"></line></g>
                <g transform="translate(392,26) scale(1.7)"><circle cx="20" cy="6" r="5"></circle><line x1="20" y1="11" x2="9" y2="22"></line><line x1="20" y1="11" x2="31" y2="22"></line><circle cx="9" cy="26" r="4"></circle><circle cx="31" cy="26" r="4"></circle><line x1="9" y1="30" x2="3" y2="40"></line><line x1="9" y1="30" x2="15" y2="40"></line><line x1="31" y1="30" x2="25" y2="40"></line><line x1="31" y1="30" x2="37" y2="40"></line></g>
              </g>
              <text x="336" y="114" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#5b6673" textAnchor="middle">3 balanced LightGBM, seeds 0 / 1 / 2, hundreds of trees each</text>
              <g stroke="var(--accent)" strokeWidth="1.2" fill="#ffffff" opacity="0.5">
                <g transform="translate(212,126) scale(1.7)"><circle cx="20" cy="6" r="5"></circle><line x1="20" y1="11" x2="9" y2="22"></line><line x1="20" y1="11" x2="31" y2="22"></line><circle cx="9" cy="26" r="4"></circle><circle cx="31" cy="26" r="4"></circle><line x1="9" y1="30" x2="3" y2="40"></line><line x1="9" y1="30" x2="15" y2="40"></line><line x1="31" y1="30" x2="25" y2="40"></line><line x1="31" y1="30" x2="37" y2="40"></line></g>
                <g transform="translate(302,126) scale(1.7)"><circle cx="20" cy="6" r="5"></circle><line x1="20" y1="11" x2="9" y2="22"></line><line x1="20" y1="11" x2="31" y2="22"></line><circle cx="9" cy="26" r="4"></circle><circle cx="31" cy="26" r="4"></circle><line x1="9" y1="30" x2="3" y2="40"></line><line x1="9" y1="30" x2="15" y2="40"></line><line x1="31" y1="30" x2="25" y2="40"></line><line x1="31" y1="30" x2="37" y2="40"></line></g>
                <g transform="translate(392,126) scale(1.7)"><circle cx="20" cy="6" r="5"></circle><line x1="20" y1="11" x2="9" y2="22"></line><line x1="20" y1="11" x2="31" y2="22"></line><circle cx="9" cy="26" r="4"></circle><circle cx="31" cy="26" r="4"></circle><line x1="9" y1="30" x2="3" y2="40"></line><line x1="9" y1="30" x2="15" y2="40"></line><line x1="31" y1="30" x2="25" y2="40"></line><line x1="31" y1="30" x2="37" y2="40"></line></g>
              </g>
              <text x="336" y="222" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#5b6673" textAnchor="middle">same trees on the mirrored file, then Side I / II swapped back</text>
              <path d="M474 68 L 588 108" stroke="#8a97a3"></path><path d="M474 168 L 588 128" stroke="#8a97a3"></path>
              <circle cx="620" cy="118" r="30" fill="#ffffff" stroke="var(--accent)" strokeWidth="1.5"></circle>
              <text x="620" y="123" fontFamily="IBM Plex Mono, monospace" fontSize="13" fill="#1d2633" textAnchor="middle">avg</text>
              <path d="M654 118h40" stroke="var(--accent)" strokeWidth="2"></path><path d="M686 112l8 6-8 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
              <rect x="700" y="90" width="160" height="56" rx="6" fill="#ffffff" stroke="var(--accent)" strokeWidth="1.5"></rect>
              <text x="780" y="112" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#5b6673" textAnchor="middle">class prior</text><text x="780" y="132" fontFamily="IBM Plex Mono, monospace" fontSize="13" fill="#1d2633" textAnchor="middle">Side II × 1.25</text>
              <path d="M864 118h40" stroke="var(--accent)" strokeWidth="2"></path><path d="M896 112l8 6-8 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
              <rect x="910" y="90" width="170" height="56" rx="6" fill="#ffffff" stroke="#a66500" strokeWidth="1.5"></rect>
              <text x="995" y="112" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#1d2633" textAnchor="middle">argmax</text><text x="995" y="132" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#a66500" textAnchor="middle">speed &lt; 20 → Normal</text>
            </svg>
        </div>
        <div className="ov-side" style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: "32px", alignItems: "start" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            <div className="callout"><span className="mono" style={{ fontSize: "10px", letterSpacing: ".12em", color: "#006d73" }}>SUMMARY</span><br />Rail corrugation produces a repeated vibration across the axle boxes on one side of the train, while an isolated impact affects fewer boxes. The model compares vibration coherence and level on each side.</div>
            <div className="kv">
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Input</div><div>One 1 s, 10 kHz <span className="mono" style={{ fontSize: "12px" }}>.csv</span> per run: tachometer pulse + 64 vibration and 64 shock channels (8 positions × 8 cars); drop the whole Test folder</div>
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Model</div><div>Three balanced LightGBM (seeds 0 / 1 / 2) on 201 features incl. same-side Welch coherence; mirror training + mirror test-time averaging</div>
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Output</div><div><span className="mono" style={{ fontSize: "12px" }}>file_id, prediction</span> · Normal / Side I / Side II</div>
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Selection rationale</div><div>Only 14 Side I training files: mirroring doubles the minority. Coherence lifted selection CV 0.8365 to 0.8441 and the portal score 0.7994 to 0.8310 for a 794 KB artifact</div>
          </div>
            <div>
              <div className="eyebrow" style={{ marginBottom: "4px" }}>Architecture &middot; 5 pipeline steps</div>
              <div className="step"><div className="num">1</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Speed.</strong> Edges of the 90-tooth pulse and a 0.85 m wheel give speed and distance. Non-finite samples become zero.</div></div>
            <div className="step"><div className="num">2</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Per-channel features.</strong> Centred vibration: RMS, peaks, skew, kurtosis, shape factors, Welch bands 20 to 5000 Hz. Odd positions 1/3/5/7 = Side I, even = Side II. Side medians / maxima / top-3 means, Side I minus II contrasts, per-car votes. Shock channels off.</div></div>
            <div className="step"><div className="num">3</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Coherence.</strong> Magnitude-squared coherence between the six same-side box pairs of each car, averaged over 48 pairs per side into seven Hz bands, giving Side I, Side II and difference summaries. 201 columns in total.</div></div>
            <div className="step"><div className="num">4</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Classify.</strong> Three seeds of balanced LightGBM, trained on each file plus its exact left / right mirror with labels swapped. At inference the mirrored input is scored too and the side probabilities swapped back and averaged. Side II × 1.25 prior, then argmax.</div></div>
            <div className="step"><div className="num" style={{ background: "#a66500" }}>5</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Post-rule (flagged).</strong> <span className="mono" style={{ fontSize: "12px" }}>speed &lt; 20 km/h → Normal</span> is a dataset shortcut, not physics: transfer to routes with low-speed faults is uncertain. Duplicate recordings are grouped before any split.</div></div>
            </div>
          </div>
          <div style={{ background: "#f7f8fa", borderRadius: "6px", padding: "14px 16px", display: "flex", flexDirection: "column" }}>
            <div className="eyebrow" style={{ marginBottom: "4px" }}>Supporting research</div>
            <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R235</span><span>De Rosa et al., <em>Applied Sciences</em> 2024. Detecting corrugation from on-board measurements by isolating other excitation sources: the reason for side contrasts and the same-side coherence idea.</span></div>
            <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R233</span><span>Liu et al., <em>Vehicle System Dynamics</em> 2023. Corrugation maintenance limit from the axle-box acceleration spectrum (IEC 61373 bands): source of the fixed Hz band layout.</span></div>
            <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R239</span><span>Pieringer &amp; Kropp, <em>Applied Acoustics</em> 2022. Model-based rail roughness from axle-box acceleration: why wavelength (distance-domain) features were built and kept in the ladder.</span></div>
            <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R237</span><span>Yu et al., <em>Measurement</em> 2025. Transfer-function corrugation detection from ABA: axle-box amplitude scales with speed, hence the v² normalisation rows and the flagged speed rule.</span></div>
            <div className="ref" style={{ borderBottom: "none", fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R246</span><span>Jahan et al., 2024. CNNs on camera and ABA data for rail surface defects: the deep comparison point, and the reason mirror augmentation was tried first.</span></div>
          </div>
        </div>
      </div>
    </>)}

    {/* page 4 SHM */}
    {m === 3 && (<>
      <div className="card ov-card-lg" style={{ padding: "28px 32px 32px", display: "flex", flexDirection: "column", gap: "18px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          <div className="ov-row" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}><span></span><span className="chip">0.9813 (1 − MAPE) · sparse linear · models/ps3/shm.pkl</span></div>
          <h3 style={{ margin: "0", fontSize: "24px", fontWeight: "600", lineHeight: "1.2" }}>Cumulative fatigue-damage regression</h3>
        </div>
        <div className="ov-diagram-box" style={{ border: "1px solid #dfe3e7", borderRadius: "6px", padding: "16px 20px 10px", display: "flex", flexDirection: "column", gap: "6px", background: "#ffffff" }}>
          <div className="eyebrow">Model diagram · Lasso in log space, blended with a physics estimate</div>
          <svg className="ov-diagram" width="1094" height="232" viewBox="0 0 1094 232" style={{ width: "100%", height: "auto", display: "block" }} aria-hidden="true">
              <rect x="20" y="66" width="140" height="56" rx="6" fill="#ffffff" stroke="#8a97a3"></rect>
              <text x="90" y="88" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#5b6673" textAnchor="middle">109 features</text><text x="90" y="108" fontFamily="IBM Plex Mono, monospace" fontSize="13" fill="#1d2633" textAnchor="middle">log + scale</text>
              <path d="M164 94h40" stroke="var(--accent)" strokeWidth="2"></path><path d="M196 88l8 6-8 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
              <rect x="210" y="20" width="320" height="146" rx="6" fill="#ffffff" stroke="var(--accent)" strokeWidth="1.5"></rect>
              <text x="370" y="42" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#5b6673" textAnchor="middle">Lasso weights: most are exactly 0</text>
              <line x1="226" y1="128" x2="514" y2="128" stroke="#dfe3e7"></line>
              <g fill="var(--accent)"><rect x="226" y="99" width="8" height="29"></rect><rect x="241" y="125" width="8" height="3"></rect><rect x="256" y="125" width="8" height="3"></rect><rect x="271" y="80" width="8" height="48"></rect><rect x="286" y="125" width="8" height="3"></rect><rect x="301" y="125" width="8" height="3"></rect><rect x="316" y="115" width="8" height="13"></rect><rect x="331" y="125" width="8" height="3"></rect><rect x="346" y="125" width="8" height="3"></rect><rect x="361" y="67" width="8" height="61"></rect><rect x="376" y="125" width="8" height="3"></rect><rect x="391" y="125" width="8" height="3"></rect><rect x="406" y="109" width="8" height="19"></rect><rect x="421" y="125" width="8" height="3"></rect><rect x="436" y="125" width="8" height="3"></rect><rect x="451" y="93" width="8" height="35"></rect><rect x="466" y="125" width="8" height="3"></rect><rect x="481" y="125" width="8" height="3"></rect><rect x="496" y="118" width="8" height="10"></rect></g>
              <text x="370" y="154" fontFamily="IBM Plex Mono, monospace" fontSize="13" fill="#1d2633" textAnchor="middle">Σ w·x + b = log D̂</text>
              <path d="M534 94h40" stroke="var(--accent)" strokeWidth="2"></path><path d="M566 88l8 6-8 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
              <rect x="580" y="66" width="150" height="56" rx="6" fill="#ffffff" stroke="var(--accent)" strokeWidth="1.5"></rect>
              <text x="655" y="88" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#5b6673" textAnchor="middle">exp × bias factor</text><text x="655" y="108" fontFamily="IBM Plex Mono, monospace" fontSize="13" fill="#1d2633" textAnchor="middle">D̂ lasso</text>
              <rect x="580" y="150" width="150" height="56" rx="6" fill="#ffffff" stroke="#8a97a3"></rect>
              <text x="655" y="172" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#5b6673" textAnchor="middle">rainflow + Miner</text><text x="655" y="192" fontFamily="IBM Plex Mono, monospace" fontSize="13" fill="#1d2633" textAnchor="middle">Miner, m = 5</text>
              <path d="M734 94 L 802 116" stroke="#8a97a3"></path><path d="M734 178 L 802 136" stroke="#8a97a3" strokeDasharray="4 3"></path>
              <rect x="806" y="94" width="130" height="64" rx="32" fill="#ffffff" stroke="var(--accent)" strokeWidth="1.5"></rect>
              <text x="871" y="121" fontFamily="IBM Plex Mono, monospace" fontSize="13" fill="#1d2633" textAnchor="middle">50 / 50 blend</text><text x="871" y="141" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#5b6673" textAnchor="middle">if skew &gt; 0</text>
              <path d="M940 126h40" stroke="var(--accent)" strokeWidth="2"></path><path d="M972 120l8 6-8 6" fill="none" stroke="var(--accent)" strokeWidth="2"></path>
              <rect x="986" y="100" width="90" height="52" rx="6" fill="#e6f1ea" stroke="#1f8059"></rect>
              <text x="1031" y="132" fontFamily="IBM Plex Mono, monospace" fontSize="16" fill="#1f8059" textAnchor="middle">D̂</text>
              <text x="20" y="224" fontFamily="IBM Plex Sans, sans-serif" fontSize="12" fill="#5b6673">Scaler, Lasso α, bias factor and the rainflow intercept are all refit inside each training fold; nothing is tuned on a held-out file.</text>
            </svg>
        </div>
        <div className="ov-side" style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: "32px", alignItems: "start" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            <div className="callout"><span className="mono" style={{ fontSize: "10px", letterSpacing: ".12em", color: "#006d73" }}>SUMMARY</span><br />Each load cycle contributes to metal fatigue, with larger cycles contributing more. Rainflow counting measures the cycles, a power law weights them, and a sparse linear model adjusts the estimate.</div>
            <div className="kv">
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Input</div><div>Headerless <span className="mono" style={{ fontSize: "12px" }}>.csv</span>, one stress sample per line (581,120 samples); no sample rate given, so spectra use cycles per sample</div>
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Model</div><div>StandardScaler + LassoCV on log-damage over 109 features, multiplicative bias correction, 50 / 50 blend with an m = 5 rainflow estimate when stress skew &gt; 0</div>
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Output</div><div><span className="mono" style={{ fontSize: "12px" }}>file_id, prediction</span> · positive damage, 9 significant digits</div>
            <div className="eyebrow" style={{ paddingTop: "2px" }}>Selection rationale</div><div>The metric is relative error and fatigue is a power law, so log space fits both. The blend was selected in 62 / 64 nested outer folds and cut nested MAPE 0.0203 to 0.0187</div>
          </div>
            <div>
              <div className="eyebrow" style={{ marginBottom: "4px" }}>Architecture &middot; 5 pipeline steps</div>
              <div className="step"><div className="num">1</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Load.</strong> Read as float64; interpolate non-finite samples if they are ≤ 1 % of the record, otherwise reject the file.</div></div>
            <div className="step"><div className="num">2</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Four feature families (109 columns).</strong> <em>stats</em>: ranges, SD, skew / kurtosis, turning points, level crossings. <em>rainflow</em>: 4-point cycles + half-cycle residue, Miner sums for S-N exponents 3 to 12, damage-equivalent loads. <em>spectral</em>: Welch moments, Dirlik, Tovo-Benasciutti. <em>fds</em>: eleven octave SDOF bands.</div></div>
            <div className="step"><div className="num">3</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Regress.</strong> Positive-scale features are log-transformed; StandardScaler + LassoCV predict log-damage inside every training fold; exponentiate and multiply by the training-only bias factor.</div></div>
            <div className="step"><div className="num">4</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Physics blend.</strong> If the stress skew is positive, average 50 / 50 with an m = 5 rainflow Miner estimate whose log intercept was fitted on training data. Negative-skew records keep the Lasso result.</div></div>
            <div className="step"><div className="num">5</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Emit.</strong> Clip to 1e-6 … 1e6; a non-finite result falls back to the training median. The downsampled trace and rainflow explanation in the app never alter the CSV.</div></div>
            </div>
          </div>
          <div style={{ background: "#f7f8fa", borderRadius: "6px", padding: "14px 16px", display: "flex", flexDirection: "column" }}>
            <div className="eyebrow" style={{ marginBottom: "4px" }}>Supporting research</div>
            <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R269</span><span>Zorman, Slavič &amp; Boltežar, <em>MSSP</em> 2023. Vibration fatigue by spectral methods, with the open-source FLife code: the reference implementation our spectral and rainflow features were checked against.</span></div>
            <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R270</span><span>Marsh et al., <em>Int. J. Fatigue</em> 2016. Rainflow residue processing: why the half-cycle residue convention was chosen and why 4-point counting reproduces the labels.</span></div>
            <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R275</span><span>Benasciutti &amp; Tovo, <em>Int. J. Fatigue</em> 2005. Spectral lifetime prediction for wide-band processes: the Tovo-Benasciutti damage feature.</span></div>
            <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R276</span><span>Dirlik &amp; Benasciutti, <em>Metals</em> 2021. Historical review of the Dirlik and TB spectral methods: the Dirlik feature and its limits on non-Gaussian records.</span></div>
            <div className="ref" style={{ borderBottom: "none", fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R274</span><span>Proner &amp; Mucchi, <em>MSSP</em> 2025. Fatigue damage spectrum for random vibration: the eleven-band SDOF FDS family.</span></div>
          </div>
        </div>
      </div>
    </>)}
  </section>
    </>
  );
}
