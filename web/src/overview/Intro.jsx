// Hero, headline metrics, the four-step pipeline, the cab animation and the two rules.
// Numbers, dataset counts and reference IDs come from results/ps3, docs/ps3_*.md and
// docs/research/references.md; the markup mirrors the design canvas (docs/design/Main.dc.html).

export default function Intro({ openTwin, openTour }) {
  return (
    <>
{/* hero: short intro + single cab */}
  <section id="top" style={{ flex: "none", position: "relative", height: "580px", overflow: "hidden", background: "#ffffff", borderBottom: "1px solid #dfe3e7" }}>
    <img src="/overview/cab_iso.png" alt="Front cab of the metro train model" style={{ position: "absolute", left: "630px", top: "100px", width: "1000px", height: "auto" }} />
    <div style={{ position: "absolute", right: "0", top: "0", width: "160px", height: "580px", background: "linear-gradient(90deg, rgba(255,255,255,0), #ffffff)" }}></div>
    <div style={{ position: "absolute", left: "120px", top: "104px", width: "470px", display: "flex", flexDirection: "column", gap: "18px" }}>
      <div className="eyebrow">Team Bus MRT Walk · Problem Statement 3</div>
      <h1 style={{ margin: "0", fontSize: "40px", lineHeight: "1.15", fontWeight: "600", letterSpacing: "-.01em" }}>A digital twin for a metro train's condition monitoring</h1>
      <p style={{ margin: "0", fontSize: "17px", lineHeight: "1.55", color: "#5b6673" }}>The Train Digital Twin is a virtual copy of the train that runs the same four fault-detection models we submit: doors, air-conditioning, rail corrugation and structural fatigue. Upload a released Test file, press RUN, and watch the component light up on the train model with an explanation for every row.</p>
      <div style={{ display: "flex", gap: "12px", marginTop: "4px" }}>
        <a href="?page=predict" onClick={openTwin} className="btn primary" style={{ textDecoration: "none" }}>Open Digital Twin</a>
        <a href="?page=predict&tour=1" onClick={openTour} className="btn" style={{ textDecoration: "none" }}>Take the 2-minute tour</a>
      </div>
      <div className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>Runs offline · app and predict.py call the same functions</div>
    </div>
  </section>

  {/* headline metrics */}
  <section style={{ flex: "none", padding: "32px 120px 8px", display: "flex", flexDirection: "column", gap: "12px" }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
      <div className="eyebrow">Honest headlines · nested / outer CV on training data only</div>
      <div style={{ fontSize: "12px", color: "#5b6673" }}>Every number names its JSON key in <span className="mono">results/ps3/leaderboard.md</span></div>
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "12px" }}>
      <div className="card" style={{ padding: "18px 20px", display: "flex", flexDirection: "column", gap: "6px" }}>
        <div className="eyebrow">Door · IoU-weighted F1</div>
        <div className="mono" style={{ fontSize: "34px", fontWeight: "500", lineHeight: "1" }}>0.9818</div>
        <div style={{ fontSize: "13px", color: "#5b6673" }}>± 0.0364 · 5 contiguous outer blocks</div>
        <div style={{ fontSize: "12px", marginTop: "4px" }}>Separate Open / Close logistic, 24 physics features</div>
      </div>
      <div className="card" style={{ padding: "18px 20px", display: "flex", flexDirection: "column", gap: "6px" }}>
        <div className="eyebrow">ACV · rank decay</div>
        <div className="mono" style={{ fontSize: "34px", fontWeight: "500", lineHeight: "1" }}>0.9792</div>
        <div style={{ fontSize: "13px", color: "#5b6673" }}>± 0.0510 · leave-one-case-out, 6 cases</div>
        <div style={{ fontSize: "12px", marginTop: "4px" }}>Pre-registered hot-cooling peer-delta rule</div>
      </div>
      <div className="card" style={{ padding: "18px 20px", display: "flex", flexDirection: "column", gap: "6px" }}>
        <div className="eyebrow">Rail · macro F1</div>
        <div className="mono" style={{ fontSize: "34px", fontWeight: "500", lineHeight: "1" }}>0.8051</div>
        <div style={{ fontSize: "13px", color: "#5b6673" }}>± 0.1291 · 15 grouped folds · portal 0.8310</div>
        <div style={{ fontSize: "12px", marginTop: "4px" }}>3-seed LightGBM with same-side axle-box coherence</div>
      </div>
      <div className="card" style={{ padding: "18px 20px", display: "flex", flexDirection: "column", gap: "6px" }}>
        <div className="eyebrow">SHM · 1 − MAPE</div>
        <div className="mono" style={{ fontSize: "34px", fontWeight: "500", lineHeight: "1" }}>0.9813</div>
        <div style={{ fontSize: "13px", color: "#5b6673" }}>MAPE 0.0187 · leave-one-file-out, 64 files · portal 0.9728</div>
        <div style={{ fontSize: "12px", marginTop: "4px" }}>Log-damage Lasso + positive-skew rainflow blend</div>
      </div>
    </div>
  </section>

  {/* how the twin works: 4 steps */}
  <section style={{ flex: "none", padding: "40px 120px 32px", display: "flex", flexDirection: "column", gap: "24px" }}>
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      <div className="eyebrow">How the digital twin works</div>
      <h2 style={{ margin: "0", fontSize: "28px", fontWeight: "600" }}>Four steps from a raw Test file to a prediction on the train</h2>
      <p style={{ margin: "0", fontSize: "15px", color: "#5b6673", maxWidth: "820px", lineHeight: "1.55" }}>The same four functions run whether you press RUN in the browser or call <span className="mono" style={{ fontSize: "13px" }}>python predict.py --input …</span>. The table on screen is the CSV you download, and the train model lights up the component behind the selected row.</p>
    </div>

    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "20px" }}>
      <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <svg width="285" height="150" viewBox="0 0 285 150" style={{ width: "100%", height: "150px", background: "#f1f3f5" }} aria-hidden="true">
          <rect x="34" y="30" width="72" height="90" rx="6" fill="#ffffff" stroke="#b8c4cc"></rect>
          <rect x="52" y="22" width="72" height="90" rx="6" fill="#ffffff" stroke="#b8c4cc"></rect>
          <rect x="70" y="14" width="72" height="90" rx="6" fill="#ffffff" stroke="#8a97a3"></rect>
          <path d="M120 14v20h22" fill="none" stroke="#8a97a3"></path>
          <text x="82" y="56" fontFamily="IBM Plex Mono, monospace" fontSize="11" fill="#1d2633">.csv</text>
          <text x="82" y="72" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#5b6673">10 kHz</text>
          <text x="82" y="86" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#5b6673">129 col</text>
          <path d="M160 64h44" stroke="#1a4fa3" strokeWidth="2"></path><path d="M198 58l8 6-8 6" fill="none" stroke="#1a4fa3" strokeWidth="2"></path>
          <rect x="214" y="40" width="52" height="48" rx="6" fill="#ffffff" stroke="#1a4fa3" strokeWidth="1.5"></rect>
          <text x="240" y="60" fontFamily="IBM Plex Mono, monospace" fontSize="10" fill="#1a4fa3" textAnchor="middle">QUEUE</text>
          <text x="240" y="78" fontFamily="IBM Plex Mono, monospace" fontSize="12" fill="#1d2633" textAnchor="middle">68 / 68</text>
          <rect x="34" y="130" width="232" height="6" rx="3" fill="#2f62c4"></rect>
        </svg>
        <div style={{ padding: "16px 18px 18px", display: "flex", flexDirection: "column", gap: "8px" }}>
          <div className="mono" style={{ fontSize: "11px", color: "#1a4fa3" }}>01 · UPLOAD</div>
          <div style={{ fontSize: "16px", fontWeight: "600" }}>Pick a subsystem, queue files</div>
          <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>One Door stream, one ACV workbook, 68 Rail files or 16 SHM files. The browser sends at most 32 files per request against one session, so the table fills as batches return.</div>
          <div className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>web/src/predict/</div>
        </div>
      </div>
      <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <svg width="285" height="150" viewBox="0 0 285 150" style={{ width: "100%", height: "150px", background: "#f1f3f5" }} aria-hidden="true">
          <rect x="20" y="22" width="150" height="106" rx="4" fill="#ffffff" stroke="#b8c4cc"></rect>
          <rect x="20" y="22" width="150" height="20" rx="4" fill="#eef1f3"></rect>
          <text x="28" y="36" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#5b6673">Time</text>
          <text x="62" y="36" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1a4fa3">Car 3 - Indoor Avg T</text>
          <line x1="20" y1="62" x2="170" y2="62" stroke="#eef1f3"></line><line x1="20" y1="82" x2="170" y2="82" stroke="#eef1f3"></line><line x1="20" y1="102" x2="170" y2="102" stroke="#eef1f3"></line>
          <text x="28" y="56" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633">14:02:10</text><text x="96" y="56" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633">27.9</text>
          <text x="28" y="76" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633">14:02:11</text><text x="96" y="76" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633">28.1</text>
          <text x="28" y="96" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633">14:02:12</text><text x="96" y="96" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#a66500">NaN</text>
          <text x="28" y="116" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633">14:02:13</text><text x="96" y="116" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633">28.0</text>
          <path d="M178 40h18" stroke="#1a4fa3" strokeWidth="2"></path><path d="M191 34l6 6-6 6" fill="none" stroke="#1a4fa3" strokeWidth="2"></path>
          <rect x="204" y="26" width="70" height="28" rx="4" fill="#ffffff" stroke="#1a4fa3"></rect>
          <text x="239" y="38" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1a4fa3" textAnchor="middle">canonical</text>
          <text x="239" y="48" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1d2633" textAnchor="middle">indoor_temp</text>
          <text x="204" y="78" fontFamily="IBM Plex Sans, sans-serif" fontSize="9" fill="#5b6673">ms timestamps</text>
          <text x="204" y="94" fontFamily="IBM Plex Sans, sans-serif" fontSize="9" fill="#5b6673">alias mapping</text>
          <text x="204" y="110" fontFamily="IBM Plex Sans, sans-serif" fontSize="9" fill="#5b6673">unknowns reported</text>
        </svg>
        <div style={{ padding: "16px 18px 18px", display: "flex", flexDirection: "column", gap: "8px" }}>
          <div className="mono" style={{ fontSize: "11px", color: "#1a4fa3" }}>02 · LOAD</div>
          <div style={{ fontSize: "16px", fontWeight: "600" }}>Schema-aware loader</div>
          <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>Parses timestamps to the millisecond, discovers <span className="mono" style={{ fontSize: "12px" }}>Car N - parameter</span> columns by pattern, maps header aliases to canonical fields and reports unknown columns instead of guessing.</div>
          <div className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>nebulax/ps3/&lt;task&gt;.py</div>
        </div>
      </div>
      <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <svg width="285" height="150" viewBox="0 0 285 150" style={{ width: "100%", height: "150px", background: "#f1f3f5" }} aria-hidden="true">
          <polyline fill="none" stroke="#1d2633" strokeWidth="1.5" points="18,80 26,60 34,96 42,52 50,88 58,70 66,98 74,44 82,86 90,64 98,92 106,56 114,84 122,72 130,90 138,50 146,82"></polyline>
          <text x="18" y="124" fontFamily="IBM Plex Sans, sans-serif" fontSize="9" fill="#5b6673">raw signal (10 kHz)</text>
          <path d="M156 74h18" stroke="#1a4fa3" strokeWidth="2"></path><path d="M169 68l6 6-6 6" fill="none" stroke="#1a4fa3" strokeWidth="2"></path>
          <rect x="184" y="34" width="84" height="8" rx="4" fill="#b3c7ea"></rect><rect x="184" y="34" width="60" height="8" rx="4" fill="#2f62c4"></rect>
          <rect x="184" y="50" width="84" height="8" rx="4" fill="#b3c7ea"></rect><rect x="184" y="50" width="30" height="8" rx="4" fill="#2f62c4"></rect>
          <rect x="184" y="66" width="84" height="8" rx="4" fill="#b3c7ea"></rect><rect x="184" y="66" width="72" height="8" rx="4" fill="#2f62c4"></rect>
          <rect x="184" y="82" width="84" height="8" rx="4" fill="#b3c7ea"></rect><rect x="184" y="82" width="44" height="8" rx="4" fill="#2f62c4"></rect>
          <text x="184" y="108" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#5b6673">RMS  band  coherence</text>
          <text x="184" y="124" fontFamily="IBM Plex Sans, sans-serif" fontSize="9" fill="#5b6673">201 features per file</text>
        </svg>
        <div style={{ padding: "16px 18px 18px", display: "flex", flexDirection: "column", gap: "8px" }}>
          <div className="mono" style={{ fontSize: "11px", color: "#1a4fa3" }}>03 · FEATURISE</div>
          <div style={{ fontSize: "16px", fontWeight: "600" }}>Physics features</div>
          <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>Door-cycle currents, peer temperature deltas, axle-box spectra and coherence, rainflow damage sums. Baselines, templates and scalers always come from training data only.</div>
          <div className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>nebulax/ps3/&lt;task&gt;_features.py</div>
        </div>
      </div>
      <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden", borderColor: "#1a4fa3" }}>
        <svg width="285" height="150" viewBox="0 0 285 150" style={{ width: "100%", height: "150px", background: "#f1f3f5" }} aria-hidden="true">
          <rect x="18" y="52" width="70" height="46" rx="6" fill="#ffffff" stroke="#8a97a3"></rect>
          <text x="53" y="72" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#5b6673" textAnchor="middle">features</text>
          <text x="53" y="86" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#1d2633" textAnchor="middle">x₁ … xₙ</text>
          <path d="M92 75h18" stroke="#1a4fa3" strokeWidth="2"></path><path d="M105 69l6 6-6 6" fill="none" stroke="#1a4fa3" strokeWidth="2"></path>
          <rect x="116" y="40" width="84" height="70" rx="6" fill="#ffffff" stroke="#1a4fa3" strokeWidth="1.5"></rect>
          <text x="158" y="60" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="#1a4fa3" textAnchor="middle">models/ps3/</text>
          <text x="158" y="74" fontFamily="IBM Plex Mono, monospace" fontSize="10" fill="#1d2633" textAnchor="middle">door.pkl</text>
          <text x="158" y="92" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#5b6673" textAnchor="middle">frozen artifact</text>
          <path d="M204 75h18" stroke="#1a4fa3" strokeWidth="2"></path><path d="M217 69l6 6-6 6" fill="none" stroke="#1a4fa3" strokeWidth="2"></path>
          <rect x="228" y="46" width="44" height="22" rx="11" fill="#e6f1ea" stroke="#1f8059"></rect>
          <text x="250" y="61" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#1f8059" textAnchor="middle">Normal</text>
          <rect x="228" y="82" width="44" height="22" rx="11" fill="#fbe9e8" stroke="#c43d36"></rect>
          <text x="250" y="97" fontFamily="IBM Plex Mono, monospace" fontSize="8" fill="#c43d36" textAnchor="middle">Abnorm.</text>
          <text x="18" y="128" fontFamily="IBM Plex Sans, sans-serif" fontSize="9" fill="#5b6673">one row per cycle / car / file, then the organiser CSV</text>
        </svg>
        <div style={{ padding: "16px 18px 18px", display: "flex", flexDirection: "column", gap: "8px" }}>
          <div className="mono" style={{ fontSize: "11px", color: "#1a4fa3" }}>04 · PREDICT</div>
          <div style={{ fontSize: "16px", fontWeight: "600" }}>Saved artifact, frozen</div>
          <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>Each <span className="mono" style={{ fontSize: "12px" }}>.pkl</span> holds the fitted model and its preprocessing state; the adjacent <span className="mono" style={{ fontSize: "12px" }}>.json</span> records the configuration and honest headline. The output is validated against the organiser schema before download.</div>
          <div className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>models/ps3/*.pkl · nebulax/ps3/submission.py</div>
        </div>
      </div>
    </div>

    {/* animated: where each subsystem lives on one cab */}
    <div className="card" style={{ overflow: "hidden" }}>
      <div style={{ padding: "20px 24px 0", display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <div className="eyebrow">Where each subsystem lives · one cab, cycling every 4 seconds</div>
        <div style={{ fontSize: "12px", color: "#5b6673" }}>The component behind the selected row lights up green (normal) or red (fault)</div>
      </div>
      <div style={{ position: "relative", width: "1200px", height: "300px", overflow: "hidden" }}>
        <img src="/overview/cab_elev.png" alt="Side elevation of the first cab of the train model" style={{ position: "absolute", left: "0", top: "0", width: "1200px", height: "260px" }} />
        <div className="spot"></div>
        <div className="cap cap1"><span className="chip" style={{ background: "#1a4fa3", color: "#ffffff" }}>DOOR</span><span style={{ fontSize: "13px" }}>Door leaves on each car side: motor current per open / close cycle</span></div>
        <div className="cap cap2"><span className="chip" style={{ background: "#1a4fa3", color: "#ffffff" }}>ACV</span><span style={{ fontSize: "13px" }}>Roof-mounted air-conditioning unit: cabin temperature against the sibling cars</span></div>
        <div className="cap cap3"><span className="chip" style={{ background: "#1a4fa3", color: "#ffffff" }}>RAIL</span><span style={{ fontSize: "13px" }}>Axle boxes on every bogie: 64 vibration channels for corrugation</span></div>
        <div className="cap cap4"><span className="chip" style={{ background: "#1a4fa3", color: "#ffffff" }}>SHM</span><span style={{ fontSize: "13px" }}>Structural member under the floor: fatigue damage from its stress record</span></div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "16px", padding: "0 24px 20px" }}>
        <div style={{ position: "relative", borderRadius: "6px", overflow: "hidden", background: "#f1f3f5" }}>
          <img src="/overview/view_door.png" alt="Close-up render of a door" style={{ width: "100%", height: "140px", objectFit: "cover", display: "block" }} />
          <div className="mono" style={{ position: "absolute", left: "10px", bottom: "8px", fontSize: "11px", background: "rgba(255,255,255,.9)", padding: "3px 8px", borderRadius: "4px" }}>01 · DOOR</div>
          <div className="ring ring1"></div>
        </div>
        <div style={{ position: "relative", borderRadius: "6px", overflow: "hidden", background: "#f1f3f5" }}>
          <img src="/overview/cab_roof.png" alt="Close-up render of the roof units" style={{ width: "100%", height: "140px", objectFit: "cover", display: "block" }} />
          <div className="mono" style={{ position: "absolute", left: "10px", bottom: "8px", fontSize: "11px", background: "rgba(255,255,255,.9)", padding: "3px 8px", borderRadius: "4px" }}>02 · ACV</div>
          <div className="ring ring2"></div>
        </div>
        <div style={{ position: "relative", borderRadius: "6px", overflow: "hidden", background: "#f1f3f5" }}>
          <img src="/overview/view_bogie.png" alt="Close-up render of a bogie" style={{ width: "100%", height: "140px", objectFit: "cover", display: "block" }} />
          <div className="mono" style={{ position: "absolute", left: "10px", bottom: "8px", fontSize: "11px", background: "rgba(255,255,255,.9)", padding: "3px 8px", borderRadius: "4px" }}>03 · RAIL</div>
          <div className="ring ring3"></div>
        </div>
        <div style={{ position: "relative", borderRadius: "6px", overflow: "hidden", background: "#f1f3f5" }}>
          <img src="/overview/cab_underframe.png" alt="Close-up render of the underframe" style={{ width: "100%", height: "140px", objectFit: "cover", display: "block" }} />
          <div className="mono" style={{ position: "absolute", left: "10px", bottom: "8px", fontSize: "11px", background: "rgba(255,255,255,.9)", padding: "3px 8px", borderRadius: "4px" }}>04 · SHM</div>
          <div className="ring ring4"></div>
        </div>
      </div>
    </div>

    <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "16px" }}>
      <div className="card" style={{ padding: "18px 20px", display: "flex", flexDirection: "column", gap: "8px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "15px", fontWeight: "600" }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1a4fa3" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
          The fold-local rule
        </div>
        <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>Every scaler, threshold, template, feature selection, bias correction and augmentation is fitted on the training partition only. A held-out file is transformed once and never augmented. Oracle or transductive rows are labelled diagnostics, never headlines.</div>
      </div>
      <div className="card" style={{ padding: "18px 20px", display: "flex", flexDirection: "column", gap: "8px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "15px", fontWeight: "600" }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1a4fa3" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5"></path></svg>
          Submission integrity
        </div>
        <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}><span className="mono" style={{ fontSize: "12px" }}>submission/nebulax/predictions.zip</span> holds the four root-level CSVs produced by this pipeline. The organiser Test inputs are unlabelled; no Test label is ever used. Portal scores and CSV hashes are logged in <span className="mono" style={{ fontSize: "12px" }}>results/ps3/portal_scores.md</span>.</div>
      </div>
    </div>
  </section>
    </>
  );
}
