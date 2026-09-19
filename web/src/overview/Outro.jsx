
// export function Glossary() {
//   return (
//   <section className="ov-sec" style={{ paddingTop: "4px", paddingBottom: "32px", display: "flex", flexDirection: "column", gap: "14px" }}>
//     <div className="ov-g4" style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "12px", fontSize: "13px", lineHeight: "1.45" }}>
//       <div className="gl"><div style={{ fontWeight: "600" }}>IoU-weighted F1</div><div style={{ color: "#5b6673" }}>Door metric: a predicted segment counts only if it overlaps a true segment of the same label; the overlap fraction weights the match.</div></div>
//       <div className="gl"><div style={{ fontWeight: "600" }}>Rank decay</div><div style={{ color: "#5b6673" }}>ACV metric: (n − (r − 1)) / n where r is the rank given to the true leaking car; 1.0 means it was listed first.</div></div>
//       <div className="gl"><div style={{ fontWeight: "600" }}>Macro F1</div><div style={{ color: "#5b6673" }}>Rail metric: F1 of Normal, Side I and Side II averaged equally, so the 14-file class matters as much as the 234-file class.</div></div>
//       <div className="gl"><div style={{ fontWeight: "600" }}>MAPE</div><div style={{ color: "#5b6673" }}>SHM metric: mean absolute percentage error; the score is max(0, 1 − MAPE).</div></div>
//       <div className="gl"><div style={{ fontWeight: "600" }}>Nested CV</div><div style={{ color: "#5b6673" }}>Model selection happens inside each outer fold, so the reported number was never used to select the model that produced it.</div></div>
//       <div className="gl"><div style={{ fontWeight: "600" }}>Fold-local</div><div style={{ color: "#5b6673" }}>Any quantity learned from data (scaler, threshold, template) is refit on the training fold only; held-out files are transformed, never fitted.</div></div>
//       <div className="gl"><div style={{ fontWeight: "600" }}>Mirror TTA</div><div style={{ color: "#5b6673" }}>Swap the train's left and right sides, predict again, swap the Side I / Side II probabilities back and average the two views.</div></div>
//       <div className="gl"><div style={{ fontWeight: "600" }}>Coherence</div><div style={{ color: "#5b6673" }}>How much two signals move together at a given frequency (0 to 1). Same-side axle boxes cohere on corrugated rail.</div></div>
//     </div>
//   </section>
//   );
// }

/** What the project is built with, grouped by layer; versions from the app lock file and package.json. */
const STACK = [
  {
    layer: "Models",
    icon: <path d="M4 19V5M4 19h16M8 15v-4M12 15V7M16 15v-2" />,
    items: [
      ["scikit-learn 1.8", "Door logistic classifiers, SHM Lasso, scalers"],
      ["LightGBM 4.7", "Rail three-seed ensemble; ladder rows for every task"],
      ["NumPy · SciPy 1.17", "Welch spectra, same-side coherence, wavelets, rainflow"],
      ["pandas 2.3 · pyarrow", "loaders, feature tables, cached parquet"],
    ],
    note: "Ladder only: PyTorch, aeon, pyod, Chronos / MOMENT / Mantis embeddings (results/ps3).",
  },
  {
    layer: "Backend",
    icon: <><rect x="3" y="4" width="18" height="6" rx="1.5" /><rect x="3" y="14" width="18" height="6" rx="1.5" /><path d="M7 7h.01M7 17h.01" /></>,
    items: [
      ["Python 3.11", "one package, nebulax/ps3/, shared by the app and predict.py"],
      ["FastAPI 0.141 · Uvicorn", "REST upload sessions, batch predict, streaming preview"],
      ["openpyxl", "ACV workbook inspection and loading"],
      ["pytest", "task, API, CLI and contract tests under tests/"],
    ],
  },
  {
    layer: "Frontend",
    icon: <><rect x="2" y="4" width="20" height="13" rx="2" /><path d="M8 21h8M12 17v4" /></>,
    items: [
      ["React 19 · Vite 8", "overview page and the fixed-board prediction console"],
      ["three.js · react-three-fiber", "the orbitable 8-car train, components tinted by prediction"],
      ["IBM Plex Sans / Mono", "one type family across page, console and charts"],
      ["Inline SVG", "every diagram and chart on this page, no chart library"],
    ],
  },
  {
    layer: "Assets & delivery",
    icon: <><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z" /><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" /></>,
    items: [
      ["Blender (headless)", "parametric train model → r151.glb and the renders on this page"],
      ["Docker Compose", "one image: Node build stage + Python runtime"],
      ["Google Cloud Run", "the deployed app, plus a Cloud Run Job for Cloud Storage batches"]
    ],
  },
];

function StackCard({ layer, icon, items, note }) {
  return (
    <div className="card" style={{ padding: "18px 20px", display: "flex", flexDirection: "column", gap: "12px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "15px", fontWeight: "600" }}>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{icon}</svg>
        {layer}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        {items.map(([name, what]) => (
          <div key={name} style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
            <span className="mono" style={{ fontSize: "12px", color: "#1d2633" }}>{name}</span>
            <span style={{ fontSize: "12px", color: "#5b6673", lineHeight: "1.45" }}>{what}</span>
          </div>
        ))}
      </div>
      {note && <div style={{ fontSize: "11px", color: "#5b6673", lineHeight: "1.45", paddingTop: "6px", borderTop: "1px solid #eef1f3" }}>{note}</div>}
    </div>
  );
}

export default function Outro({ openTwin }) {
  return (
    <>

  {/* tech stack */}
  <section id="stack" className="ov-sec" style={{ paddingTop: "40px", paddingBottom: "40px", borderTop: "1px solid #dfe3e7", display: "flex", flexDirection: "column", gap: "20px" }}>
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      {/* <div className="eyebrow">Tech stack</div> */}
      <h2 style={{ margin: "0", fontSize: "28px", fontWeight: "600" }}>Tech Stack</h2>
      {/* <p style={{ margin: "0", fontSize: "15px", color: "#5b6673", maxWidth: "820px", lineHeight: "1.55" }}>Small, inspectable models on a standard Python scientific stack, served by one FastAPI process that also hosts the React build. Everything runs offline on a laptop after the first install.</p> */}
    </div>
    <div className="ov-g4" style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "16px", alignItems: "stretch" }}>
      {STACK.map((s) => <StackCard key={s.layer} {...s} />)}
    </div>
  </section>

  {/* closing call to action */}
  <section className="ov-sec" style={{ paddingTop: "40px", paddingBottom: "40px", borderTop: "1px solid #dfe3e7", display: "flex", flexDirection: "column", alignItems: "flex-start", gap: "14px" }}>
    <div className="eyebrow">Try it</div>
    <h2 style={{ margin: "0", fontSize: "28px", fontWeight: "600" }}>Run a Test file</h2>
    <p style={{ margin: "0", fontSize: "15px", color: "#5b6673", maxWidth: "620px", lineHeight: "1.55" }}>Pick a system, drop the organiser Test file or folder, press RUN. The workspace is built for a desktop browser.</p>
    <a href="?page=predict" onClick={openTwin} className="btn primary" style={{ textDecoration: "none" }}>Open the prediction workspace</a>
  </section>

  {/* team */}
  <section className="ov-sec" style={{ paddingTop: "32px", paddingBottom: "40px", display: "flex", flexDirection: "column", gap: "24px", borderTop: "1px solid #dfe3e7" }}>
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      <div className="eyebrow">The team</div>
      <h2 style={{ margin: "0", fontSize: "28px", fontWeight: "600" }}>Team Bus MRT Walk (BMW)</h2>
    </div>
    <div className="ov-g4" style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "24px" }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", gap: "14px" }}>
        <img src="/overview/team_sun.png" alt="Portrait of Sun Sitong" className="ov-avatar" style={{ width: "160px", height: "160px", borderRadius: "80px", objectFit: "cover", border: "2px solid #1d2633" }} />
        <div style={{ fontSize: "17px", fontWeight: "600" }}>Sun Sitong</div>
        <div style={{ fontSize: "14px", color: "#5b6673", lineHeight: "1.45" }}>Y4 Computer Science;<br />Renaissance Engineering Programme</div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", gap: "14px" }}>
        <img src="/overview/team_justin.png" alt="Portrait of Justin Timothy Bersamin" className="ov-avatar" style={{ width: "160px", height: "160px", borderRadius: "80px", objectFit: "cover", border: "2px solid #1d2633" }} />
        <div style={{ fontSize: "17px", fontWeight: "600" }}>Justin Timothy Bersamin</div>
        <div style={{ fontSize: "14px", color: "#5b6673", lineHeight: "1.45" }}>Y4 Electronic and Electrical Engineering;<br />Renaissance Engineering Programme</div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", gap: "14px" }}>
        <img src="/overview/team_gan.png" alt="Portrait of Gan Qing Rong" className="ov-avatar" style={{ width: "160px", height: "160px", borderRadius: "80px", objectFit: "cover", border: "2px solid #1d2633" }} />
        <div style={{ fontSize: "17px", fontWeight: "600" }}>Gan Qing Rong</div>
        <div style={{ fontSize: "14px", color: "#5b6673", lineHeight: "1.45" }}>Y4 Computer Science;<br />Renaissance Engineering Programme</div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", gap: "14px" }}>
        <img src="/overview/team_nurdiyanah.png" alt="Portrait of Nurdiyanah Umairah Binte Abdul Rani" className="ov-avatar" style={{ width: "160px", height: "160px", borderRadius: "80px", objectFit: "cover", border: "2px solid #1d2633" }} />
        <div style={{ fontSize: "17px", fontWeight: "600" }}>Nurdiyanah Umairah Binte Abdul Rani</div>
        <div style={{ fontSize: "14px", color: "#5b6673", lineHeight: "1.45" }}>Y4 Civil Engineering;<br />Renaissance Engineering Programme</div>
      </div>
    </div>
  </section>

  <footer className="ov-sec ov-footer" style={{ paddingTop: "24px", paddingBottom: "32px", borderTop: "1px solid #dfe3e7" }}>
    <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px", color: "#5b6673" }}>
      <div className="eyebrow">References and results</div>
      <div className="mono" style={{ fontSize: "12px", display: "flex", gap: "20px", flexWrap: "wrap" }}><span>results/ps3/leaderboard.md</span><span>docs/ps3_writeup.md</span><span>docs/ps3_model_pipeline.md</span><span>docs/research/references.md</span><span>results/ps3/portal_scores.md</span><span>docs/run_app.md</span></div>
    </div>
    <div className="mono" style={{ fontSize: "11px", color: "#5b6673", textAlign: "right", lineHeight: "1.6" }}>Team Bus MRT Walk (BMW) · Track 3 · PS3</div>
  </footer>
    </>
  );
}
