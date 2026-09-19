// Glossary, team and footer.
// Numbers, dataset counts and reference IDs come from results/ps3, docs/ps3_*.md and
// docs/research/references.md; the markup mirrors the design canvas (docs/design/Main.dc.html).

export default function Outro({  }) {
  return (
    <>
{/* glossary */}
  <section style={{ flex: "none", padding: "32px 120px 24px", display: "flex", flexDirection: "column", gap: "14px" }}>
    <div className="eyebrow">Glossary · the words on this page</div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "12px", fontSize: "13px", lineHeight: "1.45" }}>
      <div className="gl"><div style={{ fontWeight: "600" }}>IoU-weighted F1</div><div style={{ color: "#5b6673" }}>Door metric: a predicted segment counts only if it overlaps a true segment of the same label; the overlap fraction weights the match.</div></div>
      <div className="gl"><div style={{ fontWeight: "600" }}>Rank decay</div><div style={{ color: "#5b6673" }}>ACV metric: (n − (r − 1)) / n where r is the rank given to the true leaking car; 1.0 means it was listed first.</div></div>
      <div className="gl"><div style={{ fontWeight: "600" }}>Macro F1</div><div style={{ color: "#5b6673" }}>Rail metric: F1 of Normal, Side I and Side II averaged equally, so the 14-file class matters as much as the 234-file class.</div></div>
      <div className="gl"><div style={{ fontWeight: "600" }}>MAPE</div><div style={{ color: "#5b6673" }}>SHM metric: mean absolute percentage error; the score is max(0, 1 − MAPE).</div></div>
      <div className="gl"><div style={{ fontWeight: "600" }}>Nested CV</div><div style={{ color: "#5b6673" }}>Model choice happens inside each outer fold, so the reported number was never used to pick the model that produced it.</div></div>
      <div className="gl"><div style={{ fontWeight: "600" }}>Fold-local</div><div style={{ color: "#5b6673" }}>Any quantity learned from data (scaler, threshold, template) is refit on the training fold only; held-out files are transformed, never fitted.</div></div>
      <div className="gl"><div style={{ fontWeight: "600" }}>Mirror TTA</div><div style={{ color: "#5b6673" }}>Swap the train's left and right sides, predict again, swap the Side I / Side II probabilities back and average the two views.</div></div>
      <div className="gl"><div style={{ fontWeight: "600" }}>Coherence</div><div style={{ color: "#5b6673" }}>How much two signals move together at a given frequency (0 to 1). Same-side axle boxes cohere on corrugated rail.</div></div>
    </div>
  </section>

  {/* team */}
  <section style={{ flex: "none", padding: "32px 120px 40px", display: "flex", flexDirection: "column", gap: "24px", borderTop: "1px solid #dfe3e7" }}>
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      <div className="eyebrow">The team · Team Bus MRT Walk (BMW)</div>
      <h2 style={{ margin: "0", fontSize: "28px", fontWeight: "600" }}>Four final-year engineers from the Renaissance Engineering Programme</h2>
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "24px" }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", gap: "14px" }}>
        <img src="/overview/team_sun.png" alt="Portrait of Sun Sitong" style={{ width: "160px", height: "160px", borderRadius: "80px", objectFit: "cover", border: "2px solid #1d2633" }} />
        <div style={{ fontSize: "17px", fontWeight: "600" }}>Sun Sitong</div>
        <div style={{ fontSize: "14px", color: "#5b6673", lineHeight: "1.45" }}>Y4 Computer Science;<br />Renaissance Engineering Programme</div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", gap: "14px" }}>
        <img src="/overview/team_justin.png" alt="Portrait of Justin Timothy Bersamin" style={{ width: "160px", height: "160px", borderRadius: "80px", objectFit: "cover", border: "2px solid #1d2633" }} />
        <div style={{ fontSize: "17px", fontWeight: "600" }}>Justin Timothy Bersamin</div>
        <div style={{ fontSize: "14px", color: "#5b6673", lineHeight: "1.45" }}>Y4 Electronic and Electrical Engineering;<br />Renaissance Engineering Programme</div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", gap: "14px" }}>
        <img src="/overview/team_gan.png" alt="Portrait of Gan Qing Rong" style={{ width: "160px", height: "160px", borderRadius: "80px", objectFit: "cover", border: "2px solid #1d2633" }} />
        <div style={{ fontSize: "17px", fontWeight: "600" }}>Gan Qing Rong</div>
        <div style={{ fontSize: "14px", color: "#5b6673", lineHeight: "1.45" }}>Y4 Computer Science;<br />Renaissance Engineering Programme</div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", gap: "14px" }}>
        <img src="/overview/team_nurdiyanah.png" alt="Portrait of Nurdiyanah Umairah Binte Abdul Rani" style={{ width: "160px", height: "160px", borderRadius: "80px", objectFit: "cover", border: "2px solid #1d2633" }} />
        <div style={{ fontSize: "17px", fontWeight: "600" }}>Nurdiyanah Umairah Binte Abdul Rani</div>
        <div style={{ fontSize: "14px", color: "#5b6673", lineHeight: "1.45" }}>Y4 Civil Engineering;<br />Renaissance Engineering Programme</div>
      </div>
    </div>
  </section>

  <footer style={{ marginTop: "auto", padding: "24px 120px 32px", borderTop: "1px solid #dfe3e7", display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "40px" }}>
    <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px", color: "#5b6673" }}>
      <div className="eyebrow">Evidence</div>
      <div className="mono" style={{ fontSize: "12px", display: "flex", gap: "20px", flexWrap: "wrap" }}><span>results/ps3/leaderboard.md</span><span>docs/ps3_writeup.md</span><span>docs/ps3_model_pipeline.md</span><span>docs/research/references.md</span><span>results/ps3/portal_scores.md</span><span>docs/run_app.md</span></div>
    </div>
    <div className="mono" style={{ fontSize: "11px", color: "#5b6673", textAlign: "right", lineHeight: "1.6" }}>Team Bus MRT Walk (BMW) · Track 3 · PS3<br />python scripts/start_app.py → http://127.0.0.1:8765/</div>
  </footer>
    </>
  );
}
