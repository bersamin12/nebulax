// Exploratory systems carousel: bogie / axle bearing and brake air supply.
// Numbers, dataset counts and reference IDs come from results/ps3, docs/ps3_*.md and
// docs/research/references.md; the markup mirrors the design canvas (docs/design/Main.dc.html).
import { useState } from "react";

export default function Extra({  }) {
  const [x, setX] = useState(0);
  return (
    <>
{/* Additional systems carousel */}
  <section style={{ flex: "none", padding: "8px 120px 40px", display: "flex", flexDirection: "column", gap: "20px" }}>
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      <div className="eyebrow">Research datasets · outside PS3 scoring</div>
      <h2 style={{ margin: "0", fontSize: "28px", fontWeight: "600" }}>Additional condition-monitoring datasets</h2>
      <p style={{ margin: "0", fontSize: "15px", color: "#5b6673", maxWidth: "900px", lineHeight: "1.55" }}>These public datasets support exploratory work on axle bearings and brake-air supply. They are not part of PS3 scoring. In the prediction workspace, each appears as a read-only dataset profile with a short recorded example.</p>
    </div>

    <div style={{ display: "flex", alignItems: "flex-end", gap: "12px" }}>
      <button type="button" className="arrow" onClick={() => setX((x + 1) % 2)} aria-label="Previous system"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M15 18l-6-6 6-6"></path></svg></button>
      <div style={{ flexGrow: "1", display: "flex", gap: "8px" }}>
        <button type="button" className={"tab" + (x === 0 ? " on" : "")} onClick={() => setX(0)}><span className="mono" style={{ fontSize: "11px", letterSpacing: ".12em", color: "#1a4fa3" }}>05 · BOGIE · AXLE BEARING</span><span style={{ fontSize: "14px", fontWeight: "600" }}>Bearing condition from vibration</span><span className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>UORED-VAFCLS · 0.828 macro F1 · bearing-grouped</span></button>
        <button type="button" className={"tab" + (x === 1 ? " on" : "")} onClick={() => setX(1)}><span className="mono" style={{ fontSize: "11px", letterSpacing: ".12em", color: "#1a4fa3" }}>06 · BRAKE AIR SUPPLY</span><span style={{ fontSize: "14px", fontWeight: "600" }}>Compressor air-leak detection</span><span className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>MetroPT-3 · 4 / 4 leaks · 0.105 false alarms per day</span></button>
      </div>
      <button type="button" className="arrow" onClick={() => setX((x + 1) % 2)} aria-label="Next system"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 6l6 6-6 6"></path></svg></button>
    </div>

    {/* page 1 BOGIE */}
    {x === 0 && (<>
      <div className="card" style={{ padding: "28px 32px 32px", display: "flex", flexDirection: "column", gap: "24px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "440px 1fr", gap: "32px", alignItems: "start" }}>
          <div style={{ position: "relative", borderRadius: "6px", overflow: "hidden", background: "#f1f3f5" }}>
            <img src="/overview/view_bogie.png" alt="Close-up render of a bogie with axle boxes under the train model" style={{ width: "100%", height: "300px", objectFit: "cover", display: "block" }} />
            <div className="mono" style={{ position: "absolute", left: "12px", bottom: "10px", fontSize: "11px", background: "rgba(255,255,255,.92)", padding: "4px 8px", borderRadius: "4px" }}>axle boxes 1L to 4R on every bogie</div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}><span className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>page 1 / 2</span><span className="chip">research benchmark · nebulax/sim/bearing.py</span></div>
            <h3 style={{ margin: "0", fontSize: "24px", fontWeight: "600", lineHeight: "1.2" }}>Rolling-element bearing condition from axle-box vibration</h3>
            <div className="callout"><span className="mono" style={{ fontSize: "10px", letterSpacing: ".12em", color: "#006d73" }}>SUMMARY</span><br />A developing bearing fault produces characteristic vibration frequencies and additional heat. The research benchmark compares those vibration patterns with healthy recordings from a public laboratory dataset.</div>
            <div className="kv" style={{ gridTemplateColumns: "120px 1fr" }}>
              <div className="eyebrow" style={{ paddingTop: "2px" }}>Dataset</div><div>UORED-VAFCLS, University of Ottawa (Mendeley Data, CC BY): 20 bearings × 3 conditions = 60 recordings of 10 s, accelerometer + microphone + load cell + hall-effect recorded together</div>
              <div className="eyebrow" style={{ paddingTop: "2px" }}>Labels</div><div>healthy / fault-developing / faulty, one label per bearing, constant load and speed</div>
              <div className="eyebrow" style={{ paddingTop: "2px" }}>In the workspace</div><div>Open Axle bearing to view a recorded vibration excerpt and its location on the train. This view does not run a PS3 prediction.</div>
              <div className="eyebrow" style={{ paddingTop: "2px" }}>Simulator</div><div>8 boxes per car at 1 Hz: Palmgren friction torque feeds one thermal node per box (steady rise scales as v<sup>0.43</sup>, calibrated to 20 to 26 K at 80 km/h); vibration is a direct feature stream, no waveform</div>
            </div>
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1.15fr 1fr 1fr", gap: "24px", alignItems: "start" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}><div className="eyebrow">Model comparison · macro F1</div><div className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>bearing-grouped 5-fold</div></div>
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              <div className="bar" style={{ gridTemplateColumns: "190px 1fr 52px" }}><div>logreg on envelope feat.</div><div className="track" style={{ height: "14px" }}><div className="fill sel" style={{ width: "82.8%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>0.828</div></div>
              <div className="bar" style={{ gridTemplateColumns: "190px 1fr 52px" }}><div>LightGBM on envelope feat.</div><div className="track" style={{ height: "14px" }}><div className="fill" style={{ width: "81.5%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>0.815</div></div>
              <div className="bar" style={{ gridTemplateColumns: "190px 1fr 52px" }}><div>QUANT, raw window</div><div className="track" style={{ height: "14px" }}><div className="fill" style={{ width: "63.2%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>0.632</div></div>
              <div className="bar" style={{ gridTemplateColumns: "190px 1fr 52px" }}><div>LITETime, raw window</div><div className="track" style={{ height: "14px" }}><div className="fill" style={{ width: "61.5%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>0.615</div></div>
              <div className="bar" style={{ gridTemplateColumns: "190px 1fr 52px" }}><div>MultiRocket-Hydra, raw</div><div className="track" style={{ height: "14px" }}><div className="fill" style={{ width: "60.3%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>0.603</div></div>
            </div>
            <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>Hand-built envelope features (time + envelope spectrum, 0.25 s windows) beat every raw-series model once no bearing is allowed in both train and test. The raw-series rows also cost 50 to 500 s to fit against 0.1 s for the logistic model.</div>
            <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>Anomaly detection on the same split: PCA SPE / T² reaches AUROC 0.983, Mahalanobis 0.977, Isolation Forest 0.960. The scored slice is two-thirds positive, so we judge those rows on AUROC, not on VUS-PR.</div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            <div className="eyebrow">How the benchmark scores it</div>
            <div className="step" style={{ padding: "4px 0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>a</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Bearing-grouped folds.</strong> All three recordings of a bearing stay in one fold; random window splits inflate accuracy from a genuine 20 to 60 % to 99.9 % on the same data.</div></div>
            <div className="step" style={{ padding: "4px 0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>b</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Normal-only training.</strong> Detectors see healthy recordings only; thresholds come from the validation slice, never from test.</div></div>
            <div className="step" style={{ padding: "4px 0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>c</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Synthetic fleet replay.</strong> On the simulated bearing fleet (leave-one-train-out) <span className="mono" style={{ fontSize: "12px" }}>cusum_cycle_scalar</span> scores VUS-PR 0.658, 10 / 12 injected events, 0.008 false-alarm episodes per scored day.</div></div>
            <div className="step" style={{ padding: "4px 0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>d</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Severity constraint.</strong> Published rig data shows the axle box gains about 1 K per severity step against an 8 K healthy spread, so temperature alone cannot flag a defect; only an acute hot axle box drives it hard.</div></div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div style={{ background: "#f7f8fa", borderRadius: "6px", padding: "14px 16px", display: "flex", flexDirection: "column", gap: "8px" }}>
              <div className="eyebrow">Limitations</div>
              <div style={{ fontSize: "13px", lineHeight: "1.5" }}>Laboratory proxy: constant load and speed, no temperature channel, and a fabricated timeline. Read the classification numbers as evidence; do not read the synthetic false-alarm clock as field performance.</div>
              <div style={{ fontSize: "13px", lineHeight: "1.5" }}>Single seed everywhere on the research leaderboard: no interval on it is a confidence interval.</div>
            </div>
            <div style={{ background: "#f7f8fa", borderRadius: "6px", padding: "14px 16px", display: "flex", flexDirection: "column" }}>
              <div className="eyebrow" style={{ marginBottom: "4px" }}>Sources and next data</div>
              <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R119</span><span>Sehri, Dumond et al., <em>Data in Brief</em> 2023. The UORED-VAFCLS data paper; per-bearing structure is what makes bearing-wise splitting possible.</span></div>
              <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R115</span><span>Vieira et al., <em>MSSP</em> 2026. Why bearing fault diagnosis must be evaluated with bearing-disjoint splits.</span></div>
              <div className="ref" style={{ borderBottom: "none", fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R118</span><span>uOttawa variable-speed set with an encoder channel: the planned primary, because a metro axle box never runs at constant speed. CITEF railway axle-box rig (R140) adds graded severity.</span></div>
            </div>
          </div>
        </div>
      </div>
    </>)}

    {/* page 2 BRAKE AIR SUPPLY */}
    {x === 1 && (<>
      <div className="card" style={{ padding: "28px 32px 32px", display: "flex", flexDirection: "column", gap: "24px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "440px 1fr", gap: "32px", alignItems: "start" }}>
          <div style={{ position: "relative", borderRadius: "6px", overflow: "hidden", background: "#f1f3f5" }}>
            <img src="/overview/view_apu.png" alt="Close-up render of the air production unit under the train model" style={{ width: "100%", height: "300px", objectFit: "cover", display: "block" }} />
            <div className="mono" style={{ position: "absolute", left: "12px", bottom: "10px", fontSize: "11px", background: "rgba(255,255,255,.92)", padding: "4px 8px", borderRadius: "4px" }}>air production unit beneath Car 3</div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}><span className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>page 2 / 2</span><span className="chip">research benchmark · nebulax/sim/pneumatic.py</span></div>
            <h3 style={{ margin: "0", fontSize: "24px", fontWeight: "600", lineHeight: "1.2" }}>Compressor air-leak detection from pneumatic telemetry</h3>
            <div className="callout"><span className="mono" style={{ fontSize: "10px", letterSpacing: ".12em", color: "#006d73" }}>SUMMARY</span><br />The compressor supplying the brakes and air springs cycles between 8.06 and 10.2 bar. Air leaks make it run more often and for longer, so the benchmark monitors changes from the normal cycle.</div>
            <div className="kv" style={{ gridTemplateColumns: "120px 1fr" }}>
              <div className="eyebrow" style={{ paddingTop: "2px" }}>Dataset</div><div>MetroPT-3, Metro do Porto (UCI 791, CC BY 4.0): 1,516,948 rows at 1 Hz, Feb to Aug 2020; 7 analogue channels (TP2, TP3, H1, DV_pressure, Reservoirs, Motor_current, Oil_temperature) + 8 digital</div>
              <div className="eyebrow" style={{ paddingTop: "2px" }}>Labels</div><div>4 high-severity air-leak windows (18 Apr, 29 to 30 May, 5 to 7 Jun, 15 Jul 2020); MetroPT-2 adds the oil-leak class this release lacks</div>
              <div className="eyebrow" style={{ paddingTop: "2px" }}>In the workspace</div><div>Open Brake air supply to view a recorded compressor excerpt and its location beneath Car 3. This view does not run a PS3 prediction.</div>
              <div className="eyebrow" style={{ paddingTop: "2px" }}>Simulator</div><div>Screw compressor, oil sump, twin-tower dryer and a 290 L main reservoir at 1 Hz with MetroPT-3 column names; all 31 constants are measured from the failure-free Feb to Mar 2020 window (3,141 cycles) by <span className="mono" style={{ fontSize: "12px" }}>scripts/calibrate_pneumatic.py</span></div>
            </div>
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1.15fr 1fr 1fr", gap: "24px", alignItems: "start" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}><div className="eyebrow">Model comparison · VUS-PR</div><div className="mono" style={{ fontSize: "11px", color: "#5b6673" }}>temporal split · 6 h windows</div></div>
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              <div className="bar" style={{ gridTemplateColumns: "190px 1fr 52px" }}><div>k-of-n corroboration</div><div className="track" style={{ height: "14px" }}><div className="fill" style={{ width: "78.3%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>0.783</div></div>
              <div className="bar" style={{ gridTemplateColumns: "190px 1fr 52px" }}><div>conformal threshold</div><div className="track" style={{ height: "14px" }}><div className="fill" style={{ width: "60.3%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>0.603</div></div>
              <div className="bar" style={{ gridTemplateColumns: "190px 1fr 52px" }}><div>LightGBM residual (selected)</div><div className="track" style={{ height: "14px" }}><div className="fill sel" style={{ width: "58.7%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>0.587</div></div>
              <div className="bar" style={{ gridTemplateColumns: "190px 1fr 52px" }}><div>moving-window variance</div><div className="track" style={{ height: "14px" }}><div className="fill" style={{ width: "57.2%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>0.572</div></div>
              <div className="bar" style={{ gridTemplateColumns: "190px 1fr 52px" }}><div>single-feature threshold</div><div className="track" style={{ height: "14px" }}><div className="fill" style={{ width: "44.3%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>0.443</div></div>
              <div className="bar" style={{ gridTemplateColumns: "190px 1fr 52px" }}><div>TCN autoencoder</div><div className="track" style={{ height: "14px" }}><div className="fill" style={{ width: "21.3%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>0.213</div></div>
              <div className="bar" style={{ gridTemplateColumns: "190px 1fr 52px" }}><div>LSTM autoencoder</div><div className="track" style={{ height: "14px" }}><div className="fill" style={{ width: "8.3%" }}></div></div><div className="mono" style={{ textAlign: "right" }}>0.083</div></div>
            </div>
            <div style={{ fontSize: "13px", color: "#5b6673", lineHeight: "1.5" }}>The no-skill floor is the 3 % positive rate, so 0.587 is an 18.8× lift. Simple residual and corroboration rules beat both neural autoencoders on four real leaks; the selected default is the row with all 4 / 4 leaks caught at the lowest false-alarm rate, 0.105 episodes per scored day, with a 17 h lead to failure.</div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            <div className="eyebrow">How the benchmark scores it</div>
            <div className="step" style={{ padding: "4px 0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>a</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Temporal split.</strong> Train on the earliest healthy months, calibrate the threshold on a validation slice, score the rest in time order. No point adjustment.</div></div>
            <div className="step" style={{ padding: "4px 0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>b</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Alarm episodes, not points.</strong> An alarm needs 3 consecutive windows over threshold; alarms less than 1 h apart merge. A leak counts as caught if an alarm lands within 48 h before it.</div></div>
            <div className="step" style={{ padding: "4px 0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>c</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>False alarms per scored day.</strong> Every row reports episodes per day of the scored slice, because a detector that catches 4 / 4 leaks with a false alarm every hour is useless to a depot.</div></div>
            <div className="step" style={{ padding: "4px 0" }}><div className="num" style={{ background: "#eef1f3", color: "#1d2633" }}>d</div><div style={{ fontSize: "13px", lineHeight: "1.5" }}><strong>Synthetic fleet replay.</strong> On the simulated pneumatic fleet <span className="mono" style={{ fontSize: "12px" }}>sparse_autoencoder</span> scores VUS-PR 0.780, 3 / 3 injected leaks, 0.110 false-alarm episodes per day.</div></div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div style={{ background: "#f7f8fa", borderRadius: "6px", padding: "14px 16px", display: "flex", flexDirection: "column", gap: "8px" }}>
              <div className="eyebrow">Limitations</div>
              <div style={{ fontSize: "13px", lineHeight: "1.5" }}>Model selection was deferred: the validation slice held no faults to select on, so the default was selected manually and is labelled as such on the leaderboard.</div>
              <div style={{ fontSize: "13px", lineHeight: "1.5" }}>MetroPT-3 has air leaks only, so there is no multi-class pneumatic fault identification from it alone. Recall figures are over 2 to 4 events, one seed.</div>
            </div>
            <div style={{ background: "#f7f8fa", borderRadius: "6px", padding: "14px 16px", display: "flex", flexDirection: "column" }}>
              <div className="eyebrow" style={{ marginBottom: "4px" }}>Sources</div>
              <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R88</span><span>Davari, Veloso, Ribeiro and Gama. <em>MetroPT-3 Dataset</em>, UCI Machine Learning Repository #791, 2023.</span></div>
              <div className="ref" style={{ fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R93</span><span>Jakobs, Veloso and Gama, <em>Int. J. Data Science and Analytics</em> 2026. Interpretable rules for online failure prediction on the Metro do Porto sets: the interpretable one-feature-rule baseline that our single-feature threshold row mirrors.</span></div>
              <div className="ref" style={{ borderBottom: "none", fontSize: "11px", gridTemplateColumns: "38px 1fr" }}><span className="mono" style={{ color: "#006d73" }}>R89</span><span>MetroPT-2 (Zenodo 7766691): 7.1 M records with one air leak and one oil leak; source of the oil-leak class and the R93 protocol.</span></div>
            </div>
          </div>
        </div>
      </div>
    </>)}
  </section>
    </>
  );
}
