import { C } from "../lib/format.js";

function Fact({ label, children }) {
  return (
    <div className="nx-dataset-fact">
      <span>{label}</span>
      <strong>{children}</strong>
    </div>
  );
}

export default function DatasetPanel({ info, detail = false, height }) {
  return (
    <section className="nx-info-card" style={{ height, minHeight: 0, overflowY: "auto", padding: 18 }}>
      <span className="nx-eyebrow">{detail ? "RESEARCH CONTEXT" : "DATASET OVERVIEW"}</span>
      <h2>{detail ? info.title : info.dataset}</h2>
      {detail ? (
        <>
          <p>Background data for the component shown above. This read-only view does not run a PS3 model.</p>
          <Fact label="Model explored">{info.model}</Fact>
          <Fact label="Evaluation">{info.evaluation}</Fact>
          <Fact label="Train location">{info.location}</Fact>
          <a className="nx-source-link" href={info.sourceUrl} target="_blank" rel="noreferrer">
            DATASET SOURCE ↗
          </a>
        </>
      ) : (
        <>
          <p>{info.blurb}</p>
          <Fact label="Source">{info.source}</Fact>
          <Fact label="Records">{info.size}</Fact>
          <Fact label="Measurements">{info.signals}</Fact>
          <div style={{ marginTop: 20, color: C.dim, lineHeight: 1.55 }}>
            Uploads and submission CSVs are not available for this dataset.
          </div>
        </>
      )}
    </section>
  );
}
