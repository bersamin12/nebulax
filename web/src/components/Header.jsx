// Shared header for the Overview and Predict pages.
import { C } from "../lib/format.js";

function PageNav({ page, onPage }) {
  const link = (id, label) => (
    <button
      key={id}
      type="button"
      className={"nx-nav" + (page === id ? " is-active" : "")}
      aria-current={page === id ? "page" : undefined}
      onClick={() => page !== id && onPage?.(id)}
    >
      {label}
    </button>
  );
  return (
    <nav style={{ display: "flex", gap: 4, flex: "none" }} aria-label="Pages">
      {link("overview", "OVERVIEW")}
      {link("predict", "PREDICT")}
    </nav>
  );
}

function TutorialButton({ onPage }) {
  return (
    <button
      type="button"
      className="nx-btn"
      style={{ height: 26, flex: "none", whiteSpace: "nowrap" }}
      onClick={() => onPage?.("predict", { tour: true })}
      title="Tour the prediction workflow"
    >
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="10" />
        <path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3" />
        <line x1="12" y1="17" x2="12" y2="17" />
      </svg>
      TUTORIAL
    </button>
  );
}

export default function Header({ page = "overview", onPage }) {
  return (
    <div
      style={{
        height: 56,
        flex: "none",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 16,
        padding: "0 20px",
        background: C.panel,
        borderBottom: `1px solid ${C.line}`,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 14, flex: "none" }}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke={C.accent} strokeWidth="1.6" strokeLinecap="round">
          <path d="M5 4h14v11a3 3 0 0 1-3 3H8a3 3 0 0 1-3-3z" />
          <path d="M5 9h14" />
          <path d="M8 18l-2.5 3" />
          <path d="M16 18l2.5 3" />
          <path d="M9 13h.01" />
          <path d="M15 13h.01" />
        </svg>
        <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: "0.13em" }}>
            <span style={{ color: C.accentBright }}>TEAM BUS MRT WALK</span> &middot; TRAIN DIGITAL TWIN
          </div>
          <div style={{ fontSize: 9.5, letterSpacing: "0.16em", color: C.dim }}>
            TRACK 3 &middot; PROBLEM STATEMENT 3
          </div>
        </div>
      </div>

      <PageNav page={page} onPage={onPage} />

      <div style={{ display: "flex", alignItems: "center", gap: 10, flex: "none" }}>
        <span className="nx-tag" style={{ color: C.dim, borderColor: C.line2 }}>
          4 PREDICTION MODELS
        </span>
        {page === "predict" ? (
          <TutorialButton onPage={onPage} />
        ) : (
          <>
            <button type="button" className="nx-btn" style={{ height: 26, flex: "none" }} onClick={() => onPage?.("predict", { tour: true })}>
              TAKE A TOUR
            </button>
            <button type="button" className="nx-btn nx-btn--primary" style={{ height: 26, flex: "none" }} onClick={() => onPage?.("predict")}>
              OPEN PREDICTION WORKSPACE
            </button>
          </>
        )}
      </div>
    </div>
  );
}
