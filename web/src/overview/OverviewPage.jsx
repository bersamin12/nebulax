// Overview landing page: what the Train Digital Twin is, the four results, how a prediction is
// made, and where each subsystem sits on the train. The technical material (each model in
// detail, the dataset profiles, the ablation ladders, the two research datasets, the glossary)
// sits below in folds that are collapsed by default, so a first visit reads in under a minute
// and a reviewer can still open everything. Static prose and numbers, no API calls.
//
// It is the default page (`?page=overview`) and an ordinary responsive document (App.jsx renders
// it outside the scaled console). "Open the prediction workspace" goes to Predict and
// "Take the tour" opens it with the tutorial running (`?tour=1`, predict/Tutorial.jsx).
import { useCallback, useState } from "react";
import "./overview.css";
import Intro from "./Intro.jsx";
import Models from "./Models.jsx";
import Eda from "./Eda.jsx";
import Ablation from "./Ablation.jsx";
import Extra from "./Extra.jsx";
import Outro from "./Outro.jsx";

/** A collapsible section: eyebrow + title + one-line blurb in the header, content below. */
function Fold({ id, eyebrow, title, blurb, defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section id={id} className={"ov-fold" + (open ? " is-open" : "")}>
      <button type="button" className="ov-fold-head" onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-controls={`${id}-body`}>
        <div className="ov-fold-title">
          <div className="eyebrow">{eyebrow}</div>
          <h2>{title}</h2>
          {blurb && <div className="ov-fold-blurb">{blurb}</div>}
        </div>
        <span className="ov-fold-chev" aria-hidden="true">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 9l6 6 6-6" /></svg>
        </span>
      </button>
      <div id={`${id}-body`} className="ov-fold-body">{open && children}</div>
    </section>
  );
}

export default function OverviewPage({ onPage }) {
  const openTwin = useCallback(
    (e) => {
      e.preventDefault();
      onPage?.("predict");
    },
    [onPage]
  );
  const openTour = useCallback(
    (e) => {
      e.preventDefault();
      onPage?.("predict", { tour: true });
    },
    [onPage]
  );
  return (
    <div className="nx-overview">
      <Intro openTwin={openTwin} openTour={openTour} />

      <Fold id="models" eyebrow="Prediction models" title="How each of the four models works" blurb="Input, decision method, validation and the literature behind each model. All four are under 1 MB and run on a laptop.">
        <Models />
      </Fold>
      <Fold id="eda" eyebrow="Exploratory data analysis" title="Released dataset profiles" blurb="What the training data looks like for each task and what that meant for the design.">
        <Eda />
      </Fold>
      <Fold id="ablation" eyebrow="Ablation studies" title="Effect of each design choice" blurb="The model ladder for each task: what was tried, what was kept, and why.">
        <Ablation />
      </Fold>
      <Fold id="research" eyebrow="Research datasets · outside PS3 scoring" title="Axle bearing and brake air supply" blurb="Two public datasets explored alongside PS3; they appear in the workspace as read-only profiles.">
        <Extra />
      </Fold>
      {/* <Fold id="glossary" eyebrow="Glossary" title="Terms used on this page">
        <Glossary />
      </Fold> */}

      <Outro openTwin={openTwin} />
    </div>
  );
}
