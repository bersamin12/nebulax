// Overview landing page: what the Train Digital Twin is, how the four PS3 models work, the EDA,
// the ablation ladders, the two exploratory systems, and the team. Static prose and numbers, no
// API calls; the only interaction is the carousels and the two calls to action.
//
// It is the default page (`?page=overview`); "Open Digital Twin" goes to the predict page and
// "Take the 2-minute tour" opens it with the tutorial running (`?tour=1`, predict/Tutorial.jsx).
// The console shell is a fixed 1440x900 board, so the page scrolls inside it.
import { useCallback } from "react";
import "./overview.css";
import Intro from "./Intro.jsx";
import Models from "./Models.jsx";
import Eda from "./Eda.jsx";
import Ablation from "./Ablation.jsx";
import Extra from "./Extra.jsx";
import Outro from "./Outro.jsx";

export default function OverviewPage({ height = 844, onPage }) {
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
    <div className="nx-overview nx-scroll" style={{ height, overflowY: "auto", overflowX: "hidden" }}>
      <div style={{ width: 1440, display: "flex", flexDirection: "column", position: "relative" }}>
        <Intro openTwin={openTwin} openTour={openTour} />
        <Models />
        <Eda />
        <Ablation />
        <Extra />
        <Outro />
      </div>
    </div>
  );
}
