// Anomaly-score trace with the false-alarm-budget threshold and alert episodes tinted.
import LineChart from "./LineChart.jsx";
import { DAY, fmtDay, fmtDayHm, fmtScore, toMs } from "../lib/format.js";

// points from GET .../scores: [[ts, score, alert], ...]
function bandsFromAlerts(rows) {
  const out = [];
  let start = null;
  let prev = null;
  for (const r of rows) {
    if (r.alert) {
      if (start === null) start = r.x;
    } else if (start !== null) {
      out.push([start, prev ?? r.x]);
      start = null;
    }
    prev = r.x;
  }
  if (start !== null && prev !== null) out.push([start, prev]);
  return out;
}

export default function ScoreTrace({
  points = [],
  threshold = null,
  width = 438,
  height = 88,
  empty = "no scores in this window",
}) {
  const rows = (points || [])
    .map((p) => ({ x: toMs(p[0]), y: Number(p[1]), alert: Boolean(p[2]) }))
    .filter((r) => Number.isFinite(r.x) && Number.isFinite(r.y));

  // day-only labels repeat when the window is short; fall back to day + hh:mm
  const span = rows.length ? rows[rows.length - 1].x - rows[0].x : 0;
  const xFormat = span > 2 * DAY ? fmtDay : fmtDayHm;

  // A CUSUM statistic and its false-alarm threshold routinely sit ten decades apart (the
  // bearing winner's threshold is 6e7 against scores of 0). Switch the axis to symlog as soon
  // as the window, threshold included, spans more than three decades, or the trace is a flat
  // line on the floor with the threshold nowhere on the chart.
  let top = Number.isFinite(threshold) ? threshold : 0;
  let smallest = Infinity;
  for (const r of rows) {
    if (r.y > top) top = r.y;
    if (r.y > 0 && r.y < smallest) smallest = r.y;
  }
  if (Number.isFinite(threshold) && threshold > 0 && threshold < smallest) smallest = threshold;
  const logY = Number.isFinite(smallest) && top / smallest > 1e3;

  return (
    <LineChart
      points={rows.map((r) => [r.x, r.y])}
      bands={bandsFromAlerts(rows)}
      threshold={Number.isFinite(threshold) ? threshold : null}
      thresholdLabel={Number.isFinite(threshold) ? fmtScore(threshold) : null}
      logY={logY}
      width={width}
      height={height}
      color="#008f95"
      markLast={rows.length ? (rows[rows.length - 1].alert ? "#c43d36" : "#006d73") : null}
      xFormat={xFormat}
      yFormat={fmtScore}
      empty={empty}
    />
  );
}
