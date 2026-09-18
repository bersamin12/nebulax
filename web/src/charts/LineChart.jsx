// Plain-SVG line chart: no library, tabular numerics, dashed threshold, tinted alert bands.
import { MONO } from "../lib/format.js";

const PAD = { l: 40, r: 16, t: 10, b: 17 };

function niceTicks(lo, hi) {
  if (!(hi > lo)) return [lo];
  return [hi, (hi + lo) / 2, lo];
}

/**
 * Symmetric-log mapping for the anomaly-score axis. A CUSUM statistic runs from an exact 0 to
 * ~1e11 while its false-alarm threshold sits somewhere in between, so a linear axis draws the
 * whole trace flat on the floor with the threshold off the top. `floor` is the smallest positive
 * value worth resolving; everything at or below it lands on the axis origin.
 */
const symlog = (floor) => (v) => (v <= floor ? 0 : Math.log10(v / floor));

export function extent(points, idx) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of points) {
    const v = p[idx];
    if (v === null || v === undefined || Number.isNaN(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return Number.isFinite(lo) ? [lo, hi] : null;
}

/**
 * points        [[x:number, y:number], ...] sorted by x
 * series        optional extra faint traces: [{points, color, dash, normalise}]
 * threshold     number | null, drawn dashed
 * bands         [[x0, x1], ...] tinted spans (alert rows)
 * xFormat/yFormat  (v) => string
 */
export default function LineChart({
  points = [],
  series = [],
  width = 438,
  height = 96,
  color = "#008f95",
  threshold = null,
  thresholdColor = "#c43d36",
  thresholdLabel = null,
  logY = false,
  bands = [],
  xFormat = (v) => String(v),
  yFormat = (v) => String(v),
  markLast = null,
  empty = "no data",
}) {
  const x0 = PAD.l;
  const x1 = width - PAD.r;
  const y0 = PAD.t;
  const y1 = height - PAD.b;
  const pw = x1 - x0;
  const ph = y1 - y0;

  const clean = points.filter(
    (p) => p && Number.isFinite(p[0]) && p[1] !== null && p[1] !== undefined && Number.isFinite(p[1])
  );
  const xe = extent(clean, 0);
  let ye = extent(clean, 1);
  if (ye && threshold !== null && threshold !== undefined && Number.isFinite(threshold)) {
    ye = [Math.min(ye[0], threshold), Math.max(ye[1], threshold)];
  }

  const frame = (
    <g>
      <rect x={x0} y={y0} width={pw} height={ph} fill="#f8fafb" />
      <line x1={x0} y1={y0} x2={x1} y2={y0} stroke="#eaf0f2" strokeWidth="1" />
      <line x1={x0} y1={(y0 + y1) / 2} x2={x1} y2={(y0 + y1) / 2} stroke="#eaf0f2" strokeWidth="1" />
      <line x1={x0} y1={y0} x2={x0} y2={y1} stroke="#bdcbd2" strokeWidth="1" />
      <line x1={x0} y1={y1} x2={x1} y2={y1} stroke="#bdcbd2" strokeWidth="1" />
    </g>
  );

  if (!clean.length || !xe || !ye) {
    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: "block" }}>
        {frame}
        <text
          x={(x0 + x1) / 2}
          y={(y0 + y1) / 2 + 3}
          textAnchor="middle"
          fontFamily={MONO}
          fontSize="9"
          fill="#64748b"
        >
          {empty}
        </text>
      </svg>
    );
  }

  let [ylo, yhi] = ye;
  // On a log axis the floor is the smallest positive value in the window (scores are >= 0 and
  // are frequently exactly 0), and the axis is padded in decades rather than in units.
  let tx = (v) => v;
  let inv = (t) => t;
  if (logY && yhi > 0) {
    let smallest = Infinity;
    for (const p of clean) if (p[1] > 0 && p[1] < smallest) smallest = p[1];
    if (Number.isFinite(threshold) && threshold > 0 && threshold < smallest) smallest = threshold;
    const floor = Number.isFinite(smallest) ? smallest : yhi / 1e6;
    tx = symlog(floor);
    inv = (t) => (t <= 0 ? 0 : floor * 10 ** t);
    ylo = 0;
    yhi = Math.max(tx(yhi) * 1.08, 0.5);
  } else if (yhi - ylo < 1e-9) {
    const pad = Math.max(Math.abs(yhi) * 0.05, 0.5);
    ylo -= pad;
    yhi += pad;
  } else {
    const pad = (yhi - ylo) * 0.08;
    ylo -= pad;
    yhi += pad;
  }
  const [xlo, xhi] = xe;
  const sx = (v) => (xhi === xlo ? x0 + pw / 2 : x0 + ((v - xlo) / (xhi - xlo)) * pw);
  // sy takes a *data* value; syT takes an already-transformed one (used for the axis ticks).
  const syT = (t) => y1 - ((t - ylo) / (yhi - ylo)) * ph;
  const sy = (v) => syT(tx(v));

  const path = clean.map((p) => `${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ");
  const last = clean[clean.length - 1];
  // ticks are positions on the transformed axis; each is labelled with the score it stands for
  const ticks = niceTicks(ylo, yhi);

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: "block" }}>
      {frame}

      {bands.map(([b0, b1], i) => {
        const bx = sx(Math.max(b0, xlo));
        const bw = Math.max(1.5, sx(Math.min(b1, xhi)) - bx);
        return <rect key={i} x={bx} y={y0} width={bw} height={ph} fill="#c43d36" opacity="0.14" />;
      })}

      {ticks.map((t, i) => (
        <text
          key={i}
          x={x0 - 6}
          y={syT(t) + 3}
          textAnchor="end"
          fontFamily={MONO}
          fontSize="8.5"
          fill="#64748b"
        >
          {yFormat(inv(t))}
        </text>
      ))}

      {series.map((s, i) => {
        const sc = (s.points || []).filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
        if (!sc.length) return null;
        const se = extent(sc, 1);
        const norm = s.normalise && se && se[1] > se[0];
        const ny = (v) => (norm ? y1 - ((v - se[0]) / (se[1] - se[0])) * ph : sy(v));
        return (
          <polyline
            key={i}
            points={sc.map((p) => `${sx(p[0]).toFixed(1)},${ny(p[1]).toFixed(1)}`).join(" ")}
            fill="none"
            stroke={s.color || "#bdcbd2"}
            strokeWidth={s.width || 1}
            strokeDasharray={s.dash || undefined}
            opacity={s.opacity ?? 0.8}
          />
        );
      })}

      {Number.isFinite(threshold) && (
        <>
          <line
            x1={x0}
            y1={sy(threshold)}
            x2={x1}
            y2={sy(threshold)}
            stroke={thresholdColor}
            strokeWidth="1"
            strokeDasharray="4 3"
          />
          <text
            x={x1 - 2}
            y={Math.max(y0 + 7, sy(threshold) - 3)}
            textAnchor="end"
            fontFamily={MONO}
            fontSize="7.5"
            fill={thresholdColor}
          >
            {thresholdLabel ?? yFormat(threshold)}
          </text>
        </>
      )}

      <polyline points={path} fill="none" stroke={color} strokeWidth="1.6" />
      {markLast && <circle cx={sx(last[0])} cy={sy(last[1])} r="2.8" fill={markLast} />}

      <text x={x0} y={height - 4} fontFamily={MONO} fontSize="8" fill="#64748b">
        {xFormat(xlo)}
      </text>
      <text
        x={(x0 + x1) / 2}
        y={height - 4}
        textAnchor="middle"
        fontFamily={MONO}
        fontSize="8"
        fill="#64748b"
      >
        {xFormat((xlo + xhi) / 2)}
      </text>
      <text x={x1} y={height - 4} textAnchor="end" fontFamily={MONO} fontSize="8" fill="#64748b">
        {xFormat(xhi)}
      </text>
    </svg>
  );
}
