// WS /api/replay probe - node 22 built-in WebSocket, no dependencies.
//
//   node web/dev/wsProbe.mjs [ws://127.0.0.1:8000/api/replay] [seconds]
//
// 1. plays the replay for `seconds` wall-clock seconds and reports the frame rate, the sim time
//    covered and how alerts / kpis move over the month;
// 2. seeks to 2026-09-14T08:00:00Z and reports the components that are `crit` there.
// Exits non-zero if the socket never delivers a frame.

const URL_ = process.argv[2] ?? "ws://127.0.0.1:8000/api/replay";
const SECONDS = Number(process.argv[3] ?? 20);
const SEEK_TS = "2026-09-14T08:00:00Z";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const n = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "-");

const ws = new WebSocket(URL_);
const frames = [];
let lastError = null;

ws.onmessage = (ev) => {
  let f;
  try {
    f = JSON.parse(ev.data);
  } catch {
    return;
  }
  if (f.error) {
    lastError = f.error;
    return;
  }
  frames.push({ wall: Date.now(), f });
};

await new Promise((res, rej) => {
  ws.onopen = res;
  ws.onerror = () => rej(new Error(`cannot open ${URL_}`));
});
console.log(`connected ${URL_}`);

// ---------------------------------------------------------------- 1. play
ws.send(JSON.stringify({ cmd: "seek", ts: "2026-09-01T00:00:00Z" }));
ws.send(JSON.stringify({ cmd: "speed", speed: 8640 }));
const mark = frames.length;
const t0 = Date.now();
ws.send(JSON.stringify({ cmd: "play" }));
await sleep(SECONDS * 1000);
ws.send(JSON.stringify({ cmd: "pause" }));
const wall = (Date.now() - t0) / 1000;
const run = frames.slice(mark);

if (!run.length) {
  console.error(`FAIL: no frames in ${SECONDS} s (last error: ${lastError})`);
  process.exit(1);
}

const tsOf = (r) => Date.parse(r.f.ts);
const simSpan = (tsOf(run.at(-1)) - tsOf(run[0])) / 3600000;
const gaps = run.slice(1).map((r, i) => r.wall - run[i].wall).sort((a, b) => a - b);

console.log(`\n--- play, ${SECONDS} s wall ---`);
console.log(`frames                ${run.length}`);
console.log(`rate                  ${n(run.length / wall)} Hz over ${n(wall)} s wall`);
console.log(`inter-frame ms        median ${gaps[gaps.length >> 1]}  min ${gaps[0]}  max ${gaps.at(-1)}`);
console.log(`sim ts                ${run[0].f.ts} -> ${run.at(-1).f.ts}`);
console.log(`sim hours covered     ${n(simSpan, 1)} h (${n(simSpan / 24, 2)} days) at speed ${run[0].f.speed}`);
console.log(`ts strictly advancing ${run.every((r, i) => i === 0 || tsOf(r) >= tsOf(run[i - 1]))}`);
console.log(`trains per frame      ${run[0].f.trains.length} (${run[0].f.trains.map((t) => t.train_id).join(",")})`);
const nComp = (f) => f.trains.reduce((s, t) => s + t.components.length, 0);
console.log(`components per frame  ${nComp(run[0].f)} -> ${nComp(run.at(-1).f)}` +
  `  (a component appears once it has a scored row at or before ts)`);

const kpiKeys = ["open_alerts", "median_lead_h", "fa_per_train_day", "events_detected", "events_total"];
const first = run[0].f;
const last = run.at(-1).f;
console.log(`\nalerts   ${first.alerts.length} -> ${last.alerts.length}` +
  `   distinct counts over the run: ${new Set(run.map((r) => r.f.alerts.length)).size}`);
for (const k of kpiKeys) console.log(`kpi ${k.padEnd(18)} ${first.kpis[k]} -> ${last.kpis[k]}`);
console.log(`ticker lines ${first.ticker.length} -> ${last.ticker.length}`);

// ---------------------------------------------------------------- 2. seek
frames.length = 0;
ws.send(JSON.stringify({ cmd: "seek", ts: SEEK_TS }));
await sleep(700);
const seeked = frames.at(-1)?.f;
console.log(`\n--- seek ${SEEK_TS} ---`);
if (!seeked) {
  console.error("FAIL: no frame after seek");
  process.exit(1);
}
console.log(`frame ts              ${seeked.ts}   playing=${seeked.playing}`);
const crit = [];
const warn = [];
for (const t of seeked.trains)
  for (const c of t.components) {
    if (c.health === "crit") crit.push(`${t.train_id} car ${c.car} ${c.component_id} ${c.score?.toExponential(2)} > ${c.threshold?.toExponential(2)}`);
    if (c.health === "warn") warn.push(`${t.train_id} ${c.component_id}`);
  }
console.log(`components            ${seeked.trains.reduce((s, t) => s + t.components.length, 0)}`);
console.log(`crit components       ${crit.length}`);
for (const c of crit) console.log(`  ${c}`);
console.log(`warn components       ${warn.length}${warn.length ? ` (${warn.join(", ")})` : ""}`);
console.log(`open alerts           ${seeked.alerts.filter((a) => !a.t_end).length} of ${seeked.alerts.length}`);
console.log(`kpis                  ${JSON.stringify(seeked.kpis)}`);
console.log(`\n${crit.length ? "OK" : "FAIL"}: ${crit.length} crit component(s) at ${SEEK_TS}`);

ws.close();
process.exit(crit.length ? 0 : 1);
