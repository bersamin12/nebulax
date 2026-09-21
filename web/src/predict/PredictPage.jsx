// PS3 prediction workflow. The former `?page=twin` fleet console is archived under
// web/archive/fleet-dashboard.
//
// One board under the 56 px header: optional API banner, 268 px model stage,
// prediction workspace and footer. The workspace is the default visible page.
//
// Flow: choose a subsystem (GET /api/ps3/tasks says what it accepts and what it scored), queue
// files, RUN. The queue is uploaded in batches of `max_files_per_request` (32) against one
// session token, so the 68 rail files never travel in one request; the server answers with every
// row accumulated so far, which is what the table, the 3D twin and the CSV download all read.
import { useCallback, useEffect, useMemo, useState } from "react";
import TrainViewport from "../components/viewport/TrainViewport.jsx";
import { ps3HealthMap } from "../components/viewport/componentMap.js";
import { ps3CsvUrl } from "../api.js";
import { C } from "../lib/format.js";
import DatasetPanel from "./DatasetPanel.jsx";
import ExplanationPanel from "./ExplanationPanel.jsx";
import ResultsTable from "./ResultsTable.jsx";
import ResearchExamplePanel from "./ResearchExamplePanel.jsx";
import TaskColumn from "./TaskColumn.jsx";
import Tutorial from "./Tutorial.jsx";
import researchExamples from "./researchExamples.json";
import { explanationFor, selectionFor } from "./selection.js";
import { columnsFor, INFO_META, SYSTEM_ORDER, TASK_META, TASK_ORDER } from "./taskMeta.js";


const VIEWPORT_H = 268;
const FOOTER_H = 68;
const BANNER_H = 30;
const LEFT_W = 340;
const RIGHT_W = 396;

/** `?page=predict&task=rail` - read once on load, and kept in the URL as the tab changes. */
function readTask() {
  if (typeof window === "undefined") return null;
  const t = new URLSearchParams(window.location.search).get("task");
  return SYSTEM_ORDER.includes(t) ? t : null;
}

export default function PredictPage({ height = 844, tourKey = 0, predictor }) {
  const p = predictor;
  // the tutorial is opt-in: the header's TUTORIAL button, the overview's "Take the tour" or
  // `?tour=N` (tourKey); it never starts by itself
  const [tour, setTour] = useState(() => tourKey > 0);
  useEffect(() => {
    if (tourKey > 0) setTour(true);
  }, [tourKey]);
  const [task, setActiveSystem] = useState(() => readTask() || p.task);
  const [resetKey, setResetKey] = useState(0);
  const [acvCar, setAcvCar] = useState(null);
  const info = INFO_META[task] || null;
  const examples = researchExamples[task] || [];
  const [exampleId, setExampleId] = useState(null);
  const [exampleIndex, setExampleIndex] = useState(0);
  const [examplePlaying, setExamplePlaying] = useState(true);
  const example = examples.find((item) => item.id === exampleId) || examples[1] || examples[0] || null;
  const examplePoint = example?.points[exampleIndex % example.points.length] || null;
  const rows = info ? [] : p.rows;
  const explanations = info ? [] : p.explanations;
  const files = info ? [] : p.files;
  const running = !info && p.running;
  const fatal = !info && p.fatal;
  const [selIdx, setSelIdx] = useState(-1);
  const [picked, setPicked] = useState(null); // what the 3D twin has highlighted

  const selectSystem = (name) => {
    if (!SYSTEM_ORDER.includes(name)) return;
    if (TASK_ORDER.includes(name)) p.setTask(name);
    setActiveSystem(name);
    setSelIdx(-1);
    setAcvCar(null);
    setPicked(null);
  };

  // Keep the deep link current: the tab is part of the URL, but not a history entry per click.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const q = new URLSearchParams(window.location.search);
    if (q.get("page") === "predict" && q.get("task") === task && !q.has("systems")) return;
    q.set("page", "predict");
    q.set("task", task);
    q.delete("systems"); // all six systems are always shown now; old links carrying it still work
    window.history.replaceState(null, "", `${window.location.pathname}?${q}`);
  }, [task]);

  // a new subsystem starts unselected; the first row of a result set is then selected for it,
  // so the twin is never blank - but a row the user picked survives the next batch's rows.
  useEffect(() => {
    setSelIdx(-1);
    setAcvCar(null);
    setExampleId(null);
    setExampleIndex(0);
    setExamplePlaying(true);
  }, [task]);
  useEffect(() => {
    if (!info || !example || !examplePlaying) return undefined;
    const timer = window.setInterval(() => setExampleIndex((i) => (i + 1) % example.points.length), 260);
    return () => window.clearInterval(timer);
  }, [info, example, examplePlaying]);
  useEffect(() => {
    setSelIdx((i) => (rows.length ? (i >= 0 && i < rows.length ? i : 0) : -1));
  }, [rows.length]);

  const columns = useMemo(() => columnsFor(task, rows), [task, rows]);
  const row = selIdx >= 0 && selIdx < rows.length ? rows[selIdx] : null;
  const explanation = useMemo(() => explanationFor(task, explanations, row), [task, explanations, row]);

  useEffect(() => {
    if (info || task === "acv") {
      setPicked(null); // ACV opens on all eight cars; a rank chip opts into a car close-up.
      return;
    }
    setPicked(explanation ? selectionFor(task, explanation) : null);
  }, [task, explanation, info]);

  // A healthMap is always passed, even with no rows: it puts the page in map mode, so the strip
  // never falls back to a replay frame it has nothing to do with, and every mesh reads `nodata`.
  const healthMap = useMemo(
    () =>
      info
        ? {
            task,
            caption: `${info.tab} · built-in ${example?.label || "dataset"} · ${examplePoint?.label || ""}`,
            components: examplePoint ? {
              [task === "pneumatic" ? "3|pneumatic|apu_1" : "1|bearing|axlebox_1L"]: {
                health: example?.condition === "faulty" || examplePoint.alert ? "crit" : "ok",
                label: `${example?.record} · ${examplePoint.value} ${example?.unit}`,
              },
            } : {},
            railHealth: {},
            acHealth: {},
          }
        : rows.length
        ? ps3HealthMap({ task, rows, explanations, fileId: row && row.file_id !== undefined ? row.file_id : null })
        : { task, caption: `${(TASK_META[task] || {}).title || task} · no prediction yet`, components: {}, railHealth: {}, acHealth: {} },
    [task, rows, explanations, row, info, example, examplePoint]
  );

  const onPick = useCallback(
    (sel) => {
      setPicked(sel);
      if (!sel) return;
      if (task === "acv" && sel.subsystem === "acv") {
        const id = String(sel.car).padStart(2, "0");
        const i = rows.findIndex((r) => String(r.ranked_cars || "").split("|").some((c) => +c === +id));
        if (i >= 0) setSelIdx(i);
        setAcvCar(id);
        return;
      }
      // clicking the twin selects the row that painted that mesh, when there is one
      const i = rows.findIndex((r) => {
        const e = explanationFor(task, explanations, r);
        const s = e ? selectionFor(task, e) : null;
        return !!s && s.car === sel.car && s.subsystem === sel.subsystem && s.component_id === sel.component_id;
      });
      if (i >= 0) setSelIdx(i);
    },
    [rows, explanations, task]
  );

  const onSelectCar = (id, i) => {
    setSelIdx(i);
    setAcvCar(id);
    setPicked({ train_id: "PS3", car: Number(id), subsystem: "acv", component_id: "ac_1" });
  };

  const onSelectRow = (i) => {
    setSelIdx(i);
    if (task === "acv") {
      setAcvCar(null);
      setPicked(null);
    }
  };

  const meta = TASK_META[task] || {};
  const bodyH = height - VIEWPORT_H - FOOTER_H - (fatal ? BANNER_H : 0);
  const innerH = bodyH - 12;
  const csvHref = p.csvUrl ? (p.csvUrl.startsWith("/") ? p.csvUrl : ps3CsvUrl(p.session)) : null;
  const completedCount = p.nDone || files.filter((file) => file.status === "done").length;

  return (
    <div style={{ height, display: "flex", flexDirection: "column", minHeight: 0 }}>
      {fatal && (
        <div
          style={{
            height: BANNER_H,
            flex: "none",
            display: "flex",
            alignItems: "center",
            gap: 10,
            padding: "0 20px",
            background: C.critBg,
            borderBottom: `1px solid ${C.crit}`,
            color: C.crit,
            fontSize: 11,
          }}
        >
          <span style={{ width: 7, height: 7, background: C.crit, display: "block", flex: "none" }} />
          <strong style={{ letterSpacing: "0.08em" }}>PREDICTION SERVICE UNAVAILABLE</strong>
          <span style={{ color: C.crit, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {String(fatal.message || fatal)}. You can review the page, but predictions are unavailable until the service reconnects.
          </span>
          <button type="button" className="nx-btn" style={{ marginLeft: "auto", flex: "none" }} onClick={p.reload}>
            RETRY
          </button>
        </div>
      )}

      <div data-tour="stage" style={{ height: VIEWPORT_H, flex: "none", borderBottom: `1px solid ${C.line}`, overflow: "hidden" }}>
        <TrainViewport
          healthMap={healthMap}
          trainId={null}
          selected={picked}
          onSelect={onPick}
          height={VIEWPORT_H}
          viewMode={task}
        />
      </div>

      <div
        style={{
          height: bodyH,
          flex: "none",
          display: "grid",
          gridTemplateColumns: `${LEFT_W}px minmax(0, 1fr) ${RIGHT_W}px`,
          gap: 12,
          padding: "12px 20px 0 20px",
        }}
      >
        <TaskColumn
          tasks={p.tasks}
          task={task}
          setTask={selectSystem}
          cloud={p}
          resetKey={resetKey}
          meta={info ? null : p.meta}
          files={files}
          addFiles={p.addFiles}
          addLocalPaths={p.addLocalPaths}
          removeFile={p.removeFile}
          run={p.run}
          cancel={p.cancel}
          running={running}
          done={p.done}
          total={p.total}
          notice={p.notice}
          error={p.error}
          dropFields={p.dropFields}
          toggleDropField={p.toggleDropField}
          width={LEFT_W}
          height={innerH}
        />

        <div data-tour="table" style={{ minWidth: 0 }}>
        {info ? (
          <ResearchExamplePanel
            info={info}
            examples={examples}
            example={example}
            index={exampleIndex % (example?.points.length || 1)}
            playing={examplePlaying}
            onSelect={(id) => { setExampleId(id); setExampleIndex(0); setExamplePlaying(true); }}
            onToggle={() => setExamplePlaying((v) => !v)}
            onSeek={setExampleIndex}
            height={innerH}
          />
        ) : (
          <ResultsTable
            task={task}
            rows={rows}
            columns={columns}
            selected={selIdx}
            selectedCar={acvCar}
            onSelect={onSelectRow}
            onSelectCar={onSelectCar}
            onShowAll={() => { setAcvCar(null); setPicked(null); }}
            height={innerH}
            running={running}
            progress={p.total ? `${p.done} / ${p.total} files` : p.nDone ? `${p.nDone} files` : ""}
          />
        )}
        </div>

        <div data-tour="explanation" style={{ minWidth: 0 }}>
        {info ? (
          <DatasetPanel info={info} detail height={innerH} />
        ) : (
          <ExplanationPanel task={task} row={row} explanation={explanation} columns={columns} selectedCar={acvCar} height={innerH} />
        )}
        </div>
      </div>

      <div
        style={{
          height: FOOTER_H,
          flex: "none",
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "0 20px",
          borderTop: `1px solid ${C.line}`,
          background: C.panel,
        }}
      >
        {info ? (
          <div style={{ fontSize: 11, color: C.text2 }}>
            <strong>{info.tab}</strong> · recorded research excerpt · no PS3 prediction or CSV output
          </div>
        ) : <><a
          data-tour="download"
          href={csvHref || undefined}
          target="_blank"
          rel="noreferrer"
          className="nx-btn"
          aria-disabled={!csvHref || !rows.length}
          style={{
            height: 28,
            fontSize: 10.5,
            textDecoration: "none",
            borderColor: rows.length ? C.violet : C.line2,
            color: rows.length ? C.violetBright : C.dim2,
            background: rows.length ? C.accentBg : C.panel2,
            pointerEvents: csvHref && rows.length ? "auto" : "none",
          }}
          download={p.outputFilename || undefined}
        >
          DOWNLOAD {p.outputFilename || (meta.tab || task).toUpperCase() + " CSV"}
        </a>
        <button
          type="button"
          className="nx-btn"
          style={{ height: 28, fontSize: 10.5 }}
          onClick={() => {
            p.clear();
            setResetKey((k) => k + 1);
            setSelIdx(-1);
            setPicked(null);
            setAcvCar(null);
          }}
          disabled={running || (!files.length && !rows.length)}
        >
          CLEAR SESSION
        </button>

        <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0, flex: "1 1 auto" }}>
          <div style={{ fontSize: 10.5, color: C.text2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            <span style={{ color: C.dim, letterSpacing: "0.1em", fontSize: 9 }}>SCORING&nbsp;&nbsp;</span>
            <span className="mono" style={{ color: C.violet }}>{meta.metric}</span>
            <span style={{ color: C.dim }}>: {meta.scored}</span>
          </div>
          <div className="mono" style={{ fontSize: 9.5, color: C.dim2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {p.session
              ? `Session ${String(p.session).slice(0, 8)}… · ${completedCount} file${completedCount === 1 ? "" : "s"} · ${rows.length} result${rows.length === 1 ? "" : "s"} · CSV ready`
              : "The CSV will be available after the first successful result."}
          </div>
        </div></>}
      </div>

      <Tutorial open={tour} startStep={tourKey} onClose={() => setTour(false)} />
    </div>
  );
}
