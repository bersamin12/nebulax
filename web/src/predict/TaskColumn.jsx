// Left column: which subsystem, what it accepts, what it scored in cross-validation, and the
// upload queue. The CV line, the accepted suffixes and the batch size all come from
// GET /api/ps3/tasks - a task whose module or artefact is missing says so and cannot be run.
import { useCallback, useEffect, useRef, useState } from "react";
import { C } from "../lib/format.js";
import { inspectPs3LocalPath, validateAcvWorkbook } from "../api.js";
import { cvScore, fmtBytes, INFO_META, SYSTEM_ORDER, TASK_META, TASK_ORDER, TASK_SUFFIXES } from "./taskMeta.js";
import { filesFromDataTransfer } from "./usePs3Predict.js";

const STATUS_COLOR = { queued: C.dim2, waiting: C.dim, running: C.accent, done: C.ok, error: C.crit };
const STATUS_WORD = { queued: "queued", waiting: "waiting", running: "running", done: "done", error: "error" };

function TaskTab({ name, entry, active, onClick }) {
  const infoOnly = !!INFO_META[name];
  const meta = TASK_META[name] || INFO_META[name] || {};
  const cv = cvScore(entry && entry.cv);
  const off = entry && entry.available === false;
  return (
    <button
      type="button"
      onClick={onClick}
      className="nx-cell"
      style={{
        borderTop: `3px solid ${active ? C.accent : C.line2}`,
        outline: active ? `1px solid ${C.accentBright}` : "none",
        outlineOffset: -1,
        background: active ? C.accentBg : C.panel2,
        opacity: off ? 0.62 : 1,
        gap: 3,
        padding: "7px 8px",
        minHeight: 57,
      }}
      title={
        off
          ? entry.detail || "task unavailable"
          : [meta.blurb, infoOnly ? `${meta.dataset} · dataset profile only` : entry && entry.cv ? `CV: ${entry.cv.scheme || "?"}${entry.cv.metric ? ` · ${entry.cv.metric}` : ""}` : "no CV results file yet"].join("\n")
      }
    >
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 6 }}>
        <span style={{ fontSize: 11.5, fontWeight: 600, color: active ? C.accentBright : C.text }}>
          {meta.tab || name}
        </span>
        <span className="mono" style={{ fontSize: 9, color: off ? C.crit : cv ? C.violet : C.dim2 }}>
          {infoOnly ? "INFO ONLY" : off ? "UNAVAILABLE" : cv ? `${cv.label} ${Number(cv.value).toFixed(3)}` : entry && entry.cv ? "cv · no headline" : "no cv yet"}
        </span>
      </div>
      <div style={{ fontSize: 9.5, color: C.dim, lineHeight: 1.25, whiteSpace: "normal" }}>{meta.blurb}</div>
    </button>
  );
}

export default function TaskColumn({
  tasks,
  task,
  setTask,
  showAll = false,
  setShowAll,
  resetKey = 0,
  meta,
  files,
  addFiles,
  addLocalPaths,
  removeFile,
  run,
  running,
  done,
  total,
  notice,
  error,
  animateOnTwin = true,
  setAnimateOnTwin,
  width = 340,
  height = 546,
}) {
  const [hover, setHover] = useState(false);
  const fileRef = useRef(null);
  const dirRef = useRef(null);
  const checkAbort = useRef(null);
  const [checking, setChecking] = useState(false);
  const [checkMessage, setCheckMessage] = useState("");
  const [checkFailed, setCheckFailed] = useState(false);
  const [pickMessage, setPickMessage] = useState("");
  const [localPath, setLocalPath] = useState("");
  const infoOnly = !!INFO_META[task];
  const tmeta = TASK_META[task] || INFO_META[task] || {};
  const queued = files.filter((f) => f.status === "queued").length;
  const errored = files.filter((f) => f.status === "error").length;
  const unavailable = meta && meta.available === false;

  // A browser keeps the last selected path on a file input. Remounting on Clear and resetting
  // both refs here makes choosing the same file or folder again reliable across browsers.
  useEffect(() => {
    checkAbort.current?.abort();
    checkAbort.current = null;
    setChecking(false);
    setCheckMessage("");
    setCheckFailed(false);
    setPickMessage("");
    setLocalPath("");
    if (fileRef.current) fileRef.current.value = "";
    if (dirRef.current) dirRef.current.value = "";
  }, [resetKey, task]);

  const queuePicked = useCallback(async (list) => {
    if (!list.length) return;
    if (task !== "acv") { addFiles(list); return; }
    checkAbort.current?.abort();
    const ctrl = new AbortController();
    checkAbort.current = ctrl;
    setChecking(true);
    setCheckMessage("Checking ACV workbook fields…");
    setCheckFailed(false);
    const valid = [];
    const errors = [];
    for (const file of list) {
      if (!file.name.toLowerCase().endsWith(".xlsx")) {
        errors.push(`${file.name}: choose a .xlsx workbook`);
        continue;
      }
      try {
        const report = await validateAcvWorkbook(file, { signal: ctrl.signal });
        if (report.valid) valid.push(file);
        else errors.push(`${file.name}: ${report.errors.join("; ")}`);
      } catch (err) {
        if (ctrl.signal.aborted) return;
        errors.push(`${file.name}: ${err.message || err}`);
      }
    }
    if (ctrl.signal.aborted || checkAbort.current !== ctrl) return;
    if (valid.length) addFiles(valid);
    setCheckMessage(errors.length ? errors.join(" · ") : `${valid.length} ACV workbook${valid.length === 1 ? "" : "s"} passed the format check`);
    setCheckFailed(errors.length > 0);
    setChecking(false);
    checkAbort.current = null;
  }, [addFiles, task]);

  const onDrop = useCallback(
    async (e) => {
      e.preventDefault();
      setHover(false);
      const list = await filesFromDataTransfer(e.dataTransfer);
      await queuePicked(list);
    },
    [queuePicked]
  );

  const onPick = useCallback(
    (e) => {
      const input = e.currentTarget;
      const list = Array.from(input.files || []);
      if (!list.length) {
        setPickMessage("The picker returned no files. Try dropping a file into the box above.");
        return;
      }
      setPickMessage("");
      // Leave the selected filename visible. Firefox may still be completing the native
      // picker handoff when this change handler returns.
      void queuePicked(list)
        .catch((err) => setPickMessage(`Could not queue the selection: ${err.message || err}`));
    },
    [queuePicked]
  );

  const onAddLocal = useCallback(async () => {
    const path = localPath.trim();
    if (!path) return;
    checkAbort.current?.abort();
    const ctrl = new AbortController();
    checkAbort.current = ctrl;
    setChecking(true);
    setCheckMessage("Checking local path…");
    setCheckFailed(false);
    try {
      const result = await inspectPs3LocalPath(task, path, { signal: ctrl.signal });
      if (ctrl.signal.aborted) return;
      addLocalPaths(result.files || []);
      setLocalPath("");
      setCheckMessage(`${result.files.length} local file${result.files.length === 1 ? "" : "s"} added to queue`);
    } catch (err) {
      if (ctrl.signal.aborted) return;
      setCheckMessage(err.message || String(err));
      setCheckFailed(true);
    } finally {
      if (checkAbort.current === ctrl) {
        checkAbort.current = null;
        setChecking(false);
      }
    }
  }, [addLocalPaths, localPath, task]);

  const accept = (task === "acv" ? [".xlsx"] : meta && meta.accepts ? meta.accepts : TASK_SUFFIXES[task] || [".csv"]).join(",");

  const systemPicker = (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, flex: "none" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6 }}>
        <span style={{ fontSize: 9.5, letterSpacing: "0.14em", color: C.dim }}>SYSTEMS</span>
        <button type="button" className="nx-filter" onClick={() => setShowAll?.(!showAll)} aria-pressed={showAll}>
          {showAll ? "ALL SIX" : "PREDICTIONS (4)"}
        </button>
      </div>
      <div className="nx-system-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
        {(showAll ? SYSTEM_ORDER : TASK_ORDER).map((name) => (
          <TaskTab
            key={name}
            name={name}
            entry={(tasks || []).find((t) => t.name === name) || null}
            active={name === task}
            onClick={() => setTask(name)}
          />
        ))}
      </div>
    </div>
  );

  if (infoOnly) {
    return (
      <div style={{ width, height, display: "flex", flexDirection: "column", gap: 8, minHeight: 0 }}>
        {systemPicker}
        <div className="nx-info-card" style={{ flex: 1, minHeight: 0, padding: 16 }}>
          <span className="nx-eyebrow">DATASET PROFILE</span>
          <h2>{tmeta.dataset}</h2>
          <p>{tmeta.blurb}</p>
          <p><strong>{tmeta.size}</strong><br />{tmeta.signals}</p>
          <span className="nx-info-badge">INFORMATION ONLY · NO UPLOAD OR PREDICTION</span>
        </div>
      </div>
    );
  }

  return (
    <div style={{ width, height, display: "flex", flexDirection: "column", gap: 8, minHeight: 0 }}>
      {systemPicker}

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setHover(true);
        }}
        onDragLeave={() => setHover(false)}
        onDrop={onDrop}
        style={{
          flex: "none",
          border: `1px dashed ${hover ? C.accentBright : C.line2}`,
          background: hover ? C.accentBg : C.panel,
          padding: "9px 10px",
          display: "flex",
          flexDirection: "column",
          gap: 6,
        }}
      >
        <div style={{ fontSize: 10.5, color: hover ? C.accentBright : C.dim }}>
          drop {tmeta.multiple ? "files or a folder" : "the file"} here — {accept}
          {meta && meta.max_files_per_request ? `, ${meta.max_files_per_request} per request` : ""}
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <span className="nx-eyebrow">FILES</span>
            <input
              key={`file-${task}-${resetKey}`}
              ref={fileRef}
              type="file"
              multiple={tmeta.multiple !== false}
              accept={accept}
              onClick={(e) => { e.currentTarget.value = ""; }}
              onChange={onPick}
              className="nx-file-pick-input"
              aria-label={`${tmeta.tab || task} input files`}
              title="Choose files to add to the queue"
            />
          </div>
          {tmeta.directory && (
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <span className="nx-eyebrow">FOLDER</span>
              <input
                key={`folder-${task}-${resetKey}`}
                ref={dirRef}
                type="file"
                multiple
                webkitdirectory=""
                directory=""
                onClick={(e) => { e.currentTarget.value = ""; }}
                onChange={onPick}
                className="nx-file-pick-input"
                aria-label={`${tmeta.tab || task} input folder`}
                title="Choose a folder to add its files to the queue"
              />
            </div>
          )}
        </div>
        {pickMessage && <div role="alert" style={{ fontSize: 10, color: C.crit }}>{pickMessage}</div>}
        {meta?.local_paths && (
          <div style={{ display: "flex", flexDirection: "column", gap: 3, borderTop: `1px solid ${C.line}`, paddingTop: 6 }}>
            <span className="nx-eyebrow">LOCAL FILE OR TEST FOLDER PATH · FIREFOX FALLBACK</span>
            <div style={{ display: "flex", gap: 5 }}>
              <input
                type="text"
                value={localPath}
                onChange={(e) => setLocalPath(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") void onAddLocal(); }}
                placeholder="/absolute/path/to/Test.csv"
                aria-label="Local file or folder path"
                style={{ flex: 1, minWidth: 0, height: 25, padding: "0 6px", border: `1px solid ${C.line2}`, fontSize: 10, color: C.text, background: C.panel }}
              />
              <button type="button" className="nx-btn" onClick={() => void onAddLocal()} disabled={checking || running || !localPath.trim()} style={{ height: 25, padding: "0 7px", fontSize: 10 }}>ADD PATH</button>
            </div>
          </div>
        )}
      </div>

      <div
        style={{
          flex: "1 1 auto",
          minHeight: 0,
          background: C.panel,
          border: `1px solid ${C.line}`,
          display: "flex",
          flexDirection: "column",
        }}
      >
        <div
          style={{
            height: 24,
            flex: "none",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0 9px",
            borderBottom: `1px solid ${C.line}`,
            background: C.panel2,
          }}
        >
          <span style={{ fontSize: 9.5, letterSpacing: "0.16em", color: C.dim }}>QUEUE</span>
          <span className="mono" style={{ fontSize: 9.5, color: errored ? C.crit : C.dim2 }}>
            {running || done ? `${done} / ${total || files.length} processed` : `${files.length} file${files.length === 1 ? "" : "s"}`}
            {errored ? ` · ${errored} failed` : ""}
          </span>
        </div>
        <div className="nx-scroll" style={{ flex: 1, minHeight: 0 }}>
          {!files.length ? (
            <div style={{ padding: "10px 9px", fontSize: 10.5, color: C.dim2 }}>nothing queued</div>
          ) : (
            files.map((f) => (
              <div
                key={f.id}
                style={{
                  display: "flex",
                  alignItems: "baseline",
                  gap: 6,
                  padding: "3px 9px",
                  borderBottom: `1px solid ${C.grid}`,
                  borderLeft: `3px solid ${STATUS_COLOR[f.status] || C.line2}`,
                }}
                title={f.message || f.name}
              >
                <span
                  className="mono"
                  style={{ fontSize: 10, color: C.text2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: "1 1 auto" }}
                >
                  {f.name}
                </span>
                <span className="mono" style={{ fontSize: 9, color: C.dim2, flex: "none" }}>
                  {fmtBytes(f.size)}
                </span>
                <span className="mono" style={{ fontSize: 9, color: STATUS_COLOR[f.status] || C.dim2, flex: "none", width: 44, textAlign: "right" }}>
                  {STATUS_WORD[f.status] || f.status}
                </span>
                {!running && (
                  <button
                    type="button"
                    onClick={() => removeFile(f.id)}
                    style={{ flex: "none", background: "none", border: "none", color: C.dim2, cursor: "pointer", fontSize: 11, padding: 0, lineHeight: 1 }}
                    aria-label={`remove ${f.name}`}
                  >
                    ×
                  </button>
                )}
              </div>
            ))
          )}
          {files
            .filter((f) => f.status === "error" && f.message)
            .slice(0, 4)
            .map((f) => (
              <div key={`${f.id}-msg`} style={{ padding: "3px 9px 4px 12px", fontSize: 9.5, color: C.crit, lineHeight: 1.3 }}>
                {f.name}: {f.message}
              </div>
            ))}
        </div>
      </div>

      {(notice || error || unavailable || checkMessage) && (
        <div style={{ flex: "none", fontSize: 9.5, lineHeight: 1.3, color: unavailable || error || checkFailed ? C.crit : notice || checking ? C.warn : C.ok }}>
          {unavailable ? meta.detail || "this task cannot run yet" : error || notice || checkMessage}
        </div>
      )}

      <div style={{ flex: "none", display: "flex", flexDirection: "column", gap: 6 }}>
        {setAnimateOnTwin && (
          <label style={{ display: "inline-flex", alignItems: "center", gap: 6, cursor: "pointer", fontSize: 10, color: C.text2, userSelect: "none" }}>
            <input
              type="checkbox"
              checked={animateOnTwin}
              onChange={(e) => setAnimateOnTwin(e.target.checked)}
              style={{ accentColor: C.accent, cursor: "pointer" }}
            />
            <span style={{ fontWeight: 600, letterSpacing: "0.04em" }}>INCLUDE REPLAY FRAMES</span>
            <span className="mono" style={{ fontSize: 9, color: C.dim2 }}>
              (stream frames)
            </span>
          </label>
        )}

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button
            type="button"
            className="nx-btn"
            onClick={run}
            disabled={running || checking || !queued || unavailable}
            style={{ borderColor: C.accent, background: C.accentBg, color: C.accentBright, height: 26, fontSize: 10.5 }}
          >
            {running ? `RUNNING ${done}/${total}` : `RUN ${queued || ""}`}
          </button>
          <span className="mono" style={{ fontSize: 9.5, color: C.dim2 }}>
            {running
              ? animateOnTwin
                ? "streaming preview frames, one file at a time"
                : "uploading in batches, one file predicted at a time"
              : queued
                ? `${queued} queued · ${fmtBytes(files.filter((f) => f.status === "queued").reduce((a, f) => a + f.size, 0))}`
                : "queue files to run"}
          </span>
        </div>
      </div>
    </div>
  );
}
