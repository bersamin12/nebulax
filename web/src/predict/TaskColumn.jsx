// Left column: which subsystem, what it accepts, what it scored in cross-validation, and the
// upload queue. The CV line, the accepted suffixes and the batch size all come from
// GET /api/ps3/tasks - a task whose module or artefact is missing says so and cannot be run.
import { useCallback, useEffect, useRef, useState } from "react";
import { C } from "../lib/format.js";
import { inspectPs3LocalPath } from "../api.js";
import ColumnToggles from "./ColumnToggles.jsx";
import FileConfirm from "./FileConfirm.jsx";
import { checkFiles } from "./fileCheck.js";
import { cvScore, fmtBytes, INFO_META, SYSTEM_ORDER, TASK_META, TASK_SUFFIXES } from "./taskMeta.js";
import { filesFromDataTransfer } from "./usePs3Predict.js";

const STATUS_COLOR = { queued: C.dim2, waiting: C.dim, running: C.accent, done: C.ok, error: C.crit };
const STATUS_WORD = { queued: "queued", waiting: "waiting", running: "running", done: "done", error: "error" };

/**
 * One system row of the accordion: a thumbnail, the name and its headline (the CV score of the
 * selected model, or RESEARCH for the two dataset profiles). The active row opens to show the
 * description underneath; choosing another row closes it again.
 */
function SystemRow({ name, entry, active, open, onClick }) {
  const infoOnly = !!INFO_META[name];
  const meta = TASK_META[name] || INFO_META[name] || {};
  const cv = cvScore(entry && entry.cv);
  const off = entry && entry.available === false;
  return (
    <div className={"nx-system-row" + (active ? " is-active" : "") + (infoOnly ? " is-research" : "")}>
      <button
        type="button"
        onClick={onClick}
        className="nx-system-btn"
        aria-pressed={active}
        aria-expanded={active && open}
        style={{ opacity: off ? 0.62 : 1 }}
        title={off ? entry.detail || "System unavailable" : undefined}
      >
        <span className="nx-system-thumb" aria-hidden="true">
          {meta.thumb ? <img src={meta.thumb} alt="" loading="lazy" /> : null}
        </span>
        <span className="nx-system-name">{meta.tab || name}</span>
        <span className="nx-system-cv mono" style={{ color: infoOnly ? C.dim2 : off ? C.crit : cv ? C.violet : C.dim2 }}>
          {infoOnly ? "RESEARCH" : off ? "UNAVAILABLE" : cv ? `${cv.label} ${Number(cv.value).toFixed(3)}` : entry && entry.cv ? "cv: no headline" : "no cv yet"}
        </span>
        <svg className="nx-system-chev" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M6 9l6 6 6-6" /></svg>
      </button>
      {active && open && <SystemInfo name={name} entry={entry} />}
    </div>
  );
}

/** The active system's description: what it does, what it eats, and how it was validated. */
function SystemInfo({ name, entry }) {
  const infoOnly = !!INFO_META[name];
  const meta = TASK_META[name] || INFO_META[name] || {};
  const cv = cvScore(entry && entry.cv);
  return (
    <div className="nx-system-info">
      <div style={{ fontSize: 11, fontWeight: 600, color: C.text }}>
        {meta.title || name}
        {infoOnly && <span className="mono" style={{ fontSize: 8.5, color: C.dim2, marginLeft: 6 }}>READ-ONLY DATASET · NO PREDICTION</span>}
      </div>
      <div style={{ fontSize: 10.5, color: C.text2, lineHeight: 1.4 }}>{meta.blurb}</div>
      <div style={{ fontSize: 9.5, color: C.dim, lineHeight: 1.4 }}>
        {infoOnly
          ? `${meta.dataset} · ${meta.size}`
          : `input: ${meta.input}${entry && entry.cv && entry.cv.scheme ? ` · CV: ${entry.cv.scheme}` : ""}${cv ? ` · ${cv.label} ${Number(cv.value).toFixed(3)}` : ""}`}
      </div>
    </div>
  );
}

export default function TaskColumn({
  tasks,
  task,
  setTask,
  resetKey = 0,
  meta,
  files,
  addFiles,
  addLocalPaths,
  removeFile,
  run,
  running,
  cancel,
  done,
  total,
  notice,
  error,
  dropFields = [],
  toggleDropField,
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
  const [pending, setPending] = useState(null); // { items: [{file, report}] } while the confirm dialog is up
  const [showInfo, setShowInfo] = useState(true);
  // canonical -> raw column of the last confirmed file, so the column toggles can mark absent ones
  const [lastFound, setLastFound] = useState(null);
  const droppable = (meta && meta.droppable_fields) || [];
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
    setPending(null);
    setLastFound(null);
    if (fileRef.current) fileRef.current.value = "";
    if (dirRef.current) dirRef.current.value = "";
  }, [resetKey, task]);

  /**
   * A pick never queues directly: the files are read (header, or the ACV workbook report from
   * the server), the dialog shows what they carry against what the model expects, and CONFIRM
   * queues the ones that match.
   */
  const queuePicked = useCallback(async (list) => {
    if (!list.length) return;
    checkAbort.current?.abort();
    const ctrl = new AbortController();
    checkAbort.current = ctrl;
    setChecking(true);
    setCheckMessage("");
    setCheckFailed(false);
    setPending({ items: list.map((file) => ({ file, report: { name: file.name, size: file.size, ok: false, columns: [], found: {}, missing: [], problems: [], detail: "" } })) });
    try {
      const reports = await checkFiles(task, list, { signal: ctrl.signal });
      if (ctrl.signal.aborted || checkAbort.current !== ctrl) return;
      setPending({ items: list.map((file, i) => ({ file, report: reports[i] })) });
    } catch (err) {
      if (ctrl.signal.aborted) return;
      setPending(null);
      setCheckMessage(`could not read the files: ${err.message || err}`);
      setCheckFailed(true);
    } finally {
      if (checkAbort.current === ctrl) {
        checkAbort.current = null;
        setChecking(false);
      }
    }
  }, [task]);

  const confirmPicked = useCallback((valid) => {
    const total = pending ? pending.items.length : valid.length;
    const first = pending && pending.items.find((it) => it.report.ok && valid.includes(it.file));
    if (first && task === "door") setLastFound(first.report.found || {});
    setPending(null);
    if (valid.length) addFiles(valid);
    setCheckMessage(
      valid.length === total
        ? `${valid.length} file${valid.length === 1 ? "" : "s"} added`
        : `${valid.length} of ${total} files were added. The other ${total - valid.length} did not match the required format.`
    );
    setCheckFailed(valid.length < total);
  }, [addFiles, pending, task]);

  const cancelPicked = useCallback(() => {
    checkAbort.current?.abort();
    checkAbort.current = null;
    setChecking(false);
    setPending(null);
    setCheckMessage("File selection cancelled");
    setCheckFailed(false);
    if (fileRef.current) fileRef.current.value = "";
    if (dirRef.current) dirRef.current.value = "";
  }, []);

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
        setPickMessage("No files were selected. Choose a file or drop it here.");
        return;
      }
      setPickMessage("");
      // Leave the selected filename visible. Firefox may still be completing the native
      // picker handoff when this change handler returns.
      void queuePicked(list)
        .catch((err) => setPickMessage(`Could not add the selection: ${err.message || err}`));
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

  const entryOf = (name) => (tasks || []).find((t) => t.name === name) || null;
  const systemPicker = (
    <div data-tour="systems" style={{ display: "flex", flexDirection: "column", gap: 6, flex: "none" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6 }}>
        <span style={{ fontSize: 9.5, letterSpacing: "0.14em", color: C.dim }}>SYSTEMS</span>
        <span className="mono" style={{ fontSize: 9, color: C.dim2 }}>4 prediction models · 2 research datasets</span>
      </div>
      <div className="nx-system-list">
        {SYSTEM_ORDER.map((name) => (
          <SystemRow
            key={name}
            name={name}
            entry={entryOf(name)}
            active={name === task}
            open={showInfo}
            onClick={() => {
              if (name === task) setShowInfo((v) => !v);
              else {
                setShowInfo(true);
                setTask(name);
              }
            }}
          />
        ))}
      </div>
    </div>
  );

  const confirmDialog = pending ? (
    <FileConfirm
      task={task}
      items={pending.items}
      checking={checking}
      onConfirm={confirmPicked}
      onCancel={cancelPicked}
      droppable={droppable}
      dropFields={dropFields}
      toggleDropField={toggleDropField}
    />
  ) : null;
  const columnStrip = droppable.length ? (
    <ColumnToggles fields={droppable} dropped={dropFields} onToggle={toggleDropField} found={lastFound} compact disabled={running} />
  ) : null;

  if (infoOnly) {
    return (
      <div style={{ width, height, display: "flex", flexDirection: "column", gap: 8, minHeight: 0 }}>
        {systemPicker}
        <div className="nx-info-card" style={{ flex: 1, minHeight: 0, padding: 16 }}>
          <span className="nx-eyebrow">DATASET PROFILE</span>
          <h2>{tmeta.dataset}</h2>
          <p>{tmeta.blurb}</p>
          <p><strong>{tmeta.size}</strong><br />{tmeta.signals}</p>
          <span className="nx-info-badge">READ-ONLY DATASET · NO PREDICTION</span>
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
        data-tour="files"
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
          Drop {tmeta.multiple ? "files or a folder" : "the file"} here ({accept}
          {meta && meta.max_files_per_request ? `, ${meta.max_files_per_request} per request` : ""})
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
              title="Choose files"
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
                title="Choose a folder"
              />
            </div>
          )}
        </div>
        {pickMessage && <div role="alert" style={{ fontSize: 10, color: C.crit }}>{pickMessage}</div>}
        {meta?.local_paths && (
          <div style={{ display: "flex", flexDirection: "column", gap: 3, borderTop: `1px solid ${C.line}`, paddingTop: 6 }}>
            <span className="nx-eyebrow">LOCAL PATH · USE IF THE FILE PICKER DOES NOT WORK</span>
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

      {columnStrip}

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
            <div style={{ padding: "10px 9px", fontSize: 10.5, color: C.dim2 }}>No files added</div>
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
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button
            type="button"
            className="nx-btn"
            data-tour="run"
            onClick={run}
            disabled={running || checking || !queued || unavailable}
            style={{ borderColor: C.accent, background: C.accentBg, color: C.accentBright, height: 26, fontSize: 10.5 }}
          >
            {running ? `PROCESSING ${done} OF ${total}` : queued ? `RUN ${queued} FILE${queued === 1 ? "" : "S"}` : "RUN"}
          </button>
          {running && (
            <button
              type="button"
              className="nx-btn"
              onClick={cancel}
              title="Stop after the current file. Completed results will remain, and unprocessed files will stay in the queue."
              style={{ borderColor: C.crit, color: C.crit, height: 26, fontSize: 10.5 }}
            >
              STOP
            </button>
          )}
          <span className="mono" style={{ fontSize: 9.5, color: C.dim2 }}>
            {running
              ? "Processing files"
              : queued
                ? `${queued} queued · ${fmtBytes(files.filter((f) => f.status === "queued").reduce((a, f) => a + f.size, 0))}`
                : "Add files to begin"}
          </span>
        </div>
      </div>
      {confirmDialog}
    </div>
  );
}
