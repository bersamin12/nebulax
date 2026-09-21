// The predict page's state machine: the task list, the upload queue, and the batched session.
//
// Cloud deployments upload directly to GCS with resumable sessions, then send file references.
// Local fallback: `POST /api/ps3/{task}/predict` takes bounded multipart requests and predicts the
// files one at a time, appending the rows to an upload *session*. Cloud Run HTTP/1 rejects a
// whole request above 32 MiB, so batches are capped by both file count and 30 MiB of file data.
// The page sends the first batch with no session and passes its token with every following batch.
// The server answers with **all** rows accumulated so far (so `rows` is a replace, not an append) and with
// the explanations of *that batch only* (so `explanations` is a merge by file_id).
//
// Nothing is retried automatically: a file the server refuses comes back in `errors` and is
// marked on its queue row, and the batch after it still runs.
//
// STOP aborts the request in flight (`cancel`): the rows already returned stay, the file or batch
// that was cut off goes back to `queued` with the rest, and nothing is treated as a server fault.
// The server may still finish the cut-off file on its side; a re-run then reports it as "already
// predicted", which the page reads as done rather than as an error.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getSavedRun, predictSavedInput, retryRunSave, deletePs3Session, getPs3Tasks, postPs3LocalStream, postPs3Predict, postPs3Stream, ps3CsvUrl } from "../api.js";
import { TASK_ORDER, TASK_SUFFIXES } from "./taskMeta.js";
import { createUploadBatch, getUploadSession, predictUploadedFile } from "../api.js";
import { resumableUpload } from "./resumableUpload.js";
import { r2MultipartUpload } from "./r2MultipartUpload.js";
import { completeUpload } from "../api.js";

let SEQ = 0;
const nextId = () => `f${++SEQ}`;

// Leaves roughly 2 MiB below Cloud Run's 32 MiB HTTP/1 request limit for multipart headers.
export const MAX_UPLOAD_BATCH_BYTES = 30 * 1024 * 1024;

/** Preserve queue order while bounding both the file count and total bytes in each request. */
export function uploadBatches(items, maxFiles = 32, maxBytes = MAX_UPLOAD_BATCH_BYTES) {
  const countLimit = Math.max(1, Number(maxFiles) || 1);
  const byteLimit = Math.max(1, Number(maxBytes) || 1);
  const batches = [];
  let batch = [];
  let bytes = 0;
  for (const item of items || []) {
    const size = Math.max(0, Number(item && (item.size ?? (item.file && item.file.size))) || 0);
    if (batch.length && (batch.length >= countLimit || bytes + size > byteLimit)) {
      batches.push(batch);
      batch = [];
      bytes = 0;
    }
    batch.push(item);
    bytes += size;
  }
  if (batch.length) batches.push(batch);
  return batches;
}

/** Accepted-suffix test, the same rule the API applies (case-insensitive, leading dot). */
export function accepts(suffixes, name) {
  const dot = String(name || "").lastIndexOf(".");
  const suf = dot < 0 ? "" : String(name).slice(dot).toLowerCase();
  return (suffixes || []).some((s) => s.toLowerCase() === suf);
}

/** Every File under a dropped DataTransfer, folders included (webkitGetAsEntry, depth-first). */
async function filesFromDataTransfer(dt) {
  const out = [];
  const items = dt && dt.items ? Array.from(dt.items) : [];
  const entries = items
    .map((it) => (it.webkitGetAsEntry ? it.webkitGetAsEntry() : null))
    .filter(Boolean);
  if (!entries.length) return Array.from((dt && dt.files) || []);
  const walk = async (entry) => {
    if (entry.isFile) {
      const file = await new Promise((res, rej) => entry.file(res, rej));
      out.push(file);
      return;
    }
    if (!entry.isDirectory) return;
    const reader = entry.createReader();
    for (;;) {
      // readEntries hands back at most ~100 at a time; keep calling until it is empty
      const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
      if (!batch.length) break;
      for (const e of batch) await walk(e);
    }
  };
  for (const e of entries) await walk(e);
  return out;
}

export { filesFromDataTransfer };

export function usePs3Predict(initialTask) {
  const [tasks, setTasks] = useState(null); // null = still loading
  const [tasksError, setTasksError] = useState(null);
  const [task, setTaskState] = useState(
    TASK_ORDER.includes(initialTask) ? initialTask : TASK_ORDER[0]
  );
  // per-task queue + results, so switching tabs never mixes two subsystems in one session
  const [state, setState] = useState(() => Object.fromEntries(TASK_ORDER.map((t) => [t, blank()])));
  const [running, setRunning] = useState(false);
  const [fatal, setFatal] = useState(null); // "the server is unreachable" banner
  // simulated missing columns per task (Door / ACV): canonical fields sent as `drop_columns`
  const [dropFields, setDropFieldsState] = useState(() => Object.fromEntries(TASK_ORDER.map((t) => [t, []])));
  const abort = useRef(null);
  const uploads = useRef(new Map());

  const reload = useCallback(() => {
    setTasksError(null);
    return getPs3Tasks()
      .then((d) => {
        setTasks(Array.isArray(d) ? d : []);
        setFatal(null);
        return d;
      })
      .catch((e) => {
        setTasks([]);
        setTasksError(e);
        setFatal(e);
        return null;
      });
  }, []);

  useEffect(() => {
    let live = true;
    getPs3Tasks()
      .then((d) => live && (setTasks(Array.isArray(d) ? d : []), setFatal(null)))
      .catch((e) => live && (setTasks([]), setTasksError(e), setFatal(e)));
    return () => {
      live = false;
    };
  }, []);

  const meta = useMemo(
    () => (tasks || []).find((t) => t.name === task) || null,
    [tasks, task]
  );
  const cur = state[task];

  const patch = useCallback(
    (name, fn) => setState((s) => ({ ...s, [name]: { ...s[name], ...fn(s[name]) } })),
    []
  );

  const setTask = useCallback((name) => {
    if (TASK_ORDER.includes(name)) setTaskState(name);
  }, []);

  /** Queue files, splitting off the ones this subsystem does not accept (they never upload). */
  const addFiles = useCallback(
    (list) => {
      const suffixes = (meta && meta.accepts) || TASK_SUFFIXES[task];
      const maxBytes = (meta && meta.max_file_bytes) || 64 * 1024 * 1024;
      patch(task, (s) => {
        const seen = new Set(s.files.map((f) => f.name));
        const add = [];
        let ignored = 0;
        for (const file of list) {
          const name = String(file.name || "").split(/[\\/]/).pop();
          const cloudRun = file.cloudRun || null;
          const localPath = typeof file.localPath === "string" ? file.localPath : null;
          if (!name || seen.has(name)) {
            ignored += name ? 1 : 0;
            continue;
          }
          seen.add(name);
          if (!accepts(suffixes, name)) {
            ignored += 1;
            continue;
          }
          const tooBig = file.size > maxBytes;
          add.push({
            id: nextId(),
            file: localPath || cloudRun ? null : file,
            cloudRun,
            localPath,
            name,
            size: file.size,
            status: tooBig ? "error" : "queued",
            message: tooBig ? `larger than the ${Math.round(maxBytes / 1024 ** 2)} MB per-file limit` : "",
          });
        }
        return {
          files: [...s.files, ...add],
          notice: ignored
            ? `${ignored} file${ignored > 1 ? "s" : ""} ignored (this task accepts ${suffixes.join(" or ")}, and each name once per session)`
            : "",
        };
      });
    },
    [meta, patch, task]
  );

  const addLocalPaths = useCallback(
    (entries) => addFiles(entries.map((entry) => ({ ...entry, localPath: entry.path }))),
    [addFiles]
  );

  const removeFile = useCallback(
    (id) => patch(task, (s) => ({ files: s.files.filter((f) => f.id !== id) })),
    [patch, task]
  );

  /** Switch one optional field off or on for the current task (applies to the next RUN). */
  const toggleDropField = useCallback(
    (field) =>
      setDropFieldsState((d) => {
        const cur = d[task] || [];
        return { ...d, [task]: cur.includes(field) ? cur.filter((f) => f !== field) : [...cur, field] };
      }),
    [task]
  );
  const setDropFields = useCallback((fields) => setDropFieldsState((d) => ({ ...d, [task]: [...fields] })), [task]);

  /** Upload + predict every queued file, using per-file requests only for local paths. */
  const run = useCallback(async () => {
    const name = task;
    const snapshot = state[name];
    const queued = snapshot.files.filter((f) => f.status === "queued");
    if (!queued.length || running || !meta) return;
    const drop = dropFields[name] || [];
    const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    abort.current = ctrl;
    setRunning(true);
    patch(name, (s) => ({
      total: queued.length,
      done: 0,
      notice: "",
      error: "",
      files: s.files.map((f) => (f.status === "queued" ? { ...f, status: "waiting", message: "" } : f)),
    }));

    let session = snapshot.session;
    let runId = snapshot.storage?.run?.id;
    let stopped = null;

    const direct = !!meta?.direct_uploads;
    if (direct) {
      const fresh = queued.filter((f) => f.file && !uploads.current.has(f.id));
      if (fresh.length) {
        try {
          const batch = await createUploadBatch(name, fresh, snapshot.runName, ctrl?.signal);
          for (const item of fresh) uploads.current.set(item.id, { batchId: batch.id });
        } catch (err) {
          patch(name, (s) => ({ ...stoppedState(s, queued.map((f) => f.id)), error: ctrl?.signal.aborted ? "" : err.message }));
          abort.current = null;
          setRunning(false);
          return;
        }
      }
    }

    if (direct || queued.some((item) => item.localPath || item.cloudRun)) {
      for (const item of queued) {
        patch(name, (s) => ({
          files: s.files.map((f) => (f.id === item.id ? { ...f, status: "running" } : f)),
        }));
        let res;
        try {
          if (direct && item.file) {
            const upload = uploads.current.get(item.id);
            if (!upload.complete) {
              if (!upload.url) Object.assign(upload, await getUploadSession(upload.batchId, item.name, ctrl?.signal));
              if (!upload.complete) {
                try {
                  const options = { signal: ctrl?.signal,
                    onProgress: (sent, total) => patch(name, (s) => ({ files: s.files.map((f) => f.id === item.id
                      ? { ...f, message: `Uploading to storage: ${Math.floor(sent / total * 100)}%` } : f) })) };
                  if (upload.provider === "r2") await r2MultipartUpload(item.file, { ...options,
                    getSession: () => getUploadSession(upload.batchId, item.name, ctrl?.signal),
                    complete: () => completeUpload(upload.batchId, item.name, ctrl?.signal) });
                  else await resumableUpload(item.file, upload.url, options);
                  upload.complete = true;
                  delete upload.url;
                } catch (err) {
                  if (err.expired) delete upload.url;
                  throw err;
                }
              }
            }
            patch(name, (s) => ({ files: s.files.map((f) => f.id === item.id ? { ...f, message: "Predicting from cloud storage" } : f) }));
            res = await predictUploadedFile(upload.batchId, item.name, drop, ctrl?.signal, runId);
            runId = res.storage?.run?.id || runId || upload.batchId;
          } else res = item.cloudRun
            ? await predictSavedInput(name, item, session, { signal: ctrl?.signal, runName: snapshot.runName, dropFields: drop })
            : item.localPath
            ? await postPs3LocalStream(name, item.localPath, session, { signal: ctrl ? ctrl.signal : undefined, runName: snapshot.runName, dropFields: drop })
            : await postPs3Stream(name, item.file, session, { signal: ctrl ? ctrl.signal : undefined, dropFields: drop, runName: snapshot.runName });
        } catch (err) {
          if (err.cancelled || (ctrl && ctrl.signal.aborted)) {
            stopped = err;
            patch(name, (s) => stoppedState(s, [item.id]));
            break;
          }
          const msg = err.detail || err.message || String(err);
          const perFile = err.status === 400 || err.status === 413;
          const finished = donePreviously(msg); // cut off by STOP, but the server had finished it
          patch(name, (s) => ({
            files: s.files.map((f) =>
              f.id === item.id
                ? finished
                  ? { ...f, status: "done", message: "predicted before the stop" }
                  : { ...f, status: direct && item.file && !perFile ? "queued" : "error", message: msg }
                : f.status === "waiting" && !perFile
                  ? { ...f, status: "queued" }
                  : f
            ),
            error: finished ? "" : msg,
            done: s.done + 1,
          }));
          if (perFile) continue;
          stopped = err;
          if (!err.status && !(direct && item.file)) setFatal(err);
          break;
        }

        const failed = new Map((res.errors || []).map((e) => [String(e.file), String(e.message)]));
        patch(name, (s) => ({
          session: res.session,
          storage: res.storage || s.storage,
          csvUrl: res.csv_url || ps3CsvUrl(res.session),
          outputFilename: res.output_filename || s.outputFilename,
          rows: Array.isArray(res.rows) ? res.rows : s.rows,
          explanations: mergeExplanations(s.explanations, res.explanations),
          done: s.done + 1,
          nDone: Number(res.n_done) || s.nDone,
          expiresAt: res.expires_at || s.expiresAt,
          error: "",
          files: s.files.map((f) =>
            f.id === item.id
              ? failed.has(f.name)
                ? donePreviously(failed.get(f.name))
                  ? { ...f, status: "done", message: "predicted before the stop" }
                  : { ...f, status: "error", message: failed.get(f.name) }
                : { ...f, status: "done", message: "" }
              : f
          ),
        }));
        session = res.session;
        runId = res.storage?.run?.id || runId;
        setFatal(null);
      }
    } else {
      const batchSize = Math.max(1, Math.min(32, Number(meta && meta.max_files_per_request) || 32));
      for (const batch of uploadBatches(queued, batchSize)) {
        const ids = new Set(batch.map((f) => f.id));
        patch(name, (s) => ({
          files: s.files.map((f) => (ids.has(f.id) ? { ...f, status: "running" } : f)),
        }));
        let res;
        try {
          res = await postPs3Predict(name, batch.map((f) => f.file), session, {
            signal: ctrl ? ctrl.signal : undefined,
            dropFields: drop,
            runName: snapshot.runName,
          });
        } catch (err) {
          if (err.cancelled || (ctrl && ctrl.signal.aborted)) {
            stopped = err;
            patch(name, (s) => stoppedState(s, [...ids]));
            break;
          }
          const msg = err.detail || err.message || String(err);
          const perBatch = err.status === 400 || err.status === 413;
          const finished = donePreviously(msg); // a one-file batch cut off by STOP that the server finished
          patch(name, (s) => ({
            files: s.files.map((f) =>
              ids.has(f.id)
                ? finished
                  ? { ...f, status: "done", message: "predicted before the stop" }
                  : { ...f, status: "error", message: msg }
                : f.status === "waiting" && !perBatch ? { ...f, status: "queued" } : f
            ),
            error: finished ? "" : msg,
            done: s.done + (perBatch ? batch.length : 0),
          }));
          if (perBatch) continue;
          stopped = err;
          if (!err.status) setFatal(err);
          break;
        }
        const failed = new Map((res.errors || []).map((e) => [String(e.file), String(e.message)]));
        patch(name, (s) => ({
          session: res.session,
          storage: res.storage || s.storage,
          csvUrl: res.csv_url || ps3CsvUrl(res.session),
          outputFilename: res.output_filename || s.outputFilename,
          rows: Array.isArray(res.rows) ? res.rows : s.rows,
          explanations: mergeExplanations(s.explanations, res.explanations),
          done: s.done + batch.length,
          nDone: Number(res.n_done) || s.nDone,
          expiresAt: res.expires_at || s.expiresAt,
          error: "",
          files: s.files.map((f) =>
            ids.has(f.id)
              ? failed.has(f.name)
                ? donePreviously(failed.get(f.name))
                  ? { ...f, status: "done", message: "predicted before the stop" }
                  : { ...f, status: "error", message: failed.get(f.name) }
                : { ...f, status: "done", message: "" }
              : f
          ),
        }));
        session = res.session;
        setFatal(null);
      }
    }
    abort.current = null;
    setRunning(false);
    if (!stopped) patch(name, (s) => ({ files: s.files.map((f) => (f.status === "waiting" ? { ...f, status: "queued" } : f)) }));
  }, [dropFields, meta, patch, running, state, task]);

  const cancel = useCallback(() => {
    if (abort.current) abort.current.abort();
  }, []);

  /** Forget the session server-side (it deletes the temp files) and empty the page. */
  const clear = useCallback(() => {
    const name = task;
    const token = state[name].session;
    if (token) deletePs3Session(token).catch(() => {});
    patch(name, () => blank());
  }, [patch, state, task]);

  const setRunName = useCallback((value) => patch(task, () => ({ runName: value })), [patch, task]);
  const openSavedRun = useCallback(async (id) => {
    const name = task;
    const res = await getSavedRun(id);
    if (res.task !== name) throw new Error("This run belongs to another model");
    patch(name, () => ({ ...blank(), rows: res.rows || [], explanations: res.explanations || [],
      files: (res.run.files || []).map((f) => ({ ...f, id: nextId(), cloudRun: id, status: "done", message: "saved input" })),
      csvUrl: res.csv_url, outputFilename: res.output_filename, nDone: res.n_done,
      runName: res.run.name, storage: res.storage,
      notice: res.imported_batch ? "Imported batch results loaded without inference. No explanations were stored for this older run." : "Saved results loaded without running the model. Clear to start a new run." }));
  }, [patch, task]);
  const queueSavedRun = useCallback(async (id) => {
    const name = task;
    const res = await getSavedRun(id);
    if (res.task !== name) throw new Error("This run belongs to another model");
    patch(name, () => ({ ...blank(), runName: `${res.run.name} rerun`.slice(0, 80),
      files: res.run.files.map((f) => ({ ...f, id: nextId(), cloudRun: id, status: "queued", message: "from cloud storage" })),
      notice: "Saved inputs queued. Run will use the current model and column settings." }));
  }, [patch, task]);
  const retrySave = useCallback(async () => {
    const name = task;
    const storage = await retryRunSave(state[name].session);
    patch(name, () => ({ storage }));
  }, [patch, task, state]);

  return {
    setRunName, openSavedRun, queueSavedRun, retrySave,
    tasks,
    tasksError,
    reload,
    task,
    setTask,
    meta,
    running,
    fatal,
    addFiles,
    addLocalPaths,
    removeFile,
    dropFields: dropFields[task] || [],
    toggleDropField,
    setDropFields,
    run,
    cancel,
    clear,
    ...cur,
  };
}

/** The queue after STOP: the cut-off items and everything still waiting go back to `queued`. */
function stoppedState(s, cutIds) {
  const cut = new Set(cutIds);
  const back = s.files.filter((f) => cut.has(f.id) || f.status === "waiting").length;
  return {
    files: s.files.map((f) => (cut.has(f.id) || f.status === "waiting" ? { ...f, status: "queued", message: "" } : f)),
    error: "",
    notice: `Stopped after ${s.done} of ${s.total} files. ${back} ${back === 1 ? "is" : "are"} still queued; select Run to continue.`,
  };
}

/** The server's answer when a file cut off by STOP had been finished on its side after all. */
function donePreviously(message) {
  return /already predicted in this session/i.test(String(message || ""));
}

function blank() {
  return {
    runName: "",
    storage: null,
    files: [],
    rows: [],
    explanations: [],
    session: null,
    csvUrl: null,
    outputFilename: null,
    expiresAt: null,
    done: 0,
    total: 0,
    nDone: 0,
    notice: "",
    error: "",
  };
}

/** Keep one explanation per file_id (the newest wins), appending the batch that just returned. */
function mergeExplanations(prev, incoming) {
  if (!Array.isArray(incoming) || !incoming.length) return prev;
  const out = [...(prev || [])];
  for (const e of incoming) {
    const i = out.findIndex((p) => String(p.file_id) === String(e.file_id));
    if (i >= 0) out[i] = e;
    else out.push(e);
  }
  return out;
}
