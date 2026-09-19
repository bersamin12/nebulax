// The predict page's state machine: the task list, the upload queue, and the batched session.
//
// Why batches: `POST /api/ps3/{task}/predict` takes at most `max_files_per_request` (32) files
// per call and predicts them one at a time, appending the rows to an upload *session*. The 68
// released rail files are ~1.1 GB, so the page slices the queue into batches, sends the first
// one with no session, and passes the token it gets back with every following batch. The server
// answers with **all** rows accumulated so far (so `rows` is a replace, not an append) and with
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
import { deletePs3Session, getPs3Tasks, postPs3LocalStream, postPs3Predict, postPs3Stream, ps3CsvUrl } from "../api.js";
import { TASK_ORDER, TASK_SUFFIXES } from "./taskMeta.js";

let SEQ = 0;
const nextId = () => `f${++SEQ}`;

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
  const abort = useRef(null);

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
            file: localPath ? null : file,
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

  /** Upload + predict every queued file, using per-file requests only for local paths. */
  const run = useCallback(async () => {
    const name = task;
    const snapshot = state[name];
    const queued = snapshot.files.filter((f) => f.status === "queued");
    if (!queued.length || running) return;
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
    let stopped = null;

    if (queued.some((item) => item.localPath)) {
      for (const item of queued) {
        patch(name, (s) => ({
          files: s.files.map((f) => (f.id === item.id ? { ...f, status: "running" } : f)),
        }));
        let res;
        try {
          res = item.localPath
            ? await postPs3LocalStream(name, item.localPath, session, { signal: ctrl ? ctrl.signal : undefined })
            : await postPs3Stream(name, item.file, session, { signal: ctrl ? ctrl.signal : undefined });
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
                  : { ...f, status: "error", message: msg }
                : f.status === "waiting" && !perFile
                  ? { ...f, status: "queued" }
                  : f
            ),
            error: finished ? "" : msg,
            done: s.done + 1,
          }));
          if (perFile) continue;
          stopped = err;
          if (!err.status) setFatal(err);
          break;
        }

        const failed = new Map((res.errors || []).map((e) => [String(e.file), String(e.message)]));
        patch(name, (s) => ({
          session: res.session,
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
        setFatal(null);
      }
    } else {
      const batchSize = Math.max(1, Math.min(32, Number(meta && meta.max_files_per_request) || 32));
      for (let i = 0; i < queued.length; i += batchSize) {
        const batch = queued.slice(i, i + batchSize);
        const ids = new Set(batch.map((f) => f.id));
        patch(name, (s) => ({
          files: s.files.map((f) => (ids.has(f.id) ? { ...f, status: "running" } : f)),
        }));
        let res;
        try {
          res = await postPs3Predict(name, batch.map((f) => f.file), session, {
            signal: ctrl ? ctrl.signal : undefined,
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
  }, [meta, patch, running, state, task]);

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

  return {
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
