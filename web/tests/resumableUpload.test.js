import assert from "node:assert/strict";
import test from "node:test";
import { resumableUpload } from "../src/predict/resumableUpload.js";

const response = (status, range) => ({ status, headers: new Headers(range ? { Range: range } : {}) });

test("resumes from the storage offset and sends bounded chunks", async () => {
  const file = new Blob([new Uint8Array(20 * 1024 * 1024)]);
  const calls = [];
  const replies = [response(308, "bytes=0-8388607"), response(308, "bytes=0-16777215"), response(200)];
  await resumableUpload(file, "https://storage.example/session", { fetcher: async (_, req) => {
    calls.push(req); return replies.shift();
  } });
  assert.deepEqual(calls.map((c) => c.headers["Content-Range"]), [
    "bytes */20971520", "bytes 8388608-16777215/20971520", "bytes 16777216-20971519/20971520",
  ]);
  assert.deepEqual(calls.map((c) => c.body.size), [0, 8388608, 4194304]);
});

test("completed uploads are not sent again", async () => {
  let calls = 0;
  await resumableUpload(new Blob(["abc"]), "unused", { fetcher: async () => { calls++; return response(200); } });
  assert.equal(calls, 1);
});

test("expired sessions and cancellation remain recoverable", async () => {
  await assert.rejects(resumableUpload(new Blob(["abc"]), "unused", {
    fetcher: async () => response(410),
  }), (e) => e.expired === true);
  const ctrl = new AbortController();
  ctrl.abort();
  await assert.rejects(resumableUpload(new Blob(["abc"]), "unused", { signal: ctrl.signal,
    fetcher: async () => assert.fail("must not send after stop"),
  }), { name: "AbortError" });
});

test("lost chunk response probes the committed offset before sending more bytes", async () => {
  const ranges = [];
  let i = 0;
  await resumableUpload(new Blob(["abcdef"]), "unused", { fetcher: async (_, req) => {
    ranges.push(req.headers["Content-Range"]);
    if (++i === 1) return response(308);
    if (i === 2) throw new Error("connection lost after commit");
    return response(200);
  } });
  assert.deepEqual(ranges, ["bytes */6", "bytes 0-5/6", "bytes */6"]);
});
