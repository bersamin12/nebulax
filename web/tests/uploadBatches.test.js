import assert from "node:assert/strict";
import test from "node:test";

import { MAX_UPLOAD_BATCH_BYTES, uploadBatches } from "../src/predict/usePs3Predict.js";

const MiB = 1024 * 1024;
const files = (count, size) => Array.from({ length: count }, (_, id) => ({ id, size }));

test("keeps released Rail files in separate Cloud Run requests", () => {
  const batches = uploadBatches(files(68, 16.7 * MiB), 32);
  assert.equal(batches.length, 68);
  assert.ok(batches.every((batch) => batch.length === 1));
});

test("groups SHM files without crossing the byte cap", () => {
  const batches = uploadBatches(files(16, 6.54 * MiB), 32);
  assert.deepEqual(batches.map((batch) => batch.length), [4, 4, 4, 4]);
  assert.ok(batches.every((batch) => batch.reduce((sum, file) => sum + file.size, 0) <= MAX_UPLOAD_BATCH_BYTES));
});

test("also respects the server file-count limit", () => {
  assert.deepEqual(uploadBatches(files(7, 1), 3).map((batch) => batch.length), [3, 3, 1]);
});

test("keeps an individually oversized file as its own request", () => {
  const large = { id: "large", size: MAX_UPLOAD_BATCH_BYTES + 1 };
  const batches = uploadBatches([{ id: "small", size: 1 }, large, { id: "last", size: 1 }]);
  assert.deepEqual(batches.map((batch) => batch.map((file) => file.id)), [["small"], ["large"], ["last"]]);
});
