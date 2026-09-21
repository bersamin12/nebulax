import test from 'node:test';
import assert from 'node:assert/strict';
import { r2MultipartUpload } from '../src/predict/r2MultipartUpload.js';

test('resume skips accepted parts and completes through API', async () => {
  const puts = [];
  let completed = false;
  await r2MultipartUpload(new Blob(['abcdefgh']), {
    getSession: async () => ({ chunk_size: 4, parts: [
      { number: 1, size: 4, done: true, url: 'part1' },
      { number: 2, size: 4, done: false, url: 'part2' }] }),
    fetcher: async (url, options) => { puts.push([url, await options.body.text()]); return { ok: true }; },
    complete: async () => { completed = true; }
  });
  assert.deepEqual(puts, [['part2', 'efgh']]);
  assert.equal(completed, true);
});

test('aborted upload never completes', async () => {
  const controller = new AbortController();
  let completed = false;
  await assert.rejects(r2MultipartUpload(new Blob(['abcd']), {
    signal: controller.signal,
    getSession: async () => ({ chunk_size: 4, parts: [{ number: 1, size: 4, done: false, url: 'part1' }] }),
    fetcher: async () => { controller.abort(); throw new Error('stopped'); },
    complete: async () => { completed = true; }
  }));
  assert.equal(completed, false);
});

test('already completed session returns without uploading', async () => {
  await r2MultipartUpload(new Blob(['abcd']), {
    getSession: async () => ({ complete: true }),
    fetcher: async () => { assert.fail('must not upload'); },
    complete: async () => { assert.fail('must not complete twice'); }
  });
});
