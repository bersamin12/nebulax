// GCS session URLs are upload capabilities: keep them in memory, never log them.
export async function resumableUpload(file, url, { signal, onProgress = () => {}, fetcher = fetch } = {}) {
  const chunkSize = 8 * 1024 * 1024; // GCS requires multiples of 256 KiB except the final chunk.
  let offset = 0;
  let failures = 0;
  let probe = true; // A previous attempt may already have committed some or all bytes.
  while (true) {
    signal?.throwIfAborted();
    const end = Math.min(offset + chunkSize, file.size);
    let res;
    try {
      res = await fetcher(url, {
        method: "PUT", signal, credentials: "omit",
        headers: { "Content-Range": probe ? `bytes */${file.size}` : `bytes ${offset}-${end - 1}/${file.size}` },
        body: probe ? new Blob([]) : file.slice(offset, end),
      });
      if (res.status >= 500 || res.status === 429) throw new Error("Storage temporarily unavailable");
    } catch (err) {
      if (signal?.aborted) throw err;
      if (++failures > 4) throw new Error("Upload interrupted. Select Run to resume the remaining bytes.");
      await new Promise((resolve) => setTimeout(resolve, Math.min(500 * 2 ** failures, 8000)));
      probe = true;
      continue;
    }
    if (res.status === 200 || res.status === 201) {
      onProgress(file.size, file.size);
      return;
    }
    if (res.status !== 308) {
      const err = new Error(res.status === 404 || res.status === 410
        ? "Upload session expired. Select Run to start a fresh upload."
        : `Storage upload failed (${res.status}). Select Run to retry.`);
      err.expired = res.status === 404 || res.status === 410;
      throw err;
    }
    const range = res.headers.get("Range");
    const match = range && /^bytes=0-(\d+)$/.exec(range);
    const next = match ? Number(match[1]) + 1 : 0;
    if (next < 0 || next > file.size || (!probe && next <= offset)) {
      throw new Error("Storage did not acknowledge upload progress. Select Run to retry.");
    }
    offset = next;
    onProgress(offset, file.size);
    failures = 0;
    probe = false;
  }
}
