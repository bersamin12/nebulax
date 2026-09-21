// Refreshing the plan queries R2's accepted parts; failed requests can be retried safely.
// Presigned URLs stay in memory and are never logged or put into localStorage.
export async function r2MultipartUpload(file, { getSession, complete, signal,
  onProgress = () => {}, fetcher = fetch } = {}) {
  let failures = 0;
  while (true) {
    signal?.throwIfAborted();
    const plan = await getSession();
    if (plan.complete) return;
    let sent = plan.parts.filter(p => p.done).reduce((sum, p) => sum + p.size, 0);
    onProgress(sent, file.size);
    try {
      for (const part of plan.parts) {
        signal?.throwIfAborted();
        if (part.done) continue;
        const start = (part.number - 1) * plan.chunk_size;
        const res = await fetcher(part.url, { method: "PUT", credentials: "omit", signal,
          body: file.slice(start, start + part.size) });
        if (!res.ok) throw new Error(`Upload part failed (${res.status})`);
        sent += part.size;
        onProgress(sent, file.size);
      }
      await complete();
      return;
    } catch (error) {
      if (signal?.aborted) throw error;
      if (++failures > 3) throw new Error("Upload interrupted. Select Run to resume accepted parts.");
      await new Promise(resolve => setTimeout(resolve, 500 * 2 ** failures));
    }
  }
}
