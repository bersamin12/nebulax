# NebulaX on Render Free

Prepared in the separate `nebulax-free` checkout on branch `migration/hf-r2` (the name reflects the initial migration attempt). Google deployment and storage remain independent.

## Local verification (22 September 2026)

The Render Docker image built successfully. All 37 backend/API tests passed, including concurrency rejection and lock recovery; the shared frontend previously passed all 11 tests and its production build. A local Linux container limited to **512 MiB RAM, no additional swap, and 0.1 CPU** completed all four HTTP prediction requests. Peak cgroup memory across the test was **291.3 MiB**, with no OOM event.

| Model | Test input | Request time |
|---|---|---|
| Door | Three recorded door cycles | 22.3 s |
| ACV | 30-row workbook excerpt | 8.5 s |
| Rail | 10,000 rows / 16.5 MiB, expanded by repeating the fixture | 27.9 s |
| SHM | 581,120 samples / 5.5 MiB, expanded by repeating the fixture | 6.8 s |

These are local capacity probes with model loading included, not production latency guarantees or accuracy benchmarks. Tests used the multipart HTTP path without R2 credentials. Live direct uploads and storage operations still need verification after account setup. The machine-readable local report is ignored at `data/render-resource-check.json`; `scripts/verify_free_container.py` reproduces the requests against a running container named `nebulax-render-check` on localhost port 8769.

## Account setup

1. Sign up at https://dashboard.render.com/ and connect the GitHub account with access to `bersamin12/nebulax`. Authorize Render to read that repository. Use a free Hobby workspace and the **Free** service instance; do not choose a paid instance.
2. Under **Account Settings → API Keys**, create a key and enter it in the existing local `.env` as `RENDER_API_KEY`. Do not paste credentials into chat.
3. Leave `RENDER_OWNER_ID` blank if the key sees just one workspace. If there are several, set it to the intended workspace ID shown in Render's workspace settings.
4. Fill the R2 fields: `NEBULAX_RUN_BUCKET`, `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`. Use a private Standard bucket and an Object Read & Write token restricted to that bucket. Hugging Face fields are no longer needed.
5. Tell the assistant the file is ready. It can check access, copy and verify the Google archives, and create the Render service.

The .env already contains repository, branch and service name defaults. Those values are not credentials. The key is read locally and only sent to Render's API; R2 secrets are supplied as service environment variables. `.env` is excluded from Git and Docker builds.

## Deployment

```powershell
Set-Location C:\Users\ganqi\Projects\nebulax-free
.venv\Scripts\python.exe scripts/render_deploy.py --check
.venv\Scripts\python.exe scripts/copy_gcs_to_r2.py --gcloud-auth
.venv\Scripts\python.exe scripts/copy_gcs_to_r2.py --gcloud-auth --copy
.venv\Scripts\python.exe scripts/render_deploy.py --publish
```

The branch must be pushed before Render can build it. `render_deploy.py` without flags prints a preview. The publish command explicitly selects `plan=free`, one instance, Singapore, the Render Dockerfile and disabled automatic deployment. It never falls back to a paid plan. If a same-named service exists, it prints the ID without changing it. A deployment receipt is saved under ignored `data/` without credentials.

Alternatively, create a Render Blueprint from the repository's `render.yaml`, select branch `migration/hf-r2`, and enter the prompted R2 settings. Use either the API or Blueprint creation path to avoid creating two services. The Blueprint uses the same Free plan and Dockerfile.

After the service has a URL, set R2's bucket CORS policy in the Cloudflare dashboard. Use `configs/r2_cors.json` with the placeholder replaced by the exact `https://...onrender.com` origin. The old `.hf.space` placeholder is illustrative only; use Render's actual URL. Keep public bucket access disabled.

## Resource choices

- One Uvicorn worker and one prediction request at a time. A simultaneous prediction receives HTTP 503 with an explanatory message; health checks and the website remain available.
- BLAS/OpenMP thread defaults are one, and malloc arenas are limited to two.
- Browser uploads go directly to R2; the API receives references and reads each stored file to predict.
- The existing 64 MiB per-file processing limit remains. Passing a test workload does not prove every 64 MiB input fits into 512 MB.
- This is a controlled showcase deployment, not a high-concurrency production service. Sessions can retain output data; persistent archives live in R2.

## Free-tier constraints

Render Free has 512 MB memory and a small CPU share. It sleeps after 15 minutes without inbound traffic; waking can take about a minute. Local files are temporary. There are 750 instance-hours per workspace per month, plus bandwidth and build-minute allowances. Without a payment method, exceeding relevant allowances results in suspension or disabled builds instead of overage billing; adding a payment method can enable overage charges. Keep the Free plan and review workspace spending settings.

Render may suspend a free service with unusually high external traffic, including transfers to object storage. Direct browser uploads reduce traffic through Render, but model inputs still need to be downloaded by the service. R2 has its own free allowances and can charge beyond them; the 8 GiB migration guard is not an ongoing spending cap.

Sources: [Render Free](https://render.com/docs/free), [Docker hosting](https://render.com/docs/docker), [Blueprint specification](https://render.com/docs/blueprint-spec), [API authentication](https://render.com/docs/api), [R2 pricing](https://developers.cloudflare.com/r2/pricing/).

## Files

| File | Responsibility |
|---|---|
| `deploy/render/Dockerfile` | Build the website and Python inference runtime for Render. |
| `render.yaml` | Reproducible Free service definition; secrets prompted separately. |
| `scripts/render_deploy.py` | Preview, check access, create a Free service via API. |
| `scripts/free_env.py` | Read local settings without logging credentials. |
| `nebulax/api/free_app.py` | Gate overlapping prediction requests to limit memory pressure. |
| `nebulax/api/r2_store.py` | Durable R2 storage and multipart uploads. |
| `scripts/copy_gcs_to_r2.py` | Verify and copy existing archives without deleting Google data. |
| `tests/test_free_app.py` | Test concurrency rejection, health availability and lock recovery. |

Live verification remains pending account credentials: deploy build, test actual R2 browser uploads/resume, copied results and all four model predictions, then check Google remains accessible.
