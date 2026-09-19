# Google Cloud guide

This is the main guide for using, observing, rebuilding, and extending the NebulaX Google Cloud deployment. The current deployment is a **private development deployment** in Qwiklabs project `qwiklabs-gcp-02-ebc381898f1f`, region `us-central1`. It is not yet a public submission URL. The lab project may be temporary; confirm the competition's submission project and access requirements before relying on it for judging.

## Contents

- [Choose how to run the app](#choose-how-to-run-the-app)
- [Use the cloud app now](#use-the-cloud-app-now)
- [Publish and retrieve the submission URL](#publish-and-retrieve-the-submission-url)
- [Check health, logs, and costs](#check-health-logs-and-costs)
- [Overnight usage and pausing](#overnight-usage-and-pausing)
- [Connect the CLI](#connect-the-cli)
- [Current deployed resources and verification](#current-deployed-resources-and-verification)
- [Build and redeploy the web app](#build-and-redeploy-the-web-app)
- [Run a Cloud Storage prediction batch](#run-a-cloud-storage-prediction-batch)
- [Implementation progress](#implementation-progress)
- [Remaining plan details](#remaining-plan-details)

## Choose how to run the app

| Command | Where the app runs | Open |
|---|---|---|
| `python scripts/start_app.py` | Local Python development environment | `http://127.0.0.1:8765/` |
| `docker compose up --build` | Local Docker container | `http://127.0.0.1:8765/` |
| `gcloud run services proxy ...` | Existing Google Cloud Run deployment, viewed through a local authenticated proxy | `http://127.0.0.1:9090/` |

The proxy does not start, build, or redeploy the app. It forwards your browser requests to the private Cloud Run service. The Google Cloud Console is a separate site for managing resources. The local Docker and Cloud Run copies have separate files, sessions, and deployments. Docker remains useful for local checks and for building the image that Cloud Run executes.

## Use the cloud app now

Connect `gcloud` as described in [Connect the CLI](#connect-the-cli), then run this in PowerShell:

```powershell
$gcloud = (Get-ChildItem "$env:LOCALAPPDATA\Google\Cloud SDK" -Filter gcloud.cmd -Recurse | Select-Object -First 1).FullName
& $gcloud run services proxy nebulax-app --region us-central1 --project qwiklabs-gcp-02-ebc381898f1f --port 9090
```

Keep that terminal open and visit `http://127.0.0.1:9090/`. On **Predict**, choose Door, ACV, Rail, or SHM; upload compatible input files; press **RUN**; inspect the explanations; and download the prediction CSV. The four saved models are packaged in the deployed image. Each interactive file is limited to 64 MiB, and the browser sends at most 32 files per request. The app stores these uploads, results, and session state on its Cloud Run instance. A restart can lose them, so download results promptly.

On macOS, after [installing and authenticating `gcloud`](#macos-setup), run `gcloud run services proxy nebulax-app --region us-central1 --project qwiklabs-gcp-02-ebc381898f1f --port 9090` and open the same local URL. Each developer needs access to the private service under their own Google account.

The separate `nebulax-ps3-batch` Cloud Run Job reads larger input folders from Cloud Storage and writes a CSV back to Cloud Storage. Use the [batch instructions](#run-a-cloud-storage-prediction-batch) below. The browser does not yet start that job or show its status.

The deployed app **does not train models or accept model uploads**. To update a model, change the artifact in `models/ps3/`, rebuild the image, and redeploy it. The current Cloud Run revision contains the latest checked-in Rail and SHM artifacts plus `demo_data/`; `/api/health` reports a two-train bearing and brake replay. Cloud training and model version management are planned work.

## Publish and retrieve the submission URL

Deploy the current checkout first:

```powershell
python scripts/gcp_deploy.py web
```

The deployment script keeps the service private. When the final deployment is ready for judging, allow unauthenticated access and retrieve its Cloud Run URL:

```powershell
& $gcloud run services update nebulax-app --project qwiklabs-gcp-02-ebc381898f1f --region us-central1 --no-invoker-iam-check
& $gcloud run services describe nebulax-app --project qwiklabs-gcp-02-ebc381898f1f --region us-central1 --format="value(status.url)"
```

Submit the returned HTTPS `run.app` URL. Do not submit `127.0.0.1:8765` or `127.0.0.1:9090`; those addresses work only on the developer's computer. Test the returned URL in an incognito window without signing in. Also confirm that this Qwiklabs project remains active throughout judging; otherwise repeat the deployment in the durable competition project and submit that service's URL. To return the service to private access later, run `& $gcloud run services update nebulax-app --project qwiklabs-gcp-02-ebc381898f1f --region us-central1 --invoker-iam-check`.

## Check health, logs, and costs

While the proxy is running:

| Check | Where | What to look for |
|---|---|---|
| App health | `http://127.0.0.1:9090/api/health` | `status: "ok"`; `scores_loaded` describes Fleet data only |
| PS3 models | `http://127.0.0.1:9090/api/ps3/tasks` | `model_loaded: true` and `available: true` for Door, ACV, Rail, SHM |
| Service logs and request/instance metrics | [Cloud Run Console](https://console.cloud.google.com/run?project=qwiklabs-gcp-02-ebc381898f1f) → Services → `nebulax-app` → Logs or Metrics | Request errors, latency, instance activity |
| Batch execution and logs | Cloud Run Console → Jobs → `nebulax-ps3-batch` → Executions / Logs | Execution status and worker output |
| Batch objects | [Cloud Storage bucket](https://console.cloud.google.com/storage/browser/nebulax-ps3-qwiklabs-gcp-02-ebc381898f1f?project=qwiklabs-gcp-02-ebc381898f1f) | `inputs/` and `outputs/` folders |
| Spend | Console → Billing → Reports; filter project `qwiklabs-gcp-02-ebc381898f1f`, group by service | Cloud Run, Cloud Build, Storage, Artifact Registry, Logging |

Billing is enabled on the lab project, but the lab user may lack permission to view its reports. Cost reports can lag behind usage. If permitted, create an alert in Billing → Budgets & alerts. An **alert-only budget does not stop spending**. See [Cloud Run monitoring](https://docs.cloud.google.com/run/docs/monitoring), [logs](https://docs.cloud.google.com/run/docs/logging), [billing reports](https://docs.cloud.google.com/billing/docs/how-to/reports), and [budgets](https://docs.cloud.google.com/billing/docs/how-to/budgets).

For an authenticated API check without the browser proxy:

```powershell
$token = & $gcloud auth print-identity-token
$url = 'https://nebulax-app-673528097036.us-central1.run.app'
Invoke-RestMethod "$url/api/health" -Headers @{ Authorization = "Bearer $token" }
Invoke-RestMethod "$url/api/ps3/tasks" -Headers @{ Authorization = "Bearer $token" }
```

### Overnight usage and pausing

The web service is configured for **zero minimum instances** and **one maximum instance**. With no requests, Cloud Run normally scales it to zero. The batch job has no schedule and runs only when executed. Closing the proxy or browser does not delete cloud resources. Stored Cloud Storage objects and Artifact Registry images can still incur charges. [Cloud Run autoscaling](https://docs.cloud.google.com/run/docs/about-instance-autoscaling) explains idle behavior.

There is normally no need to pause the service overnight. To make it unavailable deliberately:

```powershell
& $gcloud run services update nebulax-app --project qwiklabs-gcp-02-ebc381898f1f --region us-central1 --scaling=0
```

Restore it before testing or judging:

```powershell
& $gcloud run services update nebulax-app --project qwiklabs-gcp-02-ebc381898f1f --region us-central1 --scaling=auto
```

Manual scaling to zero makes requests fail, including judge visits. It does not remove Storage, Artifact Registry, or other resource charges. See [Cloud Run manual scaling](https://docs.cloud.google.com/run/docs/configuring/services/manual-scaling). The current private service is not yet accessible to judges without IAM permission or a proxy.

## Connect the CLI

### Windows PowerShell setup

Install the [Google Cloud CLI for Windows](https://docs.cloud.google.com/sdk/docs/install-sdk). On this PC it is under `%LOCALAPPDATA%\Google\Cloud SDK`. Let PowerShell find the exact executable so spaces or copied line wrapping cannot corrupt the path:

```powershell
$gcloud = (Get-ChildItem "$env:LOCALAPPDATA\Google\Cloud SDK" -Filter gcloud.cmd -Recurse | Select-Object -First 1).FullName
$gcloud
& $gcloud --version
```

The printed value should end in `google-cloud-sdk\bin\gcloud.cmd`. PowerShell requires the call operator `&` when a command is stored in a variable: use `& $gcloud --version`, not `$gcloud --version`. If the search returns nothing, install the CLI or reopen PowerShell after installation. Alternatively, add its `bin` directory to your user Path and use `gcloud` directly.

```powershell
& $gcloud auth login --no-launch-browser
& $gcloud config set project qwiklabs-gcp-02-ebc381898f1f
& $gcloud config set run/region us-central1
& $gcloud auth list
& $gcloud config list
```

Open the login URL in an incognito browser and sign in with the lab account. Do not put its username/password into `.env`, Docker, Git, or a service-account key file. The project ID and region are configuration, not secrets. CLI login does not deploy anything.

### macOS setup

Each Mac developer installs the CLI on their own computer. If they already use Homebrew, Google's [Homebrew installation guide](https://docs.cloud.google.com/sdk/docs/downloads-homebrew) gives this command; otherwise use the [official macOS installer](https://docs.cloud.google.com/sdk/docs/install-sdk) for Apple silicon or Intel, as appropriate:

```bash
brew update && brew install --cask gcloud-cli
gcloud --version
```

In Terminal or another shell, authenticate with the Google account that has been granted access to the project, then select the same project and region:

```bash
gcloud auth login --no-launch-browser
gcloud config set project qwiklabs-gcp-02-ebc381898f1f
gcloud config set run/region us-central1
gcloud auth list
gcloud config list
gcloud components install cloud-run-proxy
gcloud run services proxy nebulax-app --region us-central1 --project qwiklabs-gcp-02-ebc381898f1f --port 9090
```

Keep the proxy terminal open and visit `http://127.0.0.1:9090/`. If `gcloud` is not found after installation, open a new Terminal and follow the installer's shell setup instructions. The rest of this guide's `& $gcloud` examples are PowerShell syntax; on macOS use `gcloud` directly. Shell variables also differ: use `PROJECT_ID=...` and `$PROJECT_ID` rather than PowerShell's `$project = ...`.

The current private service grants `roles/run.invoker` to the original lab user only. A Mac developer signing in with a different account will receive an authorization error until a project administrator grants that account Cloud Run Invoker on `nebulax-app`. Grant each developer their own access; do not share the lab password. For example, an administrator can run:

```bash
gcloud run services add-iam-policy-binding nebulax-app --region us-central1 --project qwiklabs-gcp-02-ebc381898f1f --member='user:developer@example.com' --role='roles/run.invoker'
```

Replace the example email with the developer's actual Google account. `roles/run.invoker` allows access to the app; building images, deploying, running jobs, or viewing billing require additional project permissions. See [Cloud Run access control](https://docs.cloud.google.com/run/docs/securing/managing-access). A Qwiklabs lab may restrict adding external users, so use the competition's shared project if the lab cannot grant their accounts access.

For **local Python code** that calls Google APIs, separately run `& $gcloud auth application-default login --no-launch-browser` on Windows or `gcloud auth application-default login --no-launch-browser` on macOS. The deployed batch worker automatically uses its assigned Cloud Run service account; it needs no local credential file. See [Application Default Credentials](https://docs.cloud.google.com/docs/authentication/application-default-credentials). The app's `.env` is for optional app settings, such as `ANTHROPIC_API_KEY`, not lab credentials.

The `cloud-run-proxy` component is installed on this PC. On another PC, install it with `gcloud components install cloud-run-proxy`. If the Windows installer reports a bundled Python update error, set `$env:CLOUDSDK_PYTHON = (Get-Command python).Source` for that PowerShell session and retry.

Cloud Shell is another option: it has `gcloud` and the console's signed-in identity. It cannot see this PC's uncommitted files; publish or transfer the checkout before building there. See [Cloud Shell](https://docs.cloud.google.com/shell/docs/launching-cloud-shell).

## Current deployed resources and verification

| Resource | Value |
|---|---|
| Project / region | `qwiklabs-gcp-02-ebc381898f1f` / `us-central1` |
| Artifact Registry repository | `us-central1-docker.pkg.dev/qwiklabs-gcp-02-ebc381898f1f/nebulax` |
| Private Cloud Run web service | `nebulax-app`, revision `nebulax-app-00003-kgm`, 2 CPU, 2 GiB, at most 1 instance |
| Web service account | `nebulax-web@qwiklabs-gcp-02-ebc381898f1f.iam.gserviceaccount.com` |
| Web image | `app@sha256:ea307967917c0d307bba864cde8f0fe5ddfd8990da05b53cd49120c34c9f2b94` |
| Cloud Run batch job | `nebulax-ps3-batch`, 1 task, 2 CPU, 4 GiB, 20-minute timeout, no retry |
| Batch service account | `nebulax-batch@qwiklabs-gcp-02-ebc381898f1f.iam.gserviceaccount.com` |
| Batch image | `ps3-batch@sha256:cf8b5fc1b1d21db9d224c2d272b8c0404da2b0d53951d2f1b0d0dfc4a6f671c7` |
| Storage bucket | `gs://nebulax-ps3-qwiklabs-gcp-02-ebc381898f1f/` |

The web service is private under Cloud Run IAM; the lab user has `roles/run.invoker`. Revision `nebulax-app-00003-kgm` became current after a concurrent deployment on 19 September 2026. Its authenticated `/api/health` returned HTTP 200 with `scores_loaded: true` and two Fleet demo trains; `/api/ps3/tasks` returned all four models as loaded and available. Revision `00002-bmr`, built by `scripts/gcp_deploy.py` from this branch immediately beforehand, also exposed the current Rail nested score (0.8051) and SHM nested score (0.9813). A generated SHM upload against the current service produced a CSV byte-identical to local inference: SHA-256 `A7FB06B24DFEEC13234FECBDFAB9996164A767350A0BF821E8ED88F5E4FD5DBB`.

Batch execution `nebulax-ps3-batch-2r8tk` completed successfully with the current batch image on the same generated SHM signal. Its output at `outputs/shm-smoke-002/shm_predictions.csv` was byte-identical to current local inference. The reusable generated input is at `inputs/shm-smoke-001/synthetic_shm.csv`. The earlier execution and output remain as historical smoke evidence.

The bucket has uniform bucket-level access, enforced public access prevention, and lifecycle rules in `configs/cloud_bucket_lifecycle.json`: delete `inputs/` objects after 7 days and `outputs/` objects after 30 days. The batch service account has object viewer and creator roles on this bucket. The app service account is separate.

## Build and redeploy the web app

`Dockerfile` builds the React frontend with Node and runs FastAPI with Python 3.11 on Cloud Run's `PORT`. It packages the four saved `models/ps3` predictors, PS3 CV results, and Fleet leaderboard. It does **not** package raw organiser datasets, generated Fleet twin data, secrets, or training outputs. One Uvicorn process and at most one Cloud Run instance are deliberate while upload sessions and Fleet overlays live in memory. `.gcloudignore` and `.dockerignore` exclude local environments, raw data, secrets, and generated builds.

From the repository root, after changing source, frontend, dependencies, or models:

```powershell
$project = 'qwiklabs-gcp-02-ebc381898f1f'
$region = 'us-central1'
$tag = "${region}-docker.pkg.dev/$project/nebulax/app:dev"
& $gcloud builds submit --region $region --tag $tag
$digest = & $gcloud artifacts docker images describe $tag --format='value(image_summary.digest)'
& $gcloud run deploy nebulax-app --region $region --image "${region}-docker.pkg.dev/$project/nebulax/app@$digest" --service-account "nebulax-web@$project.iam.gserviceaccount.com" --no-allow-unauthenticated --max-instances 1 --min-instances 0 --memory 2Gi --cpu 2 --concurrency 10 --timeout 3600 --port 8080
```

This lab's repository, service account, and service already exist. In a new project, create those resources and grant only needed IAM roles before running the deployment. Do not make the development deployment public merely to bypass the proxy. For a judge-facing deployment, decide its authentication and stable project/URL explicitly.

For local container checks, run `docker compose up --build` and open `http://127.0.0.1:8765/`. For an optional local Fleet twin, first create `data/sim` and `data/scores` using [the app run guide](run_app.md), then mount `data/` into a local container:

```powershell
docker build -t nebulax:local .
docker run --rm -p 127.0.0.1:8765:8080 -v "${PWD}/data:/app/data" nebulax:local
```

The container needs write access to the mount for upload caches and fault injection. On Linux, grant UID 10001 access. Keep credentials and organiser data out of images.

## Run a Cloud Storage prediction batch

The worker in `scripts/cloud_batch_predict.py`, packaged by `Dockerfile.batch`, reads `gs://` input objects for one task. It downloads and predicts one file at a time using the same saved model and CSV writer as the web app, validates the complete organiser CSV, and uploads only after success. A failed input leaves no partial output; retries never overwrite an existing output object. The path supports Door, ACV, Rail, and SHM and is useful for larger Rail and SHM folders.

The job and bucket already exist in this lab. Set variables for repeatable commands:

```powershell
$project = 'qwiklabs-gcp-02-ebc381898f1f'
$region = 'us-central1'
$bucket = 'nebulax-ps3-qwiklabs-gcp-02-ebc381898f1f'
$image = "${region}-docker.pkg.dev/$project/nebulax/ps3-batch:dev"
```

To rebuild the batch image after changing the worker or models, and update the existing job:

```powershell
& $gcloud builds submit --region $region --config cloudbuild.batch.yaml --substitutions "_IMAGE=$image"
$batchDigest = & $gcloud artifacts docker images describe $image --format='value(image_summary.digest)'
& $gcloud run jobs update nebulax-ps3-batch --region $region --image "${region}-docker.pkg.dev/$project/nebulax/ps3-batch@$batchDigest"
```

For a new batch, choose a unique run ID and upload only data you are authorized to store in this project. This example uses Rail:

```powershell
& $gcloud storage cp --recursive path/to/Rail_Corrugation/Test "gs://$bucket/inputs/rail-run-001/"
& $gcloud run jobs execute nebulax-ps3-batch --region $region --wait --update-env-vars "PS3_TASK=rail,PS3_INPUT_PREFIX=gs://$bucket/inputs/rail-run-001/,PS3_OUTPUT_URI=gs://$bucket/outputs/rail-run-001/rail_predictions.csv"
& $gcloud storage cp "gs://$bucket/outputs/rail-run-001/rail_predictions.csv" .
```

Inputs must be under a prefix ending in `/`. Use `.csv` for Door, Rail, and SHM, or `.xlsx`/`.xls` for ACV. The job ignores unrelated and hidden files and Excel temporary files. Duplicate basenames in nested folders fail before prediction. Keep output outside the input prefix and choose a new output URI per run. The worker logs task, input count, row count, and output URI. The bucket lifecycle deletes inputs and outputs on the schedules above.

For local testing against a bucket, install `requirements-cloud-batch.txt`, run the separate ADC login, and set `PS3_TASK`, `PS3_INPUT_PREFIX`, and `PS3_OUTPUT_URI` before `python scripts/cloud_batch_predict.py`. The deployed job uses its own service account, with object read and create permissions. See [Cloud Run Jobs](https://docs.cloud.google.com/run/docs/configuring/jobs/containers) and [Cloud Storage uploads](https://docs.cloud.google.com/storage/docs/uploading-objects).

Earlier Cloud Build source archives included small organiser-derived test fixtures before `.gcloudignore` was updated to exclude `tests/`. Those two staging archives were removed; the staging bucket's 7-day soft-delete window may retain recoverable copies temporarily. The runtime batch image did not copy the fixtures. `.gcloudignore` now excludes `tests/`, local environments, raw data, and `.env`.

## Implementation progress

Status legend: `[x]` complete and verified, `[-]` partly implemented, `[ ]` not started, `[!]` requires an owner or competition decision.

| Stage | Progress | Checklist | Next action |
|---|---:|---|---|
| 1. Web deployment | 85% | `[x]` Docker image; `[x]` Artifact Registry; `[x]` private Cloud Run service; `[x]` authenticated health/model/upload smoke test; `[x]` repeatable digest-pinned deployment script; `[x]` current Rail/SHM models and bundled Fleet demo deployed; `[!]` stable judge project and access | Decide the judge authentication and final project, then repeat the scripted deployment there. |
| 2. Durable prediction workflow | 50% | `[x]` private bucket and lifecycle; `[x]` separate batch image; `[x]` Cloud Run Job; `[x]` current model image deployed; `[x]` generated SHM cloud/local parity test; `[x]` repeatable batch deployment script; `[ ]` browser-to-Storage upload; `[ ]` API job launch/status/download; `[ ]` shared run metadata; `[ ]` Fleet generation job | Design authenticated run ownership, then add Firestore run records and scoped upload URLs before changing the browser. |
| 3. Training and evaluation | 10% | `[x]` existing local train/evaluate commands and saved evidence; `[ ]` training image; `[ ]` Vertex AI Custom Job; `[ ]` Pipeline; `[ ]` Experiments; `[ ]` Model Registry promotion | Define one reproducible task training entry point and artifact contract, starting with the smallest task. |
| 4. Shared state and telemetry | 5% | `[-]` Cloud Storage handles batch objects; `[ ]` durable interactive sessions; `[ ]` Firestore job/owner state; `[ ]` Pub/Sub ingestion; `[ ]` queryable telemetry store; `[ ]` replay reconnect/resume | Implement the run metadata schema before allowing more than one web instance. |
| 5. Operations | 20% | `[x]` Cloud Logging/Monitoring available; `[x]` scale-to-zero and one-instance cost limits; `[ ]` Secret Manager wiring; `[ ]` uptime/error/job alerts; `[ ]` budget alert; `[-]` build and focused local tests; `[ ]` automated cloud smoke gate | Add alerts and a CI smoke gate after the final project and billing permissions are known. |

The cross-platform deployment entry point is:

```powershell
python scripts/gcp_deploy.py web    # build and deploy the browser/API image
python scripts/gcp_deploy.py batch  # build and update the prediction job
python scripts/gcp_deploy.py all    # do both sequentially
```

It uses the authenticated `gcloud` CLI, resolves each pushed tag to an immutable digest, and deploys that digest. On macOS and Linux the same Python commands work when `gcloud` is on `PATH`. Project and region can be overridden with `--project` and `--region`.

## Remaining plan details

The planned direct browser upload flow needs scoped upload URLs and bucket CORS for the web origin. The API should start the job with per-run environment overrides, store the run ID, object paths, status, and owner in shared state, and expose a validated result download. See [Cloud Run job overrides](https://docs.cloud.google.com/sdk/gcloud/reference/run/jobs/execute) and [Cloud Storage upload options](https://docs.cloud.google.com/storage/docs/uploads). Cloud Run WebSockets have request timeouts, so Fleet replay also needs reconnect/resume behavior before multi-instance scaling. See [Cloud Run WebSockets](https://docs.cloud.google.com/run/docs/triggering/websockets).
