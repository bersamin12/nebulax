# Google Cloud guide

This is the main guide for using, observing, rebuilding, and extending the NebulaX Google Cloud deployment. The current deployment is a **private development deployment** in Qwiklabs project `qwiklabs-gcp-02-ebc381898f1f`, region `us-central1`. It is not yet a public submission URL. The lab project may be temporary; confirm the competition's submission project and access requirements before relying on it for judging.

## Contents

- [Choose how to run the app](#choose-how-to-run-the-app)
- [Use the cloud app now](#use-the-cloud-app-now)
- [Check health, logs, and costs](#check-health-logs-and-costs)
- [Overnight usage and pausing](#overnight-usage-and-pausing)
- [Connect the CLI](#connect-the-cli)
- [Current deployed resources and verification](#current-deployed-resources-and-verification)
- [Build and redeploy the web app](#build-and-redeploy-the-web-app)
- [Run a Cloud Storage prediction batch](#run-a-cloud-storage-prediction-batch)
- [Next cloud work](#next-cloud-work)

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
$gcloud = Join-Path $env:LOCALAPPDATA 'Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd'
& $gcloud run services proxy nebulax-app --region us-central1 --project qwiklabs-gcp-02-ebc381898f1f --port 9090
```

Keep that terminal open and visit `http://127.0.0.1:9090/`. On **Predict**, choose Door, ACV, Rail, or SHM; upload compatible input files; press **RUN**; inspect the explanations; and download the prediction CSV. The four saved models are packaged in the deployed image. Each interactive file is limited to 64 MiB, and the browser sends at most 32 files per request. The app stores these uploads, results, and session state on its Cloud Run instance. A restart can lose them, so download results promptly.

On macOS, after [installing and authenticating `gcloud`](#macos-setup), run `gcloud run services proxy nebulax-app --region us-central1 --project qwiklabs-gcp-02-ebc381898f1f --port 9090` and open the same local URL. Each developer needs access to the private service under their own Google account.

The separate `nebulax-ps3-batch` Cloud Run Job reads larger input folders from Cloud Storage and writes a CSV back to Cloud Storage. Use the [batch instructions](#run-a-cloud-storage-prediction-batch) below. The browser does not yet start that job or show its status.

The deployed app **does not train models or accept model uploads**. To update a model, change the artifact in `models/ps3/`, rebuild the image, and redeploy it. The repository now contains newer Rail and SHM artifacts than the currently running Cloud Run image; the cloud app will use those updates only after redeployment. Cloud training and model version management are planned work. The existing deployment has no Fleet replay data, so its `/api/health` reports `scores_loaded: false`. A newly built image from this checkout includes `demo_data/` and will report a two-train bearing and brake replay after deployment.

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
& $gcloud run services update nebulax-app --region us-central1 --scaling=0
```

Restore it before testing or judging:

```powershell
& $gcloud run services update nebulax-app --region us-central1 --scaling=auto
```

Manual scaling to zero makes requests fail, including judge visits. It does not remove Storage, Artifact Registry, or other resource charges. See [Cloud Run manual scaling](https://docs.cloud.google.com/run/docs/configuring/services/manual-scaling). The current private service is not yet accessible to judges without IAM permission or a proxy.

## Connect the CLI

### Windows PowerShell setup

Install the [Google Cloud CLI for Windows](https://docs.cloud.google.com/sdk/docs/install-sdk). On this PC it is installed at `%LOCALAPPDATA%\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd`. If `gcloud` is not on `PATH`, set `$gcloud` as above and use `& $gcloud ...` for every command. Alternatively, add that `bin` directory to your user Path and reopen PowerShell.

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
| Private Cloud Run web service | `nebulax-app`, revision `nebulax-app-00001-msn`, 2 CPU, 2 GiB, at most 1 instance |
| Web service account | `nebulax-web@qwiklabs-gcp-02-ebc381898f1f.iam.gserviceaccount.com` |
| Web image | `app@sha256:e32056b4540cdd8d04720fc58d171a9dd397520edcfb868b89342f46ef9c7146` |
| Cloud Run batch job | `nebulax-ps3-batch`, 1 task, 2 CPU, 4 GiB, 20-minute timeout, no retry |
| Batch service account | `nebulax-batch@qwiklabs-gcp-02-ebc381898f1f.iam.gserviceaccount.com` |
| Batch image | `ps3-batch@sha256:e1bcd208e8e099371487bb5e81ff3a31f80612a176e029e0b85565cddeab7a87` |
| Storage bucket | `gs://nebulax-ps3-qwiklabs-gcp-02-ebc381898f1f/` |

The web service is private under Cloud Run IAM; the lab user has `roles/run.invoker`. Its authenticated `/api/health` and `/api/ps3/tasks` endpoints returned HTTP 200, and all four PS3 models were loaded. Uploading a **generated** SHM file and downloading its result succeeded. The output SHA-256 matched local inference using the earlier model version: `A7FB06B24DFEEC13234FECBDFAB9996164A767350A0BF821E8ED88F5E4FD5DBB`. The repository's Rail and SHM models have since changed; that hash does not describe the updated artifacts.

Batch execution `nebulax-ps3-batch-8vjc7` completed successfully on a generated SHM signal. Its output at `outputs/shm-smoke-001/shm_predictions.csv` had the same hash as local inference. The smoke input is at `inputs/shm-smoke-001/synthetic_shm.csv`.

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

## Next cloud work

| Stage | Current state and next work |
|---|---|
| 1. Web deployment | Private Cloud Run service is built and smoke tested. For judging, move to a durable competition project and provide reviewer access or a suitable public entry point. |
| 2. Durable prediction workflow | Cloud Storage bucket and batch job are deployed. Add scoped direct browser uploads, API job launch/status/download controls, shared run metadata, and Fleet generation jobs. |
| 3. Training and evaluation | Add a separate training image and [Vertex AI custom training](https://docs.cloud.google.com/vertex-ai/docs/training/overview), [Pipelines](https://docs.cloud.google.com/vertex-ai/docs/pipelines/introduction), Experiments, and Model Registry. Version data, splits, configs, and artifacts; use held-out release checks. |
| 4. Shared state and telemetry | Move PS3 sessions, uploads, results, and Fleet overlays to durable storage such as Cloud Storage and Firestore before increasing web instances. For real telemetry, consider Pub/Sub, Dataflow or Cloud Run, and BigQuery. |
| 5. Operations | Supply optional API keys through Secret Manager; add error, latency, job failure, and cost alerts. Add CI checks for image build, health, model loading, and representative uploads. |

The planned direct browser upload flow needs scoped upload URLs and bucket CORS for the web origin. The API should start the job with per-run environment overrides, store the run ID, object paths, status, and owner in shared state, and expose a validated result download. See [Cloud Run job overrides](https://docs.cloud.google.com/sdk/gcloud/reference/run/jobs/execute) and [Cloud Storage upload options](https://docs.cloud.google.com/storage/docs/uploads). Cloud Run WebSockets have request timeouts, so Fleet replay also needs reconnect/resume behavior before multi-instance scaling. See [Cloud Run WebSockets](https://docs.cloud.google.com/run/docs/triggering/websockets).
