#!/usr/bin/env python3
"""Build and deploy NebulaX Cloud Run resources with immutable image digests.

The script uses the already authenticated Google Cloud CLI. It stores no credentials and
defaults to the current lab resources documented in ``docs/google_cloud.md``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = "qwiklabs-gcp-02-ebc381898f1f"
DEFAULT_REGION = "us-central1"


def find_gcloud() -> str:
    """Return the CLI executable, including the standard per-user Windows location."""
    found = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidate = Path(local) / "Google" / "Cloud SDK" / "google-cloud-sdk" / "bin" / "gcloud.cmd"
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("gcloud was not found; install Google Cloud CLI and open a new terminal")


def call(gcloud: str, *args: str, capture: bool = False) -> str:
    command = [gcloud, *args]
    print("+", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return (result.stdout or "").strip()


def digest_ref(gcloud: str, tag: str) -> str:
    digest = call(
        gcloud,
        "artifacts", "docker", "images", "describe", tag,
        "--format=value(image_summary.digest)",
        capture=True,
    )
    if not digest.startswith("sha256:"):
        raise RuntimeError(f"Artifact Registry returned an invalid digest for {tag!r}: {digest!r}")
    return f"{tag.rsplit(':', 1)[0]}@{digest}"


def deploy_web(
    gcloud: str,
    project: str,
    region: str,
    *,
    public: bool = True,
    autoscale: bool = True,
) -> tuple[str, str]:
    tag = f"{region}-docker.pkg.dev/{project}/nebulax/app:dev"
    call(gcloud, "builds", "submit", "--region", region, "--tag", tag, ".")
    image = digest_ref(gcloud, tag)
    access_flag = "--no-invoker-iam-check" if public else "--invoker-iam-check"
    scaling = ["--scaling", "auto", "--max-instances", "1", "--min-instances", "0"] if autoscale else [
        "--scaling", "0"
    ]
    call(
        gcloud,
        "run", "deploy", "nebulax-app",
        "--project", project,
        "--region", region,
        "--image", image,
        "--service-account", f"nebulax-web@{project}.iam.gserviceaccount.com",
        access_flag,
        *scaling,
        "--memory", "2Gi",
        "--cpu", "2",
        "--concurrency", "10",
        "--timeout", "3600",
        "--port", "8080",
        "--quiet",
    )
    url = call(
        gcloud,
        "run", "services", "describe", "nebulax-app",
        "--project", project,
        "--region", region,
        "--format=value(status.url)",
        capture=True,
    )
    if not url.startswith("https://"):
        raise RuntimeError(f"Cloud Run returned an invalid service URL: {url!r}")
    return image, url


def deploy_batch(gcloud: str, project: str, region: str) -> str:
    tag = f"{region}-docker.pkg.dev/{project}/nebulax/ps3-batch:dev"
    call(
        gcloud,
        "builds", "submit",
        "--project", project,
        "--region", region,
        "--config", "cloudbuild.batch.yaml",
        "--substitutions", f"_IMAGE={tag}",
        ".",
    )
    image = digest_ref(gcloud, tag)
    call(
        gcloud,
        "run", "jobs", "update", "nebulax-ps3-batch",
        "--project", project,
        "--region", region,
        "--image", image,
        "--quiet",
    )
    return image


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("target", choices=("web", "batch", "all"), nargs="?", help="backward-compatible target")
    targets = out.add_mutually_exclusive_group()
    targets.add_argument("--all", dest="target_flag", action="store_const", const="all", help="deploy web and batch")
    targets.add_argument("--web", dest="target_flag", action="store_const", const="web", help="deploy only the web service")
    targets.add_argument("--batch", dest="target_flag", action="store_const", const="batch", help="deploy only the batch job")
    visibility = out.add_mutually_exclusive_group()
    visibility.add_argument("--public", dest="public", action="store_true", help="allow public web access (default)")
    visibility.add_argument("--private", dest="public", action="store_false", help="require IAM authentication")
    scaling = out.add_mutually_exclusive_group()
    scaling.add_argument("--autoscale", dest="autoscale", action="store_true", help="enable scale-to-zero autoscaling (default)")
    scaling.add_argument(
        "--pause", "--no-autoscale", dest="autoscale", action="store_false",
        help="deploy the web service at zero manual instances",
    )
    out.set_defaults(public=True, autoscale=True, target_flag=None)
    out.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT))
    out.add_argument("--region", default=os.getenv("GOOGLE_CLOUD_REGION", DEFAULT_REGION))
    return out


def main(argv: list[str] | None = None) -> int:
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    if args.target and args.target_flag:
        arg_parser.error("choose either a positional target or --all/--web/--batch, not both")
    target = args.target_flag or args.target or "all"
    gcloud = find_gcloud()
    call(gcloud, "config", "set", "project", args.project)
    images: dict[str, str] = {}
    web_url: str | None = None
    if target in ("web", "all"):
        images["web"], web_url = deploy_web(
            gcloud, args.project, args.region, public=args.public, autoscale=args.autoscale,
        )
    if target in ("batch", "all"):
        images["batch"] = deploy_batch(gcloud, args.project, args.region)
    for name, image in images.items():
        print(f"{name}: {image}")
    if web_url:
        state = "public" if args.public else "private"
        scaling_state = "autoscale" if args.autoscale else "paused"
        print(f"web_url: {web_url} ({state}, {scaling_state})")
    if target in ("batch", "all"):
        print(
            "batch_console: "
            f"https://console.cloud.google.com/run/jobs/details/{args.region}/"
            f"nebulax-ps3-batch/executions?project={args.project}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
