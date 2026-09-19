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


def deploy_web(gcloud: str, project: str, region: str) -> str:
    tag = f"{region}-docker.pkg.dev/{project}/nebulax/app:dev"
    call(gcloud, "builds", "submit", "--region", region, "--tag", tag, ".")
    image = digest_ref(gcloud, tag)
    call(
        gcloud,
        "run", "deploy", "nebulax-app",
        "--project", project,
        "--region", region,
        "--image", image,
        "--service-account", f"nebulax-web@{project}.iam.gserviceaccount.com",
        "--no-allow-unauthenticated",
        "--max-instances", "1",
        "--min-instances", "0",
        "--memory", "2Gi",
        "--cpu", "2",
        "--concurrency", "10",
        "--timeout", "3600",
        "--port", "8080",
        "--quiet",
    )
    return image


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
    out.add_argument("target", choices=("web", "batch", "all"), nargs="?", default="all")
    out.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT))
    out.add_argument("--region", default=os.getenv("GOOGLE_CLOUD_REGION", DEFAULT_REGION))
    return out


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    gcloud = find_gcloud()
    call(gcloud, "config", "set", "project", args.project)
    images: dict[str, str] = {}
    if args.target in ("web", "all"):
        images["web"] = deploy_web(gcloud, args.project, args.region)
    if args.target in ("batch", "all"):
        images["batch"] = deploy_batch(gcloud, args.project, args.region)
    for name, image in images.items():
        print(f"{name}: {image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
