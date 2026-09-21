"""Create one explicitly free Render Docker service. Preview by default; no secret logging."""
import argparse
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

from free_env import load_env, require_env


def api(method, path, payload=None):
    request = urllib.request.Request("https://api.render.com/v1" + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": "Bearer " + os.environ["RENDER_API_KEY"], "Content-Type": "application/json"},
        method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Error bodies may echo submitted environment values; don't print them.
        raise SystemExit(f"Render API returned HTTP {exc.code}. Check account/repository access in the dashboard; no paid fallback attempted.") from None


def configuration(owner):
    return {
        "type": "web_service", "name": os.getenv("RENDER_SERVICE_NAME") or "nebulax-free",
        "ownerId": owner, "repo": os.getenv("RENDER_REPO_URL") or "https://github.com/bersamin12/nebulax",
        "branch": os.getenv("RENDER_BRANCH") or "migration/hf-r2", "autoDeploy": "no",
        "serviceDetails": {"runtime": "docker", "plan": "free", "region": "singapore", "numInstances": 1,
                           "healthCheckPath": "/api/health", "envSpecificDetails": {
                               "dockerfilePath": "./deploy/render/Dockerfile", "dockerContext": ".", "dockerCommand": ""}},
        "envVars": [{"key": "NEBULAX_STORAGE_PROVIDER", "value": "r2"}] +
            [{"key": key, "value": os.getenv(key, "")} for key in
             ("NEBULAX_RUN_BUCKET", "R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")],
    }


def main():
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = configuration(os.getenv("RENDER_OWNER_ID", ""))
    print("Render target:", config["name"], "plan=free, region=singapore, one instance")
    print("Source:", config["repo"], "branch:", config["branch"])
    if not args.publish and not args.check:
        print("Preview only. No remote resources changed.")
        return
    require_env("RENDER_API_KEY")
    owner = os.getenv("RENDER_OWNER_ID")
    if not owner:
        owners = [item["owner"] for item in api("GET", "/owners?limit=100")]
        if len(owners) != 1:
            print("Available workspace IDs:", ", ".join(item["id"] for item in owners))
            raise SystemExit("Set RENDER_OWNER_ID in .env to select the intended workspace.")
        owner = owners[0]["id"]
    config["ownerId"] = owner
    print("Workspace:", owner)
    existing = api("GET", "/services?" + urllib.parse.urlencode({"ownerId": owner, "name": config["name"], "limit": 100}))
    matches = [item["service"] for item in existing if item["service"]["name"] == config["name"]]
    if matches:
        print("Existing service:", matches[0]["id"])
        print("No changes made. Inspect the existing service before deploying an update.")
        return
    if args.check:
        print("Render account accessible. No resources changed.")
        return
    require_env("NEBULAX_RUN_BUCKET", "R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    result = api("POST", "/services", config)
    service = result["service"]
    receipt = {"id": service["id"], "deploy_id": result.get("deployId"),
               "url": service.get("serviceDetails", {}).get("url"), "plan": "free"}
    folder = Path(__file__).resolve().parents[1] / "data"
    folder.mkdir(exist_ok=True)
    (folder / "render-deployment.json").write_text(json.dumps(receipt, indent=2))
    print("Created free service:", receipt["id"])
    print("Dashboard: https://dashboard.render.com/web/" + receipt["id"])
    print("Build submitted; verify the live app after it finishes. Google resources were not changed.")


if __name__ == "__main__":
    main()
