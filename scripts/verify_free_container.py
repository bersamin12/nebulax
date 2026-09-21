"""Exercise a local Render-like container. Synthetic Rail/SHM size probes, not accuracy benchmarks."""
import json
from pathlib import Path
import subprocess
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8769"
CONTAINER = "nebulax-render-check"


def memory():
    code = "from pathlib import Path;import json;print(json.dumps({k:int(Path('/sys/fs/cgroup/'+k).read_text()) for k in ['memory.current','memory.peak']}))"
    return json.loads(subprocess.check_output(["docker", "exec", CONTAINER, "python", "-c", code], text=True))


def main():
    client = requests.Session()
    client.trust_env = False
    deadline = time.monotonic() + 180
    while True:
        try:
            client.get(BASE + "/api/health", timeout=5).raise_for_status()
            break
        except requests.RequestException:
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)
    results = {"limits": {"memory_mib": 512, "cpu": 0.1}, "idle": memory(), "models": []}
    fixtures = ROOT / "tests/fixtures/ps3"
    rail = (fixtures / "rail/train1_slice.csv").read_text().splitlines()
    shm = (fixtures / "shm/train01_slice.csv").read_text().splitlines()
    cases = [
        ("door", "door.csv", (fixtures / "door/train_slice.csv").read_bytes(), "3 recorded cycles"),
        ("acv", "acv.xlsx", (fixtures / "acv/case01_slice.xlsx").read_bytes(), "30-row recorded excerpt"),
        ("rail", "rail.csv", (rail[0] + "\n" + "\n".join(rail[1:] * 100) + "\n").encode(), "10000 rows repeated from fixture"),
        ("shm", "shm.csv", ("\n".join(shm[i % len(shm)] for i in range(581120)) + "\n").encode(), "581120 samples repeated from fixture"),
    ]
    for task, filename, data, description in cases:
        start = time.monotonic()
        response = client.post(BASE + f"/api/ps3/{task}/predict", files={"files": (filename, data)}, timeout=600)
        response.raise_for_status()
        payload = response.json()
        assert payload["rows"], f"No predictions for {task}"
        result = {"task": task, "input": description, "input_bytes": len(data), "rows": len(payload["rows"]),
                  "wall_seconds": round(time.monotonic() - start, 2), **memory()}
        results["models"].append(result)
        print(json.dumps(result), flush=True)
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    (folder / "render-resource-check.json").write_text(json.dumps(results, indent=2))
    print("All four model requests passed. Local results saved; real Render/R2 verification still required.")


if __name__ == "__main__":
    main()
