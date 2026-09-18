#!/usr/bin/env python3
"""Bootstrap and start the local NEBULA X app with one command.

    python scripts/start_app.py
    python scripts/start_app.py --prepare-fleet
    python scripts/start_app.py --check

Uses only the standard library until the managed Python environment has been prepared.
Run it from any working directory. Ctrl-C stops the server.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
CONDA_ENV = ROOT / ".conda-app"
REQUIREMENTS = ROOT / "submission" / "nebulax" / "app" / "requirements.lock.txt"
FLEET_REQUIREMENTS = ROOT / "requirements-fleet.txt"
WEB = ROOT / "web"
MODELS = ROOT / "models" / "ps3"
ANACONDA_TOS_CHANNELS = frozenset({
    "https://repo.anaconda.com/pkgs/main",
    "https://repo.anaconda.com/pkgs/r",
    "https://repo.anaconda.com/pkgs/msys2",
})


def env_python(prefix: Path) -> Path:
    if os.name == "nt":
        return prefix / ("python.exe" if prefix == CONDA_ENV else "Scripts/python.exe")
    return prefix / "bin/python"


def env_root(interpreter: Path) -> Path:
    return CONDA_ENV if interpreter == env_python(CONDA_ENV) else VENV


def run(command: list[str | Path], *, cwd: Path = ROOT) -> None:
    args = [str(x) for x in command]
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def is_python_311(command: list[str]) -> bool:
    try:
        result = subprocess.run(
            [*command, "-c", "import sys; print(sys.version_info[:2] == (3, 11))"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip() == "True"


def python_311_command() -> list[str] | None:
    candidates = [[sys.executable]]
    if os.name == "nt":
        candidates.append(["py", "-3.11"])
    candidates.extend([["python3.11"], ["python"]])
    return next((command for command in candidates if is_python_311(command)), None)


def create_conda_environment(conda: str) -> None:
    """Create Python 3.11, accepting the named Anaconda channels if Conda requires it."""
    command = [conda, "create", "--prefix", str(CONDA_ENV), "python=3.11", "-y"]
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    if result.returncode == 0:
        return

    output = result.stdout + "\n" + result.stderr
    if "CondaToSNonInteractiveError" not in output:
        raise subprocess.CalledProcessError(result.returncode, command)
    required = re.findall(r"(?m)^\s*-\s*(https://repo\.anaconda\.com/pkgs/(?:main|r|msys2))\s*$", output)
    channels = list(dict.fromkeys(required))
    if not channels or any(channel not in ANACONDA_TOS_CHANNELS for channel in channels):
        raise RuntimeError("Conda requested channel terms that the launcher could not identify safely")
    print("Accepting Conda Terms of Service for the channels listed above, as requested", flush=True)
    for channel in channels:
        run([conda, "tos", "accept", "--override-channels", "--channel", channel])
    print("Retrying Conda environment creation", flush=True)
    run(command)


def prepare_python() -> Path:
    for prefix in (VENV, CONDA_ENV):
        interpreter = env_python(prefix)
        if interpreter.is_file() and is_python_311([str(interpreter)]):
            return interpreter

    command = python_311_command()
    if command:
        print("Creating local Python 3.11 environment in .venv", flush=True)
        run([*command, "-m", "venv", VENV])
        return env_python(VENV)

    conda = shutil.which("conda")
    if conda:
        print("Python 3.11 was not found; creating .conda-app with Conda", flush=True)
        create_conda_environment(conda)
        return env_python(CONDA_ENV)

    raise RuntimeError("Python 3.11 is required. Install it or Conda, then rerun this command.")


def install_python_dependencies(interpreter: Path) -> None:
    if not REQUIREMENTS.is_file():
        raise RuntimeError(f"Missing dependency lock: {REQUIREMENTS}")
    digest = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    stamp = env_root(interpreter) / ".nebulax-requirements.sha256"
    if stamp.is_file() and stamp.read_text(encoding="ascii").strip() == digest:
        return
    print("Installing app dependencies from the submission lock", flush=True)
    run([interpreter, "-m", "pip", "install", "-r", REQUIREMENTS])
    stamp.write_text(digest + "\n", encoding="ascii")


def install_fleet_dependencies(interpreter: Path) -> None:
    digest = hashlib.sha256(FLEET_REQUIREMENTS.read_bytes()).hexdigest()
    stamp = env_root(interpreter) / ".nebulax-fleet-requirements.sha256"
    if stamp.is_file() and stamp.read_text(encoding="ascii").strip() == digest:
        return
    print("Installing additional Fleet twin model dependencies", flush=True)
    run([interpreter, "-m", "pip", "install", "-r", FLEET_REQUIREMENTS])
    stamp.write_text(digest + "\n", encoding="ascii")


def build_web() -> None:
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if not npm:
        raise RuntimeError("Node.js and npm are required to build web/dist. Install Node.js and rerun.")
    lock = WEB / "package-lock.json"
    modules = WEB / "node_modules"
    stamp = modules / ".nebulax-lock.sha256"
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    if not modules.is_dir() or not stamp.is_file() or stamp.read_text(encoding="ascii").strip() != digest:
        print("Installing frontend dependencies with npm ci", flush=True)
        run([npm, "ci"], cwd=WEB)
        stamp.write_text(digest + "\n", encoding="ascii")
    print("Building the React frontend", flush=True)
    run([npm, "run", "build"], cwd=WEB)
    if not (WEB / "dist" / "index.html").is_file():
        raise RuntimeError("The frontend build finished without web/dist/index.html")


def check_models() -> None:
    missing = [str(MODELS / f"{task}.pkl") for task in ("door", "acv", "rail", "shm")
               if not (MODELS / f"{task}.pkl").is_file()]
    if missing:
        raise RuntimeError("Saved PS3 models are missing: " + ", ".join(missing))


def prepare_fleet(interpreter: Path) -> None:
    sim = ROOT / "data" / "sim" / "index.json"
    scores = ROOT / "data" / "scores" / "scores.parquet"
    if sim.is_file() and scores.is_file():
        print("Synthetic fleet and scores already exist; skipping generation", flush=True)
        return
    if not (ROOT / "results" / "runs.parquet").is_file():
        raise RuntimeError("Fleet scoring needs results/runs.parquet from the benchmark")
    if not sim.is_file():
        print("Generating the synthetic fleet; this can take many minutes", flush=True)
        run([interpreter, ROOT / "scripts" / "generate.py", "--subsystem", "all",
             "--n-trains", "10", "--days", "30", "--seed", "0", "--out", "data/sim",
             "--p-healthy", "0.4", "--max-faults", "1", "--store-every", "10",
             "--workers", "4"])
    print("Scoring the fleet; skipping MetroPT-3 if its raw data is absent", flush=True)
    args = [interpreter, ROOT / "scripts" / "score_for_demo.py"]
    if not (ROOT / "data" / "raw" / "metropt3").is_dir():
        args.append("--skip-metropt3")
    run(args)


def open_when_ready(process: subprocess.Popen[bytes], url: str) -> None:
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError(f"The app exited before becoming ready (exit {process.returncode})")
        try:
            with urllib.request.urlopen(url + "api/health", timeout=1) as response:
                if response.status == 200:
                    print(f"App ready: {url}", flush=True)
                    webbrowser.open(url)
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"The app did not become ready at {url} within 30 seconds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="local bind address (default: 127.0.0.1)")
    parser.add_argument("--port", default=8765, type=int, help="server port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--prepare-fleet", action="store_true", help="generate and score the optional fleet twin")
    parser.add_argument("--check", action="store_true", help="show prerequisite status without installing or starting")
    args = parser.parse_args(argv)

    if args.check:
        print(f"Repository: {ROOT}")
        local_python = any(env_python(prefix).is_file() and is_python_311([str(env_python(prefix))])
                           for prefix in (VENV, CONDA_ENV))
        print(f"Python 3.11: {'available' if local_python or python_311_command() else 'Conda bootstrap needed' if shutil.which('conda') else 'missing'}")
        print(f"npm: {'available' if shutil.which('npm.cmd' if os.name == 'nt' else 'npm') else 'missing'}")
        print(f"Frontend built: {(WEB / 'dist' / 'index.html').is_file()}")
        full_fleet = (ROOT / 'data' / 'sim' / 'index.json').is_file() and (ROOT / 'data' / 'scores' / 'scores.parquet').is_file()
        demo_fleet = (ROOT / 'demo_data' / 'sim' / 'index.json').is_file() and (ROOT / 'demo_data' / 'scores' / 'scores.parquet').is_file()
        print(f"Fleet replay ready: {full_fleet or demo_fleet}")
        check_models()
        print("All four saved PS3 models are present")
        return 0

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    os.chdir(ROOT)
    check_models()
    interpreter = prepare_python()
    install_python_dependencies(interpreter)
    build_web()
    if args.prepare_fleet:
        install_fleet_dependencies(interpreter)
        prepare_fleet(interpreter)
    elif not ((ROOT / "data" / "scores" / "scores.parquet").is_file()
              or (ROOT / "demo_data" / "scores" / "scores.parquet").is_file()):
        print("Fleet replay data is absent. Use --prepare-fleet to generate it; PS3 prediction is ready.", flush=True)

    command = [str(interpreter), "-m", "uvicorn", "nebulax.api.main:app",
               "--host", args.host, "--port", str(args.port)]
    print("+", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=ROOT)
    url = f"http://127.0.0.1:{args.port}/"
    try:
        if not args.no_browser and args.host in ("127.0.0.1", "localhost"):
            open_when_ready(process, url)
        else:
            print(f"App starting at {url}", flush=True)
        return process.wait()
    except KeyboardInterrupt:
        print("\nStopping app...", flush=True)
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"start_app: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
