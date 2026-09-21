"""Read the migration checkout's .env without logging or expanding secret values."""
import os
from pathlib import Path

KEYS = {"HF_SPACE_ID", "HF_TOKEN", "NEBULAX_RUN_BUCKET", "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "RENDER_API_KEY", "RENDER_OWNER_ID",
        "RENDER_REPO_URL", "RENDER_BRANCH", "RENDER_SERVICE_NAME"}


def load_env(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1] / ".env"
    if not path.exists():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or key not in KEYS:
            raise ValueError(f"Unsupported .env entry at line {number}")
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unclosed .env quote at line {number}")
            value = value[1:-1]
        # No variable interpolation or shell execution. Existing process values win.
        os.environ.setdefault(key, value)


def require_env(*keys):
    missing = [key for key in keys if not os.getenv(key) or "YOUR_" in os.environ[key]]
    if missing:
        raise SystemExit("Complete these .env fields: " + ", ".join(missing))
