"""Patchable runtime configuration. Consumers import this module, not its values."""

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")  # sightline-ok: 9 - env files are configuration, read once per process
load_dotenv(_ROOT / "panda.env")  # sightline-ok: 9 - Panda credentials use the same one-time read


def _path(key: str, default: Path) -> Path:
    return Path(os.environ.get(key) or default).expanduser()


DATA_DIR = _path("LEAGUE_PREDICTOR_DATA_DIR", _ROOT / "data")
RAW_DIR = _path("LEAGUE_PREDICTOR_RAW_DIR", DATA_DIR / "raw")
DB_PATH = _path("LEAGUE_PREDICTOR_DB_PATH", DATA_DIR / "league_predictor.db")
MODELS_DIR = _path("LEAGUE_PREDICTOR_MODELS_DIR", _ROOT / "models")
OE_GDRIVE_BASE = "https://drive.google.com/uc?id="
OE_GDRIVE_IDS = {
    2026: "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm",
    2025: "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",
    2024: "1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN",
}
OE_YEARS = sorted(OE_GDRIVE_IDS)
PANDASCORE_TOKEN = os.environ.get("PANDA") or None
ROLE_WEIGHTS = {"mid": 1.5, "jng": 1.5, "bot": 1.2, "top": 1.0, "sup": 0.5}
DASHBOARD_DEFAULT_PORT = 5000
