"""Shared paths for the WaveFlow inference package."""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("WAVEFLOW_REPO", HERE.parent)).resolve()


def load_downloads() -> dict:
    import yaml

    path = HERE / "downloads.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text())


def ensure_dirs() -> None:
    for name in ("weights", "downloads", "output", "data"):
        (REPO / name).mkdir(parents=True, exist_ok=True)
