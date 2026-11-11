"""Shared helpers for the WaveFlow figure-reproduction package.

The public tree keeps the original repository layout, so the release tools
locate everything relative to the repository root (``reproduce/..``).
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("WAVEFLOW_REPO", HERE.parent)).resolve()


def load_yaml(name: str):
    import yaml
    with open(HERE / name) as fh:
        return yaml.safe_load(fh)


def load_panels():
    return load_yaml("panels.yaml")["panels"]


def load_downloads():
    p = HERE / "downloads.yaml"
    if not p.exists():
        return {}
    return load_yaml("downloads.yaml")


def ensure_dirs():
    for d in ("weights", "downloads", "reproduced", "output", "data"):
        (REPO / d).mkdir(parents=True, exist_ok=True)
