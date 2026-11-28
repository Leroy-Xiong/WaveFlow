#!/usr/bin/env python
"""Download the released checkpoints and the model assets.

URLs, sizes and SHA256 checksums come from ``waveflow/downloads.yaml``.
Downloads are resumable and verified. Weights land in ``weights/``; the asset
archive (mesh coordinates, clusters, normalization statistics, ocean boundary)
is extracted at the repository root.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: E402

REPO = bootstrap.REPO
ARCHIVE_DIR = REPO / "downloads"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024.0


def verify(path: Path, sha256: str = "", size: int = 0) -> bool:
    if not path.exists():
        print(f"  MISSING {path}")
        return False
    if size and path.stat().st_size != int(size):
        print(f"  BAD SIZE {path}: {path.stat().st_size} != {size}")
        return False
    if sha256:
        got = sha256_file(path)
        if got.lower() != sha256.lower():
            print(f"  BAD SHA256 {path}\n    got  {got}\n    want {sha256}")
            return False
    print(f"  OK {path} ({_fmt(path.stat().st_size)})")
    return True


def download(url: str, dest: Path, sha256: str = "", size: int = 0):
    """Resumable download with optional checksum verification."""
    if not url:
        raise SystemExit(f"no URL configured for {dest.name}; see WEIGHTS.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and verify(dest, sha256, size):
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    pos = tmp.stat().st_size if tmp.exists() else 0
    headers = {}
    if pos:
        headers["Range"] = f"bytes={pos}-"
    req = urllib.request.Request(url, headers=headers)
    print(f"downloading {url}\n  -> {dest} (resume at {_fmt(pos)})")
    t0 = time.time()
    with urllib.request.urlopen(req) as resp:
        total = resp.headers.get("Content-Length")
        total = int(total) + pos if total else int(size or 0)
        mode = "ab" if pos and resp.status == 206 else "wb"
        if mode == "wb":
            pos = 0
        with open(tmp, mode) as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                pos += len(chunk)
                if total:
                    pct = 100.0 * pos / total
                    print(f"\r  {pct:5.1f}%  {_fmt(pos)} / {_fmt(total)}",
                          end="", flush=True)
    print(f"\r  done in {time.time() - t0:.0f}s ({_fmt(tmp.stat().st_size)})")
    tmp.replace(dest)
    if not verify(dest, sha256, size):
        raise SystemExit(f"checksum verification failed for {dest}")
    return dest


def safe_extract_tar(path: Path, root: Path):
    root = root.resolve()
    with tarfile.open(path) as tf:
        for member in tf.getmembers():
            target = (root / member.name).resolve()
            if not str(target).startswith(str(root) + os.sep) and target != root:
                raise SystemExit(f"refusing unsafe path in archive: {member.name}")
        try:
            tf.extractall(root, filter="data")
        except TypeError:  # Python < 3.12
            tf.extractall(root)


def get_weights(dl: dict) -> None:
    """Download every released checkpoint plus the shared model assets."""
    spec = (dl or {}).get("weights") or {}
    files = spec.get("files") or {}
    if not files:
        raise SystemExit("downloads.yaml has no weights section")
    for _, item in files.items():
        dest = REPO / "weights" / item["name"]
        download(item["url"], dest, item.get("sha256", ""),
                 int(item.get("size_bytes", 0) or 0))
    assets = spec.get("assets") or {}
    if assets.get("url"):
        arch = ARCHIVE_DIR / "waveflow-model-assets.tar.gz"
        download(assets["url"], arch, assets.get("sha256", ""),
                 int(assets.get("size_bytes", 0) or 0))
        print(f"extracting model assets at {REPO} ...")
        safe_extract_tar(arch, REPO)
