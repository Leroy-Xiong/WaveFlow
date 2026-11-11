#!/usr/bin/env python
"""Download the released weights and figure-source-data archives.

URLs, sizes and SHA256 checksums come from ``reproduce/downloads.yaml`` (written
when the Zenodo records are published). Downloads are resumable and verified.

    python download.py --weights
    python download.py --figure-data
    python download.py --all
    python download.py --forcing          # optional global IFS-ENS case forcing

Weights land in ``weights/``; the asset/figure archives are extracted at the
repository root, restoring the exact paths the paper scripts expect.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: E402

REPO = bootstrap.REPO
ARCHIVE_DIR = REPO / "downloads"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
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
        raise SystemExit(
            f"no URL configured for {dest.name}: the Zenodo record has not been "
            "published yet. See WEIGHTS.md / DATA.md.")
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


def safe_extract_zip(path: Path, root: Path):
    root = root.resolve()
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            target = (root / name).resolve()
            if not str(target).startswith(str(root) + os.sep) and target != root:
                raise SystemExit(f"refusing unsafe path in zip: {name}")
        zf.extractall(root)


def maybe_unzip_dtm():
    """Fig. 6 re-inference needs the LandsD DTM as .asc; the release ships it
    zipped to save space."""
    dtm_dir = REPO / "data/hk_coastline/dtm_landsd_official"
    z = dtm_dir / "Whole_HK_DTM_5m.zip"
    asc = dtm_dir / "Whole_HK_DTM_5m.asc"
    if z.exists() and not asc.exists():
        print(f"unzipping {z.name} ...")
        # the archive contains a single .asc (possibly inside a folder)
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".asc"):
                    with zf.open(name) as src, open(asc, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"  -> {asc}")
                    return
        print("  warning: no .asc found in the DTM zip")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def get_weights(dl):
    spec = dl.get("weights") or {}
    files = spec.get("files") or {}
    if not files:
        raise SystemExit("downloads.yaml has no weights section")
    for key, item in files.items():
        dest = REPO / "weights" / item["name"]
        download(item["url"], dest, item.get("sha256", ""), int(item.get("size_bytes", 0) or 0))
    assets = spec.get("assets") or {}
    if assets.get("url"):
        arch = ARCHIVE_DIR / "waveflow-model-assets.tar.gz"
        download(assets["url"], arch, assets.get("sha256", ""),
                 int(assets.get("size_bytes", 0) or 0))
        print(f"extracting model assets at {REPO} ...")
        safe_extract_tar(arch, REPO)


def get_figure_data(dl):
    spec = dl.get("figure_data") or {}
    if not spec.get("url"):
        raise SystemExit("downloads.yaml has no figure_data URL yet")
    arch = ARCHIVE_DIR / spec.get("archive", "waveflow-v1.0.0-figure-data.tar.gz")
    download(spec["url"], arch, spec.get("sha256", ""),
             int(spec.get("size_bytes", 0) or 0))
    print(f"extracting figure data at {REPO} ...")
    safe_extract_tar(arch, REPO)
    maybe_unzip_dtm()


def get_forcing(dl):
    spec = dl.get("forcing") or {}
    if not spec.get("url"):
        raise SystemExit(
            "no optional forcing archive published; fetch the IFS-ENS members "
            "from WeatherBench2 as described in REPRODUCE.md")
    arch = ARCHIVE_DIR / spec.get("archive", "waveflow-v1.0.0-global-forecast-forcing.tar.gz")
    download(spec["url"], arch, spec.get("sha256", ""),
             int(spec.get("size_bytes", 0) or 0))
    print(f"extracting forcing at {REPO} ...")
    safe_extract_tar(arch, REPO)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--weights", action="store_true")
    g.add_argument("--figure-data", action="store_true")
    g.add_argument("--forcing", action="store_true")
    g.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)
    dl = bootstrap.load_downloads()
    if not dl:
        raise SystemExit("reproduce/downloads.yaml is missing; this package was "
                         "not built with published URLs.")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    if args.weights or args.all:
        get_weights(dl)
    if args.figure_data or args.all:
        get_figure_data(dl)
    if args.all:
        if dl.get("forcing", {}).get("url"):
            get_forcing(dl)
    if args.forcing:
        get_forcing(dl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
