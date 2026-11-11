#!/usr/bin/env python
"""Crop the release canvases into panel images, and build reference panels.

Panel coordinates in ``panels.yaml`` are millimetres measured from the
top-left corner of the original full canvas (the figures are laid out with
absolute millimetre positions, so a crop is a pure fraction of the canvas and
does not depend on the rendering DPI).

Two modes
---------
    # build-time: crop the paper figures into reference/<name>.png
    python crop_panels.py --reference --repo ROOT --panels PANELS.yaml \
        --out-dir ROOT/reference

    # public: crop the raw panel canvases produced by reproduce/panels/*.py
    python crop_panels.py --raw-dir reproduced/_raw --out-dir reproduced

USAGE
-----
    python crop_panels.py --reference --repo . --panels panels.yaml --out-dir reference
    python crop_panels.py --raw-dir reproduced/_raw --out-dir reproduced --only fig3def
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from PIL import Image


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------
def load_panels(path):
    with open(path) as fh:
        return yaml.safe_load(fh)["panels"]


def open_image(path: Path, dpi: int = 300) -> Image.Image:
    """Open a PNG or render the first page of a PDF at ``dpi``."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        try:
            import pymupdf as fitz
        except ImportError:  # PyMuPDF < 1.24
            import fitz
        doc = fitz.open(path)
        page = doc[0]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    img = Image.open(path)
    return img.convert("RGB")


def crop_mm(img: Image.Image, bbox_mm, canvas_mm) -> Image.Image:
    """Crop ``bbox_mm`` (x0, y0, x1, y1 from top-left) out of a canvas."""
    x0, y0, x1, y1 = (float(v) for v in bbox_mm)
    cw, ch = (float(v) for v in canvas_mm)
    w, h = img.size
    px = (int(round(x0 / cw * w)), int(round(y0 / ch * h)),
          int(round(x1 / cw * w)), int(round(y1 / ch * h)))
    px = (max(0, px[0]), max(0, px[1]), min(w, px[2]), min(h, px[3]))
    if px[2] <= px[0] or px[3] <= px[1]:
        raise ValueError(f"empty crop {bbox_mm} on {w}x{h} canvas")
    return img.crop(px)


def _save(img: Image.Image, out: Path, dpi: int | None = None):
    out.parent.mkdir(parents=True, exist_ok=True)
    if dpi:
        img.save(out, dpi=(dpi, dpi))
    else:
        img.save(out)
    print(f"  wrote {out}  ({img.width}x{img.height})")


# ---------------------------------------------------------------------------
# reference generation (build time)
# ---------------------------------------------------------------------------
def make_references(panels, repo: Path, out_dir: Path, only=None):
    for key, spec in panels.items():
        if only and key not in only:
            continue
        if spec["kind"] == "panel":
            canvas = spec.get("canvas_mm")
            for crop in spec["crops"]:
                src = Path(repo) / crop["reference_source"]
                dpi = int(crop.get("reference_dpi", 300))
                img = open_image(src, dpi=dpi)
                if canvas is None:
                    raise ValueError(f"{key}: canvas_mm missing")
                out = Path(out_dir) / f"{crop['name']}.png"
                _save(crop_mm(img, crop["bbox_mm"], canvas), out)
        else:  # whole figure
            src = Path(repo) / spec["reference_source"]
            dpi = int(spec.get("reference_dpi", 300))
            img = open_image(src, dpi=dpi)
            out = Path(out_dir) / f"{key}.png"
            _save(img, out)


# ---------------------------------------------------------------------------
# public: crop raw canvases
# ---------------------------------------------------------------------------
def crop_outputs(panels, raw_dir: Path, out_dir: Path, only=None):
    written = []
    for key, spec in panels.items():
        if only and key not in only:
            continue
        if spec["kind"] != "panel":
            continue
        raw = Path(raw_dir) / f"{key}.png"
        if not raw.exists():
            raise FileNotFoundError(
                f"missing raw canvas {raw}; run reproduce.py {key} first")
        img = Image.open(raw).convert("RGB")
        canvas = spec["canvas_mm"]
        for crop in spec["crops"]:
            out = Path(out_dir) / f"{crop['name']}.png"
            _save(crop_mm(img, crop["bbox_mm"], canvas), out,
                  dpi=int(spec.get("dpi", 300)))
            written.append(out)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", default=str(Path(__file__).parent / "panels.yaml"))
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]),
                    help="repository root (build time)")
    ap.add_argument("--reference", action="store_true",
                    help="crop the paper figures into reference panels")
    ap.add_argument("--raw-dir", default=None,
                    help="directory with full-canvas panel outputs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default=None,
                    help="comma-separated panel ids to process")
    args = ap.parse_args(argv)

    panels = load_panels(args.panels)
    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None
    if args.reference:
        make_references(panels, Path(args.repo), Path(args.out_dir), only=only)
    else:
        if not args.raw_dir:
            raise SystemExit("--raw-dir is required unless --reference is used")
        crop_outputs(panels, Path(args.raw_dir), Path(args.out_dir), only=only)


if __name__ == "__main__":
    sys.exit(main())
