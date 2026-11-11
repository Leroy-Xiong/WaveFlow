#!/usr/bin/env python
"""Render release panels in-process and return PIL images (for notebooks).

The regular CLI writes a PNG per panel and reads it back. In a notebook we can
instead run the same panel script in the current kernel, intercept the
matplotlib ``Figure.savefig`` call, crop the figure in memory to the panel
box from ``reproduce/panels.yaml``, and display the Figure/Image directly.

    import sys; sys.path.insert(0, "reproduce")
    import notebook_render
    for name, img in notebook_render.render("fig2a"):
        display(img)
"""
from __future__ import annotations

import io
import os
import runpy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.figure import Figure
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PANELS = yaml.safe_load((REPO / "reproduce" / "panels.yaml").read_text())["panels"]

_ORIG_SAVEFIG = Figure.savefig
_ORIG_CLOSE = plt.close


def _crop_mm(img: Image.Image, bbox_mm, canvas_mm) -> Image.Image:
    x0, y0, x1, y1 = (float(v) for v in bbox_mm)
    cw, ch = (float(v) for v in canvas_mm)
    w, h = img.size
    px = (int(round(x0 / cw * w)), int(round(y0 / ch * h)),
          int(round(x1 / cw * w)), int(round(y1 / ch * h)))
    px = (max(0, px[0]), max(0, px[1]), min(w, px[2]), min(h, px[3]))
    return img.crop(px)


def _run(spec: dict) -> list[Figure]:
    captured: list[Figure] = []

    def savefig(self, *args, **kwargs):
        if self not in captured:
            captured.append(self)
        return None

    def close(*args, **kwargs):
        return None

    Figure.savefig = savefig
    plt.close = close
    old_argv = sys.argv
    try:
        if spec.get("kind") == "panel":
            script = REPO / "reproduce" / spec["script"]
            sys.argv = [str(script), "--out", "/tmp/waveflow_panel.png",
                        "--dpi", str(int(spec.get("dpi", 300)))]
        else:
            script = REPO / spec["script"]
            sys.argv = [str(script)]
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in (spec.get("env") or {}).items()})
        if spec.get("kind") == "whole":
            if "fig6" in str(script):
                env.setdefault("FIG6_SKIP_PDF", "1")
            if "ed_fields_global" in str(script):
                env.setdefault("ED6_SKIP_PDF", "1")
        old_env = os.environ.copy()
        os.environ.update(env)
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        finally:
            os.environ.clear()
            os.environ.update(old_env)
        return captured
    finally:
        sys.argv = old_argv
        Figure.savefig = _ORIG_SAVEFIG
        plt.close = _ORIG_CLOSE


def _strip_panel_labels(fig: Figure):
    """Remove the bold single-letter panel labels added by the paper scripts."""
    for artist_list in [fig.texts] + [ax.texts for ax in fig.axes]:
        for t in list(artist_list):
            if (t.get_text().strip() in "abcdefgh"
                    and abs(float(t.get_fontsize()) - 9.0) < 0.6
                    and str(t.get_fontweight()) in ("bold", "700", "heavy")):
                t.remove()


def render(name: str):
    """Yield ``(label, PIL.Image)`` for one release panel/figure."""
    spec = PANELS[name]
    dpi = int(spec.get("dpi", 300))
    for fig in _run(spec):
        if spec.get("kind") == "panel":
            _strip_panel_labels(fig)
        fig.canvas.draw()
        buf = io.BytesIO()
        _ORIG_SAVEFIG(fig, buf, format="png", dpi=dpi, facecolor="white")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        if spec.get("kind") == "panel":
            for crop in spec.get("crops", []):
                yield crop["name"], _crop_mm(img, crop["bbox_mm"],
                                             spec["canvas_mm"])
        else:
            yield name, img
