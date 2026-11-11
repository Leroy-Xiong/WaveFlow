#!/usr/bin/env python
"""Render only panel a of main Fig. 2 (atmospheric forcing and discretisations).

The original paper script (``scripts/figures/figure2/plot_figure2.py``) is
imported unchanged; only ``draw_panel_a`` is called, so the layout, fonts and
colours are exactly the paper ones. The full 180 x 255 mm canvas is saved;
``reproduce.py`` crops the requested panel box from it.

Needs: HK + global coordinates, one ERA5 region-2 frame, the static 24x24
ocean mask (all shipped) and cartopy's GSHHS shapefiles (auto-downloaded).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
import bootstrap  # noqa: E402

REPO = bootstrap.REPO


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts/figures"))
    sys.path.insert(0, str(REPO / "scripts/figures/figure2"))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import plot_figure2 as F2
    import typography as TY

    fig = plt.figure(figsize=(F2.FIG_W, F2.FIG_H))
    F2.draw_panel_a(fig)
    fig.canvas.draw()
    TY.enforce_glyph_floor(fig)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor="white")
    print(f"[fig2a] saved full canvas {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
