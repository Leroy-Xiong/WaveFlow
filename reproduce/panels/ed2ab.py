#!/usr/bin/env python
"""Render panels a and b of Extended Data Fig. 2 (mesh spacing + global
per-node RMSE).

Imports the paper module (``plot_ed_mesh_coastal.py``) and reuses its data
loader and drawing helpers; only the two requested axes are created. The full
180 x 113 mm canvas is saved and cropped by ``reproduce.py``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bootstrap  # noqa: E402

REPO = bootstrap.REPO


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args(argv)

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts/figures"))
    sys.path.insert(0, str(REPO / "scripts/figures/figure_ed_mesh_coastal"))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import plot_ed_mesh_coastal as M

    mesh_rows, hk_rows, _hk_share, g_rows, _h_rows = M.load()
    fig = plt.figure(figsize=(M.FW * M.MM, M.FH * M.MM))

    ax = M.ax_at(fig, M.COLS[0], M.ROWS_Y[0], M.PW, M.PH)
    M.draw_mesh_spacing(ax, mesh_rows, hk_rows)
    M.panel(ax, "a")

    ax = M.ax_at(fig, M.COLS[1], M.ROWS_Y[0], M.PW, M.PH)
    M.draw_rmse(ax, g_rows, M.MODELS, with_legend="lower right")
    M.panel(ax, "b", "Global")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor="white")
    print(f"[ed2ab] saved full canvas {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
