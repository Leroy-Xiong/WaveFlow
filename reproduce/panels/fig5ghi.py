#!/usr/bin/env python
"""Render panels g, h and i of main Fig. 5 (hurricane case fields).

Reuses the paper drawing helpers from ``plot_figure5_case.py`` (Hong Kong /
Kompasu) and ``plot_figure5_global.py`` (global / Hurricane Ian) and the exact
millimetre layout of ``plot_figure5_main.py``. The full 180 x 200 mm canvas is
saved; ``reproduce.py`` crops the three panels.

The reference columns are hindcast fields: supply the two frames described in
DATA.md (or set WF_GT_HK / WF_GT_GLOBAL). The model-side samples are shipped.
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

CANVAS_W, CANVAS_H = 180.0, 200.0
G_X0, G_W, G_GAP = 6.6, 30.34, 2.0
G_TOP = 110.6
G_H = G_W / 1.4286
H_TOP = 162.9
H_GAP = 1.5
H_PITCH = G_W + G_GAP
H_W = H_PITCH - H_GAP
H_X0 = G_X0 + 0.5 * G_W - 0.5 * H_W
H_H = H_W / 1.9717
I_TOP = H_TOP + H_H + 2.0
CASE_TS = "20220928T12"
CASE_TMPL = {12: "forecast_ian_memens_lead12h",
             24: "forecast_ian_memens_lead24h",
             48: "forecast_ian_memens_lead48h",
             72: "forecast_ian_memens_lead72h"}
CASE_ZOOM = (-90.0, -74.0, 23.0, 31.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts/figures"))
    sys.path.insert(0, str(REPO / "scripts/figures/figure5"))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    import plot_figure5_case as CASE
    import plot_figure5_global as GLOB
    import typography as TY

    TY.use_nature_fonts()

    # fail early with a precise message instead of a page of traceback
    missing = []
    for lead in CASE.LEADS:
        f = Path(CASE.case_sample_dir(lead)) / f"sample_{CASE.T_MAP}_ens000.npz"
        if not f.exists():
            missing.append(str(f))
    for lead, tmpl in CASE_TMPL.items():
        f = Path(GLOB.RUN) / tmpl / "samples" / f"sample_{CASE_TS}_ens000.npz"
        if not f.exists():
            missing.append(str(f))
    if missing:
        raise SystemExit(
            "missing Fig. 5g/h/i model samples:\n  " + "\n  ".join(missing) +
            "\nrun `python reproduce.py setup --figure-data` first")

    hk = np.load(CASE.COORDS)
    hk_lats, hk_lons = hk[:, 0], hk[:, 1]
    try:
        gt_hk = CASE.gt_hs(CASE.T_MAP)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"missing Hong Kong hindcast frame for Fig. 5g: {exc}\n"
            "see DATA.md (WF_GT_HK)") from exc
    gc = np.load(GLOB.COORDS)
    g_lats, g_lons = gc[:, 0], gc[:, 1]
    try:
        gt_global = GLOB.gt_hs(CASE_TS)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"missing global hindcast frame for Fig. 5h/i: {exc}\n"
            "see DATA.md (WF_GT_GLOBAL)") from exc

    MM = 1 / 25.4
    fig = plt.figure(figsize=(CANVAS_W * MM, CANVAS_H * MM))

    CASE.draw_mean_map_row(fig, hk_lons, hk_lats, gt_hk, x0=G_X0,
                           y_top=G_TOP, map_w=G_W, dx=G_W + G_GAP,
                           canvas=(CANVAS_W, CANVAS_H), left_labels=True)
    CASE.draw_prob_map_row(fig, hk_lons, hk_lats, gt_hk, x0=G_X0,
                           y_top=G_TOP + G_H + G_GAP, map_w=G_W,
                           dx=G_W + G_GAP, canvas=(CANVAS_W, CANVAS_H))

    GLOB.draw_global_mean_row(fig, g_lons, g_lats, gt_global, x0=H_X0,
                              y_top=H_TOP, map_w=H_W, dx=H_PITCH,
                              canvas=(CANVAS_W, CANVAS_H), rmse=None,
                              show_rmse=False, ts=CASE_TS, tmpls=CASE_TMPL)
    GLOB.draw_zoom_row(fig, g_lons, g_lats, gt_global, x0=G_X0, y_top=I_TOP,
                       map_w=G_W, dx=G_W + G_GAP, canvas=(CANVAS_W, CANVAS_H),
                       ts=CASE_TS, tmpls=CASE_TMPL, region=CASE_ZOOM,
                       dot_s=0.6)

    def letter(x, y, s):
        fig.text(x / CANVAS_W, 1 - y / CANVAS_H, s, fontsize=9,
                 fontweight="bold", ha="left", va="top")

    letter(2, G_TOP - 2.0, "g")
    letter(2, H_TOP - 2.0, "h")
    letter(2, I_TOP - 2.0, "i")

    fig.canvas.draw()
    TY.enforce_glyph_floor(fig)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor="white")
    print(f"[fig5ghi] saved full canvas {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
