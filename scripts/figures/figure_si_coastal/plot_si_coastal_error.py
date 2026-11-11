#!/usr/bin/env python
"""Supplementary figure: where in the domain the unstructured mesh pays off.

Upper row: the global mesh. Lower row: the Hong Kong mesh. In both cases the
per-node RMSE of significant wave height is split by the distance from the
node to the nearest coastline, and WaveFlow is compared with the baselines for
which per-node errors could be recovered.

INPUTS (read-only)
  global
    data/global_hindcast/global_coordinates.npy
    data/global_hindcast/ocean_boundary_wkb.npz
    earthformer_repro/.../comparison_cfm/spatial_rmse.npz        ('cfm','ef')
    oceancastnet_repro/.../comparison_cfm/spatial_rmse.npz       ('ocn')
  Hong Kong
    data/hindcast/hk_coordinates.npy
    output/results/si_coastal_hk/{waveflow,vit,ef,ocn}_sqerr.npz
        produced by compute_hk_node_rmse.py, 'sqerr' + 'count' (7788,)
    coastline: GSHHS full-resolution land polygons (cartopy cache), clipped to
        the Hong Kong box
  WaveFlow-ViT is absent from the global row: its node-level predictions for
  the global test set were not retained.

OUTPUTS
  output/plot/figure_si_coastal/figure_si_coastal.{png,pdf}
  output/plot/figure_si_coastal/coastal_error_by_distance.csv

USAGE
  python scripts/figures/figure_si_coastal/plot_si_coastal_error.py
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "figures",
                                "figure_si_mesh_resolution"))
import plot_mesh_resolution as MESH  # noqa: E402

OUT_DIR = os.path.join(ROOT, "output/plot/figure_si_coastal")
OUT_BASE = os.path.join(OUT_DIR, "figure_si_coastal")

GLOBAL_BANDS = [(0, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 200),
                (200, 500), (500, np.inf)]
GLOBAL_LABELS = ["0-5", "5-10", "10-25", "25-50", "50-100", "100-200",
                 "200-500", ">500"]
HK_BANDS = [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, np.inf)]
HK_LABELS = ["0-0.5", "0.5-1", "1-2", "2-5", "5-10", ">10"]

EF_NPZ = ("earthformer_repro/output/results/global_hs/train_20260804_120424/"
          "comparison_cfm/spatial_rmse.npz")
OCN_NPZ = ("oceancastnet_repro/output/results/global_swh/"
           "train_b192_20260803_113451/comparison_cfm/spatial_rmse.npz")
# Release override: the baseline reproduction repos are not distributed;
# the per-node squared-error arrays behind Extended Data Fig. 2 are shipped
# as figure source data under output/results/si_coastal_global/.
if not os.path.exists(EF_NPZ):
    EF_NPZ = "output/results/si_coastal_global/waveflow_ef_spatial_rmse.npz"
if not os.path.exists(OCN_NPZ):
    OCN_NPZ = "output/results/si_coastal_global/ocn_spatial_rmse.npz"
HK_NPZ = "output/results/si_coastal_hk/{}_sqerr.npz"
GLOBAL_VIT_NPZ = "output/results/si_coastal_global/vit_global_sqerr.npz"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "TeX Gyre Heros",
                        "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 5.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
})

C_OURS = "#B2182B"
C_VIT = "#ED8D5A"
C_EF = "#1f5fa9"
C_OCN = "#4d9f7f"
R_EARTH_KM = 6371.0088


# --------------------------------------------------------------------------
# distance to coastline
# --------------------------------------------------------------------------
def global_distance():
    coords = np.load("data/global_hindcast/global_coordinates.npy")
    xyz = MESH.to_xyz(coords[:, 1], coords[:, 0])
    coast, _ = MESH.coastline_points()
    return cKDTree(MESH.to_xyz(coast[:, 0], coast[:, 1])).query(xyz, k=1)[0]


def hk_distance():
    """Local-projection distance from each HK node to the GSHHS coastline."""
    from cartopy.io import shapereader as shpreader
    from shapely.geometry import box

    coords = np.load("data/hindcast/hk_coordinates.npy")   # (lat, lon)
    lat0 = np.radians(np.mean(coords[:, 0]))

    def to_km(lon, lat):
        return np.column_stack([np.radians(lon) * np.cos(lat0) * R_EARTH_KM,
                                np.radians(lat) * R_EARTH_KM])

    clip = box(113.4, 22.0, 114.8, 22.8)          # minx, miny, maxx, maxy
    pts = []
    for geom in shpreader.Reader(shpreader.gshhs(scale="f", level=1)).geometries():
        if not geom.intersects(clip):
            continue
        inter = geom.intersection(clip)
        for poly in getattr(inter, "geoms", [inter]):
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            pts.append(np.asarray(poly.exterior.coords))
    pts = np.vstack(pts)
    xyz = to_km(coords[:, 1], coords[:, 0])
    return cKDTree(to_km(pts[:, 0], pts[:, 1])).query(xyz, k=1)[0]


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def bin_stats(dist, arrays, bands, labels):
    rows = []
    for (lo, hi), label in zip(bands, labels):
        m = (dist >= lo) & (dist < hi)
        if m.sum() < 30:
            continue
        row = {"band_km": label, "n_nodes": int(m.sum()),
               "median_distance_km": float(np.median(dist[m]))}
        for name, arr in arrays.items():
            row[f"median_{name}"] = float(np.nanmedian(arr[m]))
        for name in ("vit", "ef", "ocn"):
            row[f"ratio_{name}"] = (
                float(np.nanmedian(arrays[name][m] / arrays["ours"][m]))
                if name in arrays else np.nan)
        rows.append(row)
    return rows


def draw_row(ax_r, ax_q, rows, series, label_r, label_q, vit_label):
    x = np.array([r["median_distance_km"] for r in rows])
    for key, label, colour, marker in series:
        ax_r.plot(x, [r[f"median_{key}"] for r in rows], marker=marker, ms=3.0,
                  lw=1.0, color=colour, markeredgecolor="white",
                  markeredgewidth=0.3, label=label)
    ax_r.set_xscale("log")
    ax_r.set_xlabel("distance to nearest coastline (km)")
    ax_r.set_ylabel("median per-node\nRMSE of $H_{\\mathrm{s}}$ (m)")
    ax_r.set_title(label_r, loc="left", fontweight="bold")

    ax_q.axhline(1.0, color="#888888", lw=0.7, ls="--", zorder=0)
    for rkey, label, colour, marker in [
            ("ratio_vit", vit_label, C_VIT, "D"),
            ("ratio_ef", "EarthFormer / WaveFlow", C_EF, "s"),
            ("ratio_ocn", "OceanCastNet / WaveFlow", C_OCN, "^")]:
        y = np.array([r[rkey] for r in rows], dtype=float)
        if not np.isfinite(y).any():
            continue
        ax_q.plot(x, y, marker=marker, ms=3.0, lw=1.0, color=colour,
                  markeredgecolor="white", markeredgewidth=0.3, label=label)
    ax_q.set_xscale("log")
    ax_q.set_xlabel("distance to nearest coastline (km)")
    ax_q.set_ylabel("median ratio\nto WaveFlow")
    ax_q.set_title(label_q, loc="left", fontweight="bold")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ef_npz = np.load(EF_NPZ)
    g_arr = {"ours": ef_npz["cfm"], "ef": ef_npz["ef"],
             "ocn": np.load(OCN_NPZ)["ocn"]}
    if os.path.exists(GLOBAL_VIT_NPZ):
        gv = np.load(GLOBAL_VIT_NPZ)
        g_arr["vit"] = np.sqrt(gv["sqerr"] / np.maximum(gv["count"], 1))
    g_rows = bin_stats(global_distance(), g_arr, GLOBAL_BANDS, GLOBAL_LABELS)

    hk = {}
    for name in ("waveflow", "vit", "ef", "ocn"):
        d = np.load(HK_NPZ.format(name))
        hk[name] = np.sqrt(d["sqerr"] / np.maximum(d["count"], 1))
    hk_arr = {"ours": hk["waveflow"], "vit": hk["vit"], "ef": hk["ef"],
              "ocn": hk["ocn"]}
    h_rows = bin_stats(hk_distance(), hk_arr, HK_BANDS, HK_LABELS)

    with open(os.path.join(OUT_DIR, "coastal_error_by_distance.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["domain", "band_km", "n_nodes", "median_distance_km",
                    "median_ours", "median_vit", "median_ef", "median_ocn",
                    "ratio_vit", "ratio_ef", "ratio_ocn"])
        for dom, rows in (("global", g_rows), ("hongkong", h_rows)):
            for r in rows:
                w.writerow([dom, r["band_km"], r["n_nodes"],
                            f'{r["median_distance_km"]:.3f}',
                            f'{r["median_ours"]:.4f}',
                            f'{r.get("median_vit", float("nan")):.4f}',
                            f'{r["median_ef"]:.4f}', f'{r["median_ocn"]:.4f}',
                            f'{r["ratio_vit"]:.3f}', f'{r["ratio_ef"]:.3f}',
                            f'{r["ratio_ocn"]:.3f}'])

    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 120 / 25.4))
    draw_row(axes[0][0], axes[0][1], g_rows,
             [("ours", "WaveFlow", C_OURS, "o"),
              ("vit", "WaveFlow-ViT (0.25$^\\circ$)", C_VIT, "D"),
              ("ef", "EarthFormer (0.25$^\\circ$)", C_EF, "s"),
              ("ocn", "OceanCastNet (0.5$^\\circ$)", C_OCN, "^")],
             "a", "b", "WaveFlow-ViT (0.25$^\\circ$) / WaveFlow")
    draw_row(axes[1][0], axes[1][1], h_rows,
             [("ours", "WaveFlow", C_OURS, "o"),
              ("vit", "WaveFlow-ViT", C_VIT, "D"),
              ("ef", "EarthFormer", C_EF, "s"),
              ("ocn", "OceanCastNet", C_OCN, "^")],
             "c", "d", "WaveFlow-ViT ($\\sim$1 km) / WaveFlow")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax, loc in ((axes[0][0], "lower right"), (axes[0][1], "upper right"),
                    (axes[1][1], "upper right")):
        ax.legend(frameon=False, loc=loc, handlelength=1.5,
                  borderaxespad=0.2)
    # the Hong Kong panel is too crowded for an inside legend, so it sits above
    axes[1][0].legend(frameon=False, loc="lower right",
                      bbox_to_anchor=(1.0, 1.015), ncol=2, handlelength=1.4,
                      columnspacing=1.0, borderaxespad=0.0)

    fig.subplots_adjust(wspace=0.36, hspace=0.46)
    fig.savefig(OUT_BASE + ".png", dpi=300)
    fig.savefig(OUT_BASE + ".pdf")
    print("saved:", OUT_BASE + ".png/.pdf")
    for tag, rows in (("global", g_rows), ("hongkong", h_rows)):
        print(f"-- {tag}")
        for r in rows:
            print(f'   {r["band_km"]:>9} km n={r["n_nodes"]:6d} '
                  f'ours={r["median_ours"]:.3f} '
                  f'ef={r["median_ef"]:.3f} ocn={r["median_ocn"]:.3f} '
                  f'ratios vit={r["ratio_vit"]:.2f} ef={r["ratio_ef"]:.2f} '
                  f'ocn={r["ratio_ocn"]:.2f}')


if __name__ == "__main__":
    main()
