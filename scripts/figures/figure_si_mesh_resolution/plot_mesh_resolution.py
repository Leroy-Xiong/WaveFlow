#!/usr/bin/env python
"""Supplementary figure: how the global unstructured mesh distributes its nodes.

Shows that WaveFlow's 651,652-node global ocean mesh is refined towards the
coast rather than being uniform: the median node spacing is 4.3 km within
10 km of the shoreline and rises to ~48 km beyond 500 km offshore, so 36% of
the nodes sit within 10 km of a coast and 56% within 25 km, where a 0.25 deg
grid would place 28 km between neighbours everywhere.

INPUTS (all read-only, nothing is regenerated here):
  data/global_hindcast/global_coordinates.npy    (N, 2) float, (lat, lon) deg
  data/global_hindcast/ocean_boundary_wkb.npz    cached node-derived coastline
                                                 ('coast_wkb', shapely WKB)

METHOD
  Node spacing  : great-circle distance to the nearest other node (cKDTree,
                  k=2) on a sphere of radius 6371.0088 km; chord distance is
                  used, which equals the arc distance for the separations here.
  Distance to coast : the coastline cached by modules/plot/global_ocean_boundary
                  is densified to 1 km before the nearest-coastline query, so
                  that long straight boundary segments do not inflate the
                  distance. No meridian seam segments were found in the cache
                  (checked: 0 of 1727 lines lie on lon = 0 or +/-180), so no
                  artificial shoreline is introduced.
  Sanity check  : 482,185 km of coastline x a 10 km band = 4.8e6 km^2 holding
                  234,901 nodes gives 4.5 km equivalent spacing, matching the
                  measured 4.3 km median.

OUTPUTS
  output/plot/figure_si_mesh_resolution/figureS_mesh_resolution.png  (dpi=300)
  output/plot/figure_si_mesh_resolution/figureS_mesh_resolution.pdf
  output/plot/figure_si_mesh_resolution/mesh_resolution_table.csv

DRAFT SI CAPTION
  Supplementary Fig. S1 | The global mesh concentrates its nodes at the coast.
  a, Median node spacing as a function of distance to the nearest coastline
  (solid line, shaded band: interquartile range), for the 651,652-node global
  ocean mesh. The dashed line marks the 28 km spacing of a uniform 0.25 deg
  grid. b, Share of mesh nodes in each distance band; 36% of the nodes lie
  within 10 km of a coastline and 56% within 25 km. Distances are great-circle
  distances to a coastline derived from the mesh boundary itself.

USAGE
  conda activate flow_and_diffusion
  python scripts/figures/figure_si_mesh_resolution/plot_mesh_resolution.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely.wkb
from scipy.spatial import cKDTree

R_EARTH_KM = 6371.0088
BANDS = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 500),
         (500, np.inf)]
BAND_LABELS = ["0-10", "10-25", "25-50", "50-100", "100-200", "200-500", ">500"]
UNIFORM_025_DEG_KM = 27.8          # meridional spacing of a 0.25 deg grid
COASTLINE_DENSIFY_KM = 1.0

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
COORD_PATH = os.path.join(ROOT, "data/global_hindcast/global_coordinates.npy")
COAST_PATH = os.path.join(ROOT, "data/global_hindcast/ocean_boundary_wkb.npz")
OUT_DIR = os.path.join(ROOT, "output/plot/figure_si_mesh_resolution")

# Nature-style rcParams (same block as scripts/figures/figure3/plot_figure3.py)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "TeX Gyre Heros",
                        "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

C_MAIN = "#1f5fa9"
C_BAND = "#9dc3e6"
C_BAR = "#c9d9ea"
C_REF = "0.35"


def to_xyz(lon_deg, lat_deg):
    """Unit-sphere-scaled Cartesian coordinates (km)."""
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return R_EARTH_KM * np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        axis=1)


def coastline_points():
    """Densified coastline vertices from the cached node-derived boundary."""
    cache = np.load(COAST_PATH, allow_pickle=True)
    geom = cache["coast_wkb"].item()
    if isinstance(geom, (bytes, bytearray, np.bytes_)):
        geom = shapely.wkb.loads(bytes(geom))
    parts, seam_lines = [], 0
    for line in geom.geoms:
        pts = np.asarray(line.coords)
        if np.all(np.abs(np.abs(pts[:, 0]) - 180.0) < 0.02) or \
           np.all(np.abs(pts[:, 0]) < 0.02):
            seam_lines += 1                      # 0/180 deg meridian artefacts
            continue
        parts.append(pts)
    if seam_lines:
        print(f"  dropped {seam_lines} meridian seam line(s)")
    dense = []
    for pts in parts:
        seg = np.linalg.norm(np.diff(to_xyz(pts[:, 0], pts[:, 1]), axis=0),
                             axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(int(cum[-1] / COASTLINE_DENSIFY_KM) + 1, 2)
        t = np.linspace(0.0, cum[-1], n)
        dense.append(np.column_stack([np.interp(t, cum, pts[:, 0]),
                                      np.interp(t, cum, pts[:, 1])]))
    return np.vstack(dense), sum(
        np.linalg.norm(np.diff(to_xyz(p[:, 0], p[:, 1]), axis=0),
                       axis=1).sum() for p in parts)


def analyse():
    coords = np.load(COORD_PATH)
    lat, lon = coords[:, 0], coords[:, 1]        # file stores (lat, lon)
    xyz = to_xyz(lon, lat)

    step, _ = cKDTree(xyz).query(xyz, k=2)
    spacing = step[:, 1]                          # km to nearest other node

    coast, coast_len = coastline_points()
    dist, _ = cKDTree(to_xyz(coast[:, 0], coast[:, 1])).query(xyz, k=1)

    rows = []
    for (lo, hi), label in zip(BANDS, BAND_LABELS):
        m = (dist >= lo) & (dist < hi)
        rows.append(dict(band=label, lo=lo,
                         hi=(None if np.isinf(hi) else hi),
                         n_nodes=int(m.sum()),
                         share_pct=100.0 * m.mean(),
                         median_spacing_km=float(np.median(spacing[m])),
                         mean_spacing_km=float(spacing[m].mean()),
                         q25_spacing_km=float(np.percentile(spacing[m], 25)),
                         q75_spacing_km=float(np.percentile(spacing[m], 75))))
    return spacing, dist, rows, coast_len


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    spacing, dist, rows, coast_len = analyse()
    print(f"nodes={len(spacing)}  coastline={coast_len:,.0f} km")
    for r in rows:
        print(f"  {r['band']:>8} km  n={r['n_nodes']:7d}  "
              f"share={r['share_pct']:5.1f}%  median={r['median_spacing_km']:.2f} km")

    with open(os.path.join(OUT_DIR, "mesh_resolution_table.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    x = np.array([0.5 * (r["lo"] + (r["hi"] if r["hi"] else 2 * r["lo"]))
                  for r in rows])
    med = np.array([r["median_spacing_km"] for r in rows])
    q25 = np.array([r["q25_spacing_km"] for r in rows])
    q75 = np.array([r["q75_spacing_km"] for r in rows])
    share = np.array([r["share_pct"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 62 / 25.4),
                             gridspec_kw=dict(width_ratios=[1.15, 1.0],
                                              wspace=0.32))
    ax = axes[0]
    ax.fill_between(x, q25, q75, color=C_BAND, alpha=0.55, lw=0,
                    label="interquartile range")
    ax.plot(x, med, "-o", color=C_MAIN, ms=3.0, label="median node spacing")
    ax.axhline(UNIFORM_025_DEG_KM, color=C_REF, ls=(0, (4, 2)), lw=0.9)
    ax.text(1.35, UNIFORM_025_DEG_KM * 1.10, "0.25\u00b0 grid (28 km)",
            fontsize=5.8, color=C_REF)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.2, 1200)
    ax.set_ylim(2.5, 90)
    ax.set_yticks([3, 5, 10, 20, 50])
    ax.set_yticklabels(["3", "5", "10", "20", "50"])
    ax.set_xlabel("distance to nearest coastline (km)")
    ax.set_ylabel("node spacing (km)")
    ax.legend(frameon=False, loc="lower right", handlelength=1.5)
    ax.set_title("refined towards the coast", fontsize=6.8, pad=3)
    ax.text(-0.16, 1.04, "a", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")

    ax = axes[1]
    pos = np.arange(len(rows))
    ax.bar(pos, share, color=C_BAR, edgecolor=C_MAIN, lw=0.6, width=0.72)
    for p, s in zip(pos, share):
        ax.text(p, s + 1.2, f"{s:.0f}", ha="center", va="bottom", fontsize=5.8,
                color="0.25")
    ax.set_xticks(pos)
    ax.set_xticklabels(BAND_LABELS, rotation=45, ha="right")
    ax.set_xlabel("distance to nearest coastline (km)")
    ax.set_ylabel("share of mesh nodes (%)")
    ax.set_ylim(0, max(share) * 1.22)
    ax.set_title("36% of nodes within 10 km of a coast", fontsize=6.8, pad=3)
    ax.text(-0.20, 1.04, "b", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"figureS_mesh_resolution.{ext}"),
                    dpi=300)
    print(f"wrote {OUT_DIR}/figureS_mesh_resolution.png / .pdf")


if __name__ == "__main__":
    main()
