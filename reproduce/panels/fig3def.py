#!/usr/bin/env python
"""Render panels d, e and f of main Fig. 3 (fixed patch partitions and the
patch-boundary field check).

This is a panel-only re-implementation of the corresponding blocks of
``scripts/figures/figure3/plot_figure3.py``. The original script draws all
panels at import time and needs data for panels a/b/c/g that are not part of
the release; this file repeats the d/e/f layout and drawing code verbatim and
is validated against the paper crop by ``reproduce.py verify``.

Data (all shipped): HK/global coordinates + clusters, the 10-member global
ensemble at 2022-09-24 00Z, the node-derived ocean geometry, and cartopy's
Natural Earth coastlines.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bootstrap  # noqa: E402

REPO = bootstrap.REPO

# ---------------------------------------------------------------------------
# Nature-style rcParams (same as the paper script)
# ---------------------------------------------------------------------------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors  # noqa: E402
import matplotlib.ticker  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "TeX Gyre Heros",
                        "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/figures"))
import typography as TY  # noqa: E402
TY.use_nature_fonts()

CANVAS_W, CANVAS_H = 180.0, 189.0
C_Y = 77.0
HS_CMAP = plt.get_cmap("turbo")

PATCH_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948",
    "#B07AA1", "#FF9DA7", "#86BCB6", "#F1CE63", "#D37295", "#8CD17D",
    "#A0CBE8", "#FFBE7D", "#B6992D", "#499894",
]

ZOOM_REGIONS = [
    ("Western North Pacific", (110.0, 138.0, 0.5, 28.5)),
    ("Caribbean Sea", (-95.0, -63.0, 0.7, 32.7)),
    ("Bay of Bengal", (76.0, 96.0, 0.1, 20.1)),
]
GLOBE_CENTERS = [(125.0, 14.0), (-79.0, 16.0), (86.0, 10.0), (0.0, 90.0)]
G_D = 30.0
G_XS = [49.0, 82.0, 115.0, 148.0]
G_Y = C_Y + 1.0

BCOL_XS = [4.0, 28.5, 53.0]
BCOL_W = 22.5
BCB_X, BCB_W = 76.5, 1.4
BROW_YS = [114.5, 139.0, 163.5]
BC_TS = "20220924T00"


def ax_mm(fig, x, y, w, h, **kwargs):
    return fig.add_axes([x / CANVAS_W, (CANVAS_H - y - h) / CANVAS_H,
                         w / CANVAS_W, h / CANVAS_H], **kwargs)


def panel_label(fig, x_mm, y_mm, label):
    fig.text(x_mm / CANVAS_W, (CANVAS_H - y_mm) / CANVAS_H, label,
             fontsize=9, fontweight="bold", va="top", ha="left")


def add_gray_land(ax, extent=None, scale="10m", coastline=True):
    """Light-grey land + thin coastline (figure-2 style, no satellite tiles)."""
    from shapely.geometry import box
    for s in (scale, "50m", "110m"):
        try:
            land = cfeature.NaturalEarthFeature("physical", "land", s)
            if extent is not None:
                clip = box(extent[0] - 1, extent[2] - 1,
                           extent[1] + 1, extent[3] + 1)
                proj = ccrs.PlateCarree()
                ax.add_geometries(
                    [g.intersection(clip) for g in land.geometries()], proj,
                    facecolor="#e8e8e8", edgecolor="none", zorder=1)
                if coastline:
                    coast = cfeature.NaturalEarthFeature(
                        "physical", "coastline", s)
                    ax.add_geometries(
                        [g.intersection(clip) for g in coast.geometries()],
                        proj, facecolor="none", edgecolor="0.35",
                        linewidth=0.4, zorder=5)
            else:
                ax.add_feature(land.with_scale(s), facecolor="#e8e8e8",
                               edgecolor="none", zorder=1)
                if coastline:
                    ax.add_feature(cfeature.COASTLINE.with_scale(s),
                                   edgecolor="0.35", linewidth=0.4, zorder=5)
            return
        except Exception:
            continue
    ax.add_feature(cfeature.LAND, facecolor="#e8e8e8", zorder=1)
    if coastline:
        ax.add_feature(cfeature.COASTLINE, edgecolor="0.35",
                       linewidth=0.4, zorder=5)


def patch_color_map(labels, coords_ll):
    """Greedy graph colouring of patches (same as the paper script)."""
    la, lo = np.deg2rad(coords_ll[:, 0]), np.deg2rad(coords_ll[:, 1])
    xyz = np.column_stack([np.cos(la) * np.cos(lo),
                           np.cos(la) * np.sin(lo), np.sin(la)])
    _, idx = cKDTree(xyz).query(xyz, k=9)
    src = np.repeat(np.arange(len(xyz)), 8)
    dst = idx[:, 1:].ravel()
    keep = labels[src] != labels[dst]
    adj = defaultdict(set)
    for a, b in zip(labels[src[keep]], labels[dst[keep]]):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    n = int(labels.max()) + 1
    pal = [matplotlib.colors.to_rgb(c) for c in PATCH_PALETTE]
    col = {}
    for p in sorted(range(n), key=lambda q: -len(adj[q])):
        used = {col[q] for q in adj[p] if q in col}
        c = next((i for i in range(len(pal)) if i not in used), None)
        if c is None:
            c = len(pal)
            pal.append(matplotlib.colors.to_rgb("#cccccc"))
        col[p] = c
    return np.asarray([pal[col[int(l)]] for l in labels])


def load_global_samples(samples_dir: Path, members: int = 10):
    files = [samples_dir / f"sample_{BC_TS}_ens{k:03d}.npz" for k in range(members)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(
            "missing global ensemble samples for Fig. 3f:\n  " +
            "\n  ".join(missing) +
            "\nrun `python reproduce.py setup --figure-data` first")
    acc = None
    for f in files:
        v = np.load(f)["sample"].astype(np.float64)
        if v.ndim == 2:
            v = v[:, 0]
        acc = v if acc is None else acc + v
    return acc / members


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--samples-dir", default=None,
                    help="global Hs samples for 2022-09-24 00Z (default: the "
                         "path stored in the figure-data archive)")
    args = ap.parse_args(argv)

    os.chdir(REPO)
    from modules.utils import HK_REGION

    samples_dir = args.samples_dir or os.environ.get(
        "WF_GLOBAL_HS_SAMPLES",
        "output/results/global/new_data_hs/"
        "patch2048_4frames_concat_wucos_ema999gc_xpred_huge/"
        "sample_epoch99_20260726_194530/samples")
    bc_hs = load_global_samples(Path(samples_dir))

    # ------------------------------------------------------------------
    # data for panels d/e/f
    # ------------------------------------------------------------------
    coords_ll = np.load("data/hindcast/hk_coordinates.npy")        # (N, 2)
    _z = np.load("data/hindcast/hk_clusters_512.npz")
    hk_labels = _z["labels"]
    hk_centers = _z["centers"][:, ::-1]                            # lon/lat
    hk_nodes = coords_ll[:, ::-1].copy()
    hk_rgb = patch_color_map(hk_labels, coords_ll)

    _gc = np.load("data/global_hindcast/global_coordinates.npy")
    _gz = np.load("data/global_hindcast/global_clusters_2048.npz")
    g_labels = _gz["labels"]
    g_centers = _gz["centers"][:, ::-1]
    g_lon, g_lat = _gc[:, 1], _gc[:, 0]
    g_rgb = patch_color_map(g_labels, _gc)
    from modules.plot.global_ocean_boundary import get_global_ocean_geometry
    g_ocean, g_coast = get_global_ocean_geometry()

    fig = plt.figure(figsize=(CANVAS_W / 25.4, CANVAS_H / 25.4))

    # ==================================================================
    # panel d: HK mesh, one colour per patch
    # ==================================================================
    panel_label(fig, 0.0, C_Y - 4.0, "d")
    HK_EXTENT = list(HK_REGION)
    BX0, BX1, BY0, BY1 = HK_EXTENT
    HK_W = 44.0
    HK_H = HK_W * (BY1 - BY0) / (BX1 - BX0)
    ax_hk = ax_mm(fig, 3.0, C_Y - 1, HK_W, HK_H, projection=ccrs.PlateCarree())
    ax_hk.set_extent(HK_EXTENT, crs=ccrs.PlateCarree())
    add_gray_land(ax_hk, HK_EXTENT, "10m")
    ax_hk.scatter(hk_nodes[:, 0], hk_nodes[:, 1], s=0.9, c=hk_rgb,
                  linewidths=0, zorder=5, transform=ccrs.PlateCarree())
    ax_hk.scatter(hk_centers[:, 0], hk_centers[:, 1], s=1.0, c="black",
                  linewidths=0, zorder=6, transform=ccrs.PlateCarree())

    # ==================================================================
    # panel e: global mesh on four orthographic globes
    # ==================================================================
    panel_label(fig, 48.0, C_Y - 4.0, "e")

    def draw_globe(ax):
        add_gray_land(ax, None, "110m", coastline=False)
        ax.add_geometries([g_coast], ccrs.PlateCarree(), facecolor="none",
                          edgecolor="0.35", linewidth=0.4, alpha=0.9,
                          zorder=4)
        ax.gridlines(lw=0.3, color="grey", alpha=0.6, linestyle="-", zorder=2)
        ax.scatter(g_lon, g_lat, s=0.1, c=g_rgb, linewidths=0, zorder=3,
                   rasterized=True, transform=ccrs.PlateCarree())
        ax.scatter(g_centers[:, 0], g_centers[:, 1], s=0.8, c="black",
                   linewidths=0, zorder=6, rasterized=True,
                   transform=ccrs.PlateCarree())

    globe_axes = []
    for (clon, clat), gx in zip(GLOBE_CENTERS, G_XS):
        ax_gl = ax_mm(fig, gx, G_Y, G_D, G_D,
                      projection=ccrs.Orthographic(clon, clat))
        draw_globe(ax_gl)
        globe_axes.append(ax_gl)
    for (_reg_name, (l0, l1, b0, b1)), ax_gl in zip(ZOOM_REGIONS, globe_axes):
        _t = np.linspace(0.0, 1.0, 60)
        _bx = np.concatenate([l0 + (l1 - l0) * _t, np.full_like(_t, l1),
                              l1 - (l1 - l0) * _t, np.full_like(_t, l0)])
        _by = np.concatenate([np.full_like(_t, b0), b0 + (b1 - b0) * _t,
                              np.full_like(_t, b1), b1 - (b1 - b0) * _t])
        ax_gl.plot(_bx, _by, color="black", lw=0.8, linestyle="--",
                   transform=ccrs.PlateCarree(), zorder=5)
    for ax_gl, gl_title in zip(globe_axes, ["W. N. Pacific", "Caribbean Sea",
                                            "Bay of Bengal", "Arctic"]):
        ax_gl.text(0.5, 1.02, gl_title, transform=ax_gl.transAxes,
                   ha="center", va="bottom", fontsize=6)

    # ==================================================================
    # panel f: 3 x 3 square-map block over the hotspot regions
    # ==================================================================
    panel_label(fig, 0.2, 108.8, "f")
    for _cx, _ct in zip(BCOL_XS, ["clusters", "Hs, nodes (m)",
                                  "Hs, interpolated (m)"]):
        fig.text((_cx + BCOL_W / 2) / CANVAS_W, (CANVAS_H - 113.0) / CANVAS_H,
                 _ct, ha="center", va="bottom", fontsize=5.5)

    for (reg_name, (l0, l1, b0, b1)), ry in zip(ZOOM_REGIONS, BROW_YS):
        rh = BCOL_W * (b1 - b0) / (l1 - l0)
        m = ((g_lon >= l0) & (g_lon <= l1) & (g_lat >= b0) & (g_lat <= b1))

        def _sat_base(ax):
            ax.set_extent([l0, l1, b0, b1], crs=ccrs.PlateCarree())
            add_gray_land(ax, (l0, l1, b0, b1), "50m", coastline=False)

        def _coast(ax, z=5):
            ax.add_geometries([g_coast], ccrs.PlateCarree(), facecolor="none",
                              edgecolor="black", linewidth=0.5, alpha=0.8,
                              zorder=z)

        ax_r = ax_mm(fig, BCOL_XS[0], ry, BCOL_W, rh,
                     projection=ccrs.PlateCarree())
        _sat_base(ax_r)
        ax_r.scatter(g_lon[m], g_lat[m], s=1.0, c=g_rgb[m], linewidths=0,
                     zorder=3, rasterized=True, transform=ccrs.PlateCarree())
        _mc = ((g_centers[:, 0] >= l0) & (g_centers[:, 0] <= l1)
               & (g_centers[:, 1] >= b0) & (g_centers[:, 1] <= b1))
        ax_r.scatter(g_centers[_mc, 0], g_centers[_mc, 1], s=1.2, c="black",
                     linewidths=0, zorder=5, transform=ccrs.PlateCarree())
        _coast(ax_r, z=4)
        fig.text(1.5 / CANVAS_W, (CANVAS_H - ry - rh / 2) / CANVAS_H, reg_name,
                 rotation=90, ha="center", va="center", fontsize=6)

        _vmin = float(np.floor(np.nanmin(bc_hs[m]) * 10) / 10)
        _vmax = float(np.ceil(np.nanmax(bc_hs[m]) * 10) / 10)

        ax_n = ax_mm(fig, BCOL_XS[1], ry, BCOL_W, rh,
                     projection=ccrs.PlateCarree())
        _sat_base(ax_n)
        ax_n.scatter(g_lon[m], g_lat[m], c=bc_hs[m], s=1.0, cmap=HS_CMAP,
                     vmin=_vmin, vmax=_vmax, edgecolors="none", zorder=3,
                     rasterized=True, transform=ccrs.PlateCarree())
        _coast(ax_n, z=4)

        ax_i = ax_mm(fig, BCOL_XS[2], ry, BCOL_W, rh,
                     projection=ccrs.PlateCarree())
        _sat_base(ax_i)
        nx_g = 420
        ny_g = max(2, int(round(nx_g * (b1 - b0) / (l1 - l0))))
        gx = np.linspace(l0, l1, nx_g)
        gy = np.linspace(b0, b1, ny_g)
        GX, GY = np.meshgrid(gx, gy)
        zi = griddata(np.column_stack([g_lon[m], g_lat[m]]), bc_hs[m],
                      (GX, GY), method="linear")
        from shapely import contains_xy
        zi[~contains_xy(g_ocean, GX.ravel(), GY.ravel()).reshape(GX.shape)] = np.nan
        _t = cKDTree(np.column_stack([g_lon[m], g_lat[m]]))
        _dd, _ = _t.query(np.column_stack([GX.ravel(), GY.ravel()]))
        _thr = 2.5 * np.median(_dd)
        zi[_dd.reshape(GX.shape) > _thr] = np.nan
        _im = ax_i.imshow(zi, extent=[l0, l1, b0, b1], origin="lower",
                          cmap=HS_CMAP, vmin=_vmin, vmax=_vmax,
                          transform=ccrs.PlateCarree(),
                          interpolation="bilinear", zorder=3)
        _cba = ax_mm(fig, BCB_X, ry, BCB_W, rh)
        _cbr = fig.colorbar(
            _im, cax=_cba, ticks=np.round(np.linspace(_vmin, _vmax, 4), 1),
            format=matplotlib.ticker.FormatStrFormatter("%.1f"))
        _cba.tick_params(labelsize=5.5, length=1.5, pad=0.7)
        _cbr.outline.set_linewidth(0.5)
        _coast(ax_i, z=4)

    fig.canvas.draw()
    TY.enforce_glyph_floor(fig)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor="white")
    print(f"[fig3def] saved full canvas {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
