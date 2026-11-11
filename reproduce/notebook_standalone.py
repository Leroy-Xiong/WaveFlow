"""Standalone, readable panel figures for the notebook.

The paper scripts use absolute-millimetre canvases. For the notebook we redraw
the simpler panels as normal matplotlib subplot grids, without pixel matching.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/figures"))
sys.path.insert(0, str(REPO / "scripts/figures/figure2"))


def _helpers():
    import plot_figure2 as F2
    return F2


def fig2a():
    F2 = _helpers()
    from shapely.geometry import box

    coords_hk = np.load(REPO / "data/hindcast/hk_coordinates.npy")
    coords_global = np.load(REPO / "data/global_hindcast/global_coordinates.npy")
    min_lon, max_lon, min_lat, max_lat = F2.HK_REGION
    m = ((coords_global[:, 1] >= min_lon) & (coords_global[:, 1] <= max_lon)
         & (coords_global[:, 0] >= min_lat) & (coords_global[:, 0] <= max_lat))
    coords_global_hk = coords_global[m]

    def regular(res):
        lons = np.arange(np.ceil(min_lon / res) * res, max_lon + res * 0.5, res)
        lats = np.arange(np.ceil(min_lat / res) * res, max_lat + res * 0.5, res)
        lon_g, lat_g = np.meshgrid(lons, lats)
        lo, la = lon_g.ravel(), lat_g.ravel()
        inside = ((lo >= min_lon) & (lo <= max_lon)
                  & (la >= min_lat) & (la <= max_lat))
        lo, la = lo[inside], la[inside]
        if res == 0.25:
            ocean = F2._ai025_ocean_mask()
            if ocean is not None:
                row = np.rint((la - 19.0) / 0.25).astype(int)
                col = np.rint((lo - 111.0) / 0.25).astype(int)
                keep = ocean[row, col]
                keep &= ~F2._land_away_mask(lo, la, "10m", 0.01, F2.HK_REGION)
                lo, la = lo[keep], la[keep]
            return np.column_stack([la, lo])
        return F2.mask_ocean(np.column_stack([la, lo]))

    coords_025 = regular(0.25)
    coords_0083 = regular(0.083)

    fig = plt.figure(figsize=(12.5, 5.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1], wspace=0.28, hspace=0.32)
    ax0 = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    ext = [110.0, 118.0, 18.0, 26.0]
    ax0.set_extent(ext, crs=ccrs.PlateCarree())
    ax0.add_geometries([box(*ext)], ccrs.PlateCarree(), facecolor="white",
                       edgecolor="none", zorder=1)
    land = F2._gshhs_land(ext)
    ax0.add_geometries(land, ccrs.PlateCarree(), facecolor="#e8e8e8",
                       edgecolor="none", zorder=2)
    ax0.add_geometries(land, ccrs.PlateCarree(), facecolor="none",
                       edgecolor="0.35", linewidth=0.4, zorder=3)
    raw_lons = np.arange(0.0, 360.0, 0.25)
    raw_lats = np.arange(90.0, -90.0 - 0.125, -0.25)
    e_lons = raw_lons[(raw_lons >= 111) & (raw_lons < 117)]
    e_lats = raw_lats[(raw_lats >= 19) & (raw_lats < 25)]
    lon_g, lat_g = np.meshgrid(e_lons, e_lats)
    ax0.scatter(lon_g.ravel(), lat_g.ravel(), c="#2166ac", s=12,
                transform=ccrs.PlateCarree(), zorder=4)
    ax0.plot([min_lon, max_lon, max_lon, min_lon, min_lon],
             [min_lat, min_lat, max_lat, max_lat, min_lat],
             color="#b2182b", lw=1.6, transform=ccrs.PlateCarree(), zorder=6)
    ax0.set_title("ERA5 forcing (0.25°, 24×24) + HK mesh box", fontsize=10)
    gl = ax0.gridlines(draw_labels=True, linewidth=0.3, color="0.5",
                       alpha=0.5, zorder=5)
    gl.top_labels = False
    gl.right_labels = False

    panels = [(coords_025, 26, "0.25° grid (AI wave model)"),
              (coords_0083, 7, "0.083° grid (numerical wave model)"),
              (coords_global_hk, 7, "Global unstructured mesh (ours)"),
              (coords_hk, 0.8, "HK unstructured mesh (ours)")]
    for i, (coords, s, title) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, 1 + i % 2], projection=ccrs.PlateCarree())
        ax.set_extent([min_lon, max_lon, min_lat, max_lat],
                      crs=ccrs.PlateCarree())
        ax.add_geometries([box(min_lon, min_lat, max_lon, max_lat)],
                          ccrs.PlateCarree(), facecolor="white",
                          edgecolor="none", zorder=1)
        hk_land = F2._gshhs_land([min_lon, max_lon, min_lat, max_lat])
        ax.add_geometries(hk_land, ccrs.PlateCarree(), facecolor="#e8e8e8",
                          edgecolor="none", zorder=2)
        ax.add_geometries(hk_land, ccrs.PlateCarree(), facecolor="none",
                          edgecolor="0.35", linewidth=0.4, zorder=3)
        ax.scatter(coords[:, 1], coords[:, 0], c="#b2182b", s=s,
                   transform=ccrs.PlateCarree(), zorder=4)
        ax.set_title(f"{title}\n{coords.shape[0]} nodes", fontsize=9)
        gl = ax.gridlines(draw_labels=False, linewidth=0.25, color="0.7",
                          alpha=0.5)
    fig.suptitle("WaveFlow target discretisations", fontsize=11)
    return fig
