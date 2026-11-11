"""
Draft Figure 2 for Nature Computational Science submission.

Theme: advantage of high-resolution irregular (unstructured) meshes.

Panels (absolute mm layout, Nature formatting):
  a) Multi-scale task setup: ERA5 forcing domain (0.25 deg, with the HK
     wave mesh domain marked) + zoom into the HK box showing four meshes
     at different resolutions (0.25 deg / 0.083 deg regular grids, global
     unstructured mesh, HK local unstructured mesh).
  b) HK local qualitative comparison (Typhoon Chaba, 2022-07-01 20UTC,
     GT Hs peak 5.29 m): 2 rows (Hs, water level elev) x 5 columns
     (GT / WaveFlow / WaveFlow-ViT (native 75x107 grid)
     / EarthFormer (24x24) / OceanCastNet (24x24)).
  c) Zoom into eastern Victoria Harbour: GT / WaveFlow / WaveFlow-ViT.
  d) Per-timestamp RMSE violin+box plots: HK Hs/Tp/Dm/Elev groups,
     WaveFlow vs baselines.
  e) Global Hs, one violin group: models vs hindcast GT.  (A second
     group vs real CMEMS L3 satellite along-track SWH observations —
     hourly-cadence inference matched to obs times, hindcast itself as
     reference — is implemented but disabled: the satellite comparison
     is used as a WaveFlow-only validation outside the main figure, not
     as a per-method benchmark.  Set INCLUDE_SAT_GROUP=True to restore.)
  (Global per-node spatial RMSE difference and global Kompasu case study
     are temporarily dropped from the layout; draw_panel_global_diff and
     draw_panel_global_case remain in the file.)

Outputs:
  output/plot/figure2/figure2_draft.pdf
  output/plot/figure2/figure2_draft.png (dpi=300)

Run:  python scripts/figures/figure2/plot_figure2.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from modules.utils import HK_REGION, ERA5_HK_LON_RANGE_2, ERA5_HK_LAT_RANGE_2

# ---------------------------------------------------------------------------
# Nature-style rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "TeX Gyre Heros", "DejaVu Sans"],
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

# 字体: Helvetica 系 (= Nature 的 Helvetica/Arial)。本机没有 Arial/Helvetica 时
# 自动用 TeX Live 里的 TeX Gyre Heros, 并让 mathtext($H_s$ 等)与正文同源;
# 必须在上面 rcParams.update 之后调用(它会改 font.sans-serif / mathtext.*)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import typography as TY
TY.use_nature_fonts()

OUT_DIR = "output/plot/figure2"

# 24x24 inference dirs of the 0.25-deg baselines (ERA5 region 2)
EF_INFER_DIR = ("earthformer_repro/output/results/ef_bu128_bs350_fixed/"
                "train_20260628_081221/inferences")
OCN_INFER_DIR = ("oceancastnet_repro/output/results/ocn_4ch_paper/"
                 "train_20260626_163344/inferences")
FIG_W = 180 / 25.4  # exactly 180 mm (Nature Portfolio max figure width)
FIG_H = 255 / 25.4  # 255 mm: lowest ink is the panel-f colorbar label
                    # (~254.2 mm); was 381 mm with ~124 mm dead space below

# Panel b HK case-study timestamp (Typhoon Chaba, Jul 2022 — GT Hs peak
# 5.29 m; bc3-calibrated WaveFlow hs RMSE 0.118 m / elev RMSE 0.106 m,
# beats ViT 0.317 / 0.164, EF 1.034 / 1.129; picked for the best water
# -level pattern match, not just the lowest Hs error)
CASE_HK_TS = "20220701T20"

# Panel e case-study timestamp (Typhoon Kompasu, South China Sea)
CASE_TS = "20211012T06"  # sample-file format
CASE_EXTENT = [105, 135, 10, 30]  # lon_min, lon_max, lat_min, lat_max

# panel-d model colours, aligned with figure 3's palette: ours takes
# figure 3's patch-model coral, the ViT ablation stays in the warm family
# (deep magenta), and the baselines are blues (fig-3 teal + dark navy)
COLOR_OURS = "#ED8D5A"      # WaveFlow — C_PATCH in figure 3
COLOR_VIT = "#C0547E"       # WaveFlow-ViT — deep magenta near ours
COLOR_EF = "#4198AC"        # EarthFormer — teal blue (fig-3 baseline blue)
COLOR_OCN = "#1F4E79"       # OceanCastNet — dark navy blue
COLOR_GT = "#b2182b"        # dark red, hindcast-vs-satellite reference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _land_feature(scale):
    return cfeature.NaturalEarthFeature(
        "physical", "land", scale, facecolor="#e8e8e8", edgecolor="none")


def _coastline_feature(scale):
    return cfeature.NaturalEarthFeature(
        "physical", "coastline", scale, facecolor="none",
        edgecolor="0.35", linewidth=0.4)


def add_map_features(ax, scale="10m", extent=None):
    """Add light-grey land and coastline, with fallback to coarser scales.

    When ``extent`` ([lon_min, lon_max, lat_min, lat_max]) is given, the
    geometries are clipped to the extent first, which keeps regional maps
    small in vector (PDF) output.
    """
    for s in (scale, "50m", "110m"):
        try:
            land = _land_feature(s)
            coast = _coastline_feature(s)
            if extent is not None:
                from shapely.geometry import box
                clip = box(extent[0] - 1, extent[2] - 1,
                           extent[1] + 1, extent[3] + 1)
                proj = ccrs.PlateCarree()
                land_geoms = [g.intersection(clip) for g in land.geometries()]
                coast_geoms = [g.intersection(clip)
                               for g in coast.geometries()]
                ax.add_geometries(land_geoms, proj, facecolor="#e8e8e8",
                                  edgecolor="none", zorder=2)
                ax.add_geometries(coast_geoms, proj, facecolor="none",
                                  edgecolor="0.35", linewidth=0.4, zorder=3)
            else:
                ax.add_feature(land, zorder=2)
                ax.add_feature(coast, zorder=3)
            return
        except Exception:
            continue
    ax.add_feature(cfeature.LAND, facecolor="#e8e8e8", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=3)


def latlon_grid(ax, xlocs, ylocs, left=False, right=False, bottom=False):
    """Light gridlines with 6-pt lat/lon tick labels on the requested sides.

    cartopy 0.25 + mpl 3.11: gridlines(draw_labels=True) wipes set_title,
    so axes using this must draw their titles with ax.text instead.
    """
    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color="0.5", alpha=0.5, zorder=5)
    # cartopy 0.25: the `zorder` kwarg only reaches the child line
    # collections, while the Gridliner artist itself defaults to zorder 2
    # and would be painted UNDER the scatter dots — lift the whole artist
    gl.set_zorder(6)
    gl.top_labels = False
    gl.left_labels = left
    gl.right_labels = right
    gl.bottom_labels = bottom
    gl.xlocator = mticker.FixedLocator(xlocs)
    gl.ylocator = mticker.FixedLocator(ylocs)
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {"size": 6}
    # latitude labels read bottom-to-top: much narrower than horizontal
    gl.ylabel_style = {"size": 6, "rotation": 90}
    # tuck the tick labels closer to the map frame (default pad is 5 pt)
    gl.xpadding = 2
    gl.ypadding = 2
    return gl


def panel_label(ax, label, y=1.14, x=-0.08):
    # 9 pt 粗体小写（作者指定：面板字母保持 9 pt，勿按 8 pt 规范改小）
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


# ---------------------------------------------------------------------------
# Absolute layout of panels a-c, in millimetres.  x is measured from the
# left edge of the 180-mm canvas, y from the top edge of the 257-mm
# canvas.  Every map box matches its data aspect exactly (no letterbox),
# so these numbers are literally what appears on paper; tweak them
# directly to move or resize a panel.
# ---------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 180.0, 255.0  # mm

# panel a: ERA5 overview (square) + 2x2 mesh maps (HK aspect 1.429)
A_ERA5 = dict(x=5.40, y=14.81, w=32.00, h=32.00)
A22_X = [42.72, 75.72]                    # 2.15 mm column gap
A22_Y = [8.31, 37.40]                     # 7.50 mm row gap (fits two-line titles)
A22_W, A22_H = 30.85, 21.59

# panel b: 5 model rows x 2 variable columns (HK aspect 1.429)
B_X = [116.84, 148.14]                    # 2 mm column gap
B_Y = [3.81, 25.82, 47.83, 69.84, 91.85]  # 1.5 mm row gap
B_W, B_H = 29.31, 20.51
# colorbar axes span their column's map box; colorbar() itself consumes
# ~5% for the extend triangle(s), so the rendered bar ends up shorter
B_CB = [dict(x=116.84, y=115.90, w=29.31, h=1.10),   # Hs
        dict(x=148.14, y=115.90, w=29.31, h=1.10)]   # water level

# panel c: 2 rows x 3 zoom maps (Victoria Harbour aspect 0.23/0.19=1.211)
C_X = [4.10, 35.46, 66.82]                # 2 mm column gap
C_Y = [67.13, 93.38]                      # 2 mm row gap
C_W, C_H = 29.36, 24.25
C_CB = [dict(x=98.18, y=67.13, w=1.10, h=24.25),   # Hs
        dict(x=98.18, y=93.38, w=1.10, h=24.25)]   # Hs error

# panel d: violin row, anchored in mm like the other panels (formerly
# row 1 of a figure-fraction gridspec; the mm box keeps the row put on
# the trimmed canvas — values are the exact cell position that the
# gridspec produced on the old 381-mm canvas)
D_CELL = dict(x=8.10, y=152.16, w=171.00, h=64.33)

# panel e: two stacked global Ida maps (2:1 aspect), flush to the left;
# titles on the left side, one horizontal colorbar under both maps
E_X, E_Y = 5.00, 190.00
E_W, E_H = 54.00, 27.00
E_ROW_GAP = 1.00
E_CB = dict(x=14.50, y=246.00, w=35.00, h=1.10)

# panel f: 5 cols x 2 rows of square Ida-zoom maps, hard 1.5 mm column
# gaps (2 mm row gaps), placed to the right of panel e; colorbars
# horizontal under the grid
F_X, F_Y = 63.00, 193.00
F_S, F_GAP = 22.00, 1.50   # column gap; row gap stays 2 mm (F_RGAP)
F_RGAP = 2.00
F_CB = [dict(x=73.00, y=244.00, w=35.00, h=1.10),    # Hs
        dict(x=134.00, y=244.00, w=35.00, h=1.10)]   # Hs error


def ax_mm(fig, x, y, w, h, projection=None):
    """Axes at an absolute mm position; y measured from the canvas top."""
    kw = {} if projection is None else {"projection": projection}
    return fig.add_axes([x / CANVAS_W, (CANVAS_H - y - h) / CANVAS_H,
                         w / CANVAS_W, h / CANVAS_H], **kw)


def mask_ocean(coords):
    """Keep only ocean points using Natural Earth land polygons."""
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import Point
    from shapely.prepared import prep

    for scale in ("10m", "50m", "110m"):
        try:
            shp = shpreader.natural_earth(
                resolution=scale, category="physical", name="land")
            geoms = [prep(g) for g in shpreader.Reader(shp).geometries()]
            break
        except Exception:
            continue
    else:
        return coords  # give up: return unfiltered

    mask = np.ones(len(coords), dtype=bool)
    for i, (lat, lon) in enumerate(coords):
        pt = Point(lon, lat)
        for g in geoms:
            if g.contains(pt):
                mask[i] = False
                break
    return coords[mask]


# Prepared Natural-Earth land polygons, shrunk inward by the requested
# distance.  A point inside a shrunk polygon is on land AND farther than
# that distance from the coastline — cells hugging the shore are kept.
# Land is judged from the real coastline, not the SCHISM model domain,
# so open sea outside the model boundary is never mistaken for land.
_land_shrink_cache = {}


def _land_away_mask(lons, lats, scale="10m", shrink_deg=0.01, clip=None):
    """mask[i] == True where the point is on land and > shrink_deg (deg)
    from the coast; sea points and shore-hugging points stay False.
    ``clip`` = (min_lon, max_lon, min_lat, max_lat) bbox limiting the
    land polygons to the view (plus a small pad)."""
    import cartopy.io.shapereader as shpreader
    import shapely
    from shapely.geometry import box
    from shapely.ops import unary_union

    key = (scale, shrink_deg, tuple(clip) if clip else None)
    if key not in _land_shrink_cache:
        try:
            shp = shpreader.natural_earth(
                resolution=scale, category="physical", name="land")
            geoms = list(shpreader.Reader(shp).geometries())
        except Exception:
            geoms = []
        if clip is not None:
            pad = 0.25
            b = box(clip[0] - pad, clip[2] - pad,
                    clip[1] + pad, clip[3] + pad)
            geoms = [g.intersection(b) for g in geoms]
        shrunk = []
        for g in geoms:
            if g.is_empty:
                continue
            h = g.buffer(-shrink_deg)
            if not h.is_empty:
                shrunk.append(h)
        # one union + vectorized contains: much faster than a per-point
        # loop over prepared polygons
        _land_shrink_cache[key] = unary_union(shrunk) if shrunk else None

    union = _land_shrink_cache[key]
    if union is None:
        return np.zeros(len(lons), dtype=bool)
    shapely.prepare(union)
    return shapely.contains(union, shapely.points(np.asarray(lons),
                                                  np.asarray(lats)))


def grey_background_hk(ax, extent):
    """Panel-a schematic style for the HK maps: white water, grey GSHHS
    full-resolution land, thin grey coastline.  Replaces the satellite-
    tile background in panels b and c."""
    min_lon, max_lon, min_lat, max_lat = extent
    from shapely.geometry import box
    ax.add_geometries([box(min_lon, min_lat, max_lon, max_lat)],
                      ccrs.PlateCarree(), facecolor="white",
                      edgecolor="none", zorder=1)
    land = _gshhs_land(extent)
    ax.add_geometries(land, ccrs.PlateCarree(), facecolor="#e8e8e8",
                      edgecolor="none", zorder=2)
    ax.add_geometries(land, ccrs.PlateCarree(), facecolor="none",
                      edgecolor="0.35", linewidth=0.4, zorder=5)


# GSHHS full-res land polygons (level 1), loaded once from the cartopy
# shapefile cache (~160 MB, ~1 min first time) and clipped per region.
_gshhs_all_land = None
_gshhs_clip_cache = {}


def _gshhs_all():
    global _gshhs_all_land
    if _gshhs_all_land is None:
        from cartopy.io import shapereader as shpreader
        _gshhs_all_land = list(shpreader.Reader(
            shpreader.gshhs(scale="f", level=1)).geometries())
    return _gshhs_all_land


def _gshhs_land(clip):
    """GSHHS full-res land polygons clipped to ``clip``
    (min_lon, max_lon, min_lat, max_lat)."""
    key = tuple(clip)
    if key not in _gshhs_clip_cache:
        from shapely.geometry import box
        b = box(clip[0], clip[2], clip[1], clip[3])
        _gshhs_clip_cache[key] = [g.intersection(b)
                                  for g in _gshhs_all() if g.intersects(b)]
    return _gshhs_clip_cache[key]


# ---------------------------------------------------------------------------
# Panel a: multi-scale task setup (ERA5 forcing domain vs HK wave mesh)
# ---------------------------------------------------------------------------

# Static 24x24 ocean mask of the ERA5 region-2 grid used by the
# EarthFormer / OceanCastNet baselines (identical for both models and for
# every timestamp).  Panel a reuses it so its 0.25-deg schematic shows
# exactly the cells contained in the panel-b baseline rows.
_AI025_OCEAN_MASK = None
_AI025_MASK_LOADED = False


def _ai025_ocean_mask():
    global _AI025_OCEAN_MASK, _AI025_MASK_LOADED
    if not _AI025_MASK_LOADED:
        _AI025_MASK_LOADED = True
        try:
            with np.load(f"{EF_INFER_DIR}/sample_{CASE_HK_TS}.npz") as d:
                _AI025_OCEAN_MASK = d["ocean_mask"] > 0
        except (OSError, KeyError):
            # Release override: the baseline reproduction repos are not part
            # of the release; the static 24x24 ERA5 region-2 ocean mask is
            # shipped as figure source data instead.
            _release_mask = "data/release_assets/ai025_ocean_mask.npy"
            try:
                _AI025_OCEAN_MASK = np.load(_release_mask) > 0
            except OSError:
                _AI025_OCEAN_MASK = None  # fall back to Natural Earth only
    return _AI025_OCEAN_MASK


def draw_panel_a(fig):
    min_lon, max_lon, min_lat, max_lat = HK_REGION
    era5_lon = ERA5_HK_LON_RANGE_2      # (111, 117)
    era5_lat = ERA5_HK_LAT_RANGE_2      # (19, 25)

    coords_hk = np.load("data/hindcast/hk_coordinates.npy")
    coords_global = np.load("data/global_hindcast/global_coordinates.npy")
    m = ((coords_global[:, 1] >= min_lon) & (coords_global[:, 1] <= max_lon)
         & (coords_global[:, 0] >= min_lat) & (coords_global[:, 0] <= max_lat))
    coords_global_hk = coords_global[m]

    def _make_regular(res):
        # snap to the real grid phase: 0.25° AI-model grids (ERA5 forcing,
        # EF/OCN outputs — cf. panel b) live on exact multiples of the
        # resolution, not on the HK_REGION corner
        lons = np.arange(np.ceil(min_lon / res) * res,
                         max_lon + res * 0.5, res)
        lats = np.arange(np.ceil(min_lat / res) * res,
                         max_lat + res * 0.5, res)
        lon_g, lat_g = np.meshgrid(lons, lats)
        lo, la = lon_g.ravel(), lat_g.ravel()
        if res == 0.25:
            # exactly the 24x24 cells drawn in panel b's baseline rows:
            # centres inside HK_REGION (panel b clips the 22.75 N row
            # too), the baseline's own static ocean_mask, and the same
            # land-away test.  Without this the schematic counted one
            # clipped node outside the box and kept/dropped a coastal
            # cell differently from panel b, so "N nodes" disagreed.
            inside = ((lo >= min_lon) & (lo <= max_lon)
                      & (la >= min_lat) & (la <= max_lat))
            lo, la = lo[inside], la[inside]
            ocean = _ai025_ocean_mask()
            if ocean is not None:
                row = np.rint((la - 19.0) / 0.25).astype(int)
                col = np.rint((lo - 111.0) / 0.25).astype(int)
                keep = ocean[row, col]
                keep &= ~_land_away_mask(lo, la, "10m", 0.01, HK_REGION)
                lo, la = lo[keep], la[keep]
                return np.column_stack([la, lo])
        return mask_ocean(np.column_stack([la, lo]))

    coords_025 = _make_regular(0.25)
    coords_0083 = _make_regular(0.083)

    # --- left: ERA5 forcing domain with the HK wave mesh domain marked ------
    ax0 = ax_mm(fig, **A_ERA5, projection=ccrs.PlateCarree())
    pad_lon, pad_lat = 1.5, 1.5
    ext = [era5_lon[0] - pad_lon, era5_lon[1] + pad_lon,
           era5_lat[0] - pad_lat, era5_lat[1] + pad_lat]
    ax0.set_extent(ext, crs=ccrs.PlateCarree())
    # GSHHS full-res land on white water (same schematic style as the
    # data panels)
    from shapely.geometry import box as _box
    ax0.add_geometries([_box(ext[0], ext[2], ext[1], ext[3])],
                       ccrs.PlateCarree(), facecolor="white",
                       edgecolor="none", zorder=1)
    _land0 = _gshhs_land(ext)
    ax0.add_geometries(_land0, ccrs.PlateCarree(), facecolor="#e8e8e8",
                       edgecolor="none", zorder=2)
    ax0.add_geometries(_land0, ccrs.PlateCarree(), facecolor="none",
                       edgecolor="0.35", linewidth=0.4, zorder=3)
    # ERA5 grid points: read the actual coordinates from a real ERA5 NPZ
    # and apply the dataset's region-2 crop (left-closed, right-open),
    # so the dots match the model input exactly.
    era5_npz = np.load("data/era5/npz_files/1970/era5_19700101T00.npz")
    raw_lons = era5_npz["_longitude"]
    raw_lats = era5_npz["_latitude"]
    e_lons = raw_lons[(raw_lons >= era5_lon[0]) & (raw_lons < era5_lon[1])]
    e_lats = raw_lats[(raw_lats >= era5_lat[0]) & (raw_lats < era5_lat[1])]
    e_lon_g, e_lat_g = np.meshgrid(e_lons, e_lats)
    ax0.scatter(e_lon_g.ravel(), e_lat_g.ravel(), c="#2166ac", s=4,
                marker="o", edgecolors="none", alpha=0.75,
                transform=ccrs.PlateCarree(), zorder=4)
    # HK wave mesh domain (red box)
    ax0.plot([min_lon, max_lon, max_lon, min_lon, min_lon],
             [min_lat, min_lat, max_lat, max_lat, min_lat],
             color="#b2182b", lw=1.2, transform=ccrs.PlateCarree(), zorder=6)
    # caption goes *below* the map, outside the axes
    ax0.text(0.5, -0.15,
             f"ERA5 forcing (0.25\u00b0, {len(e_lons)}\u00d7{len(e_lats)})\n"
             "red box: HK wave mesh",
             transform=ax0.transAxes, fontsize=6, va="top", ha="center")
    # cartopy 0.25 + mpl 3.11: gridlines(draw_labels=True) wipes
    # set_title -> draw titles manually with ax.text
    ax0.text(0.5, 1.02, "Atmospheric forcing",
             transform=ax0.transAxes, ha="center", va="bottom", fontsize=7)
    gl = ax0.gridlines(draw_labels=True, linewidth=0.3,
                       color="0.5", alpha=0.5, zorder=5)
    gl.set_zorder(6)  # Gridliner artist itself defaults to zorder 2
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(110, 119, 2))
    gl.ylocator = mticker.FixedLocator(np.arange(18, 27, 4))
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {"size": 6}
    gl.ylabel_style = {"size": 6, "rotation": 90}
    gl.xpadding = 2
    gl.ypadding = 2

    # --- right: zoom into the HK box, four meshes as a 2x2 matrix ---------
    panels = [
        # two-line full titles; the 2x2 block was moved down 2.5 mm so the
        # top-row titles clear the canvas top edge.  Bottom row: one-line
        # mesh name + "(ours)" on the second line — the unstructured node
        # meshes are what our model uses
        (coords_025, 40, "0.25\u00b0 regular grid\n(AI wave model)"),
        (coords_0083, 8, "0.083\u00b0 regular grid\n(numerical wave model)"),
        (coords_global_hk, 8, "Global unstructured mesh\n(ours)"),
        (coords_hk, 0.6, "HK unstructured mesh\n(ours)"),
    ]

    axes = [ax0]
    from shapely.geometry import box
    land_hk = _gshhs_land([min_lon, max_lon, min_lat, max_lat])
    for i, (coords, size, title) in enumerate(panels):
        ax = ax_mm(fig, A22_X[i % 2], A22_Y[i // 2], A22_W, A22_H,
                   projection=ccrs.PlateCarree())
        ax.set_extent([min_lon, max_lon, min_lat, max_lat],
                      crs=ccrs.PlateCarree())
        # same GSHHS full-res coastline as the data panels: white water,
        # grey land, thin grey coastline
        ax.add_geometries([box(min_lon, min_lat, max_lon, max_lat)],
                          ccrs.PlateCarree(), facecolor="white",
                          edgecolor="none", zorder=1)
        ax.add_geometries(land_hk, ccrs.PlateCarree(),
                          facecolor="#e8e8e8", edgecolor="none", zorder=2)
        ax.add_geometries(land_hk, ccrs.PlateCarree(),
                          facecolor="none", edgecolor="0.35",
                          linewidth=0.4, zorder=3)
        ax.scatter(coords[:, 1], coords[:, 0], c="#b2182b", s=size,
                   marker="o", edgecolors="none", alpha=0.85,
                   transform=ccrs.PlateCarree(), zorder=4)
        ax.text(0.03, 0.04, f"{coords.shape[0]} nodes",
                transform=ax.transAxes, fontsize=6, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          alpha=0.85, edgecolor="0.6", linewidth=0.4),
                zorder=5)
        # a's row gap is 4 mm, roomy enough for the bottom-row titles to
        # sit outside the maps again
        ax.text(0.5, 1.02, title, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7)
        latlon_grid(ax, [114.0], [22.2, 22.6],
                    left=(i % 2 == 0), bottom=(i // 2 == 1))
        axes.append(ax)
    # "a" label: pinned to the same absolute spot (level with the "b"
    # label, ~4 mm from the canvas top) regardless of ERA5 map shifts
    panel_label(axes[0], "a", y=1.338, x=-0.139)
    return axes


# ---------------------------------------------------------------------------
# Panel b: HK local qualitative comparison (Typhoon Chaba, 20220701T20)
# ---------------------------------------------------------------------------
def _grid24_fields(path, stats):
    """Load a 24x24 EF/OCN inference NPZ and denormalize hs / elev.

    The inference NPZs store *normalized* predictions (see
    earthformer_repro/scripts/visualize_hk.py):
      hs   = max(expm1(norm * hs_log_std + hs_log_mean), 0)   [m]
      elev = norm * elev_std + elev_mean                      [m]
    Land cells (ocean_mask == 0) are set to NaN.
    """
    d = np.load(path)
    pr = d["prediction"].astype(np.float64)
    ocean = d["ocean_mask"] > 0
    hs = np.maximum(np.expm1(pr[0] * stats["hs_log_std"]
                             + stats["hs_log_mean"]), 0.0)
    el = pr[4] * stats["elev_std"] + stats["elev_mean"]
    hs = np.where(ocean, hs, np.nan)
    el = np.where(ocean, el, np.nan)
    return hs, el


def _sanity(name, var, arr):
    """Print data range; abort on gross unit/scale mismatches."""
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    p995 = np.nanpercentile(np.abs(arr), 99.5)
    print(f"    {name:22s} {var:4s} min={lo:8.3f} max={hi:8.3f} "
          f"p99.5|.|={p995:7.3f}")
    if var == "hs" and not (0.0 <= lo and hi < 20.0):
        raise ValueError(f"{name} hs range [{lo}, {hi}] looks wrong (units?)")
    if var == "elev" and p995 > 5.0:
        raise ValueError(f"{name} elev p99.5 {p995} m looks wrong (units?)")


# Shared color scales for the HK case-study panels.  Both Hs vmax and
# elev vmin/vmax follow the GT field (0.25 m steps, see draw_panel_b);
# models outside the scale (e.g. the overpredicting 0.25-deg baselines)
# saturate — by design, the overprediction reads off the scale.
# Panel c (Victoria Harbour zoom) derives its own tighter vmax from the
# in-region data (see draw_panel_c).
HS_VMIN = 0.0
HS_CMAP = plt.get_cmap("turbo")
EL_LIM = 1.5
EL_CMAP = plt.get_cmap("RdBu_r")
# Panel b water-level maps use turbo (same sequential look as Hs);
# EL_CMAP stays RdBu_r for the diverging error maps (panels c and g).
EL_MAP_CMAP = plt.get_cmap("turbo")

# Zoom extent for panel c (eastern Victoria Harbour)
ZOOM_EXTENT = [114.15, 114.38, 22.20, 22.39]  # [min_lon, max_lon, min_lat, max_lat]


def _load_hk_case_fields(ts):
    """Load all HK case-study fields once (shared by panels b and c)."""
    min_lon, max_lon, min_lat, max_lat = HK_REGION
    print(f"  HK case fields (Typhoon Chaba {ts}):")

    # --- Ground truth: irregular HK nodes (15523 -> 7788 in HK_REGION) -----
    ds = xr.open_dataset(f"data/hindcast/processed/wave_{ts}.nc")
    sub = ds.where((ds.lon >= min_lon) & (ds.lon <= max_lon)
                   & (ds.lat >= min_lat) & (ds.lat <= max_lat), drop=True)
    f = {"gt_lon": sub.lon.values, "gt_lat": sub.lat.values,
         "gt_hs": sub.hs.values, "gt_el": sub.elev.values}
    ds.close()

    # --- WaveFlow: 10-member ensemble mean on the same 7788 nodes --------
    # base_v1 run (hk_cleaned), epoch-49 checkpoint, bc3-calibrated
    # samples (test_bc3).
    f["hk_coords"] = np.load("data/hindcast/hk_coordinates.npy")  # [lat, lon]
    ours = _ens_mean(
        "output/results/hk_cleaned/all_var/f8_blr5e-5_amp_stable_tod/"
        "size_sweep/base_v1_20260821_025917/curve_ep49/test_bc3/samples",
        ts, "sample")
    f["ours_hs"], f["ours_el"] = ours[:, 0], ours[:, 3]

    # --- WaveFlow-ViT: native 75x107 grid (NOT interpolated to nodes) ----------
    vit = _ens_mean(
        "output/results/hk_grid_cleaned/all_var/vit_patch4_hk2_f8_tod_large50/"
        "sample_epoch39_cleaned_20260809_231249/samples", ts, "sample")
    f["vit_hs"], f["vit_el"] = vit[0], vit[3]  # land cells are already NaN
    # Grid axes (configs/configs_grid.yaml: grid_shape [75, 107] over
    # HK_REGION).  Latitude orientation verified by correlating the ViT Hs
    # field with GT Hs binned onto this grid: as-is (row 0 = south, i.e.
    # ascending lat) gives corr = 0.986, flipud gives 0.24; land cells also
    # cluster at high row indices (northern mainland coast), matching the
    # 10m coastline.  -> lat is ascending S->N, no flip needed.
    f["vit_lons"] = np.linspace(min_lon, max_lon, 107)
    f["vit_lats"] = np.linspace(min_lat, max_lat, 75)

    # --- WaveFlow-ViT interpolated to the irregular nodes (for error maps) -----
    vit_nodes = _ens_mean(
        "output/results/hk_grid_cleaned/all_var/vit_patch4_hk2_f8_tod_large50/"
        "sample_epoch39_cleaned_20260809_231249/samples_nodes/samples",
        ts, "sample")
    f["vit_nodes_hs"] = vit_nodes[:, 0]

    # GT node order -> hk_coordinates order.  The filtered GT nodes are the
    # identical point set as hk_coordinates (exact NN match), but the order
    # is not guaranteed, so remap explicitly for node-wise error maps.
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([f["gt_lat"], f["gt_lon"]]))
    _, f["gt_idx_for_hk"] = tree.query(f["hk_coords"], k=1)

    # --- EarthFormer / OceanCastNet: native 24x24 grid (ERA5 region 2) -----
    # NOTE: these models were trained on the *old* WATLEV water-level data
    # (clipped at >= 0; see modules/datasets/update_watlev.py), so their
    # elev fields are never negative and are not directly comparable to the
    # new-WATLEV GT/WaveFlow/ViT elev in sign conventions.
    with open("data/hindcast/wave_data_stats.json") as fh:
        stats = json.load(fh)
    f["g_lons"] = np.arange(111, 117, 0.25)   # 111.00 .. 116.75
    f["g_lats"] = np.arange(19, 25, 0.25)     # 19.00 .. 24.75, ascending S->N
    f["ef_hs"], f["ef_el"] = _grid24_fields(
        f"{EF_INFER_DIR}/sample_{ts}.npz", stats)
    f["ocn_hs"], f["ocn_el"] = _grid24_fields(
        f"{OCN_INFER_DIR}/sample_{ts}.npz", stats)

    for name, hs, el in [("Reference", f["gt_hs"], f["gt_el"]),
                         ("WaveFlow", f["ours_hs"], f["ours_el"]),
                         ("WaveFlow-ViT", f["vit_hs"], f["vit_el"]),
                         ("EarthFormer", f["ef_hs"], f["ef_el"]),
                         ("OceanCastNet", f["ocn_hs"], f["ocn_el"])]:
        _sanity(name, "hs", hs)
        _sanity(name, "elev", el)
    return f


def draw_panel_b(fig, f):
    from modules.plot.satellite_background import _load_ocean_geometry

    min_lon, max_lon, min_lat, max_lat = HK_REGION
    extent = [min_lon, max_lon, min_lat, max_lat]
    gt_lon, gt_lat = f["gt_lon"], f["gt_lat"]
    gt_hs, gt_el = f["gt_hs"], f["gt_el"]
    hk_coords = f["hk_coords"]
    ours_hs, ours_el = f["ours_hs"], f["ours_el"]
    vit_hs, vit_el = f["vit_hs"], f["vit_el"]
    vit_lons, vit_lats = f["vit_lons"], f["vit_lats"]
    g_lons, g_lats = f["g_lons"], f["g_lats"]
    ef_hs, ef_el = f["ef_hs"], f["ef_el"]
    ocn_hs, ocn_el = f["ocn_hs"], f["ocn_el"]

    # Hs vmax from the GT peak (0.25 m steps, like the water-level scale)
    hs_vmin, hs_cmap = HS_VMIN, HS_CMAP
    hs_vmax = float(np.ceil(np.nanmax(gt_hs) / 0.25) * 0.25)
    # water-level column: asymmetric vmin/vmax from the GT 1st/99th
    # percentiles (0.25 m steps, not symmetric about 0) — robust to a few
    # near-coast outlier nodes; the tide range fills the whole scale and
    # models outside it (e.g. ViT outlier cells) extend.
    el_cmap = EL_MAP_CMAP
    el_vmin = float(np.floor(np.nanpercentile(gt_el, 1) / 0.25) * 0.25)
    el_vmax = float(np.ceil(np.nanpercentile(gt_el, 99) / 0.25) * 0.25)

    # satellite land background + white model ocean + model coastline,
    # same style as the random_samples_hk maps (SCHISM boundary polylines)
    ocean_geom, _ = _load_ocean_geometry(extent)

    # 5 rows (one per model) x 2 cols (Hs / water level) at absolute mm
    # positions (see the layout constants at the top of the file)
    row_labels = ["Reference", "WaveFlow", "WaveFlow-ViT",
                  "EarthFormer", "OceanCastNet"]
    col_titles = ["$H_s$ (m)", "Water level (m)"]

    axes = []
    for r in range(5):
        row_axes = []
        for c in range(2):
            ax = ax_mm(fig, B_X[c], B_Y[r], B_W, B_H,
                       projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            grey_background_hk(ax, extent)
            # gridlines wipe set_title (cartopy 0.25 bug) -> ax.text
            if r == 0:
                ax.text(0.5, 1.02, col_titles[c], transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7)
            if c == 0:
                # x=-0.17 (was -0.12): leave ~1.8 mm between the rotated
                # model name and the latitude tick labels, which used to
                # nearly touch at -0.12
                ax.text(-0.17, 0.5, row_labels[r], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=7)
            latlon_grid(ax, [114.0], [22.2, 22.6],
                        left=(c == 0), bottom=(r == 4))
            row_axes.append(ax)
        axes.append(row_axes)

    def _draw(r, c, kind, payload, vmin, vmax, cmap):
        ax = axes[r][c]
        if kind == "scatter":
            # GT and WaveFlow nodes: never dropped (the coastline itself
            # is derived from these SCHISM mesh nodes)
            lons, lats, vals = payload
            vals = np.ma.masked_invalid(vals)
            return ax.scatter(lons, lats, c=vals, s=0.5, cmap=cmap,
                              vmin=vmin, vmax=vmax, edgecolors="none",
                              transform=ccrs.PlateCarree(), zorder=4)
        # native regular grid -> scatter at cell centres (regular dots,
        # not interpolated to the irregular mesh); payload pre-filtered
        lons, lats, vals, size = payload
        vals = np.ma.masked_invalid(vals)
        return ax.scatter(lons, lats, c=vals,
                          s=size, cmap=cmap, vmin=vmin, vmax=vmax,
                          edgecolors="none", transform=ccrs.PlateCarree(),
                          zorder=4)

    def _grid_payload_model(lons, lats, vals, size):
        # model-coastline mask: drop a grid cell only when its entire
        # marker disc lies on land, i.e. the disc has no intersection
        # with the model ocean polygon (the same SCHISM node-boundary
        # geometry that draws the panel's black coastline — Natural Earth
        # land is NOT used here, so the mask matches the drawn coastline
        # exactly).  The disc radius is derived from the on-figure marker
        # size (scatter s = area in pt^2, converted through the axes
        # transform) so the mask matches the drawing.
        import shapely
        ax0 = axes[0][0]
        (x0, _), (x1, _) = (ax0.transData.transform((min_lon, min_lat)),
                            ax0.transData.transform((max_lon, min_lat)))
        (_, y0), (_, y1) = (ax0.transData.transform((min_lon, min_lat)),
                            ax0.transData.transform((min_lon, max_lat)))
        deg_per_px = max((max_lon - min_lon) / (x1 - x0),
                         (max_lat - min_lat) / (y1 - y0))
        r_deg = np.sqrt(size / np.pi) * (fig.dpi / 72.0) * deg_per_px
        lon_g, lat_g = np.meshgrid(lons, lats)
        lo, la = lon_g.ravel(), lat_g.ravel()
        v = np.array(vals).ravel()  # copy: never mutate the caller field
        # disc intersects the model ocean  <=>  distance(centre, ocean)
        # <= r_deg (the distance is 0 for centres inside the polygon);
        # vectorized shapely against the prepared polygon, no per-cell
        # buffer polygons
        shapely.prepare(ocean_geom)
        keep = shapely.distance(shapely.points(lo, la), ocean_geom) <= r_deg
        # kept cells the model's own ocean_mask left NaN: fill from the
        # surrounding kept cells so they do not punch white holes in the
        # map (only cells whose disc touches the model ocean are kept, so
        # the fill never paints dots on land)
        hole = keep & np.isnan(v)
        if hole.any():
            from scipy.interpolate import griddata
            src = keep & ~np.isnan(v)
            v[hole] = griddata((lo[src], la[src]), v[src],
                               (lo[hole], la[hole]), method="nearest")
        return lo[keep], la[keep], v[keep], size

    def _grid_payload(lons, lats, vals, size, land_buf):
        # Natural-Earth mask, kept for the EF/OCN rows
        lon_g, lat_g = np.meshgrid(lons, lats)
        lo, la, v = lon_g.ravel(), lat_g.ravel(), np.asarray(vals).ravel()
        keep = ~_land_away_mask(lo, la, "10m", land_buf, HK_REGION)
        return lo[keep], la[keep], v[keep], size

    # grid payloads carry a marker size: finer grid -> smaller
    land_buf = 0.01  # deg: EF/OCN rows still use the Natural-Earth mask
    rows = [
        ("scatter", (gt_lon, gt_lat, gt_hs), (gt_lon, gt_lat, gt_el)),
        ("scatter", (hk_coords[:, 1], hk_coords[:, 0], ours_hs),
                    (hk_coords[:, 1], hk_coords[:, 0], ours_el)),
        ("grid", _grid_payload_model(vit_lons, vit_lats, vit_hs, 0.5),
                 _grid_payload_model(vit_lons, vit_lats, vit_el, 0.5)),
        ("grid", _grid_payload(g_lons, g_lats, ef_hs, 55.0, land_buf),
                 _grid_payload(g_lons, g_lats, ef_el, 55.0, land_buf)),
        ("grid", _grid_payload(g_lons, g_lats, ocn_hs, 55.0, land_buf),
                 _grid_payload(g_lons, g_lats, ocn_el, 55.0, land_buf)),
    ]
    sc_hs = sc_el = None
    for r, (kind, hs_pl, el_pl) in enumerate(rows):
        sc_hs = _draw(r, 0, kind, hs_pl, hs_vmin, hs_vmax, hs_cmap)
        sc_el = _draw(r, 1, kind, el_pl, el_vmin, el_vmax, el_cmap)

    # Mark the panel-c zoom box (eastern Victoria Harbour) on the Hs map
    # of every row that panel c actually compares (GT / WaveFlow /
    # WaveFlow-ViT); the EF/OCN rows are not in panel c, so no box there
    z0, z1, z2, z3 = ZOOM_EXTENT
    for r in range(3):
        axes[r][0].plot([z0, z1, z1, z0, z0], [z2, z2, z3, z3, z2],
                        color="#00E5E5", lw=0.5, linestyle="--",
                        transform=ccrs.PlateCarree(), zorder=6)
    # "b" label: 3 mm left + 1 mm down from the default corner position
    # (axes are 29.31 x 20.51 mm)
    panel_label(axes[0][0], "b", y=0.98, x=-0.26)

    # short horizontal colorbars pinned just under the "114E" labels of
    # the last map row (absolute mm positions)
    cax_hs = ax_mm(fig, **B_CB[0])
    cbar_hs = fig.colorbar(sc_hs, cax=cax_hs, extend="max",
                           orientation="horizontal")
    # 全图第一条 hs 色标写全称 (首次出现), 其余 hs 色标保持 "$H_s$ (m)" 简称
    cbar_hs.set_label("Significant wave height $H_s$ (m)", fontsize=6,
                      labelpad=2.5)
    cax_el = ax_mm(fig, **B_CB[1])
    cbar_el = fig.colorbar(sc_el, cax=cax_el, extend="both",
                           orientation="horizontal")
    cbar_el.set_label("Water level (m)", fontsize=6, labelpad=2.5)
    for cb in (cbar_hs, cbar_el):
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_linewidth(0.5)
    return [ax for row in axes for ax in row]


# ---------------------------------------------------------------------------
# Panel c: zoom into eastern Victoria Harbour (GT / WaveFlow / WaveFlow-ViT)
# ---------------------------------------------------------------------------
def draw_panel_c(fig, f):
    z_min_lon, z_max_lon, z_min_lat, z_max_lat = ZOOM_EXTENT

    def _subset(lons, lats, vals):
        m = ((lons >= z_min_lon) & (lons <= z_max_lon)
             & (lats >= z_min_lat) & (lats <= z_max_lat))
        return lons[m], lats[m], vals[m]

    # ViT grid cells inside the zoom box, plus one extra cell beyond the
    # left / bottom frame edges: the axes clip those markers to partial
    # dots, so the frame edges show no blank strip (same look as the
    # GT / WaveFlow node maps)
    v_dx = f["vit_lons"][1] - f["vit_lons"][0]
    v_dy = f["vit_lats"][1] - f["vit_lats"][0]
    vm = ((f["vit_lons"] >= z_min_lon - v_dx)
          & (f["vit_lons"] <= z_max_lon))
    hm = ((f["vit_lats"] >= z_min_lat - v_dy)
          & (f["vit_lats"] <= z_max_lat))
    v_lons, v_lats = f["vit_lons"][vm], f["vit_lats"][hm]

    # 2 rows x 3 zoom maps at absolute mm positions (see the layout
    # constants at the top of the file)
    col_titles = ["Reference", "WaveFlow",
                  "WaveFlow-ViT"]

    axes = []
    for r in range(2):
        row = []
        for c in range(3):
            if r == 1 and c == 0:
                row.append(None)  # no map for GT - GT
                continue
            ax = ax_mm(fig, C_X[c], C_Y[r], C_W, C_H,
                       projection=ccrs.PlateCarree())
            ax.set_extent(ZOOM_EXTENT, crs=ccrs.PlateCarree())
            grey_background_hk(ax, ZOOM_EXTENT)
            # gridlines wipe set_title (cartopy 0.25 bug) -> ax.text
            if r == 0:
                ax.text(0.5, 1.02, col_titles[c], transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7)
            latlon_grid(ax, [114.2, 114.3], [22.25, 22.35],
                        left=(r == 0 and c == 0), bottom=(r == 1))
            row.append(ax)
        axes.append(row)

    def _scatter(ax, lons, lats, vals, size, vmin, vmax, cmap):
        if vals.ndim == 2:  # regular grid -> cell-centre dots; drop cells
            # whose whole marker disc lies on land, i.e. the disc has no
            # intersection with the model ocean polygon (the same
            # node-derived geometry as the black coastline — same recipe
            # as panel b's WaveFlow-ViT row, no Natural Earth)
            from modules.plot.satellite_background import _load_ocean_geometry
            import shapely
            # ocean geometry padded one grid cell past the left / bottom
            # frame edges, so the extra edge cells (which carry real
            # data there) can pass the disc test
            dx = lons[1] - lons[0] if len(lons) > 1 else 0.0
            dy = lats[1] - lats[0] if len(lats) > 1 else 0.0
            ocean_geom, _ = _load_ocean_geometry(
                [z_min_lon - dx, z_max_lon, z_min_lat - dy, z_max_lat])
            # disc radius from the on-figure marker size (scatter s = area
            # in pt^2) converted through the axes transform
            (x0, _), (x1, _) = (ax.transData.transform((z_min_lon, z_min_lat)),
                                ax.transData.transform((z_max_lon, z_min_lat)))
            (_, y0), (_, y1) = (ax.transData.transform((z_min_lon, z_min_lat)),
                                ax.transData.transform((z_min_lon, z_max_lat)))
            deg_per_px = max((z_max_lon - z_min_lon) / (x1 - x0),
                             (z_max_lat - z_min_lat) / (y1 - y0))
            r_deg = np.sqrt(size / np.pi) * (fig.dpi / 72.0) * deg_per_px
            lons, lats = np.meshgrid(lons, lats)
            lons, lats, vals = lons.ravel(), lats.ravel(), vals.ravel()
            shapely.prepare(ocean_geom)
            keep = (shapely.distance(shapely.points(lons, lats), ocean_geom)
                    <= r_deg)
            lons, lats, vals = lons[keep], lats[keep], vals[keep]
        # irregular nodes (GT / WaveFlow): never dropped — the coastline
        # itself is derived from these SCHISM mesh nodes
        vals = np.ma.masked_invalid(vals)
        return ax.scatter(lons, lats, c=vals, s=size, cmap=cmap,
                          vmin=vmin, vmax=vmax, edgecolors="none",
                          transform=ccrs.PlateCarree(), zorder=4)

    hk = f["hk_coords"]
    sc_hs = None
    # panel-c Hs scale: derived from the in-zoom data (GT + WaveFlow +
    # WaveFlow-ViT), 0.25 m steps — the zoom harbour is calmer than the
    # full region, so a dedicated vmax keeps the local contrast high
    _, _, z_gt = _subset(f["gt_lon"], f["gt_lat"], f["gt_hs"])
    _, _, z_ours = _subset(hk[:, 1], hk[:, 0], f["ours_hs"])
    z_vit = f["vit_hs"][np.ix_(hm, vm)]
    z_all = np.concatenate([z_gt, z_ours, z_vit[np.isfinite(z_vit)]])
    hs_vmax_c = float(np.ceil(np.nanmax(z_all) / 0.25) * 0.25)
    for c in range(3):
        if c == 0:
            hs_pl = _subset(f["gt_lon"], f["gt_lat"], f["gt_hs"]) + (5,)
        elif c == 1:
            hs_pl = _subset(hk[:, 1], hk[:, 0], f["ours_hs"]) + (5,)
        else:
            # WaveFlow-ViT zoom: larger dots than the node maps
            hs_pl = (v_lons, v_lats, f["vit_hs"][np.ix_(hm, vm)], 7)
        sc_hs = _scatter(axes[0][c], *hs_pl, HS_VMIN, hs_vmax_c, HS_CMAP)

    # --- error row: model - GT on the irregular GT nodes -------------------
    # WaveFlow is native on these nodes; WaveFlow-ViT uses its own node-interpolated
    # output (samples_nodes, KDTree interpolation from the sampling run).
    gt_hs_on_hk = f["gt_hs"][f["gt_idx_for_hk"]]
    err_ours = f["ours_hs"] - gt_hs_on_hk
    err_vit = f["vit_nodes_hs"] - gt_hs_on_hk
    # symmetric limit from the 99th percentile inside the zoom box
    _, _, eo = _subset(hk[:, 1], hk[:, 0], err_ours)
    _, _, ev = _subset(hk[:, 1], hk[:, 0], err_vit)
    p99 = np.nanpercentile(np.abs(np.concatenate([eo, ev])), 99)
    err_lim = float(min(2.0, max(0.5, np.ceil(p99 * 2) / 2)))
    sc_err = None
    for c, err in [(1, err_ours), (2, err_vit)]:
        lo, la, v = _subset(hk[:, 1], hk[:, 0], err)
        sc_err = _scatter(axes[1][c], lo, la, v, 5, -err_lim, err_lim,
                          EL_CMAP)

    # "c" label: same absolute x as the "a" label (0.952 mm from the
    # canvas left): (0.952 - 4.10) / 29.36 = -0.107
    panel_label(axes[0][0], "c", y=1.061, x=-0.107)

    # slim vertical colorbars on the right at absolute mm positions; the
    # strip between c and b leaves room for their text labels
    cax_hs = ax_mm(fig, **C_CB[0])
    cbar_hs = fig.colorbar(sc_hs, cax=cax_hs, extend="max")
    cbar_hs.set_label("$H_s$ (m)", fontsize=6)
    cax_err = ax_mm(fig, **C_CB[1])
    cbar_err = fig.colorbar(sc_err, cax=cax_err, extend="both")
    cbar_err.set_label("$H_s$ error (m)", fontsize=6)
    for cb in (cbar_hs, cbar_err):
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_linewidth(0.5)
    return [ax for row in axes for ax in row if ax is not None]


# ---------------------------------------------------------------------------
# Panel d: RMSE violin+box plots (one row: 4 HK variables + Global)
# ---------------------------------------------------------------------------
def _violin_group(ax, entries, title, ylabel=True):
    """Violin plot of per-timestamp Hs RMSE distributions.

    entries: list of (label, values | None, color); None reserves an
    empty slot (model still training)."""
    x = np.arange(len(entries))

    def _whisker_top(v):
        """Upper boxplot whisker (showfliers=False): Q3 + 1.5*IQR,
        clipped to the data max — same rule matplotlib uses."""
        vv = v[np.isfinite(v)]
        q1, q3 = np.percentile(vv, [25, 75])
        return min(vv.max(), q3 + 1.5 * (q3 - q1))

    # y range follows the boxplot whiskers, not the violin tails
    ylim = max((_whisker_top(v) for _, v, _ in entries if v is not None),
               default=None)
    if ylim is None:
        # every entry missing (CSVs not synced yet): draw an empty panel
        ax.set_xticks([])
        ax.set_title(title, pad=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return
    ylim *= 1.18
    for xi, (lab, vals, col) in enumerate(entries):
        if vals is None:
            continue  # reserved slot (model still training): leave empty
        # drop NaNs (timestamps where the variable is undefined, e.g.
        # elev spin-up): boxplot() silently draws nothing with NaN input
        vals = vals[np.isfinite(vals)]
        p = ax.violinplot(vals, positions=[xi], widths=0.7,
                          showextrema=False, showmedians=False)
        for b in p["bodies"]:
            b.set(facecolor=col, edgecolor="none", alpha=0.3, zorder=3)
        ax.boxplot(vals, positions=[xi], widths=0.16, showfliers=False,
                   patch_artist=True, zorder=4,
                   boxprops=dict(facecolor=col, edgecolor="0.15",
                                 linewidth=0.8),
                   medianprops=dict(color="0.15", linewidth=1.0),
                   whiskerprops=dict(color="0.15", linewidth=0.8),
                   capprops=dict(color="0.15", linewidth=0.8))
    # no per-group model names: models are color-coded, one shared legend
    # row under panel d explains the colors
    ax.set_xticks([])
    if ylabel:
        ax.set_ylabel("RMSE")
    ax.set_title(title, pad=3)
    ax.set_ylim(0, ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=6)


# Satellite-obs violin group in panel e: disabled in the main figure
# (obs-space ranking conflates the hindcast's own obs bias with model
# skill; the comparison is kept for a WaveFlow-only validation figure).
INCLUDE_SAT_GROUP = False


def draw_panel_d(fig, cell_mm):
    hk_csvs = [
        ("WaveFlow",
         "output/results/hk_cleaned/all_var/f8_blr5e-5_amp_stable_tod/"
         "size_sweep/base_v1_20260821_025917/curve_ep49/test_bc3/"
         "ensemble_rmse_results_10.csv",
         "hs_rmse", COLOR_OURS),
        ("WaveFlow-ViT",
         "output/results/hk_grid_cleaned/all_var/vit_patch4_hk2_f8_tod_large50/"
         "sample_epoch39_cleaned_20260809_231249/samples_nodes/"
         "ensemble_rmse_results_10.csv",
         "hs_rmse", COLOR_VIT),
        # WGNN dropped: graph-based method, this panel compares models
        # whose outputs live on regular grids
        ("EarthFormer",
         "earthformer_repro/output/results/ef_bu128_bs350_fixed/"
         "train_20260628_081221/per_sample_rmse.csv",
         "hs_rmse", COLOR_EF),
        ("OceanCastNet",
         "oceancastnet_repro/output/results/ocn_4ch_paper/"
         "train_20260626_163344/per_sample_rmse.csv",
         "hs_rmse", COLOR_OCN),
    ]
    glob_csvs = [
        ("WaveFlow",
         "output/results/global/new_data_hs/"
         "patch2048_4frames_concat_wucos_ema999gc_xpred_huge/"
         "sample_epoch99_20260726_194530/ensemble_rmse_results_10.csv",
         "hs_rmse", COLOR_OURS),
        ("WaveFlow-ViT",
         "output/results/global/new_data_hs_grid/p16_vit_huge_xpred/"
         "sample_epoch99_20260801_000131/ensemble_rmse_results_10_nodes.csv",
         "rmse", COLOR_VIT),
        ("EarthFormer",
         "earthformer_repro/output/results/global_hs/"
         "train_20260804_120424/comparison_cfm/compare_node_rmse.csv",
         "ef_rmse", COLOR_EF),
        ("OceanCastNet\n0.5\u00b0",
         "oceancastnet_repro/output/results/global_swh/"
         "train_b192_20260803_113451/comparison_cfm/compare_node_rmse.csv",
         # same color as HK OCN: one color per model for the shared legend
         "ocn_rmse", COLOR_OCN),
    ]
    # Global vs real satellite observations (CMEMS L3 along-track SWH):
    # hourly-cadence inference matched to obs times (see
    # scripts/figures/figure2/run_satellite_eval_global.py), plus the
    # hindcast itself as reference (the reanalysis-vs-observation gap)
    sat_csvs = [
        ("WaveFlow",
         "output/results/global/new_data_hs/"
         "patch2048_4frames_concat_wucos_ema999gc_xpred_huge/"
         "sample_epoch99_hourly/satellite_eval/global_waveflow_hs_satellite.csv",
         "rmse", COLOR_OURS),
        ("WaveFlow-ViT",
         "output/results/global/new_data_hs_grid/p16_vit_huge_xpred/"
         "sample_epoch99_hourly/satellite_eval/global_vit_hs_satellite.csv",
         "rmse", COLOR_VIT),
        ("EarthFormer",
         "earthformer_repro/output/results/global_hs/"
         "train_20260804_120424/inference_hourly/satellite_eval/"
         "global_ef_hs_satellite.csv",
         "rmse", COLOR_EF),
        ("OceanCastNet\n0.5\u00b0",
         "oceancastnet_repro/output/results/global_swh/"
         "train_b192_20260803_113451/inference_hourly/satellite_eval/"
         "global_ocn_hs_satellite.csv",
         "rmse", COLOR_OCN),
        ("Hindcast",
         "output/results/global/satellite_eval/"
         "global_hindcast_hs_satellite.csv",
         "rmse", COLOR_GT),
    ]

    def _series(csv_path, col):
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} missing, entry skipped")
            return None
        # dropna: some hourly inference runs miss a few hundred timestamps
        return pd.read_csv(csv_path)[col].dropna().to_numpy(dtype=np.float64)

    # HK + Global panels: one row of violin groups (4 HK variables,
    # then Global Hs vs hindcast GT, plus an optional Global Hs vs
    # satellite obs group when INCLUDE_SAT_GROUP is enabled)
    hk_vars = [("hs_rmse", "$H_s$ (m)"), ("tp_rmse", "Tp (s)"),
               ("dm_rmse", "Dm (\u00b0)"), ("elev_rmse", "Elev (m)")]
    # shrink the whole six-group row slightly inside its cell
    # (subgridspec() rejects left/right in mpl 3.11 -> manual add_axes)
    cell = Bbox.from_extents(
        cell_mm["x"] / CANVAS_W,
        (CANVAS_H - cell_mm["y"] - cell_mm["h"]) / CANVAS_H,
        (cell_mm["x"] + cell_mm["w"]) / CANVAS_W,
        (CANVAS_H - cell_mm["y"]) / CANVAS_H)
    # whole panel-d/e violin row shifted 5 mm left from its gridspec cell
    dx = -5.0 / CANVAS_W
    x0 = cell.x0 + 0.045 * cell.width + dx
    x1 = cell.x1 - 0.0 * cell.width + dx
    # absolute 29 mm upward shift of the whole panel-d block (30 mm made
    # the panel-e group titles collide with the panel-b colorbar labels)
    dy = 29.0 / CANVAS_H
    y0 = cell.y0 + 0.08 * cell.height + dy
    y1 = cell.y1 - 0.08 * cell.height + dy
    # the "d"/"e" labels stay at their previous absolute positions while
    # the row moves up: compensate the extra 4 mm in axes coordinates
    lbl_keep = 4.0 / ((y1 - y0) * CANVAS_H)
    n_col = 6 if INCLUDE_SAT_GROUP else 5
    ws = 0.38
    # extra 4 mm gap before the global group (k>=4): HK (panel d) and
    # global (panel e) are different regions, so they get visual separation
    gap_glob = 4.0 / CANVAS_W
    col_w = (x1 - x0 - gap_glob) / (n_col + (n_col - 1) * ws)

    def _col(k):
        return fig.add_axes([x0 + k * (1 + ws) * col_w
                             + (gap_glob if k >= 4 else 0.0),
                             y0, col_w, y1 - y0])
    axes = []
    for k, (col_name, var_lab) in enumerate(hk_vars):
        ax = _col(k)
        entries = [(lab, _series(p, col_name), col)
                   for lab, p, c, col in hk_csvs]
        _violin_group(ax, entries, var_lab, ylabel=(k == 0))
        axes.append(ax)
    # "d" label: same absolute x as the "a" label (0.952 mm from the
    # canvas left), computed from the shifted row origin
    # "d" label: same absolute x as the "a" label (0.952 mm from the
    # canvas left), computed from the shifted row origin; 3.5 mm below the
    # label baseline shared with "e"
    panel_label(axes[0], "d", x=(0.952 / CANVAS_W - x0) / col_w,
                y=1.14 - lbl_keep - 3.5 / ((y1 - y0) * CANVAS_H))

    glob_entries = [(lab, _series(p, c), col) for lab, p, c, col in glob_csvs]
    ax2 = _col(4)
    _violin_group(ax2, glob_entries, "$H_s$ (m)",
                  ylabel=False)
    # the global group forms panel "e" (panels f/g follow); label sits at
    # group-title height, left of the group, clearing the panel-b
    # colorbars above; same height as the "d" label, 7 mm to the right
    panel_label(ax2, "e", x=-0.55 + 7.0 / (col_w * CANVAS_W),
                y=1.14 - lbl_keep - 3.5 / ((y1 - y0) * CANVAS_H))

    ax3 = None
    if INCLUDE_SAT_GROUP:
        sat_entries = [(lab, _series(p, c), col) for lab, p, c, col in sat_csvs]
        ax3 = _col(5)
        _violin_group(ax3, sat_entries, "$H_s$ (m) — Global\nvs satellite obs",
                      ylabel=False)
    else:
        sat_entries = []

    # one shared legend row under panel d: color -> model
    legend_models = [("WaveFlow", COLOR_OURS),
                     ("WaveFlow-ViT", COLOR_VIT),
                     ("EarthFormer", COLOR_EF),
                     ("OceanCastNet", COLOR_OCN)]
    if INCLUDE_SAT_GROUP:
        legend_models.append(("Hindcast", COLOR_GT))
    handles = [Patch(facecolor=c, edgecolor="0.15", linewidth=0.8,
                     label=lab)
               for lab, c in legend_models]
    p0 = axes[0].get_position()
    p4 = (ax3 if ax3 is not None else ax2).get_position()
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False,
               fontsize=7, handlelength=1.2, columnspacing=1.4,
               bbox_to_anchor=((p0.x0 + p4.x1) / 2.0, p0.y0 - 0.004))

    for lab, p, c, _ in hk_csvs:
        v = _series(p, c)
        print(f"  [HK]     {lab.replace(chr(10), ' '):24s} "
              f"{v.mean():.3f} +/- {v.std():.3f} m")
    for lab, v, _ in glob_entries:
        if v is None:
            print(f"  [Global] {lab:<24s} (missing)")
        else:
            print(f"  [Global] {lab.replace(chr(10), ' '):24s} "
                  f"{v.mean():.3f} +/- {v.std():.3f} m")
    for lab, v, _ in sat_entries:
        if v is None:
            print(f"  [Sat]    {lab:<24s} (missing)")
        else:
            print(f"  [Sat]    {lab.replace(chr(10), ' '):24s} "
                  f"{v.mean():.3f} +/- {v.std():.3f} m")
    return axes + [ax2] + ([ax3] if ax3 is not None else [])


# ---------------------------------------------------------------------------
# Panel g: strongest tropical cyclone in the test set (GT vs WaveFlow, global)
# ---------------------------------------------------------------------------
# Timestamp selection: Hurricane Ida approaching its Louisiana landfall
# (catalogue case #8: basin p99.9 Hs = 10.1 m, storm centre
# (29.2 N, 87.2 W), landfall near Port Fourchon, LA).  Single-node maxima
# are excluded: the hindcast contains isolated coastal artefacts
# (e.g. 72.8 m at 59.9 N, 170.4 E on 20210104).
CASE_STORM_TS = "20210829T12"
STORM_EXTENT = [-90.46, -88.54, 28.54, 30.46]  # zoom: Hs peak off the Mississippi delta


def _interp_nodes_to_grid(lon, lat, vals, extent, res=0.02, max_dist=0.55):
    """Linear-interpolate unstructured node values onto a regular lon/lat
    grid; pixels farther than max_dist (deg) from the nearest node are
    masked (keeps land and data gaps empty)."""
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    ok = np.isfinite(vals)
    pts = np.column_stack([lon[ok], lat[ok]])
    glon = np.arange(extent[0], extent[1] + 0.5 * res, res)
    glat = np.arange(extent[2], extent[3] + 0.5 * res, res)
    LON, LAT = np.meshgrid(glon, glat)
    z = griddata(pts, vals[ok], (LON, LAT), method="linear")
    d, _ = cKDTree(pts).query(np.column_stack([LON.ravel(), LAT.ravel()]))
    z = np.ma.masked_where(d.reshape(LON.shape) > max_dist, z)
    return glon, glat, z


def _grey_bg_zoom(ax):
    """White background + Natural Earth 10m grey land for the zoomed storm
    region (panel-a schematic style, no satellite tiles)."""
    from shapely.geometry import box
    ax.add_geometries([box(STORM_EXTENT[0], STORM_EXTENT[2],
                           STORM_EXTENT[1], STORM_EXTENT[3])],
                      ccrs.PlateCarree(), facecolor="white",
                      edgecolor="none", zorder=1)
    add_map_features(ax, scale="10m", extent=STORM_EXTENT)


def draw_panel_e(fig, coords):
    ts = CASE_STORM_TS
    ymd, hh = ts[:8], ts[9:11]
    year = ymd[:4]

    gt = np.load(f"data/global_hindcast/npz_files/{year}/"
                 f"global_hindcast_{ymd}_{hh}.npz")["hs"].astype(np.float64)
    cfm = _ens_mean(
        "output/results/global/new_data_hs/"
        "patch2048_4frames_concat_wucos_ema999gc_xpred_huge/"
        "sample_epoch99_20260726_194530/samples", ts, "sample")
    # Native regular-grid outputs (not interpolated to the ocean nodes):
    # WaveFlow-ViT at 0.25 deg (720x1440), OceanCastNet at 0.5 deg (360x720).
    # Grid convention (modules/grid_interpolation.py): lat north->south
    # from 90, lon 0->360, cell centres.
    vit_g = _ens_mean(
        "output/results/global/new_data_hs_grid/p16_vit_huge_xpred/"
        "sample_epoch99_20260801_000131/samples", ts, "sample")[0]
    vit_lats = np.linspace(90.0, -89.75, 720)
    vit_lons = np.linspace(0.0, 359.75, 1440)
    vit_lons = np.where(vit_lons > 180.0, vit_lons - 360.0, vit_lons)
    ocn_g = np.load(
        "oceancastnet_repro/output/results/global_swh/"
        "train_b192_20260803_113451/inference/samples/"
        f"sample_{ts}.npz")["prediction"][0].astype(np.float64)
    # OCN inference files are z-score normalised -> back to metres
    st = json.load(open("data/norm_stats_0.5deg.json"))["target_var"]["swh"]
    ocn_g = ocn_g * st["std"] + st["mean"]
    gt_g = np.load(f"data/global_hindcast/interpolated_grid_0.5deg/"
                   f"{year}/global_hindcast_{ymd}_{hh}.npz")["data"][0]
    ocn_g = np.where(np.isfinite(gt_g), ocn_g, np.nan)  # mask land cells
    ocn_lats = np.linspace(90.0, -89.5, 360)
    ocn_lons = np.linspace(0.0, 359.5, 720)
    ocn_lons = np.where(ocn_lons > 180.0, ocn_lons - 360.0, ocn_lons)
    # Node-interpolated outputs for the error row, which is drawn on the
    # irregular ocean nodes (same convention as panel c).
    vit_n = _ens_mean(
        "output/results/global/new_data_hs_grid/p16_vit_huge_xpred/"
        "sample_epoch99_20260801_000131/samples_nodes", ts, "sample")
    ocn_n = np.load(
        "oceancastnet_repro/output/results/global_swh/"
        "train_b192_20260803_113451/inference_nodes/samples/"
        f"sample_{ts}.npz")["prediction"].astype(np.float64)
    # EarthFormer global: native 0.25 deg grid (same convention as the ViT
    # grid) plus node-interpolated output; both already in physical metres
    ef = np.load(
        "earthformer_repro/output/results/global_hs/"
        "train_20260804_120424/inference/samples/"
        f"sample_{ts}.npz")
    ef_g = ef["prediction"].astype(np.float64)
    ef_g = np.where(ef["ocean_mask"], ef_g, np.nan)  # mask land cells
    ef_lats = np.linspace(90.0, -89.75, 720)
    ef_lons = np.linspace(0.0, 359.75, 1440)
    ef_lons = np.where(ef_lons > 180.0, ef_lons - 360.0, ef_lons)
    ef_n = np.load(
        "earthformer_repro/output/results/global_hs/"
        "train_20260804_120424/inference_nodes/samples/"
        f"sample_{ts}.npz")["prediction"].astype(np.float64)

    lats, lons = coords[:, 0], coords[:, 1]
    e = STORM_EXTENT
    mz_in = ((lons >= e[0]) & (lons <= e[1])
             & (lats >= e[2]) & (lats <= e[3]))

    def _grid_payload(lons1d, lats1d, vals, size):
        # model-coastline mask, same recipe as panel b's WaveFlow-ViT row:
        # drop a grid cell only when its entire marker disc lies on land,
        # i.e. the disc has no intersection with the node-derived ocean
        # polygon (the same geometry that draws these maps' black
        # coastline — Natural Earth is NOT used).  The disc radius comes
        # from the on-figure marker size (scatter s = area in pt^2): the
        # maps are F_S mm squares spanning (e[1]-e[0]) degrees, so
        # fig.dpi cancels out of the conversion.
        import shapely
        from modules.plot.global_ocean_boundary import (
            get_global_ocean_geometry)
        ocean_geom, _ = get_global_ocean_geometry()
        r_deg = (np.sqrt(size / np.pi)
                 * (e[1] - e[0]) * 25.4 / (72.0 * F_S))
        lg, ag = np.meshgrid(lons1d, lats1d)
        # keep points up to one marker radius outside the view: the axes
        # patch clips them to partial (half) markers at the frame
        mg = ((lg >= e[0] - r_deg) & (lg <= e[1] + r_deg)
              & (ag >= e[2] - r_deg) & (ag <= e[3] + r_deg))
        lo, la, v = lg[mg], ag[mg], vals[mg]
        # disc intersects the model ocean  <=>  distance(centre, ocean)
        # <= r_deg (0 for centres inside the polygon)
        shapely.prepare(ocean_geom)
        keep = shapely.distance(shapely.points(lo, la), ocean_geom) <= r_deg
        return lo[keep], la[keep], np.ma.masked_invalid(v[keep]), size

    fields = [
        ("Reference", "nodes", (lons[mz_in], lats[mz_in],
                                   np.ma.masked_invalid(gt[mz_in]), 3.0)),
        ("WaveFlow", "nodes", (lons[mz_in], lats[mz_in],
                                           np.ma.masked_invalid(cfm[mz_in]),
                                           3.0)),
        ("WaveFlow-ViT", "grid",
         _grid_payload(vit_lons, vit_lats, vit_g, 30.0)),
        ("EarthFormer", "grid",
         _grid_payload(ef_lons, ef_lats, ef_g, 30.0)),
        ("OceanCastNet (0.5\u00b0)", "grid",
         _grid_payload(ocn_lons, ocn_lats, ocn_g, 120.0)),
    ]
    vmin, vmax = 0.0, 9.0
    cmap = plt.get_cmap("turbo")

    axes, sc = [], None
    for i, (title, kind, payload) in enumerate(fields):
        ax = ax_mm(fig, F_X + i * (F_S + F_GAP), F_Y, F_S, F_S,
                   projection=ccrs.PlateCarree())
        ax.set_extent(e, crs=ccrs.PlateCarree())
        _grey_bg_zoom(ax)
        if payload is not None:
            lo, la, vals, size = payload
            sc = ax.scatter(lo, la, c=vals, s=size,
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            edgecolors="none",
                            transform=ccrs.PlateCarree(),
                            zorder=4)
        # cartopy 0.25 + mpl 3.11: gridlines(draw_labels=True)
        # wipes set_title -> draw the column title manually
        ax.text(0.5, 1.02, title, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                          color="0.5", alpha=0.5, zorder=5)
        gl.set_zorder(6)  # Gridliner artist itself defaults to zorder 2
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = False
        gl.left_labels = (i == 0)
        gl.xlocator = mticker.FixedLocator([-90.0, -89.0])
        gl.ylocator = mticker.FixedLocator([29.0, 30.0])
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {"size": 6}
        gl.ylabel_style = {"size": 6, "rotation": 90}
        # tuck the left-side latitude labels closer to the map frame
        # (cartopy default pad is 5 pt)
        gl.ypadding = 1
        axes.append(ax)
    # "g" label: pushed 3 mm left of the default (-0.08) anchor, and its
    # baseline sits level with the "f" label (same mm anchor: f bottom is
    # at E_Y + (1-0.9079)*27 = 192.49 mm from the canvas top)
    panel_label(axes[0], "g", y=1.0233, x=-0.216)

    # --- error row: model - GT, all on the irregular ocean nodes --------
    errs = [None, cfm - gt, vit_n - gt, ef_n - gt, ocn_n - gt]
    # symmetric limit from the 99th percentile (same recipe as panel c)
    p99 = np.nanpercentile(
        np.abs(np.concatenate([errs[k][mz_in] for k in (1, 2, 3, 4)])), 99)
    err_lim = float(min(2.0, max(0.5, np.ceil(p99 * 2) / 2)))
    sc_err = None
    for i in range(5):
        f_row2_y = F_Y + F_S + F_RGAP
        if errs[i] is None:
            ax = ax_mm(fig, F_X + i * (F_S + F_GAP), f_row2_y, F_S, F_S)
            ax.axis("off")
            if i == 0:
                # right edge of the empty slot, hugging the first error map
                ax.text(1.00, 0.5, "$H_s$ error (m)", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=7)
            continue
        ax = ax_mm(fig, F_X + i * (F_S + F_GAP), f_row2_y, F_S, F_S,
                   projection=ccrs.PlateCarree())
        ax.set_extent(e, crs=ccrs.PlateCarree())
        _grey_bg_zoom(ax)
        sc_err = ax.scatter(lons[mz_in], lats[mz_in],
                            c=np.ma.masked_invalid(errs[i][mz_in]), s=3.0,
                            cmap=EL_CMAP, vmin=-err_lim, vmax=err_lim,
                            edgecolors="none",
                            transform=ccrs.PlateCarree(),
                            zorder=4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                          color="0.5", alpha=0.5, zorder=5)
        gl.set_zorder(6)  # Gridliner artist itself defaults to zorder 2
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = False
        # every error map carries its own longitude labels; the maps end
        # at y=239 mm and the horizontal colorbars start at 244 mm, so
        # the labels still clear them
        gl.bottom_labels = True
        gl.xlocator = mticker.FixedLocator([-90.0, -89.0])
        gl.ylocator = mticker.FixedLocator([29.0, 30.0])
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {"size": 6}
        gl.ylabel_style = {"size": 6, "rotation": 90}
        # tuck the bottom longitude labels closer to the map frame
        # (cartopy default pad is 5 pt)
        gl.xpadding = 2
        axes.append(ax)

    cax = ax_mm(fig, **F_CB[0])
    cbar = fig.colorbar(sc, cax=cax, extend="max", orientation="horizontal")
    cbar.set_label("$H_s$ (m)", fontsize=6)
    cax_e = ax_mm(fig, **F_CB[1])
    cbar_e = fig.colorbar(sc_err, cax=cax_e, extend="both",
                          orientation="horizontal")
    cbar_e.set_label("$H_s$ error (m)", fontsize=6)
    for cb in (cbar, cbar_e):
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_linewidth(0.5)
    return axes


# ---------------------------------------------------------------------------
# Panel f: global view of the Hurricane Ida snapshot (GT vs WaveFlow)
# ---------------------------------------------------------------------------
def draw_panel_ida_global(fig, coords):
    from modules.plot.plot_wave import (
        interpolate_wave_data_to_grid, _mask_data_to_global_ocean)

    ts = CASE_STORM_TS
    ymd, hh = ts[:8], ts[9:11]
    year = ymd[:4]

    gt = np.load(f"data/global_hindcast/npz_files/{year}/"
                 f"global_hindcast_{ymd}_{hh}.npz")["hs"].astype(np.float64)
    cfm = _ens_mean(
        "output/results/global/new_data_hs/"
        "patch2048_4frames_concat_wucos_ema999gc_xpred_huge/"
        "sample_epoch99_20260726_194530/samples", ts, "sample")

    lats, lons = coords[:, 0], coords[:, 1]
    fields = [("Reference", gt), ("WaveFlow", cfm)]
    vmin, vmax = 0.0, 9.0
    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # GSHHS full-res land, simplified to 0.02 deg (visually identical to
    # high-res at this panel size, keeps the PDF small)
    land = [g.simplify(0.02, preserve_topology=True) for g in _gshhs_all()]

    axes, sc = [], None
    for i, (title, field) in enumerate(fields):
        # stacked vertically (one column, two rows), flush to the left;
        # Robinson projection, panel-a schematic background (grey GSHHS
        # land, white ocean, thin grey coastline), Hs interpolated to a
        # regular grid and masked to the wave-node ocean (hs only - no
        # ssh contours, no direction arrows)
        ax = ax_mm(fig, E_X, E_Y + i * (E_H + E_ROW_GAP), E_W, E_H,
                   projection=ccrs.Robinson())
        ax.add_geometries(land, ccrs.PlateCarree(), facecolor="#e8e8e8",
                          edgecolor="none", zorder=1)
        ax.add_geometries(land, ccrs.PlateCarree(), facecolor="none",
                          edgecolor="0.35", linewidth=0.4, zorder=11)
        lon_mesh, lat_mesh, grid = interpolate_wave_data_to_grid(
            lons, lats, field, grid_resolution=(2000, 1000),
            max_distance=0.5)
        grid = np.clip(grid, 0, None)
        (grid,) = _mask_data_to_global_ocean(lon_mesh[0, :],
                                             lat_mesh[:, 0], grid)
        rgba = cmap(norm(grid))
        rgba[..., 3] = np.where(np.isnan(grid), 0.0, 1.0)
        ax.imshow(rgba, extent=[-180, 180, -90, 90], origin="lower",
                  transform=ccrs.PlateCarree(), zorder=10,
                  interpolation="bilinear", rasterized=True)
        ax.set_global()
        sc = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # titles on the left side (no gridlines here, but rotated side
        # titles keep the stacked rows tight)
        ax.text(-0.045, 0.5, title, transform=ax.transAxes, rotation=90,
                va="center", ha="center", fontsize=7)
        # black box: panel-g zoom extent
        e = STORM_EXTENT
        ax.plot([e[0], e[1], e[1], e[0], e[0]],
                [e[2], e[2], e[3], e[3], e[2]],
                color="k", lw=0.9, transform=ccrs.PlateCarree(), zorder=12)
        axes.append(ax)
    # "f" label: same absolute x as the "a" label (0.952 mm from the
    # canvas left): (0.952 - 5.00) / 54.00 = -0.075
    # "f" label: 2 mm lower than the old y=0.982 (map is 27 mm tall)
    panel_label(axes[0], "f", y=0.9079, x=-0.075)

    cax = ax_mm(fig, **E_CB)
    cbar = fig.colorbar(sc, cax=cax, extend="max", orientation="horizontal")
    # labelpad in pt (dpi-independent): 3 pt sits closer to the ticks
    # than the 4 pt default
    cbar.set_label("$H_s$ (m)", fontsize=6, labelpad=3)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.5)
    return axes


# ---------------------------------------------------------------------------
# Panel e (unused): global spatial RMSE difference (OCN - CFM)
# ---------------------------------------------------------------------------
def draw_panel_global_diff(fig, gs, coords):
    d = np.load("oceancastnet_repro/output/results/global_swh/"
                "train_b192_20260803_113451/comparison_cfm/spatial_rmse.npz")
    diff = d["ocn"].astype(np.float64) - d["cfm"].astype(np.float64)
    valid = (d["valid_count"] > 0) & np.isfinite(diff)
    lats, lons = coords[:, 0], coords[:, 1]

    vmin, vmax = -0.3, 0.3
    cmap = plt.get_cmap("RdBu")  # blue = positive diff = ours (CFM) better

    gs_c = gs.subgridspec(1, 2, width_ratios=[1.9, 1.0], wspace=0.12)

    ax = fig.add_subplot(gs_c[0, 0], projection=ccrs.PlateCarree())
    ax.set_global()
    add_map_features(ax, scale="110m")
    sc = ax.scatter(lons[valid], lats[valid], c=diff[valid], s=0.1,
                    cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="none",
                    transform=ccrs.PlateCarree(), zorder=4)
    ax.set_title("Per-node RMSE: OceanCastNet \u2212 WaveFlow", pad=3)
    panel_label(ax, "e")

    axz = fig.add_subplot(gs_c[0, 1], projection=ccrs.PlateCarree())
    zext = [105, 130, 15, 30]
    axz.set_extent(zext, crs=ccrs.PlateCarree())
    add_map_features(axz, scale="50m", extent=zext)
    mz = valid & (lons >= zext[0]) & (lons <= zext[1]) \
        & (lats >= zext[2]) & (lats <= zext[3])
    axz.scatter(lons[mz], lats[mz], c=diff[mz], s=1.5, cmap=cmap,
                vmin=vmin, vmax=vmax, edgecolors="none",
                transform=ccrs.PlateCarree(), zorder=4)
    axz.set_title("Zoom: South China Sea", pad=3)

    cbar = fig.colorbar(sc, ax=[ax, axz], orientation="horizontal",
                        fraction=0.05, pad=0.08, aspect=35, extend="both")
    cbar.set_label("\u0394RMSE (m)  (blue: ours better)", fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.5)
    return [ax, axz]


# ---------------------------------------------------------------------------
# Panel f (unused): Typhoon Kompasu case study (global, South China Sea zoom)
# ---------------------------------------------------------------------------
def _ens_mean(directory, ts, key, n_ens=10, fname_tpl="sample_{ts}_ens{i:03d}.npz"):
    acc, cnt = None, 0
    for i in range(n_ens):
        p = os.path.join(directory, fname_tpl.format(ts=ts, i=i))
        if not os.path.exists(p):
            continue
        arr = np.load(p)[key].astype(np.float64)
        acc = arr if acc is None else acc + arr
        cnt += 1
    if cnt == 0:
        raise FileNotFoundError(f"no ensemble files in {directory} for {ts}")
    return acc / cnt


def draw_panel_global_case(fig, gs, coords):
    ts = CASE_TS
    ymd, hh = ts[:8], ts[9:11]
    year = ymd[:4]

    gt = np.load(f"data/global_hindcast/npz_files/{year}/"
                 f"global_hindcast_{ymd}_{hh}.npz")["hs"].astype(np.float64)
    cfm = _ens_mean(
        "output/results/global/new_data_hs/"
        "patch2048_4frames_concat_wucos_ema999gc_xpred_huge/"
        "sample_epoch99_20260726_194530/samples", ts, "sample")
    vit = _ens_mean(
        "output/results/global/new_data_hs_grid/p16_vit_huge_xpred/"
        "sample_epoch99_20260801_000131/samples_nodes", ts, "sample")
    ocn = np.load(
        "oceancastnet_repro/output/results/global_swh/"
        "train_b192_20260803_113451/inference_nodes/samples/"
        f"sample_{ts}.npz")["prediction"].astype(np.float64)

    lats, lons = coords[:, 0], coords[:, 1]
    e = CASE_EXTENT
    mz = ((lons >= e[0]) & (lons <= e[1]) & (lats >= e[2]) & (lats <= e[3]))

    fields = [
        ("Reference", gt),
        ("WaveFlow", cfm),
        ("WaveFlow-ViT", vit),
        ("OceanCastNet", ocn),
    ]
    vmin, vmax = 0.0, 8.0
    cmap = plt.get_cmap("turbo")

    axes, sc = [], None
    for i, (title, field) in enumerate(fields):
        ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
        ax.set_extent(e, crs=ccrs.PlateCarree())
        add_map_features(ax, scale="50m", extent=e)
        vals = np.ma.masked_invalid(field[mz])
        sc = ax.scatter(lons[mz], lats[mz], c=vals, s=1.5, cmap=cmap,
                        vmin=vmin, vmax=vmax, edgecolors="none",
                        transform=ccrs.PlateCarree(), zorder=4)
        ax.set_title(title, pad=3)
        axes.append(ax)
    axes[0].text(0.02, 0.96, ts.replace("T", " ") + " UTC",
                 transform=axes[0].transAxes, fontsize=6, va="top",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                           alpha=0.85, edgecolor="0.6", linewidth=0.4))
    panel_label(axes[0], "f")

    cbar = fig.colorbar(sc, ax=axes, orientation="horizontal",
                        fraction=0.05, pad=0.08, aspect=40, extend="max")
    cbar.set_label("Significant wave height Hs (m)", fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.5)
    return axes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    hk_fields = _load_hk_case_fields(CASE_HK_TS)
    coords = np.load("data/global_hindcast/global_coordinates.npy")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    # All panels are placed at absolute mm positions (see the layout
    # constants at the top of the file), panel d included (D_CELL).

    print("Panel a: HK node comparison ...")
    draw_panel_a(fig)
    print(f"Panel b: HK qualitative comparison {CASE_HK_TS} ...")
    draw_panel_b(fig, hk_fields)
    print("Panel c: Victoria Harbour zoom ...")
    draw_panel_c(fig, hk_fields)
    print("Panel d: RMSE violin row ...")
    draw_panel_d(fig, D_CELL)
    print(f"Panel f: Hurricane Ida global view {CASE_STORM_TS} ...")
    draw_panel_ida_global(fig, coords)
    print(f"Panel g: Hurricane Ida zoom {CASE_STORM_TS} ...")
    draw_panel_e(fig, coords)
    # Global spatial RMSE difference and global Kompasu case panels
    # temporarily dropped — draw_panel_global_diff/draw_panel_global_case
    # remain available.

    pdf_path = os.path.join(OUT_DIR, "figure2_draft.pdf")
    png_path = os.path.join(OUT_DIR, "figure2_draft.png")
    if os.environ.get("FIG2_DUMP"):
        # debug: dump every axes' position in mm (from left / from top)
        fig.canvas.draw()
        w_mm, h_mm = FIG_W * 25.4, FIG_H * 25.4
        for i, ax in enumerate(fig.axes):
            p = ax.get_position()
            print(f"DUMP {i:02d} x={p.x0*w_mm:7.2f} y_top={(1-p.y1)*h_mm:7.2f}"
                  f" w={p.width*w_mm:6.2f} h={p.height*h_mm:6.2f}")
    # 5 pt 字形底线：先 draw 一轮生成刻度标签的 Text，再把过小的 mathtext
    # 父字号提到 7.2 pt（$H_s$ 的下标 5.04 pt）
    fig.canvas.draw()
    TY.enforce_glyph_floor(fig)
    fig.savefig(pdf_path, dpi=300, facecolor="white")
    fig.savefig(png_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
