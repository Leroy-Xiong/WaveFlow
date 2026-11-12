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
              (coords_hk, 0.3, "HK unstructured mesh (ours)")]
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


def _fig3def():
    sys.path.insert(0, str(REPO / "reproduce/panels"))
    import fig3def
    return fig3def


def fig3d():
    F = _fig3def()
    from modules.utils import HK_REGION
    from modules.plot.global_ocean_boundary import get_global_ocean_geometry  # noqa: F401
    coords = np.load(REPO / "data/hindcast/hk_coordinates.npy")
    z = np.load(REPO / "data/hindcast/hk_clusters_512.npz")
    labels = z["labels"]
    centers = z["centers"][:, ::-1]
    nodes = coords[:, ::-1].copy()
    rgb = F.patch_color_map(labels, coords)
    fig = plt.figure(figsize=(6.2, 4.6))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent(list(HK_REGION), crs=ccrs.PlateCarree())
    F.add_gray_land(ax, list(HK_REGION), "10m")
    ax.scatter(nodes[:, 0], nodes[:, 1], s=3.2, c=rgb, linewidths=0,
               transform=ccrs.PlateCarree(), zorder=5)
    ax.scatter(centers[:, 0], centers[:, 1], s=3.8, c="black", linewidths=0,
               transform=ccrs.PlateCarree(), zorder=6)
    ax.set_title("Fig. 3d: Hong Kong mesh patches (512 patches)", fontsize=10)
    return fig


def fig3e():
    F = _fig3def()
    from modules.plot.global_ocean_boundary import get_global_ocean_geometry
    gc = np.load(REPO / "data/global_hindcast/global_coordinates.npy")
    gz = np.load(REPO / "data/global_hindcast/global_clusters_2048.npz")
    labels = gz["labels"]
    centers = gz["centers"][:, ::-1]
    lon, lat = gc[:, 1], gc[:, 0]
    rgb = F.patch_color_map(labels, gc)
    g_ocean, g_coast = get_global_ocean_geometry()
    fig = plt.figure(figsize=(7.2, 7.2))
    for i, (clon, clat) in enumerate(F.GLOBE_CENTERS):
        ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.Orthographic(clon, clat))
        F.add_gray_land(ax, None, "110m", coastline=False)
        ax.add_geometries([g_coast], ccrs.PlateCarree(), facecolor="none",
                          edgecolor="0.35", linewidth=0.4, zorder=4)
        ax.gridlines(lw=0.3, color="grey", alpha=0.6)
        ax.scatter(lon, lat, s=0.35, c=rgb, linewidths=0, zorder=3,
                   rasterized=True, transform=ccrs.PlateCarree())
        ax.scatter(centers[:, 0], centers[:, 1], s=1.5, c="black",
                   linewidths=0, zorder=6, rasterized=True,
                   transform=ccrs.PlateCarree())
        ax.set_title(["W. N. Pacific", "Caribbean Sea", "Bay of Bengal",
                      "Arctic"][i], fontsize=9)
    fig.suptitle("Fig. 3e: global mesh patches (2,048 patches)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def fig3f():
    F = _fig3def()
    from modules.plot.global_ocean_boundary import get_global_ocean_geometry
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    from shapely import contains_xy
    import matplotlib
    samples_dir = os.environ.get(
        "WF_GLOBAL_HS_SAMPLES",
        "output/results/global/new_data_hs/"
        "patch2048_4frames_concat_wucos_ema999gc_xpred_huge/"
        "sample_epoch99_20260726_194530/samples")
    bc_hs = F.load_global_samples(REPO / samples_dir)
    gc = np.load(REPO / "data/global_hindcast/global_coordinates.npy")
    gz = np.load(REPO / "data/global_hindcast/global_clusters_2048.npz")
    labels = gz["labels"]
    centers = gz["centers"][:, ::-1]
    lon, lat = gc[:, 1], gc[:, 0]
    rgb = F.patch_color_map(labels, gc)
    g_ocean, g_coast = get_global_ocean_geometry()
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.0),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    for row, (name, (l0, l1, b0, b1)) in enumerate(F.ZOOM_REGIONS):
        m = (lon >= l0) & (lon <= l1) & (lat >= b0) & (lat <= b1)
        vmin = float(np.floor(np.nanmin(bc_hs[m]) * 10) / 10)
        vmax = float(np.ceil(np.nanmax(bc_hs[m]) * 10) / 10)

        def base(ax):
            ax.set_extent([l0, l1, b0, b1], crs=ccrs.PlateCarree())
            F.add_gray_land(ax, (l0, l1, b0, b1), "50m", coastline=False)

        ax = axes[row, 0]
        base(ax)
        ax.scatter(lon[m], lat[m], s=2.2, c=rgb[m], linewidths=0,
                   rasterized=True, transform=ccrs.PlateCarree())
        ax.scatter(centers[:, 0], centers[:, 1], s=2.4, c="black",
                   linewidths=0, transform=ccrs.PlateCarree())
        ax.add_geometries([g_coast], ccrs.PlateCarree(), facecolor="none",
                          edgecolor="black", linewidth=0.5, zorder=5)
        ax.set_title(f"{name}: patches", fontsize=8)

        ax = axes[row, 1]
        base(ax)
        ax.scatter(lon[m], lat[m], c=bc_hs[m], s=2.2, cmap=F.HS_CMAP,
                   vmin=vmin, vmax=vmax, edgecolors="none", rasterized=True,
                   transform=ccrs.PlateCarree())
        ax.add_geometries([g_coast], ccrs.PlateCarree(), facecolor="none",
                          edgecolor="black", linewidth=0.5, zorder=4)
        ax.set_title(f"{name}: Hs nodes", fontsize=8)

        ax = axes[row, 2]
        base(ax)
        nx = 420
        ny = max(2, int(round(nx * (b1 - b0) / (l1 - l0))))
        GX, GY = np.meshgrid(np.linspace(l0, l1, nx), np.linspace(b0, b1, ny))
        zi = griddata(np.column_stack([lon[m], lat[m]]), bc_hs[m],
                      (GX, GY), method="linear")
        zi[~contains_xy(g_ocean, GX.ravel(), GY.ravel()).reshape(GX.shape)] = np.nan
        t = cKDTree(np.column_stack([lon[m], lat[m]]))
        dd, _ = t.query(np.column_stack([GX.ravel(), GY.ravel()]))
        zi[dd.reshape(GX.shape) > 2.5 * np.median(dd)] = np.nan
        im = ax.imshow(zi, extent=[l0, l1, b0, b1], origin="lower",
                       cmap=F.HS_CMAP, vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), interpolation="bilinear")
        ax.add_geometries([g_coast], ccrs.PlateCarree(), facecolor="none",
                          edgecolor="black", linewidth=0.5, zorder=4)
        ax.set_title(f"{name}: Hs interpolated", fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                          format=matplotlib.ticker.FormatStrFormatter("%.1f"))
        cb.ax.tick_params(labelsize=6)
    fig.suptitle("Fig. 3f: patch-boundary fields at 2022-09-24 00Z", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _fig5_modules():
    sys.path.insert(0, str(REPO / "scripts/figures/figure5"))
    sys.path.insert(0, str(REPO / "reproduce/panels"))
    import fig5ghi
    import plot_figure5_case as CASE
    import plot_figure5_global as GLOB
    return fig5ghi, CASE, GLOB


def _fig5_data(CASE, GLOB):
    hk = np.load(REPO / CASE.COORDS)
    gc = np.load(REPO / GLOB.COORDS)
    gt_hk = CASE.gt_hs(CASE.T_MAP)
    gt_global = GLOB.gt_hs("20220928T12")
    return hk, gc, gt_hk, gt_global


def fig5g():
    F, CASE, GLOB = _fig5_modules()
    hk = np.load(REPO / CASE.COORDS)
    lons, lats = hk[:, 1], hk[:, 0]
    gt = CASE.gt_hs(CASE.T_MAP)

    W, H = 180.0, 292.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    x_left, x_right, map_w = 22.0, 94.0, 58.0
    map_h = map_w / 1.4286
    y0, pitch = 15.0, 52.0

    sc_mean = sc_prob = None
    for i, lead in enumerate([0] + CASE.LEADS):
        y = y0 + i * pitch
        if lead == 0:
            vals = gt
            title = "Reference"
        else:
            e = CASE.ens_hs(CASE.case_sample_dir(lead), CASE.T_MAP)
            vals = np.nanmean(e, axis=0)
            title = f"{lead} h lead"
        _ax, sc_mean = CASE.draw_map(
            fig, x_left, y, map_w, map_h, vals, lons, lats, title,
            0.0, CASE.HS_VMAX, "turbo", left_labels=True,
            bottom_labels=(i == 4), canvas=(W, H))
        sc_mean.set_sizes([2.2])
        if lead == 0:
            fig.text(x_right + map_w / 2, 1 - (y + map_h / 2) / H,
                     "P($H_s$>4 m)\n(12-72 h leads)", ha="center",
                     va="center", fontsize=9, color="0.35")
            continue
        e = CASE.ens_hs(CASE.case_sample_dir(lead), CASE.T_MAP)
        p = (e > CASE.EXCEED).mean(0).astype(np.float32)
        axp, sc_prob = CASE.draw_map(
            fig, x_right, y, map_w, map_h, p, lons, lats,
            f"{lead} h lead  P($H_s$>{CASE.EXCEED:.0f} m)", 0.0, 1.0,
            "viridis", left_labels=True, bottom_labels=(i == 4),
            canvas=(W, H))
        sc_prob.set_sizes([2.2])
        CASE.CASE4.add_gt_contour(axp, gt, lons, lats, level=CASE.EXCEED)

    cax = fig.add_axes([0.16, 0.034, 0.30, 0.008])
    cb = fig.colorbar(sc_mean, cax=cax, orientation="horizontal")
    cb.set_label("$H_s$ (m)", fontsize=8, labelpad=4)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=7)
    cax = fig.add_axes([0.56, 0.034, 0.30, 0.008])
    cb = fig.colorbar(sc_prob, cax=cax, orientation="horizontal")
    cb.set_label("probability  $P(H_s>4$ m$)$", fontsize=8, labelpad=4)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Fig. 5g: Hong Kong Kompasu case",
                 fontsize=12, y=0.995)
    return fig


def _global_field(lons, lats, vals, res=0.5):
    """Interpolate unstructured global nodes to a regular lon/lat grid."""
    from scipy.interpolate import griddata
    from shapely import contains_xy
    from modules.plot.global_ocean_boundary import get_global_ocean_geometry

    lon360 = lons % 360.0
    pts = np.column_stack([lon360, lats])
    ext_pts = np.vstack([pts,
                         np.column_stack([np.where(lon360 < 30, lon360 + 360, lon360), lats]),
                         np.column_stack([np.where(lon360 > 330, lon360 - 360, lon360), lats])])
    ext_vals = np.concatenate([vals, vals, vals])
    glon = np.arange(0.0, 360.0 + res / 2, res)
    glat = np.arange(-90.0, 90.0 + res / 2, res)
    GX, GY = np.meshgrid(glon, glat)
    zi = griddata(ext_pts, ext_vals, (GX, GY), method="linear")
    ocean, _coast = get_global_ocean_geometry()
    lon_shift = ((GX + 180.0) % 360.0) - 180.0
    keep = contains_xy(ocean, lon_shift.ravel(), GY.ravel()).reshape(GX.shape)
    zi[~keep] = np.nan
    order = np.argsort(lon_shift[0])
    return lon_shift[0][order], glat, zi[:, order]


def fig5h():
    F, CASE, GLOB = _fig5_modules()
    gc = np.load(REPO / GLOB.COORDS)
    lats, lons = gc[:, 0], gc[:, 1]
    ts = "20220928T12"
    gt = GLOB.gt_hs(ts)
    tmpls = F.CASE_TMPL

    W, H = 180.0, 152.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    map_w = 70.0
    map_h = map_w / 1.9717
    xcols = [12.0, 98.0]
    yrows = [14.0, 58.0, 102.0]
    items = [(0, 0, gt, "Reference")]
    for i, lead in enumerate(GLOB.LEADS):
        vals = GLOB.ens_mean_hs(lead, ts, tmpls.get(lead))
        items.append(((i + 1) // 2, (i + 1) % 2, vals, f"{lead} h lead"))

    im = None
    for row, col, vals, title in items:
        lon_p, lat_p, zi = _global_field(lons, lats, np.asarray(vals, dtype=float))
        ax = fig.add_axes([xcols[col] / W, 1 - (yrows[row] + map_h) / H,
                           map_w / W, map_h / H], projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(linewidth=0.3, color="#666666")
        im = ax.pcolormesh(lon_p, lat_p, zi, cmap="turbo", vmin=0.0,
                           vmax=GLOB.HS_VMAX, transform=ccrs.PlateCarree(),
                           shading="auto", rasterized=True)
        ax.set_title(title, fontsize=8, pad=1.5)

    cax = fig.add_axes([0.22, 0.035, 0.56, 0.010])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("$H_s$ (m)", fontsize=8, labelpad=4)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Fig. 5h: global Hurricane Ian case", fontsize=12, y=0.995)
    return fig


def fig5i():
    F, CASE, GLOB = _fig5_modules()
    gc = np.load(REPO / GLOB.COORDS)
    lats, lons = gc[:, 0], gc[:, 1]
    ts = "20220928T12"
    gt = GLOB.gt_hs(ts)
    tmpls = F.CASE_TMPL
    lon0, lon1, lat0, lat1 = F.CASE_ZOOM

    fields = [gt] + [GLOB.ens_mean_hs(L, ts, tmpls.get(L)) for L in GLOB.LEADS]
    sel = ((lons >= lon0) & (lons <= lon1) & (lats >= lat0) & (lats <= lat1))
    vals_all = np.concatenate([np.asarray(f)[sel] for f in fields])
    vmin = max(0.0, float(np.nanmin(vals_all)))
    vmax = float(np.nanmax(vals_all))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, GLOB.HS_VMAX

    W, H = 180.0, 152.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    map_w = 70.0
    map_h = map_w * (lat1 - lat0) / (lon1 - lon0)
    xcols = [12.0, 98.0]
    yrows = [14.0, 58.0, 102.0]
    sc = None
    for i, vals in enumerate(fields):
        row, col = i // 2, i % 2
        ax = fig.add_axes([xcols[col] / W, 1 - (yrows[row] + map_h) / H,
                           map_w / W, map_h / H],
                          projection=ccrs.PlateCarree())
        ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
        sc = ax.scatter(lons[sel], lats[sel], c=np.asarray(vals)[sel],
                        s=2.5, cmap="turbo", vmin=vmin, vmax=vmax,
                        edgecolors="none", linewidths=0,
                        transform=ccrs.PlateCarree(), rasterized=True)
        ax.coastlines(linewidth=0.4, color="#555555")
        title = "Reference" if i == 0 else f"{GLOB.LEADS[i - 1]} h lead"
        ax.set_title(title, fontsize=8, pad=1.5)
        xstep = 5
        ystep = 2
        xloc = np.arange(np.ceil(lon0 / xstep) * xstep, lon1 + 1e-9, xstep)
        yloc = np.arange(np.ceil(lat0 / ystep) * ystep, lat1 + 1e-9, ystep)
        ax.set_xticks(xloc, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(GLOB.LONGITUDE_FORMATTER)
        ax.tick_params(axis="x", labelsize=6, length=2, width=0.5, pad=1.0)
        if col == 0:
            ax.set_yticks(yloc, crs=ccrs.PlateCarree())
            ax.yaxis.set_major_formatter(GLOB.LATITUDE_FORMATTER)
            ax.tick_params(axis="y", labelsize=6, length=2, width=0.5, pad=0.6)
            for lab in ax.get_yticklabels():
                lab.set_rotation(90)
                lab.set_rotation_mode("anchor")
                lab.set_ha("center")
                lab.set_va("bottom")
        else:
            ax.tick_params(axis="y", labelleft=False, length=0)

    cax = fig.add_axes([0.22, 0.035, 0.56, 0.010])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("$H_s$ (m)", fontsize=8, labelpad=4)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=7)
    step = 2.0 if (vmax - vmin) <= 10 else 5.0
    ticks = list(np.arange(np.ceil((vmin + 1e-9) / step) * step,
                           vmax - 1e-6, step))
    if ticks:
        cb.set_ticks(ticks)
    fig.suptitle("Fig. 5i: Hurricane Ian zoom", fontsize=12, y=0.995)
    return fig
