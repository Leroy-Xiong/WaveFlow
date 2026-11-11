#!/usr/bin/env python
"""Extended Data Figure 6 (sixth ED figure) — four-variable qualitative
ensemble fields on the global mesh.

Companion of ED Figure 5 (Hong Kong): the same Reference | ensemble mean |
ensemble s.d. columns for hs, tp, dm and elev, but on the 651,652-node global
mesh and with the two cases stacked as blocks.  The claim: the global
ensemble reproduces the full-field structure of all four variables in both
a storm and an ordinary (typical) frame, with spread following the storm
track and swell structure instead of a uniform background.

Map recipe (identical to main Fig. 2f): Robinson projection, GSHHS full-res
land simplified to 0.02 deg (grey #e8e8e8, thin 0.35 grey coast), node
fields linearly interpolated to a regular 2000 x 1000 grid
(``interpolate_wave_data_to_grid``), masked to the wave-node ocean
(``_mask_data_to_global_ocean``) and drawn as a rasterised RGBA imshow in
the PlateCarree transform.  Dm is interpolated in sin/cos space and folded
back with atan2 (cyclic field, no seam).

Case selection (deterministic, recomputed on every run):
  storm   20220924T06  Hurricane Fiona landfall in Nova Scotia, same hour
                       as main Fig. 4/5;
  typical the 3-hourly frame of 2022 whose ocean fraction of GT hs > 6 m is
          closest to the median of the 192 candidate frames (the 1st and
          15th of each month, 8 frames each); ties resolve to the earliest
          frame — a representative global sea state, not an extreme.

Known data quirk: the Atlantic node at (39.49 N, 31.15 W) holds hs ~ 18 m
in EVERY frame of the hindcast product (an isolated bad node).  The hs
colour ceiling therefore follows the 99.9th percentile of the combined
reference|mean field (0.5 m steps), not the peak; out-of-range fields are
flagged by extend arrows on the colour bars.

INPUTS (read-only)
  data/global_hindcast/npy_files_daily_4var/<yyyy>/global_hindcast_<yyyymmdd>.npy
                                (8, 651652, 4), frame = hour // 3
  data/global_hindcast/global_coordinates.npy   (651652, 2) = lat, lon
  output/results/global/new_data_allvar/patch2048_4frames_concat_
    giant_tod_ext/sample_allvar_e199_full_v2_20261018_133956/samples/
                                sample_<ts>_ens<k>.npz, (651652, 4) physical

OUTPUT
  output/plot/figure_ed_fields_global/figure_ed_fields_global.{png,pdf}
  output/plot/figure_ed_fields_global/ed_fields_global_stats.csv
  (+ .alignment.json / .alignment.svg; cache/<ts>.npz node statistics cache,
   cache/<ts>_grids.npz interpolated display grids)

USAGE
  python scripts/figures/figure_ed_fields_global/plot_ed_fields_global.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator
import cartopy.crs as ccrs
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "figures", "figure2"))
import plot_figure2 as PF  # noqa: E402  _gshhs_all (GSHHS full-res land)
from modules.plot.plot_wave import (  # noqa: E402
    interpolate_wave_data_to_grid, _mask_data_to_global_ocean)

OUT_DIR = os.path.join(ROOT, "output/plot/figure_ed_fields_global")
OUT_BASE = os.path.join(OUT_DIR, "figure_ed_fields_global")
CACHE_DIR = os.path.join(OUT_DIR, "cache")

# Release override: sample and hindcast roots are redirectable.
GT_DIR = os.environ.get("WF_GT_GLOBAL_4VAR",
                        "data/global_hindcast/npy_files_daily_4var")
COORDS = "data/global_hindcast/global_coordinates.npy"
SAMPLES = os.environ.get(
    "WF_GLOBAL_4VAR_SAMPLES",
    "output/results/global/new_data_allvar/patch2048_4frames_concat_"
    "xpred_giant_tod_ext/sample_allvar_e199_full_v2_20261018_133956/"
    "samples")

TS_STORM = "20220924T06"          # Hurricane Fiona landfall (Fig. 4/5 case)
VARS = ["hs", "tp", "dm", "elev"]
N_ENS = 10
GRID_RES = (2000, 1000)           # display grid, same as main Fig. 2f

# --- layout (mm; x from the left edge, y from the top edge) ------------------
# vertical right-edge colour bars (ED house style, cf. plot_ed_joint) keep the
# total height within one printed page (<= 247 mm)
FW, FH = 180.0, 230.0
MM = 1 / 25.4
MAP_W, MAP_H = 48.0, 24.0         # Robinson global map, 2:1 like Fig. 2f
COL_X = (4.5, 56.0, 107.5)        # pitch 51.5 = map 48 + gutter 3.5
ROW_PITCH = MAP_H + 3.0           # 3 mm row gutter (per-row letters dropped)
ROWS_B1 = [7.6 + i * ROW_PITCH for i in range(4)]
ROWS_B2 = [121.6 + i * ROW_PITCH for i in range(4)]
B2_TITLE_Y, B2_COLLAB_Y = 117.0, 119.8
TITLE_Y, COLLAB_Y = 3.0, 5.8
BAR_STD_X, BAR_FLD_X, BAR_W = 157.5, 166.0, 2.0

# --- ED house style (identical to plot_ed_ablation / _validation / _joint) ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "TeX Gyre Heros",
                        "DejaVu Sans"],
    "font.size": 6.8, "axes.labelsize": 6.8, "axes.titlesize": 6.8,
    "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "legend.fontsize": 5.0,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.bbox": "tight",
})

# tp uses cividis (not viridis) so the field columns are distinguishable from
# the s.d. column, which is viridis for every variable
FIELD_CMAP = {"hs": "turbo", "tp": "cividis", "dm": "twilight", "elev": "turbo"}
STD_CMAP = "viridis"
STD_STEP = {"hs": 0.25, "tp": 0.5, "dm": 5.0, "elev": 0.05}
ROW_LABELS = {"hs": r"$H_\mathrm{s}$ (m)", "tp": r"$T_\mathrm{p}$ (s)",
              "dm": r"$D_\mathrm{m}$ (deg)", "elev": "Elev (m)"}
ROW_LABEL_SIZE = 7.5   # keeps mathtext subscripts above the 5 pt glyph floor
COL_LABELS = ["Reference", "Ens. mean (M=10)", "s.d."]

# GSHHS full-res land simplified to 0.02 deg (visually identical at this
# panel size, keeps the PDF small) — same as main Fig. 2f
_LAND = None


def _land():
    global _LAND
    if _LAND is None:
        _LAND = [g.simplify(0.02, preserve_topology=True)
                 for g in PF._gshhs_all()]
    return _LAND


def ax_at(fig, x, y, w, h, **kw):
    return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH], **kw)


def select_typical_ts():
    """Most ordinary 2022 frame: ocean fraction with GT hs > 6 m closest to
    the median over the 192 frames of the 1st and 15th of each month;
    ties -> earliest.

    Release override: ``ED6_TYPICAL_TS`` pins the frame (useful when only the
    two case days are available); if the candidate daily files are absent the
    paper's frame (2022-01-15T15) is used instead of failing.
    """
    pin = os.environ.get("ED6_TYPICAL_TS")
    if pin:
        print(f"typical case pinned by ED6_TYPICAL_TS: {pin}")
        return pin
    fracs = []
    for month in range(1, 13):
        for day in (1, 15):
            ymd = f"2022{month:02d}{day:02d}"
            path = os.path.join(GT_DIR, "2022", f"global_hindcast_{ymd}.npy")
            if not os.path.exists(path):
                fracs = None
                break
            frames = np.load(path, mmap_mode="r")
            for slot in range(8):
                frac = float((frames[slot][:, 0] > 6.0).mean())
                fracs.append((f"{ymd}T{slot * 3:02d}", frac))
        if fracs is None:
            break
    if not fracs:
        print("warning: 2022 candidate daily hindcast files are missing; "
              "using the paper frame 20220115T15 for the typical case "
              "(set ED6_TYPICAL_TS to override)")
        return "20220115T15"
    med = float(np.median([f for _, f in fracs]))
    best = min(fracs, key=lambda t: (abs(t[1] - med), t[0]))
    print(f"typical case: {best[0]}  frac(hs>6 m) {best[1] * 100:.3f}% "
          f"(candidate median {med * 100:.3f}%)")
    return best[0]


def load_members(ts):
    jobs = [os.path.join(SAMPLES, f"sample_{ts}_ens{k:03d}.npz")
            for k in range(N_ENS)]

    def one(path):
        with np.load(path) as d:
            return d["sample"].astype(np.float32)

    with ThreadPoolExecutor(max_workers=8) as ex:
        return np.stack(list(ex.map(one, jobs)))   # (M, N, 4)


def ens_stats(ens):
    """Ensemble mean/s.d. per variable, (4, N); circular for dm."""
    mean = np.empty(ens.shape[1:], np.float64)
    std = np.empty(ens.shape[1:], np.float64)
    for j, v in enumerate(VARS):
        e = ens[:, :, j].astype(np.float64)
        if v == "dm":
            rad = np.deg2rad(e)
            m = np.rad2deg(np.arctan2(np.sin(rad).mean(0),
                                      np.cos(rad).mean(0))) % 360.0
            dev = np.arctan2(np.sin(rad - np.deg2rad(m)),
                             np.cos(rad - np.deg2rad(m)))
            mean[:, j], std[:, j] = m, np.rad2deg(dev.std(0))
        else:
            mean[:, j], std[:, j] = e.mean(0), e.std(0)
    return mean, std


def load_case(ts):
    """Reference (4, N), ensemble mean and s.d. (4, N); cached on disk."""
    cache = os.path.join(CACHE_DIR, f"{ts}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["gt"], d["mean"], d["std"]
    gt = np.load(os.path.join(GT_DIR, ts[:4],
                              f"global_hindcast_{ts[:8]}.npy"),
                 mmap_mode="r")[int(ts[9:11]) // 3]      # (N, 4)
    g = np.asarray(gt).T.astype(np.float64)              # (4, N)
    mean, std = ens_stats(load_members(ts))
    if not (np.isfinite(g).all() and np.isfinite(mean).all()):
        raise ValueError(f"{ts}: non-finite values in GT or ensemble")
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, gt=g, mean=mean, std=std)
    return g, mean, std


def _interp(lon, lat, vals):
    """Node field -> masked regular display grid (main Fig. 2f recipe)."""
    lon_mesh, lat_mesh, grid = interpolate_wave_data_to_grid(
        lon, lat, vals, grid_resolution=GRID_RES, max_distance=0.5)
    (grid,) = _mask_data_to_global_ocean(lon_mesh[0, :], lat_mesh[:, 0], grid)
    return grid


def display_grids(ts, lon, lat, g, mean, std):
    """Interpolate every panel field to the display grid; cached.

    dm fields (Ref/mean) interpolate in sin/cos space and fold back with
    atan2 so the 0/360 seam never averages; dm s.d. is a linear magnitude.
    hs/tp/s.d. grids are clipped at zero (linear interpolation can
    undershoot near the coast)."""
    cache = os.path.join(CACHE_DIR, f"{ts}_grids.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    grids = {}
    for v_i, v in enumerate(VARS):
        for kind, src in (("ref", g[v_i]), ("mean", mean[:, v_i]),
                          ("std", std[:, v_i])):
            t0 = time.time()
            if v == "dm" and kind in ("ref", "mean"):
                rad = np.deg2rad(src)
                s = _interp(lon, lat, np.sin(rad))
                c = _interp(lon, lat, np.cos(rad))
                grid = np.rad2deg(np.arctan2(s, c)) % 360.0
            else:
                grid = _interp(lon, lat, src)
                if v in ("hs", "tp") or kind == "std":
                    grid = np.clip(grid, 0.0, None)
            grids[f"{v}_{kind}"] = grid.astype(np.float32)
            print(f"  [interp {ts} {v}_{kind}] {time.time() - t0:.1f}s")
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, **grids)
    return grids


def field_clims(v, g, m):
    if v == "hs":
        # robust ceiling: the hindcast has an isolated ~18 m bad node
        p999 = float(np.percentile(np.concatenate([g, m]), 99.9))
        return 0.0, max(float(np.ceil(p999 / 0.5) * 0.5), 0.5)
    if v == "tp":
        # robust ceiling like hs: global Tp is 5-12 s almost everywhere, a
        # handful of >15 s swell nodes would compress the whole field into
        # the bottom of the scale; the tail leaves via the extend arrow
        p999 = float(np.percentile(np.concatenate([g, m]), 99.9))
        return 0.0, max(float(np.ceil(p999)), 1.0)
    if v == "dm":
        return 0.0, 360.0
    # elev: symmetric robust band.  The global tide is mostly within
    # ±0.2 m, so the GT p1/p99 band (~±0.5 m) left the field mid-scale and
    # flat; V follows the 95th percentile of |reference ∪ mean| (0.05 m
    # steps), coastal megatides saturate beyond (extend arrows both ends)
    v95 = float(np.percentile(np.abs(np.concatenate([g, m])), 95))
    band = max(float(np.ceil(v95 / 0.05) * 0.05), 0.05)
    return -band, band


def std_clims(v, s):
    # robust ceiling (p99.9), same rationale as the field clims: single-node
    # spread outliers (the Atlantic bad node, storm-core peaks) otherwise
    # compress the whole map into the bottom of the viridis scale
    step = STD_STEP[v]
    p999 = float(np.percentile(s, 99.9))
    return 0.0, max(float(np.ceil(p999 / step) * step), step)


def bar_extend(fields, vmin, vmax):
    """'min'/'max'/'both'/'neither' for the shared colour bar, from whether
    any plotted field actually falls outside [vmin, vmax]."""
    lo = any((f < vmin - 1e-9).any() for f in fields)
    hi = any((f > vmax + 1e-9).any() for f in fields)
    return "both" if lo and hi else "max" if hi else "min" if lo else "neither"


def draw_map(fig, x, y, grid, cmap, vmin, vmax):
    """Robinson global map, main Fig. 2f recipe: grey GSHHS land, rasterised
    RGBA imshow of the interpolated grid in the PlateCarree transform."""
    ax = ax_at(fig, x, y, MAP_W, MAP_H, projection=ccrs.Robinson())
    ax.add_geometries(_land(), ccrs.PlateCarree(), facecolor="#e8e8e8",
                      edgecolor="none", zorder=1)
    ax.add_geometries(_land(), ccrs.PlateCarree(), facecolor="none",
                      edgecolor="0.35", linewidth=0.4, zorder=11)
    norm = Normalize(vmin, vmax)
    rgba = plt.get_cmap(cmap)(norm(grid))
    rgba[..., 3] = np.where(np.isnan(grid), 0.0, 1.0)
    ax.imshow(rgba, extent=[-180, 180, -90, 90], origin="lower",
              transform=ccrs.PlateCarree(), zorder=10,
              interpolation="bilinear", rasterized=True)
    ax.set_global()
    ax.spines["geo"].set_linewidth(0.6)
    return ax


def vbar(fig, x, y, cmap, vmin, vmax, ticks=None, nbins=5,
         extend="neither"):
    """Thin vertical right-edge colour bar, 5 pt compact tick labels."""
    cax = ax_at(fig, x, y, BAR_W, MAP_H)
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(vmin, vmax), cmap),
                      cax=cax, orientation="vertical", extend=extend)
    if ticks is not None:
        cb.set_ticks(ticks)
    else:
        cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:g}"))
    cb.ax.tick_params(labelsize=5.0, length=1.2, width=0.5, pad=0.6)
    cb.outline.set_linewidth(0.3)
    cb.ax.minorticks_off()
    return cax


def case_title(ts, storm):
    d = f"{ts[6:8]} {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                      'Sep', 'Oct', 'Nov', 'Dec'][int(ts[4:6]) - 1]} {ts[:4]}"
    if storm:
        return f"Storm — Hurricane Fiona ({d}, {ts[9:11]} UTC)"
    return f"Typical — median global sea state of 2022 ({d}, {ts[9:11]} UTC)"


def alignment_gate(fig, map_axes, letters):
    """Render-time multi-panel alignment gate (nature-figure QA)."""
    scripts = os.environ.get(
        "NATURE_FIGURE_SCRIPTS",
        os.path.expanduser("~/.claude/skills/nature-figure/scripts"))
    if not os.path.isdir(scripts):
        print(f"[alignment] skill scripts not found at {scripts}; skipped")
        return
    sys.path.insert(0, scripts)
    from audit_panel_alignment import require_matplotlib_panel_alignment
    row_groups = [letters[i * 3:(i + 1) * 3] for i in range(8)]
    col_groups = [letters[c::3] for c in range(3)]
    require_matplotlib_panel_alignment(
        fig,
        json_out=OUT_BASE + ".alignment.json",
        overlay_svg=OUT_BASE + ".alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        axes=map_axes,
        panel_ids={ax: s for ax, s in zip(map_axes, letters)},
        row_groups=row_groups, column_groups=col_groups,
        exemptions=[{
            "panels": list(letters), "checks": ["vertical-gutter"],
            "reason": "the gap between the storm and typical blocks carries "
                      "the second block's title; repeated intra-block row "
                      "gutters are 4.5 mm",
        }],
    )
    print("[alignment] gate passed")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    coords = np.load(COORDS)
    lat, lon = coords[:, 0], coords[:, 1]

    typical_ts = select_typical_ts()
    cases = {"storm": TS_STORM, "typical": typical_ts}

    data, grids = {}, {}
    for name, ts in cases.items():
        g, mean, std = load_case(ts)
        data[name] = (g, mean, std)
        for j, v in enumerate(VARS):
            print(f"[{name} {ts}] {v}: GT [{g[j].min():.3f}, {g[j].max():.3f}] "
                  f"p99.9 {np.percentile(g[j], 99.9):.3f} "
                  f"mean max {mean[:, j].max():.3f} "
                  f"std max {std[:, j].max():.3f} "
                  f"p50 {np.median(std[:, j]):.3f}")
        grids[name] = display_grids(ts, lon, lat, g, mean, std)

    fig = plt.figure(figsize=(FW * MM, FH * MM))
    map_axes, letters, cbar_axes = [], [], []
    stats_rows = []
    letter_seq = iter("abcdefghijklmnopqrstuvwxyz")

    for b, (case, ts) in enumerate(cases.items()):
        rows = ROWS_B1 if b == 0 else ROWS_B2
        g, mean, std = data[case]
        for r, v in enumerate(VARS):
            vmin, vmax = field_clims(v, g[r], mean[:, r])
            svmax = std_clims(v, std[:, r])[1]
            for k, kind in enumerate(("ref", "mean", "std")):
                ax = draw_map(fig, COL_X[k], rows[r],
                              grids[case][f"{v}_{kind}"],
                              FIELD_CMAP[v] if k < 2 else STD_CMAP,
                              vmin if k < 2 else 0.0,
                              vmax if k < 2 else svmax)
                # 面板字母不再逐图出（作者: 只有 Storm / Typical 两个块
                # 各带一个字母 a / b，位置在各块标题左侧）；这里仍收集
                # 每个轴的 id，仅供 alignment gate 使用
                map_axes.append(ax)
                letters.append(next(letter_seq))
            ext_f = bar_extend([g[r], mean[:, r]], vmin, vmax)
            ext_s = bar_extend([std[:, r]], 0.0, svmax)
            cbar_axes.append(vbar(
                fig, BAR_STD_X, rows[r], STD_CMAP, 0.0, svmax, nbins=4,
                extend=ext_s))
            cbar_axes.append(vbar(
                fig, BAR_FLD_X, rows[r], FIELD_CMAP[v], vmin, vmax,
                ticks=[0, 90, 180, 270, 360] if v == "dm" else None,
                extend=ext_f))
            stats_rows.append({
                "case": case, "timestamp": ts, "variable": v,
                "gt_min": g[r].min(), "gt_max": g[r].max(),
                "gt_p99": float(np.percentile(g[r], 99)),
                "mean_max": mean[:, r].max(),
                "mean_bias": float((((mean[:, r] - g[r]) + 180.0) % 360.0
                                    - 180.0).mean() if v == "dm"
                                   else (mean[:, r] - g[r]).mean()),
                "std_max": std[:, r].max(),
                "std_p50": float(np.median(std[:, r])),
                "field_vmin": vmin, "field_vmax": vmax,
                "std_vmax": svmax,
                "field_extend": ext_f, "std_extend": ext_s,
            })
        # block label: a = storm, b = typical (整张图只有这两个字母)
        fig.text((COL_X[0] - 3.0) / FW,
                 1 - (TITLE_Y if b == 0 else B2_TITLE_Y) / FH, "ab"[b],
                 fontsize=8.0, fontweight="bold", ha="left", va="baseline")
        fig.text(COL_X[0] / FW,
                 1 - (TITLE_Y if b == 0 else B2_TITLE_Y) / FH,
                 case_title(ts, case == "storm"),
                 fontsize=6.8, fontweight="bold", ha="left", va="baseline")
        for k, lab in enumerate(COL_LABELS):
            fig.text((COL_X[k] + MAP_W / 2) / FW,
                     1 - (COLLAB_Y if b == 0 else B2_COLLAB_Y) / FH, lab,
                     fontsize=5.8, ha="center", va="baseline")
    for rows in (ROWS_B1, ROWS_B2):
        for r, v in enumerate(VARS):
            fig.text(2.4 / FW, 1 - (rows[r] + MAP_H / 2) / FH, ROW_LABELS[v],
                     fontsize=ROW_LABEL_SIZE, rotation=90, ha="center",
                     va="center", rotation_mode="anchor")

    alignment_gate(fig, map_axes, letters)

    with open(os.path.join(OUT_DIR, "ed_fields_global_stats.csv"), "w",
              newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(stats_rows[0]))
        writer.writeheader()
        writer.writerows(stats_rows)
    print("stats csv:", os.path.join(OUT_DIR, "ed_fields_global_stats.csv"))

    fig.savefig(OUT_BASE + ".png", dpi=300, bbox_inches="tight",
                pad_inches=0.02)
    # submission artwork: exact 180 mm width, no tight-bbox expansion
    # Release override: the 24-panel vector PDF is ~300 MB and is not needed
    # for the figure-reproduction package; set ED6_SKIP_PDF=1 to skip it.
    if os.environ.get("ED6_SKIP_PDF") != "1":
        plt.rcParams["savefig.bbox"] = "standard"
        fig.savefig(OUT_BASE + ".pdf", dpi=300)
        plt.rcParams["savefig.bbox"] = "tight"
    print("saved:", OUT_BASE)


if __name__ == "__main__":
    main()
