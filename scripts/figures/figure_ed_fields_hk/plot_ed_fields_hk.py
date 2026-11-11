#!/usr/bin/env python
"""Extended Data Figure 5 (fifth ED figure) — four-variable qualitative
ensemble fields over the Hong Kong mesh.

The first four ED figures (validation / mesh_coastal / joint / ablation)
quantify scores; this one shows the actual fields.  For one storm hour and
one ordinary (typical) hour of the 2021-22 test period it maps, for every
predicted variable (hs, tp, dm, elev), the hindcast reference, the M=10
ensemble mean and the ensemble s.d.  The claim: the ensemble reproduces the
spatial structure of all four variables under both regimes, with spread
concentrated at the storm core and modest elsewhere.

Case selection (deterministic, recomputed on every run):
  storm   20220701T20  Typhoon Chaba peak, same hour as main Fig. 2
                       (reference Hs peak 5.29 m);
  typical the hour of output/plot/hk_hs_timeseries/hk_hs_timeseries_2021_2022.csv
          whose domain hs_p99 is closest to the median hs_p99 of all
          2021-22 hours (among hours whose 10 member files are all present;
          ties resolve to the earliest hour) — an ordinary sea state, not
          an extreme.

INPUTS (read-only)
  data/hindcast/processed_npz/wave_<ts>.npz   reference hs/tp/dm/elev, (7788,)
                                             node order identical to the model
  data/hindcast/hk_coordinates.npy            (7788, 2) = lat, lon
  output/results/hk_cleaned/all_var/f8_blr5e-5_amp_stable_tod/size_sweep/
    base_v1_20260821_025917/curve_ep49/test_bc3/samples/sample_<ts>_ens<k>.npz
                                             (7788, 4) physical units,
                                             columns hs, tp, dm, elev
  output/plot/hk_hs_timeseries/hk_hs_timeseries_2021_2022.csv  typical-case pick

OUTPUT
  output/plot/figure_ed_fields_hk/figure_ed_fields_hk.{png,pdf}
  output/plot/figure_ed_fields_hk/ed_fields_hk_stats.csv
  (+ .alignment.json / .alignment.svg from the panel-alignment gate)

USAGE
  python scripts/figures/figure_ed_fields_hk/plot_ed_fields_hk.py
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "figures", "figure2"))
import plot_figure2 as PF  # noqa: E402  grey_background_hk / HK_REGION / GSHHS land

OUT_DIR = os.path.join(ROOT, "output/plot/figure_ed_fields_hk")
OUT_BASE = os.path.join(OUT_DIR, "figure_ed_fields_hk")

# Release override: sample and hindcast roots are redirectable.
SAMPLES = os.environ.get(
    "WF_HK_SAMPLES",
    "output/results/hk_cleaned/all_var/f8_blr5e-5_amp_stable_tod/"
    "size_sweep/base_v1_20260821_025917/curve_ep49/test_bc3/samples")
GT_DIR = os.environ.get("WF_GT_HK", "data/hindcast/processed_npz")
COORDS = "data/hindcast/hk_coordinates.npy"
HS_CSV = "output/plot/hk_hs_timeseries/hk_hs_timeseries_2021_2022.csv"

TS_STORM = "20220701T20"          # Typhoon Chaba peak (main Fig. 2 case)
VARS = ["hs", "tp", "dm", "elev"]  # sample/GT column order
N_ENS = 10

# --- layout (mm; x from the left edge, y from the top edge) ------------------
FW, FH = 180.0, 121.5
MM = 1 / 25.4
MAP_W, MAP_H = 26.5, 26.5 * 0.7   # HK domain: 1.0 deg lon x 0.7 deg lat
COL_X = {"storm": (4.0, 32.5, 61.0), "typical": (92.5, 121.0, 149.5)}
# row pitch 28.8 mm = map 18.55 + a 10.25 mm channel that fits the 1.1 mm
# gap, the colourbar and its tick labels without bbox overlap (per-row panel
# letters were dropped in 2026-11; only the two block letters a/b remain)
ROW_Y = [7.8, 36.6, 65.4, 94.2]   # first row 0.4 mm lower: 8 pt letters must
                                  # clear the case-title descenders
CB_H, CB_BELOW, CB_INDENT = 1.6, 1.1, 2.0
CASE_TITLE_Y, COL_LABEL_Y = 3.0, 5.8
EXTENT = list(PF.HK_REGION)

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
STD_STEP = {"hs": 0.05, "tp": 0.5, "dm": 5.0, "elev": 0.02}
ROW_LABELS = {"hs": r"$H_\mathrm{s}$ (m)", "tp": r"$T_\mathrm{p}$ (s)",
              "dm": r"$D_\mathrm{m}$ (deg)", "elev": "Elev (m)"}
# 7.5 pt keeps mathtext subscripts above the 5 pt rendered-glyph floor
ROW_LABEL_SIZE = 7.5
COL_LABELS = ["Reference", "Ens. mean (M=10)", "s.d."]


def ax_at(fig, x, y, w, h, **kw):
    return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH], **kw)


def select_typical_ts():
    """Most ordinary hour of the test period: domain hs_p99 closest to the
    median hs_p99 of all 2021-22 hours (complete 10-member set; ties resolve
    to the earliest hour) — a representative sea state, not an extreme."""
    df = pd.read_csv(HS_CSV, parse_dates=["time"])
    med = float(df["hs_p99"].median())
    df = df.assign(dev=(df["hs_p99"] - med).abs()).sort_values(
        ["dev", "time"], kind="mergesort")
    for _, row in df.iterrows():
        ts = row["time"].strftime("%Y%m%dT%H")
        if all(os.path.exists(os.path.join(
                SAMPLES, f"sample_{ts}_ens{k:03d}.npz")) for k in range(N_ENS)):
            return ts, float(row["hs_p99"]), med
    raise RuntimeError("no typical hour with a complete ensemble found")


def load_case(ts):
    """Reference (4, N) and ensemble (M, N, 4), physical units."""
    gt = np.load(os.path.join(GT_DIR, f"wave_{ts}.npz"))
    g = np.stack([gt[v].astype(np.float64) for v in VARS])
    ens = np.stack([
        np.load(os.path.join(SAMPLES, f"sample_{ts}_ens{k:03d}.npz"))["sample"]
        .astype(np.float64) for k in range(N_ENS)])
    if ens.shape != (N_ENS, g.shape[1], 4):
        raise ValueError(f"{ts}: unexpected ensemble shape {ens.shape}")
    if not np.isfinite(ens).all() or not np.isfinite(g).all():
        raise ValueError(f"{ts}: non-finite values in GT or ensemble")
    return g, ens


def ens_stats(ens):
    """Ensemble mean/s.d. per variable, (4, N); circular for dm."""
    mean = np.empty_like(ens[0])
    std = np.empty_like(ens[0])
    for j, v in enumerate(VARS):
        e = ens[:, :, j]
        if v == "dm":
            # 10 deg and 350 deg must average to 0 deg, not 180 deg
            rad = np.deg2rad(e)
            m = np.rad2deg(np.arctan2(np.sin(rad).mean(0),
                                      np.cos(rad).mean(0))) % 360.0
            dev = np.arctan2(np.sin(rad - np.deg2rad(m)),
                             np.cos(rad - np.deg2rad(m)))
            mean[:, j], std[:, j] = m, np.rad2deg(dev.std(0))
        else:
            mean[:, j], std[:, j] = e.mean(0), e.std(0)
    return mean, std


def field_clims(v, g, m):
    """Reference|mean share one clim per variable and case."""
    if v == "hs":
        vmax = max(float(np.ceil(max(g.max(), m.max()) / 0.25) * 0.25), 0.25)
        return 0.0, vmax
    if v == "tp":
        # robust ceiling (same rule as the global figure's hs/tp): the bulk
        # of the field fills the scale, the swell tail leaves via extend
        p999 = float(np.percentile(np.concatenate([g, m]), 99.9))
        return 0.0, max(float(np.ceil(p999)), 1.0)
    if v == "dm":
        return 0.0, 360.0
    # elev: asymmetric reference p1/p99 in 0.25 m steps (main Fig. 2 recipe)
    lo = float(np.floor(np.percentile(g, 1) / 0.25) * 0.25)
    hi = float(np.ceil(np.percentile(g, 99) / 0.25) * 0.25)
    if hi - lo < 0.25:
        lo, hi = lo - 0.25, hi + 0.25
    return lo, hi


def std_clims(v, s):
    # robust ceiling (p99.9): single-node spread peaks otherwise compress the
    # whole map into the bottom of the viridis scale
    step = STD_STEP[v]
    p999 = float(np.percentile(s, 99.9))
    return 0.0, max(float(np.ceil(p999 / step) * step), step)


def draw_map(fig, x, y, lon, lat, vals, cmap, vmin, vmax):
    ax = ax_at(fig, x, y, MAP_W, MAP_H, projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    PF.grey_background_hk(ax, EXTENT)
    ax.scatter(lon, lat, c=vals, s=1.1, cmap=cmap, vmin=vmin, vmax=vmax,
               edgecolors="none", transform=ccrs.PlateCarree(), zorder=4)
    ax.spines["geo"].set_linewidth(0.6)
    return ax


def bar_extend(fields, vmin, vmax):
    """'min'/'max'/'both'/'neither' for the shared colour bar, from whether
    any plotted field actually falls outside [vmin, vmax]."""
    lo = any((f < vmin - 1e-9).any() for f in fields)
    hi = any((f > vmax + 1e-9).any() for f in fields)
    return "both" if lo and hi else "max" if hi else "min" if lo else "neither"


def hbar(fig, x, y, w, cmap, vmin, vmax, ticks=None, nbins=4,
         extend="neither"):
    cax = ax_at(fig, x, y, w, CB_H)
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(vmin, vmax), cmap),
                      cax=cax, orientation="horizontal", extend=extend)
    if ticks is not None:
        cb.set_ticks(ticks)
    else:
        cb.ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    cb.ax.tick_params(labelsize=5.0, length=1.2, width=0.5, pad=0.8)
    cb.outline.set_linewidth(0.3)
    cb.ax.minorticks_off()
    return cax


def case_title(ts, storm):
    d = f"{ts[6:8]} {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                      'Sep', 'Oct', 'Nov', 'Dec'][int(ts[4:6]) - 1]} {ts[:4]}"
    if storm:
        return f"Storm — Typhoon Chaba ({d}, {ts[9:11]} UTC)"
    return f"Typical — median sea state of 2021–22 ({d}, {ts[9:11]} UTC)"


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
    row_groups = [letters[r * 6:(r + 1) * 6] for r in range(4)]
    col_groups = [letters[c::6] for c in range(6)]
    require_matplotlib_panel_alignment(
        fig,
        json_out=OUT_BASE + ".alignment.json",
        overlay_svg=OUT_BASE + ".alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        axes=map_axes,
        panel_ids={ax: s for ax, s in zip(map_axes, letters)},
        row_groups=row_groups, column_groups=col_groups,
        exemptions=[{
            "panels": list(letters), "checks": ["horizontal-gutter"],
            "reason": "the 5 mm gap between the storm and typical "
                      "half-figures is intentional (two case blocks side by "
                      "side); repeated intra-block gutters are 2.0 mm",
        }],
    )
    print("[alignment] gate passed")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    coords = np.load(COORDS)
    lat, lon = coords[:, 0], coords[:, 1]

    typical_ts, typical_p99, median_p99 = select_typical_ts()
    print(f"typical case: {typical_ts}  hs_p99 {typical_p99:.3f} m "
          f"(test-period median {median_p99:.3f} m)")
    cases = {"storm": TS_STORM, "typical": typical_ts}

    data = {}
    for name, ts in cases.items():
        g, ens = load_case(ts)
        mean, std = ens_stats(ens)
        data[name] = (g, mean, std)
        for j, v in enumerate(VARS):
            print(f"[{name} {ts}] {v}: GT [{g[j].min():.3f}, {g[j].max():.3f}] "
                  f"mean max {mean[:, j].max():.3f} "
                  f"std max {std[:, j].max():.3f} p50 {np.median(std[:, j]):.3f}")

    fig = plt.figure(figsize=(FW * MM, FH * MM))
    map_axes, letters = [], []
    stats_rows = []

    for r, v in enumerate(VARS):
        for ci, (case, ts) in enumerate(cases.items()):
            g, mean, std = data[case]
            vmin, vmax = field_clims(v, g[r], mean[:, r])
            svmax = std_clims(v, std[:, r])[1]
            gv, mv, sv = g[r], mean[:, r], std[:, r]
            ext_f = bar_extend([gv, mv], vmin, vmax)
            ext_s = bar_extend([sv], 0.0, svmax)

            # 面板字母不再逐图出（作者: 只有 Storm / Typical 两个块各带
            # 一个字母 a / b，位置在各块标题左侧，见下方 fig.text）；
            # 这里仍给每个轴一个 id，仅供 alignment gate 使用
            for k, (vals, cm) in enumerate(
                    [(gv, FIELD_CMAP[v]), (mv, FIELD_CMAP[v]),
                     (sv, STD_CMAP)]):
                x = COL_X[case][k]
                ax = draw_map(fig, x, ROW_Y[r], lon, lat, vals, cm,
                              vmin if k < 2 else 0.0,
                              vmax if k < 2 else svmax)
                map_axes.append(ax)
                letters.append("abcdefghijklmnopqrstuvwxyz"[r * 6 + ci * 3 + k])

            # shared field bar over Ref+mean, separate s.d. bar; 2 mm indent
            # keeps the leftmost tick label clear of the next row's letters;
            # extend arrows whenever a plotted field leaves [vmin, vmax]
            # (e.g. water level beyond the reference p1/p99 band)
            y_cb = ROW_Y[r] + MAP_H + CB_BELOW
            field_ticks = ([0, 90, 180, 270, 360] if v == "dm" else None)
            hbar(fig, COL_X[case][0] + CB_INDENT, y_cb,
                 2 * MAP_W + 2.0 - 2 * CB_INDENT, FIELD_CMAP[v], vmin, vmax,
                 ticks=field_ticks, extend=ext_f)
            hbar(fig, COL_X[case][2] + CB_INDENT, y_cb,
                 MAP_W - 2 * CB_INDENT, STD_CMAP, 0.0, svmax, nbins=3,
                 extend=ext_s)

            stats_rows.append({
                "case": case, "timestamp": ts, "variable": v,
                "gt_min": gv.min(), "gt_max": gv.max(),
                "gt_p99": float(np.percentile(gv, 99)),
                "mean_max": mv.max(),
                "mean_bias": float((((mv - gv) + 180.0) % 360.0 - 180.0).mean()
                                   if v == "dm" else (mv - gv).mean()),
                "std_max": sv.max(), "std_p50": float(np.median(sv)),
                "field_vmin": vmin, "field_vmax": vmax, "std_vmax": svmax,
                "field_extend": ext_f, "std_extend": ext_s,
            })

    # titles, column labels, row labels
    for i, (case, ts) in enumerate(cases.items()):
        x0 = COL_X[case][0]
        # block label: a = storm, b = typical (整张图只有这两个字母)
        fig.text((x0 - 3.0) / FW, 1 - CASE_TITLE_Y / FH, "ab"[i],
                 fontsize=8.0, fontweight="bold", ha="left", va="baseline")
        fig.text(x0 / FW, 1 - CASE_TITLE_Y / FH,
                 case_title(ts, case == "storm"),
                 fontsize=6.8, fontweight="bold", ha="left", va="baseline")
        for k, lab in enumerate(COL_LABELS):
            fig.text((COL_X[case][k] + MAP_W / 2) / FW,
                     1 - COL_LABEL_Y / FH, lab,
                     fontsize=5.8, ha="center", va="baseline")
    for r, v in enumerate(VARS):
        fig.text(2.2 / FW, 1 - (ROW_Y[r] + MAP_H / 2) / FH, ROW_LABELS[v],
                 fontsize=ROW_LABEL_SIZE, rotation=90, ha="center",
                 va="center", rotation_mode="anchor")

    alignment_gate(fig, map_axes, letters)

    with open(os.path.join(OUT_DIR, "ed_fields_hk_stats.csv"), "w",
              newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(stats_rows[0]))
        writer.writeheader()
        writer.writerows(stats_rows)
    print("stats csv:",
          os.path.join(OUT_DIR, "ed_fields_hk_stats.csv"))

    fig.savefig(OUT_BASE + ".png", dpi=300, bbox_inches="tight",
                pad_inches=0.02)
    # submission artwork: exact 180 mm width, no tight-bbox expansion
    plt.rcParams["savefig.bbox"] = "standard"
    fig.savefig(OUT_BASE + ".pdf", dpi=300)
    plt.rcParams["savefig.bbox"] = "tight"
    print("saved:", OUT_BASE)


if __name__ == "__main__":
    main()
