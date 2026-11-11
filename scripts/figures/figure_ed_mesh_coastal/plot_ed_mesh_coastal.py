#!/usr/bin/env python
"""Extended Data figure: the mesh is refined towards the coast, and that is
where the accuracy advantage is largest.

Two blocks of three panels.  The left column describes the unstructured mesh:
median node spacing and the share of nodes in each coastal-distance band.  The
middle and right columns show the per-node RMSE of significant wave height in
the same distance bands, globally and in Hong Kong, together with the ratio of
each baseline to WaveFlow.

Panels
  a  median node spacing (and interquartile range) vs distance to the coast
  b  global median per-node RMSE of Hs vs distance, per model
  c  the same for the Hong Kong mesh
  d  share of the global mesh nodes in each distance band
  e  global median ratio of each baseline's per-node RMSE to WaveFlow's
  f  the same for the Hong Kong mesh

Layout: the six panels share one x variable, so the axis title is printed once
per column (on the bottom panel) instead of six times, and every glyph is at
5 pt or more (the earlier version carried 4.4-4.8 pt legends, below the floor).

Data (all cached)
  data/global_hindcast/global_coordinates.npy
  data/global_hindcast/ocean_boundary_wkb.npz
  earthformer_repro/.../comparison_cfm/spatial_rmse.npz        ('cfm','ef')
  oceancastnet_repro/.../comparison_cfm/spatial_rmse.npz       ('ocn')
  output/results/si_coastal_global/vit_global_sqerr.npz
  output/results/si_coastal_hk/{waveflow,vit,ef,ocn}_sqerr.npz
  data/hindcast/hk_coordinates.npy + GSHHS (cartopy cache)

OUTPUT
  output/plot/figure_ed_mesh_coastal/figure_ed_mesh_coastal.{png,pdf,svg}
  output/plot/figure_ed_mesh_coastal/ed_mesh_coastal.csv

USAGE
  python scripts/figures/figure_ed_mesh_coastal/plot_ed_mesh_coastal.py
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
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "figures",
                                "figure_si_mesh_resolution"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "figures",
                                "figure_si_coastal"))
import plot_mesh_resolution as MESH          # noqa: E402
import plot_si_coastal_error as COAST        # noqa: E402

OUT_DIR = os.path.join(ROOT, "output/plot/figure_ed_mesh_coastal")
OUT_BASE = os.path.join(OUT_DIR, "figure_ed_mesh_coastal")

FW, FH = 180.0, 113.0
MM = 1 / 25.4

C_OURS, C_VIT = COAST.C_OURS, COAST.C_VIT
C_EF, C_OCN = COAST.C_EF, COAST.C_OCN
# 两个网格（不是两个模型）用一套独立配色，避免与 b/c/e/f 的模型色混淆
C_GMESH, C_GMESH_BAND = MESH.C_MAIN, MESH.C_BAND
C_HKMESH, C_HKMESH_BAND = "#B2182B", "#F0BDB6"

FS_LETTER = 8.0
FS_TAG = 6.4
FS_LEGEND = 5.2
FS_NOTE = 5.2

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                        "DejaVu Sans"],
    "font.size": 6.8, "axes.labelsize": 6.6, "axes.titlesize": 6.6,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
    "legend.fontsize": FS_LEGEND,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.bbox": None,
})

MODELS = [("ours", "WaveFlow", C_OURS, "o"),
          ("vit", "WaveFlow-ViT", C_VIT, "D"),
          ("ef", "EarthFormer", C_EF, "s"),
          ("ocn", "OceanCastNet", C_OCN, "^")]
RATIOS = [("ratio_vit", "WaveFlow-ViT", C_VIT, "D"),
          ("ratio_ef", "EarthFormer", C_EF, "s"),
          ("ratio_ocn", "OceanCastNet", C_OCN, "^")]

XLABEL = "distance to nearest coastline (km)"


def ax_at(fig, x, y, w, h, **kw):
    return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH], **kw)


def plain_log_x(ax):
    """Plain '1, 10, 100' tick labels on a log axis: the default 10**k
    formatter renders the exponent at 0.7x, below the 5 pt glyph floor."""
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def panel(ax, letter, tag=""):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(-5.8, 3.5), textcoords="offset points",
                ha="left", va="bottom", fontsize=FS_LETTER, fontweight="bold")
    if tag:
        ax.annotate(tag, xy=(0, 1), xycoords="axes fraction",
                    xytext=(1.8, 3.5), textcoords="offset points",
                    ha="left", va="bottom", fontsize=FS_TAG, color="#333333")


def load():
    spacing, dist, mesh_rows, coast_len = MESH.analyse()
    # re-bin the mesh statistics on the same bands as the error panels so that
    # every panel of the figure shares one set of coastal-distance bins
    mesh_rows = []
    for (lo, hi), label in zip(COAST.GLOBAL_BANDS, COAST.GLOBAL_LABELS):
        m = (dist >= lo) & (dist < hi)
        mesh_rows.append(dict(
            band=label, lo=lo, hi=(None if np.isinf(hi) else hi),
            n_nodes=int(m.sum()), share_pct=100.0 * m.mean(),
            median_spacing_km=float(np.median(spacing[m])),
            median_distance_km=float(np.median(dist[m])),
            q25_spacing_km=float(np.percentile(spacing[m], 25)),
            q75_spacing_km=float(np.percentile(spacing[m], 75))))
    # Hong Kong mesh geometry on the same recipe: nearest-neighbour spacing over
    # the 7,788-node set, distance to the same GSHHS coastline
    hk_coords = np.load("data/hindcast/hk_coordinates.npy")      # (lat, lon)
    hk_xyz = MESH.to_xyz(hk_coords[:, 1], hk_coords[:, 0])
    hk_step, _ = cKDTree(hk_xyz).query(hk_xyz, k=2)
    hk_spacing = hk_step[:, 1]
    hk_dist = COAST.hk_distance()
    hk_rows = []
    for (lo, hi), label in zip(COAST.HK_BANDS, COAST.HK_LABELS):
        m = (hk_dist >= lo) & (hk_dist < hi)
        if not m.any():
            continue
        hk_rows.append(dict(
            band=label, n_nodes=int(m.sum()), share_pct=100.0 * m.mean(),
            median_spacing_km=float(np.median(hk_spacing[m])),
            median_distance_km=float(np.median(hk_dist[m])),
            q25_spacing_km=float(np.percentile(hk_spacing[m], 25)),
            q75_spacing_km=float(np.percentile(hk_spacing[m], 75))))
    # the HK mesh re-binned onto the global bands, so that the share panel can
    # compare the two meshes band by band
    hk_share = [100.0 * float(((hk_dist >= lo) & (hk_dist < hi)).mean())
                for lo, hi in COAST.GLOBAL_BANDS]
    ef = np.load(COAST.EF_NPZ)
    g_arr = {"ours": ef["cfm"], "ef": ef["ef"],
             "ocn": np.load(COAST.OCN_NPZ)["ocn"]}
    if os.path.exists(COAST.GLOBAL_VIT_NPZ):
        gv = np.load(COAST.GLOBAL_VIT_NPZ)
        g_arr["vit"] = np.sqrt(gv["sqerr"] / np.maximum(gv["count"], 1))
    g_rows = COAST.bin_stats(dist, g_arr, COAST.GLOBAL_BANDS,
                             COAST.GLOBAL_LABELS)
    hk = {}
    for name in ("waveflow", "vit", "ef", "ocn"):
        d = np.load(COAST.HK_NPZ.format(name))
        hk[name] = np.sqrt(d["sqerr"] / np.maximum(d["count"], 1))
    h_arr = {"ours": hk["waveflow"], "vit": hk["vit"], "ef": hk["ef"],
             "ocn": hk["ocn"]}
    h_rows = COAST.bin_stats(hk_dist, h_arr, COAST.HK_BANDS,
                             COAST.HK_LABELS)
    return mesh_rows, hk_rows, hk_share, g_rows, h_rows


def draw_mesh_spacing(ax, rows, hk_rows, xlabel=False):
    """Median node spacing (line) and IQR (band) against coastal distance.

    Both meshes are drawn in the same panel: the global 651,652-node mesh (blue)
    and the Hong Kong 7,788-node mesh (red). Each curve uses the distance bands
    of its own mesh, because the two domains resolve different scales; the x
    positions are the median node distance of each band, so the two curves are
    directly readable on one log axis.
    """
    for rr, colour, band_colour, label, marker in (
            (rows, C_GMESH, C_GMESH_BAND, "Global mesh", "o"),
            (hk_rows, C_HKMESH, C_HKMESH_BAND, "Hong Kong mesh", "s")):
        x = np.array([r["median_distance_km"] for r in rr])
        med = np.array([r["median_spacing_km"] for r in rr])
        q25 = np.array([r["q25_spacing_km"] for r in rr])
        q75 = np.array([r["q75_spacing_km"] for r in rr])
        ax.fill_between(x, q25, q75, color=band_colour, alpha=0.60, lw=0)
        ax.plot(x, med, "-", marker=marker, color=colour, ms=2.6,
                markeredgecolor="white", markeredgewidth=0.3, label=label)
    # 均匀网格参考线：0.25°(28 km) 与 1/12°(9.3 km)，都取子午向格距
    for km, label in ((MESH.UNIFORM_025_DEG_KM, "0.25\u00b0 (28 km)"),
                      (MESH.UNIFORM_025_DEG_KM / 3.0, "1/12\u00b0 (9.3 km)")):
        ax.axhline(km, color=MESH.C_REF, ls=(0, (4, 2)), lw=0.8)
        # 标签放在 x≈1.45 km 起（落在阴影带 bbox 内，碰撞审计视为 contained）；
        # 文字压到 1.2 个数量级以内，右端停在被测曲线抬升之前
        ax.text(1.45, km * 1.12, label, fontsize=FS_NOTE, color=MESH.C_REF)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_x(ax)
    ax.set_xlim(0.08, 1500)
    ax.set_ylim(0.2, 90)
    ax.set_yticks([0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0.3", "1", "3", "10", "30"])
    ax.set_ylabel("node spacing (km)")
    ax.legend(frameon=False, loc="lower right", handlelength=1.0,
              borderaxespad=0.3, labelspacing=0.25, fontsize=FS_LEGEND)
    if xlabel:
        ax.set_xlabel(XLABEL)
    tidy(ax)


def draw_mesh_share(ax, rows, hk_share, xlabel=True):
    """Share of each mesh's nodes in every coastal-distance band.

    The Hong Kong mesh is re-binned onto the global bands so that the two bars
    of a band are the same quantity (share of that mesh's nodes) and are
    therefore directly comparable.
    """
    share = np.array([r["share_pct"] for r in rows])
    hk = np.array(hk_share)
    pos = np.arange(len(rows))
    w = 0.38
    ax.bar(pos - w / 2, share, width=w, color=MESH.C_BAR, edgecolor=C_GMESH,
           lw=0.5, label="Global mesh")
    ax.bar(pos + w / 2, hk, width=w, color=C_HKMESH_BAND, edgecolor=C_HKMESH,
           lw=0.5, label="Hong Kong mesh")
    # no value labels: the two 0-5 km bars are adjacent, so 5.2 pt numbers cannot
    # clear both outlines; the numbers the panel argues are in the caption
    ax.set_xticks(pos)
    # vertical band labels: 45-degree ones collide in a 44 mm panel, and an
    # explicit "anchor" alignment pushes them up into the bars
    ax.set_xticklabels([r["band"] for r in rows], rotation=90, fontsize=5.4)
    # linear share axis (a log axis was tried on 2026-11-09 and rejected: the
    # bar baseline stops being zero, and the bands with no Hong Kong node
    # cannot be drawn at all)
    ax.set_ylim(0, max(share.max(), np.nanmax(hk)) * 1.16)
    ax.set_ylabel("share of nodes (%)")
    ax.legend(frameon=False, loc="upper right", handlelength=1.0,
              borderaxespad=0.3, labelspacing=0.25, fontsize=FS_LEGEND)
    if xlabel:
        ax.set_xlabel(XLABEL)
    tidy(ax)


def draw_rmse(ax, rows, series, with_legend=None, xlabel=False):
    x = np.array([r["median_distance_km"] for r in rows])
    for key, label, colour, marker in series:
        if f"median_{key}" not in rows[0]:
            continue
        ax.plot(x, [r[f"median_{key}"] for r in rows], marker=marker, ms=2.4,
                lw=0.9, color=colour, markeredgecolor="white",
                markeredgewidth=0.3, label=label)
    ax.set_xscale("log")
    plain_log_x(ax)
    ax.set_ylabel("median per-node\nRMSE of Hs (m)")
    if with_legend:
        ax.legend(frameon=False, loc=with_legend, handlelength=1.1,
                  borderaxespad=0.25, labelspacing=0.2, fontsize=FS_LEGEND)
    if xlabel:
        ax.set_xlabel(XLABEL)
    tidy(ax)


def draw_ratio(ax, rows, with_legend=None, xlabel=True):
    x = np.array([r["median_distance_km"] for r in rows])
    ax.axhline(1.0, color="#888888", lw=0.6, ls="--", zorder=0)
    for key, label, colour, marker in RATIOS:
        y = np.array([r[key] for r in rows], dtype=float)
        if not np.isfinite(y).any():
            continue
        ax.plot(x, y, marker=marker, ms=2.4, lw=0.9, color=colour,
                markeredgecolor="white", markeredgewidth=0.3, label=label)
    ax.set_xscale("log")
    plain_log_x(ax)
    ax.set_ylabel("median ratio\nto WaveFlow")
    if with_legend:
        ax.legend(frameon=False, loc=with_legend, handlelength=1.1,
                  borderaxespad=0.25, labelspacing=0.2, fontsize=FS_LEGEND)
    if xlabel:
        ax.set_xlabel(XLABEL)
    tidy(ax)


# final layout in mm: 3 columns x 2 rows of equal 44 x 41 mm panels
COLS = (10.0, 70.0, 130.0)
ROWS_Y = (9.0, 58.0)
PW, PH = 44.0, 41.0


def build_figure(mesh_rows, hk_rows, hk_share, g_rows, h_rows):
    fig = plt.figure(figsize=(FW * MM, FH * MM))
    axes = {}

    axes["a"] = ax = ax_at(fig, COLS[0], ROWS_Y[0], PW, PH)
    draw_mesh_spacing(ax, mesh_rows, hk_rows)
    panel(ax, "a")

    axes["b"] = ax = ax_at(fig, COLS[1], ROWS_Y[0], PW, PH)
    draw_rmse(ax, g_rows, MODELS, with_legend="lower right")
    panel(ax, "b", "Global")

    axes["c"] = ax = ax_at(fig, COLS[2], ROWS_Y[0], PW, PH)
    draw_rmse(ax, h_rows, MODELS)
    panel(ax, "c", "Hong Kong")

    axes["d"] = ax = ax_at(fig, COLS[0], ROWS_Y[1], PW, PH)
    draw_mesh_share(ax, mesh_rows, hk_share)
    panel(ax, "d")

    axes["e"] = ax = ax_at(fig, COLS[1], ROWS_Y[1], PW, PH)
    draw_ratio(ax, g_rows, with_legend="upper right")
    panel(ax, "e", "Global")

    axes["f"] = ax = ax_at(fig, COLS[2], ROWS_Y[1], PW, PH)
    draw_ratio(ax, h_rows)
    panel(ax, "f", "Hong Kong")
    return fig, axes


def run_alignment_gate(fig, axes, base):
    """Optional render-time gate; enabled by pointing NATURE_QA_SCRIPTS at the
    panel-alignment auditor directory."""
    scripts = os.environ.get("NATURE_QA_SCRIPTS")
    if not scripts:
        return
    sys.path.insert(0, scripts)
    from audit_panel_alignment import require_matplotlib_panel_alignment

    require_matplotlib_panel_alignment(
        fig,
        axes=[axes[k] for k in "abcdef"],
        panel_ids=list("abcdef"),
        row_groups=[["a", "b", "c"], ["d", "e", "f"]],
        column_groups=[["a", "d"], ["b", "e"], ["c", "f"]],
        json_out=base + ".alignment.json",
        overlay_svg=base + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mesh_rows, hk_rows, hk_share, g_rows, h_rows = load()

    with open(os.path.join(OUT_DIR, "ed_mesh_coastal.csv"), "w",
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

    # geometry table behind panels a and d (both meshes)
    with open(os.path.join(OUT_DIR, "ed_mesh_geometry.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["domain", "band_km", "n_nodes", "share_pct",
                    "median_distance_km", "median_spacing_km",
                    "q25_spacing_km", "q75_spacing_km"])
        for dom, rows in (("global", mesh_rows), ("hongkong", hk_rows)):
            for r in rows:
                w.writerow([dom, r["band"], r["n_nodes"],
                            f'{r["share_pct"]:.2f}',
                            f'{r["median_distance_km"]:.3f}',
                            f'{r["median_spacing_km"]:.4f}',
                            f'{r["q25_spacing_km"]:.4f}',
                            f'{r["q75_spacing_km"]:.4f}'])
        for r, sh in zip(mesh_rows, hk_share):
            w.writerow(["hongkong_on_global_bands", r["band"], "", f"{sh:.2f}",
                        r["median_distance_km"], "", "", ""])

    fig, axes = build_figure(mesh_rows, hk_rows, hk_share, g_rows, h_rows)
    run_alignment_gate(fig, axes, OUT_BASE)
    fig.savefig(OUT_BASE + ".png", dpi=600)
    fig.savefig(OUT_BASE + ".pdf")
    fig.savefig(OUT_BASE + ".svg")
    plt.close(fig)
    print("saved:", OUT_BASE)


if __name__ == "__main__":
    main()
