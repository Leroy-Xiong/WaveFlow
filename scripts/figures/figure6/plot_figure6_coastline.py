#!/usr/bin/env python
"""Fig. 6 panel a draft：香港海岸线分段与防御类型图。

9,083 个 ~150 m 分段按 coast_type 着色（seawall_vertical 824 /
revetment_slope 70 / natural_rocky 6,240 / natural_sandy 1,427 /
lowlying_open 522；已剔除城门河 56 个 OSM 误标段，见 common.load_segments），叠加 7,788 个模型节点（浅灰点）示网格覆盖，
标注地名（Hong Kong Island / Victoria Harbour / Lantau / Tai O /
Tolo Harbour / Cheung Chau）。底图 = 港府官方正射影像（LandsD GeoData
Store 影像瓦片，陆地显示、海面透明留白；common.add_orthophoto，
无缓存时自动退化为无底图）。

用法: python scripts/figures/figure6/plot_figure6_coastline.py
输出: output/plot/figure6/figure6_coastline_draft.{png,pdf}
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import plot_figure6_common as C

OUT_BASE = os.path.join(C.OUT_DIR, "figure6_coastline_draft")

# 绘制顺序/线宽：HK 五类为默认；helene 的堤顶分类三条目在 C.PANEL_A_TYPES
# 里（深色高脊在底、低洼蓝在最上），线宽并入同一张表
DRAW_ORDER = C.PANEL_A_TYPES
DRAW_LW = {"natural_rocky": 0.45, "natural_sandy": 0.6,
           "lowlying_open": 0.7, "revetment_slope": 1.1,
           "seawall_vertical": 1.1,
           "high_ridge": 0.45, "dune_berm": 0.6, "low_lying": 0.7}

# 地名 (lon, lat, text, ha)；可选第 5 元 = 引线指向 (lon, lat)（案例化）
PLACES = C.PLACES


def panel_a_legend_handles(segs):
    n_nodes = len(C.load_nodes()[0])
    handles = [Line2D([0], [0], marker="o", ls="", ms=2.5,
                      mfc="#BFBFBF", mec="none",
                      label=f"model node ({n_nodes:,})")]
    for ct in C.PANEL_A_TYPES:
        n = int((segs.coast_type == ct).sum())
        handles.append(Line2D([0], [0], color=C.COAST_COLORS[ct],
                              lw=1.4,
                              label=f"{C.COAST_LABELS[ct]} ({n:,})"))
    return handles


def panel_a_legend(fig, x, y_top, canvas, segs, center_x=None):
    """主图 panel a 图例（图级双列，放地图下方空档）。

    center_x（mm）给定时，图例作为整体以该 x 为中心水平居中。居中用的是
    **图例内可见元素（把手 + 文字）的并集包围盒**，而不是 legend 自身的框
    ——把手框比实际线段宽，按框居中会让可见内容整体右偏约 0.6 mm。"""
    cw, ch = canvas
    leg = fig.legend(handles=panel_a_legend_handles(segs), loc="upper left",
                     ncol=2, frameon=False, fontsize=6, handlelength=1.2,
                     columnspacing=2.4, labelspacing=0.28, borderpad=0.2,
                     bbox_to_anchor=(x / cw, 1 - y_top / ch))
    if center_x is not None:
        try:
            fig.canvas.draw()
            bb = leg.get_window_extent(fig.canvas.get_renderer())
            buf = np.asarray(fig.canvas.buffer_rgba())
            h_px, w_px = buf.shape[:2]
            r0 = max(int(h_px - bb.y1) - 2, 0)
            r1 = min(int(h_px - bb.y0) + 3, h_px)
            c0 = max(int(bb.x0) - 2, 0)
            c1 = min(int(bb.x1) + 3, w_px)
            ink = buf[r0:r1, c0:c1, :3].min(axis=2) < 200
            cols = np.where(ink.any(axis=0))[0]
            ink_cx = (c0 + (cols.min() + cols.max()) / 2) * cw / w_px
            shift = center_x - ink_cx
            leg.set_bbox_to_anchor((x / cw + shift / cw, 1 - y_top / ch),
                                   transform=fig.transFigure)
        except Exception:                      # 非 Agg 后端时退回左对齐
            pass
    return leg


def draw_panel_a(ax, segs, node_lats, node_lons, legend=True):
    """海岸线类型图（主图 panel a / 独立 draft 共用）。"""
    C.add_orthophoto(ax, alpha_land=0.85, show_sea=True, official_only=True)      # 官方正射影像底图（无缓存则跳过）
    ax.scatter(node_lons, node_lats, s=0.25, c="#BFBFBF", alpha=0.5,
               lw=0, zorder=1, transform=C.ccrs.PlateCarree(),
               rasterized=True)
    for ct in DRAW_ORDER:
        sub = segs[segs.coast_type == ct]
        C.add_segment_collection(ax, sub, lw=DRAW_LW[ct],
                                 zorder=3 if ct.startswith("natural")
                                          or ct == "lowlying_open" else 4,
                                 color_field=C.COAST_COLORS[ct])
    for lon, lat, name, ha, target in PLACES:
        ax.text(lon, lat, name, fontsize=6, style="italic", color="#222222",
                ha=ha, va="center", transform=C.ccrs.PlateCarree(), zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                          alpha=0.7))
        if target is not None:
            tail = ((lon + target[0]) / 2, (lat + target[1]) / 2)
            ax.annotate("", xy=target, xytext=tail,
                        xycoords=C.ccrs.PlateCarree(),
                        textcoords=C.ccrs.PlateCarree(),
                        arrowprops=dict(arrowstyle="-", lw=0.5,
                                        color="#222222"), zorder=6)
    ax.set_title(f"{C.PANEL_A_TITLE} ($n$ = {len(segs):,})",
                 fontsize=7, pad=2.5)
    if legend:
        ax.legend(handles=panel_a_legend_handles(segs), frameon=True,
                  framealpha=0.85,
                  edgecolor="#DDDDDD", loc="upper left", fontsize=5,
                  handlelength=1.2, borderpad=0.35, labelspacing=0.25)


def main():
    C.setup_style()
    os.makedirs(C.OUT_DIR, exist_ok=True)
    segs = C.load_segments()
    node_lats, node_lons = C.load_nodes()

    canvas = (110.0, 90.0)
    MM = 1 / 25.4
    fig = plt.figure(figsize=(canvas[0] * MM, canvas[1] * MM))
    ax, h = C.new_map_ax(fig, 14, 8, 88, canvas=canvas)
    draw_panel_a(ax, segs, node_lats, node_lons)
    C.letter(fig, 3, 3, "a", canvas=canvas)

    fig.savefig(OUT_BASE + ".png", dpi=300)
    fig.savefig(OUT_BASE + ".pdf")
    print("saved:", OUT_BASE + ".png/.pdf")


if __name__ == "__main__":
    main()
