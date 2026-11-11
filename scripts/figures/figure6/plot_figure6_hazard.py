#!/usr/bin/env python
"""Fig. 6 panel b draft：Saola 峰值时刻（2023-09-01T16 UTC = 00:00 HKT 2 Sep，
hybrid 堤顶插补后全段口径的领域平均 P 峰值）越浪危险度图 + HKO 记录在案
水浸点位 + 两个放大 inset。

主图：
- 官方正射影像底图（陆地显示、海面透明；common.add_orthophoto）；
- 工程岸段（894 段）按 P(q > 1 L/s/m)（M=100）着色，turbo 0–1；
  原 crest_flag==1 段（新生填海区无 DTM/SpotHeight 堤顶）已经邻段中位数
  / 形成高程默认值插补（crest_src='imputed_nb'/'imputed_def'），参与
  P 着色，其不确定性在 panel d 以斜纹区分并在 caption 声明；
- 非工程岸段浅灰；背景淡蓝 = 集合平均 Hs；
- HKO 水浸报告点位：统一实心黑圆（不再区分 confirmed/borderline，2026-11
  逐条复核来源后合并；see ragasa_hko_records.json 的 urls 字段）；两个 inset
  覆盖区的标注（Ma Wan Chung / Tung Chung / Lei Yue Mun / Tseung Kwan O）
  收进 inset，主图只留其余地名；
- 品红星 = panel d 的 3 个代表段（2143 / 6472 / 26）；
- 黑框矩形 = inset 范围，角上小字 b1 / b2。

inset（无 Hs 背景、无网格刻度、细框、左上角标签）：三个 inset 各自对准
一个代表段（星标），放大到 ~5x3 km：
- b1 Tai O（seg 2143 + HKO Tai O confirmed 点）
- b2 Kwai Chung–Stonecutters（seg 6472，无 HKO 点）
- b3 Lei Yue Mun（seg 26 + HKO Lei Yue Mun / Tseung Kwan O 点）

用法: python scripts/figures/figure6/plot_figure6_hazard.py
输出: output/plot/figure6/figure6_hazard_draft.{png,pdf}
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D

import plot_figure6_common as C

OUT_BASE = os.path.join(C.OUT_DIR, "figure6_hazard_draft" + C.OUT_SUFFIX)

# inset 定义：tag, region=[min_lon, max_lon, min_lat, max_lat],
# 主图矩形标签位置 (lon, lat)。三个 inset 各自对准一个代表段（现 panel d）
# （星标），放大到 ~5x3 km 街区尺度
_INSETS_BY_CASE = {
    "saola": [
        dict(tag="b1", region=[113.837, 113.887, 22.237, 22.269],
             tag_xy=(113.8385, 22.2705), stars=[2143]),   # Tai O
        dict(tag="b2", region=[114.092, 114.142, 22.317, 22.349],
             tag_xy=(114.0935, 22.3160), stars=[6472]),   # Kwai Chung–Stonecutters
        dict(tag="b3", region=[114.210, 114.270, 22.272, 22.308],
             tag_xy=(114.2115, 22.3095), stars=[26]),     # Lei Yue Mun
    ],
    # ragasa：三个 inset 对准代表段 8032/8657/6569（~5x4.7 km）
    # 2026-10-03：b1 原为 1570（大澳）；EurOtop 2018 + min 后大澳事件最大 P
    # 只剩 0.02，inset 会是一片空白，改用 8032（香港仔海洋公园对开护坡，P=1.0），
    # 并把对应的 HKO 确认点从主图标注挪进 inset。
    "ragasa": [
        dict(tag="b1", region=[114.138, 114.188, 22.217, 22.259],
             tag_xy=(114.139, 22.261), stars=[8032]),   # Aberdeen (Ocean Park)
        dict(tag="b2", region=[114.172, 114.222, 22.432, 22.474],
             tag_xy=(114.173, 22.476), stars=[8196]),   # Tai Po Ind. Est. (Tolo)
        dict(tag="b3", region=[114.209, 114.266, 22.263, 22.309],
             tag_xy=(114.2105, 22.3105), stars=[6569]),   # TKO Ind. Est.
    ],
    # helene：三个 inset 各对准一个 panel d 代表段（~0.05°x0.042°，与 HK 版同
    # 尺度，tag 锚在框左下角外侧）：b1 Clearwater Beach (seg 750) /
    # b2 Downtown St. Petersburg (seg 417) / b3 Port Manatee (seg 217)
    "helene": [
        dict(tag="b1", region=[-82.845, -82.795, 27.955, 27.997],
             tag_xy=(-82.844, 27.953), stars=[750]),    # Clearwater Beach
        dict(tag="b2", region=[-82.650, -82.600, 27.741, 27.783],
             tag_xy=(-82.649, 27.739), stars=[417]),    # Downtown St. Pete
        dict(tag="b3", region=[-82.590, -82.540, 27.618, 27.660],
             tag_xy=(-82.589, 27.616), stars=[217]),    # Port Manatee
    ],
}
INSETS = _INSETS_BY_CASE[C.CASE]

# 主图 HKO 标注（未被 inset 覆盖的点；inset 内的收进 inset 标注）：
# (name in json, label, dlon, dlat, ha)
_HKO_LABELS_MAIN_BY_CASE = {
    "saola": {
        # 6 pt 后字更宽，往西/北再让一点，避免压到 HKO 三角标记
        "Lau Fau Shan (Deep Bay)": ("Lau Fau Shan", -0.082, 0.010, "left"),
        "Shan Pui River, Yuen Long": ("Yuen Long", 0.012, 0.010, "left"),
        "Sha Tin - Shing Mun River": ("Sha Tin", -0.010, -0.006, "right"),
        "Tai Po (Tolo Harbour)": ("Tai Po", 0.016, 0.021, "left"),
        "Sai Kung town waterfront": ("Sai Kung", 0.014, 0.010, "left"),
        "Ma Wan Chung (Tung Chung)": ("Ma Wan Chung", 0.004, -0.0125, "left"),
        "Tung Chung": ("Tung Chung", 0.004, 0.004, "left"),
    },
    # ragasa：inset 覆盖的 Tai O / Kar Wo Lei / Sam Ka Tsuen 不在主图标注
    "ragasa": {
        "Chai Wan": ("Chai Wan", 0.001, -0.015, "left"),   # 点位移到嘉業街后下移，避开 b3 星标
        "Cheung Chau": ("Cheung Chau", -0.006, -0.013, "right"),
        # Aberdeen (Ocean Park hotel) 现在落在 inset b1 内，按约定不在主图标注
        "Tseung Kwan O South": ("Tseung Kwan O", 0.010, 0.008, "left"),
        "Sha Tin - Shing Mun River": ("Sha Tin", -0.010, -0.006, "right"),
        "Tai Po - Lam Tsuen River": ("Tai Po", -0.006, -0.032, "right"),  # b2 移到大埔后把标注挪到框外（西侧水面）；6 pt 后下移避开 Tai Po Kau 潮位站标注
    },
    # helene：主图标注部分水浸点（inset 覆盖的 Clearwater Beach 点不重复标注）
    "helene": {
        "Treasure Island": ("Treasure Island", -0.008, 0.010, "right"),
        "Madeira Beach": ("Madeira Beach", -0.008, 0.010, "right"),
        "Gulfport": ("Gulfport", -0.008, 0.010, "right"),
        "Davis Islands, Tampa": ("Davis Islands", 0.008, 0.006, "left"),
        "Pass-a-Grille, St. Pete Beach": ("Pass-a-Grille", -0.008, -0.012,
                                          "right"),
    },
}
HKO_LABELS_MAIN = _HKO_LABELS_MAIN_BY_CASE[C.CASE]
# inset 内标注（用户决定 inset 不写地名，留空）
HKO_LABELS_B1 = {}
HKO_LABELS_B2 = {}
HKO_LABELS_B3 = {}


def collect_hazard_data():
    z = C.load_overtopping("m100")
    segs = C.load_segments().set_index("seg_id")
    summary = C.load_summary()
    ids = z["seg_ids"]
    ts = list(z["timestamps"])
    i_pk = ts.index(C.T_PEAK)
    p_peak = z["p_gt1_lpsm"][:, i_pk]
    eng = segs.loc[ids]
    flag1 = (summary.loc[ids, "crest_flag"] == 1).values
    node_lats, node_lons = C.load_nodes()
    hs_bg = C.load_hs_background(C.T_PEAK)
    hko = C.load_hko()
    return dict(z=z, ids=ids, p_peak=p_peak, eng=eng, flag1=flag1,
                node_lats=node_lats, node_lons=node_lons, hs_bg=hs_bg,
                hko=hko, segs=segs)


def draw_layers(ax, d, hs=False, seg_lw=0.6, flag_lw=2.2, node_size=3.0):
    """共用图层：正射影像底图 / Hs 背景（可选）/ 非工程岸段 / 无堤顶数据
    （flag1，插补后应为空）灰段 / 工程岸段 P 着色。返回 P 的 mappable。"""
    C.add_orthophoto(ax, alpha_land=0.85, show_sea=True, official_only=True)
    if hs:
        hs_vmax = max(3.0, float(np.nanpercentile(d["hs_bg"], 97)))
        sc_hs = ax.scatter(d["node_lons"], d["node_lats"], c=d["hs_bg"],
                           s=node_size, cmap="turbo", vmin=0, vmax=hs_vmax,
                           alpha=1.0, lw=0, zorder=2,
                           transform=C.ccrs.PlateCarree())
        ax._hs_scatter = sc_hs
    non_eng = d["segs"].reset_index()
    non_eng = non_eng[~non_eng.seg_id.isin(d["ids"])]
    C.add_segment_collection(ax, non_eng, lw=seg_lw * 0.45, zorder=3,
                             color_field="#C9C9C9")
    eng, flag1, p = d["eng"], d["flag1"], d["p_peak"]
    if flag1.any():
        C.add_segment_collection(ax, eng[flag1], lw=flag_lw, zorder=4,
                                 color_field="#4D4D4D")
    # flag1 段 P 不可靠（堤顶钳 1.5 m 下界），不参与 P 着色，只留深灰底
    eng_ok, p_ok = eng[~flag1], p[~flag1]
    lc = C.add_segment_collection(ax, eng_ok, colors=p_ok,
                                  lw=seg_lw, zorder=5, cmap="plasma",
                                  vmin=0, vmax=1)
    return lc


def draw_hko(ax, d, label_map, ms=3.0, marker_only=()):
    """HKO 水浸报告点位（统一实心圆）；label_map 外的名字只画 marker。"""
    for f in d["hko"]["flooding_locations"]:
        ax.plot(f["lon"], f["lat"], marker="o", ms=ms, mfc="#111111",
                mec="#111111", mew=0.9, zorder=7,
                transform=C.ccrs.PlateCarree())
        if f["name"] in label_map:
            lab, dx, dy, ha = label_map[f["name"]]
            # 5 pt 是投稿底线：原来 4 pt 在成品尺寸下几乎不可读，提到 6 pt
            # 与其他地图地名（Hong Kong Island / Victoria Harbour）同一档
            ax.text(f["lon"] + dx, f["lat"] + dy, lab, fontsize=6, ha=ha,
                    va="center", color="#111111",
                    transform=C.ccrs.PlateCarree(), zorder=8,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white",
                              ec="none", alpha=0.75))


# 潮位站标注偏移（name -> (dlon, dlat, ha)）；与 HKO 水浸点区分开：
# 三角符号。saola/ragasa 站位不同，按需给偏移（缺省右上角）。
_GAUGE_LABELS_BY_CASE = {
    "saola": {},
    "ragasa": {
        "Shek Pik": (-0.008, -0.008, "right"),
        "Quarry Bay": (-0.008, 0.006, "right"),
        "Tai Miu Wan": (0.008, 0.006, "left"),
        "Tai Po Kau": (-0.014, 0.007, "right"),   # 标注放在 b2 框左侧、三角符号上方
        "Tsim Bei Tsui": (-0.008, 0.006, "right"),
    },
    "helene": {
        "Clearwater Beach": (-0.010, 0.008, "right"),
        "St. Petersburg": (-0.010, 0.006, "right"),
        "Port Manatee": (0.008, 0.006, "left"),
        "East Bay": (0.008, 0.008, "left"),
    },
}
GAUGE_LABELS = _GAUGE_LABELS_BY_CASE[C.CASE]


def draw_gauges(ax, d, label_map=None, ms=3.2):
    """HKO 潮位站（三角符号，panel e 水位对比的位置索引）。"""
    label_map = GAUGE_LABELS if label_map is None else label_map
    for g in d["hko"]["tide_gauge_maxima"]:
        ax.plot(g["lon"], g["lat"], marker="^", ms=ms, mfc="white",
                mec="#B2182B", mew=1.1, zorder=7,
                transform=C.ccrs.PlateCarree())
        if g["station"] in label_map:
            dx, dy, ha = label_map[g["station"]]
            ax.text(g["lon"] + dx, g["lat"] + dy, g["station"], fontsize=6,
                    ha=ha, va="center", color="#B2182B", zorder=8,
                    transform=C.ccrs.PlateCarree(),
                    bbox=dict(boxstyle="round,pad=0.08", fc="white",
                              ec="none", alpha=0.75))


def draw_stars(ax, d, seg_ids=None, ms=4.5):
    """代表段（现 panel d，PANEL_C_SEGS 为历史命名）品红星；seg_ids=None 画全部。"""
    for spec in C.PANEL_C_SEGS:
        if seg_ids is not None and spec["seg_id"] not in seg_ids:
            continue
        r = d["segs"].loc[spec["seg_id"]]
        ax.plot(r.mid_lon, r.mid_lat, marker="*", ms=ms, mfc="#FF00FF",
                mec="black", mew=0.5, zorder=7,
                transform=C.ccrs.PlateCarree())


def draw_inset_rectangles(ax):
    """主图上画 inset 范围矩形 + b1/b2 角标。"""
    for spec in INSETS:
        r = spec["region"]
        ax.add_patch(mpatches.Rectangle(
            (r[0], r[2]), r[1] - r[0], r[3] - r[2], fill=False,
            edgecolor="black", lw=0.8, ls="--", zorder=9,
            transform=C.ccrs.PlateCarree()))
        ax.text(spec["tag_xy"][0], spec["tag_xy"][1], spec["tag"],
                fontsize=6, fontweight="bold", ha="left", va="bottom",
                zorder=9, transform=C.ccrs.PlateCarree(),
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="black",
                          lw=0.5))


def draw_inset(ax, d, spec):
    """单个 inset：与主图相同的数据层（含 Hs 散点背景），细框 + b1/b2/b3 标签。"""
    label_map = {"b1": HKO_LABELS_B1, "b2": HKO_LABELS_B2,
                 "b3": HKO_LABELS_B3}[spec["tag"]]
    draw_layers(ax, d, hs=True, seg_lw=0.85, flag_lw=2.8, node_size=12.0)
    draw_hko(ax, d, label_map, ms=2.6)
    draw_gauges(ax, d, label_map={}, ms=2.4)   # inset 内只画符号不写站名
    draw_stars(ax, d, spec["stars"], ms=6.0)   # 标出 inset 对准的 panel d 代表段
    ax.text(0.02, 0.98, spec["tag"], transform=ax.transAxes, fontsize=7,
            fontweight="bold", ha="left", va="top", zorder=10,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.8))


def panel_b_legend_handles():
    return [
        Line2D([0], [0], marker="o", ls="", ms=3.5, mfc="#111111",
               mec="#111111", label=C.FLOOD_LEGEND),
        Line2D([0], [0], marker="*", ls="", ms=6, mfc="#FF00FF",
               mec="black", mew=0.5, label="panel d segment"),
        Line2D([0], [0], marker="^", ls="", ms=4, mfc="white",
               mec="#B2182B", mew=1.1, label=C.GAUGE_LEGEND),
    ]


def panel_b_legend(fig, x, y_top, canvas, ncol=1, center_x=None):
    """主图 panel b 图例（图级定位，ncol=1 单列 / ncol=3 横排）。

    center_x（mm）给定时，整行图例按**可见内容**居中于该 x（渲染后量墨迹宽度
    再反算锚点，与 plot_figure6_coastline.panel_a_legend 同一做法）。"""
    cw, ch = canvas
    leg = fig.legend(handles=panel_b_legend_handles(), loc="upper left", ncol=ncol,
                     frameon=False, fontsize=6, handlelength=1.2,
                     labelspacing=0.3, borderpad=0.3, columnspacing=0.9,
                     bbox_to_anchor=(x / cw, 1 - y_top / ch))
    if center_x is not None:
        try:
            import numpy as np
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
            leg.set_bbox_to_anchor(((x + (center_x - ink_cx)) / cw, 1 - y_top / ch),
                                   transform=fig.transFigure)
        except Exception:                      # 非 Agg 后端时退回左对齐
            pass
    return leg


def draw_panel_b(ax, d, legend=True):
    """峰值危险度主图（主图 panel b / 独立 draft 共用）。返回 P mappable。"""
    lc = draw_layers(ax, d, hs=True)
    draw_hko(ax, d, HKO_LABELS_MAIN)
    draw_gauges(ax, d)
    draw_stars(ax, d)
    draw_inset_rectangles(ax)
    ax.set_title(C.PANEL_B_TITLE, fontsize=7, pad=2.5)
    if legend:
        ax.legend(handles=panel_b_legend_handles(), frameon=True,
                  framealpha=0.85, edgecolor="#DDDDDD", loc="lower right",
                  fontsize=6, handlelength=1.5, borderpad=0.5,
                  labelspacing=0.35)
    return lc


def add_colorbars(fig, ax, lc, map_x, map_y_top, map_w, map_h, canvas):
    """P 色标 + Hs 色标：两条横条并排一行放在地图下方，标签在各自下方。
    两条色标作为一组相对主图水平居中（地图太窄放不下时退回左边距 2 mm）。"""
    cw, ch = canvas
    cb_w = min(34.0, (map_w - 22) / 2)
    y1 = map_y_top + map_h + 5.0
    group_w = 2 * cb_w + 18.0
    x1 = map_x + max(2.0, (map_w - group_w) / 2)
    cba = fig.add_axes([x1 / cw, 1 - (y1 + 1.2) / ch, cb_w / cw, 1.2 / ch])
    cb = fig.colorbar(lc, cax=cba, orientation="horizontal")
    cb.set_label("$P(q > 1$ L/s/m)", fontsize=6, labelpad=1.5)
    cb.ax.tick_params(labelsize=6, length=2)
    x2 = x1 + cb_w + 18.0
    cbb = fig.add_axes([x2 / cw, 1 - (y1 + 1.2) / ch, cb_w / cw, 1.2 / ch])
    # hs_bg 以 p97 截断保对比度（ragasa max 8.45 m > vmax 7.3）：extend 箭头
    # 声明存在超出色标上限的值
    cbh = fig.colorbar(ax._hs_scatter, cax=cbb, orientation="horizontal",
                       extend="max")
    cbh.set_label("ens.-mean $H_s$ (m)", fontsize=6, labelpad=1.5)
    cbh.ax.tick_params(labelsize=6, length=2)


def main():
    C.setup_style()
    os.makedirs(C.OUT_DIR, exist_ok=True)
    d = collect_hazard_data()

    canvas = (140.0, 128.0)
    MM = 1 / 25.4
    fig = plt.figure(figsize=(canvas[0] * MM, canvas[1] * MM))
    # 主图
    ax, h = C.new_map_ax(fig, 14, 8, 80, canvas=canvas)
    lc = draw_panel_b(ax, d)
    add_colorbars(fig, ax, lc, 14, 8, 80, h, canvas)
    # insets
    y0 = 8 + h + 17.0
    iw = 52.0
    for i, spec in enumerate(INSETS):
        axi, _ = C.new_map_ax(fig, 14 + i * (iw + 8), y0, iw,
                              canvas=canvas, region=spec["region"],
                              gridlines=False)
        draw_inset(axi, d, spec)
    C.letter(fig, 3, 3.5, "b", canvas=canvas)

    fig.savefig(OUT_BASE + ".png", dpi=300)
    fig.savefig(OUT_BASE + ".pdf")
    print("saved:", OUT_BASE + ".png/.pdf")


if __name__ == "__main__":
    main()
