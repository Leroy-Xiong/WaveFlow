#!/usr/bin/env python
"""Fig. 6 主图合成：生成式集合驱动的香港海岸线越浪概率预警（Saola 2023，
能力展示 demonstration, not validation）。

布局（180 × 206 mm，毫米绝对定位，panel 字母 a–d）：
  行 1  a 海岸线分段类型图（9,139 段 + 7,788 节点）
        b Saola 领域峰值时刻（2023-09-01T16 UTC = 00:00 HKT 2 Sep）P(q>1) 危险度
          图 + HKO 水浸点位 + 集合平均 Hs 淡背景（a:b 列宽 4:6）
  行 2  c 暴露汇总（event-max P≥0.5 的工程岸线 km / 占比 %，M=10 vs M=100，
          按 crest 置信度拆分；panel a 下方）
  行 3  d 4 个代表段的越浪 q 时序（M=100：q 带 + 中位数 + P(q>1) 右轴）
  行 4  e 同段最近 HKO 潮位站的模型水位带 vs 实测（十号信号/最接近时刻标注）
        d/e 在预报数据齐备时（现仅 Ragasa 版）叠加一条**单发布时次的 IFS ENS
        逐成员预报**：紫色实线+圆点 = 集合中位数，点线 = 预报 P(q>1)，
        图例另起一行（画布相应加高 FC_LEGEND_H）。

各 panel 绘制函数由同目录 draft 模块提供（plot_figure6_coastline/_hazard/
_evolution/_exposure），本文件只负责定位与装配。

用法: python scripts/figures/figure6/plot_figure6_main.py
输出: output/plot/figure6/figure6_main.{png,pdf}
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))
os.chdir(_REPO_ROOT)
sys.path.insert(0, _DIR)
os.environ.setdefault("CARTOPY_DATA_DIR", os.path.join(_REPO_ROOT,
                                                       "data/cartopy_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_figure6_common as C
import plot_figure6_coastline as CA
import plot_figure6_hazard as HAZ
import plot_figure6_evolution as EV
import plot_figure6_exposure as EX

sys.path.insert(0, os.path.dirname(_DIR))          # scripts/figures
import typography as TY  # noqa: E402

OUT_BASE = os.path.join(C.OUT_DIR, "figure6_main" + C.OUT_SUFFIX)

CANVAS = (180.0, 197.0)   # 底部内容到 194.3 mm，画布收到 197（下留白 ~2.7 mm）
FC_LEGEND_H = 3.0         # 预报图例第二行额外占用的画布高度（mm；无标题）
# 排布按**文字墨迹**量（图例盒子自带 ~1 mm 上下内边距，按盒子量会差出 4 mm）：
# 行 1 墨迹 192.3–194.3 mm、行 2 墨迹 195.3–197.3 mm -> 行距 1.0 mm；
# 画布 200 mm，底边到墨迹 2.7 mm


def main():
    C.setup_style()
    os.makedirs(C.OUT_DIR, exist_ok=True)

    # ---- 数据 ----
    segs = C.load_segments()
    node_lats, node_lons = C.load_nodes()
    d_haz = HAZ.collect_hazard_data()
    d_evo = EV.collect_evolution_data()
    d_fc = EV.collect_forecast_data()     # None = 预报数据不齐，图退回现状
    d_exp = EX.collect_exposure_data()

    MM = 1 / 25.4
    global CANVAS
    if d_fc is not None:
        # 预报图例另起一行：只加高画布，panel 都用上缘绝对坐标，位置不变
        CANVAS = (CANVAS[0], CANVAS[1] + FC_LEGEND_H)
    fig = plt.figure(figsize=(CANVAS[0] * MM, CANVAS[1] * MM))
    cw, ch = CANVAS

    # ---- 行 1: a 类型图 / b 峰值危险度图（主图 + b1/b2 inset）；
    # 两张地图上沿对齐（同为 y = 5 mm），a 左缘压到 6 mm / 宽 68 mm
    # （右缘 74 mm，与 b 的纬度标签 76.2 mm 留 2 mm 余量）----
    axa, ha = C.new_map_ax(fig, 6, 5, 68, canvas=CANVAS)
    CA.draw_panel_a(axa, segs, node_lats, node_lons, legend=False)
    C.letter(fig, 0.8, 2.2, "a", canvas=CANVAS)
    # 图例放 panel a 地图下方空档（双列横排），随地图放大整体下移
    # 图例按可见内容居中于 panel a 中线（40 mm），再按目视反馈整体左移 1 mm
    CA.panel_a_legend(fig, 6, 57.6, CANVAS, segs, center_x=6 + 68 / 2 - 1.0)

    # panel b：右列。主图 96 mm 宽（右缘 176 mm，留 4 mm 白边）；图例横排放
    # 最下方
    b_x, b_w = 80.0, 96.0
    axb, hb = C.new_map_ax(fig, b_x, 5, b_w, canvas=CANVAS)
    lc = HAZ.draw_panel_b(axb, d_haz, legend=False)
    C.letter(fig, 76.0, 2.2, "b", canvas=CANVAS)
    HAZ.add_colorbars(fig, axb, lc, b_x, 5, b_w, hb, CANVAS)

    # b1/b2/b3 的经纬跨度略有差异（纵横比 0.84/0.84/0.807），若同宽则 b1/b2
    # 更高、下沿不齐。取三者可行的最小高度（由最扁的 b3 决定），其余按各自
    # 纵横比反算宽度 → 上沿齐平且下沿对齐；间隙均分，整行与主图左右缘等宽
    # 间距固定 2 mm：宽度按各自纵横比同比缩放（保持等高），整行仍占满
    # [b_x, b_x + b_w]（左右缘与主图一致），只是把省下的宽度分给三块子图
    spans = [(s["region"][1] - s["region"][0],
              s["region"][3] - s["region"][2]) for s in HAZ.INSETS]
    gap = 2.0
    h_in = (b_w - gap * (len(spans) - 1)) / sum(dx / dy for dx, dy in spans)
    widths = [h_in * dx / dy for dx, dy in spans]
    x_in = b_x
    for spec, w_in in zip(HAZ.INSETS, widths):
        axi, ih = C.new_map_ax(fig, x_in, 88.8, w_in, canvas=CANVAS,
                               region=spec["region"], gridlines=False)
        HAZ.draw_inset(axi, d_haz, spec)
        x_in += w_in + gap
    HAZ.panel_b_legend(fig, b_x, 114.8, CANVAS, ncol=3,
                       center_x=b_x + b_w / 2)   # 整行居中于 panel b   # 下沿对齐 panel c 图例底（118.6 mm）

    # ---- 左列下半: c 暴露汇总（单图双纵轴：bar=km 左轴，菱形=share % 右轴）；
    # 整体下移 2 mm 给上面放大后的 panel a 图例让位（c 下方仍余 ~15 mm）----
    # 宽度取到右轴标签右沿 = panel a 图框右沿（74.1 mm），左沿与 a 同为字母列
    axc = C.ax_mm(fig, 12, 71.5, 54, 36, canvas=CANVAS)
    EX.draw_exposure_merged_ax(axc, d_exp)
    C.letter(fig, 0.8, 67.8, "c", canvas=CANVAS)
    EX.draw_exposure_merged_legend(fig, 12, C.EXPO_LEGEND_Y, CANVAS,
                                   ncol=C.EXPO_LEGEND_NCOL)

    # ---- 行 3: d 越浪时间演化（段节点，3 列）；e 水位同点对比（潮位站
    # 位置，与 d 不同点位，故独立 panel；右缘留 ~8 mm 给 P 副轴标签）----
    # 列宽/列距略收（39/42 → 37.5/40.5，整块右缘 173 → 168.5 mm），
    # 给第 4 列右轴标题和首列左轴标题留出画布内空间（原来两侧都被裁到 0/180）
    EV.draw_panel_c(fig, d_evo, x0=10.6, y_top=125.5, col_w=37.6, col_dx=40.3,
                    row_h_q=31, row_h_e=25, row_gap=5.1, canvas=CANVAS,
                    fc=d_fc)
    # 只有当某站既无 1-min 序列也无逐日极值（只能画 HKO 公布极值点划线时）
    # 才在共用图例里保留 "gauge max only"
    show_max_only = any(
        d_evo[s["seg_id"]].get("gauge_obs") is None
        and d_evo[s["seg_id"]].get("gauge_extrema") is None
        for s in C.PANEL_C_SEGS)
    show_extremes = any(
        d_evo[s["seg_id"]].get("gauge_extrema") is not None
        for s in C.PANEL_C_SEGS)
    leg_c = EV.panel_c_legend(fig, 190.2, CANVAS,
                              show_max_only=show_max_only,
                              show_extremes=show_extremes)   # 距 panel e 刻度下沿 1.5 mm
    leg_fc = None
    if d_fc is not None:
        # 锚点 = 图例行盒顶 = 墨迹顶 - 2.1 mm（实测内边距）
        leg_fc = EV.panel_c_forecast_legend(fig, 193.2, CANVAS)
    C.letter(fig, 0.8, 119.7, "d", canvas=CANVAS)
    C.letter(fig, 0.8, 157.6, "e", canvas=CANVAS)

    # 5 pt 字形底线：mathtext 上下标按 0.7x 缩，先 draw 一轮让刻度标签的
    # Text 生成，再把过小的 mathtext 父字号提到 7.2 pt（下标 5.04 pt）
    fig.canvas.draw()
    TY.enforce_glyph_floor(fig)
    # 图例字号统一（glyph floor 会把含 mathtext 的标签单独提到 7.2 pt）
    EV.unify_legend_fontsize((leg_c, leg_fc), size=6.0)

    fig.savefig(OUT_BASE + ".png", dpi=300)
    # PDF 里 imshow 按 figure dpi 重采样到设备像素密度；默认 100 dpi 会把
    # 正射影像压到 ~300 px。给 1200 dpi：panel a 主图嵌入 ~3600 px 宽，
    # inset ~1200 px 宽，PDF 放大后仍然清晰（矢量元素不受 dpi 影响）。
    # 反复调版时可设 FIG6_SKIP_PDF=1 只出 PNG（40 MB PDF 在 NFS 忙时很慢）
    if os.environ.get("FIG6_SKIP_PDF") != "1":
        fig.savefig(OUT_BASE + ".pdf", dpi=1200)
        print("saved:", OUT_BASE + ".png/.pdf")
    else:
        print("saved:", OUT_BASE + ".png (PDF skipped)")


if __name__ == "__main__":
    main()
