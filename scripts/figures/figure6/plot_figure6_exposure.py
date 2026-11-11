#!/usr/bin/env python
"""Fig. 6 panel d draft：暴露汇总 —— 按岸线类型、event-max P(q>1) ≥ 0.5
的工程岸线长度（km），M=10 vs M=100（统一 48 h 峰值窗口（49 个逐时刻）
2023-08-31T12Z..09-02T12Z，只比较集合大小；M=10 的 12 天全程序列裁到
同一窗口再取 event-max）。

每根 bar 按堤顶数据置信度拆分：实心 = crest 代理有实测/设计依据
（dtm/spot/design，crest_flag≠1），斜纹 = 低置信（原 crest_flag==1 新生
填海段经邻段中位数 / +5.5 mPD 形成高程默认值插补，crest_src 为
imputed_nb / imputed_def）。

独立运行同时输出附录版本（各类型 event-max P 分布 boxplot + jitter，
m100 口径）：figure6_exposure_boxplot_draft.{png,pdf}

用法: python scripts/figures/figure6/plot_figure6_exposure.py
输出: output/plot/figure6/figure6_exposure_draft.{png,pdf}
      output/plot/figure6/figure6_exposure_boxplot_draft.{png,pdf}
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_figure6_common as C

OUT_BASE = os.path.join(C.OUT_DIR, "figure6_exposure_draft" + C.OUT_SUFFIX)
OUT_BASE_BOX = os.path.join(C.OUT_DIR,
                            "figure6_exposure_boxplot_draft" + C.OUT_SUFFIX)

P_THR = 0.5
C_M10 = "#7F7F7F"
C_M100 = "#B2182B"
# share% 菱形/右轴：全稿 RdBu 蓝（= figure4/5 对比量蓝，取代旧 #1A1A6E 暗紫蓝）
C_SHARE = "#2166AC"
CTYPES = C.EXPO_CTYPES   # 案例化：HK=工程段两类，helene=堤顶高程三类


def collect_exposure_data():
    """event-max P（统一 48 h 峰值窗口，49 个逐时刻：M=100 只采样了峰值段，
    M=10 的全程时间序列裁到同一窗口再取 event-max，保证两者只差集合大小）
    + 分段长度/类型/crest 置信度。"""
    segs = C.load_segments().set_index("seg_id")
    summary = C.load_summary()
    z100 = C.load_overtopping("m100")
    z10 = C.load_overtopping("m10")
    ids = z100["seg_ids"]
    assert (ids == z10["seg_ids"]).all()
    ts100 = [str(t) for t in z100["timestamps"]]
    ts10 = np.array([str(t) for t in z10["timestamps"]])
    lo, hi = min(ts100), max(ts100)
    win = (ts10 >= lo) & (ts10 <= hi)
    print(f"common window {lo}..{hi}: m10 {int(win.sum())}/{len(ts10)} ts, "
          f"m100 {len(ts100)} ts")
    df = pd.DataFrame({"seg_id": ids})
    df["coast_type"] = summary.loc[df.seg_id, "coast_type"].values
    df["len_km"] = segs.loc[df.seg_id, "length_m"].values / 1000.0
    df["flag1"] = (summary.loc[df.seg_id, "crest_flag"].values == 1)
    df["imputed"] = summary.loc[df.seg_id, "crest_src"].astype(str) \
        .str.startswith("imputed").values
    df["lowconf"] = df["flag1"] | df["imputed"]
    df["pmax_m10"] = z10["p_gt1_lpsm"][:, win].max(axis=1)
    df["pmax_m100"] = z100["p_gt1_lpsm"].max(axis=1)
    # 打印 backing 数字
    tot = {ct: df[df.coast_type == ct].len_km.sum() for ct in CTYPES}
    for prod in ["pmax_m10", "pmax_m100"]:
        for ct in CTYPES:
            m = (df.coast_type == ct) & (df[prod] >= P_THR)
            conf = df[m & ~df.lowconf].len_km.sum()
            low = df[m & df.lowconf].len_km.sum()
            print(f"{prod} {ct}: {df[m].len_km.sum():.1f} km "
                  f"({m.sum()} segs, {df[m].len_km.sum()/tot[ct]*100:.1f}% "
                  f"of {tot[ct]:.1f} km) = confident {conf:.1f} + "
                  f"lowconf {low:.1f}")
    return df


def _bar_split(ax, x, w, conf, low, color, pct=None):
    """单根 bar：实心 confident 段 + 斜纹低置信（flag1/插补）段。
    pct 给定标注为 'km (pct%)'（pct = 占该类工程岸线总长比例）。"""
    ax.bar(x, conf, w, color=color, lw=0)
    ax.bar(x, low, w, bottom=conf, facecolor=color, alpha=0.45,
           hatch="///", edgecolor="white", lw=0)
    label = f"{conf + low:.1f}" if pct is None \
        else f"{conf + low:.1f} ({pct:.0f}%)"
    ax.annotate(label, (x, conf + low), ha="center",
                va="bottom", fontsize=6, color="#333333",
                textcoords="offset points", xytext=(0, 1))


def _exposure_handles():
    return [
        Patch(fc=C_M10, label="$M$ = 10"),
        Patch(fc=C_M100, label="$M$ = 100"),
        Patch(fc="#BBBBBB", label="crest-confident"),
        Patch(fc="#BBBBBB", alpha=0.45, hatch="///", edgecolor="white",
              label="imputed crest (new reclamation)"),
    ]


def draw_exposure_ax_legend(fig, x, y_top, canvas, ncol=1):
    """panel c 图例（图级自由定位，供主图放在轴旁/下方空白区）。"""
    cw, ch = canvas
    fig.legend(handles=_exposure_handles(), loc="upper left", frameon=False,
               ncol=ncol, fontsize=6, handlelength=1.5, labelspacing=0.5,
               columnspacing=0.8,
               bbox_to_anchor=(x / cw, 1 - y_top / ch))


def draw_exposure_ax(ax, df, legend=True, mode="km"):
    """分组堆叠 bar：x = coast_type，组 = M=10 / M=100。
    mode='km'：纵轴绝对长度（km），标注 'km (pct%)'；
    mode='pct'：纵轴占该类岸线总长比例（%），标注 'pct%'。"""
    xs = np.arange(len(CTYPES))
    w = 0.32
    tot_km = {ct: df[df.coast_type == ct].len_km.sum() for ct in CTYPES}
    for k, prod in enumerate(["pmax_m10", "pmax_m100"]):
        for i, ct in enumerate(CTYPES):
            m = (df.coast_type == ct) & (df[prod] >= P_THR)
            conf = df[m & ~df.lowconf].len_km.sum()
            low = df[m & df.lowconf].len_km.sum()
            pct = (conf + low) / tot_km[ct] * 100 if tot_km[ct] else 0.0
            if mode == "pct":
                scale = 100.0 / tot_km[ct] if tot_km[ct] else 0.0
                _bar_split(ax, xs[i] + (k - 0.5) * (w + 0.03), w,
                           conf * scale, low * scale,
                           C_M10 if prod == "pmax_m10" else C_M100)
                ax.annotate(f"{pct:.0f}%",
                            (xs[i] + (k - 0.5) * (w + 0.03),
                             (conf + low) * scale),
                            ha="center", va="bottom", fontsize=6,
                            color="#333333", textcoords="offset points",
                            xytext=(0, 1))
            else:
                _bar_split(ax, xs[i] + (k - 0.5) * (w + 0.03), w, conf, low,
                           C_M10 if prod == "pmax_m10" else C_M100, pct=pct)
    ax.set_xticks(xs)
    ax.set_xticklabels([C.COAST_LABELS[ct] for ct in CTYPES])
    if mode == "pct":
        ax.set_ylabel("share (%)")
        ax.set_ylim(*C.EXPO_YLIM_PCT)
        ax.set_title("share of each type total", fontsize=6.5, pad=2)
    else:
        # 判据 P(q>1) ≥ P_THR 写进图注，轴标签保持单行
        ax.set_ylabel("exposed shoreline (km)")
        ax.set_ylim(*C.EXPO_YLIM_KM)
        ax.set_title(C.EXPO_WINDOW_TITLE,
                     fontsize=6.5, pad=2)
    ax.set_xlim(-0.6, len(CTYPES) - 0.4)
    if legend:
        ax.legend(handles=_exposure_handles(), frameon=False, fontsize=6,
                  loc="upper right", handlelength=1.5, labelspacing=0.4)


def draw_exposure_merged_ax(ax, df):
    """合并版：单轴双纵标。bar = event-max P≥0.5 的工程岸线 km（左轴，
    按 crest 置信度拆分）；菱形 marker = 占该类岸线总长比例 %（右轴）。"""
    ax2 = ax.twinx()
    xs = np.arange(len(CTYPES))
    w = 0.32
    tot_km = {ct: df[df.coast_type == ct].len_km.sum() for ct in CTYPES}
    for k, prod in enumerate(["pmax_m10", "pmax_m100"]):
        for i, ct in enumerate(CTYPES):
            m = (df.coast_type == ct) & (df[prod] >= P_THR)
            conf = df[m & ~df.lowconf].len_km.sum()
            low = df[m & df.lowconf].len_km.sum()
            pct = (conf + low) / tot_km[ct] * 100 if tot_km[ct] else 0.0
            x = xs[i] + (k - 0.5) * (w + 0.03)
            _bar_split(ax, x, w, conf, low,
                       C_M10 if prod == "pmax_m10" else C_M100)
            # 菱形叠加在 bar 正中（白底描边保证在 bar 上可读）；
            # 若 pct 换算后超出柱顶（会与 km 标注重叠），改放柱内 55% 高度处
            km_lo, km_hi = C.EXPO_YLIM_KM
            pct_lo, pct_hi = C.EXPO_YLIM_PCT
            bar_top_pct = ((conf + low - km_lo) / (km_hi - km_lo)
                           * (pct_hi - pct_lo) + pct_lo)
            y_pct = pct if pct <= bar_top_pct else 0.55 * bar_top_pct
            ax2.plot(x, y_pct, marker="D", ms=3, mfc="white",
                     mec=C_SHARE, mew=0.9, zorder=5, clip_on=False)
            # % 标注常落在柱身内（深灰/深红底）：白描边 halo 保证可读
            ax2.annotate(f"{pct:.0f}%", (x, y_pct), fontsize=5.5,
                         color=C_SHARE, ha="left", va="center",
                         textcoords="offset points", xytext=(3.5, 0),
                         path_effects=[pe.withStroke(linewidth=1.2,
                                                     foreground="white")],
                         zorder=5, annotation_clip=False)
    ax.set_xticks(xs)
    ax.set_xticklabels([C.COAST_LABELS[ct] for ct in CTYPES])
    # 类目名较长，比常规刻度字号大一档（7 pt，与轴标题同级）更易读
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(-0.6, len(CTYPES) - 0.4)
    # 判据 P(q>1) ≥ P_THR 写进图注，轴标签保持单行
    ax.set_ylabel("exposed shoreline (km)")
    ax.set_ylim(*C.EXPO_YLIM_KM)
    ax2.set_ylabel("share of each type total (%)", color=C_SHARE)
    ax2.set_ylim(*C.EXPO_YLIM_PCT)
    ax2.tick_params(axis="y", colors=C_SHARE)
    # 全局 rcParams 关了 top/right 脊线，twinx 的右轴要显式打开才画得出来
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C_SHARE)
    ax2.spines["right"].set_linewidth(0.6)
    ax.set_title(C.EXPO_WINDOW_TITLE,
                 fontsize=6.5, pad=2)
    # twinx 的 ax2 默认在上层且 patch 透明：marker 盖在 bar 上仍可读
    return ax2


def _merged_handles():
    handles = [
        Patch(fc=C_M10, label="$M$ = 10"),
        Patch(fc=C_M100, label="$M$ = 100"),
        *([(Patch(fc="#BBBBBB", label="crest-confident"),
            Patch(fc="#BBBBBB", alpha=0.45, hatch="///", edgecolor="white",
                  label="imputed crest"))] if C.EXPO_CREST_SPLIT else []),
        Line2D([0], [0], marker="D", ls="", ms=3, mfc="white",
               mec=C_SHARE, mew=0.9, label="share (%), right"),
    ]
    return [h for pair in handles for h in
            (pair if isinstance(pair, tuple) else (pair,))]


def draw_exposure_merged_legend(fig, x, y_top, canvas, ncol=2):
    """合并版 panel c 图例（图级定位）。"""
    cw, ch = canvas
    fig.legend(handles=_merged_handles(), loc="upper left", frameon=False,
               ncol=ncol, fontsize=6, handlelength=1.5, labelspacing=0.5,
               columnspacing=0.8,
               bbox_to_anchor=(x / cw, 1 - y_top / ch))


def draw_boxplot_ax(ax, df, legend=True):
    """附录版：各类型 event-max P(q>1)（M=100）boxplot + jitter 点。"""
    rng = np.random.default_rng(0)
    data = []
    for i, ct in enumerate(CTYPES):
        sub = df[df.coast_type == ct]
        data.append(sub.pmax_m100.values)
        xj = i + rng.uniform(-0.25, 0.25, len(sub))
        cols = np.where(sub.lowconf.values, "#AAAAAA", "#333333")
        ax.scatter(xj, sub.pmax_m100.values, s=1.5, c=cols, alpha=0.5,
                   lw=0, zorder=2)
    bp = ax.boxplot(data, positions=range(len(CTYPES)), widths=0.4,
                    showfliers=False, patch_artist=True, zorder=3,
                    medianprops=dict(color="black", lw=0.8),
                    whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7),
                    boxprops=dict(lw=0.7))
    for patch, ct in zip(bp["boxes"], CTYPES):
        patch.set_facecolor(C.COAST_COLORS[ct])
        patch.set_alpha(0.5)
    ax.set_xticks(range(len(CTYPES)))
    ax.set_xticklabels([C.COAST_LABELS[ct] for ct in CTYPES])
    ax.set_ylabel("event-max $P(q > 1$ L/s/m), $M$ = 100")
    ax.set_ylim(-0.05, 1.05)
    if legend:
        handles = [
            Line2D([0], [0], marker="o", ls="", ms=2.5, mfc="#333333",
                   mec="none", label="crest-confident"),
            Line2D([0], [0], marker="o", ls="", ms=2.5, mfc="#AAAAAA",
                   mec="none", label="no crest data (new reclamation)"),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=6,
                  loc="center left", bbox_to_anchor=(0.02, 0.5))


def main():
    C.setup_style()
    os.makedirs(C.OUT_DIR, exist_ok=True)
    df = collect_exposure_data()

    # 主版：分组 bar
    canvas = (95.0, 70.0)
    MM = 1 / 25.4
    fig = plt.figure(figsize=(canvas[0] * MM, canvas[1] * MM))
    ax = C.ax_mm(fig, 18, 8, 70, 48, canvas=canvas)
    draw_exposure_ax(ax, df)
    C.letter(fig, 3, 3.5, "d", canvas=canvas)
    fig.savefig(OUT_BASE + ".png", dpi=300)
    fig.savefig(OUT_BASE + ".pdf")
    print("saved:", OUT_BASE + ".png/.pdf")

    # 附录版：boxplot
    fig2 = plt.figure(figsize=(canvas[0] * MM, canvas[1] * MM))
    ax2 = C.ax_mm(fig2, 18, 8, 70, 48, canvas=canvas)
    draw_boxplot_ax(ax2, df)
    fig2.savefig(OUT_BASE_BOX + ".png", dpi=300)
    fig2.savefig(OUT_BASE_BOX + ".pdf")
    print("saved:", OUT_BASE_BOX + ".png/.pdf")


if __name__ == "__main__":
    main()
