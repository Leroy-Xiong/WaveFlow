#!/usr/bin/env python
"""Fig. 5 案例 panel draft：台风圆规 (Kompasu) 2021-10-13 00Z，ENS 驱动预报
随 lead 的演化。

上 panel：domain-max hs 时间序列（2021-10-08 → 10-16）
  - GT（hindcast，逐小时，黑线）
  - WaveFlow + ERA5（完美强迫，bc2，12h 点）—— 上限参照
  - WaveFlow + IFS ENS @ lead 12/24/48/72h（bc2）：每 valid time 的
    per-member domain-max 集合均值（线）+ min–max 带
  （2026-10-03 应用户要求删除 CMEMS 沿轨灰点；卫星验证统计留 SI）
  竖虚线 = 下方地图的 valid time (2021-10-13 00Z)

下两行地图（valid = 2021-10-13 00Z）：
  行 1 = GT / 12h / 24h / 48h / 72h 集合平均 hs（turbo 0–6 m，同一色标）
  行 2 = 4 个 lead 的 P(hs>4 m) + GT 4 m 等值线

数据：<run>/forecast_ensmem{L}h_bc2/test/samples（2026-08-31 逐成员口径：
ENS 成员 k 驱动 WaveFlow 样本 k；旧 ens-mean 采样 forecast_lead{L}h_bc2
已由 panel a/b 撤下）、
uq_baselines/cfm_bc2/test/samples（ERA5 驱动，本地）、data/hindcast GT。

用法: python scripts/figures/figure5/plot_figure5_case.py
输出: output/plot/figure5/figure5_case_draft.{png,pdf}
"""
import os
import sys
from datetime import datetime, timedelta

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))
os.chdir(_REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "figures", "figure4"))
os.environ.setdefault("CARTOPY_DATA_DIR", os.path.join(_REPO_ROOT,
                                                       "data/cartopy_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import cartopy.crs as ccrs

import plot_figure4_case as CASE4   # make_ds / add_gt_contour / REGION
from modules.plot.plot_wave_hk import plot_wave_data

# Nature 风格（覆盖 CASE4 模块的 rcParams）
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "TeX Gyre Heros",
                        "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Release override: paths can be redirected to re-inferred samples / a
# user-supplied hindcast without editing the script.
RUN = os.environ.get(
    "WF_HK_RUN",
    "output/results/hk_cleaned/all_var/f8_blr5e-5_amp_stable_tod/"
    "size_sweep/base_v1_20260821_025917")
ERA5_SAMPLES = os.path.join(RUN, "uq_baselines/cfm_bc2/test/samples")
GT_DIR = os.environ.get("WF_GT_HK", "data/hindcast/processed_npz")
COORDS = "data/hindcast/hk_coordinates.npy"
LEADS = [12, 24, 48, 72]
LEAD_COLORS = {12: "#0072B2", 24: "#009E73", 48: "#E69F00", 72: "#CC79A7"}
N_MEMBERS = 10
T_MAP = "20211013T00"                # 地图 valid time（GT 峰值 20211012T22 后最近 12h 格点）
# panel d 地图行的强迫源（2026-10-19 起改 GenCast）：Kompasu 的 4 个 init
# （1010T00/1011T00/1012T00/1012T12）不在 --select-days 抽样窗口内，已单独
# 补建 + 采样到 _kompasu 目录 + bc3；未就绪时自动退回 ensmem bc2。
# 2026-10-27 起优先零插值强迫（native），回退旧的插值版
# 2026-10-28 应用户要求改回 IFS ENS 逐成员（与 panel c 同一条强迫链）：
# GenCast 在 Kompasu 24–72 h 的峰值低 0.4–0.6 m、P(Hs>4 m) 少 30–50%，
# 与 panel c 的 ENS 轨迹不自洽。
CASE_SAMPLES_TMPLS = ("forecast_ensmem{L}h_bc2/test/samples",
                      "forecast_gc{L}h_kompasu_native_bc3/test/samples")
CASE_SAMPLES_TMPL = CASE_SAMPLES_TMPLS[0]
CASE_SAMPLES_FALLBACK = "forecast_ensmem{L}h_bc2/test/samples"


def case_sample_dir(lead):
    """panel d 样本目录：IFS ENS 逐成员 bc2（与 panel c 同口径）优先。"""
    for tmpl in CASE_SAMPLES_TMPLS:
        d = os.path.join(RUN, tmpl.format(L=lead))
        if os.path.exists(os.path.join(d, f"sample_{T_MAP}_ens000.npz")):
            return d
    return os.path.join(RUN, CASE_SAMPLES_FALLBACK.format(L=lead))
T0 = datetime(2021, 10, 8, 0)
T1 = datetime(2021, 10, 16, 0)
HS_VMAX = 6.0
EXCEED = 4.0

OUT_DIR = "output/plot/figure5"
OUT_BASE = os.path.join(OUT_DIR, "figure5_case_draft")

CANVAS_W, CANVAS_H = 180.0, 105.0    # mm


def ax_mm(fig, x, y_top, w, h, projection=None, canvas=None):
    """毫米绝对定位：x 从画布左缘、y_top 从画布顶缘量。
    canvas=(W,H) mm，默认用本模块独立图的画布。"""
    cw, ch = canvas or (CANVAS_W, CANVAS_H)
    kw = dict(projection=projection) if projection is not None else {}
    return fig.add_axes([x / cw, 1 - (y_top + h) / ch,
                         w / cw, h / ch], **kw)


def ts12h(t0, t1):
    out, t = [], t0
    while t <= t1:
        out.append(t.strftime("%Y%m%dT%H"))
        t += timedelta(hours=12)
    return out


def gt_hs(ts):
    return np.load(os.path.join(GT_DIR, f"wave_{ts}.npz"))["hs"]


def ens_hs(sample_dir, ts, m=N_MEMBERS):
    """(M, 7788) hs。"""
    return np.stack([np.load(os.path.join(
        sample_dir, f"sample_{ts}_ens{k:03d}.npz"))["sample"][:, 0]
        for k in range(m)])


def collect_timeseries():
    """返回绘图数据 dict：GT 逐小时 domain-max、ERA5/各 lead 的集合统计、卫星点。"""
    ts12 = ts12h(T0, T1)

    # GT 逐小时 domain-max
    hours, gt_max = [], []
    t = T0
    while t <= T1:
        ts = t.strftime("%Y%m%dT%H")
        f = os.path.join(GT_DIR, f"wave_{ts}.npz")
        if os.path.exists(f):
            hours.append(t)
            gt_max.append(float(np.nanmax(np.load(f)["hs"])))
        t += timedelta(hours=1)

    # ERA5-forced bc2（完美强迫上限）：集合平均场的 domain-max
    era5_mean, era5_ts = [], []
    for ts in ts12:
        try:
            e = ens_hs(ERA5_SAMPLES, ts)
        except FileNotFoundError:
            continue
        era5_ts.append(datetime.strptime(ts, "%Y%m%dT%H"))
        era5_mean.append(float(np.nanmax(np.nanmean(e, axis=0))))

    # 各 lead：per-member domain-max 的均值/min/max（峰值预报的集合分布）
    lead_ts, lead_mean, lead_lo, lead_hi = {}, {}, {}, {}
    for L in LEADS:
        sdir = os.path.join(RUN, f"forecast_ensmem{L}h_bc2/test/samples")
        tt, mm, lo, hi = [], [], [], []
        for ts in ts12:
            try:
                e = ens_hs(sdir, ts)
            except FileNotFoundError:
                continue
            peak = np.nanmax(e, axis=1)          # (M,)
            tt.append(datetime.strptime(ts, "%Y%m%dT%H"))
            mm.append(float(peak.mean()))
            lo.append(float(peak.min()))
            hi.append(float(peak.max()))
        lead_ts[L], lead_mean[L] = tt, np.array(mm)
        lead_lo[L], lead_hi[L] = np.array(lo), np.array(hi)

    return dict(hours=hours, gt_max=np.array(gt_max),
                era5_ts=era5_ts, era5_mean=np.array(era5_mean),
                lead_ts=lead_ts, lead_mean=lead_mean,
                lead_lo=lead_lo, lead_hi=lead_hi)


def draw_map(fig, x, y, w, h, vals, lons, lats, title, vmin, vmax, cmap,
             left_labels=False, bottom_labels=True, canvas=None):
    ax = ax_mm(fig, x, y, w, h, projection=ccrs.PlateCarree(), canvas=canvas)
    plot_wave_data(CASE4.make_ds(vals, lons, lats), variable="hs",
                  region=CASE4.REGION, vmin=vmin, vmax=vmax, cmap=cmap,
                  ax=ax, return_ax=True, scatter_size=1, custom_title="",
                  show_cities=False, show_colorbar=False, show_nodes=False,
                  gridline_spacing=(0.5, 0.5), show_top_labels=False,
                  show_left_labels=left_labels,
                  show_bottom_labels=bottom_labels)
    ax.set_title(title, fontsize=7, pad=1.5)   # 与其它 panel 标题同号（2026-11-04）
    if left_labels:
        # 纬度标签竖排，且文字中心正对刻度（纬度）线：关掉 tick_params 的
        # 整体旋转，逐标签用 anchor 模式 + ha=center/va=bottom，
        # 否则默认的 center_baseline 会把标签整体偏移约 3 mm（2026-11-04）
        ax.tick_params(axis="y", pad=0.6)   # 标签贴近地图（2026-11-04）
        for lab in ax.get_yticklabels():
            lab.set_rotation(90)
            lab.set_rotation_mode("anchor")
            lab.set_ha("center")
            lab.set_va("bottom")
    return ax, ax._wave_scatter


def draw_timeseries_ax(ax, d, legend_out=False):
    """Kompasu domain-max hs 时序（主图 panel c）。"""
    ax.plot(d["hours"], d["gt_max"], color="black", lw=1.2,
            label="hindcast truth (hourly)")
    ax.plot(d["era5_ts"], d["era5_mean"], color="#555555", lw=1.0, ls="--",
            marker=".", ms=3, label="WaveFlow + ERA5 (perfect forcing)")
    for L in LEADS:
        ax.fill_between(d["lead_ts"][L], d["lead_lo"][L], d["lead_hi"][L],
                        color=LEAD_COLORS[L], alpha=0.18, lw=0)
        ax.plot(d["lead_ts"][L], d["lead_mean"][L], color=LEAD_COLORS[L],
                lw=1.1, marker="o", ms=2.5,
                label=f"WaveFlow + IFS ENS, {L} h lead")
    t_map = datetime.strptime(T_MAP, "%Y%m%dT%H")
    ax.axvline(t_map, color="black", lw=0.6, ls=":", alpha=0.6)
    ax.annotate("map time", (t_map, 0.15), fontsize=5.5, ha="center",
                va="bottom", color="#333333")
    ax.set_ylabel("domain-max $H_s$ (m)")
    ax.set_xlim(T0, T1)
    ax.set_ylim(0, 7)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    if legend_out:
        ax.legend(frameon=False, ncol=4, loc="upper left",
                  bbox_to_anchor=(0.0, -0.42), handlelength=1.3,
                  columnspacing=0.9, borderaxespad=0.0, labelspacing=0.3,
                  fontsize=5.4)
    else:
        ax.legend(frameon=False, ncol=4, loc="upper left", handlelength=1.4,
                  columnspacing=0.8, borderaxespad=0.1, labelspacing=0.35)
    ax.set_xlabel("valid time (2021, UTC)", labelpad=1.5)


def draw_mean_map_row(fig, lons, lats, gt_map, x0, y_top, map_w, dx,
                      canvas=None, left_labels=False, bottom_labels=False):
    """GT + 4 lead 集合平均 hs 地图行（主图 panel g 上行）。"""
    map_h = map_w / 1.4286               # PlateCarree equal aspect: 1.0°/0.7°
    cw, ch = canvas or (CANVAS_W, CANVAS_H)
    # 2026-11-04：首图写 Reference，其余写 "L h lead"
    titles = ["Reference"] + [f"{L} h lead" for L in LEADS]
    sc = None
    for i in range(5):
        if i == 0:
            vals = gt_map
        else:
            L = LEADS[i - 1]
            e = ens_hs(case_sample_dir(L), T_MAP)
            vals = np.nanmean(e, axis=0)
        _, sc = draw_map(fig, x0 + i * dx, y_top, map_w, map_h, vals,
                         lons, lats, titles[i], 0, HS_VMAX, "turbo",
                         left_labels=(left_labels and i == 0),
                         bottom_labels=bottom_labels, canvas=canvas)
    cba = fig.add_axes([(x0 + 4 * dx + map_w + 1.5) / cw,
                        1 - (y_top + map_h) / ch,
                        1.2 / cw, map_h / ch])
    cb = fig.colorbar(sc, cax=cba)
    cb.set_label("$H_s$ (m)", fontsize=6)
    cb.ax.tick_params(labelsize=6, length=2)


def draw_prob_map_row(fig, lons, lats, gt_map, x0, y_top, map_w, dx,
                      canvas=None):
    """4 个 lead 的 P(hs>4m) 概率图行 + GT 等值线（主图 panel g 下行）。

    2026-11-04：每块不再写标题；整行只在左侧空槽（没有概率图的首列）里放
    一个 "P(Hs>4 m)" 行标签。
    """
    map_h = map_w / 1.4286
    cw, ch = canvas or (CANVAS_W, CANVAS_H)
    scp, ax0 = None, None
    for i, L in enumerate(LEADS):
        e = ens_hs(case_sample_dir(L), T_MAP)
        p = (e > EXCEED).mean(0)
        ax, scp = draw_map(fig, x0 + (i + 1) * dx, y_top, map_w, map_h,
                           p.astype(np.float32), lons, lats, "",
                           0, 1, "viridis", left_labels=(i == 0),
                           canvas=canvas)
        ax0 = ax0 or ax
        CASE4.add_gt_contour(ax, gt_map, lons, lats, level=EXCEED)
    # 行标签竖排：右缘距本行纬度标签左缘 1 mm（2026-11-04，量渲染 bbox 定位）
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _k = 25.4 / fig.dpi
    _labs = [t for t in ax0.get_yticklabels() if t.get_text()]
    _x_lab = min(t.get_window_extent(_r).x0 for t in _labs) * _k
    fig.text((_x_lab - 1.0) / cw, 1 - (y_top + 0.5 * map_h) / ch,
             f"P($H_s$>{EXCEED:.0f} m)", ha="center", va="bottom",
             rotation=90, rotation_mode="anchor", fontsize=7)
    cbb = fig.add_axes([(x0 + 4 * dx + map_w + 1.5) / cw,
                        1 - (y_top + map_h) / ch,
                        1.2 / cw, map_h / ch])
    cbp = fig.colorbar(scp, cax=cbb)
    cbp.set_label("probability", fontsize=6)
    cbp.ax.tick_params(labelsize=6, length=2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    coords = np.load(COORDS)
    lats, lons = coords[:, 0], coords[:, 1]
    d = collect_timeseries()

    MM = 1 / 25.4
    fig = plt.figure(figsize=(CANVAS_W * MM, CANVAS_H * MM))

    # ---- panel a: 时间序列 ----
    axa = ax_mm(fig, 12, 5, 156, 32)
    draw_timeseries_ax(axa, d)
    fig.text(5 / CANVAS_W, 1 - 5 / CANVAS_H, "a", fontsize=9,
             fontweight="bold", ha="left", va="top")

    # ---- panel b: 集合平均 hs 地图行 ----
    gt_map = gt_hs(T_MAP)
    Y1, Y2 = 48.5, 74.0
    draw_mean_map_row(fig, lons, lats, gt_map, x0=12.0, y_top=Y1,
                      map_w=26.0, dx=30.0)
    fig.text(5 / CANVAS_W, 1 - Y1 / CANVAS_H, "b", fontsize=9,
             fontweight="bold", ha="left", va="top")

    # ---- panel c: P(hs>4m) 概率图行 ----
    draw_prob_map_row(fig, lons, lats, gt_map, x0=12.0, y_top=Y2,
                      map_w=26.0, dx=30.0)
    fig.text(5 / CANVAS_W, 1 - Y2 / CANVAS_H, "c", fontsize=9,
             fontweight="bold", ha="left", va="top")

    fig.savefig(OUT_BASE + ".png", dpi=300)
    fig.savefig(OUT_BASE + ".pdf")
    print("saved:", OUT_BASE + ".png/.pdf")


if __name__ == "__main__":
    main()
