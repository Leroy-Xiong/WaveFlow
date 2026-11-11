#!/usr/bin/env python
"""Fig. 5 panel e draft：全球 WaveFlow（new_data_hs patch2048 huge, epoch99）
+ IFS ENS 全球强迫的 Fiona 飓风预报（valid 2022-09-24 00Z，临近登陆
Nova Scotia）。行 1 = GT / 12h / 24h / 48h / 72h 集合平均 hs（Robinson 节点
散点，turbo 同一色标）；行 2 = 集合平均 − GT 误差（RdBu_r ±3 m）。

数据：output/results/global/new_data_hs/patch2048_concat_wucos_ema999gc_xpred_huge/
forecast_fiona_lead{L}h/samples（本地 GPU 采样，2026-08-30）；GT =
data/global_hindcast/npz_files（hs/mwd/ssh 节点产品）。

用法: python scripts/figures/figure5/plot_figure5_global.py
输出: output/plot/figure5/figure5_global_draft.{png,pdf}
"""
import json
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))
os.chdir(_REPO_ROOT)
os.environ.setdefault("CARTOPY_DATA_DIR", os.path.join(_REPO_ROOT,
                                                       "data/cartopy_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import (
    LATITUDE_FORMATTER, LONGITUDE_FORMATTER)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "TeX Gyre Heros",
                        "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Release override: paths can be redirected to re-inferred samples / a
# user-supplied hindcast without editing the script.
RUN = os.environ.get(
    "WF_GLOBAL_HS_RUN",
    "output/results/global/new_data_hs/"
    "patch2048_4frames_concat_wucos_ema999gc_xpred_huge")
GT_DIR = os.environ.get("WF_GT_GLOBAL", "data/global_hindcast/npz_files")
COORDS = "data/global_hindcast/global_coordinates.npy"
LEADS = [12, 24, 48, 72]
N_MEMBERS = 10
T_MAP = "20220924T00"
HS_VMAX = 12.0
ERR_VMAX = 2.0

OUT_DIR = "output/plot/figure5"
OUT_BASE = os.path.join(OUT_DIR, "figure5_global_draft")

CANVAS_W, CANVAS_H = 180.0, 62.0     # mm


def ax_mm(fig, x, y_top, w, h, projection=None, canvas=None):
    cw, ch = canvas or (CANVAS_W, CANVAS_H)
    kw = dict(projection=projection) if projection is not None else {}
    return fig.add_axes([x / cw, 1 - (y_top + h) / ch,
                         w / cw, h / ch], **kw)


def load_rmse_json():
    """Fiona 事件窗口逐 lead 集合平均 RMSE（run_global_rmse.py 产出）。
    2026-10-28 改回 IFS ENS 逐成员（与 panel c/d 同一条强迫链；gc 版只覆盖
    6 个时刻且 12–48 h 明显更差，与正文 0.239–0.304 m 那句一致的是 memens 版）。"""
    for name in ("forecast_fiona_rmse.json", "forecast_fiona_rmse_gc.json"):
        f = os.path.join(RUN, name)
        if os.path.exists(f):
            with open(f) as fh:
                return json.load(fh)
    return None


def load_crps_json():
    """全测试集（2021–2022，1460 ts）逐 lead fair CRPS / persistence /
    climatology（run_global_crps_full.py 产出，口径同 HK panel a）。
    0h 点用 fc0（hres_t0 确定性分析场）强迫的结果覆盖（
    forecast_fulltest_crps_fc0.json，run_global_crps_full.py
    --sample-tmpl forecast_fulltest_fc0_lead{L}h 产出）。"""
    f = os.path.join(RUN, "forecast_fulltest_crps.json")
    if not os.path.exists(f):       # 4frames：ens-mean 主线未跑（已撤下，留
        return None                 # supplement），由 member 曲线提供 clim/pers
    with open(f) as fh:
        d = json.load(fh)
    fc0 = os.path.join(RUN, "forecast_fulltest_crps_fc0.json")
    if os.path.exists(fc0):
        with open(fc0) as fh:
            d["leads"]["0"] = json.load(fh)["leads"]["0"]
    return d


def load_era5_crps_json():
    """ERA5 完美强迫参照（run_global_crps_era5.py 产出）。与 HK panel a 一致，
    优先 bc2 版（val 拟合 δ → test 应用），raw 版仅作 fallback；未跑时 None。"""
    for name in ("forecast_era5full_crps_bc2.json", "forecast_era5full_crps.json"):
        f = os.path.join(RUN, name)
        if os.path.exists(f):
            with open(f) as fh:
                return json.load(fh)
    return None


def load_member_crps_json(kind):
    """逐成员强迫曲线（2026-08-31 口径升级，member k → WaveFlow 样本 k）：
    kind='memens' → forecast_fulltest_crps_memens.json（IFS ENS number1-10）；
    kind='memens_bc2'——同系列 bc2（val 拟合 δ → test 应用，HK 同口径）。
    GenCast 全球线未评估（json 缺失 → None）。
    0h 点从 fc0 JSON 覆盖（分析场，成员间相同；bc2 口径取 _fc0_bc2）。"""
    f = os.path.join(RUN, f"forecast_fulltest_crps_{kind}.json")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        d = json.load(fh)
    fc0_name = ("forecast_fulltest_crps_fc0_bc2.json"
                if kind.endswith("_bc2") else "forecast_fulltest_crps_fc0.json")
    fc0 = os.path.join(RUN, fc0_name)
    if os.path.exists(fc0):
        with open(fc0) as fh:
            d["leads"]["0"] = json.load(fh)["leads"]["0"]
    return d


def draw_global_crps_ax(ax, d, era5=None, legend=True, member=None,
                        annotate_leads=(12, 72), note0=True,
                        annotate_member=0):
    """全球 fair CRPS vs lead（主图 panel b，样式与 HK panel a 一致）。
    era5: forecast_era5full_crps.json 的 dict，给红色虚线完美强迫参照。
    member: (label, color, marker, json_dict) 列表 —— 逐成员强迫曲线
    （member k → WaveFlow 样本 k）：[("WaveFlow + IFS ENS (ours)", "#B2182B",
    "8", memens_dict), ("WaveFlow + GenCast (AI ensemble)", "#2166AC", "s",
    gc_dict)]；给定时 ens-mean 主线撤下（留 supplement）。
    d["leads"] 含 "0" 时加 0h 点（fc0 / IFS HRES step0 分析场强迫，
    由 load_crps_json 从 forecast_fulltest_crps_fc0.json 覆盖）。
    legend=False 时不画自带图例（主图 a/b 共用一排图例），参照线数值改为
    线右端的小字标注。"""
    leads = list(LEADS)
    if "0" in d["leads"]:
        leads = [0] + leads
    x = np.array(leads)
    y = [d["leads"][str(L)]["crps_fair"] for L in leads]
    pers = [d["leads"][str(L)]["persistence_mae"] for L in leads]
    clim = d["leads"][str(leads[0])]["climatology_crps"]
    ax.axhline(clim, color="#999999", lw=0.8, ls=":",
               label="climatology")
    if era5:
        ax.axhline(era5["crps_fair"], color="#B2182B", lw=0.8, ls="--",
                   alpha=0.65,
                   label="WaveFlow + ERA5")
    ax.plot(x, pers, marker="x", ms=4, lw=1.0, ls="--", color="#7F7F7F",
            label="persistence")
    if member:
        # 逐成员口径：主图只画 member 曲线（0h 成员间相同 = fc0）
        for mi, (label, color, marker, md) in enumerate(member):
            mleads = list(LEADS)
            if "0" in md["leads"]:
                mleads = [0] + mleads
            my = [md["leads"][str(L)]["crps_fair"] for L in mleads]
            ax.plot(np.array(mleads), my, marker=marker, ms=4, lw=1.4,
                    ls="-", color=color, markeredgecolor="white",
                    markeredgewidth=0.4, label=label)
            for xi, yi in zip(mleads, my):
                if annotate_member is not None and mi != annotate_member:
                    continue
                if annotate_leads is not None and xi not in annotate_leads:
                    continue
                skill = 100 * (1 - yi / clim)
                ax.annotate(f"+{skill:.0f}%", (xi, yi),
                            textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=6,
                            color=color)
    else:
        ax.plot(x, y, marker="8", ms=4, lw=1.4, ls="-", color="#B2182B",
                markeredgecolor="white", markeredgewidth=0.4,
                label="WaveFlow + IFS ENS (ours)")
        for xi, yi in zip(x, y):
            skill = 100 * (1 - yi / clim)
            ax.annotate(f"+{skill:.0f}%", (xi, yi), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=6, color="#B2182B")
    # 参照线数值标在线右端（共用图例不带数值，panel a 数值不同）
    ax.annotate(f"{clim:.3f}", (77, clim), ha="right", va="bottom",
                fontsize=5.5, color="#999999")
    if era5:
        ax.annotate(f"{era5['crps_fair']:.3f}", (77, era5["crps_fair"]),
                    ha="right", va="top", fontsize=5.5,
                    color="#B2182B", alpha=0.65)
    if 0 in leads:
        if note0:
            ax.text(0.30, 0.02, "0 h: analysis forcing",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6, color="#999999", style="italic")
    ax.set_xlabel("forecast lead time (h)")
    ax.set_ylabel("fair CRPS, $H_s$ (m)")
    ax.set_xticks(x)
    ax.set_xlim(-5 if 0 in leads else 6, 78)
    ax.set_ylim(0, None)
    if legend:
        ax.legend(frameon=False, loc="upper center",
                  bbox_to_anchor=(0.48, -0.33), ncol=2, handlelength=1.8,
                  columnspacing=1.2)


def draw_global_mean_row(fig, lons, lats, gt, x0, y_top, map_w, dx,
                         canvas=None, rmse=None, show_rmse=True,
                         ts=None, tmpls=None):
    """GT + 4 lead 集合平均 hs Robinson 地图行（主图 panel h）。
    rmse: {lead: {...}}；show_rmse=True 时在标题第二行注窗口 RMSE。"""
    map_h = map_w / 1.9717               # Robinson 椭圆纵横比
    cw, ch = canvas or (CANVAS_W, CANVAS_H)
    sc = None
    # 2026-11-04：标题与 panel g 同口径（Reference / L h lead）
    titles = ["Reference"] + [f"{L} h lead" for L in LEADS]
    ts = ts or T_MAP
    for i in range(5):
        title = titles[i]
        if i == 0:
            vals = gt
        else:
            L = LEADS[i - 1]
            vals = ens_mean_hs(L, ts, (tmpls or {}).get(L))
            # 仅 draft 用：标题第二行注窗口 RMSE（主图传 show_rmse=False）
            if show_rmse and rmse and str(L) in rmse:
                title += f"\nwindow RMSE {rmse[str(L)]['rmse']:.2f} m"
        _, sc = draw_robinson(fig, x0 + i * dx, y_top, map_w, map_h,
                              lons, lats, vals, title, "turbo", 0,
                              HS_VMAX, canvas=canvas)
    # 色条位置与 panel g 的色条一致（x0 + 4*dx + map_w + 1.5）
    cba = fig.add_axes([(x0 + 4 * dx + map_w + 1.5) / cw,
                        1 - (y_top + map_h) / ch,
                        1.2 / cw, map_h / ch])
    cb = fig.colorbar(sc, cax=cba)
    cb.set_label("$H_s$ (m)", fontsize=6)
    cb.ax.tick_params(labelsize=6, length=2)


# Fiona 台风区放大框（lon0, lon1, lat0, lat1）：覆盖 2022-09-24 00Z 的高海况
# 区（真值 Hs>7 m 落在 64–31°W、34–44°N）以及各 lead 差异最集中的西侧。
ZOOM = (-80.0, -25.0, 25.0, 50.0)


def draw_zoom_row(fig, lons, lats, gt, x0, y_top, map_w, dx, canvas=None,
                  ts=None, tmpls=None, region=None, dot_s=2.5, titles=False):
    """panel i：台风区域的放大图（5 列与 g/h 同列距）。

    panel h 是全球 Robinson 图，台风细节（最大浪区形状、各 lead 差异）看不清，
    这里用等经纬 PlateCarree 放大局部区域。色标按**放大区域内**的实际数据定
    （1–99.5 分位，取整到 1 m），不沿用全球的 0–12 m。
    """
    lon0, lon1, lat0, lat1 = region or ZOOM
    map_h = map_w * (lat1 - lat0) / (lon1 - lon0)     # degrees 1:1
    cw, ch = canvas or (CANVAS_W, CANVAS_H)
    ts = ts or T_MAP
    fields = [gt] + [ens_mean_hs(L, ts, (tmpls or {}).get(L)) for L in LEADS]
    sel = ((lons >= lon0) & (lons <= lon1) & (lats >= lat0) & (lats <= lat1))
    vals = np.concatenate([np.asarray(f)[sel] for f in fields])
    # 用放大区内 5 张图的**真实** min/max（不取分位、也不再向上取整）
    # 2026-11-05 用户要求：最大值是多少就用多少（如 7.65 m）
    vmin = max(0.0, float(np.nanmin(vals)))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, HS_VMAX
    sc = None
    for i, vals_i in enumerate(fields):
        ax = ax_mm(fig, x0 + i * dx, y_top, map_w, map_h,
                   projection=ccrs.PlateCarree(), canvas=canvas)
        ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
        # 原始节点散点，与 panel g 同一画法（不做任何插值，2026-11-05）
        sc = ax.scatter(lons, lats, c=vals_i, s=dot_s, cmap="turbo",
                        vmin=vmin, vmax=vmax, edgecolors="none", linewidths=0,
                        transform=ccrs.PlateCarree(), rasterized=True)
        ax.coastlines(linewidth=0.4, color="#555555")
        # 经纬标签：首图左侧纬度（竖排、中心对刻度线），每图下方经度（2026-11-05）
        # 5 个亚图并排、每个约 26 mm，5° 一个标签会在相邻亚图接缝处相撞
        # （"75°W" 紧挨下一个的 "90°W"），所以窄图用 10° 步长
        span = lon1 - lon0
        if map_w < 32 and span <= 25:
            xstep = 10
        elif span <= 20:
            xstep = 5
        elif span <= 40:
            xstep = 10
        else:
            xstep = 15
        ystep = 5 if (lat1 - lat0) <= 12 else (10 if (lat1 - lat0) <= 20 else 15)
        xloc = np.arange(np.ceil(lon0 / xstep) * xstep, lon1 + 1e-9, xstep)
        yloc = np.arange(np.ceil(lat0 / ystep) * ystep, lat1 + 1e-9, ystep)
        ax.set_xticks(xloc, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax.tick_params(axis="x", labelsize=5.5, length=1.8, width=0.5,
                       pad=1.0)
        if i == 0:
            ax.set_yticks(yloc, crs=ccrs.PlateCarree())
            ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
            ax.tick_params(axis="y", labelsize=6, length=2, width=0.5, pad=0.6)
            for lab in ax.get_yticklabels():
                lab.set_rotation(90)
                lab.set_rotation_mode("anchor")
                lab.set_ha("center")
                lab.set_va("bottom")
        else:
            ax.tick_params(axis="y", labelleft=False, length=0)
        if titles:
            ax.set_title(("Reference" if i == 0 else f"{LEADS[i - 1]} h lead"),
                         fontsize=7, pad=1.5)
    cba = fig.add_axes([(x0 + 4 * dx + map_w + 1.5) / cw,
                        1 - (y_top + map_h) / ch, 1.2 / cw, map_h / ch])
    cb = fig.colorbar(sc, cax=cba)
    cb.set_label("$H_s$ (m)", fontsize=6)
    cb.ax.tick_params(labelsize=6, length=2)
    # 刻度必须落在 [vmin, vmax) 内：默认 locator 会给到 0/5/10 这种越界刻度，
    # 顶端刻度会被画到色条外面（看起来像"溢出"）。
    # 2026-11-07 用户要求：放大区真实最大值（如 7.65 m）不再单独标在色条顶端，
    # 只保留整步长刻度（2/4/6）；色条长度仍按真实 min/max 铺满。
    step = 2.0 if (vmax - vmin) <= 10 else 5.0
    ticks = list(np.arange(np.ceil((vmin + 1e-9) / step) * step,
                           vmax - 1e-6, step))
    if not ticks:                        # 窄区间一个整步长都没有时标下限
        ticks = [float(vmin)]
    cb.set_ticks(ticks)
    from matplotlib import ticker as _mticker
    cb.ax.yaxis.set_major_formatter(
        _mticker.FuncFormatter(lambda v, _: f"{v:g}"))   # 2/4/6 不乱补零
    return map_h


def draw_global_rmse_ax(ax, rmse):
    """Fiona 窗口集合平均 RMSE vs lead（主图 panel g）。"""
    x = np.array(LEADS)
    y = [rmse[str(L)]["rmse"] for L in LEADS]
    ax.plot(x, y, marker="8", ms=4, lw=1.4, ls="-", color="#B2182B",
            markeredgecolor="white", markeredgewidth=0.4,
            label="WaveFlow + IFS ENS (global)")
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=6, color="#B2182B")
    ax.set_xlabel("forecast lead time (h)")
    ax.set_ylabel("ens. mean RMSE (m)")
    ax.set_xticks(x)
    ax.set_xlim(6, 78)
    ax.set_ylim(0, 0.5)
    ax.set_title(f"Hurricane Fiona window, {rmse[str(LEADS[0])]['n_ts']} "
                 f"valid times", fontsize=6, pad=2)


def gt_hs(ts):
    return np.load(os.path.join(
        GT_DIR, ts[:4],
        f"global_hindcast_{ts[:8]}_{ts[9:11]}.npz"))["hs"]


def ens_mean_hs(lead, ts, tmpl=None):
    """集合平均场（逐成员 ENS 强迫）。

    tmpl 给定时只读该目录（案例图用，如 forecast_ian_memens_lead24h）；
    否则退回 Fiona 窗口的 gc → memens 两组目录。
    """
    for tmpl in ([tmpl] if tmpl else
                 [f"forecast_fiona_memens_lead{lead}h",
                  f"forecast_fiona_gc_lead{lead}h"]):
        sdir = os.path.join(RUN, tmpl, "samples")
        fp = os.path.join(sdir, f"sample_{ts}_ens000.npz")
        if os.path.exists(fp):
            e = np.stack([np.load(os.path.join(
                sdir, f"sample_{ts}_ens{k:03d}.npz"))["sample"]
                for k in range(N_MEMBERS)])
            return np.nanmean(e, axis=0)
    raise FileNotFoundError(
        f"no fiona samples for lead {lead} (gc/memens both missing)")


def draw_robinson(fig, x, y, w, h, lons, lats, vals, title, cmap, vmin, vmax,
                  canvas=None):
    ax = ax_mm(fig, x, y, w, h, projection=ccrs.Robinson(), canvas=canvas)
    ax.set_global()
    ax.coastlines(linewidth=0.3, color="#666666")
    sc = ax.scatter(lons, lats, c=vals, s=0.15, cmap=cmap, vmin=vmin,
                    vmax=vmax, edgecolors="none", linewidths=0,
                    transform=ccrs.PlateCarree(), rasterized=True)
    ax.set_title(title, fontsize=7, pad=1.5)   # 与 g 行标题同号
    return ax, sc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    c = np.load(COORDS)
    lats, lons = c[:, 0], c[:, 1]
    gt = gt_hs(T_MAP)
    rmse = load_rmse_json()

    MM = 1 / 25.4
    fig = plt.figure(figsize=(CANVAS_W * MM, CANVAS_H * MM))

    MAP_W = 31.0
    MAP_H = MAP_W / 1.9717             # Robinson 椭圆纵横比
    DX = 32.5
    X0 = 6.0
    Y1, Y2 = 6.0, 30.0

    draw_global_mean_row(fig, lons, lats, gt, X0, Y1, MAP_W, DX, rmse=rmse)
    fig.text(2 / CANVAS_W, 1 - Y1 / CANVAS_H, "d", fontsize=9,
             fontweight="bold", ha="left", va="top")

    sce = None
    for i, L in enumerate(LEADS):
        err = ens_mean_hs(L, T_MAP) - gt
        _, sce = draw_robinson(fig, X0 + (i + 1) * DX, Y2, MAP_W, MAP_H,
                               lons, lats, err, f"{L} h lead, mean − truth",
                               "RdBu_r", -ERR_VMAX, ERR_VMAX)
    cbb = fig.add_axes([(X0 + 5 * DX + 1.0) / CANVAS_W,
                        1 - (Y2 + MAP_H) / CANVAS_H,
                        1.2 / CANVAS_W, MAP_H / CANVAS_H])
    cbe = fig.colorbar(sce, cax=cbb)
    cbe.set_label("error (m)", fontsize=6)
    cbe.ax.tick_params(labelsize=6, length=2)
    fig.text(2 / CANVAS_W, 1 - Y2 / CANVAS_H, "e", fontsize=9,
             fontweight="bold", ha="left", va="top")

    fig.savefig(OUT_BASE + ".png", dpi=300)
    fig.savefig(OUT_BASE + ".pdf")
    print("saved:", OUT_BASE + ".png/.pdf")


if __name__ == "__main__":
    main()
