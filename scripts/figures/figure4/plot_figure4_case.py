#!/usr/bin/env python
"""Fig. S2 — 风暴案例场级地图（样本真实感 + 超越概率）。

数据源（均见 docs/figure4_plan.md 第 2 节）：
  - WaveFlow m100 样本：<run>/curve_ep49_m100/test/samples/sample_{ts}_ensNNN.npz
    （100 成员，"sample" (7788,4) float32 物理空间，列序 hs/tp/dm/elev）
  - Deep ensemble：<run>/../uq_baselines/ens_s10{1..5}_*/test/samples/
    （5 个 seed 目录各 1 成员，拼成 M=5）
  - 真值 GT：data/hindcast/processed_npz/wave_{ts}.npz（keys hs/tp/dm/elev）
  - 节点坐标：data/hindcast/hk_coordinates.npy，(7788,2) = (lat, lon)

风暴时刻 T* 的选择：扫描 test 期全部 17,519 个 GT 文件（2021–2022 逐小时，
ProcessPool 并行），按全场 nanmax(hs) 排序：
  1) 20211012T22  max=6.05 m  mean=2.04 m   <- T*（台风圆规 Kompasu 过境）
  2) 20211012T21  max=6.02 m
按全场 nanmean(hs) 排序最大为 20220824T21（max=5.45 m, mean=2.33 m，马鞍），
作为备选；本图用 max-hs 最高的 T* = 20211012T22。T* 在 m100/m10/DE 样本目录
中均有完整成员。

版面：3x3。
  a ERA5 真值 hs | b WaveFlow 集合平均 (M=100) | c WaveFlow 集合 std
  d-f 3 个 WaveFlow 成员（ens000/003/007）
  g Deep ensemble 平均 (M=5) | h Deep ensemble std | i P(hs>4m) + 真值 4 m 等值线
hs 各 panel 共享 0–6 m 色标；std 共享独立色标；概率 0–100%。
另单独导出 P(hs>4m) panel 为 figure4_exceedance.png（dpi 300）。
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")

# cartopy 离线缓存在仓库内，必须在 import cartopy 之前设置
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # scripts/figures/figure4/ 上三级
sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("CARTOPY_DATA_DIR", os.path.join(_REPO_ROOT, "data/cartopy_cache"))

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from modules.plot.plot_wave_hk import plot_wave_data

RUN = ("output/results/hk_cleaned/all_var/f8_blr5e-5_amp_stable_tod/size_sweep/"
       "base_v1_20260821_025917")
DE_GLOB = ("output/results/hk_cleaned/all_var/f8_blr5e-5_amp_stable_tod/size_sweep/"
           "uq_baselines/ens_s10*_*/test/samples")
MCDO_DIR = ("output/results/hk_cleaned/all_var/f8_blr5e-5_amp_stable_tod/size_sweep/"
            "uq_baselines/mcdrop_20260821_205134/test/samples")
NLL_DIR = ("output/results/hk_cleaned/all_var/f8_blr5e-5_amp_stable_tod/size_sweep/"
           "uq_baselines/nll_20260821_205231/test/samples")
SS = ("output/results/hk_cleaned/all_var/f8_blr5e-5_amp_stable_tod/size_sweep/"
      "uq_baselines")
# cfm_bc2/sde_g10 与 WaveFlow 共享同一 checkpoint 重采样，目录在 RUN/uq_baselines/<tag>
# (cfm_bc2 = ours 偏差订正版, 主图 panel a 的 WaveFlow 行与 P(hs>4m) 用它, M=10)
# msep_* = 输入扰动 σ=0.02/0.05/0.10，单独训练的 MSE 基线（2026-08-25 起替换
# 旧的 condn_* 共享-checkpoint 版），目录在 size_sweep/uq_baselines/<tag>
BASELINE_DIRS = {
    "cfm_bc2":  f"{RUN}/uq_baselines/cfm_bc2/test/samples",
    "sde_g10":  f"{RUN}/uq_baselines/sde_g10/test/samples",
    "msep_g02": f"{SS}/msep_g02/test/samples",
    "msep_g05": f"{SS}/msep_g05/test/samples",
    "msep_g10": f"{SS}/msep_g10/test/samples",
}
GT_DIR = "data/hindcast/processed_npz"
COORDS = "data/hindcast/hk_coordinates.npy"
OUT_DIR = "output/plot/figure4/main"

T_STAR = "20211012T22"          # 见 docstring：test 期 GT 全场 max hs 最高时刻
MEMBERS_SHOW = [0, 3, 7]        # 展示的 WaveFlow 成员编号
HS_VMAX = 6.0                   # hs 共享色标上限（案例 max = 6.05 m）
EXCEED_THRESH = 4.0             # P(hs > thresh) 阈值 (m)
from modules.utils import HK_REGION

REGION = list(HK_REGION)   # [min_lon, max_lon, min_lat, max_lat] 与 figure 2 一致

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "TeX Gyre Heros", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 13,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_hs_samples(sample_dir, ts, n_members):
    """读 n_members 个成员的 hs 列，返回 (M, 7788) float32。"""
    out = []
    for m in range(n_members):
        f = os.path.join(sample_dir, f"sample_{ts}_ens{m:03d}.npz")
        out.append(np.load(f)["sample"][:, 0])
    return np.stack(out)


def load_deep_ensemble(ts):
    """5 个 seed 目录各取 1 成员，返回 (5, 7788)。"""
    dirs = sorted(glob.glob(DE_GLOB))
    assert len(dirs) == 5, f"expect 5 DE seed dirs, got {len(dirs)}"
    out = []
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, f"sample_{ts}_ens*.npz")))
        assert files, f"no DE sample for {ts} in {d}"
        out.append(np.load(files[0])["sample"][:, 0])
    return np.stack(out)


def make_ds(values, lons, lats):
    """Wrap a (7788,) node array into an xr.Dataset for plot_wave_data."""
    import xarray as xr
    return xr.Dataset(
        {"hs": (["node"], values)},
        coords={"lon": (["node"], lons), "lat": (["node"], lats)},
    )


def draw_panel(fig, ax, values, lons, lats, title, vmin, vmax, cmap):
    plot_wave_data(make_ds(values, lons, lats), variable="hs", region=REGION,
                  vmin=vmin, vmax=vmax, cmap=cmap, ax=ax, return_ax=True,
                  scatter_size=8, custom_title=title, show_cities=False,
                  show_colorbar=False, show_nodes=False,
                  gridline_spacing=(0.5, 0.5))
    return ax._wave_scatter


def add_gt_contour(ax, gt_hs, lons, lats, level=EXCEED_THRESH, zorder=27):
    """在概率图上叠加真值 hs = 4 m 等值线（griddata 线性插值到规则网格）。

    zorder 默认 27：压过散点(6)与岸线(20)，只低于图内图例(30)——4 m 等值线是
    判断"大浪区在哪"的参照，被数据点或岸线切断就没法读了。
    """
    from scipy.interpolate import griddata
    lon_grid = np.linspace(REGION[0], REGION[1], 400)
    lat_grid = np.linspace(REGION[2], REGION[3], 400)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    grid = griddata((lons, lats), gt_hs, (lon_mesh, lat_mesh), method="linear")
    cs = ax.contour(lon_mesh, lat_mesh, grid, levels=[level], colors="black",
                    linewidths=0.8, transform=ccrs.PlateCarree(),
                    zorder=zorder)
    # matplotlib>=3.8 移除 ContourSet.collections;旧版走 collections[0]
    if hasattr(cs, "collections"):
        cs.collections[0].set_label(f"truth $H_s$ = {level:.0f} m")
    else:
        cs.set_label(f"truth $H_s$ = {level:.0f} m")


def load_case_data():
    """加载 T* 时刻的 GT / WaveFlow m100 / DE 样本，返回绘图所需全部数组。"""
    coords = np.load(COORDS)
    lats, lons = coords[:, 0], coords[:, 1]  # 已验证：第一列 lat、第二列 lon

    gt = np.load(os.path.join(GT_DIR, f"wave_{T_STAR}.npz"))["hs"].astype(np.float64)
    print(f"GT {T_STAR}: max hs = {np.nanmax(gt):.2f} m, "
          f"mean hs = {np.nanmean(gt):.3f} m, NaN nodes = {np.isnan(gt).sum()}")

    # T* 前后 ±12 h 的风暴演化（仅打印，验证空间/时间一致性）
    from datetime import datetime, timedelta
    t0 = datetime.strptime(T_STAR, "%Y%m%dT%H")
    for dt in (-12, 0, 12):
        ts = (t0 + timedelta(hours=dt)).strftime("%Y%m%dT%H")
        h = np.load(os.path.join(GT_DIR, f"wave_{ts}.npz"))["hs"]
        print(f"  T*{dt:+d}h {ts}: max={np.nanmax(h):.2f} m, mean={np.nanmean(h):.3f} m")

    cfm = load_hs_samples(os.path.join(RUN, "curve_ep49_m100/test/samples"),
                          T_STAR, 100)
    de = load_deep_ensemble(T_STAR)
    mcdo = load_hs_samples(MCDO_DIR, T_STAR, 10)
    nll = load_hs_samples(NLL_DIR, T_STAR, 10)
    base = {t: load_hs_samples(d, T_STAR, 10)
            for t, d in BASELINE_DIRS.items()}
    cfm_mean, cfm_std = cfm.mean(0), cfm.std(0)
    de_mean, de_std = de.mean(0), de.std(0)
    mcdo_mean, mcdo_std = mcdo.mean(0), mcdo.std(0)
    nll_mean, nll_std = nll.mean(0), nll.std(0)
    prob = (cfm > EXCEED_THRESH).mean(0)  # m100 经验超越概率
    spread_msg = " | ".join(
        [f"WaveFlow {cfm_std.mean():.3f}", f"DE {de_std.mean():.3f}",
         f"MCdrop {mcdo_std.mean():.3f}", f"NLL {nll_std.mean():.3f}"]
        + [f"{t} {base[t].std(0).mean():.3f}" for t in BASELINE_DIRS])
    print(f"spread (node-mean std, m): {spread_msg} | "
          f"P(hs>{EXCEED_THRESH:.0f}m) max = {prob.max():.2f}")
    print(f"WaveFlow member RMSE vs mean-field: "
          f"{np.sqrt(np.mean((cfm[MEMBERS_SHOW[0]] - cfm_mean) ** 2)):.3f} m "
          f"(member {MEMBERS_SHOW[0]:03d}, sharpness check)")

    out = {
        "lons": lons, "lats": lats, "gt": gt, "cfm": cfm, "de": de,
        "mcdo": mcdo, "nll": nll,
        "cfm_mean": cfm_mean, "cfm_std": cfm_std,
        "de_mean": de_mean, "de_std": de_std,
        "mcdo_mean": mcdo_mean, "mcdo_std": mcdo_std,
        "nll_mean": nll_mean, "nll_std": nll_std, "prob": prob,
        "std_vmax": float(np.ceil(max(cfm_std.max(), de_std.max()) * 10) / 10),
        "nice_time": f"{T_STAR[:4]}-{T_STAR[4:6]}-{T_STAR[6:8]} {T_STAR[9:]}:00",
    }
    for t in BASELINE_DIRS:
        out[t] = base[t]
        out[f"{t}_mean"], out[f"{t}_std"] = base[t].mean(0), base[t].std(0)
    return out


def draw_case_block(fig, gs, d, wspace=0.08, hspace=0.22):
    """在 gridspec 区域 gs 内绘制 3x3 案例面板 + 底部三条共享色标。"""
    inner = gs.subgridspec(4, 3, height_ratios=[1, 1, 1, 0.045],
                           hspace=hspace, wspace=wspace)
    axes = [fig.add_subplot(inner[i // 3, i % 3], projection=ccrs.PlateCarree())
            for i in range(9)]
    lons, lats, gt = d["lons"], d["lats"], d["gt"]
    cfm, de = d["cfm"], d["de"]

    hs_panels = [
        (0, gt, "(a) ERA5 truth $H_s$"),
        (1, d["cfm_mean"], "(b) WaveFlow ens. mean $H_s$ (M=100)"),
        (3, cfm[MEMBERS_SHOW[0]], f"(d) WaveFlow member {MEMBERS_SHOW[0]:03d}"),
        (4, cfm[MEMBERS_SHOW[1]], f"(e) WaveFlow member {MEMBERS_SHOW[1]:03d}"),
        (5, cfm[MEMBERS_SHOW[2]], f"(f) WaveFlow member {MEMBERS_SHOW[2]:03d}"),
        (6, d["de_mean"], "(g) Deep ens. mean $H_s$ (M=5)"),
    ]
    for idx, vals, title in hs_panels:
        draw_panel(fig, axes[idx], vals, lons, lats, title, 0.0, HS_VMAX, "turbo")
    std_panels = [
        (2, d["cfm_std"], "(c) WaveFlow ens. std $H_s$"),
        (7, d["de_std"], "(h) Deep ens. std $H_s$"),
    ]
    for idx, vals, title in std_panels:
        draw_panel(fig, axes[idx], vals, lons, lats, title, 0.0, d["std_vmax"],
                   "viridis")
    sc_prob = draw_panel(fig, axes[8], d["prob"], lons, lats,
                         f"(i) WaveFlow $P(H_s > {EXCEED_THRESH:.0f}$ m) (M=100)",
                         0.0, 1.0, "Reds")
    add_gt_contour(axes[8], gt, lons, lats)
    axes[8].legend(loc="lower left", fontsize=10, framealpha=0.9)

    # 三条共享色标：hs / std / 概率，横置于底部（占 inner 第 4 行）
    cb_gs = inner[3, :].subgridspec(1, 5, width_ratios=[0.42, 0.11, 0.17,
                                                        0.10, 0.17],
                                    wspace=0.0)
    cax_hs = fig.add_subplot(cb_gs[0, 0])
    fig.colorbar(plt.cm.ScalarMappable(
        norm=mpl.colors.Normalize(0, HS_VMAX), cmap="turbo"),
        cax=cax_hs, orientation="horizontal",
        label="Significant wave height $H_s$ (m)")
    cax_std = fig.add_subplot(cb_gs[0, 2])
    fig.colorbar(plt.cm.ScalarMappable(
        norm=mpl.colors.Normalize(0, d["std_vmax"]), cmap="viridis"),
        cax=cax_std, orientation="horizontal", label="$H_s$ ensemble std (m)")
    cax_pr = fig.add_subplot(cb_gs[0, 4])
    cbar_pr = fig.colorbar(sc_prob, cax=cax_pr, orientation="horizontal",
                           label=f"$P(H_s > {EXCEED_THRESH:.0f}$ m)")
    cbar_pr.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar_pr.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    return axes


def draw_exceedance(fig, ax, d, title=None):
    """在给定 cartopy ax 上绘制 P(hs>4m) + 真值 4 m 等值线，返回 scatter。"""
    if title is None:
        title = (f"WaveFlow $P(H_s > {EXCEED_THRESH:.0f}$ m), M=100"
                 f" — {d['nice_time']}")
    sc = draw_panel(fig, ax, d["prob"], d["lons"], d["lats"], title,
                    0.0, 1.0, "Reds")
    add_gt_contour(ax, d["gt"], d["lons"], d["lats"])
    ax.legend(loc="lower left", fontsize=11, framealpha=0.9)
    return sc


def exceedance_colorbar(fig, sc, ax, fontsize=13):
    """exceedance panel 的百分比色标（主图与单图共用）。"""
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08,
                        shrink=0.8, aspect=30)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.set_label(f"$P(H_s > {EXCEED_THRESH:.0f}$ m)", fontsize=fontsize)
    return cbar


def main():
    d = load_case_data()

    # ---------------- 3x3 大图 ----------------
    fig = plt.figure(figsize=(19, 15))
    gs = fig.add_gridspec(1, 1, left=0.04, right=0.985, top=0.94, bottom=0.09)
    draw_case_block(fig, gs[0, 0], d)
    fig.suptitle(f"Storm case: {d['nice_time']} (GT max $H_s$ = "
                 f"{np.nanmax(d['gt']):.2f} m)", fontsize=17, y=0.985)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"figure4_case.{ext}")
        fig.savefig(path, dpi=300, facecolor="white")
        print(f"saved: {path}")
    plt.close(fig)

    # ---------------- P(hs>4m) 单独导出 ----------------
    fig2 = plt.figure(figsize=(8.5, 7.5))
    ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    sc2 = draw_exceedance(fig2, ax2, d)
    exceedance_colorbar(fig2, sc2, ax2)
    path2 = os.path.join(OUT_DIR, "figure4_exceedance.png")
    fig2.savefig(path2, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved: {path2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
