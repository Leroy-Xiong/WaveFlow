#!/usr/bin/env python
"""Fig. 6 共享工具：越浪概率预警图的样式、数据加载与地图底层。

默认绘制 Saola(2023) 版；环境变量 FIG6_CASE=ragasa 切换到 Ragasa(2025)
版（数据路径、事件锚点、代表段、输出文件名后缀全部由 CASE_SPECS 驱动，
输出为 figure6_main_ragasa.{png,pdf} 等，不覆盖 Saola 版）。

- Nature 风格 rcParams（同 figure4/5：Liberation Sans，字号阶梯 9/7/6）
- 毫米绝对定位 ax_mm（同 plot_figure5_case.ax_mm）
- 数据加载：岸线分段 CSV、越浪 npz（M=10 / M=100）、分段 summary、
  HKO 灾情记录、模型节点坐标、峰值时刻集合平均 hs 背景、panel c 节点
  elev 集合（4900 个样本文件，带磁盘缓存，缓存放 output/plot/figure6/）
- 地图底层：cartopy 仅用 PlateCarree 投影与 gridlines（离线）；陆地底图 =
  港府官方正射影像（data/hk_coastline/orthophoto/hk_orthophoto_wgs84.npz，
  LandsD GeoData Store 影像瓦片，陆地显示/海面透明，陆地掩膜来自官方
  5 m DTM + NE 10m land 补深圳侧；fetch_landsd_orthophoto.py 可重建，
  无缓存时自动退化为白底）

Saola 事件锚点：峰值图 valid = 2023-09-01T16 UTC (= 00:00 HKT 2 Sep)——
hybrid 堤顶插补后 crest_flag=1 已清零，全段口径领域平均 P 的峰值时刻
（T16 0.03499 > T15 0.03423）；最接近时刻 2023-09-01 21:00 HKT
(= 13:00 UTC)；十号信号 2023-09-01 20:15 HKT .. 2023-09-02 03:40 HKT。

Ragasa 事件锚点：最接近 2025-09-24 06:00 HKT（中心在香港以南 ~120 km，
23 Sep 22:00 UTC）；十号信号 2025-09-24 02:40 .. 13:20 HKT；峰值时刻
T_PEAK 待 m10 领域平均 P 峰值确认（现为占位值）。
"""
import json
import glob
import os
import sys
from datetime import datetime, timedelta

_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))
os.chdir(REPO_ROOT)
os.environ.setdefault("CARTOPY_DATA_DIR", os.path.join(REPO_ROOT,
                                                       "data/cartopy_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def setup_style():
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
    # 字体: Helvetica 系 (本机无 Arial 时用 TeX Gyre Heros) + 同源 mathtext;
    # 必须在上面 rcParams.update 之后调用
    _figs_dir = os.path.dirname(_DIR)
    if _figs_dir not in sys.path:
        sys.path.insert(0, _figs_dir)
    import typography as TY
    TY.use_nature_fonts()


# ---- 路径 ----
# 案例开关：FIG6_CASE=ragasa 时整套图换成 Ragasa(2025) 的数据/锚点/代表段，
# 输出文件名加 _ragasa 后缀；默认 saola 与原行为完全一致。
# FIG6_CASE=helene 走美国案例（Helene 2024，Tampa Bay），数据/常量的 case
# 化见下方各 `or` 回退——所有回退默认值都等于原 HK 行为。
CASE = os.environ.get("FIG6_CASE", "saola")
CASE_SPECS = {
    "saola": dict(
        summary_csv="data/hk_coastline/overtopping/saola_segment_summary_hybrid.csv",
        npz_m100="data/hk_coastline/overtopping/saola_q_m100_hybrid.npz",
        npz_m10="data/hk_coastline/overtopping/saola_q_m10_hybrid.npz",
        hko_json="data/hk_coastline/saola_hko_records.json",
        samples_m100="case_saola2023_m100",
        cache_npz="_cache_node_ensemble_m100.npz",
        t_peak="20230901T16",
        t_closest_hkt=datetime(2023, 9, 1, 21, 0),
        sig10_hkt=(datetime(2023, 9, 1, 20, 15), datetime(2023, 9, 2, 3, 40)),
        panel_b_title="Territory peak: 16:00 UTC (00:00 HKT) 2 Sep 2023",
        closest_label="closest approach\n21:00 HKT",
        x_label="time (HKT, 2023)",
        # 48-h 跨度、49 个逐时刻（31 Aug 12Z .. 2 Sep 12Z 含两端）
        expo_window_title="48-h peak window (31 Aug 12Z – 2 Sep 12Z)",
        expo_ylim_km=(0, 5), expo_ylim_pct=(0, 27),
        out_suffix="",
        # 代表段（逐一核实 crest_flag=0、event-max P ≥ 0.9）
        panel_c_segs=[
            dict(seg_id=2143, name="Tai O (Lantau West)",
                 kind="sloping revetment"),
            dict(seg_id=6472, name="Kwai Chung–Stonecutters",
                 kind="vertical seawall"),
            dict(seg_id=26, name="Lei Yue Mun (Kwun Tong)",
                 kind="sloping revetment"),
        ],
    ),
    "ragasa": dict(
        # 2026-09-07 数据修正版：hybridspot（flag0 段也用 SpotHeight p90，
        # 堤顶中位 -2.8 m）+ typerev（z17/z19 正射影像核对的 175 段类型修正，
        # 见 output/review_coasttype/REVIEW_NOTES.md）+ gex（15 段几何落线
        # 问题段剔除：段线跨湾口/桥面/水面）；原始 hybrid 产物
        # 保留为 ragasa_*_hybrid.{csv,npz} 作对照
        summary_csv="data/hk_coastline/overtopping/ragasa_segment_summary_hybridspot_typerev_gex.csv",
        npz_m100="data/hk_coastline/overtopping/ragasa_q_m100_hybridspot_typerev_gex.npz",
        npz_m10="data/hk_coastline/overtopping/ragasa_q_m10_hybridspot_typerev_gex.npz",
        type_overrides="data/hk_coastline/overtopping/coast_type_overrides.csv",
        exclude_segs="data/hk_coastline/overtopping/geometry_exclusions.csv",
        hko_json="data/hk_coastline/ragasa_hko_records.json",
        samples_m100="case_ragasa2025_m100",
        cache_npz="_cache_node_ensemble_m100_ragasa.npz",
        # 单发布时次预报（panel d/e 叠加）：IFS ENS 逐成员强迫，init 2025-09-22
        # 12Z，valid 2025-09-23T00Z..25T00Z 每 3 h（lead 12–60 h）。产物由
        # run_ragasa_forecast_sample.sh + compute_overtopping_forecast.py 生成；
        # 文件不齐时整条预报链路自动跳过（saola 规格没有这些键 -> 恒不画）。
        forecast_init="20250922T12",
        # 100 个生成样本/时刻 = 50 个 IFS ENS 大气成员（单 cycle 全部扰动成员）
        # × 2 次生成采样（样本 j 用成员 j % 50）
        forecast_samples="forecast_ensmem_ragasa_init20250922T12_m50x2",
        forecast_npz=("data/hk_coastline/overtopping/"
                      "ragasa_q_forecast_init20250922T12_m50x2.npz"),
        forecast_members=100,
        # 图注用（图上不再画这一行，2026-11-10 作者要求）
        forecast_label=("IFS ENS forecast, issued 22 Sep 12Z "
                        "(12–60 h ahead; 50 members × 2 samples)"),
        # m10 全窗领域 mean P 峰值：20250924T03 UTC (= 11:00 HKT 24 Sep，
        # frac=0.122 meanP=0.123)；m100 窗口 23T00Z..25T00Z 含最接近时刻
        t_peak="20250924T03",
        t_closest_hkt=datetime(2025, 9, 24, 6, 0),     # 120 km 以南
        sig10_hkt=(datetime(2025, 9, 24, 2, 40), datetime(2025, 9, 24, 13, 20)),
        panel_b_title="Territory peak: 03:00 UTC (11:00 HKT) 24 Sep 2025",
        closest_label="closest approach\n06:00 HKT",
        x_label="time (HKT, 2025)",
        # 48-h 跨度、49 个逐时刻（23 Sep 00Z .. 25 Sep 00Z 含两端）
        expo_window_title="48-h peak window (23 Sep 00Z – 25 Sep 00Z)",
        expo_ylim_km=(0, 20), expo_ylim_pct=(0, 90),
        out_suffix="_ragasa",
        # 代表段：crest_flag=0 且 event-max P≥0.9，crest_src 均有实测依据
        # （dtm/spot，非插补）。2026-10-03 两次调整：
        #  (1) 越浪改用 EurOtop 2018 + min 后，原代表段 1570（大澳护坡）
        #      事件最大 P 由 1.0 掉到 0.02（堤顶较高、护坡在 min 规则下不再
        #      越浪），换成 8032（香港仔海洋公园对开护坡，P=1.0，
        #      q_max=233 L/s/m，距 HKO「Aberdeen」确认浸水点 0.54 km）；
        #  (2) 为让 panel e 展示四个**互不相同**的潮位站，且优先选模式水位
        #      报得准的站，把 8538（将军澳南，与 6569 同用 Quarry Bay）换成
        #      6760（长洲东湾护坡，P=1.0，crest 6.83 m spot，最近站 Shek Pik）。
        #      各站模式峰值(−实测峰)：Quarry Bay −0.21、Shek Pik −0.30、
        #      Tai Miu Wan −0.34、Tsim Bei Tsui −0.58 m（Tai Po Kau −0.64 更差，
        #      故未纳入）；四站里三站是最准的三个。
        # 2026-11 核实：四个段号的实际几何位置与旧表头地名不符（旧名沿用错误），
        # 已按 hk_coastline_segments.csv 的 mid 坐标 + OSM 反查改正表头：
        #   6760 = 石鼓洲 Shek Kwu Chau（22.189, 113.989；非长洲，相距 4.5 km）
        #   8657 = 屯门码头 Tuen Mun Pier（22.375, 113.970；非嘉和里，相距 2.7 km）
        #   8032 = 香港仔 Aberdeen（22.238, 114.163；与旧名一致）
        #   6569 = 柴湾东工业区嘉业街（22.271, 114.245；非将军澳工业邨，相距 2.2 km）
        # 其中 8032 距 HKO「Aberdeen」浸水点 0.4 km、6569 正落在柴湾嘉业街浸水点，
        # 另两段不对应水浸点（不必强求，见 discussion）。
        panel_c_segs=[
            dict(seg_id=6760, name="Shek Kwu Chau",
                 kind="sloping revetment", e_gauge="Shek Pik"),
            # panel e 原用 Tsim Bei Tsui（后海湾），但模式在该站几乎抹平潮汐
            # （48 h 内模式仅 0.6–1.1 m 摆动，实测 −0.57…+2.39 m，极值点
            # RMSE 0.97）。改为「吐露港岸段 + 大埔滘站」：模型在该站能跟上
            # 潮汐相位（极值点 RMSE 0.52），且段（大埔工業邨海旁，22.452/114.189）
            # 与站（大埔滘 22.443/114.184）仅相距 1.1 km，地理自洽。
            # 选取条件同其他段：crest_flag=0、event-max P=1.0、crest 来自 DTM。
            dict(seg_id=8196, name="Tai Po Ind. Est.",
                 kind="sloping revetment", e_gauge="Tai Po Kau"),
            dict(seg_id=8032, name="Aberdeen",
                 kind="sloping revetment", e_gauge="Quarry Bay"),
            dict(seg_id=6569, name="Chai Wan East Ind. Area",
                 kind="vertical seawall", e_gauge="Tai Miu Wan"),
        ],
    ),
    # ---- Figure 6-US：Hurricane Helene 2024-09-26/27，Tampa Bay（都市岸线，
    # Storm Surge Warning 24/2100Z..27/1500Z，登陆 27/03:10Z = 26 23:10 EDT，
    # CO-OPS 四站峰值 2.15–2.57 m MSL）。同版式同配色，只换数据：
    # 段表/summary/q npz/潮位/陆地底图全部由
    # scripts/figures/figure6_us/build_case_helene.py 产出（ragasa schema）。
    # out-of-sample（模型训练 ≤2020，无 hindcast GT）：SWL = 集合中位数 elev
    # 锚定 CO-OPS 站（与 Isaias 管线同一做法），crest 全部来自 CUDEM 激光雷达
    # （crest_flag=0，无插补段）。类型按堤顶高程分（低洼 <1 m / 沙丘-滩肩
    # 1–2.5 m / 高脊 >2.5 m，均 MSL 基准）。 forecast_* 键不存在 → d/e 不叠
    # 预报线（画布矮 3 mm，无叠加版式）。
    "helene": dict(
        seg_csv="data/us_coastline/helene_segments.csv",
        node_subset_npz="data/us_coastline/node_subset_helene.npz",
        summary_csv="data/us_coastline/helene/overtopping/helene_segment_summary.csv",
        npz_m100="data/us_coastline/helene/overtopping/helene_q_m100.npz",
        npz_m10="data/us_coastline/helene/overtopping/helene_q_m10.npz",
        hko_json="data/us_coastline/helene/overtopping/helene_impact_records.json",
        coords="data/global_hindcast/global_coordinates.npy",
        samples_dir="output/results/global/case_helene2024_m100/samples_subset",
        cache_npz="_cache_node_ensemble_m100_helene.npz",
        out_dir="output/plot/figure6_us",
        # 卫星底图：Esri World Imagery（fetch_satellite_tampa.py 生成，
        # 图注需 "Source: Esri, Maxar, Earthstar Geographics"）
        ortho_npz="data/us_coastline/orthophoto/tampa_satellite_wgs84.npz",
        coverage_npz="data/us_coastline/orthophoto/"
                     "tampa_satellite_coverage.npz",
        gauge_obs_dir="data/us_coastline",
        cd_to_msl=0.0,          # 潮位 CSV 直接存 MSL 值（字段名沿用 level_mcd）
        tz_offset=-4,           # EDT
        region=[-83.00, -82.20, 27.42, 27.996],   # 纵横比 0.72 与 HK 版一致
        hk_cleanup=False,       # 跳过城门河误标剔除（HK 专属清洗）
        t_peak="20240927T03",   # m100 领域平均 P 峰值（= 登陆时刻，实测潮峰 03:00–04:42 UTC）
        t_closest_hkt=datetime(2024, 9, 26, 23, 10),   # landfall (EDT)
        # NHC TCR Table: Storm Surge Warning 24/2100Z issued (含 Tampa Bay)，
        # 27/1500Z all discontinued
        sig10_hkt=(datetime(2024, 9, 24, 17, 0), datetime(2024, 9, 27, 11, 0)),
        sig10_label="Storm surge\nwarning",
        sig10_legend="Storm surge warning",
        closest_label="landfall\n23:10 EDT",
        closest_legend="landfall",
        panel_b_title="Tampa Bay peak: 03:00 UTC 27 Sep (23:00 EDT 26 Sep) 2024",
        x_label="time (EDT, 2024)",
        expo_window_title="48-h peak window (26 Sep 12Z – 28 Sep 12Z)",
        expo_ylim_km=(0, 140), expo_ylim_pct=(0, 100),
        expo_legend_y=112.8,
        expo_legend_ncol=3,
        expo_crest_split=False,
        q_ylim_max=2000,
        # 单发布时次预报（panel d/e 橙/红叠加线）：GEFS 2024-09-26 00Z（登陆前
        # 27 h），31 成员，3-h 有效时刻 26T12..28T12（lead 12-60 h）。强迫树
        # data/ifs/ens_members_helene（fetch_gefs.py），样本树 = 全场 run 的
        # 116 节点子集镜像（remote 采样后 extract_subset_remote.py 回传）
        forecast_init="20240926T00",
        forecast_samples_dir="output/results/global/case_helene2024_fc_init20240926T00/test/samples_subset",
        forecast_npz=("data/us_coastline/helene/overtopping/"
                      "helene_q_forecast_init20240926T00.npz"),
        forecast_members=31,
        forecast_label=("GEFS forecast, issued 26 Sep 00Z "
                        "(12-60 h ahead; 31 members)"),
        out_suffix="_helene",
        flood_legend="flooding report (news/NHC)",
        gauge_legend="wave buoy (NDBC)",
        # panel e 改对比 Hs：4 个最近 NDBC 浮标（fetch_buoys_helene.py）
        e_panel_var="hs",
        e_ylabel="$H_s$ (m)",
        e_med_legend="median $H_s$",
        e_band_legend="$H_s$ P10–P90",
        obs_legend="buoy obs.",
        e_fc_wl_legend="forecast median $H_s$",
        panel_a_title="Shoreline type",
        panel_a_types=["high_ridge", "dune_berm", "low_lying"],
        expo_ctypes=["low_lying", "dune_berm", "high_ridge"],
        elev_ylim=(0, 9),
        places=[
            (-82.640, 27.780, "St.\nPetersburg", "center", None),
            (-82.465, 27.947, "Tampa", "center", None),
            (-82.816, 27.858, "Gulf of\nMexico", "center", None),
            (-82.846, 27.943, "Clearwater", "center", None),
        ],
        # 代表段（build_case_helene.py --waves 排名后人工选定）：e 面板改配
        # 4 个最近的有 Hs 记录的 NDBC 浮标（fetch_buoys_helene.py），类型
        # 覆盖三个堤顶等级
        panel_c_segs=[
            dict(seg_id=750, name="Clearwater Beach",
                 kind="low-lying (<1 m)", e_gauge="NDBC 42098 (Egmont)"),
            dict(seg_id=417, name="Downtown St. Petersburg",
                 kind="dune/berm (1-2.5 m)", e_gauge="NDBC 42036 (W Tampa)"),
            dict(seg_id=217, name="Port Manatee",
                 kind="high ridge (>2.5 m)",
                 e_gauge="NDBC 42097 (Pulley Ridge)"),
            dict(seg_id=250, name="East Bay",
                 kind="dune/berm (1-2.5 m)", e_gauge="NDBC 42095 (Satan Shoal)"),
        ],
    ),
}
_SPEC = CASE_SPECS[CASE]

# 混合堤顶产品（官方 DTM + flag1/2 段 SpotHeight 替换）：主图唯一数据源
SEG_CSV = _SPEC.get("seg_csv", "data/hk_coastline/hk_coastline_segments.csv")
# Release override: the q products can be redirected (e.g. to the output of
# compute_overtopping_saola.py rerun on freshly inferred samples).
SUMMARY_CSV = os.environ.get("WF_FIG6_SUMMARY_CSV", _SPEC["summary_csv"])
NPZ_M100 = os.environ.get("WF_FIG6_Q_NPZ", _SPEC["npz_m100"])
NPZ_M10 = os.environ.get("WF_FIG6_Q_NPZ_M10", _SPEC["npz_m10"])
# 纯官方 DTM 版（无 SpotHeight 替换）：附录敏感性对照（HK 专属）
SUMMARY_CSV_DTM = "data/hk_coastline/overtopping/saola_segment_summary_official.csv"
NPZ_M100_DTM = "data/hk_coastline/overtopping/saola_q_m100_official.npz"
# Esri 中国（香港）转发的同一 LandsD DTM（重采样）：附录交叉检验用
SUMMARY_CSV_ESRI = "data/hk_coastline/overtopping/saola_segment_summary_landsd.csv"
NPZ_M100_ESRI = "data/hk_coastline/overtopping/saola_q_m100_landsd.npz"
# Terrarium (SRTM) 旧产品：仅附录对比用
SUMMARY_CSV_TERR = "data/hk_coastline/overtopping/saola_segment_summary.csv"
NPZ_M100_TERR = "data/hk_coastline/overtopping/saola_q_m100.npz"
HKO_JSON = _SPEC["hko_json"]
DIST_CSV = "data/hk_coastline/landsd_official/osm_vs_landsd_distances.csv"
COORDS = _SPEC.get("coords", "data/hindcast/hk_coordinates.npy")
# Release override: the run root and the 100-member sample dir are
# redirectable (e.g. to freshly inferred samples).
RUN = os.environ.get(
    "WF_HK_RUN",
    _SPEC.get("run", ("output/results/hk_cleaned/all_var/"
                      "f8_blr5e-5_amp_stable_tod/"
                      "size_sweep/base_v1_20260821_025917")))
SAMPLES_M100 = os.environ.get(
    "WF_FIG6_SAMPLES_M100",
    os.environ.get(
        "FIG6_SAMP_M100",
        _SPEC.get("samples_dir")
        or os.path.join(RUN, _SPEC["samples_m100"], "test/samples")))
# FIG6_SUBSET=1：SAMPLES_M100 指向远端镜像回来的"节点子集"树（sample 已是
# (n_subset, 4)，节点顺序 == 排序后的 node_subset npz），加载时不再按节点号索引
SUBSET_MODE = os.environ.get("FIG6_SUBSET", "0") == "1"

# ---- 预报（panel d/e 叠加）：单发布时次 IFS ENS 逐成员强迫 ----
FORECAST_INIT = _SPEC.get("forecast_init")
# forecast_samples = RUN 下的相对目录名（ragasa 惯例，自动拼 test/samples）；
# forecast_samples_dir = verbatim 路径（helene：远端镜像回来的子集树）
FORECAST_SAMPLES = (_SPEC.get("forecast_samples_dir")
                    or (os.path.join(RUN, _SPEC["forecast_samples"],
                                     "test/samples")
                        if _SPEC.get("forecast_samples") else None))
FORECAST_NPZ = _SPEC.get("forecast_npz")
FORECAST_LABEL = _SPEC.get("forecast_label", "")
# 预报三色按 panel 分工（2026-11-10 作者反馈：紫色与蓝/红区分不开）：
#   panel d 的 q 中位数用橙、P(q>1) 用深橙点线；panel e 的水位用红
C_FC_Q = "#E66101"        # forecast median q (panel d, 实线+圆点)
C_FC_P = "#E66101"        # forecast P(q>1) (panel d, 点线, 右轴)
# 与同 panel 的预报 q 实线同色（2026-11-10 作者要求），靠线型/纵横轴区分；
# panel e 的实测另用 #E69F00
C_FC_WL = "#D62728"       # forecast median water level (panel e, 实线+圆点)

OUT_DIR = _SPEC.get("out_dir", "output/plot/figure6")
CACHE_NPZ = os.path.join(OUT_DIR, _SPEC["cache_npz"])

# 地图范围 [min_lon, max_lon, min_lat, max_lat]：包全部分段 + 少量边距
REGION = _SPEC.get("region", [113.80, 114.48, 22.11, 22.60])
MAP_ASPECT = ((REGION[3] - REGION[2]) / (REGION[1] - REGION[0]))  # h/w

# 港府官方正射影像底图缓存（LandsD GeoData Store 影像瓦片，港内为
# TDOP2025 级真射影像；由 scripts/fetch_landsd_orthophoto.py 生成）。
# helene 案例用同布局的灰陆地底图（build_case_helene.py 生成，海洋 alpha=0）。
ORTHO_NPZ = _SPEC.get("ortho_npz",
                      "data/hk_coastline/orthophoto/hk_orthophoto_wgs84.npz")
# 官方 LandsD 5 m DTM（EPSG:2326；海面=0、境外=NODATA），用于陆地掩膜
ASC_DTM = "data/hk_coastline/dtm_landsd_official/Whole_HK_DTM_5m.asc"
_LANDMASK_CACHE = "data/hk_coastline/orthophoto/_landmask_cache.npz"
COVERAGE_NPZ = _SPEC.get("coverage_npz",
                         "data/hk_coastline/orthophoto/"
                         "hk_official_coverage.npz")
# 预报未配齐时的 panel d/e 事件锚点文案（helene 案例：NHC 风暴潮预警 + 登陆）
SIG10_LABEL = _SPEC.get("sig10_label", "Signal No. 10")
SIG10_LEGEND = _SPEC.get("sig10_legend", _SPEC.get("sig10_label",
                                                   "Signal No. 10"))
CLOSEST_LEGEND = _SPEC.get("closest_legend", "closest approach")
FLOOD_LEGEND = _SPEC.get("flood_legend", "flooding report (HKO)")
GAUGE_LEGEND = _SPEC.get("gauge_legend", "tide gauge (HKO)")
PANEL_A_TITLE = _SPEC.get("panel_a_title", "Coastline defence types")
PANEL_A_TYPES = _SPEC.get("panel_a_types",
                          ["seawall_vertical", "revetment_slope",
                           "natural_rocky", "natural_sandy", "lowlying_open"])
PLACES = _SPEC.get("places", [
    (114.185, 22.256, "Hong Kong\nIsland", "center", None),
    (114.070, 22.296, "Victoria\nHarbour", "center", (114.168, 22.290)),
    (113.965, 22.244, "Lantau Island", "center", None),
    (113.826, 22.262, "Tai O", "center", (113.858, 22.256)),
    (114.235, 22.465, "Tolo Harbour", "center", None),
    (114.028, 22.185, "Cheung Chau", "center", None),
])
EXPO_CTYPES = _SPEC.get("expo_ctypes", ["seawall_vertical", "revetment_slope"])
# panel c 图例的纵位/列数：helene 的两行类目刻度标签比单行高 ~2.3 mm，
# 图例顺移并改单行 5 项（ncol=5），避免上碰刻度、下碰 panel d 标题
EXPO_LEGEND_Y = _SPEC.get("expo_legend_y", 111.5)
EXPO_LEGEND_NCOL = _SPEC.get("expo_legend_ncol", 3)
# helene：crest 全部来自 CUDEM 激光雷达（无插补/无低置信段），图例去掉
# crest-confident / imputed crest 两项（图上不存在对应图元）
EXPO_CREST_SPLIT = _SPEC.get("expo_crest_split", True)
ELEV_YLIM = _SPEC.get("elev_ylim", (-0.4, 2.6))
Q_YLIM_MAX = _SPEC.get("q_ylim_max", None)
GAUGE_OBS_DIR = _SPEC.get("gauge_obs_dir", "data/hk_coastline")
TZ_OFFSET = _SPEC.get("tz_offset", 8)
# panel e 对比量：HK=潮位（elev），helene=NDBC 浮标波高（hs）
E_VAR = _SPEC.get("e_panel_var", "elev")
E_YLABEL = _SPEC.get("e_ylabel", "water level (m, MSL)")
E_MED_LEGEND = _SPEC.get("e_med_legend", "median water level")
E_BAND_LEGEND = _SPEC.get("e_band_legend", "WL P10–P90")
OBS_LEGEND = _SPEC.get("obs_legend", "gauge obs.")
E_FC_WL_LEGEND = _SPEC.get("e_fc_wl_legend", "forecast median water level")
# 2019 DTM 之后的新填海补丁多边形（WGS84；影像/官方 2026 岸线已含，
# 2019 DTM 仍是海）：3RS 三跑岛 / 港珠澳大桥口岸岛 / 东涌东扩展区。
# （石鼓洲 IWMF 在瓦片影像里只是服务拼缝的纯色块、无真实地物，不打补丁。）
EXTRA_LAND_POLYS = [
    [(113.875, 22.312), (113.875, 22.327), (113.908, 22.330),
     (113.913, 22.335), (113.928, 22.335), (113.938, 22.329),
     (113.946, 22.320), (113.946, 22.306), (113.936, 22.302),
     (113.920, 22.306), (113.905, 22.309)],                       # 3RS
    [(113.946, 22.305), (113.946, 22.323), (113.956, 22.326),
     (113.963, 22.322), (113.963, 22.308), (113.956, 22.303)],   # HKBCF 岛
    [(113.952, 22.291), (113.953, 22.306), (113.964, 22.3065),
     (113.969, 22.299), (113.967, 22.290)],                       # 东涌东
]

T_PEAK = _SPEC["t_peak"]            # 领域平均 P 峰值（hybrid crest，m100 窗口）
T_CLOSEST_HKT = _SPEC["t_closest_hkt"]
SIG10_HKT = _SPEC["sig10_hkt"]
PANEL_B_TITLE = _SPEC["panel_b_title"]
CLOSEST_LABEL = _SPEC["closest_label"]
X_LABEL = _SPEC["x_label"]
EXPO_WINDOW_TITLE = _SPEC["expo_window_title"]
EXPO_YLIM_KM = _SPEC["expo_ylim_km"]
EXPO_YLIM_PCT = _SPEC["expo_ylim_pct"]
OUT_SUFFIX = _SPEC["out_suffix"]
TAIPOKAU_MAX_MCD = 3.32          # 23:19 HKT 1 Sep
CD_TO_MSL = _SPEC.get("cd_to_msl", 1.38)   # HK Chart Datum ≈ MSL 以下 1.38 m
TAIPOKAU_MAX_MSL = TAIPOKAU_MAX_MCD - CD_TO_MSL   # ≈ 1.94 m

COAST_COLORS = {
    "seawall_vertical": "#D62728",
    "revetment_slope": "#E69F00",
    "natural_rocky": "#6E6E6E",
    "natural_sandy": "#D8B73A",
    "lowlying_open": "#56B4E9",
    # helene（US）案例的堤顶高程分类：颜色沿用全稿调色板
    "low_lying": "#56B4E9",
    "dune_berm": "#D8B73A",
    "high_ridge": "#6E6E6E",
}
COAST_LABELS = {
    "seawall_vertical": "vertical seawall",
    "revetment_slope": "sloping revetment",
    "natural_rocky": "natural rocky",
    "natural_sandy": "natural sandy",
    "lowlying_open": "low-lying open",
    "low_lying": "low-lying\n(<1 m)",
    "dune_berm": "dune/berm\n(1-2.5 m)",
    "high_ridge": "high ridge\n(>2.5 m)",
}

# 代表段（现 panel d）：Saola 版逐一核实 crest_flag=0、event-max P ≥ 0.9
# （6758/8332 在 LandsD 下变成 flag=1 无堤顶数据、1877 maxP 降至 0.38，
# 8616 在官方 ASC 堤顶（3.77 m，比 Esri 版高 2.2 m）下 maxP 降至 0，
# 故更换；6472 与 8616 同片区同为直立墙）。Ragasa 版待 m10 结果选出。
PANEL_C_SEGS = _SPEC["panel_c_segs"]


def ax_mm(fig, x, y_top, w, h, projection=None, canvas=(180.0, 200.0)):
    """毫米绝对定位：x 从画布左缘、y_top 从画布顶缘量。"""
    cw, ch = canvas
    kw = dict(projection=projection) if projection is not None else {}
    return fig.add_axes([x / cw, 1 - (y_top + h) / ch, w / cw, h / ch], **kw)


def letter(fig, x, y_top, s, canvas=(180.0, 200.0)):
    cw, ch = canvas
    # 9 pt 粗体小写（作者指定：面板字母保持 9 pt，勿按 8 pt 规范改小）
    fig.text(x / cw, 1 - y_top / ch, s, fontsize=9, fontweight="bold",
             ha="left", va="top")


# ---- 数据加载 ----
def load_segments(drop_river_mislabel=True):
    """岸段表。默认剔除城门河误标段：OSM way 750031066 把
    natural=coastline 画进了沙田城门河（官方 LandsD 岸线交叉检验确认，
    见 data/hk_coastline/landsd_official/VALIDATION_REPORT.md）——
     midpoint 距官方岸线 >250 m 且落在沙田内陆 bbox 的段不是海岸线。
    这些段均非工程段（不进越浪计算），只影响 panel a 展示与段数。"""
    segs = pd.read_csv(SEG_CSV)
    ovr = _SPEC.get("type_overrides")
    if ovr and os.path.exists(ovr):
        # 影像核对过的类型修正（panel a 展示与段数统计同步用修正后类型）
        d = pd.read_csv(ovr).set_index("seg_id")
        segs = segs.set_index("seg_id")
        segs.loc[d.index, "coast_type"] = d["new_type"]
        segs = segs.reset_index()
    if (drop_river_mislabel and _SPEC.get("hk_cleanup", True)
            and os.path.exists(DIST_CSV)):
        d = pd.read_csv(DIST_CSV)
        bad = d[(d.dist_to_landsd_m > 250)
                & d.mid_lat.between(22.35, 22.42)
                & d.mid_lon.between(114.17, 114.22)]
        segs = segs[~segs.seg_id.isin(bad.seg_id)].reset_index(drop=True)
    excl = _SPEC.get("exclude_segs")
    if excl and os.path.exists(excl):
        # 几何落线问题段（段线跨湾口/桥面/水面）：panel a 也不再展示为岸线
        ex = pd.read_csv(excl)
        segs = segs[~segs.seg_id.isin(ex.seg_id)].reset_index(drop=True)
    return segs


def load_summary():
    return pd.read_csv(SUMMARY_CSV).set_index("seg_id")


def load_overtopping(which="m100"):
    path = NPZ_M100 if which == "m100" else NPZ_M10
    return np.load(path, allow_pickle=True)


def forecast_available(members=None):
    """预报产品是否齐备（只 stat，不加载数值）。

    saola 规格没有 forecast_* 键 -> 恒 False，旧版图完全不受影响；ragasa 需要
    q 产品 npz + 每个预报时刻的 members 个样本文件全部存在才返回 True。"""
    if not (FORECAST_NPZ and FORECAST_SAMPLES):
        return False
    if not (os.path.exists(FORECAST_NPZ) and os.path.isdir(FORECAST_SAMPLES)):
        return False
    members = int(members or _SPEC.get("forecast_members", 10))
    with np.load(FORECAST_NPZ, allow_pickle=True) as z:
        ts = [str(t) for t in z["timestamps"]]
    missing = sum(1 for t in ts for k in range(members)
                  if not os.path.exists(os.path.join(
                      FORECAST_SAMPLES, f"sample_{t}_ens{k:03d}.npz")))
    if missing:
        print(f"forecast: {missing}/{len(ts) * members} sample files missing in "
              f"{FORECAST_SAMPLES}; forecast curves skipped")
    return missing == 0


def load_nodes():
    c = np.load(COORDS)
    return c[:, 0], c[:, 1]          # lats, lons


# ---- 官方正射影像底图 ----
_ORTHO_RGBA = None


def _land_sea_mask(h, w, extent):
    """陆地掩膜 = 官方 5 m DTM（海面=0、陆地为正——与 crest 提取同一
    数据源，岸线对位最准）∪ NaturalEarth 10m land（补深圳侧大陆；
    DTM 把深圳也存成 0，与海面同值，无法按 NODATA 区分）∪ 新填海补丁
    多边形（EXTRA_LAND_POLYS：2019 DTM 之后才完成的 3RS 等）。最后陆侧
    外扩 1 px。结果按 (h, w, extent) 磁盘缓存。返回 land (h, w) bool。"""
    from affine import Affine
    from scipy.ndimage import binary_dilation
    import cartopy.io.shapereader as shp
    from pyproj import Transformer
    from rasterio.features import rasterize
    lon0, lon1, lat0, lat1 = extent

    if os.path.exists(_LANDMASK_CACHE):
        c = np.load(_LANDMASK_CACHE)
        if (c["land"].shape == (h, w)
                and np.allclose(c["extent"], extent)):
            return c["land"]

    # --- HKSAR: 官方 DTM 5 m（ArcInfo ASCII, EPSG:2326）---
    with open(ASC_DTM) as fh:
        hdr = {}
        for _ in range(6):
            k, v = fh.readline().split()
            hdr[k.lower()] = float(v)
    dtm = np.loadtxt(ASC_DTM, skiprows=6, dtype=np.float32)
    ncols, nrows = int(hdr["ncols"]), int(hdr["nrows"])
    cs, xll, yll = hdr["cellsize"], hdr["xllcorner"], hdr["yllcorner"]
    ys, xs = np.mgrid[0:h, 0:w]
    lons = lon0 + xs / (w - 1) * (lon1 - lon0)
    lats = lat1 - ys / (h - 1) * (lat1 - lat0)
    fwd = Transformer.from_crs(4326, 2326, always_xy=True)
    ee, nn = fwd.transform(lons.ravel(), lats.ravel())
    # ASC 行 0 = 北缘：row = (yll + nrows*cs - N) / cs
    col = ((ee - xll) / cs).astype(int)
    row = ((yll + nrows * cs - nn) / cs).astype(int)
    ok = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)
    v = np.full(ee.shape, -9999.0, dtype=np.float32)
    v[ok] = dtm[row[ok], col[ok]]
    land = (v > 0.0).reshape(h, w)          # 海面=0，NODATA=-9999

    # --- NaturalEarth 10m land 并集（深圳侧大陆；DTM 把深圳也存成 0，
    # 与海面同值，故不能用 NODATA 区分境内外。NE 粗岸线最多把近岸
    # 真彩色海水像素多显示 ~100 m，视觉上无害）---
    tf = Affine((lon1 - lon0) / w, 0, lon0, 0, -(lat1 - lat0) / h, lat1)
    shapes = []
    reader = shp.Reader(shp.natural_earth(resolution="10m",
                                          category="physical", name="land"))
    for geom in reader.geometries():
        if (geom.bounds[2] < lon0 or geom.bounds[0] > lon1
                or geom.bounds[3] < lat0 or geom.bounds[1] > lat1):
            continue
        shapes.append((geom, 1))
    ne = rasterize(shapes, out_shape=(h, w), transform=tf, fill=0,
                   dtype="uint8").astype(bool)
    land |= ne

    # --- 2019 DTM 之后的新填海补丁（3RS / HKBCF / 东涌东 / IWMF）---
    from matplotlib.path import Path
    pts = np.column_stack([lons.ravel(), lats.ravel()])
    for poly in EXTRA_LAND_POLYS:
        land |= Path(np.asarray(poly)).contains_points(pts).reshape(h, w)

    land = binary_dilation(land, iterations=1)
    np.savez_compressed(_LANDMASK_CACHE, land=land,
                        extent=np.asarray(extent, dtype=float))
    return land


def load_orthophoto_rgba(alpha_land=0.85, show_sea=False, official_only=False):
    """官方正射影像 -> RGBA。show_sea=False（默认）：陆地保留影像
    （alpha_land），海面透明（露出轴面近白底色，保证 panel b 的 Hs
    散点对比度）；show_sea=True：整幅影像含海面一起显示。
    official_only=True：官方覆盖区（港界内含水域，瓦片服务 z16 探测，
    hk_official_coverage.npz）之外留白——界外瓦片是服务的异源回退
    底图，色调不一致。返回 (rgba (H,W,4) float32, extent)；缓存不存在
    时返回 None。"""
    global _ORTHO_RGBA
    if _ORTHO_RGBA is not None:
        return _ORTHO_RGBA
    if not os.path.exists(ORTHO_NPZ):
        return None
    d = np.load(ORTHO_NPZ, allow_pickle=True)
    rgb = d["rgb"].astype(np.float32) / 255.0
    extent = [float(v) for v in d["extent"]]
    if show_sea:
        alpha = np.full(rgb.shape[:2], alpha_land, dtype=np.float32)
    else:
        land = _land_sea_mask(*rgb.shape[:2], extent)
        alpha = np.where(land, alpha_land, 0.0).astype(np.float32)
    if official_only and os.path.exists(COVERAGE_NPZ):
        cov = np.load(COVERAGE_NPZ)["coverage"]
        if cov.shape == alpha.shape:
            alpha = np.where(cov, alpha, 0.0).astype(np.float32)
        else:
            print(f"warning: coverage mask shape {cov.shape} != image grid "
                  f"{alpha.shape}; coverage masking skipped (rebuild with "
                  f"fetch_landsd_orthophoto.py --coverage-mask --res ...)")
    rgba = np.dstack([rgb, alpha])
    _ORTHO_RGBA = (rgba, extent)
    return _ORTHO_RGBA


_INSET_CACHES = None


def _find_inset_cache(lon0, lon1, lat0, lat1):
    """轴范围完整落在某个 inset 高清缓存（hk_orthophoto_inset_*.npz，
    z16，~3.3 m/px）内时返回其 (rgb uint8, extent)，否则 None。"""
    global _INSET_CACHES
    if _INSET_CACHES is None:
        _INSET_CACHES = []
        for p in sorted(glob.glob(os.path.join(
                os.path.dirname(ORTHO_NPZ), "*_orthophoto_inset_*.npz"))):
            d = np.load(p)
            _INSET_CACHES.append((d["rgb"], [float(v) for v in d["extent"]]))
    for rgb, ext in _INSET_CACHES:
        if lon0 >= ext[0] and lon1 <= ext[1] and lat0 >= ext[2] \
                and lat1 <= ext[3]:
            return rgb, ext
    return None


def add_orthophoto(ax, alpha_land=0.85, zorder=0, show_sea=False,
                   official_only=False):
    """把官方正射影像底图加到地图轴（默认陆地影像 + 海面透明；
    show_sea=True 时海面也显示影像；official_only=True 时官方覆盖区
    之外留白）。轴范围落在 inset 高清缓存内时自动改用 z16 高清版
    （inset 深陷官方覆盖区内部，无需 coverage 掩膜）。无缓存时
    静默跳过（返回 False）。"""
    try:
        lon0, lon1, lat0, lat1 = ax.get_extent()
    except Exception:
        lon0 = lon1 = lat0 = lat1 = None
    if lon0 is not None:
        hit = _find_inset_cache(lon0, lon1, lat0, lat1)
    else:
        hit = None
    if hit is not None:
        rgb, ext = hit
        alpha = np.full(rgb.shape[:2], alpha_land, dtype=np.float32)
        rgba = np.dstack([rgb.astype(np.float32) / 255.0, alpha])
        ax.imshow(rgba, extent=ext, origin="upper", zorder=zorder,
                  transform=ccrs.PlateCarree(), interpolation="bilinear")
        return True
    out = load_orthophoto_rgba(alpha_land, show_sea=show_sea,
                               official_only=official_only)
    if out is None:
        return False
    rgba, extent = out
    ax.imshow(rgba, extent=extent, origin="upper", zorder=zorder,
              transform=ccrs.PlateCarree(), interpolation="bilinear")
    return True


def load_hko():
    with open(HKO_JSON) as fh:
        return json.load(fh)


def nearest_gauge(lat, lon):
    """距 (lat, lon) 最近的 HKO 潮位站（saola_hko_records.json
    tide_gauge_maxima），返回 dict(name, mcd, time_hkt, msl, dist_km)。
    msl = mcd - CD_TO_MSL。"""
    import math
    best = None
    for g in load_hko()["tide_gauge_maxima"]:
        d = math.hypot((lat - g["lat"]) * 111.0,
                       (lon - g["lon"]) * 111.0 * math.cos(math.radians(lat)))
        if best is None or d < best["dist_km"]:
            best = dict(name=g["station"], mcd=g["max_level_mcd"],
                        time_hkt=g["max_level_time_hkt"],
                        msl=g["max_level_mcd"] - CD_TO_MSL, dist_km=d)
    return best


def ts_to_dt(ts):
    """'%Y%m%dT%H' UTC 字符串数组 -> datetime 列表（UTC）。"""
    return [datetime.strptime(str(t), "%Y%m%dT%H") for t in ts]


def utc_to_hkt(dts):
    """UTC -> 案例本地时间（HK=HKT +8；helene=EDT -4，函数名沿用历史）。"""
    return [t + timedelta(hours=TZ_OFFSET) for t in dts]


def seg_lines(df):
    """DataFrame -> LineCollection 用的线段列表 [(lon0,lat0)-(lon1,lat1)]。"""
    return [((r.start_lon, r.start_lat), (r.end_lon, r.end_lat))
            for r in df.itertuples()]


def new_map_ax(fig, x, y_top, w, canvas, left_labels=True, bottom_labels=True,
               region=None, gridlines=True):
    """PlateCarree 地图轴（高度按 region 纵横比推出），自绘岸线底图。
    gridlines=False 用于 inset：无网格无刻度，只留细框。"""
    region = region or REGION
    h = w * (region[3] - region[2]) / (region[1] - region[0])
    ax = ax_mm(fig, x, y_top, w, h, projection=ccrs.PlateCarree(),
               canvas=canvas)
    ax.set_extent(region, crs=ccrs.PlateCarree())
    if gridlines:
        gl = ax.gridlines(draw_labels=True, alpha=0.3, lw=0.4, zorder=1)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = left_labels
        gl.bottom_labels = bottom_labels
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {"size": 6}
        gl.ylabel_style = {"size": 6, "rotation": 90, "va": "center"}
        import matplotlib.ticker as mticker
        gl.xlocator = mticker.MultipleLocator(0.2)
        # 纬度标签竖排后字长占纵向空间，0.1° 间距会互相叠字，放宽到 0.2°
        gl.ylocator = mticker.MultipleLocator(0.2)
    return ax, h


def add_segment_collection(ax, df, colors=None, lw=0.6, zorder=3,
                           color_field=None, cmap=None, vmin=0, vmax=1):
    """把分段画成 LineCollection；给 colors 数组则按值着色（返回 mappable）。"""
    lc = LineCollection(seg_lines(df), linewidths=lw, zorder=zorder,
                        transform=ccrs.PlateCarree(), capstyle="round")
    if colors is not None:
        lc.set_array(np.asarray(colors, dtype=float))
        lc.set_cmap(cmap)
        lc.set_clim(vmin, vmax)
    else:
        lc.set_color(color_field)
    ax.add_collection(lc)
    return lc


def load_hs_background(ts=T_PEAK, m=100):
    """峰值时刻集合平均 hs（N_nodes,），供 panel b 淡色背景。

    SUBSET_MODE：从节点子集镜像读出 106 个节点的集合平均，再按节点号散布回
    全长数组（窗口外 NaN，scatter 的 bad 色=透明，不画出界外点）。"""
    if SUBSET_MODE:
        sub = np.load(_SPEC["node_subset_npz"])["node_ids"]
        arr = np.stack([np.load(os.path.join(
            SAMPLES_M100, f"sample_{ts}_ens{k:03d}.npz"))["sample"][:, 0]
            for k in range(m)])
        hs_sub = np.nanmean(arr, axis=0).astype(np.float32)
        out = np.full(len(np.load(COORDS)), np.nan, np.float32)
        out[sub] = hs_sub
        return out
    arr = np.stack([np.load(os.path.join(
        SAMPLES_M100, f"sample_{ts}_ens{k:03d}.npz"))["sample"][:, 0]
        for k in range(m)])
    return np.nanmean(arr, axis=0).astype(np.float32)


def load_node_ensemble(node_ids, m=100):
    """m100 窗口 49 ts × m 成员的节点 4 变量集合 (T, m, N, 4)，带磁盘缓存。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    key = np.asarray(node_ids, dtype=int)
    if os.path.exists(CACHE_NPZ):
        c = np.load(CACHE_NPZ)
        if (c["node_ids"].shape == key.shape
                and (c["node_ids"] == key).all()
                and c["data"].shape[1] == m):
            return c["timestamps"], c["data"]
    z = load_overtopping("m100")
    timestamps = z["timestamps"]
    data = np.empty((len(timestamps), m, len(node_ids), 4), dtype=np.float32)
    if SUBSET_MODE:
        sub = np.load(_SPEC["node_subset_npz"])["node_ids"]
        pos_sub = {int(n): i for i, n in enumerate(sub)}
        sel = np.array([pos_sub[int(n)] for n in key])
    for ti, ts in enumerate(timestamps):
        for k in range(m):
            d = np.load(os.path.join(
                SAMPLES_M100, f"sample_{ts}_ens{k:03d}.npz"))
            data[ti, k] = (d["sample"][sel] if SUBSET_MODE
                           else d["sample"][key])
    np.savez_compressed(CACHE_NPZ, node_ids=key, timestamps=timestamps,
                        data=data)
    return timestamps, data
