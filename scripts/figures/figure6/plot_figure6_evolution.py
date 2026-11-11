#!/usr/bin/env python
"""Fig. 6 panel c draft：3 个代表工程岸段的越浪危险度时间演化（M=100，
2023-08-31T12 .. 09-02T12 UTC，x 轴 HKT）。

每列一段（seg 2143 大澳护面 / seg 6472 葵涌–昂船洲海堤 /
seg 26 鲤鱼门护面，均 crest_flag=0 且 event-max P ≥ 0.9）：
  上行：q 集合 P10–P90 带 + 中位线（log y，L/s/m），副轴 P(q>1) 虚线；
  下行：elev 集合带（P10–P90 + 中位，m，MSL 基准）为**距潮位站最近 3 个
        模型节点的 IDW 插值**（站在岸线陆侧、网格外，无法重心插值；与
        实测同点对比。q/P 上行仍用段代表节点），
        深灰实线 = 最近 HKO 潮位站的实测水位过程线（IOC/VLIZ 1-min 开放
        序列**逐小时平均**，与 figure_ed_validation 同口径，mCD 换算 MSL；
        仅 Quarry Bay / Shek Pik 有，见
        data/hk_coastline/tide_gauge_ragasa/README.md）；无开放序列的站
        （Saola 全部、Ragasa 的 Tsim Bei Tsui）退化为 HKO 公布极值的点
        划线参考（极值不写字标注）。
竖虚线 = 最接近时刻 21:00 HKT 1 Sep；灰带 = 十号信号时段
（20:15 HKT 1 Sep .. 03:40 HKT 2 Sep）。

预报曲线（仅当 C.forecast_available() 为真，现只有 Ragasa 版）：紫色实线+圆点 =
单发布时次 IFS ENS 逐成员强迫预报的 q / 水位集合中位数（3 h 时效点，不插值
加密），紫色点线 = 预报 P(q>1)（右轴）；发布时次与 lead 范围见 C.FORECAST_LABEL。

用法: python scripts/figures/figure6/plot_figure6_evolution.py
输出: output/plot/figure6/figure6_evolution_draft.{png,pdf}
"""
import os
import sys
import warnings
from datetime import datetime

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_figure6_common as C

OUT_BASE = os.path.join(C.OUT_DIR, "figure6_evolution_draft" + C.OUT_SUFFIX)

# 全稿 RdBu 约定（同 figure4/5）：蓝 = 模型集合量，红 = 危险度/超越概率，
# 灰 = 观测/中性；带用同族浅色实 tint（#92C5DE = figure5 的 C_GC_LT）
C_BAND = "#92C5DE"
C_MED = "#2166AC"
# panel d 统一配色（2026-11-10 作者要求）：分析场 = 蓝（q 实线 + P 深蓝虚线），
# 预报 = 橙/琥珀。P 用深一号的蓝，与 q 中位数同族但线型可区分，右侧副轴同色。
C_PROB = "#08519C"
C_REF = "#7F7F7F"
# panel e 实测（1-min 过程线 + HKO 逐日极值点）：橙黄（2026-11-10 作者要求），
# 与模型绿、预报红都能分开；白心圆点保证在灰白底上也清楚
C_OBS = "#E69F00"
C_WL_BAND = "#66C2A4"   # 水位集合带（BuGn 浅绿，与蓝/红/灰可区分）
C_WL_MED = "#006B4F"
C_FC_Q = C.C_FC_Q       # panel d 预报 q 中位数（橙）
C_FC_P = C.C_FC_P       # panel d 预报 P(q>1)（与预报 q 同橙，点线）
C_FC_WL = C.C_FC_WL     # panel e 预报水位（红）


def station_nodes(lat, lon):
    """距站位最近的 3 个模型节点及 IDW（反距离平方）权重。

    分析场（panel e 下行）与预报曲线共用这一套权重，保证两条线同点可比。
    潮位站多在岸线陆侧、落在模型网格外，故用 IDW 而不是重心插值。"""
    node_lats, node_lons = C.load_nodes()
    d = np.hypot(node_lats - lat, (node_lons - lon) * np.cos(np.radians(lat)))
    k = np.argsort(d)[:3]
    w = 1.0 / np.maximum(d[k], 1e-6) ** 2
    return k, w / w.sum()


def load_gauge_obs(station):
    """该潮位站的 1-min 实测水位（data/hk_coastline/tide_gauge_ragasa/，
    IOC/VLIZ 下载，HKO 供数；见该目录 README）。返回 (t_hkt, msl) 或
    None（该站无开放序列时，如 Tsim Bei Tsui / Tai Miu Wan）。
    helene 案例：CO-OPS 6-min 序列存 data/us_coastline/tide_gauge_helene/，
    level_mcd 列即 MSL 值（cd_to_msl=0），本地时区 EDT（tz_offset=-4）。"""
    slug = station.lower().replace(" ", "_")
    p = os.path.join(C.GAUGE_OBS_DIR, f"tide_gauge_{C.CASE}",
                     f"{slug}_1min_{C.CASE}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, parse_dates=["time_utc"])
    # 逐小时平均，与 figure_ed_validation 的潮位口径一致：模型输出是逐小时
    # 的，1-min 原始记录直接抽点会把 ~2 h 的海港静振和分钟级抖动全带进图里，
    # 与模型/集合带不可比（60 个样本的箱式平均去掉次小时方差）。
    df = (df.set_index("time_utc").level_mcd
          .resample("1h").mean().dropna())
    return df.index + pd.Timedelta(hours=C.TZ_OFFSET), (df - C.CD_TO_MSL).values


def load_gauge_extrema(station):
    """HKO "Daily Observed Sea Levels" 逐日潮汐极值（HKO CIS 开放数据，
    站位级真值锚点：HHM/LHM/HLW/LLW 高度+时刻）。无文件返回 None。"""
    slug = station.lower().replace(" ", "_")
    p = os.path.join(C.GAUGE_OBS_DIR, f"tide_gauge_{C.CASE}",
                     f"{slug}_daily_extrema_{C.CASE}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, parse_dates=["time_hkt"])
    return df.time_hkt, df.level_msl.values


def collect_evolution_data():
    """返回 {seg_id: dict(t_hkt, q_p10/50/90, p, elev_p10/50/90)}。

    注意：q/P 用段代表节点（驱动越浪计算的那个），但 elev 下行是与潮位站
    实测做同点对比。潮位站在岸线陆侧、落在模型网格外（重心插值不可用，
    只有 Tai Po Kau 在网格内），故 elev 取**距站最近 3 个模型节点的
    IDW（反距离平方）插值**——站离岸最近节点仅 0.01–0.37 km，IDW 与
    最近节点差异 <1 cm 量级，但形式上更严谨。"""
    z = C.load_overtopping("m100")
    ids = list(z["seg_ids"])
    t_hkt = C.utc_to_hkt(C.ts_to_dt(z["timestamps"]))
    segs = C.load_segments().set_index("seg_id")
    gauges = {x["station"]: (x["lat"], x["lon"])
              for x in C.load_hko()["tide_gauge_maxima"]}
    ginfo = {x["station"]: x for x in C.load_hko()["tide_gauge_maxima"]}

    specs = []
    for spec in C.PANEL_C_SEGS:
        seg = segs.loc[spec["seg_id"]]
        if spec.get("e_gauge"):      # panel e 指定站位（如 Tai Miu Wan）
            e = ginfo[spec["e_gauge"]]
            gauge = dict(name=e["station"], mcd=e["max_level_mcd"],
                         time_hkt=e["max_level_time_hkt"],
                         msl=e["max_level_mcd"] - C.CD_TO_MSL, dist_km=np.nan)
        else:
            gauge = C.nearest_gauge(seg.mid_lat, seg.mid_lon)
        knodes, kw = station_nodes(*gauges[gauge["name"]])
        specs.append((spec, seg, gauge, knodes, kw))
    # 合并所有需要的节点，一次加载集合
    all_nodes = sorted({int(n) for _, _, _, kn, _ in specs for n in kn})
    _, node_ens = C.load_node_ensemble(all_nodes)
    pos = {n: j for j, n in enumerate(all_nodes)}
    e_ch = 3 if C.E_VAR == "elev" else 0       # helene：panel e 取 Hs 通道
    out = {}
    for spec, seg, gauge, knodes, kw in specs:
        sid = spec["seg_id"]
        i = ids.index(sid)
        q = z["q_lpsm"][i]                     # (T, M)
        elev = sum(w * node_ens[:, :, pos[int(n)], e_ch]
                   for n, w in zip(knodes, kw))   # (T, M) IDW
        out[sid] = dict(
            t=t_hkt,
            q_p10=np.percentile(q, 10, axis=1),
            q_med=np.percentile(q, 50, axis=1),
            q_p90=np.percentile(q, 90, axis=1),
            p=z["p_gt1_lpsm"][i],
            elev_p10=np.percentile(elev, 10, axis=1),
            elev_med=np.percentile(elev, 50, axis=1),
            elev_p90=np.percentile(elev, 90, axis=1),
            name=spec["name"], kind=spec["kind"],
            gauge=gauge, gauge_nodes=[int(n) for n in knodes],
            gauge_w=list(np.round(kw, 3)),
            gauge_obs=load_gauge_obs(gauge["name"]),
            gauge_extrema=load_gauge_extrema(gauge["name"]),
        )
    return out


def collect_forecast_data(members=None):
    """单发布时次预报的 q / 水位集合统计（panel d/e 紫色曲线）。

    q/P 取自 compute_overtopping_forecast.py 的 npz（与 panel d 的分析场同段
    同序）；水位取预报样本在**同一组最近 3 节点**上的 IDW 插值，与分析场
    panel e 完全同点。预报数据不齐时返回 None（图退回现状）。"""
    if not C.forecast_available():
        return None
    z = np.load(C.FORECAST_NPZ, allow_pickle=True)
    ids = list(z["seg_ids"])
    ts_list = [str(t) for t in z["timestamps"]]
    t_hkt = C.utc_to_hkt(C.ts_to_dt(ts_list))
    leads = [int(v) for v in z["leads"]] if "leads" in z.files else []
    # q 用成员中位数（q_lpsm 是 (S, T, M) 的逐成员值，画进图会变成 10 条线）
    q_med_all = z["q_med_lpsm"]
    q_p10_all, q_p90_all = z["q_p10_lpsm"], z["q_p90_lpsm"]
    p_all = z["p_gt1_lpsm"]
    m = z["q_lpsm"].shape[2] if members is None else int(members)
    segs = C.load_segments().set_index("seg_id")
    ginfo = {x["station"]: x for x in C.load_hko()["tide_gauge_maxima"]}

    out = {}
    for spec in C.PANEL_C_SEGS:
        sid = spec["seg_id"]
        if sid not in ids:
            continue
        i = ids.index(sid)
        if spec.get("e_gauge"):
            e = ginfo[spec["e_gauge"]]
            lat, lon, gname = e["lat"], e["lon"], e["station"]
        else:
            seg = segs.loc[sid]
            lat, lon, gname = seg.mid_lat, seg.mid_lon, spec["name"]
        knodes, kw = station_nodes(lat, lon)
        # helene：panel e 对比量是 Hs（通道 0），采样文件走远端镜像回来的
        # 节点子集树（FIG6_SUBSET=1，与 collect 的分析场同一映射）
        e_ch = 3 if C.E_VAR == "elev" else 0
        elev = np.full((len(ts_list), m), np.nan, dtype=np.float64)
        if C.SUBSET_MODE:
            sub = np.load(C._SPEC["node_subset_npz"])["node_ids"]
            pos_sub = {int(n): i for i, n in enumerate(sub)}
            fsel = np.array([pos_sub[int(n)] for n in knodes])
        for ti, ts in enumerate(ts_list):
            for k in range(m):
                p = os.path.join(C.FORECAST_SAMPLES,
                                 f"sample_{ts}_ens{k:03d}.npz")
                if not os.path.exists(p):
                    continue
                with np.load(p) as d:
                    if C.SUBSET_MODE:
                        vals = d["sample"][fsel, e_ch]
                    else:
                        vals = d["sample"][knodes, e_ch]
                    elev[ti, k] = float(np.sum(kw * vals))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            elev_med = np.nanmedian(elev, axis=1)
            elev_p10 = np.nanpercentile(elev, 10, axis=1)
            elev_p90 = np.nanpercentile(elev, 90, axis=1)
        out[sid] = dict(
            t=t_hkt, leads=leads, q_med=q_med_all[i], p=p_all[i],
            q_p10=q_p10_all[i], q_p90=q_p90_all[i],
            elev_med=elev_med, elev_p10=elev_p10, elev_p90=elev_p90,
            gauge_name=gname,
            q_peak=float(np.nanmax(q_med_all[i])),
            p_peak=float(np.nanmax(p_all[i])))
    return dict(init=C.FORECAST_INIT, label=C.FORECAST_LABEL,
                t=t_hkt, leads=leads, segs=out)


def _mark_events(ax, t0, t1, label_sig10=False):
    ax.axvspan(C.SIG10_HKT[0], C.SIG10_HKT[1], color="#999999", alpha=0.15,
               lw=0, zorder=0)
    ax.axvline(C.T_CLOSEST_HKT, color="black", lw=0.7, ls=":", zorder=1)
    if label_sig10:
        ax.annotate(C.CLOSEST_LABEL, (C.T_CLOSEST_HKT, 1.0),
                    textcoords="offset points", xytext=(3, -2),
                    fontsize=6, ha="left", va="top", color="#333333")
        # 放灰带底部中央，避开顶部 closest approach 标注与 q 曲线
        mid = C.SIG10_HKT[0] + (C.SIG10_HKT[1] - C.SIG10_HKT[0]) / 2
        ax.annotate(C.SIG10_LABEL.replace("\n", " "), (mid, 0.0),
                    xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(0, 2),
                    fontsize=6, ha="center", va="bottom", color="#666666")
    ax.set_xlim(t0, t1)


def _fmt_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    ax.tick_params(axis="x", which="minor", length=1.5)


def draw_panel_c_col(ax_q, ax_e, d, t0, t1, col=0, ncols=3, fc=None):
    """单段两联轴：上 q 带 + P 曲线，下 elev 带；fc=该段的预报曲线（可选）。"""
    t = d["t"]
    # --- 上行：q（log）+ P(q>1) 副轴 ---
    ax_q.fill_between(t, np.clip(d["q_p10"], 1e-2, None),
                      np.clip(d["q_p90"], 1e-2, None),
                      color=C_BAND, alpha=0.55, lw=0, zorder=2)
    ax_q.plot(t, np.clip(d["q_med"], 1e-2, None), color=C_MED, lw=1.1,
              zorder=3, label="median $q$")
    ax_q.set_yscale("log")
    # 轴顶：Rc<0 的成员走堰流项 q 会破千（helene SWL 2–2.5 m 越过低矮堤顶，
    # q 峰值 ~6000，spec 给 2000）；有预报时 ragasa 也抬到 2000
    if C.Q_YLIM_MAX is not None:
        ax_q.set_ylim(0.02, C.Q_YLIM_MAX)
    else:
        ax_q.set_ylim(0.02, 2000 if fc is not None else 1000)
    ax_q.set_yticks([0.1, 1, 10, 100])
    ax_q.set_yticklabels(["0.1", "1", "10", "100"])
    axp = ax_q.twinx()
    axp.plot(t, d["p"], color=C_PROB, lw=0.9, ls="--", zorder=3,
             label="$P(q > 1)$")
    if fc is not None:
        # 单发布时次预报：3 h 时效点，不插值加密（见模块 docstring）
        ax_q.plot(fc["t"], np.clip(fc["q_med"], 1e-2, None), color=C_FC_Q,
                  lw=1.0, marker="o", ms=1.5, mec="white", mew=0.3,
                  zorder=4, label="forecast median $q$")
        axp.plot(fc["t"], fc["p"], color=C_FC_P, lw=0.9, ls=":", zorder=4,
                 label="forecast $P(q>1)$, right")
    axp.set_ylim(0, 1.05)
    axp.spines["right"].set_visible(True)
    axp.spines["right"].set_color(C_PROB)
    axp.tick_params(axis="y", colors=C_PROB)
    if col == ncols - 1:
        axp.set_ylabel("$P(q > 1$ L/s/m)", color=C_PROB, fontsize=7)
    else:
        axp.set_yticklabels([])
    if col == 0:
        ax_q.set_ylabel("$q$ (L/s/m)", labelpad=1.5)   # 贴近刻度标签（默认 4 pt）
    else:
        ax_q.set_yticklabels([])   # 右侧三列共用首列刻度，左轴只留刻度线不留数字
    _mark_events(ax_q, t0, t1)   # 图内文字注记（closest approach / Signal No. 10）已去掉，只留灰带+竖虚线
    ax_q.tick_params(labelbottom=False)
    ax_q.set_title(f"seg {d['sid']} · {d['name']}\n{d['kind']}",
                   fontsize=6.2, pad=2.5)
    # --- 下行（panel e）：站位处的模型 elev（IDW 插值）vs 实测 ---
    ax_e.fill_between(t, d["elev_p10"], d["elev_p90"], color=C_WL_BAND,
                      alpha=0.6, lw=0, zorder=2)
    ax_e.plot(t, d["elev_med"], color=C_WL_MED, lw=1.1, zorder=3)
    if fc is not None:
        ax_e.plot(fc["t"], fc["elev_med"], color=C_FC_WL, lw=1.0, marker="o",
                  ms=1.5, mec="white", mew=0.3, zorder=4,
                  label="forecast median water level")
    g = d.get("gauge")
    if g is not None:
        ax_e.set_title(f"{g['name']}", fontsize=6, pad=2)
        obs = d.get("gauge_obs")
        ext = d.get("gauge_extrema")
        if obs is not None:
            # 有实测过程线（IOC/VLIZ 开放序列 / helene: NDBC 浮标）：画过程线
            ax_e.plot(obs[0], obs[1], color=C_OBS, lw=0.8, zorder=3,
                      label=C.OBS_LEGEND)
        elif ext is not None:
            # 只有 HKO 逐日极值（Daily Observed Sea Levels）：真值锚点散点
            ax_e.plot(ext[0], ext[1], marker="o", ms=2.4, ls="",
                      color=C_OBS, mec="white", mew=0.4, zorder=4,
                      label="gauge obs. extremes")
        else:
            # 无观测数据：只给 HKO 公布极值的点划线参考
            ax_e.axhline(g["msl"], color=C_REF, lw=0.8, ls="-.", zorder=2)
    if col == 0:
        ax_e.set_ylabel(C.E_YLABEL)
    else:
        ax_e.set_yticklabels([])   # 四列共用同一 y 范围，只留首列刻度数字
    ax_e.set_ylim(*C.ELEV_YLIM)
    _mark_events(ax_e, t0, t1)
    _fmt_time_axis(ax_e)
    return axp


def panel_c_legend(fig, y_top, canvas, show_max_only=True, show_extremes=True):
    """panel c 共用图例（图级，一行居中）。

    show_max_only=False 时不列 "gauge max only" —— 该条目对应"站点只知 HKO
    公布极值"的灰色点划线，四个站都有观测（1-min 序列或逐日极值）时图上并不
    存在这条线（Ragasa 版即如此）。show_extremes=False 同理：helene 四站都有
    连续观测序列，图上没有极值散点，条目随之消失。"""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], color=C_MED, lw=1.1, label="median $q$"),
        Patch(fc=C_BAND, alpha=0.55, lw=0, label="$q$ P10–P90"),
        Line2D([0], [0], color=C_PROB, lw=0.9, ls="--",
               label="$P(q>1)$, right"),
        Line2D([0], [0], color=C_WL_MED, lw=1.1, label=C.E_MED_LEGEND),
        Patch(fc=C_WL_BAND, alpha=0.6, lw=0, label=C.E_BAND_LEGEND),
        Line2D([0], [0], color=C_OBS, lw=0.8, label=C.OBS_LEGEND),
        *([Line2D([0], [0], marker="o", ls="", ms=2.4, mfc=C_OBS, mec="white",
                  mew=0.4, label=f"{C.OBS_LEGEND} extremes")]
          if show_extremes else []),
        *([Line2D([0], [0], color=C_REF, lw=0.8, ls="-.",
                   label="gauge max only")] if show_max_only else []),
        Patch(fc="#999999", alpha=0.25, lw=0, label=C.SIG10_LEGEND),
        Line2D([0], [0], color="black", lw=0.7, ls=":",
               label=C.CLOSEST_LEGEND),
    ]
    cw, ch = canvas
    # 字号与其它图例统一（6 pt）；10 项仍排一行，靠收紧列距/handle 宽度保证
    # 不超出画布（5 pt 时整行 145 mm，6 pt 约 174 mm）
    return fig.legend(handles=handles, loc="upper center", ncol=10,
                      frameon=False, fontsize=6.0, handlelength=0.9,
                      columnspacing=0.30, handletextpad=0.3,
                      bbox_to_anchor=(0.5, 1 - y_top / ch))


def panel_c_forecast_legend(fig, y_top, canvas):
    """预报图例（第二行）：q 中位数 / P(q>1) / 水位中位数（发布时次写在图注里）。"""
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=C_FC_Q, lw=1.0, marker="o", ms=1.8,
               mec="white", mew=0.3, label="forecast median $q$"),
        Line2D([0], [0], color=C_FC_P, lw=0.9, ls=":",
               label="forecast $P(q>1)$, right"),
        Line2D([0], [0], color=C_FC_WL, lw=1.0, marker="o", ms=1.8,
               mec="white", mew=0.3, label=C.E_FC_WL_LEGEND),
    ]
    cw, ch = canvas
    return fig.legend(handles=handles, loc="upper center",
                      ncol=len(handles), frameon=False, fontsize=6.0,
                      handlelength=1.4, columnspacing=1.2,
                      bbox_to_anchor=(0.5, 1 - y_top / ch))


def unify_legend_fontsize(legends, size=6.0):
    """把图例文字统一到同一字号。

    TY.enforce_glyph_floor 会把含 mathtext 的标签提到 7.2 pt（为了 5 pt 的
    上下标下限），于是同一行里 "median $q$" 7.2 pt、"gauge obs." 6.0 pt，
    看起来大小不一。这两个图例的标签没有上下标，直接统一到 size 即可。"""
    for leg in legends:
        if leg is None:
            continue
        for t in leg.get_texts():
            t.set_fontsize(size)


def draw_panel_c(fig, data, x0, y_top, col_w, col_dx, row_h_q, row_h_e,
                 row_gap, canvas, fc=None):
    """主图装配用：3 列 × 2 行；fc=collect_forecast_data() 的结果（可选）。"""
    z = C.load_overtopping("m100")
    t_hkt = C.utc_to_hkt(C.ts_to_dt(z["timestamps"]))
    t0, t1 = t_hkt[0], t_hkt[-1]
    for j, spec in enumerate(C.PANEL_C_SEGS):
        d = data[spec["seg_id"]]
        d["sid"] = spec["seg_id"]
        ax_q = C.ax_mm(fig, x0 + j * col_dx, y_top, col_w, row_h_q,
                       canvas=canvas)
        ax_e = C.ax_mm(fig, x0 + j * col_dx, y_top + row_h_q + row_gap,
                       col_w, row_h_e, canvas=canvas)
        draw_panel_c_col(ax_q, ax_e, d, t0, t1, col=j,
                         ncols=len(C.PANEL_C_SEGS),
                         fc=None if fc is None
                         else fc["segs"].get(spec["seg_id"]))


def main():
    C.setup_style()
    os.makedirs(C.OUT_DIR, exist_ok=True)
    data = collect_evolution_data()
    fc = collect_forecast_data()

    canvas = (180.0, 92.0 + (2.5 if fc is not None else 0.0))
    n = len(C.PANEL_C_SEGS)
    # 4 段（Ragasa）沿用主图的列宽/列距，3 段（Saola）保持旧的宽松排布
    col_w, col_dx = (37.6, 40.3) if n >= 4 else (47.0, 54.0)
    show_max_only = any(
        data[s["seg_id"]].get("gauge_obs") is None
        and data[s["seg_id"]].get("gauge_extrema") is None
        for s in C.PANEL_C_SEGS)
    MM = 1 / 25.4
    fig = plt.figure(figsize=(canvas[0] * MM, canvas[1] * MM))
    draw_panel_c(fig, data, x0=10, y_top=10, col_w=col_w, col_dx=col_dx,
                 row_h_q=30, row_h_e=24, row_gap=13, canvas=canvas, fc=fc)
    leg_c = panel_c_legend(fig, 84, canvas, show_max_only=show_max_only)
    # 锚点按墨迹排布（图例盒自带 ~1 mm 内边距）：行 1 墨迹 86.1–88.1，
    # 行 2 墨迹 89.1–91.1 -> 行距 1.0 mm
    leg_fc = (panel_c_forecast_legend(fig, 87.0, canvas)
              if fc is not None else None)
    C.letter(fig, 3, 3.5, "c", canvas=canvas)

    fig.savefig(OUT_BASE + ".png", dpi=300)
    fig.savefig(OUT_BASE + ".pdf")
    print("saved:", OUT_BASE + ".png/.pdf")


if __name__ == "__main__":
    main()
