#!/usr/bin/env python
"""A5 分层 log-affine 成员校准（val 拟合 → test 应用）。

对集合样本本身做仿射校正（区别于 pit-calib 的分位重排，不坍缩成员）：
每变量、按 ensemble-mean hs 分层，在变换空间对成员做
    x'_i = μ − δ_L + ρ_L · (x_i − μ)
  δ_L：该层系统性偏差（μ − y 的均值）
  ρ_L：该层残差/成员离散度方差比（把 spread 拉到与误差一致）
变换空间：hs/tp 用 log(x+c)（异方差随量级缩放），elev 线性，dm 圆周。

协议：一切参数只在 val 拟合，test 只做应用（与 uq_posthoc_baselines 一致）。
输出：--out 下与输入同格式的 sample_{ts}_ens{NNN}.npz + calib_params.json，
直接用 eval_uq_ensemble.py 评估。

用法:
  python scripts/uq_affine_calib.py \
      --fit-samples <run>/curve_ep49/val/samples \
      --apply-samples <run>/curve_ep49/test/samples \
      --out <run>/uq_baselines/cfm_cal/test/samples [--members 10] [--workers 8]
"""
import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

VARS = ["hs", "tp", "dm", "elev"]
# 分层边界与 pit-calib/posthoc 保持一致（按 ensemble-mean hs, m）
LAYERS = np.array([0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, np.inf])
NL = len(LAYERS) - 1
# v2（修极端档）：顶部细分为 2.5/3/4/5m 以上各自独立拟合 δ/ρ。
# v1 的 2.5–∞ 档被 2.5–4m 数据主导，>4m 的 δ 被低估（cal 后 >4m CRPS 反而变差）。
LAYERS_V2 = np.array([0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5,
                      3.0, 4.0, 5.0, np.inf])
# 变换空间常数：log(x + C)
C_LOG = {"hs": 0.01, "tp": 0.1, "dm": 0.0, "elev": 0.0}
RHO_CLIP = (0.2, 5.0)
DELTA_CLIP = 1.0
MIN_LAYER_N = 500      # 该层 node 数不足则回退到全局参数
GT_DIR = "./data/hindcast/processed_npz"


def list_ts(sample_dir):
    pat = re.compile(r"sample_(\d{8}T\d{2})_ens\d+\.npz$")
    return sorted({m.group(1) for f in os.listdir(sample_dir)
                   if (m := pat.match(f))})


def fwd(x, v):
    """变换空间（dm 圆周特殊处理在调用处）。"""
    return np.log(x + C_LOG[v]) if C_LOG[v] > 0 else x


def inv(t, v):
    return np.exp(t) - C_LOG[v] if C_LOG[v] > 0 else t


def load_ens_gt(item):
    ts, sample_dir, m = item
    ens = np.stack([np.load(os.path.join(sample_dir, f"sample_{ts}_ens{k:03d}.npz"))
                    ["sample"].squeeze() for k in range(m)], 0).astype(np.float32)
    g = np.load(os.path.join(GT_DIR, f"wave_{ts}.npz"))
    gt = np.stack([g[v].astype(np.float32) for v in VARS], 1)
    return ts, ens, gt


# ---------------------------------------------------------------- fit
# 每 (var, layer) 流式累积：n / Σ(μ−y) / Σ(μ−y)² / Σ(x−μ)²（dm 另存 sin/cos）


def fit_one(item):
    """返回每变量的 per-layer 累积量。非 dm: acc=(n, Σd, Σd²)；dm: acc=(n, Σsin, Σcos, Σdy²)，
    r2 = Σ(成员绕均值的残差²)（dm 为圆周绕均值）。"""
    _, ens, gt = load_ens_gt(item)
    mu = ens.mean(0)                       # (N,4)
    lay = np.clip(np.searchsorted(LAYERS, mu[:, 0], side="right") - 1, 0, NL - 1)
    out = []
    for vi, v in enumerate(VARS):
        acc = np.zeros((NL, 4))
        r2 = np.zeros(NL)
        if v != "dm":
            z = fwd(ens[:, :, vi], v)
            zm = z.mean(0)
            d = zm - fwd(gt[:, vi], v)
            r = z - zm[None]
            for L in np.unique(lay):
                m_ = lay == L
                acc[L, 0] = m_.sum()
                acc[L, 1] = d[m_].sum()
                acc[L, 2] = (d[m_] ** 2).sum()
                r2[L] = (r[:, m_] ** 2).sum()
        else:  # dm 圆周（度）
            rad = np.deg2rad(ens[:, :, vi])
            radm = np.arctan2(np.sin(rad).mean(0), np.cos(rad).mean(0))  # 圆周均值,避免跨 0/360 成员平均出 ~180° 的假中心
            dr = np.arctan2(np.sin(rad - radm[None]), np.cos(rad - radm[None]))
            gy = np.deg2rad(gt[:, vi])
            dy = np.arctan2(np.sin(gy - radm), np.cos(gy - radm))
            for L in np.unique(lay):
                m_ = lay == L
                acc[L, 0] = m_.sum()
                acc[L, 1] = np.sin(dy[m_]).sum()
                acc[L, 2] = np.cos(dy[m_]).sum()
                acc[L, 3] = (dy[m_] ** 2).sum()
                r2[L] = (dr[:, m_] ** 2).sum()
        out.append((acc, r2))
    return out


def _delta_rho(acc, r2, n, m, v):
    """从累积量出 (delta, rho)。acc 列含义见 fit_one。"""
    if v != "dm":
        delta = np.clip(acc[1] / n, -DELTA_CLIP, DELTA_CLIP)
        var_err = acc[2] / n - (acc[1] / n) ** 2
        var_mem = r2 / (n * m)
    else:
        delta = np.rad2deg(np.arctan2(acc[1], acc[2]))  # 圆周均值偏移
        var_err = acc[3] / n
        var_mem = r2 / (n * m)
    var_err = max(var_err, 1e-12)
    var_mem = max(var_mem, 1e-12)
    return float(delta), float(np.clip(np.sqrt(var_err / var_mem), *RHO_CLIP))


def fit_params(ts_list, sample_dir, m, workers, log):
    S = {v: (np.zeros((NL, 4)), np.zeros(NL)) for v in VARS}
    items = [(t, sample_dir, m) for t in ts_list]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, res in enumerate(ex.map(fit_one, items, chunksize=16)):
            for vi, v in enumerate(VARS):
                S[v][0][:] += res[vi][0]
                S[v][1][:] += res[vi][1]
            if (k + 1) % 2000 < workers:
                print(f"  fit {k + 1}/{len(ts_list)}", flush=True)
    params = {}
    for v in VARS:
        acc, r2 = S[v]
        n = acc[:, 0]
        g_delta, g_rho = _delta_rho(acc.sum(0), r2.sum(), n.sum(), m, v)
        pv = {"layers": [], "global": {"delta": g_delta, "rho": g_rho}}
        for L in range(NL):
            if n[L] < MIN_LAYER_N:
                pv["layers"].append(None)
                continue
            d_, r_ = _delta_rho(acc[L], r2[L], n[L], m, v)
            pv["layers"].append({"n": int(n[L]), "delta": d_, "rho": r_})
        params[v] = pv
        log.append(f"{v}: global δ={g_delta:+.4f} ρ={g_rho:.3f} | "
                   + " ".join(
                       (f"[{LAYERS[L]:g}-{LAYERS[L+1]:g}]δ={pv['layers'][L]['delta']:+.3f},"
                        f"ρ={pv['layers'][L]['rho']:.2f}") if pv["layers"][L]
                       else f"[{LAYERS[L]:g}+]fallback" for L in range(NL)))
    return params


# ---------------------------------------------------------------- apply


def apply_one(item):
    ts, sample_dir, out_dir, m, params = item
    ens = np.stack([np.load(os.path.join(sample_dir, f"sample_{ts}_ens{k:03d}.npz"))
                    ["sample"].squeeze() for k in range(m)], 0).astype(np.float32)
    mu = ens.mean(0)
    lay = np.clip(np.searchsorted(LAYERS, mu[:, 0], side="right") - 1, 0, NL - 1)
    out = ens.copy()
    for vi, v in enumerate(VARS):
        pv = params[v]
        pl = [pv["layers"][L] if pv["layers"][L] else pv["global"]
              for L in range(NL)]
        deltas = np.array([p["delta"] for p in pl])[lay]     # (N,)
        rhos = np.array([p["rho"] for p in pl])[lay]
        if v != "dm":
            z = fwd(ens[:, :, vi], v)
            zm = z.mean(0)
            zc = (zm - deltas)[None] + rhos[None] * (z - zm[None])
            out[:, :, vi] = inv(zc, v)
        else:
            rad = np.deg2rad(ens[:, :, vi])
            radm = np.arctan2(np.sin(rad).mean(0), np.cos(rad).mean(0))  # 圆周均值,避免跨 0/360 成员平均出 ~180° 的假中心
            dr = np.arctan2(np.sin(rad - radm[None]), np.cos(rad - radm[None]))
            rc = radm[None] + np.deg2rad(deltas)[None] + rhos[None] * dr
            out[:, :, vi] = np.rad2deg(rc) % 360.0
    # 非负物理约束
    out[:, :, 0] = out[:, :, 0].clip(0)
    out[:, :, 1] = out[:, :, 1].clip(0)
    for k in range(m):
        np.savez(os.path.join(out_dir, f"sample_{ts}_ens{k:03d}.npz"),
                 sample=out[k])
    return ts


def main():
    global GT_DIR, LAYERS, NL
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-samples", required=True)
    ap.add_argument("--apply-samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-dir", default=GT_DIR)
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--layers", default="v1",
                    help="分层方案: 'v1'(默认,顶部 2.5-inf) / 'v2'(顶部细分 2.5/3/4/5/inf) "
                         "/ 逗号分隔数值如 0,0.5,1,2,4,inf")
    ap.add_argument("--delta-only", action="store_true",
                    help="纯偏差订正模式: 只拟合/应用 δ,强制 ρ=1(不动 spread)。"
                         "对应'静态偏差订正 bake-in'叙事,区别于完整 affine 校准")
    ap.add_argument("--freeze-top", action="store_true",
                    help="顶层(默认 v1 的 2.5-inf)δ=0,ρ=1, 即不订正。"
                         "该层 δ 由虚警节点(μ 高 y 低)主导、符号与真极端相反,"
                         "订正会恶化 >4m 档(bc 实验: 0.563->0.793)。")
    ap.add_argument("--limit-ts", type=int, default=0, help=">0 冒烟：只用前 N 个 ts")
    args = ap.parse_args()
    GT_DIR = args.gt_dir
    if args.layers == "v1":
        pass
    elif args.layers == "v2":
        LAYERS = LAYERS_V2
    else:
        LAYERS = np.array([float(x) for x in args.layers.split(",")])
    NL = len(LAYERS) - 1

    log = []
    print(f"affine calib: fit={args.fit_samples} apply={args.apply_samples} "
          f"out={args.out} M={args.members} limit_ts={args.limit_ts}")
    ts_fit = list_ts(args.fit_samples)
    if args.limit_ts > 0:
        ts_fit = ts_fit[:args.limit_ts]
    params = fit_params(ts_fit, args.fit_samples, args.members, args.workers, log)
    if args.delta_only:
        for v in VARS:
            params[v]["global"]["rho"] = 1.0
            for lp in params[v]["layers"]:
                if lp:
                    lp["rho"] = 1.0
        log.append("delta-only: all ρ forced to 1.0")
    if args.freeze_top:
        for v in VARS:
            top = params[v]["layers"][-1]
            n_top = top["n"] if top else 0
            params[v]["layers"][-1] = {"n": n_top, "delta": 0.0, "rho": 1.0}
        log.append(f"freeze-top: [{LAYERS[-2]:g}-inf) δ=0,ρ=1 (不订正)")
    for line in log:
        print(" ", line)

    os.makedirs(args.out, exist_ok=True)
    ts_apply = list_ts(args.apply_samples)
    if args.limit_ts > 0:
        ts_apply = ts_apply[:args.limit_ts]
    items = [(t, args.apply_samples, args.out, args.members, params)
             for t in ts_apply]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, _ in enumerate(ex.map(apply_one, items, chunksize=16)):
            if (k + 1) % 2000 < args.workers:
                print(f"  apply {k + 1}/{len(ts_apply)}", flush=True)

    ppath = os.path.join(os.path.dirname(args.out.rstrip("/")), "calib_params.json")
    with open(ppath, "w") as f:
        json.dump({"layers": LAYERS.tolist(), "params": params}, f, indent=1)
    print(f"CALIB DONE: {args.out} (params -> {ppath})")


if __name__ == "__main__":
    main()
