#!/usr/bin/env python
"""bc3: 连续版分层 log-affine 偏差订正（val 拟合 → test 应用）。

与 bc2（uq_affine_calib.py --delta-only --freeze-top）拟合协议完全一致：
按 ensemble-mean hs 分 9 层、每层拟合 δ（ρ 强制 1、顶层冻结 δ=0）。
唯一区别在 apply：bc2 的 δ(μ) 是分段常数，节点跨过层边界时订正量跳变，
空间/时间上会出现"分层"条带；bc3 把 δ(μ) 改为对层 δ 的线性插值：

  - 结点横坐标 = 该层拟合样本的 log(μ+0.01) 经验均值（该层 δ 的代表点），
    而不是几何中心——保证插值后每层典型位置上的订正量等于 bc2 的层 δ；
  - 冻结语义：在 u(2.5m) 处钉 δ=0 结点，订正量向顶层平滑收敛到 0；
  - 端点外常数外推。

输出与 bc2 同构：--out 下 sample_{ts}_ens{NNN}.npz + 上一级 calib_params.json，
可直接用 post_process.py rmse-multivar 评估。

用法（与 bc2 对照）:
  python scripts/uq/uq_affine_calib_v3.py \
      --fit-samples <run>/curve_ep49/val/samples \
      --apply-samples <run>/curve_ep49/test/samples \
      --out <run>/curve_ep49/test_bc3/samples --members 10 --workers 16
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uq_affine_calib as base  # 复用其常量与 GT 读取协议

C_HS = base.C_LOG["hs"]  # 插值坐标 u = log(μ_hs + C_HS)
NL = len(base.LAYERS) - 1


# ---------------------------------------------------------------- fit
# 与 base.fit_one 相同，额外累积每层 Σu（u=log(μ_hs+C)）用于定位插值结点

def fit_one(item):
    """同 base.fit_one 的累积协议（acc/r2 列含义见其注释），额外返回每层 Σu。"""
    _, ens, gt = base.load_ens_gt(item)
    mu = ens.mean(0)                       # (N,4)
    lay = np.clip(np.searchsorted(base.LAYERS, mu[:, 0], side="right") - 1, 0, NL - 1)
    u = np.log(mu[:, 0] + C_HS)
    sumu = np.zeros(NL)
    out = []
    for vi, v in enumerate(base.VARS):
        acc = np.zeros((NL, 4))
        r2 = np.zeros(NL)
        if v != "dm":
            z = base.fwd(ens[:, :, vi], v)
            zm = z.mean(0)
            d = zm - base.fwd(gt[:, vi], v)
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
    for L in np.unique(lay):
        sumu[L] = u[lay == L].sum()
    return out, sumu


def fit_one_chunk(items):
    """Worker-sized slice of fit items.

    Dispatching one future per timestamp is ~10-80x slower on this shared
    filesystem: each np.load costs a full round trip and per-item dispatch adds
    GIL contention. Handing every worker a contiguous slice keeps reads
    pipelined (same trick as eval_uq_ensemble.process_chunk).
    """
    return [fit_one(it) for it in items]


def fit_params(ts_list, sample_dir, m, workers, log):
    S = {v: (np.zeros((NL, 4)), np.zeros(NL)) for v in base.VARS}
    S_u = np.zeros(NL)
    items = [(t, sample_dir, m) for t in ts_list]
    chunks = [items[i::workers] for i in range(workers)]
    chunks = [c for c in chunks if c]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        done = 0
        for results in ex.map(fit_one_chunk, chunks):
            for res, sumu in results:
                for vi, v in enumerate(base.VARS):
                    S[v][0][:] += res[vi][0]
                    S[v][1][:] += res[vi][1]
                S_u += sumu
                done += 1
            print(f"  fit {done}/{len(ts_list)}", flush=True)
    params = {}
    for v in base.VARS:
        acc, r2 = S[v]
        n = acc[:, 0]
        g_delta, g_rho = base._delta_rho(acc.sum(0), r2.sum(), n.sum(), m, v)
        pv = {"layers": [], "global": {"delta": g_delta, "rho": g_rho}}
        for L in range(NL):
            if n[L] < base.MIN_LAYER_N:
                pv["layers"].append(None)
                continue
            d_, r_ = base._delta_rho(acc[L], r2[L], n[L], m, v)
            pv["layers"].append({"n": int(n[L]), "delta": d_, "rho": r_})
        params[v] = pv
        log.append(f"{v}: global δ={g_delta:+.4f} ρ={g_rho:.3f} | "
                   + " ".join(
                       (f"[{base.LAYERS[L]:g}-{base.LAYERS[L+1]:g}]δ={pv['layers'][L]['delta']:+.3f},"
                        f"ρ={pv['layers'][L]['rho']:.2f}") if pv["layers"][L]
                       else f"[{base.LAYERS[L]:g}+]fallback" for L in range(NL)))
    n_layer = S[base.VARS[0]][0][:, 0]
    mean_u = np.where(n_layer > 0, S_u / np.maximum(n_layer, 1), np.nan)
    return params, mean_u


def layer_knots(params, mean_u, freeze_top=True):
    """每变量 (knots_u, knots_delta)：经验层均值 u → 该层 δ；
    freeze_top 时追加 u(2.5m) 处 δ=0 结点。"""
    knots = {}
    for v in base.VARS:
        pv = params[v]
        us, ds = [], []
        for L in range(NL - 1):           # 顶层(2.5-inf)不取经验结点
            if not np.isfinite(mean_u[L]):
                continue
            p = pv["layers"][L] if pv["layers"][L] else pv["global"]
            us.append(mean_u[L])
            ds.append(p["delta"])
        if freeze_top:
            us.append(np.log(base.LAYERS[-2] + C_HS))
            ds.append(0.0)
        order = np.argsort(us)
        knots[v] = (np.array(us)[order], np.array(ds)[order])
    return knots


# ---------------------------------------------------------------- apply

def apply_one(item):
    ts, sample_dir, out_dir, m, knots = item
    ens = np.stack([np.load(os.path.join(sample_dir, f"sample_{ts}_ens{k:03d}.npz"))
                    ["sample"].squeeze() for k in range(m)], 0).astype(np.float32)
    mu = ens.mean(0)
    u = np.log(mu[:, 0] + C_HS)              # (N,) 插值坐标
    out = ens.copy()
    for vi, v in enumerate(base.VARS):
        us, ds = knots[v]
        deltas = np.interp(u, us, ds)        # (N,) 连续 δ(μ)
        if v != "dm":
            z = base.fwd(ens[:, :, vi], v)
            zm = z.mean(0)
            out[:, :, vi] = base.inv((zm - deltas)[None] + (z - zm[None]), v)
        else:
            rad = np.deg2rad(ens[:, :, vi])
            radm = np.arctan2(np.sin(rad).mean(0), np.cos(rad).mean(0))  # 圆周均值,避免跨 0/360 成员平均出 ~180° 的假中心
            dr = np.arctan2(np.sin(rad - radm[None]), np.cos(rad - radm[None]))
            out[:, :, vi] = np.rad2deg(radm[None] + np.deg2rad(deltas)[None] + dr) % 360.0
    out[:, :, 0] = out[:, :, 0].clip(0)
    out[:, :, 1] = out[:, :, 1].clip(0)
    for k in range(m):
        np.savez(os.path.join(out_dir, f"sample_{ts}_ens{k:03d}.npz"),
                 sample=out[k])
    return ts


def apply_one_chunk(items):
    return [apply_one(it) for it in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-samples", required=True)
    ap.add_argument("--apply-samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-dir", default=base.GT_DIR)
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-ts", type=int, default=0)
    ap.add_argument("--fit-stride", type=int, default=1,
                    help="拟合期每 N 个时刻取 1 个（等间隔）；分层 δ 是均值，"
                         "子样本足够，应用期不受影响")
    ap.add_argument("--fit-only", action="store_true",
                    help="只拟合并写出 calib_params.json，不重写样本"
                         "（配合 eval_uq_ensemble_bc3.py 使用）")
    args = ap.parse_args()
    base.GT_DIR = args.gt_dir  # base.load_ens_gt 用的是 base 模块全局量

    log = []
    ts_fit = base.list_ts(args.fit_samples)
    if args.fit_stride > 1:
        ts_fit = ts_fit[::args.fit_stride]
    if args.limit_ts > 0:
        ts_fit = ts_fit[:args.limit_ts]
    params, mean_u = fit_params(ts_fit, args.fit_samples, args.members,
                                args.workers, log)
    # bc3 = delta-only + freeze-top（与 bc2 口径一致）
    for v in base.VARS:
        params[v]["global"]["rho"] = 1.0
        for lp in params[v]["layers"]:
            if lp:
                lp["rho"] = 1.0
        top = params[v]["layers"][-1]
        n_top = top["n"] if top else 0
        params[v]["layers"][-1] = {"n": n_top, "delta": 0.0, "rho": 1.0}
    log.append("bc3: delta-only, freeze-top, δ(μ) 沿 log-hs 轴按层经验均值结点线性插值")
    for line in log:
        print(" ", line, flush=True)

    knots = layer_knots(params, mean_u, freeze_top=True)
    for v in base.VARS:
        us, ds = knots[v]
        print(f"  {v} knots(hs_m,δ): "
              + " ".join(f"({np.exp(u) - C_HS:.3f},{d:+.4f})" for u, d in zip(us, ds)),
              flush=True)

    ppath = os.path.join(os.path.dirname(args.out.rstrip("/")), "calib_params.json")
    os.makedirs(os.path.dirname(ppath), exist_ok=True)
    with open(ppath, "w") as f:
        json.dump({
            "mode": "bc3-interp",
            "interp_axis": f"log(mu_hs + {C_HS}), knots at per-layer empirical mean",
            "layers": base.LAYERS.tolist(),
            "params": params,
            "knots": {v: {"u": knots[v][0].tolist(), "delta": knots[v][1].tolist()}
                      for v in base.VARS},
        }, f, indent=1)
    print(f"KNOTS WRITTEN: {ppath}", flush=True)
    if args.fit_only:
        print("fit-only: skipping the apply phase")
        return

    os.makedirs(args.out, exist_ok=True)
    ts_apply = base.list_ts(args.apply_samples)
    if args.limit_ts > 0:
        ts_apply = ts_apply[:args.limit_ts]
    items = [(t, args.apply_samples, args.out, args.members, knots)
             for t in ts_apply]
    chunks = [items[i::args.workers] for i in range(args.workers)]
    chunks = [c for c in chunks if c]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        done = 0
        for res in ex.map(apply_one_chunk, chunks):
            done += len(res)
            print(f"  apply {done}/{len(ts_apply)}", flush=True)

    print(f"CALIB DONE: {args.out} (params -> {ppath})")


if __name__ == "__main__":
    main()
