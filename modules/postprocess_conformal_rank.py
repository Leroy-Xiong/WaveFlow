#!/usr/bin/env python
"""Conformal rank remapping (signed-score CPD + rank-preserving ECC), layered.

Fit (on fit_year): per (variable, layer by raw ensemble-mean hs), collect the
SIGNED normalized nonconformity score  s = (y - median(ens)) / spread(ens)
and build its empirical quantile function F_s^{-1} (a full, possibly asymmetric
predictive-distribution shape).

Apply (on apply_year): for each node-hour, remap every ensemble member's rank
u_k = (k+1)/(M+1) to  median + spread * F_s^{-1}(u_k)  (clipped >= 0 for
hs/tp/elev; dm wrapped to [0,360)). Output is a full M-member calibrated
ensemble that keeps rank order but adopts the calibrated (asymmetric) shape.

Usage:
    python modules/postprocess_conformal_rank.py --sample-dir DIR \
        --out-dir OUT --num-ensemble 100 --fit-year 2021 --apply-year 2022
"""
import argparse
import os
import re

import numpy as np
from concurrent.futures import ProcessPoolExecutor

VAR_ORDER = ["hs", "tp", "dm", "elev"]
DEFAULT_LAYER_BINS = [0.0, 0.75, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, np.inf]
N_GRID = 2001
SPREAD_EPS = 1e-3
CAP = 2_000_000  # max scores kept per (var, layer) for the quantile fn


def _fit_one(args):
    ts, sample_dir, gt_dir, num_ens, layer_bins = args
    files = [os.path.join(sample_dir, f"sample_{ts}_ens{i:03d}.npz") for i in range(num_ens)]
    if not os.path.exists(files[0]) or not os.path.exists(files[-1]):
        return None
    gt = np.load(os.path.join(gt_dir, f"wave_{ts}.npz"))
    ens = np.stack([np.load(f)["sample"].squeeze() for f in files], axis=0)
    ghs = gt["hs"].astype(np.float32).squeeze()
    valid = ~np.isnan(ghs)
    if valid.sum() == 0:
        return None
    layers = np.clip(np.searchsorted(layer_bins, ens[:, :, 0].mean(0), side="right") - 1,
                     0, len(layer_bins) - 2)
    out = {}
    for vi, var in enumerate(VAR_ORDER):
        gv = gt[var].astype(np.float32).squeeze()
        v = valid & ~np.isnan(gv)
        if v.sum() == 0:
            continue
        ev = ens[:, :, vi][:, v]
        g = gv[v]
        med = np.median(ev, axis=0)
        spread = ev.std(axis=0) + SPREAD_EPS
        if var == "dm":
            err = (g - med + 180.0) % 360.0 - 180.0
        else:
            err = g - med
        s = err / spread
        ll = layers[v]
        out[var] = (s, ll)
    return out


def _apply_one(args):
    ts, sample_dir, out_dir, gt_dir, num_ens, layer_bins, qfns = args
    files = [os.path.join(sample_dir, f"sample_{ts}_ens{i:03d}.npz") for i in range(num_ens)]
    if not os.path.exists(files[0]) or not os.path.exists(files[-1]):
        return None
    ens = np.stack([np.load(f)["sample"].squeeze() for f in files], axis=0)  # (M,N,4)
    layers = np.clip(np.searchsorted(layer_bins, ens[:, :, 0].mean(0), side="right") - 1,
                     0, len(layer_bins) - 2)
    M = ens.shape[0]
    u_grid = (np.arange(M) + 1) / (M + 1)
    yc = np.zeros_like(ens)
    for vi, var in enumerate(VAR_ORDER):
        ev = ens[:, :, vi]
        med = np.median(ev, axis=0)
        spread = ev.std(axis=0) + SPREAD_EPS
        qfn = qfns[var]  # (n_layers, N_GRID) values on u = linspace(0,1,N_GRID)
        q = qfn[layers]  # (N, N_GRID)
        u_axis = np.linspace(0, 1, N_GRID)
        vals = np.stack([np.interp(u_grid, u_axis, q[n]) for n in range(q.shape[0])])  # (N, M)
        new = med[:, None] + spread[:, None] * vals  # (N, M)
        new = new.T  # (M, N)
        if var == "dm":
            new = new % 360.0
        else:
            new = np.clip(new, 0.0, None)
        yc[:, :, vi] = new
    for i in range(M):
        data = dict(np.load(files[i]))
        data["sample"] = yc[i]
        np.savez(os.path.join(out_dir, f"sample_{ts}_ens{i:03d}.npz"), **data)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-ensemble", type=int, default=100)
    ap.add_argument("--gt-dir", default="data/hindcast/processed_npz")
    ap.add_argument("--fit-year", type=int, default=2021)
    ap.add_argument("--apply-year", type=int, default=2022)
    ap.add_argument("--layer-bins", nargs="+", type=float, default=DEFAULT_LAYER_BINS)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    sample_dir = os.path.join(args.sample_dir, "samples")
    out_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)
    ts_all = sorted({re.match(r"sample_(\d{8}T\d{2})_ens\d+\.npz$", f).group(1)
                     for f in os.listdir(sample_dir)
                     if re.match(r"sample_\d{8}T\d{2}_ens\d+\.npz$", f)})
    ts_fit = [t for t in ts_all if t.startswith(str(args.fit_year))]
    ts_apply = [t for t in ts_all if t.startswith(str(args.apply_year))]
    print(f"fit {len(ts_fit)} ts, apply {len(ts_apply)} ts", flush=True)

    layer_bins = np.array([np.inf if b == np.inf else b for b in args.layer_bins])
    n_layers = len(layer_bins) - 1

    scores = {v: [[] for _ in range(n_layers)] for v in VAR_ORDER}
    tasks = [(t, sample_dir, args.gt_dir, args.num_ensemble, layer_bins) for t in ts_fit]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_fit_one, tasks, chunksize=4):
            if res is None:
                continue
            for v in VAR_ORDER:
                if v not in res:
                    continue
                s, ll = res[v]
                for L in range(n_layers):
                    m = ll == L
                    if m.sum():
                        scores[v][L].append(s[m])
            done += 1
            if done % 2000 == 0:
                print(f"  fit {done}/{len(ts_fit)}", flush=True)

    qfns = {}
    for v in VAR_ORDER:
        qfns[v] = np.zeros((n_layers, N_GRID))
        for L in range(n_layers):
            if scores[v][L]:
                s = np.concatenate(scores[v][L])
                if len(s) > CAP:
                    rng = np.random.default_rng(0)
                    s = rng.choice(s, CAP, replace=False)
                qfns[v][L] = np.quantile(s, np.linspace(0, 1, N_GRID))
                print(f"  {v} L{L}: n={len(s):,}  F_s^-1(0.5)={np.quantile(s,0.5):+.3f} "
                      f"F_s^-1(0.95)={np.quantile(s,0.95):+.3f}", flush=True)
            else:
                qfns[v][L] = np.linspace(0, 1, N_GRID)
    print("fit done", flush=True)

    tasks = [(t, sample_dir, out_dir, args.gt_dir, args.num_ensemble, layer_bins, qfns)
             for t in ts_apply]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for _ in ex.map(_apply_one, tasks, chunksize=4):
            done += 1
            if done % 2000 == 0:
                print(f"  apply {done}/{len(ts_apply)}", flush=True)
    print("done ->", args.out_dir)


if __name__ == "__main__":
    main()
