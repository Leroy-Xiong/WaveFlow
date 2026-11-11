#!/usr/bin/env python
"""Grouped (Mondrian-style) split-conformal interval calibration for ensembles.

- Fit on <fit_year>: per (variable, layer-by-ensemble-mean-hs) collect
  normalized nonconformity scores  s = |y - median(ens)| / spread(ens).
- Apply on <apply_year>: interval = median +/- q_layer(1-alpha) * spread,
  with the finite-sample conformal quantile correction (1+1/n).

Reports coverage vs nominal at several widths, overall and per layer,
plus mean interval width (sharpness).

Usage:
    python conformal_grouped.py --sample-dir DIR --num-ensemble 100 \
        --fit-year 2021 --apply-year 2022 --widths 0.5,0.7,0.8,0.9,0.95,0.99
"""
import argparse
import os
import re
import sys

import numpy as np
from concurrent.futures import ProcessPoolExecutor

VAR_ORDER = ["hs", "tp", "dm", "elev"]
DEFAULT_LAYER_BINS = [0.0, 0.75, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, np.inf]
SPREAD_EPS = 1e-3


def _score_layer(args):
    ts, sample_dir, gt_dir, num_ens, layer_bins = args
    files = [os.path.join(sample_dir, f"sample_{ts}_ens{i:03d}.npz") for i in range(num_ens)]
    if not os.path.exists(files[0]) or not os.path.exists(files[-1]):
        return None
    gt = np.load(os.path.join(gt_dir, f"wave_{ts}.npz"))
    ens = np.stack([np.load(f)["sample"].squeeze() for f in files], axis=0)  # (M,N,4)
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
            s = np.abs(err) / spread
        else:
            s = np.abs(g - med) / spread
        ll = layers[v]
        out[var] = (s, ll, med, spread, g)
    return out


def _apply_one(args):
    ts, sample_dir, gt_dir, num_ens, layer_bins, qmaps, widths = args
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
        ll = layers[v]
        qm = qmaps[var]  # (n_layers, n_widths)
        n_layers, n_w = qm.shape
        q = qm[ll]  # (n, n_w)
        lo = med[:, None] - q * spread[:, None]
        hi = med[:, None] + q * spread[:, None]
        if var == "dm":
            inside = ((g[:, None] - lo + 180.0) % 360.0 - 180.0 >= 0) & \
                     ((g[:, None] - lo + 180.0) % 360.0 - 180.0 <= (hi - lo))
        else:
            inside = (g[:, None] >= lo) & (g[:, None] <= hi)
        cov = inside.mean(axis=0)  # (n_w,)
        width90 = float(np.mean(hi[:, widths == 0.9] - lo[:, widths == 0.9])) if 0.9 in widths else float("nan")
        # per-layer counts for weighted aggregation
        cnt = np.zeros(n_layers, dtype=np.int64)
        for L in range(n_layers):
            cnt[L] = int((ll == L).sum())
        out[var] = (cov * cnt[:, None], cnt, width90, inside, ll)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--num-ensemble", type=int, default=100)
    ap.add_argument("--gt-dir", default="data/hindcast/processed_npz")
    ap.add_argument("--fit-year", type=int, default=2021)
    ap.add_argument("--apply-year", type=int, default=2022)
    ap.add_argument("--layer-bins", nargs="+", type=float, default=DEFAULT_LAYER_BINS)
    ap.add_argument("--widths", default="0.5,0.6,0.7,0.8,0.9,0.95,0.99")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    widths = np.array([float(w) for w in args.widths.split(",")])
    sample_dir = os.path.join(args.sample_dir, "samples")
    ts_all = sorted({re.match(r"sample_(\d{8}T\d{2})_ens\d+\.npz$", f).group(1)
                     for f in os.listdir(sample_dir)
                     if re.match(r"sample_\d{8}T\d{2}_ens\d+\.npz$", f)})
    ts_fit = [t for t in ts_all if t.startswith(str(args.fit_year))]
    ts_apply = [t for t in ts_all if t.startswith(str(args.apply_year))]
    print(f"fit: {len(ts_fit)} ts ({args.fit_year}), apply: {len(ts_apply)} ts ({args.apply_year})",
          flush=True)

    layer_bins = np.array([np.inf if b == np.inf else b for b in args.layer_bins])
    n_layers = len(layer_bins) - 1
    n_w = len(widths)

    # ---------- fit ----------
    scores = {v: [[] for _ in range(n_layers)] for v in VAR_ORDER}
    tasks = [(t, sample_dir, args.gt_dir, args.num_ensemble, layer_bins) for t in ts_fit]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_score_layer, tasks, chunksize=4):
            if res is None:
                continue
            for v in VAR_ORDER:
                if v not in res:
                    continue
                s, ll, _, _, _ = res[v]
                for L in range(n_layers):
                    m = ll == L
                    if m.sum():
                        scores[v][L].append(s[m])
            done += 1
            if done % 2000 == 0:
                print(f"  fit {done}/{len(ts_fit)}", flush=True)

    qmaps = {v: np.full((n_layers, n_w), np.nan) for v in VAR_ORDER}
    n_cal = {v: np.zeros(n_layers, dtype=np.int64) for v in VAR_ORDER}
    for v in VAR_ORDER:
        for L in range(n_layers):
            if scores[v][L]:
                s = np.concatenate(scores[v][L])
                n_cal[v][L] = len(s)
                for i, w in enumerate(widths):
                    level = min(1.0, w * (1.0 + 1.0 / len(s)))
                    qmaps[v][L, i] = float(np.quantile(s, level))
    print("fit done", flush=True)

    # ---------- apply ----------
    agg_cov = {v: np.zeros(n_w) for v in VAR_ORDER}
    agg_cnt = {v: 0 for v in VAR_ORDER}
    agg_layer_cov = {v: np.zeros((n_layers, n_w)) for v in VAR_ORDER}
    agg_layer_n = {v: np.zeros(n_layers, dtype=np.int64) for v in VAR_ORDER}
    agg_w90 = {v: [] for v in VAR_ORDER}
    tasks = [(t, sample_dir, args.gt_dir, args.num_ensemble, layer_bins, qmaps, widths)
             for t in ts_apply]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_apply_one, tasks, chunksize=4):
            if res is None:
                continue
            for v in VAR_ORDER:
                if v not in res:
                    continue
                wsum, cnt, w90, _, _ = res[v]
                agg_cov[v] += wsum.sum(axis=0)
                agg_layer_cov[v] += wsum
                agg_layer_n[v] += cnt
                agg_cnt[v] += int(cnt.sum())
                if not np.isnan(w90):
                    agg_w90[v].append((w90, int(cnt.sum())))
            done += 1
            if done % 2000 == 0:
                print(f"  apply {done}/{len(ts_apply)}", flush=True)

    print("\n=== overall coverage (nominal vs achieved) ===")
    print(f"{'var':>5s} " + "".join(f"{w:>7.0%}" for w in widths))
    for v in VAR_ORDER:
        print(f"{v:>5s} " + "".join(f"{agg_cov[v][i] / agg_cnt[v]:>7.1%}"
                                    for i in range(n_w)))
    print("\n=== hs per-layer coverage ===")
    hdr = f"{'layer':>12s} " + "".join(f"{w:>7.0%}" for w in widths) + f" {'n':>10s}"
    print(hdr)
    for L in range(n_layers):
        lo = layer_bins[L]; hi = layer_bins[L + 1]
        lab = f"{lo:.2f}-{hi if hi < 10 else 'inf'}"
        print(f"{lab:>12s} " + "".join(f"{agg_layer_cov['hs'][L, i] / agg_layer_n['hs'][L]:>7.1%}"
                                       for i in range(n_w))
              + f" {agg_layer_n['hs'][L]:>10d}")
    print("\n=== mean 90% interval width (sharpness) ===")
    for v in VAR_ORDER:
        w = np.average([x[0] for x in agg_w90[v]], weights=[x[1] for x in agg_w90[v]])
        print(f"{v:>5s}: {w:.3f} (units: {'m' if v in ['hs','elev'] else 's' if v=='tp' else 'deg'})")
    print("\ndone")


if __name__ == "__main__":
    main()
