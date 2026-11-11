#!/usr/bin/env python
"""Accumulate per-node squared errors of Hs on the Hong Kong mesh.

The four models store their predictions in different places and formats, and
the gridded ones only kept node-level fields on one machine. This script reads
whichever format it is pointed at and accumulates, per mesh node, the squared
error of the ensemble mean (deterministic models contribute a single field),
so that a per-node RMSE can be formed later without moving the raw samples.

Formats
  waveflow : <root>/sample_<ts>_ens{000..009}.npz, key 'sample' (7788, 4)
  vit      : <root>/sample_<ts>_ens{000..009}.npz, key 'sample' (7788, 4)
  ef, ocn  : <root>/sample_<ts>.npz, key 'prediction' (7788, 5)

The Hs channel is index 0 in every case; the target comes from
data/hindcast/processed_npz/wave_<ts>.npz, key 'hs' (7788,).

OUTPUT (one .npz)
  sqerr  (7788,) float64  summed squared error
  count  (7788,) int32    number of contributing timestamps (no missing data
                          is expected for Hs, so this is uniform in practice)

USAGE
  python compute_hk_node_rmse.py --model waveflow \
      --root output/results/.../test_bc3/samples --out /tmp/wf.npz
  python compute_hk_node_rmse.py --model vit --root ... --out /tmp/vit.npz --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

TS_RE = re.compile(r"sample_(\d{8}T\d{2})")
GT_DIR = "data/hindcast/processed_npz"


def timestamps(root, model):
    names = os.listdir(root)
    ts = set()
    for n in names:
        m = TS_RE.match(n)
        if m:
            ts.add(m.group(1))
    return sorted(ts)


def read_pred(root, model, ts):
    if model in ("waveflow", "vit"):
        acc = None
        for k in range(10):
            p = os.path.join(root, f"sample_{ts}_ens{k:03d}.npz")
            arr = np.load(p)["sample"][:, 0].astype(np.float64)
            acc = arr if acc is None else acc + arr
        return acc / 10.0
    # The gridded baselines store node-level fields in normalized space
    # (log(1+Hs) z-scored), so they must be denormalized before comparison.
    p = os.path.join(root, f"sample_{ts}.npz")
    raw = np.load(p)["prediction"][:, 0].astype(np.float64)
    with open("data/hindcast/wave_data_stats.json") as fh:
        st = json.load(fh)
    return np.maximum(np.expm1(raw * st["hs_log_std"] + st["hs_log_mean"]), 0.0)


def read_gt(ts):
    with np.load(os.path.join(GT_DIR, f"wave_{ts}.npz")) as d:
        return d["hs"].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["waveflow", "vit", "ef", "ocn"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    ts_all = timestamps(args.root, args.model)
    if args.limit:
        step = max(len(ts_all) // args.limit, 1)
        ts_all = ts_all[::step][:args.limit]
    print(f"{args.model}: {len(ts_all)} timestamps")

    n_nodes = None
    sq = None
    cnt = None
    lock_holder = {"n": 0}

    def work(ts):
        pred = read_pred(args.root, args.model, ts)
        gt = read_gt(ts)
        return (pred - gt) ** 2

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for e2 in ex.map(work, ts_all):
            if sq is None:
                n_nodes = e2.shape[0]
                sq = np.zeros(n_nodes, dtype=np.float64)
                cnt = np.zeros(n_nodes, dtype=np.int32)
            ok = np.isfinite(e2)
            sq[ok] += e2[ok]
            cnt[ok] += 1
            lock_holder["n"] += 1
            if lock_holder["n"] % 2000 == 0:
                print(f"  {lock_holder['n']}/{len(ts_all)}")

    np.savez(args.out, sqerr=sq, count=cnt)
    rmse = np.sqrt(sq.sum() / cnt.sum())
    print(f"saved {args.out}  pooled node-hour RMSE = {rmse:.4f} m  "
          f"(n_ts={len(ts_all)})")


if __name__ == "__main__":
    main()
