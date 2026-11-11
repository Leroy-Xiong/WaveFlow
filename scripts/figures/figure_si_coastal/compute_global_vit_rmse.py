#!/usr/bin/env python
"""Accumulate per-node squared errors of Hs for the global WaveFlow-ViT run.

WaveFlow-ViT stores one node-level field per ensemble member as a single
variable (651,652 float32), so ten members are averaged per timestamp. The
reference is the global hindcast, whose Hs is stored per timestamp under
data/global_hindcast/npz_files/<year>/global_hindcast_<YYYYMMDD>_<HH>.npz.

Only the 2021--2022 test period is used, which is the 5,840-snapshot window
behind main-text Fig. 2e; the stored run also contains ten days of January
2023, which are excluded here.

OUTPUT
  sqerr  (651652,) float64   summed squared error
  count  (651652,) int32     contributing timestamps

USAGE
  python compute_global_vit_rmse.py --root <samples_nodes> --out /tmp/vit_global.npz
"""
from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

TS_RE = re.compile(r"sample_(\d{8}T\d{2})_ens\d+\.npz")
GT_PATTERN = "data/global_hindcast/npz_files/{year}/global_hindcast_{date}_{hour}.npz"


def timestamps(root):
    ts = set()
    for name in os.listdir(root):
        m = TS_RE.match(name)
        if not m:
            continue
        t = m.group(1)
        if t[:4] in ("2021", "2022"):
            ts.add(t)
    return sorted(ts)


def read_pred(root, ts):
    acc = None
    for k in range(10):
        arr = np.load(os.path.join(root, f"sample_{ts}_ens{k:03d}.npz"))["sample"]
        arr = arr.astype(np.float64)
        acc = arr if acc is None else acc + arr
    return acc / 10.0


def read_gt(ts):
    path = GT_PATTERN.format(year=ts[:4], date=ts[:8], hour=ts[9:11])
    with np.load(path) as d:
        return d["hs"].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ts_all = timestamps(args.root)
    if args.limit:
        step = max(len(ts_all) // args.limit, 1)
        ts_all = ts_all[::step][:args.limit]
    print(f"global vit: {len(ts_all)} timestamps", flush=True)

    sq = None
    cnt = None
    done = 0

    def work(ts):
        return (read_pred(args.root, ts) - read_gt(ts)) ** 2

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for e2 in ex.map(work, ts_all):
            if sq is None:
                sq = np.zeros(e2.shape[0], dtype=np.float64)
                cnt = np.zeros(e2.shape[0], dtype=np.int32)
            ok = np.isfinite(e2)
            sq[ok] += e2[ok]
            cnt[ok] += 1
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(ts_all)}", flush=True)

    np.savez(args.out, sqerr=sq, count=cnt)
    print(f"saved {args.out}  pooled node-snapshot RMSE = "
          f"{np.sqrt(sq.sum() / cnt.sum()):.4f} m  (n_ts={len(ts_all)})")


if __name__ == "__main__":
    main()
