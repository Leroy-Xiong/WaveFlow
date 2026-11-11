#!/usr/bin/env python
"""Pre-compute per-timestamp storm intensity (region-max hs) over the training set.

Scans all non-2021/2022 wave NPZ files and saves:
  ./data/hindcast/train_ts_max_hs.npz   (timestamps, max_hs, plus percentiles)

Used by WaveDataset oversampling (weighted sampling of storm hours).
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import os
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def _scan(path):
    ts = os.path.basename(path).replace("wave_", "").replace(".npz", "")
    try:
        hs = np.load(path)["hs"]
        return ts, float(np.nanmax(hs))
    except Exception:
        return ts, float("nan")


def parse_args(argv=None):
    """Parse CLI arguments (shared with load_data.py / post_process.py wrappers)."""
    return _build_parser().parse_args(argv)


def _build_parser():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/hindcast/processed_npz")
    ap.add_argument("--out", default="./data/hindcast/train_ts_max_hs.npz")
    ap.add_argument("--workers", type=int, default=32)
    return ap


def run(args):
    files = sorted(glob.glob(os.path.join(args.data_dir, "wave_*.npz")))
    files = [f for f in files if os.path.basename(f)[5:9] not in ("2021", "2022")]
    print(f"train files: {len(files)}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (ts, mx) in enumerate(ex.map(_scan, files, chunksize=64)):
            results.append((ts, mx))
            if (i + 1) % 50000 == 0:
                print(f"  scanned {i+1}/{len(files)}", flush=True)

    results.sort(key=lambda r: r[0])
    ts_arr = np.array([r[0] for r in results])
    mx_arr = np.array([r[1] for r in results])
    qs = np.nanpercentile(mx_arr, [50, 90, 95, 99, 99.5, 99.9, 100])
    print("\nregion-max hs percentiles (train set):")
    for q, v in zip([50, 90, 95, 99, 99.5, 99.9, 100], qs):
        print(f"  q{q:>5.1f} = {v:.3f} m")
    for th in [2.0, 3.0, 4.0, 5.0, 6.0]:
        print(f"  fraction ts with max_hs >= {th:.0f}m: {100.0 * np.mean(mx_arr >= th):.4f}%  "
              f"({int((mx_arr >= th).sum())} timestamps)")
    np.savez_compressed(args.out, timestamps=ts_arr, max_hs=mx_arr)
    print(f"saved -> {args.out}")


def main():
    run(parse_args())

if __name__ == "__main__":
    main()
