#!/usr/bin/env python
"""Recompute hs/tp normalization statistics under (log1p(x))^p transform.

Streams over the training split (all non-2021/2022 wave NPZ files), computes
global mean/std of y = (log1p(x))^p for hs and tp, and writes a stats JSON
with the same key names as the legacy file (hs_log_mean, hs_log_std, ...),
plus elev/dm fields copied from the original file.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def _accum(path, power):
    d = np.load(path)
    hs = d["hs"].astype(np.float64)
    tp = d["tp"].astype(np.float64)
    yh = np.power(np.maximum(np.log1p(hs), 0.0), power)
    yt = np.power(np.maximum(np.log1p(tp), 0.0), power)
    return (yh.sum(), yt.sum(), (yh * yh).sum(), (yt * yt).sum(), yh.size)


def _accum_power(args):
    return _accum(args[0], args[1])


def parse_args(argv=None):
    """Parse CLI arguments (shared with load_data.py / post_process.py wrappers)."""
    return _build_parser().parse_args(argv)


def _build_parser():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", type=float, default=0.6)
    ap.add_argument("--data-dir", default="./data/hindcast/processed_npz")
    ap.add_argument("--in-stats", default="./data/hindcast/wave_data_stats.json")
    ap.add_argument("--out", default="./data/hindcast/wave_data_stats_p06.json")
    ap.add_argument("--workers", type=int, default=48)
    return ap


def run(args):
    files = sorted(glob.glob(os.path.join(args.data_dir, "wave_*.npz")))
    files = [f for f in files if os.path.basename(f)[5:9] not in ("2021", "2022")]
    print(f"train files: {len(files)}")

    sh = st = sh2 = st2 = n = 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (a, b, c, d, e) in enumerate(
                ex.map(_accum_power, [(f, args.power) for f in files], chunksize=64)):
            sh += a; st += b; sh2 += c; st2 += d; n += e
            if (i + 1) % 50000 == 0:
                print(f"  scanned {i+1}/{len(files)}", flush=True)

    with open(args.in_stats) as f:
        stats = json.load(f)
    for key, s, s2 in [("hs", sh, sh2), ("tp", st, st2)]:
        mean = s / n
        var = s2 / n - mean * mean
        std = float(np.sqrt(max(var, 1e-12)))
        stats[f"{key}_log_mean"] = float(mean)
        stats[f"{key}_log_std"] = std
        print(f"{key}: y=(log1p)^{args.power}  mean={mean:.5f} std={std:.5f}")

    stats["_normalize_note"] = (f"hs/tp transformed as (log1p(x))^{args.power} "
                                f"then z-scored (log_power={args.power})")
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"saved -> {args.out}")


def main():
    run(parse_args())

if __name__ == "__main__":
    main()
