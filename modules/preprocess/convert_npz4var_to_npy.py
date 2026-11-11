"""Convert per-timestamp 4-var wave NPZ files to uncompressed NPY files.

- src: data/global_hindcast/npz_files_4var/YYYY/global_hindcast_YYYYMMDD_HH.npz
    (keys hs/tp/dm/elev, each [651652] float32, DEFLATE-compressed)
- dst: data/global_hindcast/npy_files_4var/YYYY/global_hindcast_YYYYMMDD_HH.npy
    float32 [651652, 4], columns (hs, tp, dm, elev), uncompressed.

Also writes wave_columns.json in the dst root so WaveDataset's npy loader
picks the 4-var column layout (mirrors wave_daily_columns.json convention).

Idempotent: existing dst files are trusted on resume. Atomic per-file write
(tmp + os.replace). Run from repo root:

    python modules/preprocess/convert_npz4var_to_npy.py --workers 48
    python modules/preprocess/convert_npz4var_to_npy.py --verify-only
"""
import argparse
import glob
import json
import os
import time
from multiprocessing import Process

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

SRC_DIR = os.path.join(REPO_ROOT, 'data/global_hindcast/npz_files_4var')
DST_DIR = os.path.join(REPO_ROOT, 'data/global_hindcast/npy_files_4var')
COLUMNS = ['hs', 'tp', 'dm', 'elev']


def convert_one(src: str, dst: str) -> None:
    data = np.load(src, allow_pickle=True)
    try:
        arr = np.stack([data[c] for c in COLUMNS], axis=-1)  # [N, 4]
    finally:
        data.close()
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp_path = dst + '.tmp'
    with open(tmp_path, 'wb') as fh:
        np.save(fh, arr)
    os.replace(tmp_path, dst)


def worker(jobs, report_path):
    """jobs: list of (src, dst)."""
    failed = []
    t0 = time.time()
    n_skip = 0
    for n, (src, dst) in enumerate(jobs):
        try:
            if not os.path.exists(dst):
                convert_one(src, dst)
            else:
                n_skip += 1
        except Exception as e:
            failed.append({'file': src, 'error': str(e)})
        if n % 500 == 499:
            rate = (n + 1) / (time.time() - t0)
            print(f"  [{os.path.basename(src)[:30]}] {n + 1}/{len(jobs)} "
                  f"({rate:.1f} files/s, skipped={n_skip}, failed={len(failed)})",
                  flush=True)
    with open(report_path, 'w') as fh:
        json.dump({'failed': failed}, fh)
    print(f"  [slice] done: {len(jobs)} files, {n_skip} skipped, "
          f"{len(failed)} failures", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--verify-only', action='store_true',
                    help='only run the verification pass (counts + byte equality)')
    ap.add_argument('--verify-sample', type=int, default=64,
                    help='number of random files for byte-equality verification')
    args = ap.parse_args()

    src_files = sorted(glob.glob(os.path.join(SRC_DIR, '*', 'global_hindcast_*.npz')))
    dst_files = [p.replace(SRC_DIR, DST_DIR).replace('.npz', '.npy') for p in src_files]
    print(f'source: {len(src_files)} npz files')

    if not args.verify_only:
        jobs = list(zip(src_files, dst_files))
        shards = [jobs[i::args.workers] for i in range(args.workers)]
        report_paths = [os.path.join('/tmp', f'npz2npy_report_{i}.json')
                        for i in range(args.workers)]
        procs = [Process(target=worker, args=(s, r))
                 for s, r in zip(shards, report_paths)]
        t0 = time.time()
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        n_failed = sum(len(json.load(open(r))['failed']) for r in report_paths)
        print(f'conversion done in {(time.time() - t0) / 60:.1f} min, '
              f'{n_failed} failures')

    # ---- verification: dst count + columns json + random byte equality ----
    n_dst = len(glob.glob(os.path.join(DST_DIR, '*', 'global_hindcast_*.npy')))
    print(f'verify: dst files={n_dst}, src files={len(src_files)}')
    if n_dst != len(src_files):
        print('verify: FILE COUNT MISMATCH — do not delete npz yet!')
        raise SystemExit(1)

    rng = np.random.RandomState(7)
    idx = rng.choice(len(src_files), size=min(args.verify_sample, len(src_files)),
                     replace=False)
    max_diff = 0.0
    for i in idx:
        d = np.load(src_files[i], allow_pickle=True)
        ref = np.stack([d[c] for c in COLUMNS], axis=-1).astype(np.float32)
        d.close()
        got = np.load(dst_files[i])
        if got.shape != (651652, 4) or got.dtype != np.float32:
            print(f'verify: bad shape/dtype in {dst_files[i]}: {got.shape} {got.dtype}')
            raise SystemExit(1)
        max_diff = max(max_diff, float(np.abs(got - ref).max()))
    print(f'verify: {len(idx)} files byte-compared, max abs diff = {max_diff}')
    if max_diff != 0.0:
        print('verify: CONTENT MISMATCH — do not delete npz yet!')
        raise SystemExit(1)

    with open(os.path.join(DST_DIR, 'wave_columns.json'), 'w') as fh:
        json.dump({'columns': COLUMNS}, fh)
    print(f"verify: OK; wrote {os.path.join(DST_DIR, 'wave_columns.json')}")


if __name__ == '__main__':
    main()
