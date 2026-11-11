"""Convert monolithic mmap caches to per-timestamp uncompressed NPY files.

Source is the verified mmap cache (byte-identical to the original NPZ data),
which avoids re-decompressing 2.6 TB of NPZ files. Output layout:

  wave: data/global_hindcast/npy_files/{year}/global_hindcast_YYYYMMDD_HH.npy
        float32 [651652, 3], columns = hs, mwd, ssh
  era5: data/era5/global_npy_files_preinterpolated/{year}/era5_preinterp_{ts}.npy
        float32 [651652, 3], columns = u10, v10, msl (as in the source NPZ)

Usage (from repo root):
    python modules/preprocess/convert_mmap_to_npy.py --kind wave --workers 48
    python modules/preprocess/convert_mmap_to_npy.py --kind era5 --workers 48
    python modules/preprocess/convert_mmap_to_npy.py --kind era5_f16 --workers 48
"""
import argparse
import json
import os
import time
from multiprocessing import Process

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def get_config(kind):
    if kind == 'wave':
        return dict(
            cache_path=os.path.join(REPO_ROOT, 'data/global_hindcast/global_wave_all_frames.npy'),
            idx_path=os.path.join(REPO_ROOT, 'data/global_hindcast/wave_mmap_ts_index.npz'),
            out_dir=os.path.join(REPO_ROOT, 'data/global_hindcast/npy_files'),
            filename=lambda ts: f"global_hindcast_{ts[:8]}_{ts[9:]}.npy",
        )
    elif kind == 'era5':
        return dict(
            cache_path=os.path.join(REPO_ROOT, 'data/era5/era5_global_all_frames.npy'),
            idx_path=os.path.join(REPO_ROOT, 'data/era5/era5_global_ts_index.npz'),
            out_dir=os.path.join(REPO_ROOT, 'data/era5/global_npy_files_preinterpolated'),
            filename=lambda ts: f"era5_preinterp_{ts}.npy",
        )
    elif kind == 'era5_f16':
        # float16 variant: u10/v10 stored directly, msl stored as (msl - 1e5)
        # because raw msl (~1e5 Pa) overflows float16 (max 65504). The dataset
        # loader (era5_file_format='npy16') adds the offset back after loading.
        return dict(
            cache_path=os.path.join(REPO_ROOT, 'data/era5/era5_global_all_frames.npy'),
            idx_path=os.path.join(REPO_ROOT, 'data/era5/era5_global_ts_index.npz'),
            out_dir=os.path.join(REPO_ROOT, 'data/era5/global_npy_files_preinterpolated_f16'),
            filename=lambda ts: f"era5_preinterp_{ts}.npy",
        )
    raise ValueError(f"unknown kind: {kind}")


def worker(kind, cache_path, rows, out_dir, filename_fn, report_path):
    """Write rows = [(row_idx, timestamp), ...] as individual NPY files."""
    mm = np.load(cache_path, mmap_mode='r')
    n_cols = mm.shape[-1]
    # Expected on-disk size: npy header (128B for v1.0 with this shape) + data.
    # Used to detect truncated files left by killed runs.
    elem_size = 2 if kind == 'era5_f16' else 4
    expected_size = 128 + mm.shape[1] * n_cols * elem_size
    failed = []
    t0 = time.time()
    for n, (idx, ts) in enumerate(rows):
        out_path = os.path.join(out_dir, ts[:4], filename_fn(ts))
        try:
            # idempotent: skip only files that exist AND have the exact expected size
            if not (os.path.exists(out_path) and os.path.getsize(out_path) == expected_size):
                if kind == 'era5_f16':
                    arr = mm[idx].astype(np.float32)
                    arr[:, 2] -= 100000.0  # msl offset: raw msl overflows float16
                    arr = arr.astype(np.float16)
                else:
                    arr = mm[idx]
                # atomic write: temp file + rename, so a killed run never
                # leaves a half-written file behind under the final name
                tmp_path = out_path + '.tmp'
                with open(tmp_path, 'wb') as fh:
                    np.save(fh, arr)
                os.replace(tmp_path, out_path)
        except Exception as e:
            failed.append({'idx': int(idx), 'ts': ts, 'error': str(e)})
        if n % 500 == 499:
            rate = (n + 1) / (time.time() - t0)
            print(f"  [{rows[0][1]}..] {n + 1}/{len(rows)} files "
                  f"({rate:.1f} files/s, failed={len(failed)})", flush=True)
    with open(report_path, 'w') as fh:
        json.dump({'failed': failed}, fh)
    print(f"  [slice @{rows[0][1]}] done, {len(failed)} failures", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['era5', 'era5_f16', 'wave'], required=True)
    ap.add_argument('--workers', type=int, default=48)
    args = ap.parse_args()

    cfg = get_config(args.kind)
    idx_data = np.load(cfg['idx_path'], allow_pickle=True)
    rows = [(int(i), str(ts)) for ts, i in zip(idx_data['timestamps'], idx_data['indices'])]
    idx_data.close()
    n = len(rows)
    print(f"kind={args.kind}: {n} timestamps -> {cfg['out_dir']}", flush=True)

    report_dir = cfg['out_dir'] + '.reports'
    os.makedirs(report_dir, exist_ok=True)
    # np.save does not create parent dirs; create all year dirs upfront
    for _, ts in rows:
        os.makedirs(os.path.join(cfg['out_dir'], ts[:4]), exist_ok=True)

    # Contiguous slices keep cache reads mostly sequential per worker.
    bounds = np.linspace(0, n, args.workers + 1).astype(int)
    procs = []
    t0 = time.time()
    for w in range(args.workers):
        s, e = int(bounds[w]), int(bounds[w + 1])
        if s == e:
            continue
        rp = os.path.join(report_dir, f'slice_{s}.json')
        p = Process(target=worker,
                    args=(args.kind, cfg['cache_path'], rows[s:e], cfg['out_dir'], cfg['filename'], rp))
        p.start()
        procs.append((p, rp))

    failed = []
    for p, rp in procs:
        p.join()
        if p.exitcode != 0:
            raise SystemExit(f"worker exited with code {p.exitcode}")
        with open(rp) as fh:
            failed.extend(json.load(fh)['failed'])

    elapsed = time.time() - t0
    print(f"all workers finished in {elapsed/60:.1f} min, {len(failed)} failures", flush=True)
    if failed:
        fail_path = cfg['out_dir'] + '.failed.json'
        with open(fail_path, 'w') as fh:
            json.dump(failed, fh, indent=2)
        print(f"WARNING: {len(failed)} failures, details in {fail_path}", flush=True)

    # Verify: file count + random spot-check against the cache
    import glob
    files = glob.glob(os.path.join(cfg['out_dir'], '**', '*.npy'), recursive=True)
    print(f"file count: {len(files)} (expected {n - len(failed)})", flush=True)
    mm = np.load(cfg['cache_path'], mmap_mode='r')
    import random
    random.seed(0)
    for idx, ts in random.sample(rows, 5):
        out_path = os.path.join(cfg['out_dir'], ts[:4], cfg['filename'](ts))
        if not os.path.exists(out_path):
            print(f"  verify {ts}: MISSING", flush=True)
            continue
        loaded = np.load(out_path)
        if args.kind == 'era5_f16':
            ref = mm[idx].astype(np.float32)
            restored = loaded.astype(np.float32)
            restored[:, 2] += 100000.0
            max_err = np.abs(restored - ref).max()
            ok = max_err <= 2.0  # f16 quantization bound for this data
            print(f"  verify {ts}: max_err={max_err:.4f} {'OK' if ok else 'TOO LARGE'}", flush=True)
        else:
            ok = np.allclose(loaded, mm[idx])
            print(f"  verify {ts}: {'OK' if ok else 'MISMATCH'}", flush=True)
    print("done", flush=True)


if __name__ == '__main__':
    main()
