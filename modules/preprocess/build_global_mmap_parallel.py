"""Parallel builder for monolithic global mmap caches.

Builds the same caches as build_era5_global_mmap.py / build_wave_mmap.py but
with multiple worker processes (the serial versions need hours; this finishes
in well under an hour on a many-core node).

Each worker owns a contiguous slice of rows and writes directly into the
shared memmap file, so no large arrays pass through IPC. The timestamp index
NPZ is only written after all rows are complete, so a half-built cache is
never picked up by WaveDataset (it requires both files to exist).

Usage (from repo root):
    python modules/preprocess/build_global_mmap_parallel.py --kind era5 --workers 48
    python modules/preprocess/build_global_mmap_parallel.py --kind wave --workers 48
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Process

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

N_NODES = 651652
N_VARS = 3


def get_config(kind):
    if kind == 'era5':
        # All years (train + test), matches build_era5_global_mmap.py
        prep_dir = os.path.join(REPO_ROOT, 'data/era5/global_npz_files_preinterpolated')
        cache_path = os.path.join(REPO_ROOT, 'data/era5/era5_global_all_frames.npy')
        idx_path = os.path.join(REPO_ROOT, 'data/era5/era5_global_ts_index.npz')
        import glob
        files = sorted(glob.glob(os.path.join(prep_dir, '**', 'era5_preinterp_*.npz'), recursive=True))
        pairs = [(f, os.path.basename(f).replace('era5_preinterp_', '').replace('.npz', ''))
                 for f in files]
    elif kind == 'wave':
        # Training years only (1959-2020), matches build_wave_mmap.py
        data_dir = os.path.join(REPO_ROOT, 'data/global_hindcast/npz_files')
        cache_path = os.path.join(REPO_ROOT, 'data/global_hindcast/global_wave_all_frames.npy')
        idx_path = os.path.join(REPO_ROOT, 'data/global_hindcast/wave_mmap_ts_index.npz')
        import glob
        files = sorted(glob.glob(os.path.join(data_dir, '**', 'global_hindcast_*.npz'), recursive=True))
        pairs = []
        for f in files:
            parts = os.path.basename(f).replace('global_hindcast_', '').replace('.npz', '').split('_')
            ts = parts[0] + 'T' + parts[1]
            if 1959 <= int(ts[:4]) <= 2020:
                pairs.append((f, ts))
    else:
        raise ValueError(f"unknown kind: {kind}")
    return pairs, cache_path, idx_path


def load_row(kind, file_path):
    """Load one NPZ file and return an [N_NODES, N_VARS] float32 array."""
    d = np.load(file_path, allow_pickle=True)
    try:
        if kind == 'era5':
            row = d['era5_on_nodes'].astype(np.float32)
        else:
            # Column order hs, mwd, ssh matches build_wave_mmap.py; the dataset's
            # NormalizeWaveData picks channel i for variables[i], so hs must stay
            # at column 0 for hs-first configs.
            row = np.stack([d['hs'].astype(np.float32),
                            d['mwd'].astype(np.float32),
                            d['ssh'].astype(np.float32)], axis=-1)
    finally:
        d.close()
    if row.shape != (N_NODES, N_VARS):
        raise ValueError(f"{file_path}: unexpected shape {row.shape}")
    return row


def worker(kind, cache_path, pairs, start, end, report_path):
    """Write rows [start, end) of the cache. Retries each file once on failure."""
    mm = np.lib.format.open_memmap(cache_path, mode='r+', dtype=np.float32,
                                   shape=(len(pairs), N_NODES, N_VARS))
    failed = []
    t0 = time.time()
    for i in range(start, end):
        f, ts = pairs[i]
        try:
            mm[i] = load_row(kind, f)
        except Exception:
            try:
                time.sleep(1.0)
                mm[i] = load_row(kind, f)
            except Exception as e:
                failed.append({'idx': i, 'ts': ts, 'file': f, 'error': str(e)})
        if (i - start) % 500 == 499:
            mm.flush()
            done = i - start + 1
            rate = done / (time.time() - t0)
            print(f"  [slice {start}] {done}/{end - start} rows "
                  f"({rate:.1f} rows/s, failed={len(failed)})", flush=True)
    mm.flush()
    with open(report_path, 'w') as fh:
        json.dump({'start': start, 'end': end, 'failed': failed}, fh)
    print(f"  [slice {start}] done, {len(failed)} failures", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['era5', 'wave'], required=True)
    ap.add_argument('--workers', type=int, default=48)
    args = ap.parse_args()

    pairs, cache_path, idx_path = get_config(args.kind)
    n_ts = len(pairs)
    print(f"kind={args.kind}: {n_ts} files", flush=True)
    print(f"  first: {pairs[0][1]}, last: {pairs[-1][1]}", flush=True)
    if n_ts == 0:
        sys.exit("no files found")

    shape = (n_ts, N_NODES, N_VARS)
    size_gb = float(np.prod(shape)) * 4 / 1024**3
    print(f"cache: {cache_path} shape={shape} = {size_gb:.1f} GB", flush=True)

    if os.path.exists(idx_path):
        sys.exit(f"index {idx_path} already exists — cache appears built. "
                 f"Delete both files to rebuild.")

    # Create (or overwrite) the raw cache file; index is written only at the end.
    mm = np.lib.format.open_memmap(cache_path, mode='w+', dtype=np.float32, shape=shape)
    del mm

    # Partition rows into contiguous slices, one per worker.
    bounds = np.linspace(0, n_ts, args.workers + 1).astype(int)
    report_dir = cache_path + '.reports'
    os.makedirs(report_dir, exist_ok=True)
    procs = []
    t0 = time.time()
    for w in range(args.workers):
        start, end = int(bounds[w]), int(bounds[w + 1])
        if start == end:
            continue
        report_path = os.path.join(report_dir, f'slice_{start}.json')
        p = Process(target=worker,
                    args=(args.kind, cache_path, pairs, start, end, report_path))
        p.start()
        procs.append((p, report_path))

    failed = []
    for p, report_path in procs:
        p.join()
        if p.exitcode != 0:
            sys.exit(f"worker exited with code {p.exitcode}; cache is incomplete, "
                     f"delete {cache_path} and rerun")
        with open(report_path) as fh:
            failed.extend(json.load(fh)['failed'])

    elapsed = time.time() - t0
    print(f"all workers finished in {elapsed/60:.1f} min, {len(failed)} failures", flush=True)

    # Save index, excluding failed timestamps (their rows stay zero-filled and
    # are simply never referenced).
    failed_ts = {f['ts'] for f in failed}
    ts_list, idx_list = [], []
    for i, (_, ts) in enumerate(pairs):
        if ts not in failed_ts:
            ts_list.append(ts)
            idx_list.append(i)
    np.savez(idx_path, timestamps=np.array(ts_list), indices=np.array(idx_list))
    print(f"saved index: {idx_path} ({len(ts_list)} entries)", flush=True)

    if failed:
        fail_path = cache_path + '.failed.json'
        with open(fail_path, 'w') as fh:
            json.dump(failed, fh, indent=2)
        print(f"WARNING: {len(failed)} files could not be read, details in {fail_path}",
              flush=True)

    # Spot check
    mm = np.load(cache_path, mmap_mode='r')
    print(f"verify: row 0 col 0 range=[{mm[0, :, 0].min():.3f}, {mm[0, :, 0].max():.3f}]", flush=True)
    print(f"verify: row -1 col 0 range=[{mm[-1, :, 0].min():.3f}, {mm[-1, :, 0].max():.3f}]", flush=True)
    print("done", flush=True)


if __name__ == '__main__':
    main()
