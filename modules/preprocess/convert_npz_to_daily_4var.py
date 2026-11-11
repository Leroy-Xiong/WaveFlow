"""Pack per-timestamp NPZ files into per-day NPY files (4-var wave + ERA5).

NPZ-source variant of convert_npy_to_daily.py (which packs from NPY sources):

- kind=wave4var: data/global_hindcast/npz_files_4var/YYYY/global_hindcast_YYYYMMDD_HH.npz
    (keys hs/tp/dm/elev) -> npy_files_daily_4var/YYYY/global_hindcast_YYYYMMDD.npy
    float32 [8, N, 4], plus wave_daily_ts_index.npz and wave_daily_columns.json.
- kind=era5: data/era5/global_npz_files_preinterpolated/YYYY/era5_preinterp_YYYYMMDDTHH.npz
    (key era5_on_nodes [N,3]) -> global_npy_files_preinterpolated_daily/YYYY/era5_preinterp_YYYYMMDD.npy
    float32 [8, N, 3], plus era5_daily_ts_index.npz.

Each output file holds one day (8 slots at 3h spacing); missing slots are
NaN-filled and excluded from the timestamp index. Layout lets the dataset read
one timestamp with a single pread() at a computed offset.

Usage (from repo root):
    python modules/preprocess/convert_npz_to_daily_4var.py --kind wave4var --workers 48
    python modules/preprocess/convert_npz_to_daily_4var.py --kind era5 --workers 48
"""
import argparse
import glob
import json
import os
import time
from multiprocessing import Process

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

N_NODES = 651652
SLOTS_PER_DAY = 8  # 3h spacing: slot = hour // 3

WAVE4VAR_COLUMNS = ['hs', 'tp', 'dm', 'elev']


def get_config(kind):
    if kind == 'wave4var':
        return dict(
            src_glob=os.path.join(REPO_ROOT, 'data/global_hindcast/npz_files_4var', '*', 'global_hindcast_*.npz'),
            src_name=lambda ts: f"global_hindcast_{ts[:8]}_{ts[9:]}.npz",
            src_dir=os.path.join(REPO_ROOT, 'data/global_hindcast/npz_files_4var'),
            out_dir=os.path.join(REPO_ROOT, 'data/global_hindcast/npy_files_daily_4var'),
            out_name=lambda day: f"global_hindcast_{day}.npy",
            idx_name='wave_daily_ts_index.npz',
            n_vars=len(WAVE4VAR_COLUMNS),
        )
    elif kind == 'era5':
        return dict(
            src_glob=os.path.join(REPO_ROOT, 'data/era5/global_npz_files_preinterpolated', '*', 'era5_preinterp_*.npz'),
            src_name=lambda ts: f"era5_preinterp_{ts}.npz",
            src_dir=os.path.join(REPO_ROOT, 'data/era5/global_npz_files_preinterpolated'),
            out_dir=os.path.join(REPO_ROOT, 'data/era5/global_npy_files_preinterpolated_daily'),
            out_name=lambda day: f"era5_preinterp_{day}.npy",
            idx_name='era5_daily_ts_index.npz',
            n_vars=3,
        )
    raise ValueError(f"unknown kind: {kind}")


def ts_of_file(kind, path):
    b = os.path.basename(path)
    if kind == 'wave4var':
        # global_hindcast_YYYYMMDD_HH.npz -> YYYYMMDDTHH
        p = b.replace('global_hindcast_', '').replace('.npz', '').split('_')
        return p[0] + 'T' + p[1]
    # era5_preinterp_YYYYMMDDTHH.npz -> YYYYMMDDTHH
    return b.replace('era5_preinterp_', '').replace('.npz', '')


def load_src(kind, path, n_vars):
    """Load one source NPZ as [N, n_vars] float32."""
    data = np.load(path, allow_pickle=True)
    try:
        if kind == 'wave4var':
            arr = np.stack([data[c] for c in WAVE4VAR_COLUMNS], axis=-1)
        else:
            arr = data['era5_on_nodes']
    finally:
        data.close()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != (N_NODES, n_vars):
        raise ValueError(f"bad shape {arr.shape}: {path}")
    return arr


def worker(cfg, kind, days, report_path):
    """Pack days = {day: [(slot, ts), ...]} into per-day NPY files."""
    index_entries = []  # (ts, day, slot)
    failed = []
    n_nan = 0
    t0 = time.time()
    n_vars = cfg['n_vars']
    for n, (day, slots) in enumerate(sorted(days.items())):
        out_path = os.path.join(cfg['out_dir'], day[:4], cfg['out_name'](day))
        try:
            if os.path.exists(out_path):
                # idempotent resume: trust existing day files, rebuild index only
                for slot, ts in slots:
                    index_entries.append((ts, day, slot))
                continue
            arr = np.full((SLOTS_PER_DAY, N_NODES, n_vars), np.nan, dtype=np.float32)
            for slot, ts in slots:
                src = os.path.join(cfg['src_dir'], ts[:4], cfg['src_name'](ts))
                arr[slot] = load_src(kind, src, n_vars)
                index_entries.append((ts, day, slot))
            n_nan += SLOTS_PER_DAY - len(slots)
            tmp_path = out_path + '.tmp'
            with open(tmp_path, 'wb') as fh:
                np.save(fh, arr)
            os.replace(tmp_path, out_path)
        except Exception as e:
            failed.append({'day': day, 'error': str(e)})
        if n % 200 == 199:
            rate = (n + 1) / (time.time() - t0)
            print(f"  [{day}] {n + 1}/{len(days)} days ({rate:.1f} days/s, "
                  f"failed={len(failed)}, nan_slots={n_nan})", flush=True)
    with open(report_path, 'w') as fh:
        json.dump({'failed': failed, 'nan_slots': n_nan,
                   'index': index_entries}, fh)
    print(f"  [slice @{days and min(days)}] done: {len(days)} days, "
          f"{len(failed)} failures, {n_nan} NaN slots", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['wave4var', 'era5'], required=True)
    ap.add_argument('--workers', type=int, default=48)
    args = ap.parse_args()

    cfg = get_config(args.kind)
    files = sorted(glob.glob(cfg['src_glob']))
    print(f"kind={args.kind}: {len(files)} source files", flush=True)

    # Group timestamps by day
    days = {}
    for f in files:
        ts = ts_of_file(args.kind, f)
        day, hh = ts[:8], int(ts[9:11])
        assert hh % 3 == 0, f"unexpected hour: {ts}"
        days.setdefault(day, []).append((hh // 3, ts))
    print(f"{len(days)} days to pack -> {cfg['out_dir']}", flush=True)

    report_dir = cfg['out_dir'] + '.reports'
    os.makedirs(report_dir, exist_ok=True)
    for day in days:
        os.makedirs(os.path.join(cfg['out_dir'], day[:4]), exist_ok=True)

    # Partition days across workers
    day_list = sorted(days.keys())
    bounds = np.linspace(0, len(day_list), args.workers + 1).astype(int)
    procs = []
    t0 = time.time()
    for w in range(args.workers):
        s, e = int(bounds[w]), int(bounds[w + 1])
        if s == e:
            continue
        sub = {d: days[d] for d in day_list[s:e]}
        rp = os.path.join(report_dir, f'slice_{s}.json')
        p = Process(target=worker, args=(cfg, args.kind, sub, rp))
        p.start()
        procs.append((p, rp))

    failed, index, n_nan = [], [], 0
    for p, rp in procs:
        p.join()
        if p.exitcode != 0:
            raise SystemExit(f"worker exited with code {p.exitcode}")
        with open(rp) as fh:
            r = json.load(fh)
        failed.extend(r['failed'])
        index.extend(r['index'])
        n_nan += r['nan_slots']

    elapsed = time.time() - t0
    print(f"all workers finished in {elapsed/60:.1f} min, "
          f"{len(failed)} failures, {n_nan} NaN slots", flush=True)
    if failed:
        for f in failed[:5]:
            print(f"  FAILED: {f}", flush=True)
        raise SystemExit("some days failed; fix and rerun (idempotent)")

    # Save timestamp index: ts -> (day, slot)
    index.sort()
    idx_path = os.path.join(cfg['out_dir'], cfg['idx_name'])
    np.savez(idx_path,
             timestamps=np.array([t for t, _, _ in index]),
             days=np.array([d for _, d, _ in index]),
             slots=np.array([s for _, _, s in index], dtype=np.int32))
    print(f"saved index: {idx_path} ({len(index)} timestamps)", flush=True)

    if args.kind == 'wave4var':
        cols_path = os.path.join(cfg['out_dir'], 'wave_daily_columns.json')
        with open(cols_path, 'w') as fh:
            json.dump({'columns': WAVE4VAR_COLUMNS, 'n_nodes': N_NODES}, fh)
        print(f"saved columns metadata: {cols_path}", flush=True)

    # Spot check against source files
    import random
    random.seed(0)
    for ts, day, slot in random.sample(index, 5):
        src = os.path.join(cfg['src_dir'], ts[:4], cfg['src_name'](ts))
        ref = load_src(args.kind, src, cfg['n_vars'])
        packed = np.load(os.path.join(cfg['out_dir'], day[:4], cfg['out_name'](day)), mmap_mode='r')
        ok = np.allclose(np.asarray(packed[slot]), ref, equal_nan=True)
        print(f"  verify {ts} (day {day} slot {slot}): {'OK' if ok else 'MISMATCH'}", flush=True)
    print("done", flush=True)


if __name__ == '__main__':
    main()
