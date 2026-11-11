"""Recompute ERA5 node-level statistics over the preinterpolated NPY corpus.

Full scan of data/era5/global_npy_files_preinterpolated/{year}/
era5_preinterp_YYYYMMDDTHH.npy ([651652, 3] float32, channels u10/v10/msl),
pooled min/max/mean/std per channel (ddof=0), then rewrite
data/era5/era5_statistics_global.json (previous file kept as .bak_*).

The historical file was aggregated on the raw 721x1440 ERA5 grid (incl.
unphysical outliers such as u10=-148 m/s) with stale HK metadata; this
script computes stats over the exact arrays training reads (ocean nodes).

    python modules/preprocess/compute_era5_preinterp_stats.py --workers 24
"""
import argparse
import glob
import json
import os
import shutil
import time
from multiprocessing import Process

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

SRC_DIR = os.path.join(REPO_ROOT, 'data/era5/global_npy_files_preinterpolated')
OUT_PATH = os.path.join(REPO_ROOT, 'data/era5/era5_statistics_global.json')

VAR_NAMES = ['10m_u_component_of_wind', '10m_v_component_of_wind',
             'mean_sea_level_pressure']

# Physical-plausibility envelope per channel: pooled min/max are taken over
# values INSIDE the envelope. The corpus contains a vanishing fraction of
# corrupted points (full-scan raw min u10 = -249.8 m/s, max v10 = +130.8 m/s);
# letting them set the min/max would compress the useful normalization range.
# Envelope bounds are physical records (ocean 10m wind <= ~100 m/s; MSL
# 870-1084 hPa) with margin, so real extremes are kept.
PHYS_ENVELOPE = [(-100.0, 100.0), (-100.0, 100.0), (85000.0, 110000.0)]


def scan_worker(files, report_path):
    acc = np.zeros((3, 5))  # per channel: n(clean), sum, sumsq, min, max
    acc[:, 3] = np.inf
    acc[:, 4] = -np.inf
    outliers = np.zeros(3)   # finite values outside the physical envelope
    failed = []
    t0 = time.time()
    for n, fp in enumerate(files):
        try:
            a = np.load(fp)  # [N, 3] float32
            for c in range(3):
                v = a[:, c]
                fin = np.isfinite(v)
                lo, hi = PHYS_ENVELOPE[c]
                clean = fin & (v >= lo) & (v <= hi)
                outliers[c] += int((fin & ~clean).sum())
                v = v[clean]
                acc[c, 0] += v.size
                acc[c, 1] += float(v.sum(dtype=np.float64))
                acc[c, 2] += float((v.astype(np.float64) ** 2).sum())
                acc[c, 3] = min(acc[c, 3], float(v.min()))
                acc[c, 4] = max(acc[c, 4], float(v.max()))
        except Exception as e:
            failed.append({'file': fp, 'error': str(e)})
        if n % 2000 == 1999:
            rate = (n + 1) / (time.time() - t0)
            print(f'  [{os.path.basename(fp)}] {n + 1}/{len(files)} '
                  f'({rate:.1f} files/s, failed={len(failed)})', flush=True)
    np.save(report_path + '.npy', acc)
    np.save(report_path + '_outliers.npy', outliers)
    with open(report_path + '.json', 'w') as fh:
        json.dump({'failed': failed, 'n_scanned': len(files)}, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=24)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SRC_DIR, '*', 'era5_preinterp_*.npy')))
    print(f'corpus: {len(files)} files')
    shards = [files[i::args.workers] for i in range(args.workers)]
    report_paths = [f'/tmp/era5stats_{i}' for i in range(args.workers)]
    procs = [Process(target=scan_worker, args=(s, r))
             for s, r in zip(shards, report_paths)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    total = np.zeros((3, 5))
    total[:, 3] = np.inf
    total[:, 4] = -np.inf
    total_outliers = np.zeros(3)
    n_failed = 0
    for r in report_paths:
        a = np.load(r + '.npy')
        # sum/sumsq/min/max merge; n merge
        total[:, 0] += a[:, 0]
        total[:, 1] += a[:, 1]
        total[:, 2] += a[:, 2]
        total[:, 3] = np.minimum(total[:, 3], a[:, 3])
        total[:, 4] = np.maximum(total[:, 4], a[:, 4])
        total_outliers += np.load(r + '_outliers.npy')
        n_failed += len(json.load(open(r + '.json'))['failed'])

    variables = []
    for c, name in enumerate(VAR_NAMES):
        n, s, ss, mn, mx = total[c]
        mean = s / n
        std = float(np.sqrt(max(ss / n - mean * mean, 0.0)))
        variables.append({'variable': name, 'min_value': float(mn),
                          'max_value': float(mx), 'mean_value': float(mean),
                          'std_value': std, 'count': int(n),
                          'outlier_count_outside_envelope': int(total_outliers[c])})
        print(f'  {name}: min={mn:.4f} max={mx:.4f} mean={mean:.6f} std={std:.4f} '
              f'(n={int(n):,}, outliers={int(total_outliers[c]):,})')

    # timestamps: min/max from filenames (era5_preinterp_YYYYMMDDTHH.npy)
    ts = [os.path.basename(f).replace('era5_preinterp_', '').replace('.npy', '')
          for f in files]
    stats = {
        'time_range': {'start': min(ts), 'end': max(ts)},
        'spatial_range': {'lon_range': [-180, 180], 'lat_range': [-90, 90],
                          'global': True,
                          'note': 'stats over preinterpolated ocean-node arrays '
                                  '[651652, 3] (u10, v10, msl)'},
        'sampling_info': {'total_files_in_range': len(files), 'sampled_files': len(files),
                          'sample_ratio': 1.0, 'successful_files': len(files) - n_failed,
                          'failed_files': n_failed, 'random_seed': None,
                          'scan': 'full corpus, pooled, ddof=0',
                          'physical_envelope': {VAR_NAMES[c]: list(PHYS_ENVELOPE[c])
                                                for c in range(3)},
                          'envelope_note': 'min/max/mean/std over values inside the '
                                           'physical envelope; outlier_count_outside_'
                                           'envelope per variable counts corrupted '
                                           'points excluded from the stats'},
        'variables': variables,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    if os.path.exists(OUT_PATH):
        bak = OUT_PATH + '.bak_' + time.strftime('%Y%m%d')
        if os.path.exists(bak):
            print(f'keeping existing backup (not overwritten): {bak}')
        else:
            shutil.copy2(OUT_PATH, bak)
            print(f'backed up previous file -> {bak}')
    with open(OUT_PATH, 'w') as fh:
        json.dump(stats, fh, indent=2)
    print(f'wrote {OUT_PATH} ({(time.time() - t0) / 60:.1f} min total, '
          f'{n_failed} failed files)')


if __name__ == '__main__':
    main()
