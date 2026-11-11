"""
计算插值网格数据的统计量（用于归一化）。

- Wave Hs: log1p mean/std (z-score 归一化), min, max
- ERA5 u10, v10: min, max, mean, std (min-max 归一化)

基于 Welford 在线算法，支持大数据集的随机采样。

Usage:
    python -m modules.datasets.compute_interp_stats
"""

import json
import os
import random
import time
import argparse

import numpy as np
from tqdm import tqdm


def compute_wave_stats(wave_dir: str, sample_ratio: float = 0.05, seed: int = 42) -> dict:
    """从采样的 0.5° wave .npy 文件计算 Hs 统计量。

    使用 Welford 在线算法计算 log1p 变换后的均值和标准差。
    """
    random.seed(seed)
    np.random.seed(seed)

    all_files = []
    for year_dir in sorted(os.listdir(wave_dir)):
        year_path = os.path.join(wave_dir, year_dir)
        if os.path.isdir(year_path) and year_dir.isdigit():
            for fname in os.listdir(year_path):
                if fname.endswith('.npy'):
                    all_files.append(os.path.join(year_path, fname))

    n_total = len(all_files)
    n_sample = max(int(n_total * sample_ratio), 100)
    files = random.sample(all_files, n_sample)
    print(f"Wave stats: sampling {n_sample:,} / {n_total:,} files")

    # Welford for log1p values
    count = 0
    log_mean = 0.0
    log_m2 = 0.0
    raw_min = float('inf')
    raw_max = float('-inf')

    for fpath in tqdm(files, desc='Wave stats'):
        data = np.load(fpath)  # (360, 720)
        ocean = data[~np.isnan(data)]
        if len(ocean) == 0:
            continue
        ocean = np.maximum(ocean, 0.0)  # clip negative

        raw_min = min(raw_min, float(ocean.min()))
        raw_max = max(raw_max, float(ocean.max()))

        log_vals = np.log1p(ocean).astype(np.float64)
        for x in log_vals:
            count += 1
            delta = x - log_mean
            log_mean += delta / count
            delta2 = x - log_mean
            log_m2 += delta * delta2

    log_std = np.sqrt(log_m2 / count) if count > 0 else 0.0
    stats = {
        'hs_log_mean': float(log_mean),
        'hs_log_std': float(log_std),
        'hs_min': float(raw_min),
        'hs_max': float(raw_max),
        'n_samples': n_sample,
        'n_ocean_pixels': count,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    print(f"  log_mean={log_mean:.6f}, log_std={log_std:.6f}")
    print(f"  raw min={raw_min:.4f}, max={raw_max:.4f}")
    print(f"  ocean pixels sampled: {count:,}")
    return stats


def compute_era5_stats(era5_dir: str, sample_ratio: float = 0.05, seed: int = 42) -> dict:
    """从采样的 0.5° ERA5 .npy 文件计算 u10/v10 统计量。

    使用 Welford 在线算法计算均值、标准差、最小值和最大值。
    """
    random.seed(seed)
    np.random.seed(seed)

    all_files = []
    for year_dir in sorted(os.listdir(era5_dir)):
        year_path = os.path.join(era5_dir, year_dir)
        if os.path.isdir(year_path) and year_dir.isdigit():
            for fname in os.listdir(year_path):
                if fname.endswith('.npy'):
                    all_files.append(os.path.join(year_path, fname))

    n_total = len(all_files)
    n_sample = max(int(n_total * sample_ratio), 100)
    files = random.sample(all_files, n_sample)
    print(f"ERA5 stats: sampling {n_sample:,} / {n_total:,} files")

    var_names = ['u10', 'v10']  # indices 0, 1
    counts = [0, 0]
    means = [0.0, 0.0]
    m2s = [0.0, 0.0]
    mins = [float('inf'), float('inf')]
    maxs = [float('-inf'), float('-inf')]

    for fpath in tqdm(files, desc='ERA5 stats'):
        data = np.load(fpath)  # (2, 360, 720)
        for vi in range(2):
            vals = data[vi].ravel().astype(np.float64)
            mins[vi] = min(mins[vi], float(vals.min()))
            maxs[vi] = max(maxs[vi], float(vals.max()))
            for x in vals:
                counts[vi] += 1
                delta = x - means[vi]
                means[vi] += delta / counts[vi]
                delta2 = x - means[vi]
                m2s[vi] += delta * delta2

    stats = {'variables': []}
    for vi, name in enumerate(var_names):
        std = np.sqrt(m2s[vi] / counts[vi]) if counts[vi] > 0 else 0.0
        s = {
            'variable': name,
            'min_value': float(mins[vi]),
            'max_value': float(maxs[vi]),
            'mean': float(means[vi]),
            'std': float(std),
        }
        stats['variables'].append(s)
        print(f"  {name}: min={mins[vi]:.4f}, max={maxs[vi]:.4f}, "
              f"mean={means[vi]:.4f}, std={std:.4f}")

    stats['n_samples'] = n_sample
    stats['n_pixels_per_var'] = counts[0]
    stats['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute normalization stats for interpolated grid data"
    )
    parser.add_argument('--wave-dir', default='./data/global_hindcast/interpolated_grid_0.5deg')
    parser.add_argument('--era5-dir', default='./data/era5/global_0.5deg')
    parser.add_argument('--wave-out', default='./data/global_hindcast/statistics_0.5deg.json')
    parser.add_argument('--era5-out', default='./data/era5/era5_statistics_global_0.5deg.json')
    parser.add_argument('--sample-ratio', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Wave stats
    print("=" * 60)
    print("Computing wave Hs statistics...")
    t0 = time.time()
    wave_stats = compute_wave_stats(args.wave_dir, args.sample_ratio, args.seed)
    with open(args.wave_out, 'w') as f:
        json.dump(wave_stats, f, indent=2)
    print(f"Saved to {args.wave_out} ({time.time() - t0:.1f}s)")

    # ERA5 stats
    print("\n" + "=" * 60)
    print("Computing ERA5 u10/v10 statistics...")
    t0 = time.time()
    era5_stats = compute_era5_stats(args.era5_dir, args.sample_ratio, args.seed)
    with open(args.era5_out, 'w') as f:
        json.dump(era5_stats, f, indent=2)
    print(f"Saved to {args.era5_out} ({time.time() - t0:.1f}s)")

    print("\nDone.")


if __name__ == '__main__':
    main()
