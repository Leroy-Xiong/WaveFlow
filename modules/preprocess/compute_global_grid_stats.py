"""
Compute normalization statistics for the global 0.25° regular grid data.

Outputs:
- data/global_hindcast/grid_statistics.json (hs log1p z-score stats)
- data/era5/era5_statistics_global_grid.json (ERA5 per-variable min/max)
- data/global_hindcast/interpolation_weights/land_mask_720x1440.npy (cropped mask)
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ERA5_VARIABLES = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'mean_sea_level_pressure',
]

GRID_SHAPE = (720, 1440)


def compute_wave_stats(wave_dir, land_mask, n_samples=5000, seed=42):
    print(f"Computing wave hs statistics from {wave_dir} ...")
    all_files = []
    for year in range(1959, 2021):
        year_dir = os.path.join(wave_dir, str(year))
        if os.path.isdir(year_dir):
            all_files.extend(sorted(glob.glob(os.path.join(year_dir, 'global_hindcast_*.npz'))))

    print(f"  Total train files: {len(all_files)}")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(all_files, size=min(n_samples, len(all_files)), replace=False)

    ocean_mask = ~land_mask
    hs_values = []
    for i, path in enumerate(sampled):
        if (i + 1) % 500 == 0:
            print(f"  Processing {i+1}/{len(sampled)} ...")
        data = np.load(path)['data']
        hs = data[0, :GRID_SHAPE[0], :GRID_SHAPE[1]]
        hs_ocean = hs[ocean_mask]
        hs_ocean = hs_ocean[~np.isnan(hs_ocean)]
        hs_ocean = np.log1p(np.maximum(hs_ocean, 0.0))
        hs_values.append(hs_ocean)

    all_hs = np.concatenate(hs_values)
    stats = {
        'hs_log_mean': float(np.mean(all_hs)),
        'hs_log_std': float(np.std(all_hs)),
        'hs': {
            'log_mean': float(np.mean(all_hs)),
            'log_std': float(np.std(all_hs)),
            'min': float(np.min(all_hs)),
            'max': float(np.max(all_hs)),
        },
        'variables': ['hs'],
        'n_samples': len(sampled),
        'n_total_files': len(all_files),
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    print(f"  hs_log_mean={stats['hs_log_mean']:.6f}, hs_log_std={stats['hs_log_std']:.6f}")
    return stats


def compute_era5_stats(era5_dir, land_mask, n_samples=5000, seed=42):
    print(f"Computing ERA5 statistics from {era5_dir} ...")
    all_files = []
    for year in range(1959, 2021):
        year_dir = os.path.join(era5_dir, str(year))
        if os.path.isdir(year_dir):
            all_files.extend(sorted(glob.glob(os.path.join(year_dir, 'era5_*.npz'))))

    print(f"  Total train files: {len(all_files)}")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(all_files, size=min(n_samples, len(all_files)), replace=False)

    ocean_mask = ~land_mask
    var_values = {v: [] for v in ERA5_VARIABLES}

    for i, path in enumerate(sampled):
        if (i + 1) % 500 == 0:
            print(f"  Processing {i+1}/{len(sampled)} ...")
        data = np.load(path)
        for var in ERA5_VARIABLES:
            arr = data[var][:GRID_SHAPE[0], :GRID_SHAPE[1]]
            vals = arr[ocean_mask]
            vals = vals[~np.isnan(vals)]
            var_values[var].append(vals)

    variables_stats = []
    for var in ERA5_VARIABLES:
        all_vals = np.concatenate(var_values[var])
        variables_stats.append({
            'variable': var,
            'min_value': float(np.min(all_vals)),
            'max_value': float(np.max(all_vals)),
            'mean_value': float(np.mean(all_vals)),
            'std_value': float(np.std(all_vals)),
            'count': int(len(all_vals)),
        })
        print(f"  {var}: min={variables_stats[-1]['min_value']:.4f}, max={variables_stats[-1]['max_value']:.4f}")

    stats = {
        'spatial_range': {'global': True, 'grid_shape': list(GRID_SHAPE)},
        'sampling_info': {
            'total_files_in_range': len(all_files),
            'sampled_files': len(sampled),
            'random_seed': seed,
        },
        'variables': variables_stats,
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    return stats


def parse_args(argv=None):
    """Parse CLI arguments (shared with load_data.py / post_process.py wrappers)."""
    return _build_parser().parse_args(argv)


def _build_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave_dir', default='data/global_hindcast/interpolated_grid')
    parser.add_argument('--era5_dir', default='data/era5/global_npz_files')
    parser.add_argument('--land_mask_file', default='data/global_hindcast/interpolation_weights/land_mask_t0.5.npy')
    parser.add_argument('--n_samples', type=int, default=5000)
    return parser


def run(args):
    land_mask_raw = np.load(args.land_mask_file)
    land_mask = land_mask_raw.reshape(721, 1440)[:GRID_SHAPE[0], :GRID_SHAPE[1]]

    mask_out = 'data/global_hindcast/interpolation_weights/land_mask_720x1440.npy'
    np.save(mask_out, land_mask)
    print(f"Saved cropped land mask: {mask_out} shape={land_mask.shape}")

    wave_stats = compute_wave_stats(args.wave_dir, land_mask, n_samples=args.n_samples)
    wave_out = 'data/global_hindcast/grid_statistics.json'
    with open(wave_out, 'w') as f:
        json.dump(wave_stats, f, indent=2)
    print(f"Saved wave stats: {wave_out}")

    era5_stats = compute_era5_stats(args.era5_dir, land_mask, n_samples=args.n_samples)
    era5_out = 'data/era5/era5_statistics_global_grid.json'
    with open(era5_out, 'w') as f:
        json.dump(era5_stats, f, indent=2)
    print(f"Saved ERA5 stats: {era5_out}")


def main():
    run(parse_args())

if __name__ == '__main__':
    main()
