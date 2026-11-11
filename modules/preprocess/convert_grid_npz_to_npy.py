"""
Convert global 0.25° grid data from NPZ to per-sample NPY format.

Wave: interpolated_grid/{year}/global_hindcast_{date}_{hour}.npz
  → interpolated_grid_npy/{year}/global_hindcast_{date}_{hour}.npy  shape (3, 720, 1440)

ERA5: global_npz_files/{year}/era5_{date}T{hour}.npz
  → global_npy_files/{year}/era5_{date}T{hour}.npy  shape (3, 720, 1440)
"""

import os
import sys
import glob
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np

GRID_SHAPE = (720, 1440)
ERA5_VARIABLES = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'mean_sea_level_pressure',
]


def convert_wave_file(src_path, src_dir, dst_dir):
    rel = os.path.relpath(src_path, src_dir)
    dst_path = os.path.join(dst_dir, rel.replace('.npz', '.npy'))
    if os.path.exists(dst_path):
        return 0
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    try:
        data = np.load(src_path)['data'][:, :GRID_SHAPE[0], :GRID_SHAPE[1]]
        np.save(dst_path, data.astype(np.float32))
        return 1
    except Exception as e:
        print(f"ERROR {src_path}: {e}", file=sys.stderr)
        return -1


def convert_era5_file(src_path, src_dir, dst_dir):
    rel = os.path.relpath(src_path, src_dir)
    dst_path = os.path.join(dst_dir, rel.replace('.npz', '.npy'))
    if os.path.exists(dst_path):
        return 0
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    try:
        npz = np.load(src_path)
        arrays = [npz[v][:GRID_SHAPE[0], :GRID_SHAPE[1]] for v in ERA5_VARIABLES]
        data = np.stack(arrays, axis=0).astype(np.float32)
        np.save(dst_path, data)
        return 1
    except Exception as e:
        print(f"ERROR {src_path}: {e}", file=sys.stderr)
        return -1


def convert_dataset(src_dir, dst_dir, pattern, convert_fn, workers):
    files = sorted(glob.glob(os.path.join(src_dir, '**', pattern), recursive=True))
    print(f"  Found {len(files)} files in {src_dir}")
    if not files:
        return

    fn = partial(convert_fn, src_dir=src_dir, dst_dir=dst_dir)
    converted, skipped, errors = 0, 0, 0
    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(fn, files, chunksize=64)):
            if result == 1:
                converted += 1
            elif result == 0:
                skipped += 1
            else:
                errors += 1
            if (i + 1) % 10000 == 0:
                print(f"  Progress: {i+1}/{len(files)} (converted={converted}, skipped={skipped}, errors={errors})")

    print(f"  Done: converted={converted}, skipped={skipped}, errors={errors}")


def parse_args(argv=None):
    """Parse CLI arguments (shared with load_data.py / post_process.py wrappers)."""
    return _build_parser().parse_args(argv)


def _build_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave_src', default='data/global_hindcast/interpolated_grid')
    parser.add_argument('--wave_dst', default='data/global_hindcast/interpolated_grid_npy')
    parser.add_argument('--era5_src', default='data/era5/global_npz_files')
    parser.add_argument('--era5_dst', default='data/era5/global_npy_files')
    parser.add_argument('--workers', type=int, default=min(32, cpu_count()))
    parser.add_argument('--skip_wave', action='store_true')
    parser.add_argument('--skip_era5', action='store_true')
    return parser


def run(args):
    if not args.skip_wave:
        print("Converting wave data (NPZ → NPY)...")
        convert_dataset(args.wave_src, args.wave_dst, 'global_hindcast_*.npz',
                        convert_wave_file, args.workers)

    if not args.skip_era5:
        print("Converting ERA5 data (NPZ → NPY)...")
        convert_dataset(args.era5_src, args.era5_dst, 'era5_*.npz',
                        convert_era5_file, args.workers)

    print("All done.")


def main():
    run(parse_args())

if __name__ == '__main__':
    main()
