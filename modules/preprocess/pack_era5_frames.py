"""
Pack 4 ERA5 frames (t, t-6h, t-12h, t-18h) into a single NPY file per wave timestamp.
Also re-save wave data as hs-only (channel 0) to reduce I/O by 3x.

Output:
- data/global_hindcast/interpolated_grid_hs/{year}/global_hindcast_{date}_{hour}.npy  shape (720, 1440)
- data/era5/global_npy_packed/{year}/era5_{date}T{hour}.npy  shape (4, 3, 720, 1440)
"""

import os
import sys
import glob
import argparse
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

import numpy as np

GRID_SHAPE = (720, 1440)


def pack_one_timestamp(wave_path, wave_src, wave_dst, era5_src, era5_dst):
    basename = os.path.basename(wave_path).replace('.npy', '')
    parts = basename.split('_')
    date_str, hour_str = parts[2], parts[3]
    ts = datetime.strptime(f"{date_str}{hour_str}", "%Y%m%d%H")
    year_str = ts.strftime('%Y')

    # Check output exists
    era5_out = os.path.join(era5_dst, year_str, f"era5_{date_str}T{hour_str}.npy")
    wave_out = os.path.join(wave_dst, year_str, f"global_hindcast_{date_str}_{hour_str}.npy")
    if os.path.exists(era5_out) and os.path.exists(wave_out):
        return 0

    # Load wave hs-only
    if not os.path.exists(wave_out):
        wave_data = np.load(wave_path)
        hs = wave_data[0].astype(np.float32)  # (720, 1440)
        os.makedirs(os.path.dirname(wave_out), exist_ok=True)
        np.save(wave_out, hs)

    # Pack 4 ERA5 frames
    if not os.path.exists(era5_out):
        frames = []
        for i in range(3, -1, -1):  # oldest first: t-18h, t-12h, t-6h, t
            frame_ts = ts - timedelta(hours=6 * i)
            frame_path = os.path.join(era5_src, frame_ts.strftime('%Y'),
                                      frame_ts.strftime('era5_%Y%m%dT%H.npy'))
            if not os.path.exists(frame_path):
                return -1
            frames.append(np.load(frame_path))
        era5_packed = np.stack(frames, axis=0).astype(np.float32)  # (4, 3, 720, 1440)
        os.makedirs(os.path.dirname(era5_out), exist_ok=True)
        np.save(era5_out, era5_packed)

    return 1


def parse_args(argv=None):
    """Parse CLI arguments (shared with load_data.py / post_process.py wrappers)."""
    return _build_parser().parse_args(argv)


def _build_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave_src', default='data/global_hindcast/interpolated_grid_npy')
    parser.add_argument('--wave_dst', default='data/global_hindcast/interpolated_grid_hs')
    parser.add_argument('--era5_src', default='data/era5/global_npy_files')
    parser.add_argument('--era5_dst', default='data/era5/global_npy_packed')
    parser.add_argument('--workers', type=int, default=min(32, cpu_count()))
    return parser


def run(args):
    wave_files = sorted(glob.glob(os.path.join(args.wave_src, '**', 'global_hindcast_*.npy'), recursive=True))
    print(f"Found {len(wave_files)} wave timestamps to pack")

    from functools import partial
    fn = partial(pack_one_timestamp,
                 wave_src=args.wave_src, wave_dst=args.wave_dst,
                 era5_src=args.era5_src, era5_dst=args.era5_dst)

    converted, skipped, errors = 0, 0, 0
    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(fn, wave_files, chunksize=32)):
            if result == 1:
                converted += 1
            elif result == 0:
                skipped += 1
            else:
                errors += 1
            if (i + 1) % 10000 == 0:
                print(f"  Progress: {i+1}/{len(wave_files)} (packed={converted}, skipped={skipped}, missing_era5={errors})")

    print(f"Done: packed={converted}, skipped={skipped}, missing_era5={errors}")


def main():
    run(parse_args())

if __name__ == '__main__':
    main()
