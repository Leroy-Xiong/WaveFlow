"""
Pack global 0.25° grid data into daily NPY files (8 timestamps per day).

Output:
- data/global_hindcast/interpolated_grid_hs_daily/{year}/wave_{YYYYMMDD}.npy  shape (8, 720, 1440) hs-only
- data/era5/global_npy_daily/{year}/era5_{YYYYMMDD}.npy  shape (8, 3, 720, 1440)

Hours per day: [00, 03, 06, 09, 12, 15, 18, 21] → index 0-7
"""

import os
import sys
import glob
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np

GRID_SHAPE = (720, 1440)
HOURS = [0, 3, 6, 9, 12, 15, 18, 21]


def pack_wave_day(year, day_file, wave_src, wave_dst):
    """Pack one day of wave data into (8, 720, 1440) hs-only."""
    basename = os.path.basename(day_file)  # e.g. "19590101" from file list
    year_str = str(year)
    date_str = basename

    dst_path = os.path.join(wave_dst, year_str, f"wave_{date_str}.npy")
    if os.path.exists(dst_path):
        return 0

    src_dir = os.path.join(wave_src, year_str)
    frames = []
    for h in HOURS:
        src = os.path.join(src_dir, f"global_hindcast_{date_str}_{h:02d}.npy")
        if not os.path.exists(src):
            return -1
        data = np.load(src)  # (3, 720, 1440)
        frames.append(data[0])  # hs only

    packed = np.stack(frames, axis=0).astype(np.float32)  # (8, 720, 1440)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    np.save(dst_path, packed)
    return 1


def pack_era5_day(year, day_file, era5_src, era5_dst):
    """Pack one day of ERA5 data into (8, 3, 720, 1440)."""
    date_str = os.path.basename(day_file)
    year_str = str(year)

    dst_path = os.path.join(era5_dst, year_str, f"era5_{date_str}.npy")
    if os.path.exists(dst_path):
        return 0

    src_dir = os.path.join(era5_src, year_str)
    frames = []
    for h in HOURS:
        src = os.path.join(src_dir, f"era5_{date_str}T{h:02d}.npy")
        if not os.path.exists(src):
            return -1
        frames.append(np.load(src))  # (3, 720, 1440)

    packed = np.stack(frames, axis=0).astype(np.float32)  # (8, 3, 720, 1440)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    np.save(dst_path, packed)
    return 1


def get_days_from_files(src_dir, pattern, year):
    """Extract unique dates from filenames."""
    year_dir = os.path.join(src_dir, str(year))
    if not os.path.isdir(year_dir):
        return []
    files = glob.glob(os.path.join(year_dir, pattern))
    dates = set()
    for f in files:
        basename = os.path.basename(f)
        # Extract YYYYMMDD from filename
        if 'global_hindcast_' in basename:
            date = basename.split('_')[2]
        elif 'era5_' in basename:
            date = basename.split('_')[1].split('T')[0]
        else:
            continue
        dates.add(date)
    return sorted(dates)


def parse_args(argv=None):
    """Parse CLI arguments (shared with load_data.py / post_process.py wrappers)."""
    return _build_parser().parse_args(argv)


def _build_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave_src', default='data/global_hindcast/interpolated_grid_npy')
    parser.add_argument('--wave_dst', default='data/global_hindcast/interpolated_grid_hs_daily')
    parser.add_argument('--era5_src', default='data/era5/global_npy_files')
    parser.add_argument('--era5_dst', default='data/era5/global_npy_daily')
    parser.add_argument('--workers', type=int, default=min(32, cpu_count()))
    parser.add_argument('--skip_wave', action='store_true')
    parser.add_argument('--skip_era5', action='store_true')
    return parser


def run(args):
    years = range(1959, 2024)

    if not args.skip_wave:
        print("Packing wave data into daily files...")
        tasks = []
        for year in years:
            days = get_days_from_files(args.wave_src, 'global_hindcast_*.npy', year)
            for day in days:
                tasks.append((year, day))
        print(f"  {len(tasks)} days to pack")

        fn = partial(pack_wave_day, wave_src=args.wave_src, wave_dst=args.wave_dst)
        converted, skipped, errors = 0, 0, 0
        with Pool(args.workers) as pool:
            for i, result in enumerate(pool.starmap(fn, tasks, chunksize=16)):
                if result == 1:
                    converted += 1
                elif result == 0:
                    skipped += 1
                else:
                    errors += 1
                if (i + 1) % 5000 == 0:
                    print(f"  Progress: {i+1}/{len(tasks)} (packed={converted}, skipped={skipped}, errors={errors})")
        print(f"  Wave done: packed={converted}, skipped={skipped}, errors={errors}")

    if not args.skip_era5:
        print("Packing ERA5 data into daily files...")
        tasks = []
        for year in years:
            days = get_days_from_files(args.era5_src, 'era5_*.npy', year)
            for day in days:
                tasks.append((year, day))
        print(f"  {len(tasks)} days to pack")

        fn = partial(pack_era5_day, era5_src=args.era5_src, era5_dst=args.era5_dst)
        converted, skipped, errors = 0, 0, 0
        with Pool(args.workers) as pool:
            for i, result in enumerate(pool.starmap(fn, tasks, chunksize=16)):
                if result == 1:
                    converted += 1
                elif result == 0:
                    skipped += 1
                else:
                    errors += 1
                if (i + 1) % 5000 == 0:
                    print(f"  Progress: {i+1}/{len(tasks)} (packed={converted}, skipped={skipped}, errors={errors})")
        print(f"  ERA5 done: packed={converted}, skipped={skipped}, errors={errors}")

    print("All done.")


def main():
    run(parse_args())

if __name__ == '__main__':
    main()
