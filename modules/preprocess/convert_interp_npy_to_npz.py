"""
Convert interpolated grid .npy files to compressed .npz format to save disk space.

- Reads .npy, writes .npz (key='data'), then deletes original .npy
- Skips already-converted files (if .npz exists)
- Multiprocessing for speed

Usage:
    python modules/preprocess/convert_interp_npy_to_npz.py --dir data/global_hindcast/interpolated_grid_0.5deg --workers 16
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


def convert_file(args: tuple) -> dict:
    """Convert a single .npy file to .npz, then delete .npy.

    Args:
        args: (npy_path, overwrite)

    Returns:
        dict: {'converted': int, 'skipped': int, 'errors': int, 'deleted_mb': float}
    """
    npy_path, overwrite = args
    npy_path = Path(npy_path)
    npz_path = npy_path.with_suffix('.npz')
    result = {'converted': 0, 'skipped': 0, 'errors': 0, 'deleted_mb': 0.0}

    if not overwrite and npz_path.exists():
        # If npz already exists and npy still there, delete npy
        if npy_path.exists():
            npy_size = npy_path.stat().st_size / (1024 * 1024)
            npy_path.unlink()
            result['deleted_mb'] = npy_size
        result['skipped'] += 1
        return result

    if not npy_path.exists():
        result['skipped'] += 1
        return result

    try:
        data = np.load(npy_path)
        npy_size = npy_path.stat().st_size / (1024 * 1024)
        np.savez_compressed(str(npz_path), data=data)
        npy_path.unlink()
        result['converted'] += 1
        result['deleted_mb'] = npy_size
    except Exception as e:
        print(f"ERROR {npy_path.name}: {e}")
        result['errors'] += 1

    return result


def main():
    parser = argparse.ArgumentParser(description="Convert interpolated .npy → .npz")
    parser.add_argument('--dir', required=True, help='Root directory with year subdirs')
    parser.add_argument('--workers', type=int, default=1, help='Worker processes')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing .npz')
    parser.add_argument('--dry-run', action='store_true', help='Only count files, no conversion')
    args = parser.parse_args()

    root = Path(args.dir)
    npy_files = sorted(root.glob('*/*.npy'))
    n_total = len(npy_files)

    if args.dry_run:
        total_mb = sum(f.stat().st_size for f in npy_files) / (1024 * 1024)
        print(f"Found {n_total:,} .npy files in {root}")
        print(f"Total size: {total_mb / 1024:.1f} GB")
        print(f"Estimated npz size: {total_mb / 1024 / 1.5:.1f} GB")
        return

    tasks = [(str(f), args.overwrite) for f in npy_files]
    totals = {'converted': 0, 'skipped': 0, 'errors': 0, 'deleted_mb': 0.0}
    t0 = time.time()

    if args.workers <= 1:
        for task in tqdm(tasks, desc="Converting", unit="file"):
            r = convert_file(task)
            for k in totals:
                totals[k] += r[k]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            with tqdm(total=len(tasks), desc="Converting", unit="file") as pbar:
                futures = {pool.submit(convert_file, t): t for t in tasks}
                for future in as_completed(futures):
                    r = future.result()
                    for k in totals:
                        totals[k] += r[k]
                    pbar.update(1)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Converted: {totals['converted']:,}  |  Skipped: {totals['skipped']:,}  |  Errors: {totals['errors']:,}")
    print(f"  Deleted: {totals['deleted_mb']/1024:.1f} GB (.npy)")


if __name__ == '__main__':
    main()
