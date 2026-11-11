#!/usr/bin/env python3
"""
Convert NetCDF (.nc) wave and ERA5 data files to NPZ format for faster data loading.

Background:
    Loading data from NFS-mounted NetCDF files via xr.load_dataset(engine='netcdf4')
    has massive metadata parsing overhead. For small files (~36 KB ERA5, ~498 KB wave),
    the NetCDF4 metadata overhead can exceed actual data read time by 10-50x over NFS.

    NPZ format uses np.load() which is just a raw NumPy array read — no metadata
    parsing, single contiguous read. This eliminates the dominant bottleneck in the
    data loading pipeline.

Usage:
    # Convert wave data (HK region)
    python preprocess/convert_to_npz.py \
        --mode wave \
        --src-dir ./data/hindcast/processed \
        --dst-dir ./data/hindcast/processed_npz

    # Convert ERA5 data
    python preprocess/convert_to_npz.py \
        --mode era5 \
        --src-dir ./data/era5/nc_files \
        --dst-dir ./data/era5/npz_files

    # Convert both with parallelism
    python preprocess/convert_to_npz.py \
        --mode all \
        --wave-src ./data/hindcast/processed \
        --wave-dst ./data/hindcast/processed_npz \
        --era5-src ./data/era5/nc_files \
        --era5-dst ./data/era5/npz_files \
        --num-workers 32

    # Dry-run to check what would be converted
    python preprocess/convert_to_npz.py --mode all --dry-run

After conversion, update configs/configs.yaml:
    file_format: 'npz'
    era5_file_format: 'npz'
    data_dir: './data/hindcast/processed_npz'
    era5_dir: './data/era5/npz_files'
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import xarray as xr
from tqdm import tqdm

# ERA5 variable names matching modules/utils.py ERA5_VARIABLE_NAMES
ERA5_VARIABLE_NAMES = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'mean_sea_level_pressure',
]

# ERA5 coordinate variable names to try (ERA5 uses 'longitude'/'latitude', some use 'lon'/'lat')
ERA5_LON_NAMES = ['longitude', 'lon']
ERA5_LAT_NAMES = ['latitude', 'lat']


def convert_wave_nc_to_npz(src_path: str, dst_dir: str) -> dict:
    """
    Convert a single wave NetCDF file to NPZ format.

    Preserves all data variables as float32 arrays.

    Returns:
        dict with keys: status, src_path, dst_path, size_mb, elapsed_s
    """
    t0 = time.time()
    try:
        # Load NC file
        ds = xr.load_dataset(src_path)
        fname = os.path.basename(src_path).replace('.nc', '.npz')

        # Preserve subdirectory structure relative to src
        rel_dir = os.path.relpath(os.path.dirname(src_path), start=os.path.dirname(src_path))
        # For flat directory structure, just use basename
        dst_path = os.path.join(dst_dir, fname)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Extract all data variables
        data_dict = {}
        for var_name in ds.data_vars:
            data_dict[var_name] = ds[var_name].values.astype(np.float32)

        ds.close()

        # Save as NPZ
        np.savez_compressed(dst_path, **data_dict)
        elapsed = time.time() - t0
        size_mb = os.path.getsize(dst_path) / (1024 * 1024)

        return {
            'status': 'ok',
            'src_path': src_path,
            'dst_path': dst_path,
            'size_mb': size_mb,
            'elapsed_s': elapsed,
            'num_vars': len(data_dict),
        }
    except Exception as e:
        return {
            'status': 'error',
            'src_path': src_path,
            'error': str(e),
            'elapsed_s': time.time() - t0,
        }


def convert_era5_nc_to_npz(src_path: str, dst_dir: str) -> dict:
    """
    Convert a single ERA5 NetCDF file to NPZ format.

    Extracts only the variables needed by the WaveDataset (u10, v10, mslp)
    and stores them using the same variable names as ERA5_VARIABLE_NAMES.

    Preserves the year-based subdirectory structure from the source.

    Returns:
        dict with keys: status, src_path, dst_path, size_mb, elapsed_s
    """
    t0 = time.time()
    try:
        ds = xr.load_dataset(src_path, engine='netcdf4')

        # Determine coordinate names
        lon_name = None
        lat_name = None
        for name in ERA5_LON_NAMES:
            if name in ds.dims or name in ds.coords:
                lon_name = name
                break
        for name in ERA5_LAT_NAMES:
            if name in ds.dims or name in ds.coords:
                lat_name = name
                break

        # Extract the 3 needed variables using ERA5_VARIABLE_NAMES
        data_dict = {}
        for var_name in ERA5_VARIABLE_NAMES:
            if var_name in ds:
                data_dict[var_name] = ds[var_name].values.astype(np.float32)
            else:
                # Try alternative short names
                short_names = {
                    '10m_u_component_of_wind': 'u10',
                    '10m_v_component_of_wind': 'v10',
                    'mean_sea_level_pressure': 'msl',
                }
                alt = short_names.get(var_name)
                if alt and alt in ds:
                    data_dict[var_name] = ds[alt].values.astype(np.float32)
                else:
                    ds.close()
                    return {
                        'status': 'error',
                        'src_path': src_path,
                        'error': f"Variable '{var_name}' not found in file. Available: {list(ds.data_vars)}",
                        'elapsed_s': time.time() - t0,
                    }

        # Also save coordinates if available (useful for interpolation)
        if lon_name and lat_name:
            data_dict['_longitude'] = ds[lon_name].values.astype(np.float32)
            data_dict['_latitude'] = ds[lat_name].values.astype(np.float32)

        ds.close()

        # Preserve year subdirectory structure
        fname = os.path.basename(src_path).replace('.nc', '.npz')
        src_parent = os.path.basename(os.path.dirname(src_path))
        # src_parent is typically a year like '2021'
        dst_path = os.path.join(dst_dir, src_parent, fname)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Save as NPZ
        np.savez_compressed(dst_path, **data_dict)
        elapsed = time.time() - t0
        size_mb = os.path.getsize(dst_path) / (1024 * 1024)

        return {
            'status': 'ok',
            'src_path': src_path,
            'dst_path': dst_path,
            'size_mb': size_mb,
            'elapsed_s': elapsed,
        }
    except Exception as e:
        return {
            'status': 'error',
            'src_path': src_path,
            'error': str(e),
            'elapsed_s': time.time() - t0,
        }


def find_nc_files(src_dir: str, pattern: str = "*.nc") -> list:
    """Find all NetCDF files recursively in the source directory."""
    files = sorted(glob.glob(os.path.join(src_dir, '**', pattern), recursive=True))
    return files


def run_conversion(files: list, convert_fn, dst_dir: str, num_workers: int,
                   desc: str, dry_run: bool = False) -> dict:
    """
    Convert a list of files using the given conversion function.

    Args:
        files: List of source file paths
        convert_fn: Function (src_path, dst_dir) -> dict
        dst_dir: Destination directory
        num_workers: Number of parallel workers
        desc: Description for progress bar
        dry_run: If True, only print what would be done

    Returns:
        dict with stats: total, ok, errors, skipped, total_size_gb, total_elapsed_s
    """
    if dry_run:
        print(f"\n[DRY RUN] Would convert {len(files)} files to {dst_dir}")
        print(f"  First 5 files:")
        for f in files[:5]:
            print(f"    {f}")
        if len(files) > 5:
            print(f"    ... and {len(files) - 5} more")
        return {'total': len(files), 'ok': 0, 'errors': 0, 'skipped': 0}

    stats = {'total': len(files), 'ok': 0, 'errors': 0, 'skipped': 0,
             'total_size_gb': 0.0, 'total_elapsed_s': 0.0}

    if num_workers > 1:
        max_workers = min(num_workers, os.cpu_count() or 1, 64)
        print(f"  Using {max_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(convert_fn, f, dst_dir): f for f in files}
            with tqdm(total=len(files), desc=desc) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result['status'] == 'ok':
                        stats['ok'] += 1
                        stats['total_size_gb'] += result.get('size_mb', 0) / 1024
                        stats['total_elapsed_s'] += result.get('elapsed_s', 0)
                    elif result['status'] == 'error':
                        stats['errors'] += 1
                        if stats['errors'] <= 5:
                            tqdm.write(f"  ERROR: {result['src_path']}: {result['error']}")
                    pbar.update(1)
    else:
        for f in tqdm(files, desc=desc):
            result = convert_fn(f, dst_dir)
            if result['status'] == 'ok':
                stats['ok'] += 1
                stats['total_size_gb'] += result.get('size_mb', 0) / 1024
                stats['total_elapsed_s'] += result.get('elapsed_s', 0)
            elif result['status'] == 'error':
                stats['errors'] += 1
                if stats['errors'] <= 5:
                    tqdm.write(f"  ERROR: {result['src_path']}: {result['error']}")

    return stats


def save_era5_coordinates(era5_src_dir: str, era5_dst_dir: str):
    """
    Extract ERA5 coordinate arrays from a sample file and save as
    era5_coordinates.npz for NPZ-format coordinate loading.
    """
    files = find_nc_files(era5_src_dir, "era5_*.nc")
    if not files:
        files = find_nc_files(era5_src_dir, "*.nc")
    if not files:
        print("  WARNING: No ERA5 files found to extract coordinates from")
        return

    # Use the first file to extract coordinates
    sample_file = files[0]
    print(f"  Extracting coordinates from: {sample_file}")
    try:
        ds = xr.load_dataset(sample_file, engine='netcdf4')

        lon_name = None
        lat_name = None
        for name in ERA5_LON_NAMES:
            if name in ds.dims or name in ds.coords:
                lon_name = name
                break
        for name in ERA5_LAT_NAMES:
            if name in ds.dims or name in ds.coords:
                lat_name = name
                break

        if lon_name and lat_name:
            coord_path = os.path.join(era5_dst_dir, 'era5_coordinates.npz')
            np.savez(
                coord_path,
                longitude=ds[lon_name].values.astype(np.float32),
                latitude=ds[lat_name].values.astype(np.float32),
            )
            print(f"  Saved coordinates to: {coord_path}")
        ds.close()
    except Exception as e:
        print(f"  WARNING: Failed to extract coordinates: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert NetCDF wave/ERA5 files to NPZ format for faster training data loading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show what would be converted (dry run)
  python preprocess/convert_to_npz.py --mode all --dry-run

  # Convert wave data only
  python preprocess/convert_to_npz.py --mode wave

  # Convert ERA5 data only with 16 workers
  python preprocess/convert_to_npz.py --mode era5 --num-workers 16

  # Convert everything with custom paths
  python preprocess/convert_to_npz.py --mode all \\
      --wave-src ./data/hindcast/processed \\
      --wave-dst ./data/hindcast/processed_npz \\
      --era5-src ./data/era5/nc_files \\
      --era5-dst ./data/era5/npz_files \\
      --num-workers 32
        """
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['wave', 'era5', 'all'],
                        help='What to convert: wave data, ERA5 data, or both')
    parser.add_argument('--wave-src', type=str, default='./data/hindcast/processed',
                        help='Source directory for wave NC files')
    parser.add_argument('--wave-dst', type=str, default='./data/hindcast/processed_npz',
                        help='Destination directory for wave NPZ files')
    parser.add_argument('--era5-src', type=str, default='./data/era5/nc_files',
                        help='Source directory for ERA5 NC files')
    parser.add_argument('--era5-dst', type=str, default='./data/era5/npz_files',
                        help='Destination directory for ERA5 NPZ files')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show what would be converted without doing it')
    args = parser.parse_args()

    print("=" * 70)
    print("NC to NPZ Converter for Flow & Diffusion Training Data")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode}")
    print(f"Parallel workers: {args.num_workers}")
    if args.dry_run:
        print("DRY RUN — no files will be written")
    print("=" * 70)

    overall_stats = {}

    if args.mode in ('wave', 'all'):
        print(f"\n{'─' * 50}")
        print(f"Converting WAVE data")
        print(f"  Source: {os.path.abspath(args.wave_src)}")
        print(f"  Dest:   {os.path.abspath(args.wave_dst)}")
        wave_files = find_nc_files(args.wave_src)
        print(f"  Found {len(wave_files)} NC files")
        if not wave_files:
            print("  WARNING: No wave files found!")
        else:
            stats = run_conversion(wave_files, convert_wave_nc_to_npz, args.wave_dst,
                                   args.num_workers, "Converting wave files", args.dry_run)
            overall_stats['wave'] = stats

    if args.mode in ('era5', 'all'):
        print(f"\n{'─' * 50}")
        print(f"Converting ERA5 data")
        print(f"  Source: {os.path.abspath(args.era5_src)}")
        print(f"  Dest:   {os.path.abspath(args.era5_dst)}")
        era5_files = find_nc_files(args.era5_src, "era5_*.nc")
        if not era5_files:
            era5_files = find_nc_files(args.era5_src, "*.nc")
        print(f"  Found {len(era5_files)} NC files")
        if not era5_files:
            print("  WARNING: No ERA5 files found!")
        else:
            stats = run_conversion(era5_files, convert_era5_nc_to_npz, args.era5_dst,
                                   args.num_workers, "Converting ERA5 files", args.dry_run)
            overall_stats['era5'] = stats

            # Save coordinate file for NPZ-format coordinate loading
            if not args.dry_run:
                save_era5_coordinates(args.era5_src, args.era5_dst)

    # Summary
    print(f"\n{'=' * 70}")
    print("Conversion Summary:")
    for data_type, stats in overall_stats.items():
        print(f"  {data_type}: {stats['ok']}/{stats['total']} converted, "
              f"{stats['errors']} errors, "
              f"{stats.get('total_size_gb', 0):.1f} GB total, "
              f"{stats.get('total_elapsed_s', 0):.1f}s elapsed")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNext step: Update configs/configs.yaml:")
    print(f"  file_format: 'npz'")
    print(f"  era5_file_format: 'npz'")
    print(f"  data_dir: '{args.wave_dst}'")
    print(f"  era5_dir: '{args.era5_dst}'")
    print("=" * 70)


if __name__ == '__main__':
    main()
