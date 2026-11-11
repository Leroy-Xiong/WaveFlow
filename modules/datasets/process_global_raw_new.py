"""
Process raw_new/ daily NetCDF files into multi-variable NPZ and statistics.

Converts 27,028 daily ERA5_schismwwm_final_YYYYMMDD.nc files (7 timesteps each)
directly into per-timestep NPZ files with selected variables.

Usage:
    # Sample: process a single day
    python -m modules.datasets.process_global_raw_new --start 20000101 --end 20000101

    # Full run with multiprocessing
    python -m modules.datasets.process_global_raw_new --workers 16
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------

RAW_NEW_DIR = Path("data/global_hindcast/raw_new")
OUTPUT_DIR = Path("data/global_hindcast/npz_files")
STATS_OUTPUT = Path("data/global_hindcast/statistics.json")

# Variables to extract: (raw_new_name, output_key)
VARIABLES = [
    ("SWH", "hs"),
    ("MWD", "mwd"),
    ("SSH", "ssh"),
]


# ---------------------------------------------------------------------------
# NPZ conversion
# ---------------------------------------------------------------------------

def process_file(args: tuple) -> dict:
    """Process a single daily NC file into 7 NPZ files (one per timestep).

    Args:
        args: (nc_path, output_dir, overwrite)

    Returns:
        dict with counts: {'written': int, 'skipped': int, 'errors': int}
    """
    nc_path, output_dir, overwrite = args
    nc_path = Path(nc_path)
    result = {"written": 0, "skipped": 0, "errors": 0}

    try:
        ds = xr.open_dataset(str(nc_path))
    except Exception as e:
        print(f"  ERROR opening {nc_path.name}: {e}")
        result["errors"] += 1
        return result

    date_str = nc_path.name.replace("ERA5_schismwwm_final_", "").replace(".nc", "")
    year = date_str[:4]

    year_dir = output_dir / year
    year_dir.mkdir(parents=True, exist_ok=True)

    n_timesteps = ds.sizes["time"]

    for t_idx in range(n_timesteps):
        # Build timestamp string: e.g. 20000101_03
        ts = ds.time.values[t_idx]
        hour = int(str(ts)[11:13]) if "T" in str(ts) else (t_idx * 3 + 3)
        timestamp = f"{date_str}_{hour:02d}"
        npz_path = year_dir / f"global_hindcast_{timestamp}.npz"

        if not overwrite and npz_path.exists():
            result["skipped"] += 1
            continue

        # Extract variables from this timestep
        var_data = {}
        for nc_var, out_key in VARIABLES:
            vals = ds[nc_var].isel(time=t_idx).values.astype(np.float32)
            var_data[out_key] = vals

        var_data["time"] = str(ts)

        try:
            np.savez_compressed(str(npz_path), **var_data)
            result["written"] += 1
        except Exception as e:
            print(f"  ERROR writing {npz_path.name}: {e}")
            result["errors"] += 1

    ds.close()
    return result


def convert_raw_new_to_npz(start_date: str | None = None,
                           end_date: str | None = None,
                           workers: int = 1,
                           overwrite: bool = False) -> dict:
    """Convert raw_new/ daily NC files to per-timestep multi-variable NPZ.

    Args:
        start_date: YYYYMMDD start (inclusive). None = all files.
        end_date: YYYYMMDD end (inclusive). None = all files.
        workers: Number of worker processes.
        overwrite: Overwrite existing NPZ files.

    Returns:
        dict with total counts.
    """
    all_files = sorted(RAW_NEW_DIR.glob("ERA5_schismwwm_final_*.nc"))

    if start_date:
        all_files = [f for f in all_files if f.stem >= f"ERA5_schismwwm_final_{start_date}"]
    if end_date:
        all_files = [f for f in all_files if f.stem <= f"ERA5_schismwwm_final_{end_date}"]

    n_files = len(all_files)
    if n_files == 0:
        print("No files to process.")
        return {}

    n_timesteps = n_files * 7
    print(f"Processing {n_files:,} daily files → ~{n_timesteps:,} NPZ files")
    print(f"Variables: {[v[1] for v in VARIABLES]}")
    print(f"Workers: {workers}")
    print(f"Output: {OUTPUT_DIR}/")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(str(f), OUTPUT_DIR, overwrite) for f in all_files]
    totals = {"written": 0, "skipped": 0, "errors": 0}
    t0 = time.time()

    if workers <= 1:
        for task in tqdm(tasks, desc="Converting", unit="file"):
            r = process_file(task)  # task is (nc_path, output_dir, overwrite)
            for k in totals:
                totals[k] += r.get(k, 0)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_file, t): t for t in tasks}
            with tqdm(total=len(tasks), desc="Converting", unit="file") as pbar:
                for future in as_completed(futures):
                    r = future.result()
                    for k in totals:
                        totals[k] += r[k]
                    pbar.update(1)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Written: {totals['written']:,}  |  Skipped: {totals['skipped']:,}  |  Errors: {totals['errors']:,}")

    return totals


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_multivar_statistics(npz_dir: str | None = None,
                                n_samples: int = 5000,
                                seed: int = 42) -> dict:
    """Compute per-variable normalization statistics from sampled NPZ files.

    For hs: log1p mean/std (z-score normalization).
    For ssh: mean/std (z-score normalization).
    For mwd: no stats needed (sin/cos decomposition).

    Args:
        npz_dir: Directory containing NPZ files (can have year subdirs).
        n_samples: Number of NPZ files to randomly sample.
        seed: Random seed for reproducibility.

    Returns:
        dict with per-variable statistics, saved to STATS_OUTPUT.
    """
    if npz_dir is None:
        npz_dir = str(OUTPUT_DIR)

    rng = np.random.RandomState(seed)
    all_files = sorted(Path(npz_dir).glob("**/global_hindcast_*.npz"))

    n_total = len(all_files)
    n_sample = min(n_samples, n_total)
    sampled = rng.choice(all_files, size=n_sample, replace=False)

    print(f"Computing statistics from {n_sample:,} / {n_total:,} NPZ files")
    print("=" * 60)

    # Accumulators (Welford's algorithm for streaming mean/variance)
    # hs: log1p space
    hs_log_count = 0
    hs_log_mean = 0.0
    hs_log_m2 = 0.0
    hs_raw_min = float("inf")
    hs_raw_max = float("-inf")

    # ssh: raw space
    ssh_count = 0
    ssh_mean = 0.0
    ssh_m2 = 0.0
    ssh_min = float("inf")
    ssh_max = float("-inf")

    for fpath in tqdm(sampled, desc="Stats", unit="file"):
        try:
            data = np.load(fpath, allow_pickle=True)
        except Exception:
            continue

        # Skip old-format files (WWM_1 only, lacking 'hs' key)
        if "hs" not in data:
            continue

        # hs: log1p stats
        hs_vals = data["hs"].ravel()
        hs_vals = hs_vals[~np.isnan(hs_vals)]
        hs_vals = np.maximum(hs_vals, 0.0)

        hs_raw_min = min(hs_raw_min, float(hs_vals.min()))
        hs_raw_max = max(hs_raw_max, float(hs_vals.max()))

        log_vals = np.log1p(hs_vals).astype(np.float64)
        for x in log_vals:
            hs_log_count += 1
            delta = x - hs_log_mean
            hs_log_mean += delta / hs_log_count
            delta2 = x - hs_log_mean
            hs_log_m2 += delta * delta2

        # ssh: raw stats
        ssh_vals = data["ssh"].ravel()
        ssh_vals = ssh_vals[~np.isnan(ssh_vals)]

        ssh_min = min(ssh_min, float(ssh_vals.min()))
        ssh_max = max(ssh_max, float(ssh_vals.max()))

        for x in ssh_vals.astype(np.float64):
            ssh_count += 1
            delta = x - ssh_mean
            ssh_mean += delta / ssh_count
            delta2 = x - ssh_mean
            ssh_m2 += delta * delta2

    hs_log_std = np.sqrt(hs_log_m2 / hs_log_count) if hs_log_count > 0 else 0.0
    hs_raw_mean = float(np.expm1(hs_log_mean))  # approximate
    ssh_std = np.sqrt(ssh_m2 / ssh_count) if ssh_count > 0 else 0.0

    stats = {
        "variables": ["hs", "mwd", "ssh"],
        "hs": {
            "log_mean": float(hs_log_mean),
            "log_std": float(hs_log_std),
            "min": float(hs_raw_min),
            "max": float(hs_raw_max),
        },
        "mwd": {
            "normalization": "sin_cos_decompose",
        },
        "ssh": {
            "mean": float(ssh_mean),
            "std": float(ssh_std),
            "min": float(ssh_min),
            "max": float(ssh_max),
        },
        "n_samples": n_sample,
        "n_total_files": n_total,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Also add flat keys that NormalizeWaveData expects:
    # hs_log_mean, hs_log_std, ssh_mean, ssh_std
    stats["hs_log_mean"] = stats["hs"]["log_mean"]
    stats["hs_log_std"] = stats["hs"]["log_std"]
    stats["ssh_mean"] = stats["ssh"]["mean"]
    stats["ssh_std"] = stats["ssh"]["std"]

    print(f"\n  hs:  log_mean={hs_log_mean:.6f}, log_std={hs_log_std:.6f}, raw_range=[{hs_raw_min:.2f}, {hs_raw_max:.2f}]")
    print(f"  ssh: mean={ssh_mean:.4f}, std={ssh_std:.4f}, range=[{ssh_min:.4f}, {ssh_max:.4f}]")
    print(f"  mwd: sin_cos_decompose (no stats needed)")

    with open(STATS_OUTPUT, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {STATS_OUTPUT}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process raw_new/ into multi-variable NPZ and statistics"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # convert
    p_conv = sub.add_parser("convert", help="Convert raw_new NC to per-timestep NPZ")
    p_conv.add_argument("--start", type=str, default=None,
                        help="Start date YYYYMMDD (inclusive)")
    p_conv.add_argument("--end", type=str, default=None,
                        help="End date YYYYMMDD (inclusive)")
    p_conv.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes (default: 1)")
    p_conv.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing NPZ files")

    # stats
    p_stats = sub.add_parser("stats", help="Compute multi-variable statistics")
    p_stats.add_argument("--npz-dir", type=str, default=str(OUTPUT_DIR),
                         help="Directory containing NPZ files")
    p_stats.add_argument("--n-samples", type=int, default=5000,
                         help="Number of NPZ files to sample (default: 5000)")
    p_stats.add_argument("--seed", type=int, default=42,
                         help="Random seed (default: 42)")

    # all (convert + stats)
    p_all = sub.add_parser("all", help="Run convert + stats")
    p_all.add_argument("--start", type=str, default=None)
    p_all.add_argument("--end", type=str, default=None)
    p_all.add_argument("--workers", type=int, default=1)
    p_all.add_argument("--overwrite", action="store_true")
    p_all.add_argument("--n-samples", type=int, default=5000)
    p_all.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "convert":
        convert_raw_new_to_npz(start_date=args.start, end_date=args.end,
                               workers=args.workers, overwrite=args.overwrite)
    elif args.command == "stats":
        compute_multivar_statistics(npz_dir=args.npz_dir,
                                    n_samples=args.n_samples, seed=args.seed)
    elif args.command == "all":
        convert_raw_new_to_npz(start_date=args.start, end_date=args.end,
                               workers=args.workers, overwrite=args.overwrite)
        compute_multivar_statistics(npz_dir=str(OUTPUT_DIR),
                                    n_samples=args.n_samples, seed=args.seed)


if __name__ == "__main__":
    main()
