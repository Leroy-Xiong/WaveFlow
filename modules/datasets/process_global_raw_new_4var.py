"""
Process raw_new/ daily NetCDF files into 4-variable (hs/tp/dm/elev) NPZ + statistics.

4-variable variant of process_global_raw_new.py: adds tp (raw T10, the spectral
energy period Tm-1,0 — the period variable used by the EurOtop overtopping
discharge formulation), and renames MWD->dm, SSH->elev to match the HK
4-variable pipeline (NormalizeWaveData keys: tp_log_*, elev_*).

NOTE: this "tp" is Tm-1,0, NOT the HK dataset's tp (WWM TPP peak period).

Usage:
    # Sample: process a single day
    python -m modules.datasets.process_global_raw_new_4var convert --start 20000101 --end 20000101

    # Full run with multiprocessing
    python -m modules.datasets.process_global_raw_new_4var convert --workers 48

    # Statistics (vectorized accumulation, safe for large n_samples)
    python -m modules.datasets.process_global_raw_new_4var stats --n-samples 5000
"""

import argparse
import json
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
OUTPUT_DIR = Path("data/global_hindcast/npz_files_4var")
STATS_OUTPUT = Path("data/global_hindcast/statistics_4var.json")

# Variables to extract: (raw_new_name, output_key)
VARIABLES = [
    ("SWH", "hs"),
    ("T10", "tp"),   # energy period Tm-1,0 (EurOtop overtopping period)
    ("MWD", "dm"),
    ("SSH", "elev"),
]

_FILL = 1e30  # treat anything above this as fill/missing


# ---------------------------------------------------------------------------
# NPZ conversion
# ---------------------------------------------------------------------------

def process_file(args: tuple) -> dict:
    """Process a single daily NC file into per-timestep NPZ files.

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
        bad = False
        for nc_var, out_key in VARIABLES:
            vals = ds[nc_var].isel(time=t_idx).values.astype(np.float32)
            vals = np.where(np.isfinite(vals) & (np.abs(vals) < _FILL), vals, np.nan).astype(np.float32)
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
    """Convert raw_new/ daily NC files to per-timestep 4-variable NPZ."""
    all_files = sorted(RAW_NEW_DIR.glob("ERA5_schismwwm_final_*.nc"))

    if start_date:
        all_files = [f for f in all_files if f.stem >= f"ERA5_schismwwm_final_{start_date}"]
    if end_date:
        all_files = [f for f in all_files if f.stem <= f"ERA5_schismwwm_final_{end_date}"]

    n_files = len(all_files)
    if n_files == 0:
        print("No files to process.")
        return {}

    print(f"Processing {n_files:,} daily files -> ~{n_files * 8:,} NPZ files")
    print(f"Variables: {[f'{a}->{b}' for a, b in VARIABLES]}")
    print(f"Workers: {workers}")
    print(f"Output: {OUTPUT_DIR}/")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(str(f), OUTPUT_DIR, overwrite) for f in all_files]
    totals = {"written": 0, "skipped": 0, "errors": 0}
    t0 = time.time()

    if workers <= 1:
        for task in tqdm(tasks, desc="Converting", unit="file"):
            r = process_file(task)
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
# Statistics computation (vectorized)
# ---------------------------------------------------------------------------

def compute_multivar_statistics(npz_dir: str | None = None,
                                n_samples: int = 5000,
                                seed: int = 42) -> dict:
    """Compute per-variable normalization statistics from sampled NPZ files.

    hs/tp: log1p mean/std (z-score normalization in log space).
    elev: mean/std (z-score normalization).
    dm: no stats needed (sin/cos decomposition).
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

    acc = {k: {"n": 0, "sum": 0.0, "sumsq": 0.0, "min": np.inf, "max": -np.inf}
           for k in ["hs_log", "tp_log", "elev"]}

    def update(key, vals):
        a = acc[key]
        a["n"] += vals.size
        a["sum"] += float(vals.sum())
        a["sumsq"] += float((vals ** 2).sum())
        a["min"] = min(a["min"], float(vals.min()))
        a["max"] = max(a["max"], float(vals.max()))

    def finalize(key):
        a = acc[key]
        mean = a["sum"] / a["n"]
        var = a["sumsq"] / a["n"] - mean ** 2
        return mean, float(np.sqrt(max(var, 0.0)))

    for fpath in tqdm(sampled, desc="Stats", unit="file"):
        try:
            data = np.load(fpath, allow_pickle=True)
        except Exception:
            continue
        if "hs" not in data or "tp" not in data:
            continue

        hs = data["hs"].ravel().astype(np.float64)
        update("hs_log", np.log1p(np.maximum(hs[np.isfinite(hs)], 0.0)))

        tp = data["tp"].ravel().astype(np.float64)
        update("tp_log", np.log1p(np.maximum(tp[np.isfinite(tp)], 0.0)))

        el = data["elev"].ravel().astype(np.float64)
        update("elev", el[np.isfinite(el)])

    hs_log_mean, hs_log_std = finalize("hs_log")
    tp_log_mean, tp_log_std = finalize("tp_log")
    elev_mean, elev_std = finalize("elev")

    stats = {
        "variables": ["hs", "tp", "dm", "elev"],
        "tp_source": "T10 (energy period Tm-1,0; EurOtop overtopping period) - NOT peak period",
        "hs": {"log_mean": hs_log_mean, "log_std": hs_log_std},
        "tp": {"log_mean": tp_log_mean, "log_std": tp_log_std},
        "dm": {"normalization": "sin_cos_decompose"},
        "elev": {"mean": elev_mean, "std": elev_std},
        "n_samples": n_sample,
        "n_total_files": n_total,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        # Flat keys that NormalizeWaveData expects
        "hs_log_mean": hs_log_mean,
        "hs_log_std": hs_log_std,
        "tp_log_mean": tp_log_mean,
        "tp_log_std": tp_log_std,
        "elev_mean": elev_mean,
        "elev_std": elev_std,
    }

    print(f"\n  hs:   log_mean={hs_log_mean:.6f}, log_std={hs_log_std:.6f}")
    print(f"  tp:   log_mean={tp_log_mean:.6f}, log_std={tp_log_std:.6f} "
          f"(median ~{np.expm1(tp_log_mean):.2f} s)")
    print(f"  elev: mean={elev_mean:.4f}, std={elev_std:.4f}")
    print(f"  dm:   sin_cos_decompose (no stats needed)")

    with open(STATS_OUTPUT, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {STATS_OUTPUT}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process raw_new/ into 4-variable (hs/tp/dm/elev) NPZ and statistics"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_conv = sub.add_parser("convert", help="Convert raw_new NC to per-timestep 4-var NPZ")
    p_conv.add_argument("--start", type=str, default=None, help="Start date YYYYMMDD (inclusive)")
    p_conv.add_argument("--end", type=str, default=None, help="End date YYYYMMDD (inclusive)")
    p_conv.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    p_conv.add_argument("--overwrite", action="store_true", help="Overwrite existing NPZ files")

    p_stats = sub.add_parser("stats", help="Compute 4-variable statistics")
    p_stats.add_argument("--npz-dir", type=str, default=str(OUTPUT_DIR))
    p_stats.add_argument("--n-samples", type=int, default=5000)
    p_stats.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "convert":
        convert_raw_new_to_npz(start_date=args.start, end_date=args.end,
                               workers=args.workers, overwrite=args.overwrite)
    elif args.command == "stats":
        compute_multivar_statistics(npz_dir=args.npz_dir,
                                    n_samples=args.n_samples, seed=args.seed)


if __name__ == "__main__":
    main()
