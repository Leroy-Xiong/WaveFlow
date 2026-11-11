"""
CMEMS MY (multi-year) L3 along-track SWH — download + process.

Downloads monthly multi-mission altimeter SWH from the MY product
(WAVE_GLO_PHY_SWH_L3_MY_014_005, merged dataset cci_obs-wave_glo_phy-swh_my_l3_PT1S-i)
for the 2021 test period, merges the per-mission NetCDF files into monthly
Parquet files compatible with modules/eval_satellite.py / modules/eval_satellite_hk.py
(columns: time, latitude, longitude, swh).

Usage (via post_process.py, recommended):
    python post_process.py satellite-download --year 2021 --action all

Standalone:
    python -m modules.datasets.load_cmems_my --year 2021 --action all --workers 4

Note: NRT product (WAVE_GLO_PHY_SWH_L3_NRT_014_001) only retains ~2 years;
2021 data is only available via the MY merged dataset.
"""

import os
import glob
import time
import shutil
import argparse
import tempfile
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

# MY merged multi-mission dataset (1991-08 ~ 2023-12, per-mission daily files)
DATASET_ID = "cci_obs-wave_glo_phy-swh_my_l3_PT1S-i"
# Default MY data root. Kept separate from the NRT dataset (./data/cmems_l3_swh)
# so MY and NRT parquet files never overwrite each other.
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "cmems_l3_swh_my",
)
# Thresholds to catch empty / truncated downloads
MIN_TOTAL_MB = 50.0     # a normal month is ~400 MB across missions
MIN_OBS = 1_000_000     # a normal month has ~5-6 M observations

_HEADER = "=" * 60


def _raw_month_dir(year, month):
    return os.path.join(DATA_DIR, "raw", str(year), f"month_{month:02d}")


def _processed_month_path(year, month):
    return os.path.join(DATA_DIR, "processed", str(year), f"month_{month:02d}.parquet")


def _month_end(year, month):
    """Last day of month as datetime string."""
    if month == 12:
        return f"{year}-12-31T23:59:59"
    nxt = pd.Timestamp(year=year, month=month + 1, day=1)
    last = (nxt - pd.Timedelta(days=1)).day
    return f"{year}-{month:02d}-{last:02d}T23:59:59"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_month(year, month, force=False, max_retries=3):
    """Download one month of MY multi-mission SWH.

    Produces one .nc per mission under
    data/cmems_l3_swh_my/raw/<year>/month_<MM>/
    """
    from copernicusmarine import subset

    raw_dir = _raw_month_dir(year, month)
    os.makedirs(raw_dir, exist_ok=True)

    existing = glob.glob(os.path.join(raw_dir, "*.nc"))
    if not force and existing:
        print(f"  [{year}-{month:02d}] {len(existing)} mission files "
              f"already present, skipping.")
        return True

    start_dt = f"{year}-{month:02d}-01T00:00:00"
    end_dt = _month_end(year, month)

    for attempt in range(1, max_retries + 1):
        tmp_dir = None
        try:
            print(f"  [{year}-{month:02d}] downloading "
                  f"(attempt {attempt}/{max_retries})...", flush=True)
            tmp_dir = tempfile.mkdtemp(prefix=f"cmems_my_{year}{month:02d}_")
            subset(
                dataset_id=DATASET_ID,
                variables=["swh"],
                minimum_longitude=-180, maximum_longitude=180,
                minimum_latitude=-90,  maximum_latitude=90,
                start_datetime=start_dt, end_datetime=end_dt,
                output_filename=f"my_{year}_{month:02d}.nc",
                output_directory=tmp_dir,
                overwrite=True,
                disable_progress_bar=True,
            )
            # Mission files land in <tmp_dir>/<name>/<mission>.nc
            out_subdir = os.path.join(tmp_dir, f"my_{year}_{month:02d}")
            nc_files = sorted(glob.glob(os.path.join(out_subdir, "*.nc")))

            total_mb = sum(os.path.getsize(f) for f in nc_files) / (1024 * 1024)
            if not nc_files:
                raise RuntimeError("no .nc files produced")
            if total_mb < MIN_TOTAL_MB:
                raise RuntimeError(
                    f"download too small ({total_mb:.1f} MB < {MIN_TOTAL_MB})"
                )

            for f in nc_files:
                shutil.move(f, os.path.join(raw_dir, os.path.basename(f)))
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"  [{year}-{month:02d}] done: {len(nc_files)} mission files, "
                  f"{total_mb:.0f} MB total.", flush=True)
            return True

        except Exception as e:
            print(f"  [{year}-{month:02d}] attempt {attempt} failed: {e}",
                  flush=True)
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if attempt < max_retries:
                wait = 30 * attempt
                print(f"  retrying in {wait}s...", flush=True)
                time.sleep(wait)
    print(f"  [{year}-{month:02d}] FAILED after {max_retries} attempts.",
          flush=True)
    return False


def process_month(year, month, force=False):
    """Merge per-mission nc files of one month into a single Parquet."""
    raw_dir = _raw_month_dir(year, month)
    out_path = _processed_month_path(year, month)

    if not force and os.path.exists(out_path):
        print(f"  [{year}-{month:02d}] already processed, skipping.")
        return True
    if not os.path.isdir(raw_dir):
        print(f"  [{year}-{month:02d}] raw dir missing, skip.")
        return False

    nc_files = sorted(glob.glob(os.path.join(raw_dir, "*.nc")))
    if not nc_files:
        print(f"  [{year}-{month:02d}] no raw files, skip.")
        return False

    print(f"  [{year}-{month:02d}] processing {len(nc_files)} mission files...",
          flush=True)
    frames = []
    for f in nc_files:
        ds = xr.open_dataset(f)
        df = ds[["swh", "latitude", "longitude"]].to_dataframe().reset_index()
        df = df[["time", "latitude", "longitude", "swh"]].dropna()
        frames.append(df)
        ds.close()

    df = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    if len(df) < MIN_OBS:
        print(f"  [{year}-{month:02d}] WARNING: only {len(df):,} obs "
              f"(< {MIN_OBS:,}), check data.", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  [{year}-{month:02d}] saved {len(df):,} obs -> {out_path}",
          flush=True)
    return True


def run_year(year, actions, workers=4, force=False):
    months = list(range(1, 13))

    if "download" in actions:
        print(f"\n{_HEADER}\nDownloading CMEMS MY L3 SWH — {year}\n{_HEADER}")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(download_month, year, m, force): m
                    for m in months}
            ok = {m: f.result() for f, m in
                  ((f, futs[f]) for f in as_completed(futs))}
        failed = [m for m, s in ok.items() if not s]
        if failed:
            print(f"DOWNLOAD FAILED for months: {failed}")
            return False

    if "process" in actions:
        print(f"\n{_HEADER}\nProcessing CMEMS MY L3 SWH — {year}\n{_HEADER}")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(process_month, year, m, force): m
                    for m in months}
            ok = {m: f.result() for f, m in
                  ((f, futs[f]) for f in as_completed(futs))}
        failed = [m for m, s in ok.items() if not s]
        if failed:
            print(f"PROCESS FAILED for months: {failed}")
            return False

    # Summary
    print(f"\n{_HEADER}\nSummary — {year}\n{_HEADER}")
    for m in months:
        out = _processed_month_path(year, m)
        raw = _raw_month_dir(year, m)
        if os.path.exists(out):
            n = len(pd.read_parquet(out, columns=["swh"]))
            nf = len(glob.glob(os.path.join(raw, "*.nc")))
            print(f"  {year}-{m:02d}: {n:>9,} obs ({nf} mission files)")
        else:
            print(f"  {year}-{m:02d}: MISSING")
    print()
    return True


def set_data_dir(path):
    """Override the module-level DATA_DIR (used by --data-dir / post_process)."""
    global DATA_DIR
    DATA_DIR = path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2021)
    p.add_argument("--action", nargs="+", default=["all"],
                   choices=["download", "process", "all"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-dir", default=None,
                   help="Root dir for MY raw/ and processed/ output "
                        "(default: ./data/cmems_l3_swh_my)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.data_dir:
        set_data_dir(args.data_dir)

    actions = ["download", "process"] if "all" in args.action else args.action
    ok = run_year(args.year, actions, workers=args.workers, force=args.force)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
