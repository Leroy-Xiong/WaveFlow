"""
Compare SWH data between data/global_hindcast/raw/ and data/global_hindcast/raw_new/.

- Basic comparison: sample multiple timesteps, compute per-node metrics (MAE, RMSE, Bias, Corr)
- Deep analysis: spatial distribution, latitude bands, SWH magnitude levels, hotspot regions

Usage:
    python -m modules.datasets.compare_global_raw
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/global_hindcast/raw")
RAW_NEW_DIR = Path("data/global_hindcast/raw_new")

# Both datasets share the same 3-hourly timesteps: 03, 06, 09, 12, 15, 18, 21
COMMON_HOURS = [3, 6, 9, 12, 15, 18, 21]

# Seasonally representative timestamps for deep analysis
DEEP_SAMPLE_TIMES = [
    "1955-01-15 12:00",  # NH winter
    "1955-07-15 12:00",  # NH summer
    "1985-01-15 12:00",  # NH winter
    "1985-07-15 12:00",  # NH summer
    "2005-04-15 12:00",  # NH spring
    "2005-10-15 12:00",  # NH autumn
    "2015-01-15 12:00",  # NH winter
    "2015-07-15 12:00",  # NH summer
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _pick_sample_timestamps(n_per_decade: int = 3, seed: int = 42) -> list:
    """Randomly sample n_per_decade timestamps from each decade (common hours only)."""
    rng = np.random.RandomState(seed)
    timestamps = []
    for decade_start in range(1950, 2020, 10):
        decade_end = min(decade_start + 9, 2022)
        dates = pd.date_range(f"{decade_start}-01-01", f"{decade_end}-12-31", freq="3h")
        dates = dates[dates.hour.isin(COMMON_HOURS)]
        if len(dates) == 0:
            continue
        chosen = rng.choice(dates, size=min(n_per_decade, len(dates)), replace=False)
        timestamps.extend(sorted(chosen))
    return timestamps


def _load_raw_swh(ts: pd.Timestamp) -> np.ndarray:
    """Load WWM_1 from raw/ for a given timestamp."""
    year = ts.year
    decade_start = (year // 10) * 10
    fpath = RAW_DIR / f"mentaschi2023swh_{decade_start}-{decade_start + 9}.nc"
    if not fpath.exists():
        raise FileNotFoundError(f"Missing: {fpath}")
    ds = xr.open_dataset(fpath, engine="netcdf4")
    swh = ds["WWM_1"].sel(time=ts, method="nearest").values
    ds.close()
    return swh


def _load_rawnew_swh(ts: pd.Timestamp) -> np.ndarray:
    """Load SWH from raw_new/ for a given timestamp.

    Note: raw_new/ daily files have 7 timesteps starting at 03:00 (same as raw/).
    """
    date_str = ts.strftime("%Y%m%d")
    fpath = RAW_NEW_DIR / f"ERA5_schismwwm_final_{date_str}.nc"
    if not fpath.exists():
        raise FileNotFoundError(f"Missing: {fpath}")
    ds = xr.open_dataset(fpath)
    # 03:00 -> idx 0, 06:00 -> idx 1, ..., 21:00 -> idx 6
    time_idx = ts.hour // 3 - 1
    swh = ds["SWH"].isel(time=time_idx).values
    ds.close()
    return swh


def _load_both(ts_str: str) -> tuple:
    """Load SWH + coordinates from both datasets for a given timestamp string.

    Returns:
        (x, y, swh_old, swh_new, ts)
    """
    ts = pd.Timestamp(ts_str)

    # raw/
    year = ts.year
    decade_start = (year // 10) * 10
    f_old = RAW_DIR / f"mentaschi2023swh_{decade_start}-{decade_start + 9}.nc"
    ds_old = xr.open_dataset(f_old, engine="netcdf4")
    swh_old = ds_old["WWM_1"].sel(time=ts, method="nearest").values
    ds_old.close()

    # raw_new/
    date_str = ts.strftime("%Y%m%d")
    f_new = RAW_NEW_DIR / f"ERA5_schismwwm_final_{date_str}.nc"
    ds_new = xr.open_dataset(f_new)
    time_idx = ts.hour // 3 - 1
    swh_new = ds_new["SWH"].isel(time=time_idx).values
    x = ds_new["x"].values
    y = ds_new["y"].values
    ds_new.close()

    return x, y, swh_old, swh_new, ts


# ---------------------------------------------------------------------------
# Temporal overlap check
# ---------------------------------------------------------------------------

def check_temporal_overlap():
    """Check temporal overlap between the two datasets."""
    print("=" * 60)
    print("0. Temporal Overlap Check")
    print("=" * 60)

    first_new = sorted(RAW_NEW_DIR.glob("*.nc"))[0]
    last_new = sorted(RAW_NEW_DIR.glob("*.nc"))[-1]
    print(f"  raw_new/ first file: {first_new.name}")
    print(f"  raw_new/ last  file: {last_new.name}")

    ds = xr.open_dataset(first_new)
    print(f"  raw_new/ daily hours: {list(ds.time.values)}")
    ds.close()

    raw_files = sorted(RAW_DIR.glob("*.nc"))
    print(f"  raw/ file count: {len(raw_files)}")
    for f in raw_files:
        print(f"    {f.name}")

    ds_first = xr.open_dataset(raw_files[0], engine="netcdf4")
    ds_last = xr.open_dataset(raw_files[-1], engine="netcdf4")
    print(f"  raw/ time range: {ds_first.time.values[0]} ~ {ds_last.time.values[-1]}")
    print(f"  raw/ daily hours: {sorted(set(ds_first.time.dt.hour.values))}")
    ds_first.close()
    ds_last.close()


# ---------------------------------------------------------------------------
# Coordinate comparison
# ---------------------------------------------------------------------------

def compare_coords():
    """Verify that both datasets share identical x, y coordinates."""
    print("=" * 60)
    print("1. Coordinate Comparison")
    print("=" * 60)

    ds_old = xr.open_dataset(RAW_DIR / "mentaschi2023swh_2000-2009.nc", engine="netcdf4")
    ds_new = xr.open_dataset(RAW_NEW_DIR / "ERA5_schismwwm_final_20000101.nc")

    for coord in ["x", "y"]:
        old_vals = ds_old[coord].values
        new_vals = ds_new[coord].values
        max_diff = np.abs(old_vals - new_vals).max()
        n_equal = np.sum(np.isclose(old_vals, new_vals))
        pct = 100 * n_equal / len(old_vals)
        print(f"  {coord}: max_diff={max_diff:.2e}, matching_nodes={n_equal}/{len(old_vals)} ({pct:.1f}%)")

    ds_old.close()
    ds_new.close()


# ---------------------------------------------------------------------------
# Basic SWH comparison
# ---------------------------------------------------------------------------

def compare_swh(n_per_decade: int = 3, seed: int = 42) -> list[dict]:
    """Sample timesteps and compare SWH values node-by-node.

    Returns:
        list of dict with per-timestep comparison metrics.
    """
    print("\n" + "=" * 60)
    print("2. SWH Comparison (raw.WWM_1 vs raw_new.SWH)")
    print("=" * 60)

    timestamps = _pick_sample_timestamps(n_per_decade=n_per_decade, seed=seed)
    print(f"\nSampled {len(timestamps)} timesteps:\n")
    for ts in timestamps:
        print(f"    {pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M')}")

    results = []
    for i, ts_np in enumerate(timestamps):
        ts = pd.Timestamp(ts_np)
        ts_str = ts.strftime("%Y-%m-%d %H:%M")
        print(f"\n  [{i+1}/{len(timestamps)}] {ts_str} ...", end=" ", flush=True)

        try:
            swh_old = _load_raw_swh(ts)
            swh_new = _load_rawnew_swh(ts)

            valid = ~(np.isnan(swh_old) | np.isnan(swh_new))
            swh_o = swh_old[valid]
            swh_n = swh_new[valid]
            diff = swh_o - swh_n

            r = {
                "ts": ts,
                "n_total": len(swh_old),
                "n_valid": len(swh_o),
                "pct_nan_old": 100 * np.isnan(swh_old).sum() / len(swh_old),
                "pct_nan_new": 100 * np.isnan(swh_new).sum() / len(swh_new),
                "max_abs_diff": float(np.abs(diff).max()),
                "mae": float(np.abs(diff).mean()),
                "rmse": float(np.sqrt((diff ** 2).mean())),
                "bias": float(diff.mean()),
                "corr": float(np.corrcoef(swh_o, swh_n)[0, 1]),
                "mean_old": float(swh_o.mean()),
                "mean_new": float(swh_n.mean()),
                "std_old": float(swh_o.std()),
                "std_new": float(swh_n.std()),
            }
            results.append(r)
            print(f"MAE={r['mae']:.4f}, RMSE={r['rmse']:.4f}, "
                  f"Bias={r['bias']:+.4f}, Corr={r['corr']:.6f}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"ts": ts, "error": str(e)})

    return results


def print_swh_summary(results: list[dict]):
    """Print summary statistics for SWH comparison."""
    print("\n" + "=" * 60)
    print("3. Basic Comparison Summary")
    print("=" * 60)

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("  No valid comparison results!")
        return

    maes = [r["mae"] for r in valid]
    rmses = [r["rmse"] for r in valid]
    biases = [r["bias"] for r in valid]
    corrs = [r["corr"] for r in valid]
    max_diffs = [r["max_abs_diff"] for r in valid]

    n_ocean = valid[0]['n_valid']
    n_total = valid[0]['n_total']
    print(f"\n  Timesteps compared: {len(valid)}")
    print(f"  Ocean nodes: {n_ocean:,} / {n_total:,}")
    print(f"\n  {'Metric':<16} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'MAE (m)':<16} {np.mean(maes):>10.4f} {np.median(maes):>10.4f} "
          f"{np.min(maes):>10.4f} {np.max(maes):>10.4f}")
    print(f"  {'RMSE (m)':<16} {np.mean(rmses):>10.4f} {np.median(rmses):>10.4f} "
          f"{np.min(rmses):>10.4f} {np.max(rmses):>10.4f}")
    print(f"  {'Bias (m)':<16} {np.mean(biases):>+10.4f} {np.median(biases):>+10.4f} "
          f"{np.min(biases):>+10.4f} {np.max(biases):>+10.4f}")
    print(f"  {'Max Abs Diff (m)':<16} {np.mean(max_diffs):>10.4f} {np.median(max_diffs):>10.4f} "
          f"{np.min(max_diffs):>10.4f} {np.max(max_diffs):>10.4f}")
    print(f"  {'Correlation':<16} {np.mean(corrs):>10.6f} {np.median(corrs):>10.6f} "
          f"{np.min(corrs):>10.6f} {np.max(corrs):>10.6f}")

    # Per-decade breakdown
    print(f"\n  {'Decade':<12} {'Samples':>6} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'Corr':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    decades = sorted(set(r["ts"].year // 10 * 10 for r in valid))
    for decade in decades:
        dr = [r for r in valid if r["ts"].year // 10 * 10 == decade]
        n = len(dr)
        print(f"  {decade}-{decade+9:<4} {n:>6} "
              f"{np.mean([r['mae'] for r in dr]):>8.4f} "
              f"{np.mean([r['rmse'] for r in dr]):>8.4f} "
              f"{np.mean([r['bias'] for r in dr]):>+8.4f} "
              f"{np.mean([r['corr'] for r in dr]):>10.6f}")

    print(f"\n  NaN pct: raw={np.mean([r['pct_nan_old'] for r in valid]):.2f}%, "
          f"raw_new={np.mean([r['pct_nan_new'] for r in valid]):.2f}%")
    print(f"  Mean ocean SWH: raw={np.mean([r['mean_old'] for r in valid]):.4f} m, "
          f"raw_new={np.mean([r['mean_new'] for r in valid]):.4f} m")


# ---------------------------------------------------------------------------
# Deep analysis
# ---------------------------------------------------------------------------

def deep_analyze(sample_times: list[str] | None = None) -> list[dict]:
    """Deep-dive analysis: spatial distribution, latitude bands, SWH levels, hotspots.

    Args:
        sample_times: list of timestamp strings to analyze.
                      Defaults to DEEP_SAMPLE_TIMES.

    Returns:
        list of dict with per-timestep analysis data.
    """
    if sample_times is None:
        sample_times = DEEP_SAMPLE_TIMES

    print("=" * 60)
    print("4. Deep Analysis: Spatial / Latitude Bands / SWH Levels / Hotspots")
    print("=" * 60)

    all_diffs = []

    for ts_str in sample_times:
        print(f"\n{'='*60}")
        print(f"  Timestep: {ts_str}")
        print(f"{'='*60}")

        x, y, swh_old, swh_new, ts = _load_both(ts_str)

        valid = ~(np.isnan(swh_old) | np.isnan(swh_new))
        swh_o = swh_old[valid]
        swh_n = swh_new[valid]
        diff = swh_o - swh_n
        abs_diff = np.abs(diff)
        x_v, y_v = x[valid], y[valid]

        n = len(swh_o)
        corr = float(np.corrcoef(swh_o, swh_n)[0, 1])
        print(f"  Valid nodes: {n:,}")
        print(f"  SWH (old):   mean={swh_o.mean():.3f}, std={swh_o.std():.3f}, max={swh_o.max():.2f}")
        print(f"  SWH (new):   mean={swh_n.mean():.3f}, std={swh_n.std():.3f}, max={swh_n.max():.2f}")
        print(f"  Diff:   mean={diff.mean():+.4f}, std={diff.std():.4f}, max_abs={abs_diff.max():.2f}")
        print(f"  Corr:   {corr:.6f}")

        # 1) Latitude band analysis
        _print_lat_band_analysis(y_v, diff, swh_o)

        # 2) SWH magnitude level analysis
        _print_swh_level_analysis(swh_o, diff)

        # 3) Top-10 nodes with largest differences
        _print_top_diff_nodes(x_v, y_v, swh_o, swh_n, diff, abs_diff)

        all_diffs.append({
            "ts": ts_str, "x": x_v, "y": y_v,
            "diff": diff, "abs_diff": abs_diff, "swh_old": swh_o,
        })

    # Cross-timestep summary
    _print_cross_time_summary(all_diffs)

    return all_diffs


def _print_lat_band_analysis(y: np.ndarray, diff: np.ndarray, swh: np.ndarray):
    """Print stats grouped by latitude bands."""
    print(f"\n  --- By Latitude Band ---")
    print(f"  {'Band':<12} {'Nodes':>10} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'Mean SWH':>8}")
    lat_bins = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]
    for lat_low, lat_high in lat_bins:
        mask = (y >= lat_low) & (y < lat_high)
        if mask.sum() == 0:
            continue
        d = diff[mask]
        print(f"  [{lat_low:>3d}, {lat_high:>3d})   {mask.sum():>10,}  "
              f"{np.abs(d).mean():>8.4f}  {np.sqrt((d**2).mean()):>8.4f}  "
              f"{d.mean():>+8.4f}  {swh[mask].mean():>8.3f}")


def _print_swh_level_analysis(swh: np.ndarray, diff: np.ndarray):
    """Print stats grouped by SWH magnitude."""
    print(f"\n  --- By SWH Level (old) ---")
    print(f"  {'SWH range (m)':<16} {'Nodes':>10} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'Rel MAE%':>8}")
    swh_bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, 100)]
    for low, high in swh_bins:
        mask = (swh >= low) & (swh < high)
        if mask.sum() == 0:
            continue
        d = diff[mask]
        s = swh[mask]
        mae = np.abs(d).mean()
        rel_mae = 100 * mae / s.mean() if s.mean() > 0 else 0
        print(f"  [{low:>3.0f}, {high:>3.0f})       {mask.sum():>10,}  {mae:>8.4f}  "
              f"{np.sqrt((d**2).mean()):>8.4f}  {d.mean():>+8.4f}  {rel_mae:>8.2f}")


def _print_top_diff_nodes(x: np.ndarray, y: np.ndarray, swh_o: np.ndarray,
                          swh_n: np.ndarray, diff: np.ndarray, abs_diff: np.ndarray,
                          n: int = 10):
    """Print the n nodes with the largest absolute differences."""
    print(f"\n  --- Top {n} Nodes by |diff| ---")
    top_idx = np.argsort(abs_diff)[-n:][::-1]
    print(f"  {'Rank':<6} {'Lon':>8} {'Lat':>8} {'SWH_old':>8} {'SWH_new':>8} {'Diff':>8} {'AbsDiff':>8}")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:<6} {x[idx]:>8.2f} {y[idx]:>8.2f} "
              f"{swh_o[idx]:>8.3f} {swh_n[idx]:>8.3f} "
              f"{diff[idx]:>+8.3f} {abs_diff[idx]:>8.3f}")


def _print_cross_time_summary(all_diffs: list[dict]):
    """Print cross-timestep difference distribution summary."""
    print(f"\n\n{'='*60}")
    print(f"  Cross-Timestep Summary")
    print(f"{'='*60}")
    for d in all_diffs:
        pct_gt_1m = 100 * (d["abs_diff"] > 1).mean()
        pct_gt_05m = 100 * (d["abs_diff"] > 0.5).mean()
        print(f"  {d['ts']}: MAE={d['abs_diff'].mean():.4f}, "
              f"|diff|>1m={pct_gt_1m:.2f}%, "
              f"|diff|>0.5m={pct_gt_05m:.2f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_temporal_overlap()
    compare_coords()
    results = compare_swh(n_per_decade=3, seed=42)
    print_swh_summary(results)
    deep_analyze()
