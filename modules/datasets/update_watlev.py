"""
Update water level (elev) channel in all hindcast data products using the new
WATLEV data from raw_new/.

HS/TP/DM are left untouched (verified identical between old and new datasets).

Usage:
    # Full update (all steps)
    python -m modules.datasets.update_watlev --workers 8

    # Step-by-step
    python load_data.py hindcast-update-elev --step processed --workers 8
    python load_data.py hindcast-update-elev --step preinterp-4ch --workers 8
    python load_data.py hindcast-update-elev --step preinterp-zoom --workers 8
    python load_data.py hindcast-update-elev --step stats
    python load_data.py hindcast-update-elev --step rebuild-cache
"""

import os
import sys
import json
import time
import h5py
import numpy as np
import xarray as xr
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_NEW_DIR = PROJECT_ROOT / 'data' / 'hindcast' / 'raw_new'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'hindcast' / 'processed'
PREINTERP_4CH_DIR = PROJECT_ROOT / 'data' / 'hindcast' / 'preinterpolated_4ch'
PREINTERP_ZOOM_DIR = PROJECT_ROOT / 'data' / 'hindcast' / 'preinterpolated_hk_zoom'
STATS_FILE = PROJECT_ROOT / 'data' / 'hindcast' / 'wave_data_stats.json'

N_NEW_WATLEV = 464064   # time steps in new WATLEV
N_OLD_TOTAL = 464591    # total time steps in old data
N_NODES = 15523         # total mesh nodes

# ---- HK Zoom interpolation grid (must match preinterpolate_hk_zoom.py) ----
from modules.utils import HK_REGION, ERA5_HK_LON_RANGE_2, ERA5_HK_LAT_RANGE_2


def _get_time_str(t):
    """Convert time index to YYYYMMDDTHH string."""
    ts = np.datetime64('1970-01-01T00:00:00') + np.timedelta64(int(t), 'h')
    return str(ts).replace('-', '').replace(':', '').replace(' ', 'T')[:11]


class WaterLevelUpdater:
    """Update elev channel in hindcast data products using new WATLEV data."""

    def __init__(self):
        self.watlev_path = RAW_NEW_DIR / 'WATLEV_merged.mat'
        self.node_path = RAW_NEW_DIR / 'node_infos.mat'

        # Open WATLEV HDF5 (lazy)
        self._watlev_file = None
        self._watlev_data = None
        self._node_lat = None
        self._node_lon = None

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------
    def _open_watlev(self):
        if self._watlev_file is None:
            self._watlev_file = h5py.File(self.watlev_path, 'r')
            self._watlev_data = self._watlev_file['NEW_watlev']
        return self._watlev_data

    def _load_node_coords(self):
        if self._node_lat is None:
            with h5py.File(self.node_path, 'r') as f:
                self._node_lat = np.squeeze(f['node_LAT'][()])
                self._node_lon = np.squeeze(f['node_LON'][()])
        return self._node_lon, self._node_lat

    def close(self):
        if self._watlev_file is not None:
            self._watlev_file.close()
            self._watlev_file = None
            self._watlev_data = None

    def load_watlev_row(self, t):
        """Read WATLEV row t (0-based), returns (15523,) float64."""
        data = self._open_watlev()
        return data[t, :].flatten().astype(np.float64)

    # ==================================================================
    # Step A: Update processed/wave_*.nc
    # ==================================================================
    @staticmethod
    def _update_processed_worker(args):
        """Worker: update elev in a range of .nc files. (module-level, pickle-safe)"""
        t_start, t_end, watlev_path = args
        updated = 0
        errors = []
        with h5py.File(watlev_path, 'r') as f:
            watlev_data = f['NEW_watlev']
            for t in range(t_start, t_end):
                ts_str = _get_time_str(t)
                nc_path = str(PROCESSED_DIR / f'wave_{ts_str}.nc')
                if not os.path.exists(nc_path):
                    continue
                try:
                    ds = xr.load_dataset(nc_path)
                    if t < N_NEW_WATLEV:
                        new_elev = watlev_data[t, :].flatten().astype(np.float64)
                        ds['elev'] = (['node'], new_elev.astype(np.float32))
                    else:
                        ds['elev'] = (['node'], np.full(N_NODES, np.nan, dtype=np.float32))
                    ds.to_netcdf(nc_path, engine='scipy')
                    ds.close()
                    updated += 1
                except Exception as e:
                    errors.append((ts_str, str(e)))
        return updated, errors

    def update_processed_nc(self, num_workers=1):
        """Update elev in all processed/wave_*.nc files sequentially.
        Reads WATLEV in memory-efficient chunks, updates one file at a time."""
        print_header("STEP A: Update processed/wave_*.nc")
        t0 = time.time()
        total = N_OLD_TOTAL
        print(f"  Total files: {total}")

        watlev = self._open_watlev()
        updated = 0
        errors = 0
        report_interval = 10000

        # Process in batches to keep HDF5 access efficient
        BATCH = 2000
        for batch_start in range(0, total, BATCH):
            batch_end = min(batch_start + BATCH, total)

            # Read a batch of WATLEV rows
            if batch_start < N_NEW_WATLEV:
                read_end = min(batch_end, N_NEW_WATLEV)
                watlev_batch = watlev[batch_start:read_end, :]  # (batch, 15523)
            else:
                watlev_batch = None  # all NaN region

            for i, t in enumerate(range(batch_start, batch_end)):
                ts_str = _get_time_str(t)
                nc_path = str(PROCESSED_DIR / f'wave_{ts_str}.nc')
                if not os.path.exists(nc_path):
                    continue
                try:
                    # Skip empty/corrupt files
                    fsize = os.path.getsize(nc_path)
                    if fsize < 100:
                        errors += 1
                        continue

                    ds = xr.load_dataset(nc_path)
                    if watlev_batch is not None and i < watlev_batch.shape[0]:
                        new_elev = watlev_batch[i, :].astype(np.float64)
                        ds['elev'] = (['node'], new_elev.astype(np.float32))
                    else:
                        ds['elev'] = (['node'], np.full(N_NODES, np.nan, dtype=np.float32))
                    ds.to_netcdf(nc_path, engine='scipy')
                    ds.close()
                    updated += 1
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"  Error at {ts_str}: {e}", flush=True)

            if (batch_start // BATCH) % (report_interval // BATCH) == 0:
                pct = 100 * batch_end / total
                elapsed = time.time() - t0
                speed = batch_end / elapsed if elapsed > 0 else 0
                eta = (total - batch_end) / speed if speed > 0 else 0
                print(f"  {batch_end}/{total} ({pct:.1f}%) — "
                      f"{speed:.0f} files/s, eta={eta/3600:.1f}h, "
                      f"errors={errors}", flush=True)

        elapsed = time.time() - t0
        print(f"  Updated: {updated} files in {elapsed/60:.1f} min ({errors} errors)")
        return updated

    # ==================================================================
    # Step B: Update preinterpolated_4ch/ (KDTree fast-path)
    # ==================================================================
    def update_preinterp_4ch(self, num_workers=1):
        """Update elev (ch3) in all preinterpolated_4ch/ files using KDTree NN."""
        print_header("STEP B: Update preinterpolated_4ch/")
        t0 = time.time()

        from scipy.spatial import cKDTree
        lons, lats = self._load_node_coords()

        # Precompute KDTree for 24x24 grid
        # Match the ERA5 grid definition from preinterpolate_wave.py:
        #   np.arange(start, stop, 0.25) — ERA5 is a fixed 0.25° grid
        H, W = 24, 24
        lon_grid = np.arange(ERA5_HK_LON_RANGE_2[0], ERA5_HK_LON_RANGE_2[1], 0.25)
        lat_grid = np.arange(ERA5_HK_LAT_RANGE_2[0], ERA5_HK_LAT_RANGE_2[1], 0.25)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        xi = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
        tree = cKDTree(np.column_stack([lons, lats]))
        _, nn_indices = tree.query(xi, k=1)
        print(f"  Precomputed KDTree: 24x24 grid, {len(nn_indices)} cells")

        # Build file list
        file_list = []
        for t in range(N_OLD_TOTAL):
            ts_str = _get_time_str(t)
            year = ts_str[:4]
            fpath = PREINTERP_4CH_DIR / year / f'wave_{ts_str}.npy'
            if fpath.exists():
                file_list.append((t, str(fpath)))

        print(f"  Files found: {len(file_list)}")
        if not file_list:
            print("  No files found — skipping")
            return 0

        # Sequential fast update
        watlev = self._open_watlev()
        updated, errors = 0, 0
        BATCH = 2000
        for batch_start in range(0, len(file_list), BATCH):
            batch_end = min(batch_start + BATCH, len(file_list))
            batch = file_list[batch_start:batch_end]

            for t, fpath in batch:
                try:
                    grid = np.load(fpath)
                    if grid.shape[0] < 4:
                        continue
                    if t < N_NEW_WATLEV:
                        elev_nodes = watlev[t, :].flatten().astype(np.float64)
                        elev_grid = elev_nodes[nn_indices].reshape(H, W)
                        grid[3] = np.maximum(elev_grid, 0.0)
                    else:
                        grid[3] = np.nan
                    np.save(fpath, grid)
                    updated += 1
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  Error at {fpath}: {e}", flush=True)

            pct = 100 * batch_end / len(file_list)
            elapsed = time.time() - t0
            speed = batch_end / elapsed if elapsed > 0 else 0
            eta = (len(file_list) - batch_end) / speed if speed > 0 else 0
            print(f"  {batch_end}/{len(file_list)} ({pct:.1f}%) — "
                  f"{speed:.0f} files/s, eta={eta/60:.1f} min, errors={errors}", flush=True)

        elapsed = time.time() - t0
        print(f"  Updated: {updated} files in {elapsed/60:.1f} min ({errors} errors)")
        return updated

    # ==================================================================
    # Step C: Update preinterpolated_hk_zoom/ (KDTree fast-path)
    # ==================================================================
    def update_preinterp_hk_zoom(self, num_workers=1):
        """Update elev channel in all preinterpolated_hk_zoom/ files using KDTree NN."""
        print_header("STEP C: Update preinterpolated_hk_zoom/")
        t0 = time.time()

        from scipy.spatial import cKDTree
        lons, lats = self._load_node_coords()

        # Precompute KDTree for 75x107 grid
        min_lon, max_lon, min_lat, max_lat = HK_REGION
        BUFFER = 0.3
        mask = (
            (lons >= min_lon - BUFFER) & (lons <= max_lon + BUFFER) &
            (lats >= min_lat - BUFFER) & (lats <= max_lat + BUFFER)
        )
        node_idx_buffered = np.where(mask)[0]
        tree = cKDTree(np.column_stack([lons[mask], lats[mask]]))

        n_lat, n_lon = 75, 107
        lon_edges = np.linspace(min_lon, max_lon, n_lon + 1)
        lat_edges = np.linspace(min_lat, max_lat, n_lat + 1)
        lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        lon_mesh, lat_mesh = np.meshgrid(lon_c, lat_c)
        xi = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
        _, nn_local = tree.query(xi, k=1)
        nn_global = node_idx_buffered[nn_local]  # map to global node indices
        print(f"  Precomputed KDTree: {n_lat}x{n_lon} grid, {len(nn_global)} cells")

        # Build file list
        file_list = []
        for t in range(N_OLD_TOTAL):
            ts_str = _get_time_str(t)
            year = ts_str[:4]
            fpath = PREINTERP_ZOOM_DIR / year / f'wave_{ts_str}.npy'
            if fpath.exists():
                file_list.append((t, str(fpath)))

        print(f"  Files found: {len(file_list)}")
        if not file_list:
            print("  No files found — skipping")
            return 0

        # Sequential fast update
        watlev = self._open_watlev()
        updated, errors = 0, 0
        BATCH = 2000
        for batch_start in range(0, len(file_list), BATCH):
            batch_end = min(batch_start + BATCH, len(file_list))
            batch = file_list[batch_start:batch_end]

            for t, fpath in batch:
                try:
                    grid = np.load(fpath)
                    if grid.shape[0] < 4:
                        continue
                    if t < N_NEW_WATLEV:
                        elev_nodes = watlev[t, :].flatten().astype(np.float64)
                        elev_grid = elev_nodes[nn_global].reshape(n_lat, n_lon)
                        grid[3] = np.maximum(elev_grid, 0.0)
                    else:
                        grid[3] = np.nan
                    np.save(fpath, grid)
                    updated += 1
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  Error at {fpath}: {e}", flush=True)

            pct = 100 * batch_end / len(file_list)
            elapsed = time.time() - t0
            speed = batch_end / elapsed if elapsed > 0 else 0
            eta = (len(file_list) - batch_end) / speed if speed > 0 else 0
            print(f"  {batch_end}/{len(file_list)} ({pct:.1f}%) — "
                  f"{speed:.0f} files/s, eta={eta/60:.1f} min, errors={errors}", flush=True)

        elapsed = time.time() - t0
        print(f"  Updated: {updated} files in {elapsed/60:.1f} min ({errors} errors)")
        return updated

    # ==================================================================
    # Step D: Update wave_data_stats.json (elev stats only)
    # ==================================================================
    def update_stats(self):
        """Recompute elev statistics from new WATLEV data (HK region only)."""
        print_header("STEP D: Update wave_data_stats.json")

        # Load existing stats
        with open(STATS_FILE) as f:
            stats = json.load(f)

        print(f"  Old elev stats: min={stats.get('elev_min')}, max={stats.get('elev_max')}, "
              f"mean={stats.get('elev_mean')}, std={stats.get('elev_std')}")

        # Load HK node mask
        lons, lats = self._load_node_coords()
        min_lon, max_lon, min_lat, max_lat = HK_REGION
        mask_hk = (lons >= min_lon) & (lons <= max_lon) & (lats >= min_lat) & (lats <= max_lat)
        print(f"  HK nodes: {mask_hk.sum()}")

        # Compute elev stats from new WATLEV (chunked)
        data = self._open_watlev()
        n_time = min(N_NEW_WATLEV, data.shape[0])

        elev_min = float('inf')
        elev_max = float('-inf')
        elev_sum = 0.0
        elev_sum2 = 0.0
        n_total = 0

        chunk = 5000
        for start in range(0, n_time, chunk):
            end = min(start + chunk, n_time)
            chunk_data = data[start:end, :][:, mask_hk]
            valid = chunk_data[np.isfinite(chunk_data)]
            if len(valid) == 0:
                continue
            elev_min = min(elev_min, float(valid.min()))
            elev_max = max(elev_max, float(valid.max()))
            elev_sum += float(valid.sum())
            elev_sum2 += float((valid ** 2).sum())
            n_total += len(valid)

        elev_mean = elev_sum / n_total
        elev_std = float(np.sqrt(elev_sum2 / n_total - elev_mean ** 2))

        # Update stats
        stats['source_elev'] = str(self.watlev_path)
        stats['elev_min'] = elev_min
        stats['elev_max'] = elev_max
        stats['elev_mean'] = elev_mean
        stats['elev_std'] = elev_std

        # Save
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"  New elev stats: min={elev_min:.4f}, max={elev_max:.4f}, "
              f"mean={elev_mean:.4f}, std={elev_std:.4f}")
        print(f"  Saved to {STATS_FILE}")
        return stats

    # ==================================================================
    # Step E: Rebuild 24x24 train/test caches (elev_norm channel only)
    # ==================================================================
    def rebuild_4ch_cache(self):
        """Rebuild wave_preinterp_4ch_{train,test}.npy — update only ch[4] (elev_norm)."""
        print_header("STEP E: Rebuild 24x24 train/test caches")

        with open(STATS_FILE) as f:
            stats = json.load(f)

        elev_mean = stats['elev_mean']
        elev_std = stats['elev_std']
        hs_log_mean = stats['hs_log_mean']
        hs_log_std = stats['hs_log_std']
        tp_log_mean = stats['tp_log_mean']
        tp_log_std = stats['tp_log_std']
        print(f"  elev: mean={elev_mean:.4f}, std={elev_std:.4f}")
        print(f"  hs log: mean={hs_log_mean:.4f}, std={hs_log_std:.4f}")
        print(f"  tp log: mean={tp_log_mean:.4f}, std={tp_log_std:.4f}")

        # Determine train/test split
        test_years = (2021, 2022)
        train_files = []
        test_files = []
        for year_dir in sorted(os.listdir(str(PREINTERP_4CH_DIR))):
            ypath = PREINTERP_4CH_DIR / year_dir
            if not ypath.is_dir():
                continue
            year = int(year_dir)
            for fname in sorted(os.listdir(str(ypath))):
                if fname.endswith('.npy') and not fname.startswith('wave_preinterp'):
                    fpath = ypath / fname
                    if year in test_years:
                        test_files.append(str(fpath))
                    else:
                        train_files.append(str(fpath))

        print(f"  Train files: {len(train_files)}, Test files: {len(test_files)}")

        for split, file_list in [('train', train_files), ('test', test_files)]:
            if not file_list:
                print(f"  No {split} files — skipping")
                continue

            cache_path = PREINTERP_4CH_DIR / f'wave_preinterp_4ch_{split}.npy'
            n_samples = len(file_list)
            print(f"\n  Building {split} cache: {n_samples} samples, 5 channels, 24x24")

            # Open mmap cache
            cache = np.lib.format.open_memmap(
                str(cache_path), mode='r+', dtype=np.float32,
            )

            n_updated = 0
            for i, fpath in enumerate(file_list):
                raw = np.load(fpath)  # (4, 24, 24) — raw hs, tp, dm, elev
                ocean = ~np.isnan(raw[0])

                # Recompute elev_norm (ch4) using new stats
                elev_ocean = raw[3][ocean]
                elev_norm = np.full_like(raw[3], np.nan, dtype=np.float32)
                if len(elev_ocean) > 0:
                    elev_norm[ocean] = (elev_ocean - elev_mean) / elev_std
                cache[i, 4] = elev_norm

                # Verify ch0-3 are unchanged (optional integrity check)
                hs_norm = np.full_like(raw[0], np.nan, dtype=np.float32)
                hs_norm[ocean] = (np.log1p(np.maximum(raw[0][ocean], 0)) - hs_log_mean) / hs_log_std
                tp_norm = np.full_like(raw[1], np.nan, dtype=np.float32)
                tp_norm[ocean] = (np.log1p(np.maximum(raw[1][ocean], 0)) - tp_log_mean) / tp_log_std

                cache[i, 0] = hs_norm
                cache[i, 1] = tp_norm
                # dm_sin, dm_cos — keep as-is (no change)
                dm_rad = np.deg2rad(raw[2])
                dm_sin = np.full_like(raw[2], np.nan, dtype=np.float32)
                dm_cos = np.full_like(raw[2], np.nan, dtype=np.float32)
                dm_sin[ocean] = np.sin(dm_rad[ocean])
                dm_cos[ocean] = np.cos(dm_rad[ocean])
                cache[i, 2] = dm_sin
                cache[i, 3] = dm_cos

                n_updated += 1
                if (i + 1) % 10000 == 0:
                    print(f"    {i+1}/{n_samples} done")

            cache.flush()
            print(f"  {split} cache updated: {n_updated} samples")
            print(f"  Saved: {cache_path}")

        return True


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_update(step='all', num_workers=8):
    """Main entry point for CLI."""
    updater = WaterLevelUpdater()
    try:
        if step in ('all', 'processed'):
            updater.update_processed_nc(num_workers=num_workers)
        if step in ('all', 'preinterp-4ch'):
            updater.update_preinterp_4ch(num_workers=num_workers)
        if step in ('all', 'preinterp-zoom'):
            updater.update_preinterp_hk_zoom(num_workers=num_workers)
        if step in ('all', 'stats'):
            updater.update_stats()
        if step in ('all', 'rebuild-cache'):
            updater.rebuild_4ch_cache()
    finally:
        updater.close()
    print("\nDone.")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Update water level in hindcast data products')
    ap.add_argument('--step', default='all',
                    choices=['all', 'processed', 'preinterp-4ch', 'preinterp-zoom',
                             'stats', 'rebuild-cache'])
    ap.add_argument('--workers', '-w', type=int, default=8)
    args = ap.parse_args()
    run_update(args.step, args.workers)
