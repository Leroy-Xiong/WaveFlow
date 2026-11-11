"""
HK Zoom Preinterpolator: interpolate hindcast wave data from scattered
unstructured mesh nodes to a dense regular grid focused on Hong Kong waters.

Target grid: ~0.00935° equal-resolution grid within HK_REGION
(113.6–114.6°E, 22.0–22.7°N), producing ~8,000 pixels that approximately
match the 7,788 HK mesh nodes. Land pixels are masked to NaN.

Usage:
    from modules.datasets.preinterpolate_hk_zoom import HKZoomPreinterpolator

    interp = HKZoomPreinterpolator()
    grid = interp.interpolate_single('data/hindcast/processed/wave_19700101T00.nc')
    # grid.shape -> (3, 75, 107)  [hs, tp, dm], land=NaN

    # Batch processing:
    interp.run(timestamps=['19700101T00', '19700101T01'], num_workers=4)

    # Visualise a sample:
    interp.visualize_sample('data/hindcast/processed/wave_20180916T00.nc',
                            save_path='output/plot/data/hk/interpolation/sample.png')
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from tqdm import tqdm

from modules.utils import HK_REGION

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_DIR = './data/hindcast/processed'
DEFAULT_OUTPUT_DIR = './data/hindcast/preinterpolated_hk_zoom'
DEFAULT_GRID_SHAPE = (75, 107)  # (n_lat, n_lon) — 8,025 pixels, equal Δ≈0.00935°
DEFAULT_VARIABLES = ('hs', 'tp', 'dm', 'elev')
DEFAULT_BUFFER_DEG = 0.3  # expand HK_REGION by this much for source nodes
DEFAULT_LAND_DISTANCE = 0.02  # ° — grid cells farther than this from any node → land

# ---------------------------------------------------------------------------
# Module-level worker helpers for process pools
# ---------------------------------------------------------------------------
# ProcessPoolExecutor pickles the submitted callable for *every* task.  A
# bound method drags the whole HKZoomPreinterpolator instance along (~29 MB,
# dominated by the 464k-entry _ts_to_path index) → ~0.25 s/task of pure
# serialization in the parent (~30 h total).  Module-level functions pickle
# by reference, so the worker reads its (small) context from a module global
# that forked workers inherit — this makes append_elev O(file) instead of
# O(file × 29 MB).

_WORKER_CTX = None  # (xi, land_mask, grid_shape, buffer_bounds)


def _interpolate_single_variable_static(ctx, nc_path: str, var: str) -> np.ndarray:
    """Interpolate one variable to the dense HK grid (no instance).

    Module-level twin of ``HKZoomPreinterpolator.interpolate_single_variable``
    so process-pool workers can run it without pickling the instance.
    """
    xi, land_mask, grid_shape, buffer_bounds = ctx
    b_min_lon, b_max_lon, b_min_lat, b_max_lat = buffer_bounds

    ds = xr.load_dataset(nc_path)
    node_lon = ds['lon'].values
    node_lat = ds['lat'].values
    mask = (
        (node_lon >= b_min_lon) & (node_lon <= b_max_lon)
        & (node_lat >= b_min_lat) & (node_lat <= b_max_lat)
    )
    points = np.column_stack([node_lon[mask], node_lat[mask]])
    orig_vals = ds[var].values[mask]

    n_lat, n_lon = grid_shape
    result = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)

    if var == 'dm':
        rad = np.deg2rad(orig_vals)
        cubic_sin = griddata(points, np.sin(rad), xi, method='cubic')
        nan_s = np.isnan(cubic_sin)
        if nan_s.any():
            cubic_sin = np.where(nan_s, griddata(points, np.sin(rad), xi, method='nearest'), cubic_sin)
        cubic_cos = griddata(points, np.cos(rad), xi, method='cubic')
        nan_c = np.isnan(cubic_cos)
        if nan_c.any():
            cubic_cos = np.where(nan_c, griddata(points, np.cos(rad), xi, method='nearest'), cubic_cos)
        dm_recon = np.rad2deg(np.arctan2(cubic_sin, cubic_cos))
        dm_recon = np.where(dm_recon < 0, dm_recon + 360.0, dm_recon)
        grid_vals = dm_recon
    elif var in ('hs', 'tp'):
        cubic = griddata(points, orig_vals, xi, method='cubic')
        nan_mask = np.isnan(cubic)
        if nan_mask.any():
            nearest = griddata(points, orig_vals, xi, method='nearest')
            grid_vals = np.where(nan_mask, nearest, cubic)
        else:
            grid_vals = cubic
        grid_vals = np.maximum(grid_vals, 0.0)
    else:
        # elev (and other signed variables): NO clip — negative values are
        # physical (low tide / z-scored), unlike hs/tp where they are
        # pure cubic overshoot artifacts.
        cubic = griddata(points, orig_vals, xi, method='cubic')
        nan_mask = np.isnan(cubic)
        if nan_mask.any():
            nearest = griddata(points, orig_vals, xi, method='nearest')
            grid_vals = np.where(nan_mask, nearest, cubic)
        else:
            grid_vals = cubic

    result[0] = grid_vals.reshape(n_lat, n_lon)
    result[:, land_mask] = np.nan
    ds.close()
    return result  # (1, n_lat, n_lon)


def _append_elev_worker_static(args) -> tuple:
    """Read existing (3,H,W) file, interpolate elev, save as (4,H,W).

    Module-level twin of ``HKZoomPreinterpolator._append_elev_worker`` that
    reads its context from the ``_WORKER_CTX`` module global (set by
    append_elev before the pool starts; inherited by forked workers), so
    per-task pickling stays small.
    """
    ts, nc_path, out_path, force = args
    try:
        existing = np.load(out_path)
        if existing.shape[0] == 4 and not force:
            return (ts, True, 'already_4ch')
        elev_grid = _interpolate_single_variable_static(_WORKER_CTX, nc_path, 'elev')
        if existing.shape[0] == 4:
            # Replace elev channel in place (force mode)
            existing[3] = elev_grid[0]
            np.save(out_path, existing)
        else:
            stacked = np.concatenate([existing, elev_grid], axis=0)
            np.save(out_path, stacked)
        return (ts, True, 'ok')
    except Exception as e:
        return (ts, False, str(e))


class HKZoomPreinterpolator:
    """Interpolate unstructured wave data to a dense regular grid over HK.

    Parameters
    ----------
    source_dir : str
        Directory containing per-timestamp wave_*.nc files.
    output_dir : str
        Root output directory. Per-year subdirectories are created under it.
    hk_region : tuple
        (min_lon, max_lon, min_lat, max_lat) bounding box.
    grid_shape : tuple
        (n_lat, n_lon) pixel dimensions of the output grid.
    variables : tuple of str
        Variable names to interpolate (must exist in the .nc files).
    buffer_deg : float
        Expand the HK bounding box by this many degrees when selecting
        source nodes for interpolation (avoids edge extrapolation artifacts).
    """

    def __init__(
        self,
        source_dir: str = DEFAULT_SOURCE_DIR,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        hk_region: tuple = HK_REGION,
        grid_shape: tuple = DEFAULT_GRID_SHAPE,
        variables: tuple = DEFAULT_VARIABLES,
        buffer_deg: float = DEFAULT_BUFFER_DEG,
        land_distance_threshold: float = DEFAULT_LAND_DISTANCE,
    ):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.hk_region = hk_region
        self.grid_shape = grid_shape  # (n_lat, n_lon)
        self.variables = list(variables)
        self.buffer_deg = buffer_deg
        self.land_distance_threshold = land_distance_threshold

        min_lon, max_lon, min_lat, max_lat = hk_region
        n_lat, n_lon = grid_shape

        # Build target regular grid
        self.lon_vals = np.linspace(min_lon, max_lon, n_lon)
        self.lat_vals = np.linspace(min_lat, max_lat, n_lat)
        self.LON2D, self.LAT2D = np.meshgrid(self.lon_vals, self.lat_vals)
        # Query points for griddata: (n_lat * n_lon, 2)
        self.XI = np.column_stack([self.LON2D.ravel(), self.LAT2D.ravel()])

        # Buffer region for source node selection
        self._buffer_bounds = (
            min_lon - buffer_deg,
            max_lon + buffer_deg,
            min_lat - buffer_deg,
            max_lat + buffer_deg,
        )

        # Build index of available .nc files: ts -> nc_path
        self._ts_to_path = {}
        if os.path.isdir(source_dir):
            for fname in sorted(os.listdir(source_dir)):
                if fname.startswith('wave_') and fname.endswith('.nc'):
                    ts = fname.replace('wave_', '').replace('.nc', '')
                    self._ts_to_path[ts] = os.path.join(source_dir, fname)

        # Build land mask (True = land, set grid cells to NaN)
        # Must be after _ts_to_path (needs node coords) and _buffer_bounds
        self.land_mask = self._build_land_mask()

        print(f"HKZoomPreinterpolator ready:")
        print(f"  HK region:  lon [{min_lon:.2f}, {max_lon:.2f}]  "
              f"lat [{min_lat:.2f}, {max_lat:.2f}]")
        print(f"  Grid:       {n_lon} (lon) × {n_lat} (lat) = {n_lon * n_lat} pixels")
        print(f"  Resolution: {self.lon_vals[1] - self.lon_vals[0]:.4f}° lon × "
              f"{self.lat_vals[1] - self.lat_vals[0]:.4f}° lat")
        print(f"  Buffer:     {buffer_deg}°")
        print(f"  Land mask:  {self.land_mask.sum()} / {self.land_mask.size} cells "
              f"({self.land_mask.sum() / self.land_mask.size * 100:.1f}%), "
              f"threshold={land_distance_threshold}°")
        print(f"  Variables:  {self.variables}")
        print(f"  Source:     {source_dir}  ({len(self._ts_to_path)} .nc files)")
        print(f"  Output:     {output_dir}")

    # ------------------------------------------------------------------
    # Node filtering
    # ------------------------------------------------------------------
    def _filter_nodes_to_buffer(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """Return boolean mask of nodes within the buffered HK region."""
        b_min_lon, b_max_lon, b_min_lat, b_max_lat = self._buffer_bounds
        return (
            (lons >= b_min_lon)
            & (lons <= b_max_lon)
            & (lats >= b_min_lat)
            & (lats <= b_max_lat)
        )

    # ------------------------------------------------------------------
    # Land mask (mesh-node-distance based)
    # ------------------------------------------------------------------
    def _build_land_mask(self) -> np.ndarray:
        """Return a boolean land mask (True=land) for the target grid.

        Uses distance from each grid cell to the nearest mesh node.
        Grid cells farther than ``land_distance_threshold`` degrees from
        any node are marked as land.  This is inherently more precise than
        a generic coastline shapefile because the mesh boundary *is* the
        model's definition of the ocean domain.

        The default threshold (0.02°) is ~2× the grid resolution and ~2.7×
        the median node spacing, which safely separates ocean from land
        while respecting the 95th-percentile node spacing (0.017°).
        """
        # Load node coordinates once from the first available .nc file
        if not self._ts_to_path:
            return np.zeros(self.LON2D.shape, dtype=bool)

        first_nc = next(iter(self._ts_to_path.values()))
        ds = xr.load_dataset(first_nc)
        node_lon = ds['lon'].values
        node_lat = ds['lat'].values
        ds.close()

        # Use all nodes within the buffered region
        mask = self._filter_nodes_to_buffer(node_lon, node_lat)
        points = np.column_stack([node_lon[mask], node_lat[mask]])

        # KD-tree: for each grid cell, find distance to nearest mesh node
        tree = cKDTree(points)
        grid_pts = np.column_stack([self.LON2D.ravel(), self.LAT2D.ravel()])
        distances, _ = tree.query(grid_pts)

        land = distances.reshape(self.LON2D.shape) > self.land_distance_threshold
        return land

    # ------------------------------------------------------------------
    # Core interpolation
    # ------------------------------------------------------------------
    def interpolate_single(self, nc_path: str) -> np.ndarray:
        """Interpolate one wave .nc file to the dense HK grid.

        Parameters
        ----------
        nc_path : str
            Path to a wave_*.nc file.

        Returns
        -------
        np.ndarray, shape (n_vars, n_lat, n_lon), dtype float32.
        """
        ds = xr.load_dataset(nc_path)
        node_lon = ds['lon'].values
        node_lat = ds['lat'].values

        # Filter to buffered HK region for efficiency
        mask = self._filter_nodes_to_buffer(node_lon, node_lat)
        if mask.sum() < 10:
            ds.close()
            raise RuntimeError(
                f"Only {mask.sum()} nodes in buffered region for {nc_path}"
            )

        points = np.column_stack([node_lon[mask], node_lat[mask]])

        n_vars = len(self.variables)
        n_lat, n_lon = self.grid_shape
        result = np.full((n_vars, n_lat, n_lon), np.nan, dtype=np.float32)

        for j, var in enumerate(self.variables):
            orig_vals = ds[var].values[mask]

            # ---- dm: decompose sin/cos for circular interpolation ----
            if var == 'dm':
                # Convert to radians, decompose
                rad = np.deg2rad(orig_vals)
                sin_vals = np.sin(rad)
                cos_vals = np.cos(rad)

                # Cubic interpolate sin component
                cubic_sin = griddata(points, sin_vals, self.XI, method='cubic')
                nan_sin = np.isnan(cubic_sin)
                if nan_sin.any():
                    near_sin = griddata(points, sin_vals, self.XI, method='nearest')
                    cubic_sin = np.where(nan_sin, near_sin, cubic_sin)

                # Cubic interpolate cos component
                cubic_cos = griddata(points, cos_vals, self.XI, method='cubic')
                nan_cos = np.isnan(cubic_cos)
                if nan_cos.any():
                    near_cos = griddata(points, cos_vals, self.XI, method='nearest')
                    cubic_cos = np.where(nan_cos, near_cos, cubic_cos)

                # Reconstruct direction from sin/cos, map to [0, 360)
                dm_recon = np.rad2deg(np.arctan2(cubic_sin, cubic_cos))
                dm_recon = np.where(dm_recon < 0, dm_recon + 360.0, dm_recon)
                grid_vals = dm_recon

            # ---- hs / tp: cubic + clip >= 0 ----
            elif var in ('hs', 'tp'):
                # 1st pass: cubic interpolation
                cubic = griddata(points, orig_vals, self.XI, method='cubic')

                # 2nd pass: nearest-neighbour fallback for NaN positions
                nan_mask = np.isnan(cubic)
                if nan_mask.any():
                    nearest = griddata(points, orig_vals, self.XI, method='nearest')
                    grid_vals = np.where(nan_mask, nearest, cubic)
                else:
                    grid_vals = cubic

                # Clip non-negative (cubic overshoots negative near zero boundaries)
                grid_vals = np.maximum(grid_vals, 0.0)

            # ---- elev (and other signed variables): cubic, NO clip ----
            # elev can be negative (low tide / z-scored) — clipping destroys
            # real tidal signal.  Negative values in hs/tp are pure cubic
            # overshoot artifacts (node data is non-negative); in elev they
            # are physical.
            else:
                cubic = griddata(points, orig_vals, self.XI, method='cubic')

                nan_mask = np.isnan(cubic)
                if nan_mask.any():
                    nearest = griddata(points, orig_vals, self.XI, method='nearest')
                    grid_vals = np.where(nan_mask, nearest, cubic)
                else:
                    grid_vals = cubic

            result[j] = grid_vals.reshape(n_lat, n_lon)

        # Apply land mask: set land grid cells to NaN
        result[:, self.land_mask] = np.nan

        ds.close()
        return result

    # ------------------------------------------------------------------
    # Single-file worker (for parallel map)
    # ------------------------------------------------------------------
    def interpolate_single_variable(self, nc_path: str, var: str) -> np.ndarray:
        """Interpolate a single variable to the dense HK grid.

        Faster than ``interpolate_single`` when only one channel is needed.
        """
        return _interpolate_single_variable_static(
            (self.XI, self.land_mask, self.grid_shape, self._buffer_bounds),
            nc_path, var)

    # ------------------------------------------------------------------
    # Elev appending
    # ------------------------------------------------------------------
    def _append_elev_worker(self, args) -> tuple:
        """Read existing (3,H,W) file, interpolate elev, save as (4,H,W)."""
        global _WORKER_CTX
        if _WORKER_CTX is None:
            _WORKER_CTX = (self.XI, self.land_mask, self.grid_shape,
                           self._buffer_bounds)
        return _append_elev_worker_static(args)

    def append_elev(self, timestamps: list = None, num_workers: int = 1,
                    force: bool = False):
        """Append (or re-compute) elev channel of preinterpolated files.

        Reads each (3, H, W) file, interpolates elev from the source .nc,
        and saves as (4, H, W) — hs, tp, dm, elev.

        force=True re-computes the elev channel even for 4-channel files
        (replaces channel 3) — used after fixing the clip<0 bug that
        truncated negative (low-tide) elev values to zero.
        """
        if timestamps is not None:
            ts_map = {ts: self._ts_to_path[ts] for ts in timestamps
                      if ts in self._ts_to_path}
        else:
            ts_map = self._ts_to_path

        tasks = []
        for ts, nc_path in ts_map.items():
            year = ts[:4]
            out_path = os.path.join(self.output_dir, year, f'wave_{ts}.npy')
            if not os.path.exists(out_path):
                continue  # skip — file hasn't been preinterpolated yet
            tasks.append((ts, nc_path, out_path, force))

        if not tasks:
            print("No existing preinterpolated files found to append elev to.")
            return

        print(f"Files to append elev: {len(tasks)} (force={force})")
        t0 = time.time()

        if num_workers > 1:
            # Pass the small interpolation context as a module global (fork
            # inheritance) instead of the bound method — pickling the whole
            # instance per task is ~0.25 s/task and dominates the runtime.
            global _WORKER_CTX
            _WORKER_CTX = (self.XI, self.land_mask, self.grid_shape,
                           self._buffer_bounds)
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                futures = {ex.submit(_append_elev_worker_static, t): t[0]
                           for t in tasks}
                results = []
                for f in tqdm(as_completed(futures), total=len(tasks),
                              desc='Appending elev'):
                    results.append(f.result())
        else:
            results = []
            for task in tqdm(tasks, desc='Appending elev'):
                results.append(self._append_elev_worker(task))

        ok = sum(1 for r in results if r[1])
        already = sum(1 for r in results if r[2] == 'already_4ch')
        fail = sum(1 for r in results if not r[1])
        elapsed = time.time() - t0
        print(f"\nDone. {ok} OK ({already} already 4ch), {fail} failed "
              f"in {elapsed:.1f}s ({elapsed/len(results):.2f}s/file)")

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------
    def _interpolate_worker(self, args) -> tuple:
        """Worker: interpolate one file and save. Returns (ts, success)."""
        ts, nc_path, out_path = args
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            grid = self.interpolate_single(nc_path)
            np.save(out_path, grid)
            return (ts, True)
        except Exception as e:
            print(f"[ERROR] {ts}: {e}")
            return (ts, False)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def run(
        self,
        timestamps: list = None,
        num_workers: int = 1,
        skip_existing: bool = True,
    ):
        """Interpolate and save multiple timestamps.

        Parameters
        ----------
        timestamps : list of str or None
            Timestamp strings 'YYYYMMDDTHH'. If None, processes all available.
        num_workers : int
            Number of parallel worker processes.
        skip_existing : bool
            If True, skip files that already exist with the correct shape.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Resolve timestamps
        if timestamps is not None:
            ts_map = {ts: self._ts_to_path[ts] for ts in timestamps
                      if ts in self._ts_to_path}
            if not ts_map:
                raise ValueError("No matching timestamps found in source_dir")
        else:
            ts_map = self._ts_to_path

        n_lat, n_lon = self.grid_shape
        n_vars = len(self.variables)
        expected_shape = (n_vars, n_lat, n_lon)
        expected_land_nan = self.land_mask.sum() * n_vars

        # Build task list
        tasks = []
        skipped = 0
        for ts, nc_path in ts_map.items():
            year = ts[:4]
            out_path = os.path.join(self.output_dir, year, f'wave_{ts}.npy')
            if skip_existing and os.path.exists(out_path):
                try:
                    d = np.load(out_path)
                    if d.shape == expected_shape and np.isnan(d).sum() == expected_land_nan:
                        skipped += 1
                        continue
                except Exception:
                    pass
            tasks.append((ts, nc_path, out_path))

        if skipped:
            print(f"Skipping {skipped} already-processed files")
        if not tasks:
            print("All files already processed — nothing to do.")
            return

        print(f"Files to process: {len(tasks)}")
        t0 = time.time()

        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                futures = {
                    ex.submit(self._interpolate_worker, task): task[0]
                    for task in tasks
                }
                results = []
                for f in tqdm(as_completed(futures), total=len(tasks),
                              desc='Interpolating'):
                    results.append(f.result())
        else:
            results = []
            for task in tqdm(tasks, desc='Interpolating'):
                results.append(self._interpolate_worker(task))

        ok = sum(r[1] for r in results)
        fail = len(results) - ok
        elapsed = time.time() - t0
        print(f"\nDone. {ok} OK, {fail} failed in {elapsed:.1f}s "
              f"({elapsed / len(results):.2f}s/file)")

        # Quick verification on first saved file
        if ok > 0:
            first_out = tasks[0][2]
            d = np.load(first_out)
            print(f"Verification ({os.path.basename(first_out)}): "
                  f"shape={d.shape}, dtype={d.dtype}, "
                  f"range=[{np.nanmin(d):.3f}, {np.nanmax(d):.3f}], "
                  f"land_NaN={np.isnan(d).sum()}/{d.size}")

    # ------------------------------------------------------------------
    # Build aggregated train / test cache files
    # ------------------------------------------------------------------
    def build_cache(
        self,
        stats_file: str = None,
        test_years: tuple = (2021, 2022),
        normalize: bool = True,
    ):
        """Concatenate per-timestamp .npy files into train/test cache arrays.

        Uses WaveDataset to obtain the correct, filtered file ordering so that
        the cache index matches what the training loop expects.

        Parameters
        ----------
        stats_file : str or None
            Path to wave_data_stats.json for normalisation constants.
            If None and normalize=True, stats are computed on the fly.
        test_years : tuple
            Years assigned to the test split.
        normalize : bool
            If True, apply hs/tp log1p+zscore and dm→sin normalisation.
        """
        import json
        from modules.datasets.wave_dataset import WaveDataset

        n_lat, n_lon = self.grid_shape
        n_vars = len(self.variables)

        # Load or skip normalisation stats
        if normalize:
            if stats_file and os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                hs_log_mean = stats['hs_log_mean']
                hs_log_std = stats['hs_log_std']
                tp_log_mean = stats['tp_log_mean']
                tp_log_std = stats['tp_log_std']
            else:
                # Will compute on the fly from raw data
                hs_log_mean = hs_log_std = tp_log_mean = tp_log_std = None

        for split in ['train', 'test']:
            t0 = time.time()
            print(f"\n{'=' * 60}")
            print(f"Building {split} cache...")

            ds = WaveDataset(
                data_dir=self.source_dir,
                era5_dir='./data/era5/nc_files',
                variables=self.variables,
                split=split,
                load_era5=True,
                era5_region=2,
                num_era5_frames=2,
            )
            file_list = ds.file_list
            n_samples = len(file_list)
            print(f"  {n_samples} samples in {time.time() - t0:.1f}s")

            cache_path = os.path.join(self.output_dir, f'wave_zoom_{split}.npy')
            cache = np.lib.format.open_memmap(
                cache_path, mode='w+', dtype=np.float32,
                shape=(n_samples, n_vars, n_lat, n_lon),
            )

            for i, fpath in enumerate(tqdm(file_list, desc=f'  {split}')):
                fname = os.path.basename(fpath)
                ts = fname.replace('wave_', '').replace('.nc', '')
                year = ts[:4]
                npy_path = os.path.join(self.output_dir, year, f'wave_{ts}.npy')

                if not os.path.exists(npy_path):
                    raise FileNotFoundError(f"Missing zoom preinterp file: {npy_path}")

                raw = np.load(npy_path)
                if normalize:
                    raw = self._normalize_sample(raw, hs_log_mean, hs_log_std,
                                                 tp_log_mean, tp_log_std)
                cache[i] = raw

            cache.flush()
            del cache

            size_gb = os.path.getsize(cache_path) / 1e9
            elapsed = time.time() - t0
            print(f"  Saved {cache_path} ({size_gb:.2f} GB) in {elapsed:.1f}s "
                  f"({elapsed / n_samples * 1000:.1f}ms/sample)")

    @staticmethod
    def _normalize_sample(raw, hs_log_mean, hs_log_std, tp_log_mean, tp_log_std):
        """Apply hs/tp log1p+zscore and dm→sin normalization to one sample."""
        result = raw.copy()
        # hs
        valid = ~np.isnan(raw[0])
        if valid.any():
            clipped = np.maximum(raw[0][valid], 0.0)
            result[0][valid] = (np.log1p(clipped) - hs_log_mean) / hs_log_std
        # tp
        valid = ~np.isnan(raw[1])
        if valid.any():
            clipped = np.maximum(raw[1][valid], 0.0)
            result[1][valid] = (np.log1p(clipped) - tp_log_mean) / tp_log_std
        # dm → sin
        valid = ~np.isnan(raw[2])
        if valid.any():
            result[2][valid] = np.sin(raw[2][valid] * np.pi / 180.0)
        return result

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------
    def visualize_sample(
        self,
        nc_path: str,
        save_path: str = None,
        dpi: int = 150,
        figsize: tuple = (18, 10),
    ):
        """Interpolate one timestamp and produce a diagnostic figure.

        Left column: scattered source nodes (coloured by variable value).
        Right column: interpolated dense grid.

        Parameters
        ----------
        nc_path : str
            Path to a wave_*.nc file.
        save_path : str or None
            If given, save the figure to this path.
        dpi : int
        figsize : tuple
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ds = xr.load_dataset(nc_path)
        grid = self.interpolate_single(nc_path)

        min_lon, max_lon, min_lat, max_lat = self.hk_region

        fig, axes = plt.subplots(
            3, 2, figsize=figsize,
            subplot_kw={'projection': ccrs.PlateCarree()},
        )

        var_info = {
            'hs':  {'label': 'Hs (m)',        'cmap': 'Spectral_r', 'vmin': 0, 'vmax': 4},
            'tp':  {'label': 'Tp (s)',         'cmap': 'viridis',    'vmin': 0, 'vmax': 16},
            'dm':  {'label': 'Dm (°)',         'cmap': 'twilight',   'vmin': 0, 'vmax': 360},
        }

        for row, var in enumerate(self.variables):
            vi = var_info.get(var, {})
            # --- Left: scattered nodes ---
            ax = axes[row, 0]
            ax.set_extent([min_lon, max_lon, min_lat, max_lat])
            ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=0)
            ax.coastlines(resolution='10m', linewidth=0.5, zorder=1)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='grey', alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False

            vals = ds[var].values
            sc = ax.scatter(
                ds['lon'].values, ds['lat'].values,
                c=vals, cmap=vi.get('cmap', 'turbo'),
                s=3, vmin=vi.get('vmin'), vmax=vi.get('vmax'),
                transform=ccrs.PlateCarree(),
            )
            plt.colorbar(sc, ax=ax, label=vi.get('label', var), shrink=0.85)
            ax.set_title(f'{var.upper()} — Scattered nodes ({len(vals)} pts)')

            # --- Right: interpolated grid ---
            ax = axes[row, 1]
            ax.set_extent([min_lon, max_lon, min_lat, max_lat])
            ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=0)
            ax.coastlines(resolution='10m', linewidth=0.5, zorder=1)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='grey', alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False

            im = ax.pcolormesh(
                self.LON2D, self.LAT2D, grid[row],
                cmap=vi.get('cmap', 'turbo'),
                vmin=vi.get('vmin'), vmax=vi.get('vmax'),
                transform=ccrs.PlateCarree(), shading='auto',
            )
            plt.colorbar(im, ax=ax, label=vi.get('label', var), shrink=0.85)
            ax.set_title(f'{var.upper()} — Interpolated grid '
                         f'{self.grid_shape[1]}×{self.grid_shape[0]}')

        # Extract timestamp for suptitle
        ts_str = os.path.basename(nc_path).replace('wave_', '').replace('.nc', '')
        fig.suptitle(f'HK Zoom Interpolation — {ts_str}', fontsize=13, y=1.01)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Saved: {save_path}")

        plt.show()
        ds.close()
        return fig
