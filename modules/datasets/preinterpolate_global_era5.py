"""
Global ERA5 Regridder: regrid ERA5 surface variables from native 0.25°
(721 × 1440) to a 0.5° regular lat/lon grid (360 × 720).

Target grid:
  - Latitude:  89.5° → -90° (360 points, 0.5° spacing, drops 90°N pole row)
  - Longitude: 0° → 360° (720 points, 0.5° spacing)

Only u10 and v10 are extracted (OceanCastNet atmospheric inputs).

Method: xarray .interp() with bilinear interpolation (method='linear').

Usage:
    from modules.datasets.preinterpolate_global_era5 import GlobalERA5Regridder

    regridder = GlobalERA5Regridder()
    regridder.run(start='19590101T00', end='20231231T21', num_workers=8)
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_DIR = './data/era5/global_nc_files'
DEFAULT_OUTPUT_DIR = './data/era5/global_0.5deg'
DEFAULT_TARGET_SHAPE = (360, 720)  # (n_lat, n_lon) — 0.5° resolution
DEFAULT_VARIABLES = ('10m_u_component_of_wind', '10m_v_component_of_wind')


def _regrid_worker(args) -> Tuple[str, bool]:
    """Worker function (module-level for multiprocessing).

    Loads an ERA5 NC file, regrids to 0.5°, and saves as .npy.

    Parameters
    ----------
    args : tuple
        (ts, nc_path, out_path, target_lat, target_lon, variables)

    Returns
    -------
    tuple : (ts, success_bool)
    """
    import xarray as xr

    ts, nc_path, out_path, target_lat, target_lon, variables = args
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        ds = xr.load_dataset(nc_path, engine='netcdf4')

        # Detect coordinate names ('latitude'/'lat', 'longitude'/'lon')
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'

        # Bilinear interpolation to target 0.5° grid
        ds_regrid = ds.interp(
            {lat_name: target_lat, lon_name: target_lon},
            method='linear',
        )

        # Stack variables: [u10, v10] → (2, 360, 720)
        frames = []
        for var_name in variables:
            vals = ds_regrid[var_name].values
            if vals.ndim == 3:
                # (time, lat, lon) — squeeze time dim if present
                vals = vals.squeeze(0)
            frames.append(vals.astype(np.float32))

        result = np.stack(frames, axis=0)  # (2, 360, 720)

        ds.close()
        ds_regrid.close()

        np.save(out_path, result)
        return (ts, True)
    except Exception as e:
        print(f"[ERROR] {ts}: {e}")
        return (ts, False)


class GlobalERA5Regridder:
    """Regrid global ERA5 from 0.25° to 0.5° regular grid.

    Parameters
    ----------
    source_dir : str
        Root directory containing per-year subdirectories of era5_*.nc files.
    output_dir : str
        Root output directory. Per-year subdirectories are created under it.
    target_shape : tuple
        (n_lat, n_lon) pixel dimensions of the output grid.
    variables : tuple of str
        ERA5 variable names to extract and regrid.
    """

    def __init__(
        self,
        source_dir: str = DEFAULT_SOURCE_DIR,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        target_shape: Tuple[int, int] = DEFAULT_TARGET_SHAPE,
        variables: tuple = DEFAULT_VARIABLES,
    ):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.target_shape = target_shape
        self.variables = list(variables)

        n_lat, n_lon = target_shape

        # Build target 0.5° grid
        # Latitude: 89.5° → -90° (360 points, drops 90°N degenerate pole row)
        # Longitude: 0° → 360° (720 points, matches ERA5 0-360 convention)
        self.lon_vals = np.linspace(0, 360, n_lon, endpoint=False, dtype=np.float64)
        self.lat_vals = np.linspace(89.5, -90, n_lat, dtype=np.float64)

        # Build index of available NC files
        self._ts_to_path = self._build_file_index()

        self._report_init()

    def _report_init(self):
        n_lat, n_lon = self.target_shape
        print(f"GlobalERA5Regridder ready:")
        print(f"  Target grid:  {n_lon} (lon) × {n_lat} (lat) = {n_lat * n_lon:,} pixels")
        print(f"  Resolution:   {self.lon_vals[1] - self.lon_vals[0]:.3f}° lon × "
              f"{self.lat_vals[0] - self.lat_vals[1]:.3f}° lat")
        print(f"  Lat range:    [{self.lat_vals[-1]:.1f}, {self.lat_vals[0]:.1f}] N→S")
        print(f"  Lon range:    [{self.lon_vals[0]:.1f}, {self.lon_vals[-1]:.1f}]")
        print(f"  Variables:    {self.variables}")
        print(f"  Source dir:   {self.source_dir}  ({len(self._ts_to_path)} NC files)")
        print(f"  Output dir:   {self.output_dir}")

    # ------------------------------------------------------------------
    # File index
    # ------------------------------------------------------------------
    def _build_file_index(self) -> dict:
        """Scan source directory for per-year/era5_*.nc files.

        Returns
        -------
        dict : timestamp (YYYYMMDDTHH) → file path
        """
        ts_to_path = {}
        if not os.path.isdir(self.source_dir):
            return ts_to_path

        with os.scandir(self.source_dir) as it:
            for entry in it:
                if entry.is_dir() and len(entry.name) == 4 and entry.name.isdigit():
                    year_dir = entry.path
                    with os.scandir(year_dir) as year_it:
                        for f in year_it:
                            if f.is_file() and f.name.startswith('era5_') and f.name.endswith('.nc'):
                                # era5_YYYYMMDDTHH.nc → YYYYMMDDTHH
                                ts = f.name.replace('era5_', '').replace('.nc', '')
                                ts_to_path[ts] = f.path
        return ts_to_path

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def run(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        timestamps: Optional[list] = None,
        num_workers: int = 1,
        skip_existing: bool = True,
    ):
        """Regrid multiple ERA5 timestamps.

        Parameters
        ----------
        start : str or None
            Start timestamp 'YYYYMMDDTHH' (inclusive).
        end : str or None
            End timestamp 'YYYYMMDDTHH' (inclusive).
        timestamps : list of str or None
            Specific timestamps. If None, uses start/end or all available.
        num_workers : int
            Number of parallel worker processes.
        skip_existing : bool
            If True, skip files that already exist with the correct shape.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Select timestamps
        if timestamps is not None:
            selected = {ts: self._ts_to_path[ts] for ts in timestamps if ts in self._ts_to_path}
        elif start is not None and end is not None:
            selected = {
                ts: path for ts, path in self._ts_to_path.items()
                if start <= ts <= end
            }
        else:
            selected = self._ts_to_path

        if not selected:
            print("No matching timestamps found.")
            return

        # Build task list, skip existing
        tasks = []
        skipped = 0
        expected_shape = (len(self.variables), *self.target_shape)
        for ts in sorted(selected):
            nc_path = selected[ts]
            year = ts[:4]
            out_path = os.path.join(self.output_dir, year, f'era5_{ts}.npy')
            if skip_existing and os.path.exists(out_path):
                try:
                    existing = np.load(out_path)
                    if existing.shape == expected_shape:
                        skipped += 1
                        continue
                except Exception:
                    pass
            tasks.append((
                ts, nc_path, out_path,
                self.lat_vals, self.lon_vals,
                self.variables,
            ))

        if skipped:
            print(f"Skipping {skipped} already-processed files")
        if not tasks:
            print("All files already processed — nothing to do.")
            return

        print(f"Files to process: {len(tasks)}")
        t0 = time.time()

        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                futures = {ex.submit(_regrid_worker, t): t[0] for t in tasks}
                results = []
                for f in tqdm(as_completed(futures), total=len(tasks),
                              desc='Regridding ERA5'):
                    results.append(f.result())
        else:
            results = []
            for task in tqdm(tasks, desc='Regridding ERA5'):
                results.append(_regrid_worker(task))

        ok = sum(r[1] for r in results)
        fail = len(results) - ok
        elapsed = time.time() - t0
        print(f"\nDone. {ok} OK, {fail} failed in {elapsed:.1f}s "
              f"({elapsed / len(results):.2f}s/file avg)")

        # Quick verification on first output
        if ok > 0 and tasks:
            first_out = tasks[0][2]
            d = np.load(first_out)
            print(f"Verification ({os.path.basename(first_out)}): "
                  f"shape={d.shape}, dtype={d.dtype}, "
                  f"u10 range=[{d[0].min():.3f}, {d[0].max():.3f}], "
                  f"v10 range=[{d[1].min():.3f}, {d[1].max():.3f}]")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    regridder = GlobalERA5Regridder()
    if regridder._ts_to_path:
        first_ts = sorted(regridder._ts_to_path)[0]
        first_path = regridder._ts_to_path[first_ts]
        print(f"\nTesting regrid on {first_ts}: {first_path}")
        t0 = time.time()
        import xarray as xr
        ds = xr.load_dataset(first_path, engine='netcdf4')
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        ds_regrid = ds.interp(
            {lat_name: regridder.lat_vals, lon_name: regridder.lon_vals},
            method='linear',
        )
        frames = []
        for var_name in regridder.variables:
            vals = ds_regrid[var_name].values
            if vals.ndim == 3:
                vals = vals.squeeze(0)
            frames.append(vals.astype(np.float32))
        result = np.stack(frames, axis=0)
        print(f"  Regrid took {time.time() - t0:.2f}s, shape={result.shape}, "
              f"u10 range=[{result[0].min():.3f}, {result[0].max():.3f}], "
              f"v10 range=[{result[1].min():.3f}, {result[1].max():.3f}]")
        ds.close()
        ds_regrid.close()
