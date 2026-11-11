"""
Global Hindcast Preinterpolator: interpolate global hindcast wave data from
unstructured mesh nodes (~652k) to a regular 0.25° lat/lon grid (721 × 1440).

Target grid matches the ERA5 0.25° resolution and Aurora model input format:
  - Latitude: 90° → -90° (721 points, decreasing, as Aurora requires)
  - Longitude: 0° → 360° (1440 points, increasing)

Method: cKDTree in 3D Cartesian space + inverse-distance-weighted (IDW)
interpolation from the K nearest source mesh nodes.

Usage:
    from modules.datasets.preinterpolate_global_hindcast import GlobalHindcastPreinterpolator

    interp = GlobalHindcastPreinterpolator()
    interp.precompute_weights(K=9)  # one-time precomputation
    grid = interp.interpolate_single('data/global_hindcast/npz_files/2018/global_hindcast_20180101_00.npz')
    # grid.shape -> (721, 1440), float32, land=NaN

    # Batch processing:
    interp.run(start='20180101T00', end='20181231T21', num_workers=8)
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_DIR = './data/global_hindcast/npz_files'
DEFAULT_COORD_FILE = './data/global_hindcast/global_coordinates.npy'
DEFAULT_OUTPUT_DIR = './data/global_hindcast/interpolated_grid'
DEFAULT_WEIGHTS_DIR = './data/global_hindcast/interpolation_weights'
DEFAULT_TARGET_SHAPE = (721, 1440)  # (n_lat, n_lon) — 0.25° resolution
DEFAULT_VARIABLE = ['hs', 'mwd', 'ssh']  # variables to interpolate; single str for backward compat
DEFAULT_K = 9  # number of nearest neighbors for IDW interpolation
DEFAULT_LAND_DISTANCE = 0.5  # ° — grid cells farther than this from any node → land


def latlon_to_cartesian(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Convert latitude/longitude to 3D Cartesian coordinates on unit sphere.

    Args:
        lats: Latitudes in degrees, shape (...).
        lons: Longitudes in degrees, shape (...).

    Returns:
        np.ndarray: Cartesian coordinates, shape (..., 3).
    """
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    x = np.cos(lats_rad) * np.cos(lons_rad)
    y = np.cos(lats_rad) * np.sin(lons_rad)
    z = np.sin(lats_rad)
    return np.stack([x, y, z], axis=-1)


class GlobalHindcastPreinterpolator:
    """Interpolate global unstructured hindcast wave data to a regular lat/lon grid.

    Parameters
    ----------
    source_dir : str
        Directory containing per-timestamp NPZ files (may be in year subdirectories).
    coord_file : str
        Path to the global coordinates NumPy file (N, 2) with [lat, lon] per row.
    output_dir : str
        Root output directory. Per-year subdirectories are created under it.
    weights_dir : str
        Directory where precomputed KDTree neighbour indices and IDW weights are stored.
    target_shape : tuple
        (n_lat, n_lon) pixel dimensions of the output grid.
    variable : str or list of str
        Variable name(s) in the NPZ files to interpolate.
        Single str → output shape (H, W). List → output shape (C, H, W).
    k_neighbors : int
        Number of nearest source nodes for IDW interpolation.
    land_distance_threshold : float
        Grid cells whose nearest source node is farther than this (in degrees)
        are masked as land (NaN).
    """

    def __init__(
        self,
        source_dir: str = DEFAULT_SOURCE_DIR,
        coord_file: str = DEFAULT_COORD_FILE,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        weights_dir: str = DEFAULT_WEIGHTS_DIR,
        target_shape: Tuple[int, int] = DEFAULT_TARGET_SHAPE,
        variable: str = DEFAULT_VARIABLE,
        k_neighbors: int = DEFAULT_K,
        land_distance_threshold: float = DEFAULT_LAND_DISTANCE,
    ):
        self.source_dir = source_dir
        self.coord_file = coord_file
        self.output_dir = output_dir
        self.weights_dir = weights_dir
        self.target_shape = target_shape  # (n_lat, n_lon)
        self.variable = variable
        self.k_neighbors = k_neighbors
        self.land_distance_threshold = land_distance_threshold

        n_lat, n_lon = target_shape

        # Build target regular grid matching Aurora/ERA5 conventions:
        #   lat: decreasing, lon: increasing (0 → 360)
        self.lon_vals = np.linspace(0, 360, n_lon, endpoint=False, dtype=np.float64)
        # lat: by default inclusive at both poles (e.g. 361 → 90°…-90° at 0.5°).
        # When n_lat == 360 (0.5° without a pole row) the range is shifted by
        # half a cell so that the spacing stays exactly 0.5°.
        if n_lat == 360:
            self.lat_vals = np.linspace(89.5, -90, n_lat, dtype=np.float64)
        else:
            self.lat_vals = np.linspace(90, -90, n_lat, dtype=np.float64)
        self.LON2D, self.LAT2D = np.meshgrid(self.lon_vals, self.lat_vals)
        # Flattened target points: (n_lat * n_lon, 2) -> [lon, lat]
        self._target_flat = np.column_stack([self.LON2D.ravel(), self.LAT2D.ravel()])

        # Load source node coordinates (these are fixed)
        if not os.path.exists(coord_file):
            raise FileNotFoundError(f"Coordinate file not found: {coord_file}")
        self.source_coords = np.load(coord_file)  # (N, 2) — [lat, lon]
        self.num_source_nodes = self.source_coords.shape[0]

        # Convert source nodes to 3D Cartesian for KDTree
        self._src_cartesian = latlon_to_cartesian(
            self.source_coords[:, 0],
            self.source_coords[:, 1],
        ).astype(np.float64)

        # Build index of available NPZ files: timestamp -> npz_path
        self._ts_to_path = self._build_file_index()

        # Precomputed weights: loaded lazily
        self._neighbor_indices: Optional[np.ndarray] = None  # (N_target, K)
        self._neighbor_weights: Optional[np.ndarray] = None  # (N_target, K)
        self._land_mask: Optional[np.ndarray] = None         # (N_target,) bool

        self._report_init()

    def _report_init(self):
        n_lat, n_lon = self.target_shape
        print(f"GlobalHindcastPreinterpolator ready:")
        print(f"  Source nodes: {self.num_source_nodes:,}")
        print(f"  Target grid:  {n_lon} (lon) × {n_lat} (lat) = {n_lat * n_lon:,} pixels")
        print(f"  Resolution:   {(self.lon_vals[1] - self.lon_vals[0]):.3f}° lon × "
              f"{(self.lat_vals[0] - self.lat_vals[1]):.3f}° lat")
        print(f"  Lat range:    [{self.lat_vals[-1]:.1f}, {self.lat_vals[0]:.1f}] "
              f"(decreasing: {np.all(self.lat_vals[1:] < self.lat_vals[:-1])})")
        print(f"  Lon range:    [{self.lon_vals[0]:.1f}, {self.lon_vals[-1]:.1f}] "
              f"(increasing: {np.all(self.lon_vals[1:] > self.lon_vals[:-1])})")
        print(f"  KDTree K:     {self.k_neighbors}")
        print(f"  Land thresh:  {self.land_distance_threshold}°")
        print(f"  Variable:     {self.variable}")
        print(f"  Source dir:   {self.source_dir}  ({len(self._ts_to_path)} NPZ files)")
        print(f"  Output dir:   {self.output_dir}")

    # ------------------------------------------------------------------
    # File index
    # ------------------------------------------------------------------
    def _build_file_index(self) -> dict:
        """Build a mapping from timestamp string to NPZ file path.

        Supports two layouts:
          - Flat:      source_dir/global_hindcast_YYYYMMDD_HH.npz
          - Year sub:  source_dir/YYYY/global_hindcast_YYYYMMDD_HH.npz

        Uses ``os.scandir`` for efficiency with large directories.
        """
        ts_to_path = {}

        def _scan_dir(root: str):
            if not os.path.isdir(root):
                return
            with os.scandir(root) as it:
                for entry in it:
                    if entry.is_file() and entry.name.startswith('global_hindcast_') and entry.name.endswith('.npz'):
                        # Extract timestamp: global_hindcast_YYYYMMDD_HH.npz
                        ts = entry.name.replace('global_hindcast_', '').replace('.npz', '')
                        parts = ts.split('_')
                        if len(parts) == 2:
                            ts_key = f"{parts[0]}T{parts[1]}"  # YYYYMMDDTHH
                            ts_to_path[ts_key] = entry.path

        # Scan flat layout
        _scan_dir(self.source_dir)

        # Scan year-subdirectory layout (only check dirs with 4-digit names)
        if os.path.isdir(self.source_dir):
            with os.scandir(self.source_dir) as it:
                for entry in it:
                    if entry.is_dir() and len(entry.name) == 4 and entry.name.isdigit():
                        _scan_dir(entry.path)

        return ts_to_path

    # ------------------------------------------------------------------
    # Precompute KDTree neighbour indices & IDW weights (run once)
    # ------------------------------------------------------------------
    def precompute_weights(
        self,
        K: Optional[int] = None,
        force: bool = False,
        land_distance: Optional[float] = None,
    ):
        """Build KDTree, find nearest K source nodes for each target grid cell,
        compute IDW weights, and derive a land mask.  Results are saved to
        ``weights_dir`` and re-loaded on subsequent runs.

        Parameters
        ----------
        K : int or None
            Number of nearest neighbours (defaults to ``self.k_neighbors``).
        force : bool
            If True, recompute even if cached weight files exist.
        land_distance : float or None
            Override ``self.land_distance_threshold``.
        """
        if K is not None:
            self.k_neighbors = K
        if land_distance is not None:
            self.land_distance_threshold = land_distance

        K = self.k_neighbors
        n_target = self.target_shape[0] * self.target_shape[1]
        thresh = self.land_distance_threshold

        idx_path = os.path.join(self.weights_dir, f'indices_k{K}.npy')
        wgt_path = os.path.join(self.weights_dir, f'weights_k{K}.npy')
        msk_path = os.path.join(self.weights_dir, f'land_mask_t{thresh:.1f}.npy')

        os.makedirs(self.weights_dir, exist_ok=True)

        if not force and os.path.exists(idx_path) and os.path.exists(wgt_path) and os.path.exists(msk_path):
            print(f"Loading precomputed weights from {self.weights_dir} ...")
            self._neighbor_indices = np.load(idx_path)
            self._neighbor_weights = np.load(wgt_path)
            self._land_mask = np.load(msk_path)
            print(f"  Indices: {self._neighbor_indices.shape}, "
                  f"weights: {self._neighbor_weights.shape}, "
                  f"land mask: {self._land_mask.sum():,} / {self._land_mask.size:,} cells")
            return

        # Build KDTree from source nodes (3D Cartesian)
        print(f"Building KDTree from {self.num_source_nodes:,} source nodes ...")
        t0 = time.time()
        tree = cKDTree(self._src_cartesian)
        print(f"  KDTree built in {time.time() - t0:.1f}s")

        # Convert target grid points to 3D Cartesian
        print(f"Converting {n_target:,} target grid points to 3D Cartesian ...")
        t0 = time.time()
        target_cart = latlon_to_cartesian(
            self._target_flat[:, 1],  # lat
            self._target_flat[:, 0],  # lon
        ).astype(np.float64)
        print(f"  Conversion done in {time.time() - t0:.1f}s")

        # Query K nearest neighbours for all target points
        # Process in chunks to manage memory
        chunk_size = 50_000  # process 50k target points at a time
        n_chunks = int(np.ceil(n_target / chunk_size))

        indices = np.empty((n_target, K), dtype=np.int32)
        distances = np.empty((n_target, K), dtype=np.float64)
        # chord_distance = 2 * sin(great_circle_dist / 2)
        # great_circle_dist = 2 * arcsin(chord_dist / 2)
        # For IDW we use great-circle distance ~ chord distance (monotonic, so weights same)

        print(f"Querying {K} nearest neighbours for {n_target:,} target points "
              f"({n_chunks} chunks of {chunk_size}) ...")
        t0 = time.time()
        for i in tqdm(range(n_chunks), desc='  KDTree query'):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_target)
            dists, idxs = tree.query(target_cart[start:end], k=K)
            # dists here are 3D Euclidean (= chord length on unit sphere)
            indices[start:end] = idxs.astype(np.int32)
            distances[start:end] = dists

        elapsed = time.time() - t0
        print(f"  Query done in {elapsed:.1f}s ({elapsed / n_target * 1e6:.1f} µs/point)")

        # Compute IDW weights
        print("Computing IDW weights ...")
        t0 = time.time()
        eps = 1e-10
        weights = 1.0 / (distances + eps)  # (N_target, K)
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights /= weights_sum  # normalize
        weights = weights.astype(np.float32)
        print(f"  Weights computed in {time.time() - t0:.1f}s")

        # Compute land mask from min distance per grid cell
        # Convert 3D chord distance to approximate great-circle degrees:
        #   chord = 2 * sin(gc_rad / 2)  =>  gc_deg = rad2deg(2 * arcsin(chord/2))
        print("Computing land mask ...")
        t0 = time.time()
        min_chord = distances[:, 0]  # distance to nearest node
        # Clamp chord values to [-1, 1] for arcsin safety
        min_chord = np.clip(min_chord / 2.0, -1.0, 1.0)
        min_gc_deg = np.degrees(2.0 * np.arcsin(min_chord))
        land_mask = min_gc_deg > thresh
        print(f"  Land mask computed in {time.time() - t0:.1f}s")

        # Save
        print(f"Saving to {self.weights_dir} ...")
        np.save(idx_path, indices)
        np.save(wgt_path, weights)
        np.save(msk_path, land_mask)

        self._neighbor_indices = indices
        self._neighbor_weights = weights
        self._land_mask = land_mask

        n_land = land_mask.sum()
        n_ocean = (~land_mask).sum()
        print(f"  Saved: indices={indices.shape}, weights={weights.shape}, "
              f"land={n_land:,} ({n_land / land_mask.size * 100:.1f}%), "
              f"ocean={n_ocean:,} ({n_ocean / land_mask.size * 100:.1f}%)")

    # ------------------------------------------------------------------
    # Single timestep interpolation
    # ------------------------------------------------------------------
    def interpolate_single(self, npz_path: str) -> np.ndarray:
        """Interpolate one NPZ file to the regular grid.

        Parameters
        ----------
        npz_path : str
            Path to a global_hindcast_*.npz file.

        Returns
        -------
        np.ndarray, shape (n_lat, n_lon) if single variable,
                    shape (C, n_lat, n_lon) if multiple variables.
            dtype float32. Land cells are NaN.
        """
        if self._neighbor_indices is None or self._neighbor_weights is None:
            self.precompute_weights()

        data = np.load(npz_path)
        variables = self.variable if isinstance(self.variable, list) else [self.variable]

        channels = []
        for var in variables:
            if var not in data:
                raise KeyError(f"Variable '{var}' not found in {npz_path}. "
                               f"Available: {list(data.keys())}")
            src_vals = data[var].astype(np.float32)  # (N_source,)

            if len(src_vals) != self.num_source_nodes:
                raise ValueError(
                    f"Source node count mismatch: expected {self.num_source_nodes}, "
                    f"got {len(src_vals)} in {npz_path}"
                )

            # Weighted average using precomputed neighbours
            gathered = src_vals[self._neighbor_indices]  # (N_target, K)
            result = (gathered * self._neighbor_weights).sum(axis=1)  # (N_target,)
            result[self._land_mask] = np.nan

            n_lat, n_lon = self.target_shape
            channels.append(result.reshape(n_lat, n_lon).astype(np.float32))

        if len(channels) == 1:
            return channels[0]
        return np.stack(channels, axis=0)  # (C, H, W)

    # ------------------------------------------------------------------
    # Worker (for multiprocessing)
    # ------------------------------------------------------------------
    def _interpolate_worker(self, args) -> Tuple[str, bool]:
        """Process a single file: interpolate and save. Returns (ts, success)."""
        ts, npz_path, out_path = args
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            grid = self.interpolate_single(npz_path)
            np.savez_compressed(out_path, data=grid)
            return (ts, True)
        except Exception as e:
            print(f"[ERROR] {ts}: {e}")
            return (ts, False)

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
        """Interpolate and save multiple timestamps.

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
            If True, skip files that already exist.
        """
        # Ensure weights are precomputed
        if self._neighbor_indices is None:
            self.precompute_weights()

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
        for ts in sorted(selected):
            npz_path = selected[ts]
            year = ts[:4]
            out_path = os.path.join(self.output_dir, year, f'global_hindcast_{ts.replace("T", "_")}.npz')
            if skip_existing and os.path.exists(out_path):
                # Quick shape check
                try:
                    existing = np.load(out_path)
                    if existing.shape == self.target_shape:
                        skipped += 1
                        continue
                except Exception:
                    pass
            tasks.append((ts, npz_path, out_path))

        if skipped:
            print(f"Skipping {skipped} already-processed files")
        if not tasks:
            print("All files already processed — nothing to do.")
            return

        print(f"Files to process: {len(tasks)}")
        t0 = time.time()

        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                futures = {ex.submit(self._interpolate_worker, t): t[0] for t in tasks}
                results = []
                for f in tqdm(as_completed(futures), total=len(tasks),
                              desc='Interpolating global hindcast'):
                    results.append(f.result())
        else:
            results = []
            for task in tqdm(tasks, desc='Interpolating global hindcast'):
                results.append(self._interpolate_worker(task))

        ok = sum(r[1] for r in results)
        fail = len(results) - ok
        elapsed = time.time() - t0
        print(f"\nDone. {ok} OK, {fail} failed in {elapsed:.1f}s "
              f"({elapsed / len(results):.2f}s/file avg)")

        # Quick verification
        if ok > 0:
            first_out = tasks[0][2]
            d = np.load(first_out)
            ocean_vals = d[~self._land_mask.reshape(self.target_shape)]
            print(f"Verification ({os.path.basename(first_out)}): "
                  f"shape={d.shape}, dtype={d.dtype}, "
                  f"ocean range=[{np.nanmin(d):.4f}, {np.nanmax(d):.4f}], "
                  f"land_NaN={np.isnan(d).sum()}/{d.size}")

    # ------------------------------------------------------------------
    # Visualization helper
    # ------------------------------------------------------------------
    def visualize_sample(
        self,
        npz_path: str,
        save_path: Optional[str] = None,
        dpi: int = 150,
        figsize: tuple = (18, 6),
    ):
        """Interpolate one timestep and produce a diagnostic figure.

        Left: scattered source nodes (all nodes).
        Right: interpolated regular grid.

        Parameters
        ----------
        npz_path : str
            Path to a global_hindcast_*.npz file.
        save_path : str or None
            If given, save the figure to this path.
        dpi : int
        figsize : tuple
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        grid = self.interpolate_single(npz_path)

        # Load source values
        data = np.load(npz_path)
        src_vals = data[self.variable].astype(np.float32)

        fig, axes = plt.subplots(
            1, 2, figsize=figsize,
            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        )

        # Use interpolated grid's range as the unified colorbar scale
        vmax = np.nanmax(grid)

        # --- Left: scattered nodes (all nodes) ---
        ax = axes[0]
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=0)
        ax.coastlines(resolution='110m', linewidth=0.3, zorder=1)
        ax.set_title(f'Source nodes ({self.num_source_nodes:,} total)')

        sc = ax.scatter(
            self.source_coords[:, 1], self.source_coords[:, 0],
            c=src_vals, cmap='Spectral_r', s=0.1,
            vmin=0, vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        plt.colorbar(sc, ax=ax, label='SWH (m)', shrink=0.7)

        # --- Right: interpolated grid ---
        ax = axes[1]
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=0)
        ax.coastlines(resolution='110m', linewidth=0.3, zorder=1)
        ax.set_title(f'Interpolated grid {self.target_shape[1]}×{self.target_shape[0]} '
                     f'(0.25°, K={self.k_neighbors})')

        im = ax.pcolormesh(
            self.lon_vals, self.lat_vals, grid,
            cmap='Spectral_r', vmin=0, vmax=vmax,
            transform=ccrs.PlateCarree(), shading='auto',
        )
        plt.colorbar(im, ax=ax, label='SWH (m)', shrink=0.7)

        ts_str = os.path.basename(npz_path).replace('global_hindcast_', '').replace('.npz', '')
        fig.suptitle(f'Global Hindcast Interpolation — {ts_str}', fontsize=13, y=0.98)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Saved: {save_path}")

        plt.show()
        return fig


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    interp = GlobalHindcastPreinterpolator()
    interp.precompute_weights(K=9)
    # Test on first available file
    if interp._ts_to_path:
        first_ts = sorted(interp._ts_to_path)[0]
        first_path = interp._ts_to_path[first_ts]
        print(f"\nTesting interpolation on {first_ts}: {first_path}")
        t0 = time.time()
        g = interp.interpolate_single(first_path)
        print(f"  Interpolation took {time.time() - t0:.2f}s, "
              f"shape={g.shape}, ocean range=[{np.nanmin(g):.4f}, {np.nanmax(g):.4f}]")
