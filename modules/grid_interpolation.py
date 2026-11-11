"""
GridInterpolator: Interpolates between the 75x107 regular grid and
the 7788-node irregular mesh for the HK Zoom region.

Used in post-processing to convert grid-format model outputs back to
node-format for RMSE comparison against the irregular mesh model.
"""

import os
import time

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


class GridInterpolator:
    """Handles interpolation between regular grid and irregular mesh nodes.

    The HK Zoom grid is 75×107 cells covering approximately:
      lat [22.0, 22.7], lon [113.6, 114.6].
    The irregular mesh has 7788 nodes covering the same domain.
    """

    def __init__(
        self,
        grid_shape: tuple = (75, 107),
        lat_range: tuple = (22.0, 22.6998),
        lon_range: tuple = (113.6005, 114.5996),
        node_coord_path: str = './data/hindcast/hk_coordinates.npy',
        valid_mask_path: str = None,
    ):
        """Initialize the interpolator with grid and node coordinate geometry.

        Args:
            grid_shape: (H, W) of the regular grid.
            lat_range: (min_lat, max_lat) of the grid domain.
            lon_range: (min_lon, max_lon) of the grid domain.
            node_coord_path: Path to irregular node coordinates (N, 2) lat, lon.
            valid_mask_path: Path to precomputed valid mask (H, W) bool,
                             True = ocean pixel. If None, derived from data.
        """
        H, W = grid_shape
        self.grid_shape = grid_shape

        # Create grid pixel-centre coordinates
        lats_1d = np.linspace(lat_range[0], lat_range[1], H)
        lons_1d = np.linspace(lon_range[0], lon_range[1], W)
        self.lon_grid, self.lat_grid = np.meshgrid(lons_1d, lats_1d)
        # shape: (H, W) each

        # Flattened grid coordinates as source points: (H*W, 2) [lon, lat]
        self.grid_points = np.column_stack([
            self.lon_grid.ravel(),
            self.lat_grid.ravel(),
        ])

        # Load irregular node coordinates (target points)
        node_coords = np.load(node_coord_path)  # (N_nodes, 2) [lat, lon]
        # Convert to [lon, lat] for griddata
        self.node_coords = np.column_stack([
            node_coords[:, 1],  # lon
            node_coords[:, 0],  # lat
        ])  # (N_nodes, 2)
        self.num_nodes = node_coords.shape[0]

        # Valid mask: True = ocean pixel (valid wave data).
        # The land_mask file on disk uses True = land, so invert it.
        if valid_mask_path:
            raw_mask = np.load(valid_mask_path).astype(bool)
            # Heuristic: if fewer than half the pixels are True, it's a land mask
            if raw_mask.sum() < raw_mask.size * 0.5:
                self.valid_mask = ~raw_mask  # invert: True = ocean
            else:
                self.valid_mask = raw_mask
        else:
            self.valid_mask = None

        # Precompute valid source indices for fast interpolation
        self._valid_indices = None
        if self.valid_mask is not None:
            self._valid_indices = self.valid_mask.ravel()

    def set_valid_mask(self, valid_mask: np.ndarray):
        """Set the valid (ocean) pixel mask and precompute source indices.

        Args:
            valid_mask: (H, W) bool array, True = valid ocean pixel.
        """
        self.valid_mask = valid_mask.astype(bool)
        self._valid_indices = self.valid_mask.ravel()

    def set_valid_mask_from_data(self, grid_sample: np.ndarray):
        """Derive valid mask from a grid sample's NaN pattern.

        Args:
            grid_sample: (C, H, W) numpy array. Uses first channel to detect NaN.
        """
        valid = ~np.isnan(grid_sample[0])  # (H, W)
        self.set_valid_mask(valid)

    def grid_to_nodes(
        self, grid_data: np.ndarray, method: str = 'linear'
    ) -> np.ndarray:
        """Interpolate grid data to irregular node positions.

        Args:
            grid_data: (C, H, W) numpy array. NaN values at land pixels.
            method: Interpolation method ('linear', 'nearest', 'cubic').

        Returns:
            (N_nodes, C) numpy array of interpolated values.
        """
        C, H, W = grid_data.shape
        assert (H, W) == self.grid_shape, \
            f"Expected grid shape {self.grid_shape}, got ({H}, {W})"

        if self._valid_indices is not None:
            valid = self._valid_indices
        else:
            # Fall back to non-NaN points from the data
            valid = ~np.isnan(grid_data.reshape(C, -1)).any(axis=0)

        source_points = self.grid_points[valid]

        result = np.zeros((self.num_nodes, C), dtype=np.float32)
        for c in range(C):
            values = grid_data[c].ravel()[valid]
            if values.size == 0:
                result[:, c] = np.nan
                continue
            interpolated = griddata(
                source_points, values,
                self.node_coords, method=method,
            )
            # Fill any remaining NaN with nearest-neighbour
            nan_mask = np.isnan(interpolated)
            if nan_mask.any():
                interpolated[nan_mask] = griddata(
                    source_points, values,
                    self.node_coords[nan_mask], method='nearest',
                )
            result[:, c] = interpolated.astype(np.float32)

        return result

    def grid_batch_to_nodes(
        self, grid_batch: np.ndarray, method: str = 'linear'
    ) -> np.ndarray:
        """Interpolate a batch of grid samples to node positions.

        Args:
            grid_batch: (B, C, H, W) numpy array.
            method: Interpolation method.

        Returns:
            (B, N_nodes, C) numpy array.
        """
        B = grid_batch.shape[0]
        results = []
        for b in range(B):
            results.append(self.grid_to_nodes(grid_batch[b], method=method))
        return np.stack(results, axis=0)


# ---------------------------------------------------------------------------
# Global Grid → Node interpolator (KDTree + IDW, with precomputed weights)
# ---------------------------------------------------------------------------

class GlobalGridToNodeInterpolator:
    """KDTree + IDW: interpolate from global regular grid (720×1440) to
    irregular ocean nodes (~652k).

    Precomputes interpolation weights once (cKDTree over ocean grid cells,
    IDW from K nearest grid cells to each node), then applies them
    efficiently to any number of grid-format sample files.

    Conventions
    -----------
    * Model grid: 720×1440 at 0.25°.
      lat = linspace(90, -89.75, 720)  — north→south, drops south-pole row
      lon = linspace(0, 359.75, 1440)  — 0→360
    * Node coords: ``global_coordinates.npy`` — [lat, lon] with lon in
      [-180, 180).  Automatic conversion to 0–360 for grid matching.
    * Land mask: ``land_mask_720x1440.npy`` — (720, 1440) bool — True=land
      by convention (fewer True than False pixels).  Inverted internally to
      ``valid_mask`` (True=ocean).
    """

    DEFAULT_GRID_SHAPE = (720, 1440)
    DEFAULT_LAT_RANGE = (90.0, -89.75)   # north→south
    DEFAULT_LON_RANGE = (0.0, 359.75)    # 0→360
    DEFAULT_K_NEIGHBORS = 4

    def __init__(
        self,
        grid_shape: tuple = DEFAULT_GRID_SHAPE,
        lat_range: tuple = DEFAULT_LAT_RANGE,
        lon_range: tuple = DEFAULT_LON_RANGE,
        node_coord_path: str = './data/global_hindcast/global_coordinates.npy',
        land_mask_path: str = './data/global_hindcast/interpolation_weights/land_mask_720x1440.npy',
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
    ):
        H, W = grid_shape
        self.grid_shape = (H, W)
        self.n_grid = H * W

        # Build grid pixel-centre coordinates (lat-major, lon-fastest ravel)
        lats_1d = np.linspace(lat_range[0], lat_range[1], H, dtype=np.float64)
        lons_1d = np.linspace(lon_range[0], lon_range[1], W, dtype=np.float64)
        self.lon_grid, self.lat_grid = np.meshgrid(lons_1d, lats_1d)
        # self.lon_grid, self.lat_grid: (H, W)

        # Load / derive valid (ocean) mask
        if land_mask_path and os.path.exists(land_mask_path):
            raw = np.load(land_mask_path).astype(bool)
            if raw.ndim == 1:
                raw = raw.reshape(H, W)
            # Heuristic: if fewer than half True → True = land → invert
            if raw.sum() < raw.size * 0.5:
                self.valid_mask = ~raw   # True = ocean
            else:
                self.valid_mask = raw
        else:
            self.valid_mask = None

        # Ocean grid cells only — flat indices and (lat, lon) pairs
        if self.valid_mask is not None:
            self._ocean_flat = np.where(self.valid_mask.ravel())[0].astype(np.int32)
        else:
            self._ocean_flat = np.arange(self.n_grid, dtype=np.int32)

        ocean_lats = self.lat_grid.ravel()[self._ocean_flat]
        ocean_lons = self.lon_grid.ravel()[self._ocean_flat]
        self._ocean_xy = np.column_stack([ocean_lats, ocean_lons])  # [N_ocean, 2]
        self.n_ocean = len(self._ocean_flat)

        # Node coordinates — convert lon -180..180 → 0..360
        node_coords = np.load(node_coord_path)  # [N, 2] — [lat, lon]
        node_lats = node_coords[:, 0].astype(np.float64)
        node_lons = node_coords[:, 1].astype(np.float64)
        node_lons_360 = np.where(node_lons < 0, node_lons + 360, node_lons)
        self.node_xy = np.column_stack([node_lats, node_lons_360])  # [N, 2]
        self.n_nodes = node_coords.shape[0]

        self.k_neighbors = k_neighbors
        self._indices = None   # [N_nodes, K] flat grid indices
        self._weights = None   # [N_nodes, K] IDW weights

        print(f"GlobalGridToNodeInterpolator ready:")
        print(f"  Grid:       {H}×{W} = {self.n_grid:,} pixels")
        print(f"  Ocean:      {self.n_ocean:,} pixels")
        print(f"  Nodes:      {self.n_nodes:,}")
        print(f"  K:          {k_neighbors}")
        print(f"  Lat range:  [{lats_1d[0]:.2f}, {lats_1d[-1]:.2f}]")
        print(f"  Lon range:  [{lons_1d[0]:.2f}, {lons_1d[-1]:.2f}]")

    # ------------------------------------------------------------------
    # Precompute weights
    # ------------------------------------------------------------------
    def precompute_weights(
        self,
        weights_dir: str = './data/global_hindcast/interpolation_weights',
        force: bool = False,
    ):
        """Build KDTree over ocean grid cells, query K nearest per node,
        compute normalized IDW weights, and save to disk.

        Parameters
        ----------
        weights_dir : str
            Directory to save ``grid_to_node_k{K}.npz``.
        force : bool
            Recompute even if cached file exists.
        """
        K = self.k_neighbors
        out_path = os.path.join(weights_dir, f'grid_to_node_k{K}.npz')

        if not force and os.path.exists(out_path):
            print(f"Loading cached weights from {out_path}")
            self._load_weights(out_path)
            return

        os.makedirs(weights_dir, exist_ok=True)

        # Build KDTree on ocean grid cells
        print(f"Building KDTree on {self.n_ocean:,} ocean grid cells ...")
        t0 = time.time()
        tree = cKDTree(self._ocean_xy)
        print(f"  KDTree built in {time.time() - t0:.1f}s")

        # Query K nearest ocean grid cells for each node
        print(f"Querying K={K} nearest grid cells for {self.n_nodes:,} nodes ...")
        t0 = time.time()
        distances, ocean_subset_idx = tree.query(self.node_xy, k=K)
        # distances: [N, K] chord distances
        # ocean_subset_idx: [N, K] — indices into _ocean_xy / _ocean_flat
        print(f"  Query done in {time.time() - t0:.1f}s")

        # Convert ocean-subset indices → flat grid indices
        flat_indices = self._ocean_flat[ocean_subset_idx]  # [N, K]

        # IDW weights: 1 / (d + eps), normalised per node
        eps = 1e-10
        w = 1.0 / (distances + eps)
        w /= w.sum(axis=1, keepdims=True)
        w = w.astype(np.float32)

        # Save
        np.savez_compressed(
            out_path,
            indices=flat_indices.astype(np.int32),
            weights=w,
            K=K,
        )
        print(f"  Saved {out_path}  ({os.path.getsize(out_path) / 1024**2:.1f} MB)")

        self._indices = flat_indices
        self._weights = w

    def _load_weights(self, path: str):
        """Load precomputed weights from an npz file."""
        d = np.load(path)
        self._indices = d['indices']   # [N, K]
        self._weights = d['weights']   # [N, K]
        self.k_neighbors = int(d.get('K', self._indices.shape[1]))
        print(f"  Loaded indices {self._indices.shape}, weights {self._weights.shape}")

    def load_weights(self, path: str):
        """Public method — load precomputed weights from a .npz path."""
        self._load_weights(path)

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------
    def interpolate_one(
        self, grid_data: np.ndarray,
    ) -> np.ndarray:
        """Interpolate a single grid field to node positions.

        Parameters
        ----------
        grid_data : np.ndarray
            (H, W) or (C, H, W) — denormalised physical values,
            NaN = land.

        Returns
        -------
        np.ndarray  — (N_nodes,) or (N_nodes, C) float32.
        """
        if self._indices is None or self._weights is None:
            raise RuntimeError("Weights not precomputed. Call precompute_weights() or load_weights() first.")

        if grid_data.ndim == 2:
            return self.interpolate_from_flat(grid_data.ravel())

        # Multi-channel: (C, H, W)
        C = grid_data.shape[0]
        return self.interpolate_from_flat(grid_data.reshape(C, -1))  # handles C=1 squeeze

    def interpolate_from_flat(
        self, grid_flat: np.ndarray,
    ) -> np.ndarray:
        """Interpolate from a pre-flattened grid array.

        Parameters
        ----------
        grid_flat : np.ndarray
            (H*W,) or (C, H*W) float32 — already raveled.

        Returns
        -------
        np.ndarray  — (N_nodes,) or (N_nodes, C) float32.
        """
        if grid_flat.ndim == 1:
            gathered = grid_flat[self._indices]
            return (gathered * self._weights).sum(axis=1).astype(np.float32)

        C = grid_flat.shape[0]
        gathered = grid_flat[:, self._indices]       # [C, N, K]
        result = (gathered * self._weights).sum(axis=-1)  # [C, N]
        result = result.T.astype(np.float32)           # [N, C]
        if C == 1:
            return result.squeeze(axis=-1)            # [N]
        return result

    # ------------------------------------------------------------------
    # File-level operations
    # ------------------------------------------------------------------
    def interpolate_file(self, npz_path: str, output_path: str) -> bool:
        """Load a sample .npz, interpolate the 'sample' key, save to output.

        Returns True on success, False on error.
        """
        try:
            data = np.load(npz_path)
            sample_grid = data['sample']  # (C, H, W) or (H, W)
            sample_nodes = self.interpolate_one(sample_grid)

            meta = {}
            for k in ('timestamp', 'sampling_steps', 'ensemble_index', 'rank'):
                if k in data:
                    meta[k] = data[k]
            data.close()

            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            np.savez_compressed(output_path, sample=sample_nodes, **meta)
            return True
        except Exception:
            return False
