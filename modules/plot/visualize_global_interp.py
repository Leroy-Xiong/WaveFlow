"""
Global hindcast interpolation visualization.

Generates a side-by-side comparison figure:
  Left  — source unstructured mesh nodes (coloured by SWH value).
  Right — interpolated regular 0.25° grid.

Supports both source NPZ files (interpolates on the fly) and pre-interpolated
NPY files (loads directly).

Usage:
    from modules.plot.visualize_global_interp import visualize_interpolation

    # From source NPZ — interpolates on the fly:
    visualize_interpolation(
        npz_path='data/global_hindcast/npz_files/global_hindcast_20180916_12.npz',
        save_path='output/plot/data/global_hindcast/interpolation/global_hindcast_interp_20180916_12.png',
    )

    # From pre-interpolated NPY:
    visualize_interpolation(
        npy_path='data/global_hindcast/interpolated_grid/2018/global_hindcast_20180916_12.npy',
        save_path='output/plot/data/global_hindcast/interpolation/global_hindcast_interp_20180916_12.png',
    )
"""

import os
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_COORD_FILE = './data/global_hindcast/global_coordinates.npy'
DEFAULT_WEIGHTS_DIR = './data/global_hindcast/interpolation_weights'
DEFAULT_TARGET_SHAPE = (721, 1440)
DEFAULT_OUTPUT_DIR = './output/plot/data/global_hindcast/interpolation'
DEFAULT_DPI = 150


def _latlon_to_cartesian(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Convert lat/lon (degrees) → 3D Cartesian on unit sphere."""
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    x = np.cos(lats_rad) * np.cos(lons_rad)
    y = np.cos(lats_rad) * np.sin(lons_rad)
    z = np.sin(lats_rad)
    return np.stack([x, y, z], axis=-1)


def _load_interpolation_assets(
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    target_shape: tuple = DEFAULT_TARGET_SHAPE,
):
    """Load precomputed neighbour indices, IDW weights, land mask, and build
    target grid coordinates.  Returns a dict with all assets."""
    k = 9
    thresh = 0.5  # must match the precompute run

    idx_path = os.path.join(weights_dir, f'indices_k{k}.npy')
    wgt_path = os.path.join(weights_dir, f'weights_k{k}.npy')
    msk_path = os.path.join(weights_dir, f'land_mask_t{thresh:.1f}.npy')

    for p in (idx_path, wgt_path, msk_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Precomputed weight file not found: {p}\n"
                f"Run 'python load_data.py global-hindcast-preinterp precompute' first."
            )

    n_lat, n_lon = target_shape
    lon_vals = np.linspace(0, 360, n_lon, endpoint=False, dtype=np.float64)
    lat_vals = np.linspace(90, -90, n_lat, dtype=np.float64)

    return {
        'indices': np.load(idx_path),
        'weights': np.load(wgt_path),
        'land_mask': np.load(msk_path),
        'lon_vals': lon_vals,
        'lat_vals': lat_vals,
        'target_shape': target_shape,
    }


def _interpolate_from_source(npz_path: str, assets: dict) -> np.ndarray:
    """Interpolate one source NPZ to the regular grid using precomputed assets.

    Returns (n_lat, n_lon) float32 array; land cells are NaN.
    """
    data = np.load(npz_path)
    if 'WWM_1' not in data:
        raise KeyError(f"'WWM_1' missing from {npz_path}.  Keys: {list(data.keys())}")
    src_vals = data['WWM_1'].astype(np.float32)

    gathered = src_vals[assets['indices']]
    result = (gathered * assets['weights']).sum(axis=1)
    result[assets['land_mask']] = np.nan

    n_lat, n_lon = assets['target_shape']
    return result.reshape(n_lat, n_lon).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_interpolation(
    npz_path: Optional[str] = None,
    npy_path: Optional[str] = None,
    save_path: Optional[str] = None,
    coord_file: str = DEFAULT_COORD_FILE,
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    dpi: int = DEFAULT_DPI,
    title: Optional[str] = None,
):
    """Visualise global hindcast interpolation: source nodes vs regular grid.

    Exactly one of ``npz_path`` (source unstructured NPZ) or ``npy_path``
    (pre-interpolated grid NPY) must be provided.

    Parameters
    ----------
    npz_path : str or None
        Path to a ``global_hindcast_*.npz`` source file.  Interpolation is
        performed on the fly using precomputed weights.
    npy_path : str or None
        Path to a pre-interpolated ``global_hindcast_*.npy`` grid file.
        The corresponding source NPZ file is looked up automatically.
    save_path : str or None
        Output PNG path.  Auto-generated from ``{output_dir}/{basename}.png``
        if not given.
    coord_file : str
        Path to ``global_coordinates.npy``.
    weights_dir : str
        Directory containing precomputed KDTree weights.
    dpi : int
        Figure DPI.
    title : str or None
        Figure suptitle override.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # --- resolve inputs ---
    if (npz_path is None) == (npy_path is None):
        raise ValueError("Exactly one of `npz_path` or `npy_path` must be given.")

    if npy_path is not None:
        # Load pre-interpolated grid (supports .npy and .npz)
        npy_path = os.path.abspath(npy_path)
        loaded = np.load(npy_path, allow_pickle=True)
        grid = loaded['data'] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
        fname = os.path.basename(npy_path).replace('.npz', '').replace('.npy', '')
        ts = fname.replace('global_hindcast_', '').replace('_', 'T', 1)
        year = ts[:4]
        # Try to find the source NPZ
        src_dir = './data/global_hindcast/npz_files'
        src_fname = fname + '.npz'
        npz_path = os.path.join(src_dir, src_fname)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Cannot find source NPZ for {npy_path}.\n"
                f"Expected: {npz_path}"
            )
    else:
        npz_path = os.path.abspath(npz_path)
        fname = os.path.basename(npz_path).replace('.npz', '')
        ts = fname.replace('global_hindcast_', '').replace('_', 'T', 1)
        year = ts[:4]

    # --- load source data ---
    src_data = np.load(npz_path)
    src_vals = src_data['WWM_1'].astype(np.float32)

    # --- load source coordinates ---
    source_coords = np.load(coord_file)  # (N, 2) — [lat, lon]

    # --- interpolate if needed ---
    if npy_path is None:
        assets = _load_interpolation_assets(weights_dir)
        grid = _interpolate_from_source(npz_path, assets)
        lon_vals = assets['lon_vals']
        lat_vals = assets['lat_vals']
        target_shape = assets['target_shape']
    else:
        assets = _load_interpolation_assets(weights_dir)
        lon_vals = assets['lon_vals']
        lat_vals = assets['lat_vals']
        target_shape = assets['target_shape']

    # --- generate save_path if not given ---
    if save_path is None:
        output_dir = DEFAULT_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'global_hindcast_interp_{fname}.png')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    # --- unified colour range ---
    vmax = np.nanmax(grid)

    # --- plot ---
    fig, axes = plt.subplots(
        1, 2, figsize=(18, 6),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
    )

    # -- Left: source nodes --
    ax = axes[0]
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=0)
    ax.coastlines(resolution='110m', linewidth=0.3, zorder=1)
    ax.set_title(f'Source nodes ({len(source_coords):,} total)')

    sc = ax.scatter(
        source_coords[:, 1], source_coords[:, 0],
        c=src_vals, cmap='Spectral_r', s=0.1,
        vmin=0, vmax=vmax,
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(sc, ax=ax, label='SWH (m)', shrink=0.7)

    # -- Right: interpolated grid --
    ax = axes[1]
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=0)
    ax.coastlines(resolution='110m', linewidth=0.3, zorder=1)
    ax.set_title(f'Interpolated grid {target_shape[1]}×{target_shape[0]} '
                 f'(0.25°, K=9)')

    im = ax.pcolormesh(
        lon_vals, lat_vals, grid,
        cmap='Spectral_r', vmin=0, vmax=vmax,
        transform=ccrs.PlateCarree(), shading='auto',
    )
    plt.colorbar(im, ax=ax, label='SWH (m)', shrink=0.7)

    # -- title --
    fig.suptitle(title or f'Global Hindcast Interpolation — {fname}',
                 fontsize=13, y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {save_path}")
