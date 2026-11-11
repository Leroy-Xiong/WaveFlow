"""
Visualize random samples from the global 0.5° interpolated data.

Shows u10, v10, wind speed (with arrows), and SWH side-by-side on world maps
with lat/lon gridlines and coastlines.

Usage as module:
    from modules.plot.visualize_05deg import visualize_samples
    visualize_samples(n=4, seed=42)

Usage via CLI:
    python load_data.py global-0.5deg-vis --n-samples 4 --seed 42
"""

import os
import random
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ERA5_DIR = './data/era5/global_0.5deg'
DEFAULT_WAVE_DIR = './data/global_hindcast/interpolated_grid_0.5deg'
DEFAULT_OUTPUT_DIR = './output/plot/data/global_hindcast/0.5deg_samples'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_timestamps(era5_dir: str, wave_dir: str, n: int, seed: int) -> list:
    """Return up to *n* random (ts, era5_path, wave_path) tuples that exist in both dirs."""
    random.seed(seed)

    # Collect all matching ERA5 NPY paths
    pairs = []
    for year in sorted(os.listdir(era5_dir)):
        era5_year = os.path.join(era5_dir, year)
        wave_year = os.path.join(wave_dir, year)
        if not os.path.isdir(era5_year) or not os.path.isdir(wave_year):
            continue
        for fname in sorted(os.listdir(era5_year)):
            if not fname.startswith('era5_') or not fname.endswith('.npy'):
                continue
            ts = fname.replace('era5_', '').replace('.npy', '')
            # Wave file: global_hindcast_YYYYMMDD_HH.npy
            hc_ts = ts.replace('T', '_')
            wave_path = os.path.join(wave_year, f'global_hindcast_{hc_ts}.npy')
            if os.path.exists(wave_path):
                pairs.append((ts, os.path.join(era5_year, fname), wave_path))

    random.shuffle(pairs)
    return pairs[:n]


def _add_basemap(ax):
    """Add global map features, gridlines, and coastlines to an axis."""
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=0)
    ax.coastlines(resolution='110m', linewidth=0.3, zorder=1)
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.2, color='grey', alpha=0.4, linestyle='--',
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}


def _plot_one_sample(axes, era5_data, swh_data, ts, lon, lat):
    """Plot a single sample: u10 | v10 | wind speed+arrows | SWH.

    Parameters
    ----------
    axes : array of 4 Axes (all cartopy GeoAxes)
    era5_data : np.ndarray [2, 360, 720] — u10, v10
    swh_data : np.ndarray [360, 720] — SWH
    ts : str — timestamp 'YYYYMMDDTHH'
    lon, lat : np.ndarray — 1-D coordinate arrays
    """
    u10 = era5_data[0]
    v10 = era5_data[1]
    wspd = np.sqrt(u10**2 + v10**2)

    # --- Subsample for wind arrows (every stride-th grid point) ---
    stride = 18  # 360/18=20, 720/18=40 → 800 arrows
    lon_sub = lon[::stride]
    lat_sub = lat[::stride]
    u10_sub = u10[::stride, ::stride]
    v10_sub = v10[::stride, ::stride]

    # Adjust aspect ratio for quiver: platecarree lon/lat distorts at high lat
    # Scale u component by 1/cos(lat) to make arrows visually correct
    lat_2d = lat_sub[:, None]  # broadcast for [20, 40]
    cos_lat = np.cos(np.deg2rad(np.abs(lat_2d)))
    cos_lat = np.clip(cos_lat, 0.1, 1.0)
    u_scaled = u10_sub / cos_lat

    # --- Panel layout ---
    panels = [
        (axes[0], u10, 'u10 (m/s)', 'RdBu_r', -25, 25, False),
        (axes[1], v10, 'v10 (m/s)', 'RdBu_r', -25, 25, False),
        (axes[2], wspd, 'Wind Speed (m/s)', 'YlOrRd', 0, 20, True),
        (axes[3], swh_data, 'SWH (m)', 'Spectral_r', 0, 8, False),
    ]

    for ax, data, title, cmap, vmin, vmax, is_wind_speed in panels:
        _add_basemap(ax)
        im = ax.pcolormesh(
            lon, lat, data,
            cmap=cmap, vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), shading='auto',
        )
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)

        # Overlay wind arrows on the wind speed panel
        if is_wind_speed:
            ax.quiver(
                lon_sub, lat_sub, u_scaled, v10_sub,
                transform=ccrs.PlateCarree(),
                scale=200, width=0.003, color='black', alpha=0.7,
                headwidth=3, headlength=4,
            )

    # Timestamp label on the leftmost axis
    dt = datetime.strptime(ts, '%Y%m%dT%H')
    axes[0].set_title(
        f'{dt.strftime("%Y-%m-%d %H:00")} UTC',
        fontsize=10, loc='left', fontweight='bold',
    )


def visualize_samples(
    n: int = 4,
    seed: int = 42,
    era5_dir: str = DEFAULT_ERA5_DIR,
    wave_dir: str = DEFAULT_WAVE_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    dpi: int = 150,
):
    """Randomly sample and visualize input→output pairs from 0.5° data.

    Parameters
    ----------
    n : int
        Number of random timestamps to visualize.
    seed : int
        Random seed for reproducibility.
    era5_dir : str
        Root directory of ERA5 0.5° NPY files.
    wave_dir : str
        Root directory of wave 0.5° NPY files.
    output_dir : str
        Directory to save PNG figures.
    dpi : int
        Output DPI.
    """
    # --- Build target lat/lon arrays ---
    lon = np.linspace(0, 360, 720, endpoint=False)
    lat = np.linspace(89.5, -90, 360)

    # --- Sample ---
    samples = _sample_timestamps(era5_dir, wave_dir, n, seed)
    if not samples:
        print("No matching ERA5+wave pairs found.")
        return

    print(f"Visualizing {len(samples)} random samples (seed={seed})")
    os.makedirs(output_dir, exist_ok=True)

    for i, (ts, era5_path, wave_path) in enumerate(samples):
        era5 = np.load(era5_path).astype(np.float32)  # [2, 360, 720]
        swh = np.load(wave_path).astype(np.float32)    # [360, 720]

        fig, axes = plt.subplots(
            1, 4, figsize=(32, 6),
            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        )
        _plot_one_sample(axes, era5, swh, ts, lon, lat)

        dt_label = datetime.strptime(ts, '%Y%m%dT%H').strftime('%Y%m%d_%H%M')
        out_path = os.path.join(output_dir, f'sample_{i:02d}_{dt_label}.png')
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        u10_range = f'[{era5[0].min():.1f}, {era5[0].max():.1f}]'
        v10_range = f'[{era5[1].min():.1f}, {era5[1].max():.1f}]'
        swh_range = f'[{np.nanmin(swh):.2f}, {np.nanmax(swh):.2f}]'
        print(f"  [{i+1}/{len(samples)}] {ts}  "
              f"u10={u10_range}  v10={v10_range}  SWH={swh_range}")
        print(f"    → {out_path}")

    print(f"\nSaved {len(samples)} figures to {output_dir}/")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    visualize_samples(n=2, seed=123)
