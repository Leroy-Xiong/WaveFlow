"""
Regular-grid visualisation for HK Zoom (75×107) wave forecasts.

Produces 2×3 comparison figures (GT | Mean | Std | Sample1 | Sample2 | Diff)
using ``pcolormesh`` on Cartopy axes — no scatter interpolation needed since
the data already lives on a regular lat-lon grid.
"""

from __future__ import annotations

import os
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from typing import Optional

from modules.utils import WAVE_VARIABLES

# ---------------------------------------------------------------------------
# Grid geometry (matches HKZoomPreinterpolator / GridWaveDataset)
# ---------------------------------------------------------------------------
HK_GRID_SHAPE = (75, 107)
HK_LON_RANGE = (113.6005, 114.5996)
HK_LAT_RANGE = (22.0, 22.6998)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _grid_coords():
    """Return (lon_1d, lat_1d) for the 75×107 HK zoom grid."""
    lons = np.linspace(HK_LON_RANGE[0], HK_LON_RANGE[1], HK_GRID_SHAPE[1])
    lats = np.linspace(HK_LAT_RANGE[0], HK_LAT_RANGE[1], HK_GRID_SHAPE[0])
    return lons, lats


def _load_grid_gt(ts: str, gt_dir: str = "./data/hindcast/preinterpolated_hk_zoom") -> np.ndarray:
    """Load ground-truth grid data for a timestamp.

    Returns (4, 75, 107) float32 array [hs, tp, dm, elev].
    Land pixels are NaN.
    """
    year = ts[:4]
    path = os.path.join(gt_dir, year, f"wave_{ts}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Grid GT file not found: {path}")
    return np.load(path).astype(np.float32)


def _load_grid_sample(path: str) -> np.ndarray:
    """Load a single grid-format sample .npz file.  Returns (4, 75, 107)."""
    data = np.load(path)
    sample = data["sample"]
    data.close()
    return sample


def _setup_cartopy_ax(ax, lons, lats, title=""):
    """Configure a Cartopy GeoAxes for the HK region."""
    ax.set_extent(
        [lons[0], lons[-1], lats[0], lats[-1]],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="10m", linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    # draw_labels=False avoids Cartopy LinearRing crash on small extents
    gl = ax.gridlines(
        draw_labels=False, linewidth=0.3, color="gray", alpha=0.5,
        linestyle="--",
    )
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")


def _plot_grid_panel(ax, data_2d, lons, lats, vmin, vmax, cmap, title):
    """Plot a single (H, W) array as pcolormesh on a Cartopy axis."""
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    # Mask NaN (land) pixels
    mask = np.isnan(data_2d)
    masked = np.ma.masked_where(mask, data_2d)
    im = ax.pcolormesh(
        lon_mesh, lat_mesh, masked,
        transform=ccrs.PlateCarree(),
        vmin=vmin, vmax=vmax, cmap=cmap,
        shading="auto",
    )
    _setup_cartopy_ax(ax, lons, lats, title)
    return im


# ---------------------------------------------------------------------------
# main figure builder
# ---------------------------------------------------------------------------

def plot_grid_ensemble(
    ts: str,
    sample_files: list[str],
    gt_dir: str = "./data/hindcast/preinterpolated_hk_zoom",
    save_path: Optional[str] = None,
    variables: Optional[list[str]] = None,
    seed: int = 42,
    dpi: int = 150,
):
    """Create a 2×3 comparison figure for a timestamp on the regular grid.

    Parameters
    ----------
    ts : str
        Timestamp string (YYYYMMDDTHH).
    sample_files : list[str]
        Paths to ensemble sample .npz files for this timestamp.
    gt_dir : str
        Directory containing preinterpolated ground-truth .npy files.
    save_path : str or None
        If provided, save figures to ``<save_path>_<variable>.jpg``.
    variables : list[str] or None
        Which variables to plot.  Default: ['hs', 'tp', 'dm', 'elev'].
    seed : int
        Random seed for picking ensemble members to display.
    dpi : int
        Figure DPI.
    """
    rng = np.random.RandomState(seed)
    variables = variables or ["hs", "tp", "dm", "elev"]
    lons, lats = _grid_coords()

    # Load ground truth
    gt_grid = _load_grid_gt(ts, gt_dir=gt_dir)  # (4, 75, 107)

    # Load ensemble samples
    samples = np.stack([_load_grid_sample(f) for f in sorted(sample_files)], axis=0)
    # (N_ens, 4, 75, 107)

    n_ens = samples.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(samples, axis=0)
        std = np.nanstd(samples, axis=0)

    # Pick two ensemble members
    idx = sorted(rng.choice(n_ens, size=min(2, n_ens), replace=False))

    var_indices = {v: i for i, v in enumerate(["hs", "tp", "dm", "elev"])}

    for var in variables:
        vi = var_indices[var]
        var_info = WAVE_VARIABLES.get(var, {})
        label = var_info.get("plot_label", var)
        units = var_info.get("units", "")
        cmap = "turbo"
        std_cmap = "viridis"
        diff_cmap = "seismic"

        # --- colorbar range ---
        gt_vals = gt_grid[vi][~np.isnan(gt_grid[vi])]
        if gt_vals.size == 0:
            # All land or missing — skip this variable
            plt.close(fig)
            continue
        vmin, vmax = float(np.min(gt_vals)), float(np.max(gt_vals))
        diff_max = max(abs(vmin - np.nanmin(mean[vi])), abs(vmax - np.nanmax(mean[vi])), 0.01)
        _std_val = np.nanmax(std[vi])
        std_vmax = max(float(_std_val) * 0.8, 0.01) if not np.isnan(_std_val) else 0.01

        fig, axes = plt.subplots(
            2, 3, figsize=(18, 9),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        axes = axes.flatten()

        # Panel 1: Ground Truth
        im = _plot_grid_panel(axes[0], gt_grid[vi], lons, lats,
                              vmin, vmax, cmap, f"GT {label}")
        plt.colorbar(im, ax=axes[0], shrink=0.7, label=units)

        # Panel 2: Ensemble Mean
        im = _plot_grid_panel(axes[1], mean[vi], lons, lats,
                              vmin, vmax, cmap, f"Ensemble Mean {label}")
        plt.colorbar(im, ax=axes[1], shrink=0.7, label=units)

        # Panel 3: Ensemble Std
        im = _plot_grid_panel(axes[2], std[vi], lons, lats,
                              0, std_vmax, std_cmap, f"Ensemble Std {label}")
        plt.colorbar(im, ax=axes[2], shrink=0.7, label=units)

        # Panel 4: Sample 1
        im = _plot_grid_panel(axes[3], samples[idx[0], vi], lons, lats,
                              vmin, vmax, cmap, f"Sample #{idx[0]} {label}")
        plt.colorbar(im, ax=axes[3], shrink=0.7, label=units)

        # Panel 5: Sample 2
        if len(idx) > 1:
            im = _plot_grid_panel(axes[4], samples[idx[1], vi], lons, lats,
                                  vmin, vmax, cmap, f"Sample #{idx[1]} {label}")
            plt.colorbar(im, ax=axes[4], shrink=0.7, label=units)
        else:
            axes[4].set_visible(False)

        # Panel 6: Difference (Mean - GT)
        diff = mean[vi] - gt_grid[vi]
        im = _plot_grid_panel(axes[5], diff, lons, lats,
                              -diff_max, diff_max, diff_cmap,
                              f"Diff (Mean − GT) {label}")
        plt.colorbar(im, ax=axes[5], shrink=0.7, label=units)

        fig.suptitle(f"{ts}  —  {var.upper()}  ({n_ens} members)",
                     fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout()

        if save_path:
            sp = f"{save_path}_{ts}_{var}.jpg"
            os.makedirs(os.path.dirname(sp), exist_ok=True)
            fig.savefig(sp, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Global grid (720×1440) visualization
# ---------------------------------------------------------------------------

GLOBAL_GRID_SHAPE = (720, 1440)
GLOBAL_LON_RANGE = (-180, 180)
GLOBAL_LAT_RANGE = (90, -90)


def _global_grid_coords():
    lons = np.linspace(GLOBAL_LON_RANGE[0], GLOBAL_LON_RANGE[1], GLOBAL_GRID_SHAPE[1])
    lats = np.linspace(GLOBAL_LAT_RANGE[0], GLOBAL_LAT_RANGE[1], GLOBAL_GRID_SHAPE[0])
    return lons, lats


def _load_global_gt(ts: str, gt_dir: str) -> np.ndarray:
    """Load global grid GT. Returns (H, W) HS field."""
    year = ts[:4]
    path = os.path.join(gt_dir, year, f"global_hindcast_{ts[:8]}_{ts[9:]}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Global GT not found: {path}")
    data = np.load(path)
    return data[0].astype(np.float32) if data.ndim == 3 else data.astype(np.float32)


def plot_global_grid_ensemble(
    ts: str,
    sample_files: list[str],
    gt_dir: str = "./data/global_hindcast/interpolated_grid_npy",
    save_path: Optional[str] = None,
    seed: int = 42,
    dpi: int = 150,
):
    """2×3 global grid figure: GT | Mean | Std | Member1 | Member2 | Diff.

    Parameters
    ----------
    ts : str
        Timestamp (YYYYMMDDTHH).
    sample_files : list[str]
        Paths to ensemble .npz files (each has 'sample' key, shape (1, H, W)).
    gt_dir : str
        Directory with global_hindcast_YYYYMMDD_HH.npy files.
    save_path : str or None
        Save prefix; produces ``<save_path>_<ts>_hs.jpg``.
    seed : int
        Seed for picking displayed members.
    dpi : int
        Figure DPI.
    """
    rng = np.random.RandomState(seed)
    lons, lats = _global_grid_coords()
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)

    gt = _load_global_gt(ts, gt_dir)
    ens = np.stack([np.load(f)["sample"][0] for f in sorted(sample_files)], axis=0)
    n_ens = ens.shape[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(ens, axis=0)
        std = np.nanstd(ens, axis=0)

    idx = sorted(rng.choice(n_ens, size=min(2, n_ens), replace=False))
    diff = mean - gt

    vmin, vmax = 0, float(np.nanmax(gt)) * 0.95
    std_vmax = max(float(np.nanmax(std)) * 0.8, 0.01)
    diff_lim = max(abs(float(np.nanmin(diff))), abs(float(np.nanmax(diff))), 0.01)

    proj = ccrs.Robinson()
    pc = ccrs.PlateCarree()

    def _panel(ax, data, lo, hi, cmap, title):
        masked = np.ma.masked_where(np.isnan(data), data)
        im = ax.pcolormesh(lon_mesh, lat_mesh, masked, transform=pc,
                           vmin=lo, vmax=hi, cmap=cmap, shading="auto")
        ax.set_global()
        ax.coastlines(resolution="110m", linewidth=0.4)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
        ax.set_title(title, fontsize=9, fontweight="bold")
        return im

    fig, axes = plt.subplots(2, 3, figsize=(14, 7),
                             subplot_kw={"projection": proj})
    fig.suptitle(f"Global Grid HS — {ts} ({n_ens} members)",
                 fontsize=12, fontweight="bold")

    im_gt = _panel(axes[0, 0], gt, vmin, vmax, "turbo", "Ground Truth")
    _panel(axes[0, 1], mean, vmin, vmax, "turbo", "Ensemble Mean")
    im_std = _panel(axes[0, 2], std, 0, std_vmax, "viridis", "Std")
    _panel(axes[1, 0], ens[idx[0]], vmin, vmax, "turbo", f"Member {idx[0]}")
    _panel(axes[1, 1], ens[idx[1]], vmin, vmax, "turbo", f"Member {idx[1]}")
    im_diff = _panel(axes[1, 2], diff, -diff_lim, diff_lim, "seismic", "Mean − GT")

    fig.colorbar(im_gt, ax=axes[0, 0], fraction=0.02, pad=0.02)
    fig.colorbar(im_std, ax=axes[0, 2], fraction=0.02, pad=0.02)
    fig.colorbar(im_diff, ax=axes[1, 2], fraction=0.02, pad=0.02)

    plt.tight_layout()
    if save_path:
        sp = f"{save_path}_{ts}_hs.jpg"
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        fig.savefig(sp, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
