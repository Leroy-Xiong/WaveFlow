"""
Dual-model comparison figure (6-panel, 2×3) for irregular-mesh vs regular-grid
ensemble predictions on the same timestamp.

Panels:
  (1,1) Ground Truth (scatter, 7788 nodes)
  (1,2) Irregular model ensemble mean (scatter)
  (1,3) Regular model ensemble mean — interpolated to nodes (scatter)
  (2,1) Regular model ensemble mean — native grid (pcolormesh)
  (2,2) Error: Irregular Mean − GT (scatter)
  (2,3) Error: Regular Mean (nodes) − GT (scatter)

Colormaps: turbo for GT / means, seismic for error maps.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from typing import Optional

from modules.utils import WAVE_VARIABLES


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_node_samples(file_list: list[str]) -> np.ndarray:
    """Load irregular / node-format ensemble samples.  Returns (N_ens, N, C)."""
    samples = []
    for f in sorted(file_list):
        data = np.load(f)
        samples.append(data["sample"].squeeze())
        data.close()
    return np.stack(samples, axis=0)


def _load_grid_samples(file_list: list[str]) -> np.ndarray:
    """Load regular-grid ensemble samples.  Returns (N_ens, C, H, W)."""
    samples = []
    for f in sorted(file_list):
        data = np.load(f)
        samples.append(data["sample"])
        data.close()
    return np.stack(samples, axis=0)


def _load_grid_gt(ts: str, gt_dir: str = "./data/hindcast/preinterpolated_hk_zoom") -> np.ndarray:
    """Load regular-grid ground truth.  Returns (4, 75, 107)."""
    year = ts[:4]
    path = os.path.join(gt_dir, year, f"wave_{ts}.npy")
    return np.load(path).astype(np.float32)


def _load_node_gt(ts: str, gt_dir: str = "./data/hindcast/processed") -> np.ndarray:
    """Load NC ground truth for the HK region, returning (7788, 4)."""
    import xarray as xr
    from modules.utils import HK_REGION
    path = os.path.join(gt_dir, f"wave_{ts}.nc")
    ds = xr.load_dataset(path, engine="netcdf4")
    # Filter to HK region
    min_lon, max_lon = HK_REGION[0], HK_REGION[1]
    min_lat, max_lat = HK_REGION[2], HK_REGION[3]
    lon_name = "longitude" if "longitude" in ds else "lon"
    lat_name = "latitude" if "latitude" in ds else "lat"
    ds = ds.where(
        (ds[lon_name] >= min_lon) & (ds[lon_name] < max_lon) &
        (ds[lat_name] >= min_lat) & (ds[lat_name] < max_lat),
        drop=True,
    )
    vals = np.column_stack([
        ds["hs"].values.squeeze().astype(np.float32),
        ds["tp"].values.squeeze().astype(np.float32),
        ds["dm"].values.squeeze().astype(np.float32),
        ds["elev"].values.squeeze().astype(np.float32),
    ])
    ds.close()
    return vals  # (7788, 4)


def _make_node_ds(values, lons, lats) -> "xr.Dataset":
    """Wrap a (7788,) node array into an xr.Dataset for plot_wave_data."""
    import xarray as xr
    return xr.Dataset(
        {"hs": (["node"], values)},
        coords={"lon": (["node"], lons), "lat": (["node"], lats)},
    )


# ---------------------------------------------------------------------------
# main figure builder
# ---------------------------------------------------------------------------

def plot_model_comparison(
    ts: str,
    irregular_files: list[str],
    regular_grid_files: list[str],
    regular_node_files: list[str],
    node_lons: np.ndarray,
    node_lats: np.ndarray,
    node_token_files: Optional[list[str]] = None,
    node_gt_dir: str = "./data/hindcast/processed",
    grid_gt_dir: str = "./data/hindcast/preinterpolated_hk_zoom",
    save_path: Optional[str] = None,
    variables: Optional[list[str]] = None,
    seed: int = 42,
    dpi: int = 150,
):
    """Create a comparison figure for one timestamp.

    With node_token_files: 9-panel (3×3) including node-token GraphTransformer.
    Without: 6-panel (2×3) irregular vs regular grid.

    Parameters
    ----------
    ts : str
        Timestamp YYYYMMDDTHH.
    irregular_files : list[str]
        Paths to irregular-model ensemble .npz files (7788, 4) each.
    regular_grid_files : list[str]
        Paths to regular-model grid ensemble .npz files (4, 75, 107) each.
    regular_node_files : list[str]
        Paths to regular-model node-interpolated ensemble .npz files (7788, 4).
    node_token_files : list[str] or None
        Paths to node-token GraphTransformer ensemble .npz files (7788, 4).
    node_lons, node_lats : np.ndarray
        (7788,) lon/lat for node scatter plots.
    node_gt_dir, grid_gt_dir : str
    save_path : str or None
    variables : list[str] or None
    seed : int
    dpi : int
    """
    from modules.plot.plot_grid import _plot_grid_panel, _grid_coords, _setup_cartopy_ax
    from modules.plot.plot_wave import plot_wave_data

    variables = variables or ["hs", "tp", "dm", "elev"]
    var_indices = {v: i for i, v in enumerate(["hs", "tp", "dm", "elev"])}
    grid_lons, grid_lats = _grid_coords()
    use_three = bool(node_token_files)

    # ---- load data ----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        irr_arr = _load_node_samples(irregular_files)        # (ens, 7788, 4)
        reg_grid_arr = _load_grid_samples(regular_grid_files) # (ens, 4, 75, 107)
        reg_node_arr = _load_node_samples(regular_node_files) # (ens, 7788, 4)
        if use_three:
            nt_arr = _load_node_samples(node_token_files)     # (ens, 7788, 4)

    node_gt = _load_node_gt(ts, gt_dir=node_gt_dir)          # (7788, 4)
    grid_gt = _load_grid_gt(ts, gt_dir=grid_gt_dir)          # (4, 75, 107)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        irr_mean = np.nanmean(irr_arr, axis=0)               # (7788, 4)
        reg_grid_mean = np.nanmean(reg_grid_arr, axis=0)     # (4, 75, 107)
        reg_node_mean = np.nanmean(reg_node_arr, axis=0)     # (7788, 4)
        if use_three:
            nt_mean = np.nanmean(nt_arr, axis=0)             # (7788, 4)

    # ---- per-variable loop ----
    for var in variables:
        vi = var_indices[var]
        info = WAVE_VARIABLES.get(var, {})
        label = info.get("plot_label", var)
        units = info.get("units", "")

        # Node GT for this variable
        gt_node_vals = node_gt[:, vi]          # (7788,)
        # Ensemble means
        irr_vals = irr_mean[:, vi]             # (7788,)
        reg_node_vals = reg_node_mean[:, vi]   # (7788,)
        reg_grid_vals = reg_grid_mean[vi]      # (75, 107)
        grid_gt_vals = grid_gt[vi]             # (75, 107)
        if use_three:
            nt_vals = nt_mean[:, vi]           # (7788,)
            nt_err = nt_vals - gt_node_vals

        # Error maps (on nodes)
        irr_err = irr_vals - gt_node_vals
        reg_err = reg_node_vals - gt_node_vals

        # ---- colorbar ranges ----
        _gt_finite = gt_node_vals[~np.isnan(gt_node_vals)]
        if var == "elev":
            gt_vmin = float(np.percentile(_gt_finite, 0.1))
            gt_vmax = float(np.percentile(_gt_finite, 99.9))
        else:
            gt_vmin = float(np.min(_gt_finite))
            gt_vmax = float(np.max(_gt_finite))

        err_maxes = [float(np.nanmax(np.abs(irr_err))),
                     float(np.nanmax(np.abs(reg_err)))]
        if use_three:
            err_maxes.append(float(np.nanmax(np.abs(nt_err))))
        err_abs = max(err_maxes + [0.01])

        # Build node datasets for scatter panels
        ds_gt   = _make_node_ds(gt_node_vals, node_lons, node_lats)
        ds_irr  = _make_node_ds(irr_vals, node_lons, node_lats)
        ds_regn = _make_node_ds(reg_node_vals, node_lons, node_lats)
        ds_e1   = _make_node_ds(irr_err, node_lons, node_lats)
        ds_e2   = _make_node_ds(reg_err, node_lons, node_lats)
        if use_three:
            ds_nt  = _make_node_ds(nt_vals, node_lons, node_lats)
            ds_e3  = _make_node_ds(nt_err, node_lons, node_lats)

        # ---- build figure ----
        if use_three:
            fig, axes = plt.subplots(
                3, 3, figsize=(28, 20),
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
            n_members = f"{irr_arr.shape[0]}/{nt_arr.shape[0]}/{reg_grid_arr.shape[0]}"
        else:
            fig, axes = plt.subplots(
                2, 3, figsize=(22, 12),
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
            n_members = f"{irr_arr.shape[0]}"

        # ---- Row 1: GT | Patched Mean | (Node-Token Mean or Regular Node Mean) ----
        plot_wave_data(ds_gt, variable="hs", region=list(grid_lons[[0,-1]])+list(grid_lats[[0,-1]]),
                       vmin=gt_vmin, vmax=gt_vmax, cmap="turbo", ax=axes[0,0],
                       scatter_size=10, custom_title=f"GT {label}", show_cities=False, show_colorbar=False)
        plot_wave_data(ds_irr, variable="hs", region=list(grid_lons[[0,-1]])+list(grid_lats[[0,-1]]),
                       vmin=gt_vmin, vmax=gt_vmax, cmap="turbo", ax=axes[0,1],
                       scatter_size=10, custom_title=f"Patched Mean {label}", show_cities=False, show_colorbar=False)
        if use_three:
            plot_wave_data(ds_nt, variable="hs", region=list(grid_lons[[0,-1]])+list(grid_lats[[0,-1]]),
                           vmin=gt_vmin, vmax=gt_vmax, cmap="turbo", ax=axes[0,2],
                           scatter_size=10, custom_title=f"Node-Token Mean {label}", show_cities=False, show_colorbar=False)
        else:
            plot_wave_data(ds_regn, variable="hs", region=list(grid_lons[[0,-1]])+list(grid_lats[[0,-1]]),
                           vmin=gt_vmin, vmax=gt_vmax, cmap="turbo", ax=axes[0,2],
                           scatter_size=10, custom_title=f"Regular Mean (nodes) {label}", show_cities=False, show_colorbar=False)

        # Shared colorbar for top row (turbo)
        cbar_ax_top = fig.add_axes([0.92, 0.58, 0.01, 0.30])
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=gt_vmin, vmax=gt_vmax), cmap="turbo")
        fig.colorbar(sm, cax=cbar_ax_top).set_label(units, fontsize=10)

        # ---- Row 2: Regular Grid Mean | Patched Error | (Node-Token Error or Regular Node Error) ----
        im_g = _plot_grid_panel(axes[1,0], reg_grid_vals, grid_lons, grid_lats,
                                gt_vmin, gt_vmax, "turbo", f"Regular Mean (grid) {label}")
        cbar_ax_grid = fig.add_axes([0.22, 0.08, 0.22, 0.015])
        fig.colorbar(im_g, cax=cbar_ax_grid, orientation="horizontal").set_label(units, fontsize=8)

        plot_wave_data(ds_e1, variable="hs", region=list(grid_lons[[0,-1]])+list(grid_lats[[0,-1]]),
                       vmin=-err_abs, vmax=err_abs, cmap="seismic", ax=axes[1,1],
                       scatter_size=10, custom_title=f"Patched Error {label}", show_cities=False, show_colorbar=False)
        if use_three:
            plot_wave_data(ds_e3, variable="hs", region=list(grid_lons[[0,-1]])+list(grid_lats[[0,-1]]),
                           vmin=-err_abs, vmax=err_abs, cmap="seismic", ax=axes[1,2],
                           scatter_size=10, custom_title=f"Node-Token Error {label}", show_cities=False, show_colorbar=False)
        else:
            plot_wave_data(ds_e2, variable="hs", region=list(grid_lons[[0,-1]])+list(grid_lats[[0,-1]]),
                           vmin=-err_abs, vmax=err_abs, cmap="seismic", ax=axes[1,2],
                           scatter_size=10, custom_title=f"Regular Error (nodes) {label}", show_cities=False, show_colorbar=False)

        # ---- Row 3 (only for 3-model layout): Regular Node Mean | Regular Node Error ----
        if use_three:
            plot_wave_data(ds_regn, variable="hs", region=list(grid_lons[[0,-1]])+list(grid_lats[[0,-1]]),
                           vmin=gt_vmin, vmax=gt_vmax, cmap="turbo", ax=axes[2,0],
                           scatter_size=10, custom_title=f"Regular Mean (nodes) {label}", show_cities=False, show_colorbar=False)
            plot_wave_data(ds_e2, variable="hs", region=list(grid_lons[[0,-1]])+list(grid_lats[[0,-1]]),
                           vmin=-err_abs, vmax=err_abs, cmap="seismic", ax=axes[2,1],
                           scatter_size=10, custom_title=f"Regular Error (nodes) {label}", show_cities=False, show_colorbar=False)

        # Shared colorbar for error panels
        cbar_ax_err = fig.add_axes([0.92, 0.10, 0.01, 0.30])
        sm_err = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-err_abs, vmax=err_abs), cmap="seismic")
        fig.colorbar(sm_err, cax=cbar_ax_err).set_label(units, fontsize=10)

        fig.suptitle(f"{ts}  —  {var.upper()}  |  {n_members} members",
                     fontsize=13, fontweight="bold", y=0.98)
        plt.subplots_adjust(left=0.04, right=0.90, top=0.93, bottom=0.08,
                            wspace=0.12, hspace=0.28)

        if save_path:
            sp = f"{save_path}_{ts}_{var}.jpg"
            os.makedirs(os.path.dirname(sp), exist_ok=True)
            fig.savefig(sp, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        else:
            plt.show()
