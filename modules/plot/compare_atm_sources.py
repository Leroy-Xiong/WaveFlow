"""
Qualitative comparison of ensemble-mean predictions from different
atmospheric forcing sources (ERA5 reanalysis vs IFS ENS forecasts at
multiple lead times) against the wave ground truth.

For a small set of randomly selected timestamps shared by all sources, draws
a (variables x sources) grid where the first column is the ground truth and
the remaining columns are the ensemble means from each source. This makes it
easy to visually compare how forecast errors grow with lead time and how
they compare to the analysis-driven baseline.

Usage:
    from modules.plot.compare_atm_sources import compare_ensemble_mean_across_sources

    compare_ensemble_mean_across_sources(
        sample_dirs={
            "ERA5":   ".../sample_epoch49_20260204_195847/samples",
            "ENS 6h": ".../sample_epoch49_ens6h_.../samples",
            ...
        },
        gt_data_dir="./data/hindcast/processed",
        save_dir="./output/plot/qualitative_atm_sources",
        num_samples=4,
    )
"""
from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm

import cartopy.crs as ccrs

from modules.plot.plot_wave import plot_wave_data
from modules.utils import HK_REGION, WAVE_VARIABLES


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _list_ensemble_files(sample_dir: str) -> Dict[str, List[str]]:
    """Group ensemble npz files by timestamp."""
    out: Dict[str, List[str]] = defaultdict(list)
    if not os.path.isdir(sample_dir):
        return out
    for name in os.listdir(sample_dir):
        if not (name.startswith("sample_") and name.endswith(".npz") and "_ens" in name):
            continue
        ts = name.replace("sample_", "").split("_ens")[0]
        out[ts].append(os.path.join(sample_dir, name))
    return out


def _compute_ensemble_mean(files: Sequence[str]) -> Optional[np.ndarray]:
    """Load all ensemble member npz files and return their mean."""
    arrs = []
    for f in files:
        try:
            arrs.append(np.load(f)["sample"].squeeze())
        except Exception as e:
            print(f"[WARN] failed to load {f}: {e}")
    if not arrs:
        return None
    return np.mean(np.stack(arrs, axis=0), axis=0)  # (N,) or (N, C)


def _load_gt_hk(gt_data_dir: str, timestamp: str,
                region: Sequence[float] = HK_REGION) -> Optional[xr.Dataset]:
    """Load HK ground truth NetCDF and filter to the HK_REGION extent so its
    node count matches the 7788 nodes used by the model."""
    path = os.path.join(gt_data_dir, f"wave_{timestamp}.nc")
    if not os.path.exists(path):
        return None
    try:
        ds = xr.load_dataset(path)
        min_lon, max_lon, min_lat, max_lat = region
        ds = ds.where(
            (ds.lon >= min_lon) & (ds.lon <= max_lon) &
            (ds.lat >= min_lat) & (ds.lat <= max_lat),
            drop=True,
        )
        return ds
    except Exception as e:
        print(f"[WARN] failed to open GT {path}: {e}")
        return None


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def compare_ensemble_mean_across_sources(
    sample_dirs: Dict[str, str],
    gt_data_dir: str,
    coord_file: str = "./data/hindcast/hk_coordinates.npy",
    save_dir: str = "./output/plot/qualitative_atm_sources",
    num_samples: int = 4,
    variables: Sequence[str] = ("hs", "tp", "dm", "elev"),
    region: Sequence[float] = HK_REGION,
    seed: int = 42,
    scatter_size: float = 14,
    timestamps: Optional[Sequence[str]] = None,
    filename_prefix_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Draw a (variable x source) panel for each selected timestamp.

    Args:
        sample_dirs: ordered dict {label: samples_dir}; the order controls
            column order in the figure. Example: {"ERA5": ..., "ENS 6h": ...}.
        gt_data_dir: directory holding HK wave_*.nc ground truth files.
        coord_file: path to hk_coordinates.npy (lon, lat) ordered same as nodes.
        save_dir: folder to save jpg files.
        num_samples: number of timestamps to randomly visualize when
            ``timestamps`` is not provided.
        variables: which wave variables to plot (rows).
        region: [min_lon, max_lon, min_lat, max_lat] for the map extent.
        seed: RNG seed for picking timestamps (ignored when ``timestamps`` is set).
        scatter_size: scatter point size on the map.
        timestamps: Optional explicit list of ``YYYYMMDDTHH`` strings to plot.
            When given, the random sampling step is skipped and ``num_samples``
            / ``seed`` are ignored. Any requested timestamp missing from a
            source is logged and skipped.
        filename_prefix_map: Optional dict ``{ts: prefix}`` that prepends a
            human-readable label to the saved file (e.g. typhoon name).
    """
    random.seed(seed)
    np.random.seed(seed)

    # --- 1. Find timestamps that exist in EVERY source -------------------
    per_source_groups = {label: _list_ensemble_files(d) for label, d in sample_dirs.items()}
    ts_sets = {k: set(g.keys()) for k, g in per_source_groups.items() if g}
    if not ts_sets:
        print("[ERROR] No ensemble files found in any sample_dirs.")
        return

    if timestamps is not None:
        requested = list(timestamps)
        # Keep only sources that have ALL requested timestamps; warn on the rest.
        keep = [k for k in ts_sets if all(t in ts_sets[k] for t in requested)]
        dropped = [k for k in ts_sets if k not in keep]
        if dropped:
            print(f"[WARN] Sources missing one or more requested timestamps "
                  f"dropped: {dropped}")
            for k in dropped:
                miss = [t for t in requested if t not in ts_sets[k]]
                print(f"       {k} missing: {miss}")
        if len(keep) < 2:
            print(f"[ERROR] Fewer than 2 sources have all requested "
                  f"timestamps: kept={keep}")
            return
        sample_dirs = {k: sample_dirs[k] for k in sample_dirs if k in keep}
        per_source_groups = {k: per_source_groups[k] for k in sample_dirs}
        selected_ts = list(requested)
        print(f"Using {len(selected_ts)} explicit timestamps with "
              f"{len(sample_dirs)} sources: {list(sample_dirs.keys())}")
        print(f"Selected timestamps: {selected_ts}")
    else:
        # Try full intersection first; if empty, iteratively drop the source whose
        # exclusion yields the largest remaining intersection. This handles cases
        # like ENS 6h (valid_hour = 06/18) being disjoint from 12/24/48/72h
        # (valid_hour = 00/12).
        def _intersect(keys):
            return set.intersection(*[ts_sets[k] for k in keys]) if keys else set()

        keep = list(ts_sets.keys())
        common_ts = _intersect(keep)
        dropped = []
        while not common_ts and len(keep) > 2:
            # Find which source, when dropped, yields the largest remaining intersection
            best_drop, best_size = None, -1
            for k in keep:
                cand = [x for x in keep if x != k]
                sz = len(_intersect(cand))
                if sz > best_size:
                    best_size, best_drop = sz, k
            if best_size <= 0:
                break
            dropped.append(best_drop)
            keep.remove(best_drop)
            common_ts = _intersect(keep)

        if dropped:
            print(f"[WARN] Sources with disjoint timestamp sets dropped from joint comparison: {dropped}")
        if not common_ts:
            print("[ERROR] Could not find any timestamps shared across at least two sources.")
            for k, s in ts_sets.items():
                print(f"  {k}: {len(s)} ts")
            return

        # Filter sample_dirs to only the kept sources, preserving original order
        sample_dirs = {k: sample_dirs[k] for k in sample_dirs if k in keep}
        per_source_groups = {k: per_source_groups[k] for k in sample_dirs}

        common_ts = sorted(common_ts)
        print(f"Found {len(common_ts)} timestamps common to {len(sample_dirs)} sources: {list(sample_dirs.keys())}")

        if num_samples > len(common_ts):
            print(f"  requested {num_samples} > available {len(common_ts)}; using all.")
            num_samples = len(common_ts)
        selected_ts = sorted(random.sample(common_ts, num_samples))
        print(f"Selected timestamps: {selected_ts}")

    # --- 2. Load coordinates --------------------------------------------
    coords = np.load(coord_file)
    # HK convention (matches modules/postprocess_samples.PostProcess):
    #   coords[:, 0] = latitude, coords[:, 1] = longitude
    lats = coords[:, 0]
    lons = coords[:, 1]
    num_nodes = len(lons)

    os.makedirs(save_dir, exist_ok=True)

    var_to_idx = {v: i for i, v in enumerate(variables)}
    src_labels = list(sample_dirs.keys())
    n_cols = 1 + len(src_labels)  # GT + each source
    n_rows = len(variables)

    # --- 3. Loop over selected timestamps -------------------------------
    for ts in tqdm(selected_ts, desc="Qualitative comparison"):
        # Load GT
        gt_ds = _load_gt_hk(gt_data_dir, ts, region=region)
        if gt_ds is None:
            print(f"[SKIP] no GT for {ts}")
            continue

        # Pre-compute ensemble means for each source
        src_means: Dict[str, np.ndarray] = {}
        for label, ts_groups in per_source_groups.items():
            files = ts_groups.get(ts)
            arr = _compute_ensemble_mean(files) if files else None
            if arr is None:
                print(f"[SKIP] missing samples for {label} @ {ts}")
                continue
            src_means[label] = arr

        if len(src_means) != len(src_labels):
            # incomplete; skip
            continue

        # --- 4. Build figure ---------------------------------------------
        # Compute figsize from region aspect ratio so Cartopy subplots
        # fill their grid cells without internal whitespace.
        lon_range = region[1] - region[0]
        lat_range = region[3] - region[2]
        per_row_h = 2.2
        per_col_w = per_row_h * lon_range / lat_range
        figsize = (per_col_w * n_cols, per_row_h * n_rows)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=figsize,
            subplot_kw={"projection": ccrs.PlateCarree()},
            squeeze=False,
        )
        fig.subplots_adjust(wspace=0.03, hspace=0.01)

        for row, var in enumerate(variables):
            # Variable color range: pin to the GT range (so all sources share scale)
            try:
                gt_vals = gt_ds[var].values.squeeze()
            except KeyError:
                # Try gt_var_mapping fallback (HK uses identical names)
                continue
            if var == "elev":
                vmin = float(np.nanpercentile(gt_vals, 0.5))
                vmax = float(np.nanpercentile(gt_vals, 99.5))
            else:
                vmin = float(np.nanmin(gt_vals))
                vmax = float(np.nanmax(gt_vals))

            var_meta = WAVE_VARIABLES.get(var, {})
            var_label = var_meta.get("plot_label", var)

            # Column 0: GT
            ds_gt = xr.Dataset(
                {var: (["node"], gt_vals)},
                coords={
                    "lon": (["node"], lons),
                    "lat": (["node"], lats),
                    "time": ts,
                },
            )
            is_last_row = (row == n_rows - 1)
            plot_wave_data(
                ds_gt, variable=var, region=region, ax=axes[row, 0],
                vmin=vmin, vmax=vmax, scatter_size=scatter_size,
                show_cities=False, show_colorbar=False, show_nodes=False,
                custom_title=(f"{var_label}\nGround Truth" if row == 0 else ""),
                show_left_labels=True,
                show_bottom_labels=is_last_row,
            )

            # Columns 1..N: source ensemble means
            for col, label in enumerate(src_labels, start=1):
                arr = src_means[label]
                if arr.ndim == 2:
                    vals = arr[:, var_to_idx[var]] if var in var_to_idx else arr[:, 0]
                else:
                    vals = arr
                if vals.shape[0] != num_nodes:
                    print(f"[WARN] node mismatch for {label} @ {ts}: {vals.shape} vs {num_nodes}")
                    continue
                ds_pred = xr.Dataset(
                    {var: (["node"], vals)},
                    coords={
                        "lon": (["node"], lons),
                        "lat": (["node"], lats),
                        "time": ts,
                    },
                )
                plot_wave_data(
                    ds_pred, variable=var, region=region, ax=axes[row, col],
                    vmin=vmin, vmax=vmax, scatter_size=scatter_size,
                    show_cities=False, show_colorbar=False, show_nodes=False,
                    custom_title=(f"{var_label}\n{label} (Ens. Mean)" if row == 0 else ""),
                    show_left_labels=False,
                    show_bottom_labels=is_last_row,
                )

            # Trim title size for compactness on every cell of this row
            for c in range(n_cols):
                axes[row, c].title.set_fontsize(10)

            # One shared colorbar per row, anchored to the last column.
            mappable = getattr(axes[row, n_cols - 1], "_wave_scatter", None)
            if mappable is not None:
                cbar = fig.colorbar(
                    mappable,
                    ax=list(axes[row, :]),
                    shrink=0.85, pad=0.012, aspect=30,
                )
                cbar.set_label(var_label, fontsize=10)
                cbar.ax.tick_params(labelsize=9)

        fig.suptitle(
            f"Qualitative comparison @ {ts} — Ensemble Mean across atmospheric forcings",
            fontsize=14, fontweight="bold",
        )

        prefix = (filename_prefix_map or {}).get(ts)
        fname = f"{prefix}_compare_{ts}.jpg" if prefix else f"compare_{ts}.jpg"
        out_path = os.path.join(save_dir, fname)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path}")
