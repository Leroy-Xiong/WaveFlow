"""
Post-processing class for generated wave samples.
Includes visualization and analysis tools.
"""
import numpy as np
import pandas as pd
import xarray as xr
import json
import os
import glob
import random
import concurrent.futures
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tqdm import tqdm
from modules.plot.plot_wave import plot_wave_data, plot_wave_data_global
from modules.utils import HK_REGION


def _process_timestamp_viz_standalone(args):
    """Standalone function for parallel timestamp visualization."""
    (timestamp, sample_files_list, data_dir, save_dir,
     num_nodes, lons, lats, region_mode, gt_file_format,
     gt_filename_prefix, gt_var_mapping, variables, rng_seed,
     era5_dir) = args

    import numpy as np
    import xarray as xr
    import os
    import random as _random
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from modules.utils import WAVE_VARIABLES

    _random.seed(rng_seed)
    np.random.seed(rng_seed)

    # Load ensemble samples
    ensemble_samples = []
    for sf in sample_files_list:
        try:
            ensemble_samples.append(np.load(sf)['sample'].squeeze())
        except Exception:
            continue
    if not ensemble_samples:
        return None

    ensemble_arr = np.array(ensemble_samples)

    # Determine variables
    current_variables = variables
    if current_variables is None:
        if ensemble_arr.ndim == 3 and ensemble_arr.shape[2] == 4:
            current_variables = ['hs', 'tp', 'dm', 'elev']
        elif ensemble_arr.ndim == 3 and ensemble_arr.shape[2] == 3:
            current_variables = ['hs', 'mwd', 'ssh']
        elif ensemble_arr.ndim == 2:
            current_variables = ['hs']
            ensemble_arr = ensemble_arr[:, :, np.newaxis]
        else:
            num_vars = ensemble_arr.shape[2] if ensemble_arr.ndim == 3 else 1
            current_variables = [f'var_{i}' for i in range(num_vars)]
            if ensemble_arr.ndim == 2:
                ensemble_arr = ensemble_arr[:, :, np.newaxis]
    if ensemble_arr.ndim == 2:
        ensemble_arr = ensemble_arr[:, :, np.newaxis]

    # GT file path
    if region_mode == 'global':
        if gt_file_format == 'npz':
            subdir = "npz_files"
        elif gt_file_format == 'npy':
            subdir = "npy_files_4var"
        else:
            subdir = "nc_files"
        date_part = timestamp[:8]
        hour_part = timestamp[9:11]
        year = date_part[:4]
        gt_file = os.path.join(data_dir, subdir, year, f"{gt_filename_prefix}_{date_part}_{hour_part}.{gt_file_format}")
    else:
        gt_file = os.path.join(data_dir, f"{gt_filename_prefix}_{timestamp}.{gt_file_format}")

    if not os.path.exists(gt_file):
        return None

    # Column layout of the per-timestamp NPY GT product ([N, C] float32);
    # declared by wave_columns.json in the npy directory (global-only format).
    npy_cols = None
    if gt_file_format == 'npy':
        with open(os.path.join(data_dir, subdir, 'wave_columns.json')) as fh:
            npy_cols = json.load(fh)['columns']

    # Load GT
    try:
        if gt_file_format == 'npy':
            gt_data = np.load(gt_file)
        elif gt_file_format == 'npz':
            gt_data = np.load(gt_file, allow_pickle=True)
        else:
            gt_data = xr.load_dataset(gt_file)
            # Filter to HK region when using nc format (nc files contain full domain)
            if region_mode != 'global':
                from modules.utils import HK_REGION
                min_lon, max_lon, min_lat, max_lat = HK_REGION
                gt_data = gt_data.where(
                    (gt_data.lon >= min_lon) & (gt_data.lon <= max_lon) &
                    (gt_data.lat >= min_lat) & (gt_data.lat <= max_lat),
                    drop=True)
    except Exception:
        return None

    ens_size = ensemble_arr.shape[0]
    if ens_size >= 2:
        s_idx = _random.sample(range(ens_size), 2)
    else:
        s_idx = [0, 0]

    generated_files = []

    for var_idx, variable in enumerate(current_variables):
        var_ens = ensemble_arr[:, :, var_idx]
        if variable == 'dm':
            # 循环均值/离散度：10° 与 350° 的均值应为 0° 而非 180°
            rad = np.deg2rad(var_ens)
            ensemble_mean = np.rad2deg(np.arctan2(np.mean(np.sin(rad), axis=0),
                                                  np.mean(np.cos(rad), axis=0))) % 360.0
            circ_dev = np.arctan2(np.sin(rad - np.deg2rad(ensemble_mean)),
                                  np.cos(rad - np.deg2rad(ensemble_mean)))
            ensemble_std = np.rad2deg(np.std(circ_dev, axis=0))
        else:
            ensemble_mean = np.mean(var_ens, axis=0)
            ensemble_std = np.std(var_ens, axis=0)
        sample1 = var_ens[s_idx[0]]
        sample2 = var_ens[s_idx[1]]

        # Extract GT
        var_key = gt_var_mapping.get(variable, variable)
        try:
            if gt_file_format == 'npy':
                gt_values = gt_data[:, npy_cols.index(var_key)].astype(np.float32).squeeze()
            elif gt_file_format == 'npz':
                gt_values = gt_data[var_key].astype(np.float32).squeeze()
            else:
                gt_values = gt_data[var_key].values.squeeze()
        except (KeyError, ValueError):
            continue

        if gt_values.shape[0] != num_nodes or ensemble_mean.shape[0] != num_nodes:
            continue

        difference = ensemble_mean - gt_values
        if variable == 'dm':
            # 循环差值：359° vs 1° 的真实偏差是 -2° 而非 +358°
            difference = (difference + 180.0) % 360.0 - 180.0

        # Colorbar ranges
        if variable == 'elev':
            gt_vmin = float(np.nanpercentile(gt_values, 0.1))
            gt_vmax = float(np.nanpercentile(gt_values, 99.9))
        else:
            gt_vmin = float(np.nanmin(gt_values))
            gt_vmax = float(np.nanmax(gt_values))
        std_vmin, std_vmax = 0.0, float(np.nanmax(ensemble_std))
        diff_abs = float(np.nanmax(np.abs(difference)))
        diff_vmin, diff_vmax = -diff_abs, diff_abs

        var_prop = WAVE_VARIABLES.get(variable, {})
        var_title = var_prop.get('plot_title', variable)
        var_label = var_prop.get('plot_label', variable)

        # dm 是循环变量（0°=360°），用循环色图并固定 0-360 范围，
        # 避免 turbo 在 0°/360° 处出现假的断裂
        val_cmap = 'twilight' if variable == 'dm' else 'turbo'
        # dm 误差 wrap 到 ±180°，-180 与 +180 物理等价，用循环色图
        # twilight_shifted（中点 0 为浅色，两端同色），范围固定 ±180
        diff_cmap = 'twilight_shifted' if variable == 'dm' else 'seismic'
        if variable == 'dm':
            diff_vmin, diff_vmax = -180.0, 180.0
            gt_vmin, gt_vmax = 0.0, 360.0

        def _make_ds(values):
            return xr.Dataset(
                {variable: (['node'], values)},
                coords={'lon': (['node'], lons), 'lat': (['node'], lats), 'time': timestamp}
            )

        if region_mode == 'global':
            # Global: combined interpolated + scatter figures (no individual panels)
            from modules.plot.plot_wave import interpolate_wave_data_to_grid

            # Wrap lon to [-180,180] once
            lon_w = lons.copy()
            lon_w = np.where(lon_w > 180, lon_w - 360, lon_w)

            ds_panels = [
                (gt_values,   gt_vmin,   gt_vmax,   val_cmap,
                 f'{var_title} (Ground Truth)\n{timestamp} | Nodes: {num_nodes}'),
                (ensemble_mean, gt_vmin, gt_vmax, val_cmap,
                 f'{var_title} (Ensemble Mean)\n{timestamp} | Ensemble Size: {ens_size}'),
                (ensemble_std,  std_vmin,  std_vmax,  'viridis',
                 f'{var_title} (Ensemble Std)\n{timestamp} | Ensemble Size: {ens_size}'),
                (sample1,   gt_vmin,   gt_vmax,   val_cmap,
                 f'{var_title} (Sample {s_idx[0]+1})\n{timestamp} | Ensemble Size: {ens_size}'),
                (sample2,   gt_vmin,   gt_vmax,   val_cmap,
                 f'{var_title} (Sample {s_idx[1]+1})\n{timestamp} | Ensemble Size: {ens_size}'),
                (difference, diff_vmin, diff_vmax, diff_cmap,
                 f'{var_title} (Mean - GT)\n{timestamp} | Ensemble Size: {ens_size}'),
            ]
            n_panels = len(ds_panels)

            # Interpolate all panels
            interp_data = []
            for values, vmin, vmax, cmap, title in ds_panels:
                if variable == 'dm' and cmap != 'viridis':
                    # 循环量（值 0-360°、误差 ±180°）在 sin/cos 空间插值，
                    # 避免环绕处线性插值产生断裂带
                    rad = np.deg2rad(values)
                    lm, lam, si = interpolate_wave_data_to_grid(lon_w, lats, np.sin(rad))
                    _, _, ci = interpolate_wave_data_to_grid(lon_w, lats, np.cos(rad))
                    vi = np.rad2deg(np.arctan2(si, ci))
                    if cmap == val_cmap:
                        vi = vi % 360.0
                else:
                    lm, lam, vi = interpolate_wave_data_to_grid(lon_w, lats, values)
                interp_data.append((lm, lam, vi, vmin, vmax, cmap, title))

            # --- Interpolated combined figure ---
            fig_int = plt.figure(figsize=(36, 18))
            for i, (lm, lam, vi, vmin, vmax, cmap, title) in enumerate(interp_data):
                ax = fig_int.add_subplot(2, 3, i + 1,
                                         projection=ccrs.Robinson(central_longitude=0))
                vmasked = np.ma.masked_invalid(vi)
                im = ax.pcolormesh(lm, lam, vmasked, cmap=cmap, shading='auto',
                                   transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
                try:
                    ax.add_feature(cfeature.LAND, color='whitesmoke', alpha=1.0, zorder=10)
                    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, alpha=1.0, zorder=11)
                except Exception as e:
                    print(f"  Warning: cartopy feature error (combined): {e}")
                ax.set_global()
                ax.set_title(title, fontsize=12)
                try:
                    gl = ax.gridlines(draw_labels=False, alpha=0.3,
                                      xlocs=range(-180, 181, 60), ylocs=range(-90, 91, 30))
                except Exception as e:
                    print(f"  Warning: cartopy gridlines error (combined): {e}")
                cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
                cbar.set_label(var_label, fontsize=10)
            plt.tight_layout()
            path_int = os.path.join(save_dir, f'ensemble_{timestamp}_{variable}_combined.jpg')
            fig_int.savefig(path_int, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_int)
            generated_files.append(path_int)

            # --- Scatter combined figure ---
            fig_sc = plt.figure(figsize=(36, 18))
            for i, (values, vmin, vmax, cmap, title) in enumerate(ds_panels):
                ax = fig_sc.add_subplot(2, 3, i + 1,
                                        projection=ccrs.Robinson(central_longitude=0))
                ax.set_global()
                try:
                    ax.add_feature(cfeature.LAND, color='whitesmoke', alpha=1.0, zorder=3)
                    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5, alpha=1.0, zorder=4)
                except Exception as e:
                    print(f"  Warning: cartopy feature error (scatter): {e}")
                sc = ax.scatter(lon_w, lats, c=values, cmap=cmap,
                                s=2, edgecolors='none', transform=ccrs.PlateCarree(),
                                vmin=vmin, vmax=vmax)
                ax.set_title(title, fontsize=12)
                try:
                    gl = ax.gridlines(draw_labels=False, alpha=0.3,
                                      xlocs=range(-180, 181, 60), ylocs=range(-90, 91, 30))
                except Exception as e:
                    print(f"  Warning: cartopy gridlines error (scatter): {e}")
                cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
                cbar.set_label(var_label, fontsize=10)
            plt.tight_layout()
            path_sc = os.path.join(save_dir, f'ensemble_{timestamp}_{variable}_scatter.jpg')
            fig_sc.savefig(path_sc, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_sc)
            generated_files.append(path_sc)
        else:
            # HK region: single combined figure with subplots
            from modules.utils import HK_REGION as _HK
            region_param = _HK

            ds_gt   = _make_ds(gt_values)
            ds_mean = _make_ds(ensemble_mean)
            ds_std  = _make_ds(ensemble_std)
            ds_s1   = _make_ds(sample1)
            ds_s2   = _make_ds(sample2)
            ds_diff = _make_ds(difference)

            panels = [
                (ds_gt,   gt_vmin,   gt_vmax,   val_cmap,   f'{var_title} (Ground Truth)\n{timestamp} | Nodes: {num_nodes}'),
                (ds_mean, gt_vmin,   gt_vmax,   val_cmap,   f'{var_title} (Ensemble Mean)\n{timestamp} | Ensemble Size: {ens_size}'),
                (ds_std,  std_vmin,  std_vmax,  'viridis', f'{var_title} (Ensemble Std)\n{timestamp} | Ensemble Size: {ens_size}'),
                (ds_s1,   gt_vmin,   gt_vmax,   val_cmap,   f'{var_title} (Sample {s_idx[0]+1})\n{timestamp} | Ensemble Size: {ens_size}'),
                (ds_s2,   gt_vmin,   gt_vmax,   val_cmap,   f'{var_title} (Sample {s_idx[1]+1})\n{timestamp} | Ensemble Size: {ens_size}'),
                (ds_diff, diff_vmin, diff_vmax, diff_cmap, f'{var_title} (Mean - GT)\n{timestamp} | Ensemble Size: {ens_size}'),
            ]

            from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

            fig = plt.figure(figsize=(36, 18))
            for i, (ds, vmin, vmax, cmap, title) in enumerate(panels):
                ax = fig.add_subplot(2, 3, i + 1, projection=ccrs.PlateCarree())
                plot_wave_data(
                    dataset=ds, variable=variable, region=list(region_param),
                    vmin=vmin, vmax=vmax, cmap=cmap, ax=ax, return_ax=True,
                    scatter_size=25, custom_title=title, show_cities=False,
                    use_high_res_coastline=True,
                    use_satellite_background=True,
                )
                # Add lat/lon labels manually (avoid cartopy gridlines LinearRing bug)
                lon_ticks = [113.6, 113.8, 114.0, 114.2, 114.4, 114.6]
                lat_ticks = [22.0, 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7]
                ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
                ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
                ax.xaxis.set_major_formatter(LongitudeFormatter())
                ax.yaxis.set_major_formatter(LatitudeFormatter())
                ax.tick_params(labelsize=8)
            plt.tight_layout()
            path_combined = os.path.join(save_dir, f'ensemble_{timestamp}_{variable}.jpg')
            plt.savefig(path_combined, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            generated_files.append(path_combined)

    # ---- ERA5 input visualization ----
    if era5_dir:
        _save_era5_figure_standalone(timestamp, save_dir, era5_dir, region_mode,
                                     generated_files)

    return generated_files


def _save_traj_field_figure(values, variable, timestamp, out_path,
                            vmin, vmax, cmap, lons, lats, region_mode, num_nodes,
                            show_cbar=False):
    """Plot one trajectory field (nodes) on the HK/global map and save the figure.

    Shared by the per-member and ensemble-stats trajectory workers.
    show_cbar: draw a full-height colorbar (default matplotlib sizing, no
    label, ~5 sparse ticks). Used for the ensemble summary figures only.
    Returns out_path on success, None on failure.
    """
    import numpy as np
    import xarray as xr
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from modules.plot.plot_wave import plot_wave_data

    if values.shape[0] != num_nodes:
        print(f"  Warning: node count mismatch for {out_path} ({values.shape[0]} != {num_nodes})")
        return None

    if region_mode == 'global':
        region_param = None
    else:
        from modules.utils import HK_REGION as _HK
        region_param = list(_HK)

    ds = xr.Dataset(
        {variable: (['node'], values)},
        coords={'lon': (['node'], lons), 'lat': (['node'], lats), 'time': timestamp}
    )

    fig = plt.figure(figsize=(5.8, 3.5) if show_cbar else (5, 3.5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    plot_wave_data(
        dataset=ds, variable=variable,
        region=region_param,
        vmin=vmin, vmax=vmax, cmap=cmap, ax=ax, return_ax=True,
        scatter_size=8, custom_title='', show_cities=False,
        show_colorbar=False, show_nodes=False,
        show_left_labels=False, show_bottom_labels=False,
        show_top_labels=False,
        use_high_res_coastline=True,
        use_satellite_background=(region_mode != 'global'),
    )

    if show_cbar:
        from matplotlib import ticker as mticker
        # Default-sized (full-height) colorbar, no label, sparse ticks
        cbar = plt.colorbar(ax._wave_scatter, ax=ax)
        cbar.locator = mticker.MaxNLocator(nbins=5)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    # Trajectory figures are exported as vector PDF with a transparent
    # figure background (satellite background stays raster at `dpi`)
    if out_path.lower().endswith(('.jpg', '.jpeg')):
        out_path = os.path.splitext(out_path)[0] + '.pdf'
    fig.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    return out_path


def _detect_traj_variables(n_channels):
    """Variable names from channel count (hs/tp/dm/elev convention)."""
    if n_channels == 4:
        return ['hs', 'tp', 'dm', 'elev']
    if n_channels == 3:
        return ['hs', 'mwd', 'ssh']
    if n_channels == 1:
        return ['hs']
    return [f'var_{i}' for i in range(n_channels)]


def _process_trajectory_sample_standalone(task):
    """Standalone worker: plot ODE trajectory snapshots for one (timestamp, ens) sample.

    One figure per (ODE time t, variable) — variables are exported as
    separate images, not combined into one big figure.

    Args:
        task: tuple of (timestamp, ens_idx, t_files, save_dir, num_nodes,
                        lons, lats, region_mode, variables, var_ranges)
              t_files: list of (ode_t, file_path) sorted by ode_t.
              var_ranges: optional dict {variable: (vmin, vmax)} computed from
                          ground truth; keeps the color scale of a variable
                          consistent across all t and ensemble members.
                          None -> per-figure percentile ranges.

    Returns:
        List of generated file paths.
    """
    import os as _os
    import numpy as np

    (timestamp, ens_idx, t_files, save_dir, num_nodes,
     lons, lats, region_mode, variables, var_ranges) = task

    generated_files = []

    for ode_t, file_path in t_files:
        try:
            arr = np.load(file_path)['sample']
        except Exception as e:
            print(f"  Warning: failed to load {file_path}: {e}")
            continue
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]

        current_variables = variables
        if current_variables is None:
            current_variables = _detect_traj_variables(arr.shape[-1])

        for var_idx, variable in enumerate(current_variables):
            if var_idx >= arr.shape[-1]:
                break
            values = arr[..., var_idx].astype(np.float32)

            if var_ranges and variable in var_ranges:
                # GT-based range: consistent across all t and ensemble members
                vmin, vmax = var_ranges[variable]
            else:
                vmin = float(np.nanpercentile(values, 0.1))
                vmax = float(np.nanpercentile(values, 99.9))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                    vmin, vmax = None, None

            out_path = _os.path.join(
                save_dir, f'traj_{timestamp}_ens{ens_idx:03d}_t{ode_t:.2f}_{variable}.jpg'
            )
            result = _save_traj_field_figure(
                values, variable, timestamp, out_path, vmin, vmax, 'turbo',
                lons, lats, region_mode, num_nodes)
            if result:
                generated_files.append(result)

    return generated_files


def _process_trajectory_ensemble_standalone(task):
    """Standalone worker: plot ensemble mean / std of ODE trajectory snapshots
    for one timestamp.

    One figure per (ODE time t, statistic, variable). Mean uses the GT-based
    color range (same as member figures); spread (std) figures auto-scale to
    their own min/max (no fixed vmin/vmax).

    Args:
        task: tuple of (timestamp, t_map, save_dir, num_nodes,
                        lons, lats, region_mode, variables, var_ranges)
              t_map: list of (ode_t, [member file paths]) sorted by ode_t.

    Returns:
        List of generated file paths.
    """
    import os as _os
    import numpy as np

    (timestamp, t_map, save_dir, num_nodes,
     lons, lats, region_mode, variables, var_ranges) = task

    generated_files = []

    # First pass: load members and compute mean / std per ODE time
    stats = []  # (ode_t, mean_arr, std_arr)
    for ode_t, member_paths in t_map:
        arrs = []
        for p in member_paths:
            try:
                a = np.load(p)['sample']
            except Exception as e:
                print(f"  Warning: failed to load {p}: {e}")
                continue
            if a.ndim == 1:
                a = a[:, np.newaxis]
            arrs.append(a.astype(np.float32))
        if len(arrs) < 2:
            continue
        stack = np.stack(arrs)  # (members, nodes, C)
        stats.append((ode_t, stack.mean(axis=0), stack.std(axis=0)))

    if not stats:
        return generated_files

    n_ch = stats[0][1].shape[-1]
    current_variables = variables
    if current_variables is None:
        current_variables = _detect_traj_variables(n_ch)

    for ode_t, mean_arr, std_arr in stats:
        for var_idx, variable in enumerate(current_variables):
            if var_idx >= n_ch:
                break
            mean_values = mean_arr[..., var_idx]
            std_values = std_arr[..., var_idx]

            if var_ranges and variable in var_ranges:
                vmin, vmax = var_ranges[variable]
            else:
                vmin = float(np.nanpercentile(mean_values, 0.1))
                vmax = float(np.nanpercentile(mean_values, 99.9))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                    vmin, vmax = None, None

            mean_path = _os.path.join(
                save_dir, f'traj_{timestamp}_ensmean_t{ode_t:.2f}_{variable}.jpg')
            result = _save_traj_field_figure(
                mean_values, variable, timestamp, mean_path, vmin, vmax, 'turbo',
                lons, lats, region_mode, num_nodes, show_cbar=True)
            if result:
                generated_files.append(result)

            # Spread (std) figures auto-scale per figure (no fixed vmin/vmax):
            # a shared scale would flatten the spread contrast at most t values
            std_path = _os.path.join(
                save_dir, f'traj_{timestamp}_ensstd_t{ode_t:.2f}_{variable}.jpg')
            result = _save_traj_field_figure(
                std_values, variable, timestamp, std_path,
                None, None, 'viridis',
                lons, lats, region_mode, num_nodes, show_cbar=True)
            if result:
                generated_files.append(result)

    return generated_files


def _process_trajectory_gt_standalone(task):
    """Standalone worker: plot ground truth for one timestamp, one figure per
    variable, using the same GT-based color range as the trajectory figures.

    Args:
        task: tuple of (timestamp, gt_file, gt_file_format, gt_var_mapping,
                        save_dir, num_nodes, lons, lats, region_mode,
                        variables, var_ranges)

    Returns:
        List of generated file paths.
    """
    import os as _os
    import numpy as np
    import xarray as xr

    (timestamp, gt_file, gt_file_format, gt_var_mapping, save_dir, num_nodes,
     lons, lats, region_mode, variables, var_ranges) = task

    generated_files = []

    try:
        if gt_file_format == 'npz':
            gt_data = np.load(gt_file, allow_pickle=True)
        else:
            gt_data = xr.load_dataset(gt_file)
            if region_mode != 'global':
                from modules.utils import HK_REGION
                min_lon, max_lon, min_lat, max_lat = HK_REGION
                gt_data = gt_data.where(
                    (gt_data.lon >= min_lon) & (gt_data.lon <= max_lon) &
                    (gt_data.lat >= min_lat) & (gt_data.lat <= max_lat),
                    drop=True)
    except Exception as e:
        print(f"  Warning: failed to load GT file {gt_file}: {e}")
        return generated_files

    for variable in variables:
        var_key = gt_var_mapping.get(variable, variable)
        try:
            if gt_file_format == 'npz':
                values = gt_data[var_key].astype(np.float32).squeeze()
            else:
                values = gt_data[var_key].values.squeeze().astype(np.float32)
        except KeyError:
            print(f"  Warning: variable {var_key} not found in GT file {gt_file}")
            continue

        if var_ranges and variable in var_ranges:
            vmin, vmax = var_ranges[variable]
        else:
            vmin = float(np.nanmin(values))
            vmax = float(np.nanmax(values))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                vmin, vmax = None, None

        out_path = _os.path.join(save_dir, f'traj_{timestamp}_gt_{variable}.jpg')
        result = _save_traj_field_figure(
            values, variable, timestamp, out_path, vmin, vmax, 'turbo',
            lons, lats, region_mode, num_nodes)
        if result:
            generated_files.append(result)

    return generated_files


def _save_era5_figure_standalone(timestamp, save_dir, era5_dir, region_mode,
                                 generated_files):
    """Generate a 4-panel ERA5 wind + pressure figure (standalone worker).
    Supports both NC (.nc) and NPZ (.npz) formats."""
    import os as _os
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    from modules.utils import ERA5_VARIABLE_NAMES

    ts_compact = timestamp.replace('T', '_')
    era5_nc_path = _os.path.join(era5_dir, timestamp[:4], f'era5_{timestamp}.nc')
    era5_npz_path = _os.path.join(era5_dir, timestamp[:4], f'era5_{ts_compact}.npz')

    if _os.path.exists(era5_nc_path):
        # NC format
        try:
            ds = xr.load_dataset(era5_nc_path, engine='netcdf4')
        except Exception:
            return
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        if ds[lat_name].values[0] < ds[lat_name].values[-1]:
            ds = ds.sortby(lat_name, ascending=False)
        u_data = ds[ERA5_VARIABLE_NAMES[0]].values
        v_data = ds[ERA5_VARIABLE_NAMES[1]].values
        p_data = ds[ERA5_VARIABLE_NAMES[2]].values / 100.0
        lat = ds[lat_name].values
        lon_e = ds[lon_name].values
    elif _os.path.exists(era5_npz_path):
        # NPZ format (global data)
        try:
            d = np.load(era5_npz_path, allow_pickle=True)
            coord_file = _os.path.join(era5_dir, 'era5_coordinates.npz')
            if not _os.path.exists(coord_file):
                coord_file = _os.path.join(_os.path.dirname(era5_dir), 'era5_coordinates.npz')
            c = np.load(coord_file) if _os.path.exists(coord_file) else None
            if c is None:
                d.close()
                return
            u_data = d[ERA5_VARIABLE_NAMES[0]]
            v_data = d[ERA5_VARIABLE_NAMES[1]]
            p_data = d[ERA5_VARIABLE_NAMES[2]] / 100.0
            lat = c['latitude'].astype(np.float32)
            lon_e = c['longitude'].astype(np.float32)
            d.close()
            c.close()
        except Exception:
            return
    else:
        return
    wind_speed = np.sqrt(u_data ** 2 + v_data ** 2)

    lat = ds[lat_name].values
    lon_e = ds[lon_name].values
    lon_p = np.where(lon_e > 180, lon_e - 360, lon_e)
    Lon, Lat = np.meshgrid(lon_p, lat)

    wind_vmax = max(float(np.nanmax(np.abs(u_data))),
                    float(np.nanmax(np.abs(v_data))))
    p_vmin, p_vmax = float(np.nanmin(p_data)), float(np.nanmax(p_data))
    ws_vmax = float(np.nanmax(wind_speed))

    is_global = region_mode == 'global'
    if is_global:
        fig = plt.figure(figsize=(20, 12))
        proj = ccrs.Robinson(central_longitude=0)
    else:
        fig = plt.figure(figsize=(10, 9))
        proj = ccrs.PlateCarree()

    panels = [
        (0, u_data,     'RdBu_r',  -wind_vmax, wind_vmax, 'm/s',
         f'10m U-Wind (East-West)\n{timestamp}'),
        (1, v_data,     'RdBu_r',  -wind_vmax, wind_vmax, 'm/s',
         f'10m V-Wind (North-South)\n{timestamp}'),
        (2, wind_speed, 'viridis',  0,          ws_vmax,   'm/s',
         f'10m Wind Speed\n{timestamp}'),
        (3, p_data,     'RdYlBu_r', p_vmin,     p_vmax,    'hPa',
         f'MSL Pressure\n{timestamp}'),
    ]

    for idx, data_2d, cmap, vmin, vmax, clabel, title in panels:
        ax = fig.add_subplot(2, 2, idx + 1, projection=proj)

        if is_global:
            ax.set_global()
        else:
            map_extent = [float(lon_p.min()), float(lon_p.max()),
                          float(lat.min()), float(lat.max())]
            ax.set_extent(map_extent, crs=proj)

        ax.add_feature(cfeature.LAND, color='whitesmoke', alpha=1, zorder=3)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.4, zorder=4)

        # draw_labels=False avoids Cartopy LinearRing crash on global extents
        gl = ax.gridlines(draw_labels=False, alpha=0.3,
                          xlocs=range(-180, 181, 60), ylocs=range(-90, 91, 30))
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        im = ax.pcolormesh(Lon, Lat, data_2d, cmap=cmap,
                           transform=ccrs.PlateCarree(), shading='auto',
                           vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label=clabel)

        if idx == 2:
            ss = max(1, min(len(lat), len(lon_p)) // 30)
            ax.quiver(Lon[::ss, ::ss], Lat[::ss, ::ss],
                      u_data[::ss, ::ss], v_data[::ss, ::ss],
                      transform=ccrs.PlateCarree(), alpha=0.6, width=0.002,
                      color='white', scale=500)

        ax.set_title(title, fontsize=16)

    plt.tight_layout()
    path = _os.path.join(save_dir, f'ensemble_{timestamp}_era5_input.jpg')
    try:
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    except Exception:
        pass  # cartopy may fail on gridlines draw during savefig
    plt.close(fig)
    generated_files.append(path)


def process_single_timestamp_multivar_standalone(timestamp, sample_files_list, data_dir, variables, 
                                                  num_ensemble_samples, num_nodes, gt_file_format='nc',
                                                  region_mode='hk', gt_filename_prefix='wave', 
                                                  gt_var_mapping=None):
    """
    Standalone helper function to process a single timestamp for multivar RMSE calculation.
    Intended to be used with ProcessPoolExecutor.
    
    Args:
        timestamp: Timestamp string
        sample_files_list: List of sample file paths for this timestamp
        data_dir: Directory containing ground truth files
        variables: List of variable names to calculate RMSE for
        num_ensemble_samples: Number of ensemble samples to use
        num_nodes: Expected number of nodes
        gt_file_format: Ground truth file format - 'nc' for NetCDF or 'npz' for NPZ files
        region_mode: 'global' or 'hk' - determines ground truth file naming
        gt_filename_prefix: Prefix for ground truth files ('wave' for HK, 'global_hindcast' for global)
        gt_var_mapping: Dict mapping variable names to file keys (e.g., {'hs': 'WWM_1'} for global)
    """
    if gt_var_mapping is None:
        gt_var_mapping = {}
    # Limit number of ensemble samples if specified
    if num_ensemble_samples is not None and num_ensemble_samples > 0:
        sample_files_list = sample_files_list[:num_ensemble_samples]
    
    ensemble_size = len(sample_files_list)
    
    # Load all ensemble members for this timestamp
    ensemble_samples = []
    for sample_file in sample_files_list:
        try:
            sample_data = np.load(sample_file)
            sample = sample_data['sample'].squeeze()
            ensemble_samples.append(sample)
        except Exception as e:
            # print(f"Error loading sample {os.path.basename(sample_file)}: {e}")
            continue
    
    if len(ensemble_samples) == 0:
        # print(f"Warning: No valid samples loaded for timestamp {timestamp}")
        return None
    
    # Calculate ensemble mean as the final prediction
    ensemble_samples_array = np.array(ensemble_samples) 
    
    # Calculate mean across ensemble members (axis 0)
    ensemble_mean = np.mean(ensemble_samples_array, axis=0)
    
    # Ensure correct shape (num_nodes, num_channels)
    if ensemble_mean.ndim == 1:
        # Single-channel output (e.g., global model with only hs)
        ensemble_mean = ensemble_mean.reshape(-1, 1)
    
    if ensemble_mean.ndim == 2:
        if ensemble_mean.shape[0] == num_nodes:
            pass # (nodes, channels)
        elif ensemble_mean.shape[1] == num_nodes:
            ensemble_mean = ensemble_mean.T # Transpose to (nodes, channels)
        else:
            # print(f"Warning: Dimension mismatch for {timestamp}. Nodes: {num_nodes}, Shape: {ensemble_mean.shape}")
            return None
    else:
         # print(f"Warning: Expected 2D array (nodes, channels) for ensemble mean, got {ensemble_mean.shape} for {timestamp}")
         return None
        
    if ensemble_mean.shape[1] < len(variables):
        # Fewer channels than requested variables — only process what's available
        variables = variables[:ensemble_mean.shape[1]]

    # dm is circular (degrees): replace its linear ensemble mean with the
    # circular mean (atan2 of mean sin/cos), otherwise members straddling
    # 0/360 (e.g. 350° and 10°) average to a spurious ~180° prediction.
    if 'dm' in variables:
        dm_ch = variables.index('dm')
        if dm_ch < ensemble_mean.shape[1] and ensemble_samples_array.ndim == 3:
            ens_arr = ensemble_samples_array
            if ens_arr.shape[1] != num_nodes and ens_arr.shape[2] == num_nodes:
                ens_arr = ens_arr.transpose(0, 2, 1)
            if ens_arr.shape[1] == num_nodes and dm_ch < ens_arr.shape[2]:
                rad = np.deg2rad(ens_arr[:, :, dm_ch].astype(np.float64))
                ensemble_mean[:, dm_ch] = np.rad2deg(
                    np.arctan2(np.sin(rad).mean(axis=0), np.cos(rad).mean(axis=0))) % 360.0

    # Construct ground truth file path based on region mode
    if region_mode == 'global':
        if gt_file_format == 'npz':
            subdir = "npz_files"
        elif gt_file_format == 'npy':
            subdir = "npy_files_4var"
        else:
            subdir = "nc_files"
    else:
        subdir = ""
    
    if region_mode == 'global':
        # Global format: global_hindcast_YYYYMMDD_HH.npz
        date_part = timestamp[:8]  # YYYYMMDD
        hour_part = timestamp[9:11]  # HH
        year = date_part[:4]
        formatted_timestamp = f"{date_part}_{hour_part}"
        filename = f"{gt_filename_prefix}_{formatted_timestamp}.{gt_file_format}"
    else:
        # HK format: wave_YYYYMMDDTHH.nc
        year = None
        filename = f"{gt_filename_prefix}_{timestamp}.{gt_file_format}"

    if year:
        gt_file = os.path.join(data_dir, subdir, year, filename)
    else:
        gt_file = os.path.join(data_dir, subdir, filename)
    
    if not os.path.exists(gt_file):
        # print(f"Warning: Ground truth file not found: {gt_file}. Skipping...")
        return None
    
    # Load ground truth data
    try:
        if gt_file_format == 'npy':
            # Per-timestamp NPY GT product (global-only): [N, C] float32 whose
            # column layout is declared by wave_columns.json in the npy dir.
            gt_arr = np.load(gt_file)
            with open(os.path.join(data_dir, subdir, 'wave_columns.json')) as fh:
                npy_cols = json.load(fh)['columns']

            row = {
                'timestamp': timestamp,
                'ensemble_size': ensemble_size
            }

            for i, var_name in enumerate(variables):
                var_key = gt_var_mapping.get(var_name, var_name)

                if var_key not in npy_cols:
                    row[f'{var_name}_rmse'] = np.nan
                    continue

                gt_values = gt_arr[:, npy_cols.index(var_key)].astype(np.float32)
                pred_values = ensemble_mean[:, i]

                if gt_values.shape[0] != num_nodes:
                    row[f'{var_name}_rmse'] = np.nan
                    continue

                valid_mask = ~np.isnan(gt_values)
                if np.sum(valid_mask) == 0:
                    row[f'{var_name}_rmse'] = np.nan
                    continue

                if var_name == 'dm':
                    # Circular RMSE for directional variables
                    abs_diff = np.abs(pred_values[valid_mask] - gt_values[valid_mask])
                    angular_diff = np.minimum(abs_diff, 360.0 - abs_diff)
                    rmse = np.sqrt(np.mean(angular_diff ** 2))
                else:
                    diff = pred_values[valid_mask] - gt_values[valid_mask]
                    rmse = np.sqrt(np.mean(diff ** 2))

                row[f'{var_name}_rmse'] = rmse
        elif gt_file_format == 'npz':
            # Load from NPZ file
            gt_data = np.load(gt_file, allow_pickle=True)
            
            row = {
                'timestamp': timestamp,
                'ensemble_size': ensemble_size
            }
            
            for i, var_name in enumerate(variables):
                # Map variable names to NPZ keys using region-specific mapping
                var_key = gt_var_mapping.get(var_name, var_name)
                
                if var_key not in gt_data:
                    row[f'{var_name}_rmse'] = np.nan
                    continue
                    
                gt_values = gt_data[var_key].astype(np.float32).squeeze()
                
                pred_values = ensemble_mean[:, i]
                
                # Flatten GT if needed
                if gt_values.ndim > 1 and gt_values.size == num_nodes:
                     gt_values = gt_values.flatten()
                
                if gt_values.shape[0] != num_nodes:
                     row[f'{var_name}_rmse'] = np.nan
                     continue

                # Calculate RMSE
                valid_mask = ~np.isnan(gt_values)
                if np.sum(valid_mask) == 0:
                    row[f'{var_name}_rmse'] = np.nan
                    continue
                
                # Use specific RMSE calculation for Directional Mean (dm)
                if var_name == 'dm':
                    # Circular RMSE for directional variables
                    # Formula: RMSE = sqrt(mean(min(|y - y_hat|, 360 - |y - y_hat|)^2))
                    abs_diff = np.abs(pred_values[valid_mask] - gt_values[valid_mask])
                    
                    # Calculate shortest angular distance
                    angular_diff = np.minimum(abs_diff, 360.0 - abs_diff)
                    mse = np.mean(angular_diff ** 2)
                    rmse = np.sqrt(mse)
                else:
                    # Standard RMSE for other variables
                    diff = pred_values[valid_mask] - gt_values[valid_mask]
                    mse = np.mean(diff ** 2)
                    rmse = np.sqrt(mse)
                
                row[f'{var_name}_rmse'] = rmse
        else:
            # Load from NetCDF file
            ds_gt = xr.load_dataset(gt_file)
            
            # Filter to HK region to match samples
            min_lon, max_lon, min_lat, max_lat = HK_REGION
            ds_gt = ds_gt.where(
                (ds_gt.lon >= min_lon) & (ds_gt.lon <= max_lon) & 
                (ds_gt.lat >= min_lat) & (ds_gt.lat <= max_lat), 
                drop=True
            )
            
            row = {
                'timestamp': timestamp,
                'ensemble_size': ensemble_size
            }
            
            for i, var_name in enumerate(variables):
                if var_name not in ds_gt:
                    row[f'{var_name}_rmse'] = np.nan
                    continue
                    
                gt_values = ds_gt[var_name].values.squeeze()
                
                pred_values = ensemble_mean[:, i]
                
                # Flatten GT if needed
                if gt_values.ndim > 1 and gt_values.size == num_nodes:
                     gt_values = gt_values.flatten()
                
                if gt_values.shape[0] != num_nodes:
                     row[f'{var_name}_rmse'] = np.nan
                     continue

                # Calculate RMSE
                valid_mask = ~np.isnan(gt_values)
                if np.sum(valid_mask) == 0:
                    row[f'{var_name}_rmse'] = np.nan
                    continue
                
                # Use specific RMSE calculation for Directional Mean (dm)
                if var_name == 'dm':
                    # Circular RMSE for directional variables
                    # Formula: RMSE = sqrt(mean(min(|y - y_hat|, 360 - |y - y_hat|)^2))
                    abs_diff = np.abs(pred_values[valid_mask] - gt_values[valid_mask])
                    
                    # Calculate shortest angular distance
                    angular_diff = np.minimum(abs_diff, 360.0 - abs_diff)
                    mse = np.mean(angular_diff ** 2)
                    rmse = np.sqrt(mse)
                else:
                    # Standard RMSE for other variables
                    diff = pred_values[valid_mask] - gt_values[valid_mask]
                    mse = np.mean(diff ** 2)
                    rmse = np.sqrt(mse)
                
                row[f'{var_name}_rmse'] = rmse
        
        return row
        
    except Exception as e:
        # print(f"Error processing ground truth for {timestamp}: {e}")
        return None


def process_single_timestamp_pernode_se_standalone(timestamp, sample_files_list, data_dir, variable,
                                                    num_ensemble_samples, num_nodes, gt_file_format='nc',
                                                    region_mode='hk', gt_filename_prefix='wave',
                                                    gt_var_mapping=None):
    """
    Standalone helper for computing per-node squared error (ensemble_mean - gt)^2 for a
    single timestamp, plus auxiliary statistics needed for normalized RMSE plots.

    Returns:
        dict with keys:
            'timestamp': str
            'sq_err': (num_nodes,) per-node squared error, np.nan where invalid
            'gt_values': (num_nodes,) ground-truth values (np.nan where invalid)
            'valid_mask': (num_nodes,) bool
            'ts_rmse': float, RMSE over valid spatial nodes for this timestamp
            'ts_gt_std': float, std of GT over valid spatial nodes (for time-axis normalization)
            'ts_gt_mean': float, mean of GT over valid spatial nodes
        Returns None on failure.
    """
    if gt_var_mapping is None:
        gt_var_mapping = {}

    if num_ensemble_samples is not None and num_ensemble_samples > 0:
        sample_files_list = sample_files_list[:num_ensemble_samples]

    ensemble_samples = []
    for sample_file in sample_files_list:
        try:
            sample_data = np.load(sample_file)
            sample = sample_data['sample'].squeeze()
            ensemble_samples.append(sample)
        except Exception:
            continue

    if len(ensemble_samples) == 0:
        return None

    ensemble_mean = np.mean(np.array(ensemble_samples), axis=0)

    if region_mode == 'global':
        subdir = "npz_files" if gt_file_format == 'npz' else "nc_files"
    else:
        subdir = ""

    if region_mode == 'global':
        date_part = timestamp[:8]
        hour_part = timestamp[9:11]
        formatted_timestamp = f"{date_part}_{hour_part}"
        filename = f"{gt_filename_prefix}_{formatted_timestamp}.{gt_file_format}"
        year = date_part[:4]
        gt_file = os.path.join(data_dir, subdir, year, filename)
    else:
        filename = f"{gt_filename_prefix}_{timestamp}.{gt_file_format}"
        gt_file = os.path.join(data_dir, subdir, filename)
    if not os.path.exists(gt_file):
        return None

    try:
        if gt_file_format == 'npz':
            gt_data = np.load(gt_file, allow_pickle=True)
            var_key = gt_var_mapping.get(variable, variable)
            if var_key not in gt_data:
                return None
            gt_values = gt_data[var_key].astype(np.float32).squeeze()
        else:
            ds_gt = xr.load_dataset(gt_file)
            if region_mode != 'global':
                min_lon, max_lon, min_lat, max_lat = HK_REGION
                ds_gt = ds_gt.where(
                    (ds_gt.lon >= min_lon) & (ds_gt.lon <= max_lon) &
                    (ds_gt.lat >= min_lat) & (ds_gt.lat <= max_lat),
                    drop=True
                )
            if variable not in ds_gt:
                return None
            gt_values = ds_gt[variable].values.squeeze()

        if gt_values.ndim > 1 and gt_values.size == num_nodes:
            gt_values = gt_values.flatten()

        # If sample has multiple channels (multivar), take first one matching `variable`
        if ensemble_mean.ndim == 2:
            # Assume axis 0 is nodes and axis 1 is channels; take first channel.
            if ensemble_mean.shape[0] == num_nodes:
                ensemble_mean = ensemble_mean[:, 0]
            elif ensemble_mean.shape[1] == num_nodes:
                ensemble_mean = ensemble_mean[0, :]
            else:
                return None

        if gt_values.shape[0] != num_nodes or ensemble_mean.shape[0] != num_nodes:
            return None

        valid_mask = ~np.isnan(gt_values) & ~np.isnan(ensemble_mean)
        sq_err = np.full(num_nodes, np.nan, dtype=np.float64)
        gt_out = np.full(num_nodes, np.nan, dtype=np.float64)
        if np.sum(valid_mask) == 0:
            return None
        diff = ensemble_mean[valid_mask] - gt_values[valid_mask]
        sq_err[valid_mask] = diff ** 2
        gt_out[valid_mask] = gt_values[valid_mask]

        ts_rmse = float(np.sqrt(np.mean(diff ** 2)))
        ts_gt_std = float(np.std(gt_values[valid_mask]))
        ts_gt_mean = float(np.mean(gt_values[valid_mask]))

        return {
            'timestamp': timestamp,
            'sq_err': sq_err,
            'gt_values': gt_out,
            'valid_mask': valid_mask,
            'ts_rmse': ts_rmse,
            'ts_gt_std': ts_gt_std,
            'ts_gt_mean': ts_gt_mean,
        }
    except Exception:
        return None


def process_single_timestamp_gt_stats_standalone(timestamp, data_dir, variable, num_nodes,
                                                  gt_file_format='nc', region_mode='hk',
                                                  gt_filename_prefix='wave', gt_var_mapping=None):
    """
    Lightweight helper that loads only ground truth at a given timestamp and returns
    its spatial mean and std. Used by visualize_temporal_rmse to normalize RMSE.
    """
    if gt_var_mapping is None:
        gt_var_mapping = {}

    if region_mode == 'global':
        subdir = "npz_files" if gt_file_format == 'npz' else "nc_files"
    else:
        subdir = ""

    if region_mode == 'global':
        date_part = timestamp[:8]
        hour_part = timestamp[9:11]
        formatted_timestamp = f"{date_part}_{hour_part}"
        filename = f"{gt_filename_prefix}_{formatted_timestamp}.{gt_file_format}"
        year = date_part[:4]
        gt_file = os.path.join(data_dir, subdir, year, filename)
    else:
        filename = f"{gt_filename_prefix}_{timestamp}.{gt_file_format}"
        gt_file = os.path.join(data_dir, subdir, filename)
    if not os.path.exists(gt_file):
        return None

    try:
        if gt_file_format == 'npz':
            gt_data = np.load(gt_file, allow_pickle=True)
            var_key = gt_var_mapping.get(variable, variable)
            if var_key not in gt_data:
                return None
            gt_values = gt_data[var_key].astype(np.float32).squeeze()
        else:
            ds_gt = xr.load_dataset(gt_file)
            if region_mode != 'global':
                min_lon, max_lon, min_lat, max_lat = HK_REGION
                ds_gt = ds_gt.where(
                    (ds_gt.lon >= min_lon) & (ds_gt.lon <= max_lon) &
                    (ds_gt.lat >= min_lat) & (ds_gt.lat <= max_lat),
                    drop=True
                )
            if variable not in ds_gt:
                return None
            gt_values = ds_gt[variable].values.squeeze()

        if gt_values.ndim > 1 and gt_values.size == num_nodes:
            gt_values = gt_values.flatten()
        if gt_values.shape[0] != num_nodes:
            return None

        valid = gt_values[~np.isnan(gt_values)]
        if valid.size == 0:
            return None
        return {
            'timestamp': timestamp,
            'gt_mean': float(np.mean(valid)),
            'gt_std': float(np.std(valid)),
        }
    except Exception:
        return None


def process_single_timestamp_ensemble_standalone(timestamp, sample_files_list, data_dir, variable,
                                                  num_ensemble_samples, num_nodes, gt_file_format='nc',
                                                  region_mode='hk', gt_filename_prefix='wave',
                                                  gt_var_mapping=None):
    """
    Standalone helper function to process a single timestamp for single-variable ensemble RMSE.
    Intended to be used with ProcessPoolExecutor.
    """
    if gt_var_mapping is None:
        gt_var_mapping = {}

    # Limit number of ensemble samples if specified
    if num_ensemble_samples is not None and num_ensemble_samples > 0:
        sample_files_list = sample_files_list[:num_ensemble_samples]

    ensemble_size = len(sample_files_list)

    # Load all ensemble members for this timestamp
    ensemble_samples = []
    for sample_file in sample_files_list:
        try:
            sample_data = np.load(sample_file)
            sample = sample_data['sample'].squeeze()
            ensemble_samples.append(sample)
        except Exception:
            continue

    if len(ensemble_samples) == 0:
        return None

    ensemble_samples_array = np.array(ensemble_samples)
    ensemble_mean = np.mean(ensemble_samples_array, axis=0)

    # Select the channel corresponding to the requested variable
    # Sample channels: hs=0, tp=1, dm=2, elev=3 (or dm_sin=2,dm_cos=3,elev=4 for 5ch)
    if ensemble_mean.ndim == 2 and ensemble_mean.shape[1] > 1:
        var_to_idx = {'hs': 0, 'tp': 1, 'dm': 2, 'elev': 3}
        idx = var_to_idx.get(variable)
        if idx is not None and idx < ensemble_mean.shape[1]:
            ensemble_mean = ensemble_mean[:, idx]

    # Construct ground truth file path
    if region_mode == 'global':
        subdir = "npz_files" if gt_file_format == 'npz' else "nc_files"
    else:
        subdir = ""

    if region_mode == 'global':
        date_part = timestamp[:8]
        hour_part = timestamp[9:11]
        formatted_timestamp = f"{date_part}_{hour_part}"
        filename = f"{gt_filename_prefix}_{formatted_timestamp}.{gt_file_format}"
        year = date_part[:4]
        gt_file = os.path.join(data_dir, subdir, year, filename)
    else:
        filename = f"{gt_filename_prefix}_{timestamp}.{gt_file_format}"
        gt_file = os.path.join(data_dir, subdir, filename)

    if not os.path.exists(gt_file):
        return None

    try:
        if gt_file_format == 'npz':
            gt_data = np.load(gt_file, allow_pickle=True)
            var_key = gt_var_mapping.get(variable, variable)
            if var_key not in gt_data:
                return None
            gt_values = gt_data[var_key].astype(np.float32).squeeze()
        else:
            ds_gt = xr.load_dataset(gt_file)
            if region_mode != 'global':
                min_lon, max_lon, min_lat, max_lat = HK_REGION
                ds_gt = ds_gt.where(
                    (ds_gt.lon >= min_lon) & (ds_gt.lon <= max_lon) &
                    (ds_gt.lat >= min_lat) & (ds_gt.lat <= max_lat),
                    drop=True
                )
            if variable not in ds_gt:
                return None
            gt_values = ds_gt[variable].values.squeeze()

        # Flatten if needed
        if gt_values.ndim > 1 and gt_values.size == num_nodes:
            gt_values = gt_values.flatten()

        if gt_values.shape[0] != num_nodes or ensemble_mean.shape[0] != num_nodes:
            return None

        valid_mask = ~np.isnan(gt_values)
        if np.sum(valid_mask) == 0:
            return None

        diff = ensemble_mean[valid_mask] - gt_values[valid_mask]
        mse = np.mean(diff ** 2)
        rmse = np.sqrt(mse)

        return {
            'timestamp': timestamp,
            'ensemble_size': ensemble_size,
            'rmse': rmse
        }
    except Exception:
        return None


