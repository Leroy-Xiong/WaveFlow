"""
Compare ERA5 reanalysis vs IFS ENS forecast data for multi-frame inputs.
Visualizes multi-row comparisons: ERA5 (row 1) + ENS at multiple lead times (rows 2+).
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from modules.utils import (ERA5_VARIABLE_NAMES, 
                            ERA5_HK_LON_RANGE_1, ERA5_HK_LAT_RANGE_1,
                            ERA5_HK_LON_RANGE_2, ERA5_HK_LAT_RANGE_2,
                            ERA5_HK_LON_RANGE_3, ERA5_HK_LAT_RANGE_3)


def _get_lon_lat_range(era5_region=1):
    """Get lon/lat range for a given ERA5 region preset."""
    if era5_region == 1:
        return ERA5_HK_LON_RANGE_1, ERA5_HK_LAT_RANGE_1
    elif era5_region == 2:
        return ERA5_HK_LON_RANGE_2, ERA5_HK_LAT_RANGE_2
    elif era5_region == 3:
        return ERA5_HK_LON_RANGE_3, ERA5_HK_LAT_RANGE_3
    else:
        return ERA5_HK_LON_RANGE_1, ERA5_HK_LAT_RANGE_1


def _load_nc_frame(path, lon_range, lat_range):
    """Load a single NC frame, filter by region, ensure lat descending. Returns [3, H, W]."""
    if not os.path.exists(path):
        return None
    ds = xr.load_dataset(path, engine='netcdf4')
    lon_name = 'longitude' if 'longitude' in ds else 'lon'
    lat_name = 'latitude' if 'latitude' in ds else 'lat'
    
    min_lon, max_lon = lon_range
    min_lat, max_lat = lat_range
    ds = ds.where(
        (ds[lon_name] >= min_lon) & (ds[lon_name] < max_lon) &
        (ds[lat_name] >= min_lat) & (ds[lat_name] < max_lat),
        drop=True
    )
    # Ensure lat descending (ERA5 convention)
    if ds[lat_name].values[0] < ds[lat_name].values[-1]:
        ds = ds.sortby(lat_name, ascending=False)
    
    arrays = [ds[var].values.astype(np.float32) for var in ERA5_VARIABLE_NAMES]
    return np.stack(arrays, axis=0)


def _build_ens_frame_plan(valid_time_str, ens_lead_time, num_frames,
                          ens_init=None):
    """Build ENS frame plan: same logic as WaveDataset._build_ens_frame_plan.

    ens_init ('YYYYMMDDTHH') pins the issue time (one operational cycle whose
    lead varies per valid time); None keeps init = valid - ens_lead_time."""
    dt_base = datetime.strptime(valid_time_str, "%Y%m%dT%H")
    dt_main_init = (datetime.strptime(ens_init, "%Y%m%dT%H") if ens_init
                    else dt_base - timedelta(hours=ens_lead_time))
    
    frames_info = []
    for i in range(num_frames - 1, -1, -1):
        dt_valid_frame = dt_base - timedelta(hours=i * 6)
        if dt_valid_frame >= dt_main_init:
            dt_init = dt_main_init
        else:
            hour = dt_valid_frame.hour
            dt_init = dt_valid_frame.replace(
                hour=12 if hour >= 12 else 0, minute=0, second=0
            )
        lead_hours = int((dt_valid_frame - dt_init).total_seconds() // 3600)
        frames_info.append({
            'valid_time': dt_valid_frame.strftime("%Y%m%dT%H"),
            'init_time': dt_init.strftime("%Y%m%dT%H"),
            'lead_time': lead_hours,
        })
    return frames_info


def compare_era5_vs_ens_multi_lead(timestamps, num_frames=8,
                                    ens_lead_times=[6, 12, 24, 48, 72],
                                    era5_dir='./data/era5/nc_files',
                                    ens_dir='./data/ifs/ens',
                                    era5_region=2,
                                    output_dir='./output/plot/data/era5_vs_ens',
                                    subdir_map=None):
    """
    Compare ERA5 and ENS multi-frame inputs at multiple lead times.
    
    For each timestamp and each variable, generates a multi-row plot:
    - Row 1: ERA5 (ground truth)
    - Row 2+: ENS at each lead time (6h, 12h, 24h, 48h, 72h)
    
    Each row has `num_frames` columns (8 frames = 2 days of 6-hourly data).
    
    Args:
        timestamps: List of valid time strings in format YYYYMMDDTHH.
        num_frames: Number of input frames (8 = 2 days).
        ens_lead_times: List of ENS lead times in hours.
        era5_dir: Path to ERA5 NC data.
        ens_dir: Path to ENS data.
        era5_region: ERA5 region preset (1, 2, or 3).
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    lon_range, lat_range = _get_lon_lat_range(era5_region)
    var_labels = ['u10 (m/s)', 'v10 (m/s)', 'MSL (Pa)']
    
    
    for ts in timestamps:
        print(f"\nProcessing timestamp: {ts}")
        dt_base = datetime.strptime(ts, "%Y%m%dT%H")
        
        # Build ERA5 frames (historical, 6h apart, oldest first)
        era5_frames = []
        era5_time_labels = []
        for i in range(num_frames - 1, -1, -1):
            dt_offset = dt_base - timedelta(hours=i * 6)
            t_str = dt_offset.strftime("%Y%m%dT%H")
            year = t_str[:4]
            path = os.path.join(era5_dir, year, f'era5_{t_str}.nc')
            frame = _load_nc_frame(path, lon_range, lat_range)
            if frame is None:
                print(f"  ERA5 missing: {t_str}")
                break
            era5_frames.append(frame)
            era5_time_labels.append(dt_offset.strftime('%m/%d %HZ'))
        
        if len(era5_frames) != num_frames:
            print(f"  Skipping {ts}: missing ERA5 frames")
            continue
        
        # Build ENS frames for each lead time
        all_ens_frames = {}  # lead_time -> list of frames
        all_ens_labels = {}  # lead_time -> list of labels
        available_lead_times = []
        for lt in ens_lead_times:
            ens_plan = _build_ens_frame_plan(ts, lt, num_frames)
            ens_frames = []
            ens_labels = []
            missing = False
            for info in ens_plan:
                path = os.path.join(
                    ens_dir, f"lead_time_{info['lead_time']}h",
                    info['init_time'][:4], f"ifs_ens_{info['init_time']}.nc"
                )
                frame = _load_nc_frame(path, lon_range, lat_range)
                if frame is None:
                    print(f"  ENS missing: lt={lt}h, valid={info['valid_time']}, "
                          f"file_lt={info['lead_time']}h, init={info['init_time']}")
                    missing = True
                    break
                ens_frames.append(frame)
                ens_labels.append(f"lt={info['lead_time']}h")
            
            if not missing:
                all_ens_frames[lt] = ens_frames
                all_ens_labels[lt] = ens_labels
                available_lead_times.append(lt)
            else:
                print(f"  Skipping ENS {lt}h for {ts}: missing files")
        
        if not available_lead_times:
            print(f"  Skipping {ts}: no ENS lead times available")
            continue
        
        # Plot each variable
        actual_num_rows = 1 + len(available_lead_times)
        for var_idx, (var_name, var_label) in enumerate(zip(ERA5_VARIABLE_NAMES, var_labels)):
            fig, axes = plt.subplots(actual_num_rows, num_frames,
                                     figsize=(num_frames * 2.5, actual_num_rows * 2.3))
            
            # Compute global vmin/vmax across all data for this variable
            all_vals = [era5_frames[j][var_idx] for j in range(num_frames)]
            for lt in available_lead_times:
                all_vals += [all_ens_frames[lt][j][var_idx] for j in range(num_frames)]
            data_min = min(v.min() for v in all_vals)
            data_max = max(v.max() for v in all_vals)
            if 'pressure' in var_name:
                # Pressure: match the reference era5_pressure_*_hk.png style.
                vmin, vmax = data_min, data_max
                cmap = 'RdYlBu_r'
            else:
                # Wind components (u10, v10): symmetric diverging colorbar about 0.
                amax = max(abs(data_min), abs(data_max))
                vmin, vmax = -amax, amax
                cmap = 'RdBu_r'
            
            # Row 0: ERA5
            for j in range(num_frames):
                im = axes[0, j].imshow(era5_frames[j][var_idx], vmin=vmin, vmax=vmax,
                                        cmap=cmap, aspect='auto', origin='upper')
                axes[0, j].set_title(era5_time_labels[j], fontsize=7)
                axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
            axes[0, 0].set_ylabel('ERA5', fontsize=10, fontweight='bold')
            
            # Rows 1+: ENS at each available lead time
            for row_idx, lt in enumerate(available_lead_times, start=1):
                for j in range(num_frames):
                    im = axes[row_idx, j].imshow(
                        all_ens_frames[lt][j][var_idx], vmin=vmin, vmax=vmax,
                        cmap=cmap, aspect='auto', origin='upper')
                    axes[row_idx, j].set_title(all_ens_labels[lt][j], fontsize=7)
                    axes[row_idx, j].set_xticks([]); axes[row_idx, j].set_yticks([])
                axes[row_idx, 0].set_ylabel(f'ENS {lt}h', fontsize=10, fontweight='bold')
            
            fig.suptitle(f'{var_label} — ERA5 vs ENS at multiple lead times\n'
                         f'valid_time={ts}, {num_frames} frames (2 days)',
                         fontsize=12, fontweight='bold')
            fig.subplots_adjust(right=0.88, top=0.93, hspace=0.3, wspace=0.1)
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label=var_label, pad=0.02)
            
            ts_subdir = (subdir_map or {}).get(ts)
            save_dir = os.path.join(output_dir, ts_subdir) if ts_subdir else output_dir
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{ts}_{var_name.replace(" ", "_")}_multi_lead.png')
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {save_path}")
    
    print("\nDone!")


def _snap_to_6h(dt):
    """Snap a datetime to the nearest 6-hourly slot (00/06/12/18Z)."""
    base = dt.replace(minute=0, second=0, microsecond=0)
    candidates = [
        base.replace(hour=(base.hour // 6) * 6),
        base.replace(hour=(base.hour // 6) * 6) + timedelta(hours=6),
    ]
    return min(candidates, key=lambda c: abs(c - dt))


def _check_coverage(ts, num_frames, max_lead, era5_dir, ens_dir):
    """Return True if all ERA5 input frames and the ENS init at max_lead exist."""
    dt = datetime.strptime(ts, "%Y%m%dT%H")
    for i in range(num_frames - 1, -1, -1):
        dt_offset = dt - timedelta(hours=i * 6)
        t_str = dt_offset.strftime("%Y%m%dT%H")
        path = os.path.join(era5_dir, t_str[:4], f'era5_{t_str}.nc')
        if not os.path.exists(path):
            return False
    dt_init = dt - timedelta(hours=max_lead)
    init_str = dt_init.strftime("%Y%m%dT%H")
    ens_path = os.path.join(ens_dir, f"lead_time_{max_lead}h",
                            init_str[:4], f"ifs_ens_{init_str}.nc")
    return os.path.exists(ens_path)


def default_typhoon_name_lookup():
    """Built-in mapping of ``YYYYMMDD`` date → typhoon name for the top HK
    Hs peaks in 2021-2022. Two consecutive days are included per event so a
    daylight peak still maps correctly after the 6h/12h snapping step."""
    return {
        '20211012': 'Kompasu',
        '20211013': 'Kompasu',
        '20211220': 'Rai',
        '20211221': 'Rai',
        '20220701': 'Chaba',
        '20220702': 'Chaba',
        '20220824': 'Ma-on',
        '20220825': 'Ma-on',
        '20221101': 'Nalgae',
        '20221102': 'Nalgae',
    }


def pick_top_hs_peaks(csv_path='./output/plot/hk_hs_timeseries/hk_hs_timeseries_2021_2022.csv',
                      num_peaks=5,
                      min_separation_hours=120,
                      hs_column='hs_max'):
    """Pick the top-N distinct Hs peaks from a HK Hs time-series CSV.

    Rows are ranked by ``hs_column`` (descending) and greedily kept if they
    are at least ``min_separation_hours`` away from every already-picked peak
    — this avoids selecting multiple hours of the same typhoon.

    Returns a list of ``(datetime, hs_value)`` tuples sorted by time.
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"HK Hs time-series CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(hs_column, ascending=False).reset_index(drop=True)

    sep = timedelta(hours=min_separation_hours)
    picked = []
    for _, row in df.iterrows():
        t = row['time'].to_pydatetime()
        if all(abs(t - p) >= sep for p, _ in picked):
            picked.append((t, float(row[hs_column])))
            if len(picked) >= num_peaks:
                break
    picked.sort(key=lambda x: x[0])
    return picked


def top_hs_peaks_and_compare(csv_path='./output/plot/hk_hs_timeseries/hk_hs_timeseries_2021_2022.csv',
                              num_peaks=5,
                              min_separation_hours=120,
                              hs_column='hs_max',
                              num_frames=8,
                              ens_lead_times=[12, 24, 48, 72],
                              era5_dir='./data/era5/nc_files',
                              ens_dir='./data/ifs/ens',
                              era5_region=2,
                              output_dir='./output/plot/data/era5_vs_ens',
                              typhoon_names=None):
    """
    Pick top-N distinct HK Hs peaks (i.e. typhoon events) from a hindcast
    time-series CSV and generate ERA5-vs-ENS multi-lead comparison plots
    at those valid times.

    Each event's plots are written to a per-typhoon subfolder under
    ``output_dir`` (e.g. ``output_dir/Kompasu_20211013T00/``) so the figures
    are easy to identify.

    Args:
        csv_path: HK Hs time-series CSV produced by
            ``WaveDataProcessor.plot_hk_hs_timeseries`` (columns:
            ``time, hs_mean, hs_max, hs_p95, hs_p99``).
        num_peaks: Number of distinct peak events to keep.
        min_separation_hours: Minimum spacing between selected peaks so that
            two hours within the same typhoon are not double-counted.
        hs_column: Which column of the CSV to rank by.
        num_frames, ens_lead_times, era5_dir, ens_dir, era5_region, output_dir:
            Forwarded to ``compare_era5_vs_ens_multi_lead``.
        typhoon_names: Optional dict mapping ``YYYYMMDD`` date strings to a
            typhoon name. Merged on top of the built-in defaults; pass
            ``{date: None}`` to suppress a default entry.
    """
    name_lookup = default_typhoon_name_lookup()
    if typhoon_names:
        name_lookup.update(typhoon_names)

    picked = pick_top_hs_peaks(
        csv_path=csv_path,
        num_peaks=num_peaks,
        min_separation_hours=min_separation_hours,
        hs_column=hs_column,
    )

    # Snap each peak to the nearest 6h slot, then validate coverage; if the
    # closest slot is missing data, fall back to neighbouring 6h slots within ±18h.
    max_lead = max(ens_lead_times)
    timestamps = []
    for t, hs in picked:
        candidates = [_snap_to_6h(t) + timedelta(hours=k) for k in (0, -6, 6, -12, 12, -18, 18)]
        chosen = None
        for c in candidates:
            ts = c.strftime("%Y%m%dT%H")
            if _check_coverage(ts, num_frames, max_lead, era5_dir, ens_dir):
                chosen = ts
                break
        if chosen is None:
            print(f"Skipping peak at {t} (Hs={hs:.2f}): no nearby 6h slot has full ERA5+ENS coverage")
            continue
        timestamps.append(chosen)
        print(f"Peak {t} (Hs={hs:.2f}) -> using valid_time={chosen}")

    if not timestamps:
        print("No valid timestamps with full coverage among the top peaks!")
        return

    # Build per-event subfolder names: "<TyphoonName>_<valid_time>" if known.
    subdir_map = {}
    for ts in timestamps:
        date_key = ts[:8]
        name = name_lookup.get(date_key)
        subdir_map[ts] = f"{name}_{ts}" if name else ts

    print(f"\nTop-{len(timestamps)} HK Hs peak timestamps: {timestamps}")
    print(f"Per-event output folders: {list(subdir_map.values())}")
    compare_era5_vs_ens_multi_lead(
        timestamps=timestamps,
        num_frames=num_frames,
        ens_lead_times=ens_lead_times,
        era5_dir=era5_dir,
        ens_dir=ens_dir,
        era5_region=era5_region,
        output_dir=output_dir,
        subdir_map=subdir_map,
    )


def random_sample_and_compare(num_samples=3, num_frames=8,
                               ens_lead_times=[12, 24, 48, 72],
                               era5_dir='./data/era5/nc_files',
                               ens_dir='./data/ifs/ens',
                               era5_region=2,
                               output_dir='./output/plot/data/era5_vs_ens',
                               seed=42):
    """
    Randomly sample timestamps and generate ERA5 vs ENS multi-lead comparison plots.
    
    Samples timestamps from available ERA5 data that also have sufficient ENS coverage.
    Only scans years that have ENS data available.
    """
    np.random.seed(seed)
    
    # Only scan years that have ENS data
    ens_years = set()
    sample_lt_dir = os.path.join(ens_dir, f"lead_time_{ens_lead_times[0]}h")
    if os.path.isdir(sample_lt_dir):
        for d in os.listdir(sample_lt_dir):
            if d.isdigit():
                ens_years.add(d)
    
    # Find available ERA5 timestamps (every 6h to match ENS frame spacing)
    era5_files = []
    for year_dir in sorted(os.listdir(era5_dir)):
        if year_dir not in ens_years:
            continue
        year_path = os.path.join(era5_dir, year_dir)
        if not os.path.isdir(year_path):
            continue
        for f in sorted(os.listdir(year_path)):
            if f.startswith('era5_') and f.endswith('.nc'):
                ts = f.replace('era5_', '').replace('.nc', '')
                # Consider 6-hourly timestamps (00, 06, 12, 18Z)
                if ts.endswith('T00') or ts.endswith('T06') or ts.endswith('T12') or ts.endswith('T18'):
                    era5_files.append(ts)
    
    if not era5_files:
        print("No ERA5 files found!")
        return
    
    # Filter: need num_frames * 6h = 42h of history before valid_time
    # and ENS coverage (max lead = 72h, so init must exist 72h before valid_time)
    max_lead = max(ens_lead_times)
    valid_timestamps = []
    for ts in era5_files:
        dt = datetime.strptime(ts, "%Y%m%dT%H")
        # Check ERA5 frames exist (going back num_frames-1 * 6h)
        all_exist = True
        for i in range(num_frames - 1, -1, -1):
            dt_offset = dt - timedelta(hours=i * 6)
            t_str = dt_offset.strftime("%Y%m%dT%H")
            year = t_str[:4]
            path = os.path.join(era5_dir, year, f'era5_{t_str}.nc')
            if not os.path.exists(path):
                all_exist = False
                break
        if not all_exist:
            continue
        # Check ENS init file exists for max lead time
        dt_init = dt - timedelta(hours=max_lead)
        init_str = dt_init.strftime("%Y%m%dT%H")
        ens_path = os.path.join(ens_dir, f"lead_time_{max_lead}h",
                                init_str[:4], f"ifs_ens_{init_str}.nc")
        if not os.path.exists(ens_path):
            continue
        valid_timestamps.append(ts)
    
    if not valid_timestamps:
        print("No valid timestamps with complete ERA5 history!")
        return
    
    # Random sample
    num_samples = min(num_samples, len(valid_timestamps))
    sampled = np.random.choice(valid_timestamps, size=num_samples, replace=False).tolist()
    print(f"Sampled timestamps: {sampled}")
    
    compare_era5_vs_ens_multi_lead(
        timestamps=sampled,
        num_frames=num_frames,
        ens_lead_times=ens_lead_times,
        era5_dir=era5_dir,
        ens_dir=ens_dir,
        era5_region=era5_region,
        output_dir=output_dir,
    )


# Keep old function for backward compatibility
def compare_era5_vs_ens(timestamps, num_frames=8, ens_lead_time=24,
                         era5_dir='./data/era5/nc_files', ens_dir='./data/ifs/ens',
                         era5_region=2, output_dir='./output/plot/data/era5_vs_ens'):
    """Legacy single-lead-time comparison. Use compare_era5_vs_ens_multi_lead instead."""
    compare_era5_vs_ens_multi_lead(
        timestamps=timestamps,
        num_frames=num_frames,
        ens_lead_times=[ens_lead_time],
        era5_dir=era5_dir,
        ens_dir=ens_dir,
        era5_region=era5_region,
        output_dir=output_dir,
    )
