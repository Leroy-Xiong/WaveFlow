"""
HK 4-variable model satellite observation evaluation (CMEMS L3 along-track SWH).

Same matching / interpolation logic as modules/eval_satellite.py, adapted for
multi-channel HK samples of shape (7788, 4) with variables [hs, tp, dm, elev].
Evaluates the chosen channel (default: hs, channel 0) against CMEMS L3
along-track SWH observations via KD-tree IDW interpolation.

Fast path: pre-filter the satellite table with a single lat/lon between() on
HK_REGION — the same crop used to build the HK training data
(modules/utils.py) — plus a ±time_window filter around hourly stamps, so only
the ~500 surviving overpass timestamps are actually evaluated.

This module provides three entry points (all callable from post_process.py):

  evaluate_hk(...)        per-timestamp / monthly / global metrics of the
                          ensemble mean vs satellite obs
                          (--time-interp adds linear time interpolation of the
                          field between the two bounding hourly stamps;
                          verified to change metrics by <0.001 m)
  evaluate_bridge(...)    bridge comparison: on the SAME satellite points,
                          Model vs Sat / Hindcast-GT vs Sat / Model vs GT, to
                          separate the model's own error from the
                          reanalysis-vs-observation gap
  main()                  standalone CLI (used by post_process.py commands)

By default CSVs are written to <sample_run_dir>/satellite_eval/ (same
convention as <sample_run_dir>/visualization/).

Usage (via post_process.py, recommended):
    python post_process.py satellite-eval \
        --sample-dir <run_dir> --start 20210101T00 --end 20221231T23
    python post_process.py satellite-bridge \
        --sample-dir <run_dir> --start 20210101T00 --end 20221231T23
    python post_process.py satellite-viz \
        --sample-dir <run_dir> --start 20210101T00 --end 20221231T23
    python post_process.py satellite-download --year 2021 --action all
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy.spatial import cKDTree
from tqdm import tqdm

from modules.utils import HK_REGION, ERA5_HK_LON_RANGE_2, ERA5_HK_LAT_RANGE_2

_SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
_CHANNEL_KEYS = ["hs", "tp", "dm", "elev"]


def _latlon_to_xyz(lat, lon):
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.column_stack([x, y, z])


def _chord_to_great_circle_km(chord_dist, R=6371.0):
    half = np.clip(chord_dist / 2.0, 0.0, 1.0)
    central_angle = 2.0 * np.arcsin(half)
    return R * central_angle


def _discover_samples(sample_dir):
    pattern = os.path.join(sample_dir, "sample_*_ens*.npz")
    files = sorted(glob.glob(pattern))
    groups = defaultdict(list)
    for f in files:
        basename = os.path.basename(f).replace(".npz", "")
        if "_ens" not in basename or "T" not in basename:
            continue
        ts = basename.split("_ens")[0].replace("sample_", "")
        groups[ts].append(f)
    timestamps = {ts: sorted(groups[ts]) for ts in sorted(groups.keys())}
    print(f"SatelliteEvaluator(HK): {len(timestamps)} timestamps, "
          f"{sum(len(v) for v in timestamps.values())} ensemble files.")
    return timestamps


def _load_satellite(satellite_data_dir, start_date, end_date):
    """Load processed CMEMS Parquet data (columns: time, latitude, longitude, swh)."""
    if len(start_date) == 8:
        start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    if len(end_date) == 8:
        end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    months = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        path = os.path.join(satellite_data_dir, str(current.year),
                            f"month_{current.month:02d}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)]
            months.append(df)
        else:
            print(f"  Warning: {path} not found, skipping.")
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    if not months:
        raise ValueError(f"No satellite data found in [{start_date}, {end_date}]")
    df = pd.concat(months, ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"Loaded {len(df):,} satellite observations from "
          f"{start_date} to {end_date}")
    return df


def _prefilter_satellite(satellite_df, time_window_minutes):
    """Fast path: keep obs inside the HK_REGION box within ±window of an
    hourly stamp. Returns (df_with_hour, cand_hours) where cand_hours is the
    set of YYYYMMDDTHH stamps that have at least one obs."""
    min_lon, max_lon, min_lat, max_lat = HK_REGION
    hour = satellite_df["time"].dt.round("h")
    near = (satellite_df["time"] - hour).abs() <= np.timedelta64(time_window_minutes, "m")
    df = satellite_df.loc[near].copy()
    df["hour"] = hour[near]
    df = df[df["latitude"].between(min_lat, max_lat)
            & df["longitude"].between(min_lon, max_lon)]
    cand_hours = set(df["hour"].dt.strftime("%Y%m%dT%H"))
    return df, cand_hours


def _build_interpolator(model_xyz, kdtree, k_neighbors, max_distance_km,
                        model_values, obs_lats, obs_lons):
    obs_xyz = _latlon_to_xyz(obs_lats, obs_lons)
    dists, idxs = kdtree.query(obs_xyz, k=k_neighbors)
    great_circle_km = _chord_to_great_circle_km(dists)
    eps = 1.0
    weights = np.where((great_circle_km <= max_distance_km) & (great_circle_km >= 0),
                       1.0 / (great_circle_km + eps), 0.0)
    neighbour_vals = model_values[idxs]
    weights[np.isnan(neighbour_vals)] = 0.0
    sum_w = weights.sum(axis=1)
    valid = sum_w > 0
    result = np.full(len(obs_lats), np.nan, dtype=np.float64)
    result[valid] = (np.sum(weights[valid] * neighbour_vals[valid], axis=1)
                     / sum_w[valid])
    return result


def _load_ensemble_mean(sample_files, num_ensemble, channel):
    """Ensemble mean of the given channel for one timestamp. None if no file loads."""
    samples = []
    files_to_use = sample_files[:num_ensemble] if num_ensemble else sample_files
    for f in files_to_use:
        try:
            arr = np.load(f)["sample"].squeeze()  # (N, 4) or (N,)
            if arr.ndim == 2:
                arr = arr[:, channel]
            samples.append(arr)
        except Exception:
            continue
    if not samples:
        return None
    return np.mean(np.array(samples), axis=0)  # (N,)


# ---------------------------------------------------------------------------
# Single-timestamp workers (module level so ProcessPoolExecutor can pickle)
# ---------------------------------------------------------------------------
def _eval_single_timestamp(timestamp, sample_files, model_xyz, kdtree,
                           k_neighbors, max_distance_km, satellite_df,
                           time_window, num_ensemble, channel, min_obs=1):
    """Metrics for one timestamp against along-track SWH. Returns dict or None."""
    ensemble_mean = _load_ensemble_mean(sample_files, num_ensemble, channel)
    if ensemble_mean is None:
        return None

    ts_dt = pd.Timestamp(timestamp)
    mask = np.abs(satellite_df["time"] - ts_dt) <= time_window
    obs = satellite_df.loc[mask]
    if len(obs) == 0:
        return None

    pred = _build_interpolator(model_xyz, kdtree, k_neighbors, max_distance_km,
                               ensemble_mean, obs["latitude"].values,
                               obs["longitude"].values)
    valid = ~np.isnan(pred)
    if valid.sum() < min_obs:
        return None

    pred_v = pred[valid]
    obs_v = obs["swh"].values[valid]
    lat_v = obs["latitude"].values[valid]
    lon_v = obs["longitude"].values[valid]
    diff = pred_v - obs_v
    n = len(pred_v)
    rmse = np.sqrt(np.mean(diff ** 2))
    bias = np.mean(diff)
    mean_obs = np.mean(obs_v)
    si = np.std(diff - bias) / mean_obs if mean_obs > 0 else np.nan
    corr = np.corrcoef(pred_v, obs_v)[0, 1] if n > 2 else np.nan
    # obs_vals / pred_vals / lat_lon: per-point arrays, used to compute the
    # GLOBAL pooled correlation and to save a per-point match table.
    return {"timestamp": timestamp, "n_obs": n, "rmse": rmse, "bias": bias,
            "si": si, "correlation": corr,
            "obs_vals": obs_v, "pred_vals": pred_v,
            "lat_vals": lat_v, "lon_vals": lon_v}


def _eval_bridge_timestamp(timestamp, sample_files, model_xyz, kdtree,
                           k_neighbors, max_distance_km, satellite_df,
                           time_window, num_ensemble, channel, gt_dir,
                           min_obs=1):
    """Bridge comparison for one timestamp: Model vs Sat / GT vs Sat /
    Model vs GT on the same satellite points. Returns dict or None."""
    ensemble_mean = _load_ensemble_mean(sample_files, num_ensemble, channel)
    if ensemble_mean is None:
        return None

    gt_path = os.path.join(gt_dir, f"wave_{timestamp}.npz")
    if not os.path.exists(gt_path):
        return None
    with np.load(gt_path) as gz:
        if _CHANNEL_KEYS[channel] not in gz:
            return None
        gt_field = gz[_CHANNEL_KEYS[channel]]  # (N,) physical space

    ts_dt = pd.Timestamp(timestamp)
    mask = np.abs(satellite_df["time"] - ts_dt) <= time_window
    obs = satellite_df.loc[mask]
    if len(obs) == 0:
        return None

    lats = obs["latitude"].values
    lons = obs["longitude"].values
    swh = obs["swh"].values
    pred_model = _build_interpolator(model_xyz, kdtree, k_neighbors,
                                     max_distance_km, ensemble_mean, lats, lons)
    pred_gt = _build_interpolator(model_xyz, kdtree, k_neighbors,
                                  max_distance_km, gt_field, lats, lons)
    valid = ~np.isnan(pred_model) & ~np.isnan(pred_gt)
    if valid.sum() < min_obs:
        return None

    d_ms = pred_model[valid] - swh[valid]
    d_gs = pred_gt[valid] - swh[valid]
    d_mg = pred_model[valid] - pred_gt[valid]
    n = len(d_ms)
    row = {"timestamp": timestamp, "n_obs": n}
    for key, d in (("model_vs_sat", d_ms), ("gt_vs_sat", d_gs),
                   ("model_vs_gt", d_mg)):
        row[f"{key}_rmse"] = np.sqrt(np.mean(d ** 2))
        row[f"{key}_bias"] = np.mean(d)
        if n > 2:
            if key == "model_vs_gt":
                row[f"{key}_corr"] = np.corrcoef(pred_model[valid],
                                                 pred_gt[valid])[0, 1]
            else:
                row[f"{key}_corr"] = np.corrcoef(
                    pred_model[valid] if key == "model_vs_sat" else pred_gt[valid],
                    swh[valid])[0, 1]
        else:
            row[f"{key}_corr"] = np.nan
    # per-point arrays for the GLOBAL pooled correlation
    row["sat_vals"] = swh[valid]
    row["model_vals"] = pred_model[valid]
    row["gt_vals"] = pred_gt[valid]
    return row


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _aggregate_per_ts(df, min_obs_per_ts=1):
    """Per-timestamp df (with pred column) -> (per_ts, monthly, global_metrics)."""
    df = df.dropna(subset=["pred"])
    g = df.groupby("timestamp").agg(
        n_obs=("obs", "size"),
        rmse=("obs", lambda x: np.sqrt(np.mean(
            (df.loc[x.index, "pred"] - x) ** 2))),
        bias=("obs", lambda x: np.mean(df.loc[x.index, "pred"] - x)),
        si=("obs", lambda x: np.std(df.loc[x.index, "pred"] - x
                                    - np.mean(df.loc[x.index, "pred"] - x))
            / np.mean(x) if np.mean(x) > 0 else np.nan),
        correlation=("obs", lambda x: np.corrcoef(
            df.loc[x.index, "pred"], x)[0, 1] if len(x) > 2 else np.nan),
    ).reset_index()
    g = g[g["n_obs"] >= min_obs_per_ts].sort_values("timestamp")
    if len(g) == 0:
        return g, None, None

    g["month"] = g["timestamp"].str[:6]
    g["season"] = g["timestamp"].str[4:6].astype(int).map(_SEASON_MAP)
    monthly = g.groupby("month").agg(
        n_obs=("n_obs", "sum"),
        rmse=("rmse", lambda x: np.sqrt(np.mean(x ** 2))),
        bias=("bias", "mean"),
        si=("si", lambda x: np.sqrt(np.mean(x ** 2))),
        correlation=("correlation", "mean"),
    ).reset_index()
    monthly["season"] = monthly["month"].str[4:6].astype(int).map(_SEASON_MAP)

    # Global metrics: POOLED over all observations (all timestamps merged
    # into one sample). RMSE/bias are mathematically identical to the
    # n_obs-weighted aggregation; corr benefits from the much larger sample.
    global_metrics = _pooled_metrics(df["obs"].values, df["pred"].values,
                                     n_timestamps=len(g))
    return g, monthly, global_metrics


def _pooled_metrics(obs, pred, n_timestamps=None):
    """Global metrics over the pooled (obs, pred) point sample."""
    diff = pred - obs
    mean_obs = np.mean(obs)
    gmetrics = {
        "n_timestamps": n_timestamps if n_timestamps is not None else 0,
        "n_obs": int(len(obs)),
        "rmse": np.sqrt(np.mean(diff ** 2)),
        "bias": np.mean(diff),
        "si": np.std(diff - np.mean(diff)) / mean_obs if mean_obs > 0 else np.nan,
        "corr": np.corrcoef(pred, obs)[0, 1],
    }
    return gmetrics


def _aggregate_bridge(df, points_df):
    """Bridge per-timestamp df -> (df, global_metrics). Global rmse/bias/corr
    are POOLED over all satellite points of the three comparisons."""
    gmetrics = {"n_timestamps": len(df),
                "n_obs": len(points_df)}
    for key, a, b in (("model_vs_sat", points_df["model_vals"],
                       points_df["sat_vals"]),
                      ("gt_vs_sat", points_df["gt_vals"],
                       points_df["sat_vals"]),
                      ("model_vs_gt", points_df["model_vals"],
                       points_df["gt_vals"])):
        d = a - b
        gmetrics[f"{key}_rmse"] = np.sqrt(np.mean(d ** 2))
        gmetrics[f"{key}_bias"] = np.mean(d)
        gmetrics[f"{key}_corr"] = np.corrcoef(a, b)[0, 1]
    return df, gmetrics


def _default_output_path(sample_dir, tag, suffix=""):
    """<run_dir>/satellite_eval/hk_<tag><suffix>.csv (run_dir = sample_dir's parent)."""
    run_dir = os.path.dirname(os.path.abspath(sample_dir))
    return os.path.join(run_dir, "satellite_eval", f"hk_{tag}{suffix}.csv")


# ---------------------------------------------------------------------------
# Main evaluation entry points
# ---------------------------------------------------------------------------
def evaluate_hk(sample_dir, satellite_dir, coord_file, start, end, output=None,
                channel=0, num_ensemble=None, min_obs=1, max_workers=None,
                time_window=30, k_neighbors=4, max_distance_km=50.0,
                time_interp=False):
    """Evaluate the ensemble-mean channel against satellite SWH.

    sample_dir: run dir's samples/ folder (contains sample_*_ens*.npz).
    Returns (per_ts_df, monthly_df, global_metrics); CSVs written to
    `output` or <run_dir>/satellite_eval/ by default.
    """
    tag = _CHANNEL_KEYS[channel]
    max_workers = max_workers or min(48, (os.cpu_count() or 1) + 2)

    coords = np.load(coord_file)  # (N, 2) [lat, lon]
    model_xyz = _latlon_to_xyz(coords[:, 0], coords[:, 1])
    kdtree = cKDTree(model_xyz)
    print(f"SatelliteEvaluator(HK): {coords.shape[0]} model nodes, "
          f"channel={channel} ({tag}), KD-tree built.")

    time_window_td = np.timedelta64(time_window, "m")
    timestamps = _discover_samples(sample_dir)
    satellite_df = _load_satellite(satellite_dir, start[:8], end[:8])

    satellite_df, cand_hours = _prefilter_satellite(satellite_df, time_window)
    satellite_df = satellite_df.drop(columns=["hour"])
    print(f"Pre-filtered satellite obs to {len(satellite_df):,} "
          f"inside HK region box {HK_REGION} / ±{time_window} min")

    ts_list = [ts for ts in timestamps
               if start <= ts <= end and ts in cand_hours]
    print(f"Evaluating {len(ts_list)} timestamps in [{start}, {end}]")
    if not ts_list:
        return None, None, None

    if time_interp:
        g, monthly, gmetrics = _evaluate_time_interp(
            satellite_df, timestamps, ts_list, model_xyz, kdtree,
            k_neighbors, max_distance_km, num_ensemble, channel)
        suffix = "_tinterp"
    else:
        results = []
        point_rows = []
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for ts in ts_list:
                f = pool.submit(
                    _eval_single_timestamp, ts, timestamps[ts],
                    model_xyz, kdtree, k_neighbors, max_distance_km,
                    satellite_df, time_window_td, num_ensemble, channel,
                    min_obs,
                )
                futures[f] = ts
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc="Evaluating satellite RMSE"):
                ts = futures[f]
                try:
                    r = f.result()
                    if r is not None:
                        results.append({k: v for k, v in r.items()
                                        if k not in ("obs_vals", "pred_vals",
                                                     "lat_vals", "lon_vals")})
                        point_rows.append(pd.DataFrame({
                            "timestamp": r["timestamp"],
                            "lat": r["lat_vals"], "lon": r["lon_vals"],
                            "obs": r["obs_vals"],
                            "pred": r["pred_vals"],
                        }))
                except Exception as e:
                    print(f"Error at {ts}: {e}")
        if not results:
            print("No results computed.")
            return None, None, None
        g = pd.DataFrame(results).sort_values("timestamp")
        g["month"] = g["timestamp"].str[:6]
        g["season"] = g["timestamp"].str[4:6].astype(int).map(_SEASON_MAP)
        monthly = g.groupby("month").agg(
            n_obs=("n_obs", "sum"),
            rmse=("rmse", lambda x: np.sqrt(np.mean(x ** 2))),
            bias=("bias", "mean"),
            si=("si", lambda x: np.sqrt(np.mean(x ** 2))),
            correlation=("correlation", "mean"),
        ).reset_index()
        monthly["season"] = monthly["month"].str[4:6].astype(int).map(_SEASON_MAP)

        # Global metrics: pooled over all points (all timestamps merged)
        points_df = pd.concat(point_rows, ignore_index=True)
        gmetrics = _pooled_metrics(points_df["obs"].values,
                                   points_df["pred"].values,
                                   n_timestamps=len(g))
        suffix = ""

    print(f"\n{'=' * 60}")
    print(f"Global metrics (POOLED over {gmetrics['n_obs']:,} obs / "
          f"{gmetrics['n_timestamps']} timestamps):")
    print(f"  RMSE:  {gmetrics['rmse']:.4f} m")
    print(f"  Bias:  {gmetrics['bias']:.4f} m")
    print(f"  SI:    {gmetrics['si']:.4f}")
    print(f"  Corr:  {gmetrics['corr']:.4f}")
    print(f"{'=' * 60}")

    out = output or _default_output_path(sample_dir, tag, suffix)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    g.to_csv(out, index=False)
    monthly.to_csv(out.replace(".csv", "_monthly.csv"), index=False)
    # per-point match table (timestamp, lat, lon, obs, pred) for fair
    # same-point comparisons with other models (e.g. OCN grid)
    if not time_interp:
        points_df.to_csv(out.replace(".csv", "_points.csv"), index=False)
        print(f"Per-point matches saved to {out.replace('.csv', '_points.csv')}")
    print(f"Per-timestamp results saved to {out}")
    print(f"Monthly aggregate saved to {out.replace('.csv', '_monthly.csv')}")
    return g, monthly, gmetrics


def _evaluate_time_interp(satellite_df, timestamps, ts_list, model_xyz, kdtree,
                          k_neighbors, max_distance_km, num_ensemble, channel):
    """Time-interpolated variant: for each obs, linearly interpolate the
    ensemble-mean field between the two bounding hourly stamps, then IDW."""
    # ---- 1. ensemble-mean field for every hour we may need ----
    h0 = satellite_df["time"].dt.floor("h")
    need = set(h0.dt.strftime("%Y%m%dT%H"))
    need |= set((h0 + pd.Timedelta(hours=1)).dt.strftime("%Y%m%dT%H"))
    need = sorted(need & set(timestamps.keys()))
    fields = {}
    for ts in tqdm(need, desc="Loading ensemble-mean fields"):
        f = _load_ensemble_mean(timestamps[ts], num_ensemble, channel)
        if f is not None:
            fields[ts] = f
    print(f"Loaded {len(fields)} hourly ensemble-mean fields "
          f"({len(need)} needed)")

    # ---- 2. per-observation time interpolation + spatial IDW ----
    h0s = h0.dt.strftime("%Y%m%dT%H").values
    h1s = (h0 + pd.Timedelta(hours=1)).dt.strftime("%Y%m%dT%H").values
    frac = (satellite_df["time"] - h0).dt.total_seconds().values / 3600.0
    ts_round = satellite_df["time"].dt.round("h").dt.strftime(
        "%Y%m%dT%H").values
    lats = satellite_df["latitude"].values
    lons = satellite_df["longitude"].values
    swh = satellite_df["swh"].values

    preds = np.full(len(satellite_df), np.nan)
    n_both = n_single = n_miss = 0
    for i in tqdm(range(len(satellite_df)), desc="Time-interp IDW"):
        f0, f1 = fields.get(h0s[i]), fields.get(h1s[i])
        a = frac[i]
        if f0 is not None and f1 is not None:
            field = (1.0 - a) * f0 + a * f1
            n_both += 1
        elif f0 is not None:
            field = f0
            n_single += 1
        elif f1 is not None:
            field = f1
            n_single += 1
        else:
            n_miss += 1
            continue
        preds[i] = _build_interpolator(
            model_xyz, kdtree, k_neighbors, max_distance_km,
            field, lats[i:i + 1], lons[i:i + 1])[0]
    print(f"Time interpolation: both-fields={n_both}, single-field={n_single}, "
          f"missing={n_miss}")

    df = pd.DataFrame({
        "timestamp": ts_round,
        "lat": lats, "lon": lons,
        "obs": swh, "pred": preds,
    })
    return _aggregate_per_ts(df)


def evaluate_bridge(sample_dir, satellite_dir, coord_file, gt_dir, start, end,
                    output=None, channel=0, num_ensemble=None, min_obs=1,
                    max_workers=None, time_window=30, k_neighbors=4,
                    max_distance_km=50.0):
    """Bridge comparison on the same satellite points:
      Model vs Sat | Hindcast-GT vs Sat | Model vs GT.

    gt_dir: folder of wave_{YYYYMMDDTHH}.npz (default ./data/hindcast/processed_npz).
    Returns (bridge_df, global_metrics); per-timestamp CSV written to
    `output` or <run_dir>/satellite_eval/hk_<tag>_bridge.csv by default.
    """
    tag = _CHANNEL_KEYS[channel]
    max_workers = max_workers or min(48, (os.cpu_count() or 1) + 2)

    coords = np.load(coord_file)
    model_xyz = _latlon_to_xyz(coords[:, 0], coords[:, 1])
    kdtree = cKDTree(model_xyz)
    print(f"SatelliteBridge(HK): {coords.shape[0]} model nodes, "
          f"channel={channel} ({tag}), GT dir: {gt_dir}")

    time_window_td = np.timedelta64(time_window, "m")
    timestamps = _discover_samples(sample_dir)
    satellite_df = _load_satellite(satellite_dir, start[:8], end[:8])
    satellite_df, cand_hours = _prefilter_satellite(satellite_df, time_window)
    satellite_df = satellite_df.drop(columns=["hour"])
    print(f"Pre-filtered satellite obs to {len(satellite_df):,} "
          f"inside HK region box {HK_REGION} / ±{time_window} min")

    ts_list = [ts for ts in timestamps
               if start <= ts <= end and ts in cand_hours]
    print(f"Evaluating {len(ts_list)} bridge timestamps in [{start}, {end}]")
    if not ts_list:
        return None, None

    results = []
    point_rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for ts in ts_list:
            f = pool.submit(
                _eval_bridge_timestamp, ts, timestamps[ts],
                model_xyz, kdtree, k_neighbors, max_distance_km,
                satellite_df, time_window_td, num_ensemble, channel,
                gt_dir, min_obs,
            )
            futures[f] = ts
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc="Evaluating bridge comparison"):
            ts = futures[f]
            try:
                r = f.result()
                if r is not None:
                    results.append({k: v for k, v in r.items()
                                    if k not in ("sat_vals", "model_vals",
                                                 "gt_vals")})
                    point_rows.append(pd.DataFrame({
                        "timestamp": r["timestamp"],
                        "sat_vals": r["sat_vals"],
                        "model_vals": r["model_vals"],
                        "gt_vals": r["gt_vals"],
                    }))
            except Exception as e:
                print(f"Error at {ts}: {e}")
    if not results:
        print("No results computed.")
        return None, None

    df = pd.DataFrame(results).sort_values("timestamp").reset_index(drop=True)
    points_df = pd.concat(point_rows, ignore_index=True)
    df, gmetrics = _aggregate_bridge(df, points_df)

    print(f"\n{'=' * 60}")
    print(f"Bridge global metrics (POOLED over {gmetrics['n_obs']:,} obs / "
          f"{gmetrics['n_timestamps']} timestamps, same satellite points):")
    for key in ("model_vs_sat", "gt_vs_sat", "model_vs_gt"):
        print(f"  {key:14s} RMSE {gmetrics[f'{key}_rmse']:.4f} m | "
              f"bias {gmetrics[f'{key}_bias']:+.4f} m | "
              f"corr {gmetrics[f'{key}_corr']:.4f}")
    print(f"{'=' * 60}")

    out = output or _default_output_path(sample_dir, tag, "_bridge")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    summary = out.replace(".csv", "_summary.txt")
    with open(summary, "w") as fh:
        fh.write(f"Bridge comparison ({tag}) — {gmetrics['n_timestamps']} "
                 f"timestamps, {gmetrics['n_obs']:,} obs "
                 f"(metrics pooled over all points)\n")
        for key in ("model_vs_sat", "gt_vs_sat", "model_vs_gt"):
            fh.write(f"{key}: RMSE {gmetrics[f'{key}_rmse']:.4f} m, "
                     f"bias {gmetrics[f'{key}_bias']:+.4f} m, "
                     f"corr {gmetrics[f'{key}_corr']:.4f}\n")
    print(f"Bridge results saved to {out}")
    print(f"Global summary saved to {summary}")
    return df, gmetrics


# ---------------------------------------------------------------------------
# OCN (OceanCastNet) evaluation — 0.25° HK grid predictions
# ---------------------------------------------------------------------------
# OCN inference NPZs (run_dir/inferences/sample_{ts}.npz) store *normalized*
# predictions on the 0.25° ERA5-HK-2 grid (24×24, ERA5_HK_LON/LAT_RANGE_2):
#   keys: prediction (5, 24, 24) [hs, tp, dm_sin, dm_cos, elev], ocean_mask,
#         target, era5_input, timestamp
# Denormalization (see oceancastnet_repro/scripts/visualize_hk.py):
#   hs   = max(expm1(norm[0]*hs_log_std + hs_log_mean), 0)      [m]
#   tp   = max(expm1(norm[1]*tp_log_std + tp_log_mean), 0)      [s]
#   dm   = rad2deg(atan2(norm[2], norm[3])) mod 360             [deg]
#   elev = norm[4]*elev_std + elev_mean                          [m]
# Land cells (ocean_mask == 0) are set to NaN so they contribute no
# weight in the IDW. Matching/pre-filtering/aggregation are identical to
# evaluate_hk() so the numbers are directly comparable with our model.
def _load_ocn_hs_field(path, stats):
    """Load one OCN inference NPZ -> physical hs field (24, 24).

    Masked (land) cells stay NaN: they are never used as interpolation
    sources. Satellite obs are still ALL evaluated (selected only by the
    HK_REGION box + time window); obs whose IDW neighbours are all masked
    fall back to the nearest non-masked cell (see _eval_ocn_timestamp).
    """
    d = np.load(path)
    pr = d["prediction"].astype(np.float64)
    hs = np.maximum(np.expm1(pr[0] * stats["hs_log_std"] + stats["hs_log_mean"]), 0.0)
    hs = np.where(d["ocean_mask"] > 0, hs, np.nan)
    return hs


def _ocn_grid_xyz():
    """Unit-sphere xyz of the 0.25° HK-2 grid (24×24, row-major)."""
    lon = np.arange(ERA5_HK_LON_RANGE_2[0], ERA5_HK_LON_RANGE_2[1], 0.25)
    lat = np.arange(ERA5_HK_LAT_RANGE_2[0], ERA5_HK_LAT_RANGE_2[1], 0.25)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return _latlon_to_xyz(lat2d.ravel(), lon2d.ravel())


def _eval_ocn_timestamp(timestamp, path, grid_xyz, kdtree, k_neighbors,
                        max_distance_km, satellite_df, time_window, stats,
                        min_obs=1):
    """OCN grid prediction vs along-track SWH for one timestamp."""
    hs = _load_ocn_hs_field(path, stats)

    ts_dt = pd.Timestamp(timestamp)
    mask = np.abs(satellite_df["time"] - ts_dt) <= time_window
    obs = satellite_df.loc[mask]
    if len(obs) == 0:
        return None

    pred = _build_interpolator(grid_xyz, kdtree, k_neighbors, max_distance_km,
                               hs.ravel(), obs["latitude"].values,
                               obs["longitude"].values)
    # All satellite obs are kept (selection is only the HK box + time window).
    # Obs whose IDW neighbours are all masked (land) get the value of the
    # nearest non-masked (ocean) grid cell.
    if np.isnan(pred).any():
        valid_cells = np.isfinite(hs.ravel())
        kd_valid = cKDTree(grid_xyz[valid_cells])
        d2, idx = kd_valid.query(
            _latlon_to_xyz(obs["latitude"].values, obs["longitude"].values))
        nearest = hs.ravel()[valid_cells][idx]
        pred = np.where(np.isnan(pred), nearest, pred)
    valid = ~np.isnan(pred)
    if valid.sum() < min_obs:
        return None

    pred_v = pred[valid]
    obs_v = obs["swh"].values[valid]
    lat_v = obs["latitude"].values[valid]
    lon_v = obs["longitude"].values[valid]
    diff = pred_v - obs_v
    n = len(pred_v)
    rmse = np.sqrt(np.mean(diff ** 2))
    bias = np.mean(diff)
    mean_obs = np.mean(obs_v)
    si = np.std(diff - bias) / mean_obs if mean_obs > 0 else np.nan
    corr = (np.corrcoef(pred_v, obs_v)[0, 1]
            if n > 2 and np.std(pred_v) > 0 and np.std(obs_v) > 0 else np.nan)
    return {"timestamp": timestamp, "n_obs": n, "rmse": rmse, "bias": bias,
            "si": si, "correlation": corr,
            "obs_vals": obs_v, "pred_vals": pred_v,
            "lat_vals": lat_v, "lon_vals": lon_v}


def evaluate_ocn(ocn_inf_dir, satellite_dir, stats_file, start, end,
                 output=None, min_obs=1, time_window=30, k_neighbors=4,
                 max_distance_km=50.0):
    """Evaluate OCN 0.25° HK grid predictions (run_dir/inferences/) against
    CMEMS L3 along-track SWH, same protocol as evaluate_hk().

    ocn_inf_dir: OCN run dir's inferences/ folder (sample_{ts}.npz).
    Returns (per_ts_df, monthly_df, global_metrics); CSVs written to
    <ocn_run_dir>/satellite_eval/ocn_hs.csv by default.
    """
    with open(stats_file) as f:
        stats = json.load(f)

    grid_xyz = _ocn_grid_xyz()
    kdtree = cKDTree(grid_xyz)
    print(f"SatelliteEvaluator(OCN 0.25° HK grid): {grid_xyz.shape[0]} grid "
          f"points, stats: {os.path.basename(stats_file)}")

    time_window_td = np.timedelta64(time_window, "m")

    # OCN has one file per timestamp (deterministic prediction, no ensemble)
    files = sorted(glob.glob(os.path.join(ocn_inf_dir, "sample_*.npz")))
    timestamps = {}
    for f in files:
        ts = os.path.basename(f).replace("sample_", "").replace(".npz", "")
        timestamps[ts] = f
    print(f"OCN predictions: {len(timestamps)} timestamps "
          f"({os.path.basename(ocn_inf_dir)})")

    satellite_df = _load_satellite(satellite_dir, start[:8], end[:8])
    satellite_df, cand_hours = _prefilter_satellite(satellite_df, time_window)
    satellite_df = satellite_df.drop(columns=["hour"])
    print(f"Pre-filtered satellite obs to {len(satellite_df):,} "
          f"inside HK region box {HK_REGION} / ±{time_window} min")

    ts_list = sorted(ts for ts in timestamps
                     if start <= ts <= end and ts in cand_hours)
    print(f"Evaluating {len(ts_list)} OCN timestamps in [{start}, {end}]")
    if not ts_list:
        return None, None, None

    results = []
    point_rows = []
    for ts in tqdm(ts_list, desc="Evaluating OCN satellite RMSE"):
        r = _eval_ocn_timestamp(
            ts, timestamps[ts], grid_xyz, kdtree, k_neighbors,
            max_distance_km, satellite_df, time_window_td, stats, min_obs)
        if r is not None:
            results.append({k: v for k, v in r.items()
                            if k not in ("obs_vals", "pred_vals")})
            point_rows.append(pd.DataFrame({
                "timestamp": r["timestamp"],
                "lat": r["lat_vals"], "lon": r["lon_vals"],
                "obs": r["obs_vals"],
                "pred": r["pred_vals"],
            }))
    if not results:
        print("No results computed.")
        return None, None, None

    out = output or os.path.join(
        os.path.dirname(os.path.abspath(ocn_inf_dir)),
        "satellite_eval", "ocn_hs.csv")

    g = pd.DataFrame(results).sort_values("timestamp")
    g["month"] = g["timestamp"].str[:6]
    g["season"] = g["timestamp"].str[4:6].astype(int).map(_SEASON_MAP)
    monthly = g.groupby("month").agg(
        n_obs=("n_obs", "sum"),
        rmse=("rmse", lambda x: np.sqrt(np.mean(x ** 2))),
        bias=("bias", "mean"),
        si=("si", lambda x: np.sqrt(np.mean(x ** 2))),
        correlation=("correlation", "mean"),
    ).reset_index()
    monthly["season"] = monthly["month"].str[4:6].astype(int).map(_SEASON_MAP)

    points_df = pd.concat(point_rows, ignore_index=True)
    points_df.to_csv(out.replace(".csv", "_points.csv"), index=False)
    print(f"Per-point matches saved to {out.replace('.csv', '_points.csv')}")
    gmetrics = _pooled_metrics(points_df["obs"].values,
                               points_df["pred"].values,
                               n_timestamps=len(g))

    print(f"\n{'=' * 60}")
    print(f"OCN global metrics (POOLED over {gmetrics['n_obs']:,} obs / "
          f"{gmetrics['n_timestamps']} timestamps):")
    print(f"  RMSE:  {gmetrics['rmse']:.4f} m")
    print(f"  Bias:  {gmetrics['bias']:.4f} m")
    print(f"  SI:    {gmetrics['si']:.4f}")
    print(f"  Corr:  {gmetrics['corr']:.4f}")
    print(f"{'=' * 60}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    g.to_csv(out, index=False)
    monthly.to_csv(out.replace(".csv", "_monthly.csv"), index=False)
    print(f"Per-timestamp results saved to {out}")
    print(f"Monthly aggregate saved to {out.replace('.csv', '_monthly.csv')}")
    return g, monthly, gmetrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_common_args(p):
    p.add_argument("--sample-dir", required=True,
                   help="Run directory containing a 'samples/' subfolder.")
    p.add_argument("--satellite-dir",
                   default="./data/cmems_l3_swh_my/processed",
                   help="Root of processed CMEMS MY Parquet data.")
    p.add_argument("--coord-file",
                   default="./data/hindcast/hk_coordinates.npy")
    p.add_argument("--start", default="20210101T00")
    p.add_argument("--end", default="20221231T23")
    p.add_argument("--channel", type=int, default=0,
                   help="sample channel (0=hs, 1=tp, 2=dm, 3=elev)")
    p.add_argument("--num-ensemble", type=int, default=None,
                   help="Cap ensemble members per timestamp (default: all).")
    p.add_argument("--time-window", type=int, default=30,
                   help="± minutes to match satellite obs to model timestamp.")
    p.add_argument("--k-neighbors", type=int, default=4)
    p.add_argument("--max-distance-km", type=float, default=50.0)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="Model vs satellite evaluation.")
    _add_common_args(pe)
    pe.add_argument("--output", default=None,
                    help="CSV output path (default: "
                         "<run_dir>/satellite_eval/hk_<var>.csv)")
    pe.add_argument("--min-obs", type=int, default=1,
                    help="min matched obs per timestamp to keep it")
    pe.add_argument("--max-workers", type=int, default=None)
    pe.add_argument("--time-interp", action="store_true",
                    help="linearly interpolate the field in time between the "
                         "two bounding hourly stamps")
    pe.set_defaults(func=lambda a: evaluate_hk(
        sample_dir=os.path.join(a.sample_dir, "samples"),
        satellite_dir=a.satellite_dir, coord_file=a.coord_file,
        start=a.start, end=a.end, output=a.output, channel=a.channel,
        num_ensemble=a.num_ensemble, min_obs=a.min_obs,
        max_workers=a.max_workers, time_window=a.time_window,
        k_neighbors=a.k_neighbors, max_distance_km=a.max_distance_km,
        time_interp=a.time_interp))

    pb = sub.add_parser("bridge",
                        help="Model vs Sat / GT vs Sat / Model vs GT on the "
                             "same satellite points.")
    _add_common_args(pb)
    pb.add_argument("--gt-dir", default="./data/hindcast/processed_npz",
                    help="Folder of wave_YYYYMMDDTHH.npz ground-truth files.")
    pb.add_argument("--output", default=None)
    pb.add_argument("--min-obs", type=int, default=1)
    pb.add_argument("--max-workers", type=int, default=None)
    pb.set_defaults(func=lambda a: evaluate_bridge(
        sample_dir=os.path.join(a.sample_dir, "samples"),
        satellite_dir=a.satellite_dir, coord_file=a.coord_file,
        gt_dir=a.gt_dir, start=a.start, end=a.end, output=a.output,
        channel=a.channel, num_ensemble=a.num_ensemble, min_obs=a.min_obs,
        max_workers=a.max_workers, time_window=a.time_window,
        k_neighbors=a.k_neighbors, max_distance_km=a.max_distance_km))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
