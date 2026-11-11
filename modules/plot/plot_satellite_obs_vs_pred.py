"""
Visualize model ensemble-mean predictions vs satellite along-track SWH
observations for the HK model.

Matches CMEMS L3 SWH obs to HK model timestamps, interpolates the ensemble
mean (hs channel) to obs points via KD-tree IDW, and plots:

  - obs vs pred scatter with 1:1 line + fit + metrics
  - per-timestamp obs/pred time series (hourly)
  - monthly RMSE / bias / obs-count bar summary

Outputs go to <sample_run_dir>/visualization/.

Usage (via post_process.py, recommended):
    python post_process.py satellite-viz \
        --sample-dir <run_dir> --start 20210101T00 --end 20221231T23

Standalone:
    python -m modules.plot.plot_satellite_obs_vs_pred \
        --sample-dir <run_dir>/samples --output-dir <run_dir>/visualization
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.eval_satellite_hk import (  # noqa: E402
    _latlon_to_xyz, _chord_to_great_circle_km, _discover_samples,
    _load_satellite, _load_ensemble_mean, _build_interpolator,
    _prefilter_satellite, HK_REGION,
)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
})


def collect_pairs(sample_dir, satellite_dir, coord_file, start, end, channel=0):
    """Match satellite obs to model timestamps; return DataFrame of
    (timestamp, lat, lon, obs, pred).

    Fast path: pre-filter the satellite table to (a) ±30 min of an hourly
    model timestamp, (b) inside the HK_REGION box used to build the HK
    training data, then only load ensemble samples for the surviving
    overpass timestamps (~500 total).
    """
    coords = np.load(coord_file)
    model_xyz = _latlon_to_xyz(coords[:, 0], coords[:, 1])
    from scipy.spatial import cKDTree
    kdtree = cKDTree(model_xyz)
    k_neighbors, max_dist = 4, 50.0
    time_window = np.timedelta64(30, "m")

    timestamps = _discover_samples(sample_dir)
    ts_set = {ts for ts in timestamps if start <= ts <= end}
    print(f"Total model timestamps in range: {len(ts_set)}")

    # ---- 1+2. pre-filter satellite obs: HK_REGION box + ±30min of hourly ts
    satellite_df = _load_satellite(satellite_dir, start[:8], end[:8])
    df, cand_hours = _prefilter_satellite(satellite_df, 30)
    del satellite_df
    print(f"Obs inside HK region box {HK_REGION} / ±30 min of hourly "
          f"stamps: {len(df):,}")

    # ---- 3. candidate overpass timestamps that exist in the samples ----
    cand = df["hour"].dt.strftime("%Y%m%dT%H")
    cand_ts = sorted(ts_set & set(cand))
    print(f"Overpass timestamps with model samples: {len(cand_ts)}")

    # ---- 4. per-timestamp: ensemble mean -> IDW to obs ----
    rows = []
    for ts in cand_ts:
        obs = df.loc[cand == ts]
        ensemble_mean = _load_ensemble_mean(timestamps[ts], None, channel)
        if ensemble_mean is None:
            continue
        pred = _build_interpolator(
            model_xyz, kdtree, k_neighbors, max_dist,
            ensemble_mean, obs["latitude"].values, obs["longitude"].values,
        )
        valid = ~np.isnan(pred)
        if valid.sum() < 1:
            continue
        for i in np.where(valid)[0]:
            rows.append({
                "timestamp": ts,
                "lat": obs["latitude"].values[i],
                "lon": obs["longitude"].values[i],
                "obs": obs["swh"].values[i],
                "pred": pred[i],
            })

    df_out = pd.DataFrame(rows)
    print(f"Collected {len(df_out)} obs-pred pairs over "
          f"{df_out['timestamp'].nunique() if len(df_out) else 0} timestamps")
    return df_out


def plot_scatter(df, out_path):
    """obs vs pred scatter with 1:1 line, fit and metrics."""
    obs, pred = df["obs"].values, df["pred"].values
    diff = pred - obs
    n = len(obs)
    rmse = np.sqrt(np.mean(diff ** 2))
    bias = np.mean(diff)
    si = np.std(diff - bias) / np.mean(obs)
    corr = np.corrcoef(pred, obs)[0, 1]

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.scatter(obs, pred, s=14, c="#1f77b4", alpha=0.55, edgecolors="none",
               label="Obs vs pred")
    lim = [0, max(np.percentile(obs, 99), np.percentile(pred, 99), 2.5)]
    ax.plot(lim, lim, "--", color="0.45", lw=1.0, label="1:1")
    m, b = np.polyfit(obs, pred, 1)
    xs = np.linspace(*lim, 50)
    ax.plot(xs, m * xs + b, "-", color="#d62728", lw=1.2,
            label=f"Fit (slope={m:.2f})")

    ax.text(0.04, 0.97,
            f"n = {n:,}\n"
            f"RMSE = {rmse:.3f} m\n"
            f"Bias = {bias:+.3f} m\n"
            f"SI = {si:.3f}\n"
            f"Corr = {corr:.3f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", lw=0.8))

    ax.set_xlabel("Satellite SWH obs (m)")
    ax.set_ylabel("Model ensemble mean pred (m)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_timeseries(df, out_path):
    """Per-timestamp mean obs/pred line plot."""
    g = df.groupby("timestamp").agg(
        obs_mean=("obs", "mean"),
        pred_mean=("pred", "mean"),
        n_obs=("obs", "size"),
        obs_std=("obs", "std"),
        pred_std=("pred", "std"),
    ).reset_index()
    g = g.sort_values("timestamp")
    t = pd.to_datetime(g["timestamp"], format="%Y%m%dT%H")

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(t, g["obs_mean"], "-o", ms=3.5, lw=1.0, c="#1f77b4",
            label="Satellite obs (mean)")
    ax.plot(t, g["pred_mean"], "-x", ms=3.5, lw=1.0, c="#d62728",
            label="Model ensemble mean (mean)")
    ax.set_xlabel("Time")
    ax.set_ylabel("SWH (m)")
    ax.set_title(f"Obs vs pred per overpass  (n={g['n_obs'].sum():,} obs, "
                 f"{len(g)} overpasses)", fontsize=11)
    ax.legend(fontsize=9, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_monthly(df, out_path):
    """Monthly RMSE / bias / obs count summary."""
    df = df.copy()
    df["month"] = df["timestamp"].str[:6]
    g = df.groupby("month").agg(
        n_obs=("obs", "size"),
        rmse=("obs", lambda x: np.sqrt(np.mean(
            (df.loc[x.index, "pred"] - x) ** 2))),
        bias=("obs", lambda x: np.mean(df.loc[x.index, "pred"] - x)),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True,
                                   height_ratios=[3, 1])
    x = np.arange(len(g))
    ax1.bar(x - 0.18, g["rmse"], width=0.36, color="#1f77b4",
            label="RMSE")
    ax1.bar(x + 0.18, g["bias"], width=0.36, color="#d62728",
            label="Bias")
    ax1.axhline(0, color="0.5", lw=0.7)
    ax1.set_ylabel("Error (m)")
    ax1.legend(fontsize=9, frameon=False)
    ax1.set_xticks(x)
    ax1.set_xticklabels(g["month"], rotation=45, ha="right", fontsize=8)

    ax2.bar(x, g["n_obs"], width=0.6, color="0.6")
    ax2.set_ylabel("n obs")
    ax2.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_satellite_obs_vs_pred(sample_dir, satellite_dir, coord_file,
                               start, end, channel=0, output_dir=None):
    """Collect matched pairs and render the three plots. Returns output paths.

    sample_dir: run dir's samples/ folder. output_dir defaults to
    <run_dir>/visualization/ (run_dir = sample_dir's parent).
    """
    if output_dir is None:
        run_dir = os.path.dirname(os.path.abspath(sample_dir))
        output_dir = os.path.join(run_dir, "visualization")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    df = collect_pairs(sample_dir, satellite_dir, coord_file, start, end,
                       channel)
    if len(df) == 0:
        print("No matched obs-pred pairs; nothing to plot.")
        return []

    tag = ["hs", "tp", "dm", "elev"][channel]
    outs = [
        os.path.join(output_dir, f"satellite_obs_vs_pred_{tag}_scatter.png"),
        os.path.join(output_dir, f"satellite_obs_vs_pred_{tag}_timeseries.png"),
        os.path.join(output_dir, f"satellite_obs_vs_pred_{tag}_monthly.png"),
    ]
    plot_scatter(df, outs[0])
    plot_timeseries(df, outs[1])
    plot_monthly(df, outs[2])
    return outs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-dir", required=True,
                   help="Run directory containing a 'samples/' subfolder.")
    p.add_argument("--satellite-dir",
                   default="./data/cmems_l3_swh_my/processed")
    p.add_argument("--coord-file",
                   default="./data/hindcast/hk_coordinates.npy")
    p.add_argument("--start", default="20210101T00")
    p.add_argument("--end", default="20221231T23")
    p.add_argument("--channel", type=int, default=0,
                   help="sample channel (0=hs, 1=tp, 2=dm, 3=elev)")
    p.add_argument("--output-dir", default=None,
                   help="visualization output dir (default: "
                        "<sample_run_dir>/visualization)")
    args = p.parse_args()

    plot_satellite_obs_vs_pred(
        sample_dir=os.path.join(args.sample_dir, "samples"),
        satellite_dir=args.satellite_dir,
        coord_file=args.coord_file,
        start=args.start, end=args.end,
        channel=args.channel,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
