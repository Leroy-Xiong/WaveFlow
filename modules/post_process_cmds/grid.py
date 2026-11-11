"""Post-processing entry point.

Use subcommands to choose which step to run, e.g.

    python post_process.py rmse-ensemble \\
        --sample-dir output/results/hk/all_var/cond_patch512_allVar_hk2_f8/sample_epoch49_ens72h_20260520_180045 \\
        --num-ensemble 10

    python post_process.py compare-atm-sources --num-samples 4

    python post_process.py hk-hs-timeseries

Run ``python post_process.py -h`` for the full list of subcommands and
``python post_process.py <cmd> -h`` for the options of a specific one.
"""
from __future__ import annotations

import argparse
import os
from functools import lru_cache


# Default sample run used by single-run subcommands.
DEFAULT_SAMPLE_DIR = (
    "output/results/hk/all_var/cond_patch512_allVar_hk2_f8/"
    "sample_epoch49_ens72h_20260520_180045"
)
# Default base directory shared by the multi-run comparison subcommands.
DEFAULT_BASE_DIR = "output/results/hk/all_var/cond_patch512_allVar_hk2_f8"
# Default mapping of label -> per-run sample directory used by the
# qualitative / RMSE-overlay comparisons.
DEFAULT_ATM_SOURCES = {
    "ERA5":    "sample_epoch49_20260204_195847",
    "ENS 6h":  "sample_epoch49_ens6h_20260520_173759",
    "ENS 12h": "sample_epoch49_ens12h_20260520_174528",
    "ENS 24h": "sample_epoch49_ens24h_20260520_170746",
    "ENS 48h": "sample_epoch49_ens48h_20260520_175256",
    "ENS 72h": "sample_epoch49_ens72h_20260520_180045",
}
# Subset used in the HK Hs time-series RMSE overlay (drops ENS 6h).
DEFAULT_HS_TIMESERIES_SOURCES = [
    "ERA5", "ENS 12h", "ENS 24h", "ENS 48h", "ENS 72h",
]


def _interp_init_worker(grid_shape, lon_range, lat_range, node_coord_path, land_mask_path):
    """Per-process initializer: create a GridInterpolator instance."""
    global _worker_interpolator
    from modules.grid_interpolation import GridInterpolator
    _worker_interpolator = GridInterpolator(
        grid_shape=grid_shape,
        lon_range=lon_range,
        lat_range=lat_range,
        node_coord_path=node_coord_path,
        valid_mask_path=land_mask_path,
    )


def _interp_process_one(args_tuple):
    """Process a single file (called from worker pool)."""
    global _worker_interpolator
    fpath, output_dir, method, overwrite = args_tuple
    fname = os.path.basename(fpath)
    out_path = os.path.join(output_dir, fname)
    if os.path.exists(out_path) and not overwrite:
        return 0
    try:
        import numpy as np
        data = np.load(fpath)
        sample = data["sample"]
        data.close()
        node_sample = _worker_interpolator.grid_to_nodes(sample, method=method)
        np.savez_compressed(out_path, sample=node_sample)
        return 1
    except Exception:
        return 0


def cmd_interpolate_grid(args):
    """Interpolate grid-format samples (C, H, W) → irregular node format (N, C).

    Reads .npz files with key 'sample' of shape (4, 75, 107), interpolates
    each to (7788, 4) and writes the result to --output-dir.
    Uses multiprocessing for speed.
    """
    import glob
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    sample_dir = os.path.join(args.sample_dir, "samples")
    output_dir = args.output_dir or os.path.join(args.sample_dir, "samples_nodes")
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(sample_dir, "sample_*.npz")))
    if not files:
        print(f"No sample_*.npz files found in {sample_dir}")
        return

    print(f"Found {len(files)} grid-format sample files in {sample_dir}")

    num_workers = min(args.num_workers or cpu_count(), 32)
    method_val = args.method
    overwrite_val = args.overwrite

    # Pack args for pool
    task_args = [(f, output_dir, method_val, overwrite_val) for f in files]

    print(f"Using {num_workers} workers...")
    with Pool(
        processes=num_workers,
        initializer=_interp_init_worker,
        initargs=(
            tuple(args.grid_shape), tuple(args.lon_range), tuple(args.lat_range),
            args.node_coord, args.land_mask,
        ),
    ) as pool:
        results = list(tqdm(
            pool.imap_unordered(_interp_process_one, task_args, chunksize=100),
            total=len(files),
            desc="Interpolating grid → nodes",
        ))

    done = sum(results)
    print(f"Done — {done}/{len(files)} files written to {output_dir}")
    print(f"Next: python post_process.py rmse-multivar --sample-dir {output_dir} --num-ensemble 10")


def cmd_viz_grid(args):
    """Visualize regular-grid ensemble samples vs ground truth.

    Reads grid-format samples from ``samples_grid/`` and ground truth from
    ``preinterpolated_hk_zoom/``.  Produces 2×3 pcolormesh figures per
    variable per timestamp.
    """
    import glob
    import numpy as np
    from collections import defaultdict
    from modules.plot.plot_grid import plot_grid_ensemble

    sample_dir = os.path.join(args.sample_dir, "samples_grid")
    save_dir = args.save_dir or os.path.join(args.sample_dir, "visualization", "grid_ensemble")
    os.makedirs(save_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(sample_dir, "sample_*.npz")))
    if not files:
        print(f"No sample_*.npz files found in {sample_dir}")
        return

    # Group by timestamp
    by_ts = defaultdict(list)
    for f in files:
        ts = os.path.basename(f).split("_")[1]  # sample_YYYYMMDDTHH_ensNNN.npz
        by_ts[ts].append(f)

    timestamps = sorted(by_ts.keys())
    if args.timestamps:
        chosen = [t for t in args.timestamps if t in by_ts]
        if len(chosen) != len(args.timestamps):
            missing = set(args.timestamps) - set(chosen)
            print(f"Warning: {len(missing)} timestamps not found: {missing}")
    else:
        # Match the exact random logic used by visualize_ensemble_samples_with_ground_truth
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        chosen = sorted(_random.sample(timestamps, min(args.num_samples, len(timestamps))))

    print(f"Found {len(timestamps)} timestamps, plotting {len(chosen)} samples")
    print(f"Timestamps: {', '.join(chosen)}")
    for ts in chosen:
        save_prefix = os.path.join(save_dir, f"grid")
        plot_grid_ensemble(
            ts=ts,
            sample_files=by_ts[ts],
            gt_dir=args.gt_dir,
            save_path=save_prefix,
            variables=args.variables,
            seed=args.seed,
        )
        print(f"  {ts} done ({len(by_ts[ts])} members)")

    print(f"Done — figures saved to {save_dir}")


def cmd_viz_global_grid(args):
    """Visualize global 720×1440 grid ensemble samples (HS) vs ground truth."""
    import glob
    import numpy as np
    from collections import defaultdict
    from modules.plot.plot_grid import plot_global_grid_ensemble

    sample_dir = os.path.join(args.sample_dir, "samples")
    if not os.path.isdir(sample_dir):
        sample_dir = args.sample_dir
    save_dir = args.save_dir or os.path.join(args.sample_dir, "visualization", "global_grid")
    os.makedirs(save_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(sample_dir, "sample_*.npz")))
    if not files:
        print(f"No sample_*.npz files found in {sample_dir}")
        return

    by_ts = defaultdict(list)
    for f in files:
        ts = os.path.basename(f).split("_")[1]
        by_ts[ts].append(f)

    timestamps = sorted(by_ts.keys())
    if args.timestamps:
        chosen = [t for t in args.timestamps if t in by_ts]
        if len(chosen) != len(args.timestamps):
            missing = set(args.timestamps) - set(chosen)
            print(f"Warning: {len(missing)} timestamps not found: {missing}")
    else:
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        chosen = sorted(_random.sample(timestamps, min(args.num_samples, len(timestamps))))

    print(f"Found {len(timestamps)} timestamps, plotting {len(chosen)} samples")
    save_prefix = os.path.join(save_dir, "global_grid")
    for ts in chosen:
        print(f"  {ts}...", end=" ", flush=True)
        plot_global_grid_ensemble(
            ts=ts,
            sample_files=by_ts[ts],
            gt_dir=args.gt_dir,
            save_path=save_prefix,
            seed=args.seed,
        )
        print(f"done ({len(by_ts[ts])} members)")

    print(f"Done — figures saved to {save_dir}")


def cmd_rmse_global_grid(args):
    """Compute grid-native RMSE/Bias/Correlation for global 720×1440 ensemble samples."""
    import glob
    import numpy as np
    import csv
    from collections import defaultdict

    sample_dir = os.path.join(args.sample_dir, "samples")
    if not os.path.isdir(sample_dir):
        sample_dir = args.sample_dir
    gt_dir = args.gt_dir

    files = sorted(glob.glob(os.path.join(sample_dir, "sample_*.npz")))
    if not files:
        print(f"No sample_*.npz files found in {sample_dir}")
        return

    by_ts = defaultdict(list)
    for f in files:
        ts = os.path.basename(f).split("_")[1]
        by_ts[ts].append(f)

    timestamps = sorted(by_ts.keys())
    print(f"Found {len(timestamps)} timestamps, {sum(len(v) for v in by_ts.values())} total files")

    results = []
    all_se = []
    all_err = []
    all_pred = []
    all_gt = []
    skipped = 0

    for i, ts in enumerate(timestamps):
        year = ts[:4]
        gt_path = os.path.join(gt_dir, year, f"global_hindcast_{ts[:8]}_{ts[9:]}.npy")
        if not os.path.exists(gt_path):
            skipped += 1
            continue

        gt = np.load(gt_path)
        gt_hs = gt[0].astype(np.float32) if gt.ndim == 3 else gt.astype(np.float32)

        ens = np.stack([np.load(f)["sample"][0] for f in sorted(by_ts[ts])], axis=0)
        with np.errstate(invalid="ignore"):
            ens_mean = np.nanmean(ens, axis=0)

        ocean = ~np.isnan(gt_hs) & ~np.isnan(ens_mean)
        if ocean.sum() == 0:
            skipped += 1
            continue

        err = ens_mean[ocean] - gt_hs[ocean]
        se = err ** 2
        rmse = float(np.sqrt(se.mean()))
        bias = float(err.mean())
        corr = float(np.corrcoef(ens_mean[ocean], gt_hs[ocean])[0, 1])

        results.append({
            "timestamp": ts,
            "ensemble_size": ens.shape[0],
            "ocean_pixels": int(ocean.sum()),
            "rmse": rmse,
            "bias": bias,
            "correlation": corr,
        })
        all_se.append(se.sum())
        all_err.append(err.sum())
        all_pred.append(ens_mean[ocean])
        all_gt.append(gt_hs[ocean])

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(timestamps)} timestamps...")

    if not results:
        print("No valid timestamps processed.")
        return

    n_total = sum(r["ocean_pixels"] for r in results)
    global_rmse = float(np.sqrt(sum(all_se) / n_total))
    global_bias = float(sum(all_err) / n_total)
    all_pred_cat = np.concatenate(all_pred)
    all_gt_cat = np.concatenate(all_gt)
    global_corr = float(np.corrcoef(all_pred_cat, all_gt_cat)[0, 1])
    per_ts_rmse = np.array([r["rmse"] for r in results])

    print(f"\n{'='*60}")
    print(f"Global Grid RMSE Evaluation (HS)")
    print(f"{'='*60}")
    print(f"  Timestamps evaluated:  {len(results)} (skipped: {skipped})")
    print(f"  Total ocean pixels:    {n_total:,}")
    print(f"  Global RMSE:           {global_rmse:.4f} m")
    print(f"  Global Bias:           {global_bias:.4f} m")
    print(f"  Global Correlation:    {global_corr:.6f}")
    print(f"  Per-timestamp RMSE:    {per_ts_rmse.mean():.4f} ± {per_ts_rmse.std():.4f} m")
    print(f"  Per-timestamp RMSE:    min={per_ts_rmse.min():.4f}, max={per_ts_rmse.max():.4f} m")
    print(f"{'='*60}")

    save_dir = args.save_dir or os.path.join(args.sample_dir, "evaluation")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "global_grid_rmse.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "ensemble_size", "ocean_pixels", "rmse", "bias", "correlation"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-timestamp results saved to: {csv_path}")

    summary_path = os.path.join(save_dir, "global_grid_rmse_summary.json")
    import json
    summary = {
        "num_timestamps": len(results),
        "num_skipped": skipped,
        "total_ocean_pixels": n_total,
        "global_rmse_m": global_rmse,
        "global_bias_m": global_bias,
        "global_correlation": global_corr,
        "per_ts_rmse_mean_m": float(per_ts_rmse.mean()),
        "per_ts_rmse_std_m": float(per_ts_rmse.std()),
        "per_ts_rmse_min_m": float(per_ts_rmse.min()),
        "per_ts_rmse_max_m": float(per_ts_rmse.max()),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


