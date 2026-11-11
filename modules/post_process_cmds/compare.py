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


def cmd_compare_atm_sources(args):
    from modules.plot.compare_atm_sources import compare_ensemble_mean_across_sources
    sample_dirs = {
        label: os.path.join(args.base_dir, run_dir, "samples")
        for label, run_dir in DEFAULT_ATM_SOURCES.items()
    }

    timestamps = None
    filename_prefix_map = None

    if args.timestamps:
        timestamps = list(args.timestamps)
    elif args.top_peaks:
        from datetime import datetime, timedelta
        from modules.plot.compare_era5_ens import (
            pick_top_hs_peaks, default_typhoon_name_lookup,
        )

        def _snap_to_12h(dt):
            """Snap datetime to nearest 00Z or 12Z slot (sample cadence)."""
            base = dt.replace(minute=0, second=0, microsecond=0)
            candidates = [
                base.replace(hour=0),
                base.replace(hour=12),
                (base + timedelta(days=1)).replace(hour=0),
            ]
            return min(candidates, key=lambda c: abs(c - dt))

        picked = pick_top_hs_peaks(
            csv_path=args.peaks_csv,
            num_peaks=args.top_peaks,
            min_separation_hours=args.min_separation_hours,
            hs_column=args.hs_column,
        )
        name_lookup = default_typhoon_name_lookup()
        timestamps = []
        filename_prefix_map = {}
        for t, hs in picked:
            ts = _snap_to_12h(t).strftime("%Y%m%dT%H")
            timestamps.append(ts)
            name = name_lookup.get(ts[:8])
            if name:
                filename_prefix_map[ts] = name
            print(f"  Peak {t} (Hs={hs:.2f}) -> ts={ts}"
                  + (f" [{name}]" if name else ""))

    compare_ensemble_mean_across_sources(
        sample_dirs=sample_dirs,
        gt_data_dir=args.gt_dir,
        save_dir=args.save_dir,
        num_samples=args.num_samples,
        timestamps=timestamps,
        filename_prefix_map=filename_prefix_map,
    )








def cmd_compare_models(args):
    """Comparison figure: irregular (patched) vs regular grid vs node-token GraphTransformer.

    Layout: 3×3 (with node-token) or 2×3 (without).
    """
    import glob
    import numpy as np
    import random as _random
    from collections import defaultdict
    from modules.plot.compare_models import plot_model_comparison

    # ---- gather irregular model samples ----
    irr_sample_dir = os.path.join(args.irregular_dir, "samples")
    irr_files = sorted(glob.glob(os.path.join(irr_sample_dir, "sample_*_ens*.npz")))
    irr_by_ts = defaultdict(list)
    for f in irr_files:
        ts = os.path.basename(f).split("_ens")[0].replace("sample_", "")
        irr_by_ts[ts].append(f)
    print(f"Irregular model: {len(irr_by_ts)} timestamps")

    # ---- gather regular model samples (grid + node) ----
    reg_grid_dir = os.path.join(args.regular_dir, "samples_grid")
    reg_node_dir = os.path.join(args.regular_dir, "samples_nodes")
    reg_grid_files = sorted(glob.glob(os.path.join(reg_grid_dir, "sample_*_ens*.npz")))
    reg_node_files = sorted(glob.glob(os.path.join(reg_node_dir, "sample_*_ens*.npz")))

    reg_grid_by_ts = defaultdict(list)
    for f in reg_grid_files:
        ts = os.path.basename(f).split("_ens")[0].replace("sample_", "")
        reg_grid_by_ts[ts].append(f)
    reg_node_by_ts = defaultdict(list)
    for f in reg_node_files:
        ts = os.path.basename(f).split("_ens")[0].replace("sample_", "")
        reg_node_by_ts[ts].append(f)
    print(f"Regular model: {len(reg_grid_by_ts)} grid timestamps, {len(reg_node_by_ts)} node timestamps")

    # ---- gather node-token model samples (optional) ----
    nt_by_ts = defaultdict(list)
    if args.node_token_dir:
        nt_sample_dir = os.path.join(args.node_token_dir, "samples")
        nt_files = sorted(glob.glob(os.path.join(nt_sample_dir, "sample_*_ens*.npz")))
        for f in nt_files:
            ts = os.path.basename(f).split("_ens")[0].replace("sample_", "")
            nt_by_ts[ts].append(f)
        print(f"Node-token model: {len(nt_by_ts)} timestamps")

    # ---- intersect timestamps ----
    ts_sets = [set(irr_by_ts.keys()), set(reg_grid_by_ts.keys()), set(reg_node_by_ts.keys())]
    if args.node_token_dir:
        ts_sets.append(set(nt_by_ts.keys()))
    common_ts = sorted(set.intersection(*ts_sets))
    if not common_ts:
        print("Error: no common timestamps between model samples!")
        return
    print(f"Common timestamps: {len(common_ts)}")

    # ---- select timestamps ----
    _random.seed(args.seed)
    np.random.seed(args.seed)
    chosen = sorted(_random.sample(common_ts, min(args.num_samples, len(common_ts))))
    print(f"Selected: {chosen}")

    # ---- load node coordinates ----
    coords = np.load("./data/hindcast/hk_coordinates.npy")
    node_lons = coords[:, 1]
    node_lats = coords[:, 0]

    # ---- plot per timestamp ----
    save_dir = args.save_dir or "./output/plot/model_comparison"
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for ts in chosen:
        node_token_files = nt_by_ts.get(ts, []) if args.node_token_dir else []
        plot_model_comparison(
            ts=ts,
            irregular_files=irr_by_ts[ts],
            regular_grid_files=reg_grid_by_ts[ts],
            regular_node_files=reg_node_by_ts[ts],
            node_token_files=node_token_files,
            node_lons=node_lons,
            node_lats=node_lats,
            save_path=os.path.join(save_dir, "compare"),
            variables=args.variables,
            seed=args.seed,
        )
        n_nt = f", {len(node_token_files)} nt" if node_token_files else ""
        print(f"  {ts} done ({len(irr_by_ts[ts])} irr, {len(reg_grid_by_ts[ts])} reg{n_nt} members)")

    print(f"Done — figures saved to {save_dir}")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------
