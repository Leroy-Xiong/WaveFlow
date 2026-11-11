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


from modules.post_process_cmds.common import _get_postprocess

def cmd_viz_random(args):
    pp = _get_postprocess(args.sample_dir, args.region_mode, args.gt_format)
    era5_dir = args.era5_dir if args.era5_dir and args.era5_dir.strip() else None
    # Auto-detect ERA5 directory based on region_mode if not explicitly set
    if era5_dir is None:
        era5_dir = "./data/era5/global_npz_files" if args.region_mode == "global" else "./data/era5/nc_files"
    pp.visualize_random_samples(
        num_samples=args.num_samples,
        save_dir=os.path.join(args.sample_dir, "visualization"),
        era5_dir=era5_dir,
    )


def cmd_viz_gt(args):
    pp = _get_postprocess(args.sample_dir, args.region_mode, args.gt_format)
    pp.visualize_samples_with_ground_truth(
        num_samples=args.num_samples,
        save_dir=os.path.join(args.sample_dir, "visualization"),
    )


def cmd_viz_ensemble(args):
    pp = _get_postprocess(args.sample_dir, args.region_mode, args.gt_format)
    era5_dir = args.era5_dir if args.era5_dir and args.era5_dir.strip() else None
    # Auto-detect ERA5 directory based on region_mode if not explicitly set
    if era5_dir is None:
        era5_dir = "./data/era5/global_npz_files" if args.region_mode == "global" else "./data/era5/nc_files"
    pp.visualize_ensemble_samples_with_ground_truth(
        num_samples=args.num_samples,
        save_dir=os.path.join(args.sample_dir, "visualization/random_samples"),
        era5_dir=era5_dir,
    )


def cmd_viz_trajectory(args):
    pp = _get_postprocess(args.sample_dir, args.region_mode, args.gt_format)
    pp.visualize_trajectory(
        trajectory_dir=os.path.join(args.sample_dir, "trajectory"),
        num_samples=args.num_samples,
        save_dir=os.path.join(args.sample_dir, "visualization/trajectory"),
        seed=args.seed,
    )




def cmd_viz_region(args):
    pp = _get_postprocess(args.sample_dir, args.region_mode, args.gt_format)
    region = [args.lon_range[0], args.lon_range[1], args.lat_range[0], args.lat_range[1]]
    era5_dir = args.era5_dir if args.era5_dir and args.era5_dir.strip() else None
    # Auto-detect ERA5 directory based on region_mode if not explicitly set
    if era5_dir is None:
        era5_dir = "./data/era5/global_npz_files" if args.region_mode == "global" else "./data/era5/nc_files"
    pp.visualize_region(
        timestamps=args.timestamps,
        region=region,
        save_dir=os.path.join(args.sample_dir, "visualization/region"),
        scatter_size=args.scatter_size,
        show_interpolation=args.interpolation,
        era5_dir=era5_dir,
    )






