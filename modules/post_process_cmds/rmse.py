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

def cmd_rmse_multivar(args):
    pp = _get_postprocess(args.sample_dir, args.region_mode, args.gt_format)
    pp.calculate_multivar_rmse_and_export(
        save_path=os.path.join(
            args.sample_dir, f"ensemble_rmse_results_{args.num_ensemble}.csv"
        ),
        num_ensemble_samples=args.num_ensemble,
        variables=args.variables,
    )


def cmd_spatial_rmse(args):
    pp = _get_postprocess(args.sample_dir, args.region_mode, args.gt_format)
    rmse_vis_dir = os.path.join(args.sample_dir, "visualization", "rmse_distribution")
    pp.visualize_spatial_rmse(
        save_path=os.path.join(rmse_vis_dir, f"spatial_rmse_{args.variable}.jpg"),
        variable=args.variable,
        num_ensemble_samples=args.num_ensemble,
    )


def cmd_temporal_rmse(args):
    pp = _get_postprocess(args.sample_dir, args.region_mode, args.gt_format)
    rmse_vis_dir = os.path.join(args.sample_dir, "visualization", "rmse_distribution")
    rmse_csv = args.rmse_csv or os.path.join(
        args.sample_dir, f"ensemble_rmse_results_{args.num_ensemble}.csv"
    )
    pp.visualize_temporal_rmse(
        rmse_csv_path=rmse_csv,
        save_path=os.path.join(rmse_vis_dir, f"temporal_rmse_{args.variable}.jpg"),
        variable=args.variable,
        rmse_column=args.rmse_column,
    )


# ---------------------------------------------------------------------------
# Multi-run / standalone subcommands
# ---------------------------------------------------------------------------





