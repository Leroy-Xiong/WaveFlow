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


def cmd_satellite_eval_hk(args):
    """HK 4-variable model vs CMEMS MY along-track SWH (modules/eval_satellite_hk)."""
    from modules.eval_satellite_hk import evaluate_hk

    evaluate_hk(
        sample_dir=os.path.join(args.sample_dir, "samples"),
        satellite_dir=args.satellite_dir,
        coord_file=args.coord_file,
        start=args.start, end=args.end,
        output=args.output,
        channel=args.channel,
        num_ensemble=args.num_ensemble,
        min_obs=args.min_obs,
        max_workers=args.max_workers,
        time_window=args.time_window,
        k_neighbors=args.k_neighbors,
        max_distance_km=args.max_distance_km,
        time_interp=args.time_interp,
    )


def cmd_satellite_bridge(args):
    """Bridge comparison: Model vs Sat / Hindcast-GT vs Sat / Model vs GT."""
    from modules.eval_satellite_hk import evaluate_bridge

    evaluate_bridge(
        sample_dir=os.path.join(args.sample_dir, "samples"),
        satellite_dir=args.satellite_dir,
        coord_file=args.coord_file,
        gt_dir=args.gt_dir,
        start=args.start, end=args.end,
        output=args.output,
        channel=args.channel,
        num_ensemble=args.num_ensemble,
        min_obs=args.min_obs,
        max_workers=args.max_workers,
        time_window=args.time_window,
        k_neighbors=args.k_neighbors,
        max_distance_km=args.max_distance_km,
    )


def cmd_satellite_viz(args):
    """HK model vs satellite obs scatter / timeseries / monthly plots."""
    from modules.plot.plot_satellite_obs_vs_pred import plot_satellite_obs_vs_pred

    plot_satellite_obs_vs_pred(
        sample_dir=os.path.join(args.sample_dir, "samples"),
        satellite_dir=args.satellite_dir,
        coord_file=args.coord_file,
        start=args.start, end=args.end,
        channel=args.channel,
        output_dir=args.output_dir,
    )


def cmd_satellite_eval_ocn(args):
    """OCN 0.25° HK grid predictions vs CMEMS MY along-track SWH."""
    from modules.eval_satellite_hk import evaluate_ocn

    evaluate_ocn(
        ocn_inf_dir=os.path.join(args.ocn_run_dir, "inferences"),
        satellite_dir=args.satellite_dir,
        stats_file=args.stats_file,
        start=args.start, end=args.end,
        output=args.output,
        min_obs=args.min_obs,
        time_window=args.time_window,
        k_neighbors=args.k_neighbors,
        max_distance_km=args.max_distance_km,
    )


# Module-level globals used by interpolation worker processes
_interp_config = {}
_worker_interpolator = None


