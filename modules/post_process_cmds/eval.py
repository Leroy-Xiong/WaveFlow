"""Evaluation subcommands (extreme weather, PIT calibration, global grid interp)."""
from __future__ import annotations

import argparse

from modules.postprocess_extreme import run as run_eval_extreme
from modules.plot.plot_pit import run as run_plot_pit
from modules.plot.plot_error_box import run as run_plot_error_box
from modules.preprocess.interpolate_global_grid_to_nodes import (
    cmd_precompute,
    cmd_interpolate,
    cmd_all,
)


def cmd_eval_extreme(args):
    """Extreme-weather (typhoon) evaluation of ensemble samples vs hindcast GT."""
    run_eval_extreme(args)


def cmd_plot_pit(args):
    """PIT (Probability Integral Transform) calibration diagnostics."""
    run_plot_pit(args)


def cmd_plot_error_box(args):
    """Error-vs-GT-value boxplots for ensemble samples vs hindcast GT."""
    run_plot_error_box(args)


def cmd_interpolate_grid_global(args):
    """Interpolate global grid model output (720x1440) to irregular nodes (~652k).

    Wraps modules.preprocess.interpolate_global_grid_to_nodes.
    """
    ns = argparse.Namespace(
        cmd=args.action,
        sample_dir=args.sample_dir,
        output_dir=args.output_dir,
        k=args.k,
        weights_dir=args.weights_dir,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        force=args.force,
    )
    dispatch = {'precompute': cmd_precompute, 'interpolate': cmd_interpolate, 'all': cmd_all}
    dispatch[ns.cmd](ns)
