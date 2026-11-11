"""
Post-processing for generated wave samples (re-export shim).

Implementation has been split into:
  - modules.postprocess_standalone: per-timestamp standalone processing helpers
  - modules.postprocess_class:      PostProcess class (visualization + RMSE)

This module re-exports everything for backward compatibility with
existing ``from modules.postprocess_samples import ...`` statements.
"""
from modules.postprocess_standalone import (
    _process_timestamp_viz_standalone,
    _save_traj_field_figure,
    _detect_traj_variables,
    _process_trajectory_sample_standalone,
    _process_trajectory_ensemble_standalone,
    _process_trajectory_gt_standalone,
    _save_era5_figure_standalone,
    process_single_timestamp_multivar_standalone,
    process_single_timestamp_pernode_se_standalone,
    process_single_timestamp_gt_stats_standalone,
    process_single_timestamp_ensemble_standalone,
)
from modules.postprocess_class import PostProcess

__all__ = [
    "_process_timestamp_viz_standalone",
    "_save_traj_field_figure",
    "_detect_traj_variables",
    "_process_trajectory_sample_standalone",
    "_process_trajectory_ensemble_standalone",
    "_process_trajectory_gt_standalone",
    "_save_era5_figure_standalone",
    "process_single_timestamp_multivar_standalone",
    "process_single_timestamp_pernode_se_standalone",
    "process_single_timestamp_gt_stats_standalone",
    "process_single_timestamp_ensemble_standalone",
    "PostProcess",
]
