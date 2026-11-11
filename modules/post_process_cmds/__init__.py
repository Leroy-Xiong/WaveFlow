"""
post_process command implementations, grouped by domain:
  - viz:        visualization subcommands
  - rmse:       RMSE evaluation subcommands
  - satellite:  CMEMS satellite evaluation subcommands
  - grid:       global-grid interpolation / evaluation subcommands
  - compare:    model/atmosphere-source comparison subcommands
  - common:     shared helpers (PostProcess factory)
"""
from modules.post_process_cmds.viz import (
    cmd_viz_random,
    cmd_viz_gt,
    cmd_viz_ensemble,
    cmd_viz_trajectory,
    cmd_viz_region,
)
from modules.post_process_cmds.rmse import (
    cmd_rmse_multivar,
    cmd_spatial_rmse,
    cmd_temporal_rmse,
)
from modules.post_process_cmds.compare import (
    cmd_compare_atm_sources,
    cmd_compare_models,
)
from modules.post_process_cmds.satellite import (
    cmd_satellite_eval_hk,
    cmd_satellite_bridge,
    cmd_satellite_viz,
    cmd_satellite_eval_ocn,
)
from modules.post_process_cmds.grid import (
    cmd_interpolate_grid,
    cmd_viz_grid,
    cmd_viz_global_grid,
    cmd_rmse_global_grid,
)
from modules.post_process_cmds.eval import (
    cmd_eval_extreme,
    cmd_plot_pit,
    cmd_plot_error_box,
    cmd_interpolate_grid_global,
)
from modules.post_process_cmds.calibrate import (
    cmd_pit_calib,
)
from modules.post_process_cmds.common import _get_postprocess

__all__ = [
    "cmd_viz_random",
    "cmd_viz_gt",
    "cmd_viz_ensemble",
    "cmd_viz_trajectory",
    "cmd_viz_region",
    "cmd_rmse_multivar",
    "cmd_spatial_rmse",
    "cmd_temporal_rmse",
    "cmd_compare_atm_sources",
    "cmd_compare_models",
    "cmd_satellite_eval_hk",
    "cmd_satellite_bridge",
    "cmd_satellite_viz",
    "cmd_satellite_eval_ocn",
    "cmd_interpolate_grid",
    "cmd_viz_grid",
    "cmd_viz_global_grid",
    "cmd_rmse_global_grid",
    "cmd_eval_extreme",
    "cmd_plot_pit",
    "cmd_plot_error_box",
    "cmd_interpolate_grid_global",
    "cmd_pit_calib",
    "_get_postprocess",
]
