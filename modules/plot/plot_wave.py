"""
Wave plotting utilities (re-export shim).

The plotting code has been split into focused modules:
  - plot_wave_hk:     Hong Kong region plots (nodes, ERA5 HK)
  - plot_wave_global: Global ocean plots + ocean masking
  - plot_rmse:        RMSE boxplot/bar comparisons
  - plot_cluster:     Patch cluster visualization

This module re-exports everything for backward compatibility with
existing ``from modules.plot.plot_wave import ...`` statements.
"""
from modules.plot.plot_wave_hk import (
    interpolate_wave_data_to_grid,
    plot_variable_distributions,
    plot_wave_data,
    plot_wave_data_interpolated,
    plot_era5_pressure,
    plot_era5_variable_distributions,
    plot_era5_wind,
)
from modules.plot.plot_wave_global import (
    _get_ocean_raster,
    _points_in_global_ocean,
    _mask_data_to_global_ocean,
    _add_global_satellite_with_ocean_mask,
    _add_global_graticule,
    plot_wave_data_global,
    plot_era5_wind_global,
    plot_era5_pressure_global,
)
from modules.plot.plot_rmse import (
    plot_rmse_comparison,
    plot_multi_variable_rmse_comparison,
)
from modules.plot.plot_cluster import (
    plot_cluster_results,
    _plot_cluster_spherical_views,
)

__all__ = [
    "interpolate_wave_data_to_grid",
    "plot_variable_distributions",
    "plot_wave_data",
    "plot_wave_data_interpolated",
    "plot_era5_pressure",
    "plot_era5_variable_distributions",
    "plot_era5_wind",
    "_get_ocean_raster",
    "_points_in_global_ocean",
    "_mask_data_to_global_ocean",
    "_add_global_satellite_with_ocean_mask",
    "_add_global_graticule",
    "plot_wave_data_global",
    "plot_era5_wind_global",
    "plot_era5_pressure_global",
    "plot_rmse_comparison",
    "plot_multi_variable_rmse_comparison",
    "plot_cluster_results",
    "_plot_cluster_spherical_views",
]
