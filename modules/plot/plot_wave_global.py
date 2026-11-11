import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
from numpy import ma
import pandas as pd
import seaborn as sns

from modules.utils import WAVE_VARIABLES, ERA5_VARIABLE_NAMES, MAJOR_CITIES
try:
    from modules.plot.satellite_background import add_satellite_background, add_satellite_background_global
except ImportError:
    add_satellite_background = None
    add_satellite_background_global = None

from modules.plot.plot_wave_hk import interpolate_wave_data_to_grid
# Raster resolution (deg) used to rasterize the ocean polygon for grid
# masking - much finer than the 0.25 deg ERA5 grid, so cell-center lookups
# are effectively exact.
_OCEAN_RASTER_RES = 0.05
_ocean_raster = None
# per-grid-signature cache for _points_in_global_ocean
_ocean_contains_cache = {}


def _get_ocean_raster(ocean_geom):
    """Rasterize the ocean MultiPolygon to a uint8 image (255=ocean) at
    ``_OCEAN_RASTER_RES`` resolution covering lon [-180,180], lat [-90,90].

    Rasterization via PIL is orders of magnitude faster than per-point
    point-in-polygon tests against the full geometry (~100k vertices).
    """
    global _ocean_raster
    if _ocean_raster is not None:
        return _ocean_raster
    from PIL import Image, ImageDraw
    res = _OCEAN_RASTER_RES
    nlon = int(round(360.0 / res))
    nlat = int(round(180.0 / res))

    def to_px(ring):
        pts = np.asarray(ring.coords, dtype=float)
        px = (pts[:, 0] + 180.0) / res - 0.5
        py = (90.0 - pts[:, 1]) / res - 0.5
        return list(zip(px.tolist(), py.tolist()))

    img = Image.new('L', (nlon, nlat), 0)
    draw = ImageDraw.Draw(img)
    polys = [ocean_geom] if ocean_geom.geom_type == 'Polygon' \
        else list(ocean_geom.geoms)
    for poly in polys:
        draw.polygon(to_px(poly.exterior), fill=255)
        for interior in poly.interiors:
            draw.polygon(to_px(interior), fill=0)
    _ocean_raster = np.asarray(img)
    return _ocean_raster


def _points_in_global_ocean(lon, lat, ocean_geom):
    """Boolean (nlat, nlon) mask: which regular-grid points fall inside the
    ocean polygon.  Results are cached per grid signature."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    key = (len(lon), len(lat),
           round(float(lon[0]), 6), round(float(lon[-1]), 6),
           round(float(lat[0]), 6), round(float(lat[-1]), 6))
    if key in _ocean_contains_cache:
        return _ocean_contains_cache[key]
    res = _OCEAN_RASTER_RES
    raster = _get_ocean_raster(ocean_geom)
    nlat, nlon = raster.shape
    col = np.clip(((lon + 180.0) / res).astype(int), 0, nlon - 1)
    row = np.clip(((90.0 - lat) / res).astype(int), 0, nlat - 1)
    mask = raster[np.ix_(row, col)] > 0
    _ocean_contains_cache[key] = mask
    return mask


def _mask_data_to_global_ocean(lon, lat, *arrays):
    """Mask regular-grid data to the global wave-node ocean (same ocean
    polygon as the white overlay), so ERA5 fields only appear over the model
    ocean and land keeps the satellite background. Cells outside the ocean
    become NaN.

    Args:
        lon, lat: 1D coordinate arrays of the data grid (-180..180 lon).
        *arrays: 2D fields with shape (len(lat), len(lon)).

    Returns:
        Tuple of masked copies of ``arrays`` (NaN outside the ocean). If the
        ocean geometry cannot be built, the input arrays are returned
        unchanged.
    """
    try:
        from modules.plot.global_ocean_boundary import get_global_ocean_geometry
        ocean_geom, _ = get_global_ocean_geometry()
        mask_2d = _points_in_global_ocean(lon, lat, ocean_geom)
    except Exception as e:
        print(f"Warning: global ocean geometry unavailable, data not masked: {e}")
        return arrays
    return tuple(np.where(mask_2d, arr, np.nan) for arr in arrays)


def _add_global_satellite_with_ocean_mask(ax, use_satellite):
    """Add satellite imagery with ocean masked white (same approach as HK).

    When ``use_satellite`` is True, satellite tiles are drawn and then
    overlaid with the precise ocean polygon (derived from the global wave
    node coordinates) painted white, so land shows satellite detail while
    the ocean stays white (making scatter / contour data clearly visible).
    The ocean boundary is drawn as a thin black coastline.
    """
    if not use_satellite or add_satellite_background_global is None:
        ax.add_feature(cfeature.OCEAN, color='white', alpha=1.0, zorder=3)
        ax.add_feature(cfeature.LAND, color='whitesmoke', alpha=1.0, zorder=4)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.8, zorder=11)
        return
    add_satellite_background_global(ax)
    try:
        from modules.plot.global_ocean_boundary import get_global_ocean_geometry
        ocean_geom, coastline_geom = get_global_ocean_geometry()
        ax.add_geometries([ocean_geom], ccrs.PlateCarree(), facecolor='white',
                          edgecolor='none', zorder=2)
        ax.add_geometries([coastline_geom], ccrs.PlateCarree(), facecolor='none',
                          edgecolor='black', linewidth=0.5, alpha=0.8, zorder=11)
    except Exception as e:
        print(f"Warning: failed to draw global ocean geometry: {e}")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.8, zorder=11)


def _add_global_graticule(ax):
    """Add a lon/lat graticule with degree labels to a global map.

    Grid lines are subtle and sit below the data layers; only the tick
    labels are meant to be read.

    Note: cartopy 0.25's labelled gridliner crashes in the measuring
    draws triggered by ``tight_layout`` / ``bbox_inches='tight'`` (map
    boundary path still uninitialized, all-NaN vertices).  Callers must
    do any tight layout *before* adding the graticule and save without
    ``bbox_inches='tight'``.
    """
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                      alpha=0.4, linestyle='--',
                      xlocs=np.arange(-180, 181, 60),
                      ylocs=np.arange(-60, 61, 30))
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize': 8}
    gl.ylabel_style = {'fontsize': 8}
    # Keep the axes title at its fixed position: matplotlib's
    # label-aware title repositioning would push it above the canvas
    # because the graticule labels stick out past the axes bbox.
    ax._autotitlepos = False
    return gl


def plot_wave_data_global(dataset, variable='hs', output_path=None, projection='Robinson',
                          dpi=200, cmap='turbo', vmin=None, vmax=None,
                          custom_title=None, show_interpolation=False, clip_negative=None,
                          return_interpolated=False, use_satellite_background=False,
                          contour_var=None, contour_levels=14, direction_var=None,
                          figsize=(9, 6), transparent=False, show_colorbar=True):
    """
    Plot wave data on a global map using Cartopy with Robinson or Mollweide projection.
    
    Optimized for global data visualization with proper projection handling,
    similar to plot_era5_wind_global and plot_era5_pressure_global.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Dataset containing wave data with dimensions (lat, lon) or (latitude, longitude)
    variable : str, optional
        Variable name to plot ('hs', 'tp', 'dm', 'elev'), default is 'hs'
    output_path : str, optional
        Path to save the output plot. If None, returns figure without saving
    projection : str, optional
        Map projection to use: 'Robinson', 'Mollweide', or 'PlateCarree' (default: 'Robinson')
    dpi : int, optional
        Resolution of the output figure (default: 200)
    cmap : str, optional
        Colormap name for the plot (default: 'turbo')
    vmin : float, optional
        Minimum value for color scale
    vmax : float, optional
        Maximum value for color scale
    custom_title : str, optional
        Custom title for the plot. If None, uses default title format
    show_interpolation : bool, optional
        If True, adds a second subplot with the original scatter nodes for
        comparison against the interpolated main panel (default: False)
    clip_negative : bool, optional
        If True, clip negative values to 0 for wave height/elevation variables.
        If None, automatically clip for 'hs' and 'elev' variables unless it's a difference plot.
    return_interpolated : bool, optional
        If True, returns the interpolated data along with the figure (default: False)
    contour_var : str, optional
        Name of a second dataset variable (e.g. 'ssh') to overlay as
        contour lines on top of the main panel (default: None)
    contour_levels : int, optional
        Number of contour levels for ``contour_var`` (default: 14)
    direction_var : str, optional
        Name of a wave direction variable in degrees FROM (nautical
        convention, e.g. 'mwd') to overlay as quiver arrows on top of the
        main panel (default: None)
    transparent : bool, optional
        Save with a transparent figure background instead of white. Forces the
        output extension to .png since JPEG cannot store alpha (default: False).
    show_colorbar : bool, optional
        Whether to draw the horizontal colorbar below the map (default: True).

    Returns:
    --------
    matplotlib.figure.Figure or tuple: 
        - If return_interpolated=False: The created figure object
        - If return_interpolated=True: (fig, interpolated_data) where interpolated_data is a dict
          containing 'lon_mesh', 'lat_mesh', 'values_interpolated', or None if interpolation was not performed
    """
    
    # Select projection
    if projection == 'Robinson':
        proj = ccrs.Robinson(central_longitude=0)
    elif projection == 'Mollweide':
        proj = ccrs.Mollweide(central_longitude=0)
    else:
        proj = ccrs.PlateCarree(central_longitude=0)
    
    # Create figure with subplots
    if show_interpolation:
        fig = plt.figure(figsize=(figsize[0] * 2, figsize[1]))
        ax1 = fig.add_subplot(1, 2, 1, projection=proj)
        ax2 = fig.add_subplot(1, 2, 2, projection=proj)
    else:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 1, 1, projection=proj)
        ax2 = None
    
    # Extract time for plot title
    time_str = str(dataset.time.values)[:16] if hasattr(dataset, 'time') and dataset.time is not None else 'Unknown Time'
    
    # Get wave data
    wave_data = dataset[variable].values
    
    # Get lon/lat - for global hindcast data, they are data variables, not coordinates
    # Try data variables first (global hindcast format), then coordinates (ERA5 format)
    if 'lon' in dataset:
        lon = dataset.lon.values
        lat = dataset.lat.values
    elif 'longitude' in dataset:
        lon = dataset.longitude.values
        lat = dataset.latitude.values
    else:
        raise ValueError(f"Could not find lon/lat or longitude/latitude in dataset. "
                        f"Available variables: {list(dataset.data_vars.keys())}, "
                        f"coords: {list(dataset.coords.keys())}")
    
    # Convert longitude from 0-360 to -180-180 if needed
    if np.any(lon > 180):
        lon = np.where(lon > 180, lon - 360, lon)
    
    # Flatten arrays (global hindcast data is unstructured)
    lon_flat = lon.flatten()
    lat_flat = lat.flatten()
    wave_data_flat = wave_data.flatten()
    
    # Get variable properties
    var_properties = {var: {
        'label': WAVE_VARIABLES[var]['plot_label'],
        'title': WAVE_VARIABLES[var]['plot_title']
    } for var in WAVE_VARIABLES}
    
    var_prop = var_properties.get(variable, {'label': f'{variable}', 'title': f'{variable} Distribution'})
    
    # Determine colorbar range
    if vmin is None:
        vmin = np.nanmin(wave_data_flat)
    if vmax is None:
        vmax = np.nanmax(wave_data_flat)
    
    # Count valid nodes
    valid_nodes = np.sum(~np.isnan(wave_data_flat))
    
    # --- Main subplot: data interpolated to a regular grid ---
    lon_mesh, lat_mesh, values_interpolated = interpolate_wave_data_to_grid(
        lon_flat, lat_flat, wave_data_flat, grid_resolution=(4000, 2000),
        max_distance=0.5
    )
    
    # Clip values to reasonable range (no negative values for wave height)
    # For difference plots (e.g., title contains 'Difference'), don't clip
    should_clip = clip_negative
    if should_clip is None:
        # Auto-determine: clip for hs/elev unless it's a difference plot
        is_difference_plot = custom_title is not None and 'Difference' in custom_title
        should_clip = variable in ['hs', 'elev'] and not is_difference_plot
    
    if should_clip:
        values_interpolated = np.clip(values_interpolated, 0, None)
    
    # Mask to the wave-node ocean (same polygon as the ERA5 panels), so
    # land keeps the satellite background.  Cartopy's warped pcolormesh
    # (Robinson / Mollweide) silently drops NaN / masked cells, so the
    # data is rendered as an RGBA image with alpha=0 outside the ocean -
    # cartopy's image warping honours the alpha channel.
    if use_satellite_background:
        (values_interpolated,) = _mask_data_to_global_ocean(
            lon_mesh[0, :], lat_mesh[:, 0], values_interpolated)
    
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = plt.get_cmap(cmap)(norm(values_interpolated))
    rgba[..., 3] = np.where(np.isnan(values_interpolated), 0.0, 1.0)
    im1 = ax1.imshow(rgba, extent=[-180, 180, -90, 90], origin='lower',
                     transform=ccrs.PlateCarree(), zorder=10,
                     interpolation='bilinear')
    cbar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    # Overlay contour lines of a second variable (e.g. SSH) on top
    if contour_var is not None and contour_var in dataset:
        contour_flat = np.asarray(dataset[contour_var].values,
                                  dtype=np.float64).flatten()
        _, _, contour_grid = interpolate_wave_data_to_grid(
            lon_flat, lat_flat, contour_flat,
            grid_resolution=(4000, 2000), max_distance=0.5)
        if use_satellite_background:
            (contour_grid,) = _mask_data_to_global_ocean(
                lon_mesh[0, :], lat_mesh[:, 0], contour_grid)
        if np.isfinite(contour_grid).any():
            c_lo = float(np.nanpercentile(contour_grid, 1))
            c_hi = float(np.nanpercentile(contour_grid, 99))
            c_levels = np.unique(np.round(np.linspace(c_lo, c_hi, contour_levels), 2))
            if len(c_levels) >= 2:
                # dark shadow + white main line so contours read on any color
                ax1.contour(lon_mesh, lat_mesh, contour_grid, levels=c_levels,
                            colors='black', linewidths=1.2, alpha=0.3,
                            transform=ccrs.PlateCarree(), zorder=20)
                ax1.contour(lon_mesh, lat_mesh, contour_grid, levels=c_levels,
                            colors='white', linewidths=0.5,
                            transform=ccrs.PlateCarree(), zorder=21)
    
    # Overlay wave direction arrows (degrees FROM -> arrows point TO)
    if direction_var is not None and direction_var in dataset:
        dm_flat = np.asarray(dataset[direction_var].values,
                             dtype=np.float64).flatten()
        dm_rad = np.deg2rad(dm_flat + 180.0)
        _, _, u_grid = interpolate_wave_data_to_grid(
            lon_flat, lat_flat, np.sin(dm_rad),
            grid_resolution=(1000, 500), max_distance=0.5)
        _, _, v_grid = interpolate_wave_data_to_grid(
            lon_flat, lat_flat, np.cos(dm_rad),
            grid_resolution=(1000, 500), max_distance=0.5)
        lon_d = np.linspace(-180, 180, u_grid.shape[1])
        lat_d = np.linspace(-90, 90, u_grid.shape[0])
        if use_satellite_background:
            u_grid, v_grid = _mask_data_to_global_ocean(lon_d, lat_d,
                                                        u_grid, v_grid)
        # subsample to a coarse arrow lattice, skip masked / zero cells
        step = 12
        x_d, y_d = np.meshgrid(lon_d, lat_d)
        u_sub = u_grid[::step, ::step]
        v_sub = v_grid[::step, ::step]
        x_sub = x_d[::step, ::step]
        y_sub = y_d[::step, ::step]
        ok = np.isfinite(u_sub) & np.isfinite(v_sub) \
            & (np.hypot(u_sub, v_sub) > 0.2)
        if ok.any():
            ax1.quiver(x_sub[ok], y_sub[ok], u_sub[ok], v_sub[ok],
                       color='black', alpha=0.45, scale=120, width=0.0008,
                       headwidth=2.2, headlength=2.8,
                       transform=ccrs.PlateCarree(), zorder=22)
    
    _add_global_satellite_with_ocean_mask(ax1, use_satellite_background)
    ax1.set_global()
    
    # Add colorbar (no ticks, no label — clean bar for flowchart use)
    if show_colorbar:
        cbar1 = plt.colorbar(cbar_mappable, ax=ax1, orientation='horizontal', pad=0.02, shrink=0.3)
        cbar1.set_ticks([])
    
    # --- Second subplot: original scatter nodes (if requested) ---
    if show_interpolation and ax2 is not None:
        scatter2 = ax2.scatter(lon_flat, lat_flat, c=wave_data_flat, cmap=cmap,
                               s=1, edgecolors='none',
                               transform=ccrs.PlateCarree(),
                               vmin=vmin, vmax=vmax, zorder=10)
        
        _add_global_satellite_with_ocean_mask(ax2, use_satellite_background)
        ax2.set_global()
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2, orientation='horizontal', pad=0.02, shrink=0.3)
        cbar2.set_label(var_prop['label'], fontsize=8)
        cbar2.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path:
        if transparent and output_path.lower().endswith(('.jpg', '.jpeg')):
            output_path = os.path.splitext(output_path)[0] + '.png'
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        if transparent:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', transparent=True)
        else:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Global wave plot saved as: {output_path}")
    
    # Prepare interpolated data to return if requested
    interpolated_data = None
    if return_interpolated:
        interpolated_data = {
            'lon_mesh': lon_mesh,
            'lat_mesh': lat_mesh,
            'values_interpolated': values_interpolated
        }
    
    if return_interpolated:
        return fig, interpolated_data
    return fig


def plot_era5_wind_global(dataset, output_path='wind_components_plot.png', projection='Robinson', dpi=200, use_satellite_background=False, figsize=(15, 5), transparent=False):
    """
    Plot 10m u and v wind components and wind vector field for global ERA5 data.
    
    Optimized for global data visualization with proper projection handling,
    no plt.show() blocking, and flexible coordinate name support.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Dataset containing wind components with dimensions (latitude, longitude) or (lat, lon)
    output_path : str, optional
        Path to save the output plot (default: 'wind_components_plot.png')
    projection : str, optional
        Map projection to use: 'Robinson', 'Mollweide', or 'PlateCarree' (default: 'Robinson')
    dpi : int, optional
        Resolution of the output figure (default: 200)
    transparent : bool, optional
        Save with a transparent figure background instead of white. Forces the
        output extension to .png since JPEG cannot store alpha (default: False).
    """
    
    # Detect coordinate names (global data may use 'longitude'/'latitude')
    if 'longitude' in dataset.coords:
        lon_name = 'longitude'
        lat_name = 'latitude'
    elif 'lon' in dataset.coords:
        lon_name = 'lon'
        lat_name = 'lat'
    else:
        # Try to infer from dimension names
        for coord in dataset.coords:
            if 'lon' in coord.lower():
                lon_name = coord
            if 'lat' in coord.lower():
                lat_name = coord
    
    # Select projection
    if projection == 'Robinson':
        proj = ccrs.Robinson(central_longitude=0)
    elif projection == 'Mollweide':
        proj = ccrs.Mollweide(central_longitude=0)
    else:
        proj = ccrs.PlateCarree(central_longitude=0)
    
    # Create figure with three subplots
    fig = plt.figure(figsize=figsize)
    
    ax1 = fig.add_subplot(1, 3, 1, projection=proj)
    ax2 = fig.add_subplot(1, 3, 2, projection=proj)
    ax3 = fig.add_subplot(1, 3, 3, projection=proj)
    
    # Extract time for plot title
    time_str = str(dataset.time.values)[:16] if hasattr(dataset, 'time') else 'Unknown Time'
    
    # Get wind data
    u_var = ERA5_VARIABLE_NAMES[0] if len(ERA5_VARIABLE_NAMES) > 0 else '10m_u_component_of_wind'
    v_var = ERA5_VARIABLE_NAMES[1] if len(ERA5_VARIABLE_NAMES) > 1 else '10m_v_component_of_wind'
    
    u_data = dataset[u_var].values
    v_data = dataset[v_var].values
    
    # Calculate symmetric colorbar limits
    max_abs_u = np.nanmax(np.abs(u_data))
    max_abs_v = np.nanmax(np.abs(v_data))
    max_abs_both = max(max_abs_u, max_abs_v)
    vmin = -max_abs_both
    vmax = max_abs_both
    
    # Get coordinates
    lat = dataset[lat_name].values
    lon = dataset[lon_name].values
    
    # Convert longitude from 0-360 to -180-180 if needed
    if np.any(lon > 180):
        lon = np.where(lon > 180, lon - 360, lon)
        # Sort by longitude for proper plotting
        sort_idx = np.argsort(lon)
        lon = lon[sort_idx]
        u_data = u_data[:, sort_idx]
        v_data = v_data[:, sort_idx]
    
    # Mask to the wave-node ocean so land keeps the satellite background
    # (same ocean mask as the wave scatter plots)
    if use_satellite_background:
        u_data, v_data = _mask_data_to_global_ocean(lon, lat, u_data, v_data)
    
    Lon, Lat = np.meshgrid(lon, lat)
    
    # --- Subplot 1: U-component ---
    im1 = ax1.pcolormesh(Lon, Lat, u_data, cmap='RdBu_r', 
                         transform=ccrs.PlateCarree(),
                         vmin=vmin, vmax=vmax, shading='auto', zorder=10)
    _add_global_satellite_with_ocean_mask(ax1, use_satellite_background)
    ax1.set_global()
    
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.02, shrink=0.3, label='Wind Speed (m/s)')
    cbar1.ax.tick_params(labelsize=8)
    cbar1.set_label('Wind Speed (m/s)', fontsize=8)
    
    # --- Subplot 2: V-component ---
    im2 = ax2.pcolormesh(Lon, Lat, v_data, cmap='RdBu_r', 
                         transform=ccrs.PlateCarree(),
                         vmin=vmin, vmax=vmax, shading='auto', zorder=10)
    _add_global_satellite_with_ocean_mask(ax2, use_satellite_background)
    ax2.set_global()
    
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.02, shrink=0.3, label='Wind Speed (m/s)')
    cbar2.ax.tick_params(labelsize=8)
    cbar2.set_label('Wind Speed (m/s)', fontsize=8)
    
    # --- Subplot 3: Wind vector field ---
    wind_speed = np.sqrt(u_data**2 + v_data**2)
    
    # Plot wind speed as background
    im3 = ax3.contourf(Lon, Lat, wind_speed, levels=20, cmap='viridis', 
                       transform=ccrs.PlateCarree(), alpha=0.8, zorder=10)
    
    # Subsample for quiver plot (global data needs more aggressive subsampling)
    subsample_lat = max(1, len(lat) // 30)
    subsample_lon = max(1, len(lon) // 60)
    
    quiver = ax3.quiver(Lon[::subsample_lat, ::subsample_lon], 
                        Lat[::subsample_lat, ::subsample_lon], 
                        u_data[::subsample_lat, ::subsample_lon], 
                        v_data[::subsample_lat, ::subsample_lon],
                        scale=max_abs_both*15, color='white', width=0.0015, 
                        transform=ccrs.PlateCarree(), alpha=0.8, zorder=10)
    
    _add_global_satellite_with_ocean_mask(ax3, use_satellite_background)
    ax3.set_global()
    
    cbar3 = plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.02, shrink=0.3, label='Wind Speed (m/s)')
    cbar3.ax.tick_params(labelsize=8)
    cbar3.set_label('Wind Speed (m/s)', fontsize=8)
    
    # Add quiver key
    ax3.quiverkey(quiver, 0.9, 0.05, max_abs_both, 
                  f'{max_abs_both:.1f} m/s', labelpos='E', coordinates='axes')
    
    plt.tight_layout()
    
    # Save figure
    if transparent and output_path.lower().endswith(('.jpg', '.jpeg')):
        output_path = os.path.splitext(output_path)[0] + '.png'
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    if transparent:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', transparent=True)
    else:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Global wind plot saved as: {output_path}")
    
    plt.close(fig)
    return fig


def plot_era5_pressure_global(dataset, output_path='pressure_plot.png', projection='Robinson', dpi=200, use_satellite_background=False, figsize=(9, 6), transparent=False, show_colorbar=True, max_plot_res_deg=0.5):
    """
    Plot mean sea level pressure field for global ERA5 data.

    Optimized for global data visualization with proper projection handling,
    no plt.show() blocking, and flexible coordinate name support.

    Parameters:
    -----------
    dataset : xarray.Dataset
        Dataset containing mean_sea_level_pressure with dimensions (latitude, longitude) or (lat, lon)
    output_path : str, optional
        Path to save the output plot (default: 'pressure_plot.png')
    projection : str, optional
        Map projection to use: 'Robinson', 'Mollweide', or 'PlateCarree' (default: 'Robinson')
    dpi : int, optional
        Resolution of the output figure (default: 200)
    transparent : bool, optional
        Save with a transparent figure background instead of white. Forces the
        output extension to .png since JPEG cannot store alpha (default: False).
    show_colorbar : bool, optional
        Whether to draw the horizontal colorbar below the map (default: True).
    max_plot_res_deg : float or None, optional
        Downsample all fields to at most this grid spacing (degrees) before
        plotting — visualization only, the dataset is not modified. Keeps
        vector PDF output small (default: 0.5; None disables).
    """
    
    # Detect coordinate names
    if 'longitude' in dataset.coords:
        lon_name = 'longitude'
        lat_name = 'latitude'
    elif 'lon' in dataset.coords:
        lon_name = 'lon'
        lat_name = 'lat'
    else:
        for coord in dataset.coords:
            if 'lon' in coord.lower():
                lon_name = coord
            if 'lat' in coord.lower():
                lat_name = coord
    
    # Select projection
    if projection == 'Robinson':
        proj = ccrs.Robinson(central_longitude=0)
    elif projection == 'Mollweide':
        proj = ccrs.Mollweide(central_longitude=0)
    else:
        proj = ccrs.PlateCarree(central_longitude=0)
    
    # Create figure with a single subplot
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(1, 1, 1, projection=proj)
    
    # Extract time
    time_str = str(dataset.time.values)[:16] if hasattr(dataset, 'time') else 'Unknown Time'
    
    # Get pressure data
    p_var = ERA5_VARIABLE_NAMES[2] if len(ERA5_VARIABLE_NAMES) > 2 else 'mean_sea_level_pressure'
    pressure_data = dataset[p_var].values
    pressure_hpa = pressure_data / 100.0  # Convert Pa to hPa

    # Get wind data for arrow overlay
    u_var = ERA5_VARIABLE_NAMES[0]
    v_var = ERA5_VARIABLE_NAMES[1]
    has_wind = u_var in dataset and v_var in dataset
    if has_wind:
        u_wind = dataset[u_var].values
        v_wind = dataset[v_var].values

    # Get coordinates
    lat = dataset[lat_name].values
    lon = dataset[lon_name].values

    # Convert longitude from 0-360 to -180-180 if needed
    if np.any(lon > 180):
        lon = np.where(lon > 180, lon - 360, lon)
        # Sort by longitude for proper plotting
        sort_idx = np.argsort(lon)
        lon = lon[sort_idx]
        pressure_hpa = pressure_hpa[:, sort_idx]
        if has_wind:
            u_wind = u_wind[:, sort_idx]
            v_wind = v_wind[:, sort_idx]

    # Downsample for plotting only (keeps vector PDF output small)
    if max_plot_res_deg is not None and len(lon) > 1 and len(lat) > 1:
        dlon = abs(lon[1] - lon[0])
        dlat = abs(lat[1] - lat[0])
        sx = max(1, int(round(max_plot_res_deg / dlon)))
        sy = max(1, int(round(max_plot_res_deg / dlat)))
        if sx > 1 or sy > 1:
            lon = lon[::sx]
            lat = lat[::sy]
            pressure_hpa = pressure_hpa[::sy, ::sx]
            if has_wind:
                u_wind = u_wind[::sy, ::sx]
                v_wind = v_wind[::sy, ::sx]

    Lon, Lat = np.meshgrid(lon, lat)

    p_min, p_max = np.nanmin(pressure_hpa), np.nanmax(pressure_hpa)

    # --- Pcolormesh (smooth color fill) ---
    # rasterized=True: avoids hairline gaps between vector quads in PDF viewers
    pcolor_plot = ax.pcolormesh(Lon, Lat, pressure_hpa,
                                 cmap='plasma', transform=ccrs.PlateCarree(),
                                 shading='auto', vmin=p_min, vmax=p_max, zorder=10,
                                 rasterized=True)

    # --- Overlay 10m wind arrows ---
    if has_wind:
        nlat, nlon = u_wind.shape
        step_lat = max(1, nlat // 42)
        step_lon = max(1, nlon // 84)
        u_sub = u_wind[::step_lat, ::step_lon]
        v_sub = v_wind[::step_lat, ::step_lon]
        x_sub = Lon[::step_lat, ::step_lon]
        y_sub = Lat[::step_lat, ::step_lon]
        ok = np.isfinite(u_sub) & np.isfinite(v_sub) \
            & (np.hypot(u_sub, v_sub) > 0.3)
        if ok.any():
            ax.quiver(x_sub[ok], y_sub[ok], u_sub[ok], v_sub[ok],
                       color='black', alpha=0.45, scale=1200, width=0.0008,
                       headwidth=2.2, headlength=2.8,
                       transform=ccrs.PlateCarree(), zorder=22)

    # Coastlines only — pressure covers the full globe, no need for satellite bg
    import cartopy.feature as cfeature
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.8, zorder=11)
    ax.set_global()

    if show_colorbar:
        # No ticks, no label — clean bar matching the wave-field plots
        cbar = plt.colorbar(pcolor_plot, ax=ax, orientation='horizontal', pad=0.02, shrink=0.3)
        cbar.set_ticks([])
    
    plt.tight_layout()
    
    # Save figure
    if transparent and output_path.lower().endswith(('.jpg', '.jpeg')):
        output_path = os.path.splitext(output_path)[0] + '.png'
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    if transparent:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', transparent=True)
    else:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Global pressure plot saved as: {output_path}")
    
    plt.close(fig)
    return fig


