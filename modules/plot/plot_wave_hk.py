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


def interpolate_wave_data_to_grid(lons, lats, values, grid_resolution=400, max_distance=0.5):
    """
    Interpolate unstructured wave data to a regular lat-lon grid.
    
    This function performs 2-step interpolation:
    1. Linear interpolation for the main data
    2. Nearest neighbor interpolation to fill remaining NaN values near data points
    
    Parameters:
    -----------
    lons : np.ndarray
        Longitude values (1D array, will be flattened)
    lats : np.ndarray  
        Latitude values (1D array, will be flattened)
    values : np.ndarray
        Data values (1D array, will be flattened)
    grid_resolution : int or (int, int), optional
        Grid resolution for interpolation (default: 400).  A single int
        gives a square NxN grid; a tuple (nx, ny) gives nx lon points by
        ny lat points.
    max_distance : float, optional
        Maximum distance (in degrees) to fill NaN values using nearest neighbor (default: 0.5)
    
    Returns:
    --------
    tuple: (lon_mesh, lat_mesh, values_interpolated)
        - lon_mesh: 2D array of longitudes for the grid
        - lat_mesh: 2D array of latitudes for the grid
        - values_interpolated: 2D array of interpolated values
    """
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    
    # Flatten inputs
    lon_flat = lons.flatten()
    lat_flat = lats.flatten()
    values_flat = values.flatten()
    
    # Get valid data points (non-NaN)
    mask_valid = ~np.isnan(values_flat)
    lons_valid = lon_flat[mask_valid]
    lats_valid = lat_flat[mask_valid]
    values_valid = values_flat[mask_valid]
    
    # Create a regular grid for interpolation
    if np.isscalar(grid_resolution):
        nx = ny = grid_resolution
    else:
        nx, ny = grid_resolution
    lon_grid = np.linspace(-180, 180, nx)
    lat_grid = np.linspace(-90, 90, ny)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # Interpolate using griddata with linear method
    values_interpolated = griddata(
        (lons_valid, lats_valid), 
        values_valid, 
        (lon_mesh, lat_mesh), 
        method='linear',
        fill_value=np.nan
    )
    
    # Use nearest neighbor to fill remaining NaN values near data points
    mask_nan = np.isnan(values_interpolated)
    if np.any(mask_nan):
        values_nearest = griddata(
            (lons_valid, lats_valid), 
            values_valid, 
            (lon_mesh, lat_mesh), 
            method='nearest'
        )
        
        # Build KDTree to find distance to nearest data point
        tree = cKDTree(np.column_stack([lons_valid, lats_valid]))
        grid_points = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
        distances, _ = tree.query(grid_points)
        distances = distances.reshape(lon_mesh.shape)
        
        # Only fill NaN values that are close to existing data points
        fill_mask = mask_nan & (distances < max_distance)
        values_interpolated[fill_mask] = values_nearest[fill_mask]
    
    return lon_mesh, lat_mesh, values_interpolated


def plot_variable_distributions(ds, variables=None, save_path=None, dpi=150):
    """
    Plot distributions for multiple variables (hs, tp, dm, elev) from an xarray Dataset.

    Args:
        ds (xarray.Dataset): Dataset containing variables (hs, tp, dm, elev)
        variables (list, optional): List of variable names to plot. Defaults to ['hs','tp','dm','elev']
        save_path (str, optional): Path to save the figure. If None, displays the plot
        dpi (int): Resolution of the output figure

    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    if variables is None:
        variables = ['hs', 'tp', 'dm', 'elev']

    # Validate variables
    vars_present = [v for v in variables if v in ds]
    if not vars_present:
        raise ValueError(f"None of the requested variables {variables} are present in the dataset")

    # Create 2x2 grid (support up to 4 variables)
    n_plots = min(4, len(vars_present))
    fig = plt.figure(figsize=(14, 10))

    for i, var in enumerate(vars_present[:4], start=1):
        ax = fig.add_subplot(2, 2, i)

        data = ds[var].values
        data = data.flatten()
        data = data[~np.isnan(data)]

        if var == 'dm':
            # Directional data: make a rose (polar) histogram
            ax = fig.add_subplot(2, 2, i, projection='polar')
            theta = np.deg2rad(data)
            bins = 36
            counts, bin_edges = np.histogram(theta, bins=bins)
            widths = np.diff(bin_edges)
            bin_centers = bin_edges[:-1] + widths / 2.0
            ax.bar(bin_centers, counts, width=widths, bottom=0.0, color='teal', alpha=0.7, edgecolor='k')
            ax.set_title('Wave Direction Distribution (deg)')

            # Circular mean
            if data.size > 0:
                sin_sum = np.sum(np.sin(theta))
                cos_sum = np.sum(np.cos(theta))
                mean_angle = np.arctan2(sin_sum, cos_sum)
                mean_deg = np.rad2deg(mean_angle) % 360
                ax.text(0.02, 0.98, f'Mean Dir: {mean_deg:.1f}°', transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        else:
            # Numeric variable: histogram
            ax.hist(data, bins=100, alpha=0.8, color='skyblue', edgecolor='black', linewidth=0.4)
            ax.set_yscale('log')
            ax.set_xlabel(var)
            ax.set_ylabel('Frequency (log scale)')
            # Use unified variable metadata
            title = WAVE_VARIABLES.get(var, {}).get('plot_label', var)
            ax.set_title(title)

            # Add statistics box
            if data.size > 0:
                mean_val = np.mean(data)
                std_val = np.std(data)
                min_val = np.min(data)
                max_val = np.max(data)
                stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Variable distributions saved to: {save_path}")

    return fig


def plot_wave_data(dataset, variable='hs', region=None, save_path=None, dpi=300, vmin=None, vmax=None, cmap='turbo', ax=None, return_ax=False, scatter_size=30, custom_title=None, show_cities=True, show_interpolation=False, show_colorbar=True, show_nodes=True, show_left_labels=True, show_bottom_labels=True, use_high_res_coastline=False, use_satellite_background=False, figsize=(14, 10), cbar_orientation='vertical', gridline_spacing=None, show_top_labels=True, transparent=False, show_cbar_label=True):
    """
    Plot wave data on a geographical map using Cartopy for professional coastline.
    Display the number of nodes (scatter points) in the plot region.
    
    Args:
        dataset (xarray.Dataset): Dataset containing wave data with coordinates
        variable (str): Variable name to plot ('hs', 'tp', 'dm', or 'elev')
        region (list, optional): [min_lon, max_lon, min_lat, max_lat] for map extent
        save_path (str, optional): Path to save the figure. If None, displays the plot
        dpi (int): Resolution of the output figure
        vmin (float, optional): Minimum value for color scale
        vmax (float, optional): Maximum value for color scale
        cmap (str, optional): Colormap name for the plot (default: 'turbo')
        ax (matplotlib.axes, optional): Existing axes to plot on. If None, creates new figure
        return_ax (bool): If True, returns (fig, ax) instead of just fig
        scatter_size (float or int): Size of scatter points (default: 15)
        custom_title (str, optional): Custom title for the plot. If None, uses default title format
        show_cities (bool): If True, shows major city labels on the map (default: True)
        show_interpolation (bool): If True, creates a second subplot with interpolated data filling ocean gaps (default: False)
        figsize (tuple): Figure size in inches (default: (14, 10))
        cbar_orientation (str): 'vertical' (right) or 'horizontal' (bottom) colorbar
        gridline_spacing (tuple, optional): (lon_step, lat_step) in degrees for the
            graticule; None uses the cartopy default locator
        show_top_labels (bool): If True, shows longitude labels on the top edge
            (default: True)
        transparent (bool): Save with a transparent figure background instead
            of white (default: False)
        show_cbar_label (bool): If False, the colorbar is drawn without ticks
            or label (default: True)
    
    Returns:
        matplotlib.figure.Figure or tuple: The created figure object, or (fig, ax) if return_ax=True
    """
    # Create a copy of the dataset to avoid modifying the original
    ds_plot = dataset.copy()
    
    # If region is specified, filter the dataset to the specified region
    if region is not None:
        min_lon, max_lon, min_lat, max_lat = region

        # Filter dataset to the specified region
        ds_plot = ds_plot.where(
            (ds_plot.lon >= min_lon) & (ds_plot.lon <= max_lon) & 
            (ds_plot.lat >= min_lat) & (ds_plot.lat <= max_lat), 
            drop=True
        )

    # Count the number of valid nodes (non-NaN points) in the filtered dataset
    valid_nodes = np.sum(~np.isnan(ds_plot[variable].values))
    total_nodes = len(ds_plot[variable].values.flatten())
    
    # Create figure with Cartopy projection
    if ax is None:
        if show_interpolation:
            # Create figure with two subplots side by side with reduced spacing
            fig = plt.figure(figsize=(28, 10))
            ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
            ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
            # Adjust spacing between subplots
            plt.subplots_adjust(wspace=0)  # Reduce horizontal spacing
            ax = ax1
        else:
            fig = plt.figure(figsize=figsize)
            # Use PlateCarree projection (standard lat/lon)
            ax = plt.axes(projection=ccrs.PlateCarree())
    else:
        fig = ax.get_figure()
    
    # Set map extent based on data or provided region
    if region is None:
        # Calculate region from data with 10% buffer
        lons, lats = ds_plot.lon.values, ds_plot.lat.values
        lon_buffer = (lons.max() - lons.min()) * 0.1
        lat_buffer = (lats.max() - lats.min()) * 0.1
        
        region = [
            lons.min() - lon_buffer, 
            lons.max() + lon_buffer,
            lats.min() - lat_buffer, 
            lats.max() + lat_buffer
        ]
    
    ax.set_extent(region, crs=ccrs.PlateCarree())
    
    # Extract data for plotting from the filtered dataset
    lons = ds_plot.lon.values
    lats = ds_plot.lat.values
    wave_data = ds_plot[variable].values
    
    # Add high-quality coastline data from Cartopy
    if use_satellite_background and use_high_res_coastline and add_satellite_background is not None:
        # Satellite imagery + ocean masked out using the model boundary polylines
        add_satellite_background(ax, region)
    elif use_high_res_coastline:
        ax.add_feature(cfeature.GSHHSFeature(scale='full', levels=[1]),
                       facecolor='whitesmoke', edgecolor='black', linewidth=0.6, zorder=4)
    else:
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, zorder=4)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5, alpha=0.7, zorder=4)
        # ax.add_feature(cfeature.OCEAN, color='white', alpha=0.3, zorder=2)
        ax.add_feature(cfeature.LAND, color='whitesmoke', alpha=1, zorder=3)
    
    # Define variable-specific properties
    var_properties = {var: {
        'label': WAVE_VARIABLES[var]['plot_label'],
        'title': WAVE_VARIABLES[var]['plot_title']
    } for var in WAVE_VARIABLES}
    
    var_prop = var_properties.get(variable, {'label': f'{variable}', 'title': f'{variable} Distribution'})
    
    # Determine colorbar range and extend behavior
    # Will be updated if show_interpolation is True
    cbar_vmin = vmin
    cbar_vmax = vmax
    cbar_extend = 'neither'
    
    # Create scatter plot colored by wave data
    # zorder=6 keeps the data points on top of the satellite image (1), the
    # ocean mask (2), coastline/land (4) and gridlines (5).
    scatter = ax.scatter(lons, lats, c=wave_data, cmap=cmap, 
                        s=scatter_size, edgecolors='none', 
                        transform=ccrs.PlateCarree(),
                        vmin=cbar_vmin, vmax=cbar_vmax, zorder=6)
    # Expose the mappable for callers that want to draw a shared colorbar.
    ax._wave_scatter = scatter
    
    # Add colorbar (will be updated after interpolation if needed)
    if show_colorbar:
        # A horizontal colorbar needs extra pad to clear the bottom
        # longitude labels drawn by the gridliner
        cbar_pad = 0.12 if cbar_orientation == 'horizontal' else 0.02
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=cbar_pad, extend=cbar_extend,
                            orientation=cbar_orientation, aspect=30)
        if show_cbar_label:
            cbar.set_label(var_prop['label'], fontsize=15)
            cbar.ax.tick_params(labelsize=12)
        else:
            cbar.set_ticks([])
    
    # Add gridlines with lat/lon coordinate labels.
    # NOTE: cartopy's labelled gridliner collapses figures saved with
    # bbox_inches='tight' when top labels are disabled (cartopy gridliner
    # bug), so the top-labels-off case draws an unlabelled grid plus plain
    # geo ticks instead.  Right labels are off (colorbar sits there).
    if show_top_labels:
        gl = ax.gridlines(draw_labels=True, alpha=0.3, zorder=5)
        gl.right_labels = False
        gl.top_labels = show_top_labels
        gl.left_labels = show_left_labels
        gl.bottom_labels = show_bottom_labels
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if gridline_spacing is not None:
            from matplotlib import ticker as mticker
            gl.xlocator = mticker.MultipleLocator(gridline_spacing[0])
            gl.ylocator = mticker.MultipleLocator(gridline_spacing[1])
    else:
        # Cartopy's labelled gridliner collapses figures saved with
        # bbox_inches='tight' when top labels are disabled (gridliner bug),
        # so draw the grid without labels and use plain geo ticks instead.
        ax.gridlines(draw_labels=False, alpha=0.3, zorder=5)
        from matplotlib import ticker as mticker
        min_lon, max_lon, min_lat, max_lat = region
        if gridline_spacing is not None:
            lon_step, lat_step = gridline_spacing
            xticks = np.arange(np.ceil(min_lon / lon_step) * lon_step, max_lon, lon_step)
            yticks = np.arange(np.ceil(min_lat / lat_step) * lat_step, max_lat, lat_step)
        else:
            xticks = mticker.MaxNLocator(5).tick_values(min_lon, max_lon)
            yticks = mticker.MaxNLocator(5).tick_values(min_lat, max_lat)
        if show_bottom_labels:
            ax.set_xticks(xticks, crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        if show_left_labels:
            ax.set_yticks(yticks, crs=ccrs.PlateCarree())
            ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    
    # Get time from dataset for title
    # time_str = str(ds_plot.time.values)[:16]  # Extract date part
    # formatted_time = time_str.replace('T', ' ')
    
    # Create title with node count information
    if custom_title is not None:
        title = custom_title
    else:
        title = f'{var_prop["title"]}\nDate: {ds_plot.time.values}'
        if show_nodes:
            title += f' | Nodes: {valid_nodes}'
    ax.set_title(title, fontsize=16)
    
    # Add node count as text annotation in the plot
    if show_nodes:
        ax.text(0.02, 0.98, f'Nodes: {valid_nodes}', transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
    
    # Add major cities labels if requested
    if show_cities:
        # Filter cities within the plot region
        min_lon, max_lon, min_lat, max_lat = region
        for city_name, (lon, lat) in MAJOR_CITIES.items():
            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                # Add city marker
                ax.plot(lon, lat, '^', color='red', markersize=6, markeredgewidth=0, transform=ccrs.PlateCarree(), zorder=5)
                # Add city label with white background for better visibility
                ax.text(lon + 0.1, lat + 0.1, city_name, transform=ccrs.PlateCarree(),
                       fontsize=9, fontweight='bold', 
                       ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='black', alpha=0.7, linewidth=0.5),
                       zorder=6)
    
    # Create interpolated subplot if requested
    if show_interpolation and ax is not None:
        from scipy.interpolate import griddata
        from scipy.spatial import cKDTree
        
        # Get valid data points (non-NaN)
        lons_valid = lons[~np.isnan(wave_data)]
        lats_valid = lats[~np.isnan(wave_data)]
        values_valid = wave_data[~np.isnan(wave_data)]
        
        # Create a fine grid for interpolation
        min_lon, max_lon, min_lat, max_lat = region
        grid_resolution = 400  # Increased resolution for smoother interpolation
        lon_grid = np.linspace(min_lon, max_lon, grid_resolution)
        lat_grid = np.linspace(min_lat, max_lat, grid_resolution)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Interpolate using griddata with linear method (more stable than cubic)
        values_interpolated = griddata(
            (lons_valid, lats_valid), 
            values_valid, 
            (lon_mesh, lat_mesh), 
            method='linear',
            fill_value=np.nan
        )
        
        # Use nearest neighbor to fill remaining NaN values in ocean areas
        # This prevents interpolation onto land while filling ocean gaps
        mask_nan = np.isnan(values_interpolated)
        if np.any(mask_nan):
            values_nearest = griddata(
                (lons_valid, lats_valid), 
                values_valid, 
                (lon_mesh, lat_mesh), 
                method='nearest'
            )
            
            # Build KDTree to find distance to nearest data point
            tree = cKDTree(np.column_stack([lons_valid, lats_valid]))
            grid_points = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
            distances, _ = tree.query(grid_points)
            distances = distances.reshape(lon_mesh.shape)
            
            # Only fill NaN values that are close to existing data points (within reasonable distance)
            # This prevents filling land areas that are far from ocean data
            max_distance = 0.5  # degrees, adjust based on your data resolution
            fill_mask = mask_nan & (distances < max_distance)
            values_interpolated[fill_mask] = values_nearest[fill_mask]
        
        # Clip values to reasonable range (no negative values for wave height)
        if variable in ['hs', 'elev']:  # Variables that shouldn't be negative
            values_interpolated = np.clip(values_interpolated, 0, None)
        
        # Calculate colorbar range based on interpolated data
        interp_valid = values_interpolated[~np.isnan(values_interpolated)]
        if len(interp_valid) > 0:
            interp_min = np.min(interp_valid)
            interp_max = np.max(interp_valid)
            
            # Use user-provided vmin/vmax if specified, otherwise use interpolated range
            if vmin is None:
                cbar_vmin = interp_min
            else:
                cbar_vmin = vmin
                
            if vmax is None:
                cbar_vmax = interp_max
            else:
                cbar_vmax = vmax
            
            # Determine extend behavior based on scatter data range
            scatter_min = np.nanmin(values_valid)
            scatter_max = np.nanmax(values_valid)
            
            extend_min = scatter_min < cbar_vmin
            extend_max = scatter_max > cbar_vmax
            
            if extend_min and extend_max:
                cbar_extend = 'both'
            elif extend_min:
                cbar_extend = 'min'
            elif extend_max:
                cbar_extend = 'max'
            else:
                cbar_extend = 'neither'
            
            # Update the first scatter plot with new colorbar range
            scatter.set_clim(vmin=cbar_vmin, vmax=cbar_vmax)

            # Update colorbar to reflect new extend
            cbar.update_normal(scatter)
        
        # Use ax2 for interpolated plot
        ax_interp = ax2 if show_interpolation else ax
        
        # Set map extent
        ax_interp.set_extent(region, crs=ccrs.PlateCarree())
        
        # Plot interpolated data using pcolormesh with masked array to hide NaN
        from numpy import ma
        values_masked = ma.masked_invalid(values_interpolated)
        im = ax_interp.pcolormesh(lon_mesh, lat_mesh, values_masked, 
                                   cmap=cmap, shading='auto',
                                   transform=ccrs.PlateCarree(),
                                   vmin=cbar_vmin, vmax=cbar_vmax)
        
        # Add cartopy features - IMPORTANT: Add land AFTER plotting data to cover interpolated areas on land
        if use_satellite_background and use_high_res_coastline:
            if add_satellite_background is not None:
                add_satellite_background(ax_interp, region)
        elif use_high_res_coastline:
            ax_interp.add_feature(cfeature.GSHHSFeature(scale='full', levels=[1]),
                                  facecolor='whitesmoke', edgecolor='black', linewidth=0.8, zorder=10)
        else:
            ax_interp.add_feature(cfeature.LAND, color='whitesmoke', alpha=1, zorder=3)
            ax_interp.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, zorder=4)
            ax_interp.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5, alpha=0.7, zorder=4)
        
        # Add colorbar for interpolated plot (no extend needed as it's based on interpolated range)
        cbar_interp = plt.colorbar(im, ax=ax_interp, shrink=0.8, pad=0.02, extend='neither')
        cbar_interp.set_label(var_prop['label'], fontsize=12)
        
        # Add gridlines
        gl_interp = ax_interp.gridlines(draw_labels=True, alpha=0.3)
        gl_interp.top_labels = False
        gl_interp.right_labels = False
        gl_interp.xformatter = LONGITUDE_FORMATTER
        gl_interp.yformatter = LATITUDE_FORMATTER
        gl_interp.xlabel_style = {'size': 10}
        gl_interp.ylabel_style = {'size': 10}
        
        # Set title for interpolated plot
        if custom_title is not None:
            title_interp = custom_title + ' (Interpolated)'
        else:
            title_interp = f'{var_prop["title"]} (Interpolated)\nDate: {ds_plot.time.values}'
        ax_interp.set_title(title_interp, fontsize=16)
        
        # Add cities to interpolated plot if requested
        if show_cities:
            for city_name, (lon, lat) in MAJOR_CITIES.items():
                if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                    ax_interp.plot(lon, lat, '^', color='red', markersize=6, markeredgewidth=0, transform=ccrs.PlateCarree(), zorder=5)
                    ax_interp.text(lon + 0.1, lat + 0.1, city_name, transform=ccrs.PlateCarree(),
                                  fontsize=9, fontweight='bold', 
                                  ha='left', va='bottom',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                           edgecolor='black', alpha=0.7, linewidth=0.5),
                                  zorder=6)
    
    # Save figure if path provided
    if save_path:
        if transparent and save_path.lower().endswith(('.jpg', '.jpeg')):
            save_path = os.path.splitext(save_path)[0] + '.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if transparent:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")

    # Only close the figure if we created it (not passed in)
    if not return_ax:
        plt.close(fig)
    
    if return_ax:
        return fig, ax
    return fig

def plot_wave_data_interpolated(dataset, variable='hs', region=None, save_path=None, dpi=300,
                               vmin=None, vmax=None, cmap='turbo', custom_title=None,
                               grid_resolution=1400, max_distance=0.05,
                               robust_clim=(0.1, 99.9), use_satellite_background=True,
                               direction_var=None, direction_arrows=30,
                               symmetric_clim=False, figsize=(14, 10),
                               cbar_orientation='vertical', gridline_spacing=None,
                               show_top_labels=True, show_bottom_labels=True,
                               transparent=False, show_cbar_label=True,
                               show_left_labels=True):
    """
    Plot wave data interpolated onto a regular grid over the model ocean.

    Same layout as :func:`plot_wave_data` (satellite background, colorbar,
    graticule, title), but the scattered nodes are interpolated instead of
    drawn as dots:

    1. ``griddata`` linear interpolation onto a regular grid spanning
       ``region``, plus nearest-neighbour fill for cells within
       ``max_distance`` degrees of a node (same two-step approach as the
       global plots in :func:`plot_wave_data_global`).
    2. The grid is masked to the model ocean polygon (SCHISM boundary
       polylines via ``_load_ocean_geometry``), so interpolation never
       bleeds across land and islands keep their satellite detail.  The
       point-in-polygon test uses ``matplotlib.path.Path`` (shapely
       semantics): the clipped model ring has double-wound sections where
       even-odd rasterization (e.g. PIL) wrongly punches holes in open
       water, e.g. south of Hong Kong island.
    3. When ``vmin``/``vmax`` are not given, colorbar limits are the
       ``robust_clim`` percentiles of the node values with
       ``extend='both'``, so a few outlier nodes cannot dominate the scale
       or smear into large saturated blobs.
    4. Optionally, a wave direction variable (degrees FROM, nautical
       convention, e.g. 'dm') is overlaid as quiver arrows pointing TO the
       propagation direction (same approach as ``plot_wave_data_global``).

    Args:
        dataset (xarray.Dataset): Dataset with the variable and lon/lat coords.
        variable (str): Variable name to plot ('hs', 'tp', 'dm', or 'elev').
        region (list, optional): [min_lon, max_lon, min_lat, max_lat] map extent.
        save_path (str, optional): Path to save the figure. If None, displays the plot.
        dpi (int): Resolution of the output figure.
        vmin, vmax (float, optional): Explicit colorbar limits (disables robust limits).
        cmap (str): Colormap name (default: 'turbo').
        custom_title (str, optional): Custom title. If None, uses the default format.
        grid_resolution (int): Number of longitude grid points (default: 1400);
            latitude points scale with the region aspect ratio.
        max_distance (float): Max distance (degrees) for nearest-neighbour fill.
        robust_clim (tuple): (low, high) percentiles for automatic colorbar limits.
        use_satellite_background (bool): Draw satellite imagery + model ocean mask.
        direction_var (str, optional): Name of a wave direction variable in
            degrees FROM (nautical convention) to overlay as quiver arrows.
        direction_arrows (int): Approximate number of arrows across the region
            (default: 30).
        symmetric_clim (bool): If True and vmin/vmax are not given, make the
            automatic colorbar limits symmetric about zero
            (-max(|lo|, |hi|), +max(|lo|, |hi|)); pair with a diverging cmap
            for signed variables like 'elev'.
        figsize (tuple): Figure size in inches (default: (14, 10)).
        cbar_orientation (str): 'vertical' (right) or 'horizontal' (bottom)
            colorbar; extra pad is added when horizontal to clear the bottom
            longitude labels.
        gridline_spacing (tuple, optional): (lon_step, lat_step) in degrees for
            the graticule; None uses the cartopy default locator.
        show_top_labels (bool): Longitude labels on the top edge (default: True).
        show_bottom_labels (bool): Longitude labels on the bottom edge
            (default: True).
        transparent (bool): Save with a transparent figure background instead
            of white. Forces the output extension to .png since JPEG cannot
            store alpha (default: False).
        show_cbar_label (bool): If False, the colorbar is drawn without ticks
            or label (default: True).
        show_left_labels (bool): Latitude labels on the left edge
            (default: True).

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    from matplotlib.path import Path
    from numpy import ma

    ds_plot = dataset.copy()
    if region is not None:
        min_lon, max_lon, min_lat, max_lat = region
        ds_plot = ds_plot.where(
            (ds_plot.lon >= min_lon) & (ds_plot.lon <= max_lon) &
            (ds_plot.lat >= min_lat) & (ds_plot.lat <= max_lat),
            drop=True
        )
    else:
        lons_all, lats_all = ds_plot.lon.values, ds_plot.lat.values
        region = [lons_all.min(), lons_all.max(), lats_all.min(), lats_all.max()]
        min_lon, max_lon, min_lat, max_lat = region

    lons = ds_plot.lon.values
    lats = ds_plot.lat.values
    values = ds_plot[variable].values
    valid = ~np.isnan(values)
    lons, lats, values = lons[valid], lats[valid], values[valid]

    # --- Interpolate scattered nodes onto a regular grid ---
    nx = grid_resolution
    ny = max(2, int(round(grid_resolution * (max_lat - min_lat) / (max_lon - min_lon))))
    lon_grid = np.linspace(min_lon, max_lon, nx)
    lat_grid = np.linspace(min_lat, max_lat, ny)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    tree = cKDTree(np.column_stack([lons, lats]))

    def _to_grid(vals):
        """Linear interpolation + nearest-neighbour fill onto the grid."""
        grid = griddata((lons, lats), vals, (lon_mesh, lat_mesh),
                        method='linear', fill_value=np.nan)
        mask_nan = np.isnan(grid)
        if np.any(mask_nan):
            nearest = griddata((lons, lats), vals, (lon_mesh, lat_mesh),
                               method='nearest')
            dist, _ = tree.query(np.column_stack([lon_mesh.ravel(),
                                                  lat_mesh.ravel()]))
            dist = dist.reshape(lon_mesh.shape)
            fill = mask_nan & (dist < max_distance)
            grid[fill] = nearest[fill]
        return grid

    interp = _to_grid(values)

    # --- Mask to the model ocean polygon (land keeps the satellite background) ---
    ocean_mask = None
    if use_satellite_background and add_satellite_background is not None:
        try:
            from modules.plot.satellite_background import _load_ocean_geometry
            ocean, _ = _load_ocean_geometry(region)
            pts = np.column_stack([np.tile(lon_grid, ny), np.repeat(lat_grid, nx)])
            mask_flat = np.zeros(len(pts), dtype=bool)
            polys = [ocean] if ocean.geom_type == 'Polygon' else list(ocean.geoms)
            for poly in polys:
                mask_flat |= Path(np.asarray(poly.exterior.coords)).contains_points(pts)
                for interior in poly.interiors:
                    if len(interior.coords) >= 3:
                        mask_flat &= ~Path(np.asarray(interior.coords)).contains_points(pts)
            ocean_mask = mask_flat.reshape(ny, nx)
            interp[~ocean_mask] = np.nan
        except Exception as e:
            print(f"Warning: ocean masking failed, using unmasked interpolation: {e}")

    # --- Colorbar limits ---
    if vmin is None or vmax is None:
        lo, hi = np.percentile(values, robust_clim)
        if symmetric_clim:
            abs_max = max(abs(lo), abs(hi))
            lo, hi = -abs_max, abs_max
        if vmin is None:
            vmin = lo
        if vmax is None:
            vmax = hi
        extend = 'both'
    else:
        extend = 'neither'

    # --- Figure ---
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(region, crs=ccrs.PlateCarree())

    if use_satellite_background and add_satellite_background is not None:
        add_satellite_background(ax, region)
    else:
        ax.add_feature(cfeature.GSHHSFeature(scale='full', levels=[1]),
                       facecolor='whitesmoke', edgecolor='black', linewidth=0.6, zorder=4)

    mesh = ax.pcolormesh(lon_mesh, lat_mesh, ma.masked_invalid(interp),
                         cmap=cmap, vmin=vmin, vmax=vmax, shading='auto',
                         transform=ccrs.PlateCarree(), zorder=6)
    cbar_pad = 0.12 if cbar_orientation == 'horizontal' else 0.02
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.5, pad=cbar_pad, extend=extend,
                        orientation=cbar_orientation, aspect=30)

    var_prop = {'label': WAVE_VARIABLES.get(variable, {}).get('plot_label', variable),
                'title': WAVE_VARIABLES.get(variable, {}).get('plot_title', f'{variable} Distribution')}
    if show_cbar_label:
        cbar.set_label(var_prop['label'], fontsize=15)
        cbar.ax.tick_params(labelsize=12)
    else:
        cbar.set_ticks([])

    # --- Overlay wave direction arrows (degrees FROM -> arrows point TO) ---
    if direction_var is not None and direction_var in ds_plot:
        dm = np.asarray(ds_plot[direction_var].values, dtype=np.float64)[valid]
        dm_rad = np.deg2rad(dm + 180.0)
        u_grid = _to_grid(np.sin(dm_rad))
        v_grid = _to_grid(np.cos(dm_rad))
        if ocean_mask is not None:
            u_grid[~ocean_mask] = np.nan
            v_grid[~ocean_mask] = np.nan
        step = max(1, nx // direction_arrows)
        u_sub = u_grid[::step, ::step]
        v_sub = v_grid[::step, ::step]
        x_sub = lon_mesh[::step, ::step]
        y_sub = lat_mesh[::step, ::step]
        ok = np.isfinite(u_sub) & np.isfinite(v_sub) \
            & (np.hypot(u_sub, v_sub) > 0.2)
        if ok.any():
            ax.quiver(x_sub[ok], y_sub[ok], u_sub[ok], v_sub[ok],
                      color='black', alpha=0.55, scale=40, width=0.0022,
                      headwidth=2.5, headlength=3.5, pivot='mid',
                      transform=ccrs.PlateCarree(), zorder=7)

    if show_top_labels:
        gl = ax.gridlines(draw_labels=True, alpha=0.3, zorder=5)
        gl.right_labels = False
        gl.top_labels = show_top_labels
        gl.bottom_labels = show_bottom_labels
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if gridline_spacing is not None:
            from matplotlib import ticker as mticker
            gl.xlocator = mticker.MultipleLocator(gridline_spacing[0])
            gl.ylocator = mticker.MultipleLocator(gridline_spacing[1])
    else:
        # Cartopy's labelled gridliner collapses figures saved with
        # bbox_inches='tight' when top labels are disabled (gridliner bug),
        # so draw the grid without labels and use plain geo ticks instead.
        ax.gridlines(draw_labels=False, alpha=0.3, zorder=5)
        from matplotlib import ticker as mticker
        if gridline_spacing is not None:
            lon_step, lat_step = gridline_spacing
            xticks = np.arange(np.ceil(min_lon / lon_step) * lon_step, max_lon, lon_step)
            yticks = np.arange(np.ceil(min_lat / lat_step) * lat_step, max_lat, lat_step)
        else:
            xticks = mticker.MaxNLocator(5).tick_values(min_lon, max_lon)
            yticks = mticker.MaxNLocator(5).tick_values(min_lat, max_lat)
        if show_bottom_labels:
            ax.set_xticks(xticks, crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        if show_left_labels:
            ax.set_yticks(yticks, crs=ccrs.PlateCarree())
            ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

    title = custom_title if custom_title is not None else \
        f'{var_prop["title"]}\nDate: {ds_plot.time.values}'
    ax.set_title(title, fontsize=16)

    if save_path:
        if transparent and save_path.lower().endswith(('.jpg', '.jpeg')):
            save_path = os.path.splitext(save_path)[0] + '.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if transparent:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")

    plt.close(fig)
    return fig


def plot_era5_pressure(dataset, output_path='pressure_plot.png'):
    """
    Plot mean sea level pressure field with coastlines and save to file.
    Creates two plots: one with contour lines and one with simple color fill.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Dataset containing mean_sea_level_pressure with dimensions (latitude, longitude)
    output_path : str, optional
        Path to save the output plot (default: 'pressure_plot.png')
    """
    
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Extract time for plot title with hour precision
    time_str = str(dataset.time.values)[:16]  # Get date and hour part (YYYY-MM-DDTHH:MM)
    
    # Extract pressure data (using ERA5_VARIABLE_NAMES)
    pressure_data = dataset[ERA5_VARIABLE_NAMES[2]].values
    
    # Convert from Pa to hPa for better readability
    pressure_hpa = pressure_data / 100.0
    
    # Create meshgrid for contour plot
    lat = dataset.latitude.values
    lon = dataset.longitude.values
    Lon, Lat = np.meshgrid(lon, lat)
    
    # --- Left subplot: Contour plot with lines ---
    ax1 = axes[0]
    
    # Plot pressure field as filled contours
    levels = np.arange(np.nanmin(pressure_hpa), np.nanmax(pressure_hpa), 2)  # 2 hPa intervals
    contourf_plot1 = ax1.contourf(Lon, Lat, pressure_hpa, levels=levels,
                                  cmap='RdYlBu_r', transform=ccrs.PlateCarree(), 
                                  alpha=0.8, extend='both')
    
    # Add contour lines for better readability
    contour_lines = ax1.contour(Lon, Lat, pressure_hpa, levels=levels[::2],
                               colors='black', linewidths=0.5, 
                               transform=ccrs.PlateCarree(), alpha=0.4)
    
    # Add contour labels
    ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.0f')
    
    # Add coastlines and gridlines
    ax1.coastlines()
    gl1 = ax1.gridlines(draw_labels=True, alpha=0.5)
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER
    
    # Add colorbar
    cbar1 = plt.colorbar(contourf_plot1, ax=ax1, orientation='vertical', pad=0.05, shrink=0.8)
    cbar1.set_label('Mean Sea Level Pressure (hPa)', fontsize=12)
    
    # Set title
    ax1.set_title(f'Mean Sea Level Pressure (Contour)\n{time_str}', fontsize=14)
    
    # --- Right subplot: Simple color fill plot ---
    ax2 = axes[1]
    
    # Plot pressure field with pcolormesh (smooth color fill)
    pcolor_plot = ax2.pcolormesh(Lon, Lat, pressure_hpa,
                                 cmap='RdYlBu_r', transform=ccrs.PlateCarree(),
                                 shading='auto', vmin=np.nanmin(pressure_hpa), 
                                 vmax=np.nanmax(pressure_hpa))
    
    # Add coastlines and gridlines
    ax2.coastlines()
    gl2 = ax2.gridlines(draw_labels=True, alpha=0.5)
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.xformatter = LONGITUDE_FORMATTER
    gl2.yformatter = LATITUDE_FORMATTER
    
    # Add colorbar
    cbar2 = plt.colorbar(pcolor_plot, ax=ax2, orientation='vertical', pad=0.05, shrink=0.8)
    cbar2.set_label('Mean Sea Level Pressure (hPa)', fontsize=12)
    
    # Set title
    ax2.set_title(f'Mean Sea Level Pressure (Color Fill)\n{time_str}', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure with high resolution
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_path}")
    
    # Display the plot
    plt.show()

    plt.close(fig)
    
    return fig


def plot_era5_variable_distributions(ds, save_path=None, dpi=150):
    """
    Plot distributions for ERA5 variables (u-component, v-component, pressure) from an xarray Dataset.
    
    Args:
        ds (xarray.Dataset): Dataset containing ERA5 variables
        save_path (str, optional): Path to save the figure. If None, displays the plot
        dpi (int): Resolution of the output figure
    
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    # Get ERA5 variable names from utils
    variables = ERA5_VARIABLE_NAMES
    
    # Validate that all required variables are present
    vars_present = [v for v in variables if v in ds]
    if len(vars_present) != 3:
        missing_vars = [v for v in variables if v not in ds]
        raise ValueError(f"Missing required variables: {missing_vars}")
    
    # Create 1x3 grid for three variables
    fig = plt.figure(figsize=(18, 5))
    
    # Define plot labels for ERA5 variables
    var_labels = {
        '10m_u_component_of_wind': 'U-Component of Wind (m/s)',
        '10m_v_component_of_wind': 'V-Component of Wind (m/s)',
        'mean_sea_level_pressure': 'Mean Sea Level Pressure (Pa)'
    }
    
    for i, var in enumerate(variables, start=1):
        ax = fig.add_subplot(1, 3, i)
        
        # Flatten data and remove NaN values
        data = ds[var].values.flatten()
        data = data[~np.isnan(data)]
        
        # Convert pressure from Pa to hPa for better readability
        if var == 'mean_sea_level_pressure':
            data = data / 100.0
            label = 'Mean Sea Level Pressure (hPa)'
        else:
            label = var_labels.get(var, var)
        
        # Create histogram
        ax.hist(data, bins=100, alpha=0.8, color='skyblue', edgecolor='black', linewidth=0.4)
        ax.set_yscale('log')
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Frequency (log scale)', fontsize=11)
        ax.set_title(label, fontsize=13)
        
        # Add statistics box
        if data.size > 0:
            mean_val = np.mean(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"ERA5 variable distributions saved to: {save_path}")
    
    plt.close(fig)
    
    return fig

def plot_era5_wind(dataset, output_path='wind_components_plot.png'):
    """
    Plot 10m u and v wind components and wind vector field as subplots with coastlines and save to file.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Dataset containing wind components with dimensions (latitude, longitude)
    output_path : str, optional
        Path to save the output plot (default: 'wind_components_plot.png')
    """
    
    # Create figure with three subplots using PlateCarree projection
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), 
                                       subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Extract time for plot title with hour precision
    time_str = str(dataset.time.values)[:16]  # Get date and hour part (YYYY-MM-DDTHH:MM)
    
    # Calculate symmetric colorbar limits for both components (using ERA5_VARIABLE_NAMES)
    u_data = dataset[ERA5_VARIABLE_NAMES[0]].values
    v_data = dataset[ERA5_VARIABLE_NAMES[1]].values
    
    # Find the maximum absolute value across both components
    max_abs_u = np.nanmax(np.abs(u_data))
    max_abs_v = np.nanmax(np.abs(v_data))
    max_abs_both = max(max_abs_u, max_abs_v)
    
    # Set symmetric vmin and vmax
    vmin = -max_abs_both
    vmax = max_abs_both
    
    # Plot u-component (east-west wind) (using ERA5_VARIABLE_NAMES)
    u_plot = dataset[ERA5_VARIABLE_NAMES[0]].plot(
        ax=ax1,
        cmap='RdBu_r',  # Red-Blue colormap (reversed for convention)
        add_colorbar=True,
        cbar_kwargs={'label': 'Wind Speed (m/s)'},
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree()  # Specify data coordinate system
    )
    # Add coastlines and gridlines to the u-component plot
    ax1.coastlines()
    # Show labels only on left and bottom
    gl1 = ax1.gridlines(draw_labels=True, alpha=0.5)
    gl1.top_labels = False    # Hide top labels
    gl1.right_labels = False  # Hide right labels
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER
    ax1.set_title(f'10m U-Component (East-West)\n{time_str}')
    
    # Plot v-component (north-south wind) (using ERA5_VARIABLE_NAMES)
    v_plot = dataset[ERA5_VARIABLE_NAMES[1]].plot(
        ax=ax2,
        cmap='RdBu_r',
        add_colorbar=True,
        cbar_kwargs={'label': 'Wind Speed (m/s)'},
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree()  # Specify data coordinate system
    )
    # Add coastlines and gridlines to the v-component plot
    ax2.coastlines()
    # Show labels only on left and bottom
    gl2 = ax2.gridlines(draw_labels=True, alpha=0.5)
    gl2.top_labels = False    # Hide top labels
    gl2.right_labels = False  # Hide right labels
    gl2.left_labels = False   # Hide left labels (since left subplot already has labels)
    gl2.xformatter = LONGITUDE_FORMATTER
    gl2.yformatter = LATITUDE_FORMATTER
    ax2.set_title(f'10m V-Component (North-South)\n{time_str}')
    
    # Plot wind vector field (third subplot)
    # Calculate wind speed for vector magnitude
    wind_speed = np.sqrt(u_data**2 + v_data**2)
    
    # Create wind vector plot using quiver
    # Subsample the data to avoid overcrowding the plot
    lat = dataset.latitude.values
    lon = dataset.longitude.values
    u = dataset[ERA5_VARIABLE_NAMES[0]].values
    v = dataset[ERA5_VARIABLE_NAMES[1]].values
    
    # Determine subsampling rate based on data density
    subsample = max(1, len(lat) // 20)  # Aim for about 20 vectors in latitude direction
    
    # Create meshgrid for quiver plot
    Lon, Lat = np.meshgrid(lon, lat)
    
    # Plot wind speed as background
    speed_plot = ax3.contourf(Lon, Lat, wind_speed, cmap='viridis', 
                             transform=ccrs.PlateCarree(), alpha=0.8)
    
    # Add wind vectors (subsampled for clarity)
    quiver = ax3.quiver(Lon[::subsample, ::subsample], Lat[::subsample, ::subsample], 
                       u[::subsample, ::subsample], v[::subsample, ::subsample],
                       scale=max_abs_both*20,  # Adjust scale for proper vector sizing
                       color='white', width=0.002, transform=ccrs.PlateCarree())
    
    # Add coastlines and gridlines to the vector field plot
    ax3.coastlines()
    # Show labels only on left and bottom
    gl3 = ax3.gridlines(draw_labels=True, alpha=0.5)
    gl3.top_labels = False    # Hide top labels
    gl3.right_labels = False  # Hide right labels  
    gl3.left_labels = False   # Hide left labels
    gl3.xformatter = LONGITUDE_FORMATTER
    gl3.yformatter = LATITUDE_FORMATTER
    
    # Add colorbar for wind speed
    cbar = plt.colorbar(speed_plot, ax=ax3)
    cbar.set_label('Wind Speed (m/s)')
    
    # Add quiver key (reference vector)
    ax3.quiverkey(quiver, 0.05, 0.05, max_abs_both, 
                 f'{max_abs_both:.1f} m/s', labelpos='E', coordinates='axes')
    
    ax3.set_title(f'10m Wind Vector Field\n{time_str}')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save figure with high resolution
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_path}")
    
    # Display the plot
    plt.show()

    plt.close(fig)
    
    return fig

# Cached point-in-ocean masks for regular data grids, keyed by grid signature
_ocean_contains_cache = {}

