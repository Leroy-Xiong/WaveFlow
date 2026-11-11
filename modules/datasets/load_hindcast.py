import os
from tqdm import tqdm
import h5py
import numpy as np
import xarray as xr
import pandas as pd
import json
from glob import glob
import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from modules.plot.plot_wave import plot_wave_data, plot_wave_data_interpolated, plot_variable_distributions
from modules.utils import HK_REGION, WAVE_VARIABLES, get_random_files
try:
    from modules.plot.satellite_background import add_satellite_background
except ImportError:
    add_satellite_background = None
from modules.datasets.coordinate_utils import export_coordinates, cluster_coordinates


def _compute_hk_hs_stats_single(args):
    """Worker: compute HK-region Hs statistics for a single time step file.

    Returns a tuple (timestamp, hs_mean, hs_max, hs_p95, hs_p99) or None on failure.
    """
    file_path, hk_region = args
    min_lon, max_lon, min_lat, max_lat = hk_region
    try:
        with xr.open_dataset(file_path) as ds:
            lat = ds['lat'].values
            lon = ds['lon'].values
            mask = (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)
            if not np.any(mask):
                return None
            hs = ds['hs'].values[mask]
            time_val = pd.to_datetime(ds['time'].values).to_pydatetime()
        hs_valid = hs[~np.isnan(hs)]
        if hs_valid.size == 0:
            return None
        return (
            time_val,
            float(np.mean(hs_valid)),
            float(np.max(hs_valid)),
            float(np.percentile(hs_valid, 95)),
            float(np.percentile(hs_valid, 99)),
        )
    except Exception:
        return None


class WaveDataProcessor:

    def __init__(self):
        self.base_dir = './data/hindcast'
        self.raw_base_dir = './data/hindcast/raw'
        
        self.hs_file_path = os.path.join(self.raw_base_dir, 'HS_merged_hindcast_holland.mat')
        self.tp_file_path = os.path.join(self.raw_base_dir, 'TP_merged_hindcast_holland.mat')
        self.dm_file_path = os.path.join(self.raw_base_dir, 'DM_merged_hindcast_holland.mat')
        self.elev_file_path = os.path.join(self.raw_base_dir, 'ELEV_merged_hindcast_holland.mat')
        self.node_file_path = os.path.join(self.raw_base_dir, 'node_infos.mat')
        self.processed_dir = os.path.join(self.base_dir, 'processed')
        self.cache_dir = os.path.join(self.base_dir, 'cache')

    @staticmethod
    def read_mat_v73(file_name):
        """
        Read MATLAB v7.3 format .mat file
        """
        try:
            # Use h5py to read the file
            with h5py.File(file_name, 'r') as f:
                print("Variables in the file:")
                print("=" * 50)
                
                # First, print all variable names
                print("All Variable Names:")
                variable_names = []
                
                def collect_variable_names(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        variable_names.append(name)
                
                f.visititems(collect_variable_names)
                
                # Print the list of variable names
                for i, name in enumerate(variable_names, 1):
                    print(f"{i}. {name}")
                
                print("\nVariable Details:")
                print("=" * 50)
                
                # Display detailed information for all variables
                def print_variables(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"Variable: {name}")
                        print(f"  Shape: {obj.shape}")
                        print(f"  Data Type: {obj.dtype}")
                        
                        # Get the data
                        data = obj[()]
                        print(f"  Data Dimension: {data.shape}")
                        
                        # Display data preview
                        if data.size > 0:
                            if data.ndim == 1:
                                print(f"  First 5 values: {data[:5]}")
                            elif data.ndim == 2:
                                print(f"  Data Shape: {data.shape}")
                                # Preview of the first few rows
                                print(f"  Preview of first few rows:\n{data[:3, :5] if data.shape[0] > 3 else data}")
                        print("-" * 30)
                
                f.visititems(print_variables)
                
                return f
                
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    def compute_stats(self):
        """
        Compute min/max/mean/std statistics for HK region wave data for all four variables and visualize the distribution.
            
        Returns:
        dict: Dictionary containing min, max, mean, std and source information for all four variables
        """
        ds = self.create_wave_dataset(hk_only=True)
        
        # Initialize stats dictionary
        stats = {
            "source_hs": self.hs_file_path,
            "source_tp": self.tp_file_path,
            "source_dm": self.dm_file_path,
            "source_elev": self.elev_file_path,
            "source_node": self.node_file_path
        }
        
        # Define the variables to process
        variables = ['hs', 'tp', 'dm', 'elev']
        
        # Process each variable
        for var in variables:
            # Extract data 
            var_data = ds[var].values
            var_data = var_data[~np.isnan(var_data)]  # Remove NaN values
            
            # Calculate stats
            var_min = float(np.min(var_data))
            var_max = float(np.max(var_data))
            var_mean = float(np.mean(var_data))
            var_std = float(np.std(var_data))
            
            # Add to stats dictionary
            stats[f"{var}_min"] = var_min
            stats[f"{var}_max"] = var_max
            stats[f"{var}_mean"] = var_mean
            stats[f"{var}_std"] = var_std

            # Calculate log stats for hs and tp
            if var in ['hs', 'tp']:
                log_data = np.log1p(var_data)
                stats[f"{var}_log_mean"] = float(np.mean(log_data))
                stats[f"{var}_log_std"] = float(np.std(log_data))
            
            print(f"{var.upper()} data statistics (HK region) - Min: {var_min}, Max: {var_max}, Mean: {var_mean:.4f}, Std: {var_std:.4f}")
        
        # Save statistics to JSON file
        stats_file = os.path.join(self.base_dir, "wave_data_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics saved to: {stats_file}")
        
        return stats

    def create_wave_dataset(self, hk_only=False, chunk_size=10000):
        """
        Reads the HS data file and Node info file, combining them into an xarray Dataset.
        
        Parameters:
        hk_only (bool): If True, return only the HK region data. Default is False.
        chunk_size (int): Size of chunks for processing large datasets. Default is 1000.
        """
        start_time = time.time()  # Start timing

        # Check if chunked cache exists
        chunk_files = sorted(glob(os.path.join(self.cache_dir, 'hindcast_chunk_*.nc')))
        if not chunk_files:
            # Initialize variables for outside scope if needed, though they are scoped inside 'with' now
            lat = None
            lon = None
            hs_data = None
            tp_data = None
            dm_data = None
            elev_data = None
            n_nodes = 0
            
            # 1. Process Node File
            
            # Use 'with' to open the file again safely and guarantee the handle is valid within this block
            try:
                with h5py.File(self.node_file_path, 'r') as f_node:
                    # Extract Lat/Lon and squeeze (guaranteed to work since f_node is fresh and open)
                    lat = np.squeeze(f_node['node_LAT'][()])
                    lon = np.squeeze(f_node['node_LON'][()])
                    n_nodes = lat.shape[0]
            except Exception as e:
                # Catch file-specific errors (e.g., KeyError if 'node_LAT' is wrong, or FileNotFoundError)
                raise ValueError(f"Failed to load node file or extract coordinates: {e}")


            # 2. Process Multiple Data Files

            # Use 'with' to open the HS file safely
            try:
                print("Reading HS file:")
                with h5py.File(self.hs_file_path, 'r') as f_hs:
                    # Print all variable names to confirm the correct key
                    print("HS File Variables:")
                    f_hs.visit(print)
                    # Extract Wave Height Data
                    # Note: This loads the entire dataset into memory.
                    hs_data = f_hs['NEW_HS'][()]
                    
                print("\nReading TP file:")
                # Use 'with' to open the TP file safely
                with h5py.File(self.tp_file_path, 'r') as f_tp:
                    # Print all variable names to confirm the correct key
                    print("TP File Variables:")
                    f_tp.visit(print)
                    # Extract Peak Period Data
                    tp_data = f_tp['NEW_TP'][()]
                    
                print("\nReading DM file:")
                # Use 'with' to open the DM file safely
                with h5py.File(self.dm_file_path, 'r') as f_dm:
                    # Print all variable names to confirm the correct key
                    print("DM File Variables:")
                    f_dm.visit(print)
                    # Extract Directional Mean Data
                    dm_data = f_dm['NEW_DM'][()]
                    
                print("\nReading ELEV file:")
                # Use 'with' to open the ELEV file safely
                with h5py.File(self.elev_file_path, 'r') as f_elev:
                    # Print all variable names to confirm the correct key
                    print("ELEV File Variables:")
                    f_elev.visit(print)
                    # Extract Elevation Data
                    elev_data = f_elev['NEW_ELEV'][()]
            except Exception as e:
                raise ValueError(f"Failed to load data or extract variables: {e}")
                    
            
            # 3. Check Dimensions and Construct Dataset
            
            # We expect shapes like (Time, Node) or (Node, Time)
            # The error logic is now robust as the data is guaranteed to be loaded if we reach here.
            if hs_data.shape[1] == n_nodes:
                # Shape is (Time, Node)
                dims = ('time', 'node')
                n_time_steps = hs_data.shape[0]
            elif hs_data.shape[0] == n_nodes:
                # Shape is (Node, Time)
                dims = ('node', 'time')
                n_time_steps = hs_data.shape[1]
            else:
                raise ValueError(f"Dimension mismatch: Data shape {hs_data.shape} does not match node count {n_nodes}")

            # Create hourly time coordinates starting from 1970-01-01 00:00:00
            start_time_coord = np.datetime64('1970-01-01T00:00:00')
            coords_time = start_time_coord + np.arange(n_time_steps) * np.timedelta64(1, 'h')

            print(f"\n--- Creating xarray Dataset ---")
            # 4. Construct the xarray Dataset
            ds = xr.Dataset(
                data_vars={
                    "hs": (dims, hs_data, {"units": WAVE_VARIABLES['hs']['units'], "long_name": WAVE_VARIABLES['hs']['long_name']}),
                    "tp": (dims, tp_data, {"units": WAVE_VARIABLES['tp']['units'], "long_name": WAVE_VARIABLES['tp']['long_name']}),
                    "dm": (dims, dm_data, {"units": WAVE_VARIABLES['dm']['units'], "long_name": WAVE_VARIABLES['dm']['long_name']}),
                    "elev": (dims, elev_data, {"units": WAVE_VARIABLES['elev']['units'], "long_name": WAVE_VARIABLES['elev']['long_name']})
                },
                coords={
                    "lat": (("node",), lat, {"standard_name": "latitude", "units": "degrees_north"}),
                    "lon": (("node",), lon, {"standard_name": "longitude", "units": "degrees_east"}),
                    "time": (("time",), coords_time)
                },
                attrs={
                    "description": "Merged Hindcast Wave Data converted from MATLAB v7.3",
                    "source_hs": self.hs_file_path,
                    "source_tp": self.tp_file_path,
                    "source_dm": self.dm_file_path,
                    "source_elev": self.elev_file_path,
                    "source_node": self.node_file_path
                }
            )
            
            # Save to NetCDF cache file (always save the complete dataset)
            os.makedirs(self.cache_dir, exist_ok=True)

            print("Saving as chunked files...")
            
            # If single file fails, save as chunks
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Determine the dimension along which to chunk (time)
            time_dim = ds.dims['time']
            num_chunks = max(1, (time_dim + chunk_size - 1) // chunk_size)  # Ceiling division
            
            print(f"Splitting dataset into {num_chunks} chunks of size ~{chunk_size}")
            
            # Split along time dimension and save chunks
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, time_dim)
                
                # Extract chunk
                ds_chunk = ds.isel(time=slice(start_idx, end_idx))
                
                # Save chunk
                chunk_filename = os.path.join(self.cache_dir, f'hindcast_chunk_{i:04d}.nc')
                ds_chunk.to_netcdf(chunk_filename, engine='scipy')
                print(f"Saved chunk {i+1}/{num_chunks} to: {chunk_filename}")
            
            print(f"Saved dataset as {num_chunks} chunks to: {self.cache_dir}")
            
        print(f"Loading data from {len(chunk_files)} chunked cache files")
        # Try to open multifile dataset
        ds = xr.open_mfdataset(chunk_files, combine='by_coords')
        
        # Filter to HK region only if requested
        if hk_only:
            min_lon, max_lon, min_lat, max_lat = HK_REGION
            lon_condition = (ds.lon >= min_lon) & (ds.lon <= max_lon)
            lat_condition = (ds.lat >= min_lat) & (ds.lat <= max_lat)
            mask = (lon_condition & lat_condition).compute()

            ds = ds.where(mask, drop=True).compute()
            
        # Print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print("Successfully created xarray object.")
        print(f"Function execution time: {execution_time:.2f} seconds")
        print(ds)
        
        return ds

    def split_dataset_by_time(self, ds, chunk_size=1000):
        """
        Split dataset by time and export as separate compressed files
        
        Parameters:
        ds: Original xarray Dataset
        output_dir: Output directory path
        chunk_size: Number of time points to process at once (for memory management)
        """
        # Create output directory
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Get all time points
        times = ds.time.values
        total_times = len(times)
        
        print(f"Processing {total_times} time points...")
        print(f"Dataset size: {ds.nbytes / 1024**3:.2f} GB")
        print(f"Number of nodes: {len(ds.node)}")
        
        # Process in chunks to avoid memory issues
        for i in tqdm(range(0, total_times, chunk_size)):
            end_idx = min(i + chunk_size, total_times)
            time_chunk = times[i:end_idx]
            
            # Extract data for current time chunk
            ds_chunk = ds.sel(time=time_chunk).compute()
            
            # Process each time point individually
            for j, time_point in enumerate(time_chunk):
                # Select data for single time point
                ds_single_time = ds_chunk.sel(time=time_point)
                
                # Convert time to filename-friendly format (hour precision)
                time_str = str(time_point)[:13].replace(':', '').replace(' ', 'T').replace('-', '')
                filename = f"wave_{time_str}.nc"
                filepath = os.path.join(self.processed_dir, filename)
                
                # Set compression encoding for all 4 variables
                encoding = {
                    'hs': {'zlib': True, 'complevel': 6},
                    'tp': {'zlib': True, 'complevel': 6},
                    'dm': {'zlib': True, 'complevel': 6},
                    'elev': {'zlib': True, 'complevel': 6},
                    'lat': {'zlib': True, 'complevel': 6},
                    'lon': {'zlib': True, 'complevel': 6}
                }
                
                # Export with netCDF4 engine explicitly specified
                ds_single_time.to_netcdf(filepath, engine='scipy')
        
        print(f"Processing completed! Files saved to: {self.processed_dir}")
    
    def load_and_split(self):
        ds = self.create_wave_dataset()
        self.split_dataset_by_time(ds, chunk_size=1000)

    def visualize_samples(self, timestamps: list = None, n_samples: int = 10, seed: int = 42,
                           save_dir_hk: str = './output/plot/data/hk/random_samples_hk',
                           save_dir_full: str = './output/plot/data/hk/random_samples'):
        """
        Visualize wave samples for specific or random timestamps.

        Supports three modes:
        - ``timestamps`` given → visualize those exact timestamps
        - ``timestamps`` is None → randomly select ``n_samples`` files

        For each timestamp, generates:
        - Full-domain scatter plots (4 variables)
        - HK-region zoom plots (elev interpolated onto the model ocean
          grid, other variables as scatter nodes, satellite bg)
        - HS + elev contour overlay (HK region only)

        Args:
            timestamps: List of YYYYMMDDTHH strings, or None for random mode.
            n_samples: Number of random samples (only when timestamps is None).
            seed: Random seed for reproducibility.
            save_dir_hk: Output directory for HK-region zoom + contour plots.
            save_dir_full: Output directory for full-domain plots.
        """
        variables = ['hs', 'tp', 'dm', 'elev']

        if timestamps is not None:
            files = [os.path.join(self.processed_dir, f"wave_{ts}.nc") for ts in timestamps]
            for f in files:
                if not os.path.exists(f):
                    print(f"  Warning: file not found: {f}")
            files = [f for f in files if os.path.exists(f)]
            print(f"Visualizing {len(files)} specified timestamps:")
        else:
            files = get_random_files(self.processed_dir, n=n_samples, seed=seed)
            print(f"Randomly selected {len(files)} files:")

        for f in files:
            print(f"  {f}")

        for file in tqdm(files, desc="Visualizing samples"):
            ds = xr.load_dataset(file)
            base_filename = os.path.basename(file)[:-3]  # strip ".nc"

            # Full-domain plots
            for var in variables:
                plot_wave_data(
                    ds, variable=var,
                    save_path=os.path.join(save_dir_full, f"{base_filename}_{var}.jpg"),
                )

            # HK-region zoom plots: elev interpolated onto the model ocean
            # grid, the other variables as scatter nodes (transparent PNG)
            for var in variables:
                save_path = os.path.join(save_dir_hk, f"{base_filename}_hk_{var}.png")
                if var == 'elev':
                    plot_wave_data_interpolated(
                        ds, variable=var,
                        save_path=save_path,
                        region=HK_REGION,
                        use_satellite_background=True,
                        direction_var='dm',
                        cmap='RdBu_r',
                        symmetric_clim=True,
                        figsize=(8, 5.6),
                        cbar_orientation='horizontal',
                        gridline_spacing=(0.3, 0.2),
                        show_top_labels=False,
                        transparent=True,
                    )
                else:
                    plot_wave_data(
                        ds, variable=var,
                        save_path=save_path,
                        region=HK_REGION,
                        show_cities=False,
                        show_nodes=False,
                        use_high_res_coastline=True,
                        use_satellite_background=True,
                        figsize=(8, 5.6),
                        scatter_size=8,
                        show_top_labels=False,
                        show_bottom_labels=True,
                        cbar_orientation='horizontal',
                        gridline_spacing=(0.3, 0.2),
                        transparent=True,
                    )

            # HS + elev contour overlay
            self._plot_hs_elev_contour(
                ds,
                save_path=os.path.join(save_dir_hk, f"{base_filename}_hk_hs_elev_contour.jpg"),
                region=HK_REGION,
            )

    def _plot_hs_elev_contour(self, ds, save_path, region=None):
        """
        Plot HS scatter + elev tricontour overlay (ocean-clipped).

        HS is drawn as colored scatter points. Elev contours are computed
        from Delaunay triangulation of the scattered nodes and clipped to
        the model's ocean polygon so they never appear over land.

        Args:
            ds: xarray Dataset with 'hs' and 'elev' variables plus lon/lat coords.
            save_path: Output path for the saved JPG figure.
            region: [min_lon, max_lon, min_lat, max_lat] or None.
        """
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path
        from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
        from modules.utils import WAVE_VARIABLES
        from modules.plot.satellite_background import _load_ocean_geometry

        # ---- shapely → matplotlib Path helper ----
        def _ocean_to_clip_path(ocean):
            """Convert ocean shapely Polygon to a matplotlib PathPatch."""
            if isinstance(ocean, MultiPolygon):
                # Use the largest polygon (main ocean body)
                ocean = max(ocean.geoms, key=lambda p: p.area)
            ext = np.asarray(ocean.exterior.coords)
            verts = [ext]
            codes = [[Path.MOVETO] + [Path.LINETO] * (len(ext) - 1)]
            for interior in ocean.interiors:
                ring = np.asarray(interior.coords)
                if len(ring) < 3:
                    continue
                verts.append(ring)
                codes.append([Path.MOVETO] + [Path.LINETO] * (len(ring) - 1))
            path = Path(np.vstack(verts), np.concatenate(codes))
            return PathPatch(path, transform=ccrs.PlateCarree(),
                             facecolor='none', edgecolor='none')

        # ---- Filter to region ----
        ds_plot = ds.copy()
        if region is not None:
            min_lon, max_lon, min_lat, max_lat = region
            ds_plot = ds_plot.where(
                (ds_plot.lon >= min_lon) & (ds_plot.lon <= max_lon) &
                (ds_plot.lat >= min_lat) & (ds_plot.lat <= max_lat),
                drop=True,
            )

        hs_vals = ds_plot['hs'].values
        elev_vals = ds_plot['elev'].values
        lons = ds_plot.lon.values
        lats = ds_plot.lat.values

        # Mask out NaN
        mask = ~np.isnan(hs_vals) & ~np.isnan(elev_vals)
        if not np.any(mask):
            print(f"  Warning: all nodes NaN, skipping contour plot")
            return

        hs_vals = hs_vals[mask]
        elev_vals = elev_vals[mask]
        lons = lons[mask]
        lats = lats[mask]

        hs_label = WAVE_VARIABLES['hs']['plot_label']

        # ---- Delaunay triangulation on scattered nodes ----
        tri = mtri.Triangulation(lons, lats)

        # ---- Figure ----
        extent = region if region else [lons.min(), lons.max(), lats.min(), lats.max()]
        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Satellite + coastline
        if add_satellite_background is not None:
            add_satellite_background(ax, extent)
        else:
            ax.add_feature(cfeature.GSHHSFeature(scale='full', levels=[1]),
                           facecolor='whitesmoke', edgecolor='black', linewidth=0.6, zorder=4)

        # ---- HS scatter (base layer) ----
        hs_vmin = float(np.nanmin(hs_vals))
        hs_vmax = float(np.nanmax(hs_vals))
        sc = ax.scatter(lons, lats, c=hs_vals, cmap='turbo',
                         s=15, edgecolors='none', transform=ccrs.PlateCarree(),
                         vmin=hs_vmin, vmax=hs_vmax, zorder=5, alpha=1.0)

        cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.04, aspect=30)
        cbar.set_label(hs_label, fontsize=11)
        cbar.ax.tick_params(labelsize=11)

        # ---- Wave direction arrows (dm, nautical convention: direction FROM) ----
        if 'dm' in ds_plot:
            dm_vals = ds_plot['dm'].values[mask]
            # Direction waves travel TO = FROM + 180°
            dm_rad = np.deg2rad(dm_vals + 180)
            u_dm = np.sin(dm_rad)   # eastward component
            v_dm = np.cos(dm_rad)   # northward component
            # Downsample: every N-th node
            step = max(1, len(lons) // 400)
            ax.quiver(lons[::step], lats[::step],
                       u_dm[::step], v_dm[::step],
                       transform=ccrs.PlateCarree(),
                       color='black', alpha=0.55, width=0.003,
                       scale=30, zorder=12, headwidth=4, headlength=5)

        # ---- Ocean clip path for contours ----
        ocean_clip = None
        try:
            ocean_polygon, _ = _load_ocean_geometry(extent)
            ocean_clip = _ocean_to_clip_path(ocean_polygon)
            ax.add_patch(ocean_clip)  # must be added to axes for clip to work
        except Exception:
            pass

        # ---- Elev tricontour ----
        elev_vmin = float(np.nanpercentile(elev_vals, 1))
        elev_vmax = float(np.nanpercentile(elev_vals, 99))
        elev_levels = np.linspace(elev_vmin, elev_vmax, 14)
        elev_levels = np.unique(np.round(elev_levels, 2))
        if len(elev_levels) >= 2:
            # Shadow + white lines, clipped to ocean
            cs_shadow = ax.tricontour(tri, elev_vals, levels=elev_levels,
                                       colors='black', linewidths=1.8,
                                       transform=ccrs.PlateCarree(), zorder=20, alpha=0.35)
            cs = ax.tricontour(tri, elev_vals, levels=elev_levels,
                                colors='white', linewidths=1.0,
                                transform=ccrs.PlateCarree(), zorder=21)
            ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f',
                       colors='white', inline_spacing=10)

            if ocean_clip is not None:
                cs_shadow.set_clip_path(ocean_clip)
                cs.set_clip_path(ocean_clip)

        # ---- Title ----
        timestamp = os.path.basename(save_path).split('_hk_')[0].replace('wave_', '')
        ax.set_title(f'HS + Elev Contours\n{timestamp}  |  HK Region  |  '
                      f'Elev: [{elev_vmin:.2f}, {elev_vmax:.2f}]',
                      fontsize=13, fontweight='bold')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def plot_hk_hs_timeseries(
        self,
        start_time='20210101T00',
        end_time='20221231T23',
        save_path='./output/plot/hk_hs_timeseries/hk_hs_timeseries_2021_2022.png',
        csv_save_path=None,
        num_workers=16,
        typhoon_events=None,
        model_rmse_csvs=None,
        rmse_column='hs_rmse',
        normalize_rmse=False,
    ):
        """
        Compute and plot the significant wave height (Hs) time series over the
        Hong Kong region from per-timestep hindcast files.

        For each hourly file in `self.processed_dir` between `start_time` and
        `end_time`, this aggregates Hs across nodes inside `HK_REGION` and
        records mean / 95th / 99th percentile / max. Files are read in parallel.

        Parameters
        ----------
        start_time, end_time : str
            Time range in 'YYYYMMDDTHH' format (inclusive).
        save_path : str
            Path to save the time-series figure (PNG).
        csv_save_path : str, optional
            If provided, the aggregated time series is also saved as CSV.
        num_workers : int
            Number of parallel processes for reading per-timestep files.
        typhoon_events : list of dict, optional
            Each dict like {'name': 'Lionrock', 'time': '20211008T00'} to mark
            on the figure. Defaults to a curated 2021-2022 list affecting HK.
        model_rmse_csvs : dict, optional
            Mapping from label -> CSV path. CSV must contain `timestamp`
            (format 'YYYYMMDDTHH') and `rmse_column`. Each is overlaid on a
            secondary right-side y-axis as a thin line. Use this to compare
            multiple atmospheric inputs / lead times against the wave climate.
        rmse_column : str
            Column name to use for the model RMSE (default 'hs_rmse').
        """
        # Build hourly file list
        start_dt = datetime.strptime(start_time, '%Y%m%dT%H')
        end_dt = datetime.strptime(end_time, '%Y%m%dT%H')
        if start_dt > end_dt:
            raise ValueError("start_time must be <= end_time")

        cur = start_dt
        file_args = []
        while cur <= end_dt:
            ts = cur.strftime('%Y%m%dT%H')
            fp = os.path.join(self.processed_dir, f'wave_{ts}.nc')
            if os.path.exists(fp):
                file_args.append((fp, HK_REGION))
            cur += timedelta(hours=1)

        if not file_args:
            raise FileNotFoundError(
                f"No wave files found in {self.processed_dir} for {start_time}..{end_time}"
            )

        print(f"Aggregating HK Hs from {len(file_args)} hourly files using "
              f"{num_workers} workers...")

        results = []
        if num_workers and num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                for r in tqdm(
                    ex.map(_compute_hk_hs_stats_single, file_args, chunksize=64),
                    total=len(file_args),
                    desc='HK Hs aggregation',
                ):
                    if r is not None:
                        results.append(r)
        else:
            for args in tqdm(file_args, desc='HK Hs aggregation'):
                r = _compute_hk_hs_stats_single(args)
                if r is not None:
                    results.append(r)

        if not results:
            raise RuntimeError("No valid HK Hs samples were computed.")

        results.sort(key=lambda x: x[0])
        df = pd.DataFrame(
            results,
            columns=['time', 'hs_mean', 'hs_max', 'hs_p95', 'hs_p99'],
        ).set_index('time')

        if csv_save_path is not None:
            os.makedirs(os.path.dirname(csv_save_path) or '.', exist_ok=True)
            df.to_csv(csv_save_path)
            print(f"Saved time-series CSV to: {csv_save_path}")

        # Default typhoon markers (HK-affecting events in 2021-2022).
        if typhoon_events is None:
            typhoon_events = [
                {'name': 'Koguma',     'time': '20210613T00'},
                {'name': 'Cempaka',    'time': '20210720T00'},
                {'name': 'Lupit',      'time': '20210805T00'},
                {'name': 'Lionrock',   'time': '20211008T12'},
                {'name': 'Kompasu',    'time': '20211013T12'},
                {'name': 'Chaba',      'time': '20220702T00'},
                {'name': 'Mulan',      'time': '20220810T00'},
                {'name': 'Ma-on',      'time': '20220825T00'},
                {'name': 'Hinnamnor',  'time': '20220903T00'},
                {'name': 'Nalgae',     'time': '20221103T00'},
            ]

        fig, ax = plt.subplots(figsize=(18, 7))
        ax.plot(df.index, df['hs_p99'], color='#0072B2', linewidth=0.9,
                alpha=0.9, label='Hs p99 (HK nodes)')

        # Overlay model RMSE curves on a twin y-axis if provided.
        ax2 = None
        rmse_lines = []
        rmse_labels = []
        if model_rmse_csvs:
            ax2 = ax.twinx()

            # ERA5 (and other non-ENS sources) stay solid black; ENS lead
            # times get a viridis gradient so the eye reads 12h -> 72h
            # monotonically.
            ens_labels_in_order = [
                lab for lab in model_rmse_csvs.keys()
                if lab.lower().startswith('ens')
            ]
            n_ens = max(len(ens_labels_in_order), 1)
            ens_cmap = plt.get_cmap('viridis')
            ens_color_map = {}
            for j, lab in enumerate(ens_labels_in_order):
                # Sample within [0.15, 0.85] so endpoints stay readable.
                t = 0.15 + 0.70 * (j / max(n_ens - 1, 1))
                ens_color_map[lab] = ens_cmap(t)

            for i, (label, csv_path) in enumerate(model_rmse_csvs.items()):
                if not os.path.exists(csv_path):
                    print(f"[warn] RMSE CSV missing for '{label}': {csv_path}")
                    continue
                df_r = pd.read_csv(csv_path)
                if rmse_column not in df_r.columns or 'timestamp' not in df_r.columns:
                    print(f"[warn] CSV {csv_path} missing required columns "
                          f"(need 'timestamp' and '{rmse_column}')")
                    continue
                df_r['datetime'] = pd.to_datetime(
                    df_r['timestamp'].astype(str), format='%Y%m%dT%H',
                    errors='coerce'
                )
                df_r = (df_r.dropna(subset=['datetime', rmse_column])
                            .sort_values('datetime'))
                df_r = df_r[(df_r['datetime'] >= start_dt) &
                            (df_r['datetime'] <= end_dt)]
                if df_r.empty:
                    print(f"[warn] No RMSE rows in range for '{label}'")
                    continue
                is_ens = label.lower().startswith('ens')
                color = ens_color_map[label] if is_ens else '#000000'
                lw = 0.9 if is_ens else 1.1

                y_vals = df_r[rmse_column]
                legend_label = f'{label} Hs RMSE'
                if normalize_rmse:
                    sigma = float(y_vals.std())
                    if not np.isfinite(sigma) or sigma == 0:
                        print(f"[warn] RMSE std is 0/NaN for '{label}'; skipping normalization")
                    else:
                        y_vals = y_vals / sigma
                        legend_label = f'{label} Hs RMSE (std)'

                line, = ax2.plot(
                    df_r['datetime'], y_vals,
                    color=color, linewidth=lw, linestyle='-', alpha=0.9,
                    label=legend_label,
                )
                rmse_lines.append(line)
                rmse_labels.append(legend_label)

            if normalize_rmse:
                ax2.set_ylabel('Model Hs RMSE / σ (standardized)',
                               color='#444444')
            else:
                ax2.set_ylabel('Model Hs RMSE (m, ensemble mean vs GT)',
                               color='#444444')
            ax2.tick_params(axis='y', colors='#444444')
            ax2.grid(False)
            # Compress the RMSE traces into the lower portion of the figure,
            # but leave them tall enough to read individual peaks.
            ax2.set_ylim(0, 10.0 if normalize_rmse else 2.0)

        # Lift the Hs p99 trace into the upper portion of the figure by
        # extending the primary y-axis below zero. Keep the upper bound just
        # above the data max so typhoon labels still fit.
        hs_p99_top = float(df['hs_p99'].max())
        ax.set_ylim(-3.0, hs_p99_top * 1.05)

        # Annotate typhoon events
        ymax_left = float(df['hs_p99'].max())
        for ev in typhoon_events:
            try:
                t = datetime.strptime(ev['time'], '%Y%m%dT%H')
            except ValueError:
                continue
            if t < start_dt or t > end_dt:
                continue
            ax.axvline(t, color='gray', linestyle='--',
                       linewidth=0.7, alpha=0.55)
            ax.text(t, ymax_left * 1.01, ev['name'], rotation=90,
                    va='bottom', ha='right', fontsize=8, color='dimgray')

        ax.set_title(
            f"Hong Kong Significant Wave Height (Hs) Hindcast — "
            f"{start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}"
            + ("  |  with model ensemble-mean RMSE overlay"
               if model_rmse_csvs else "")
        )
        ax.set_xlabel('Time (UTC)')
        ax.set_ylabel('Significant Wave Height Hs (m)')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # Merge legends from both axes.
        handles1, labels1 = ax.get_legend_handles_labels()
        if ax2 is not None:
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(handles1 + handles2, labels1 + labels2,
                      loc='upper left', framealpha=0.85, ncol=2, fontsize=8)
        else:
            ax.legend(loc='upper left', framealpha=0.85)
        fig.autofmt_xdate()
        fig.tight_layout()

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved HK Hs time-series figure to: {save_path}")

        return df

    def visualize_hs_distribution(self, save_path=None):
        """
        Visualize distributions for HS, TP, DM and ELEV across all time steps (HK region by default).

        This is backward compatible with the previous `visualize_hs_distribution` name but now
        produces a 2x2 plot showing distributions for `hs`, `tp`, `dm`, and `elev`.

        Parameters:
        save_path (str, optional): Path to save the figure. If None, the figure is returned (and not saved).
        """
        # Load the full dataset for HK region
        ds = self.create_wave_dataset(hk_only=True)

        # Plot distributions for the four variables
        fig = plot_variable_distributions(ds, variables=['hs', 'tp', 'dm', 'elev'], save_path=save_path)

        return fig

    def export_hk_coordinates(self, save_path=None):
        """
        Export Hong Kong region coordinates (lat, lon) as numpy array.
        
        The output shape is [N, 2] where N is the number of nodes in HK region,
        and the 2 columns are [latitude, longitude].
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the .npy file. 
            If None, saves to './data/hindcast/hk_coordinates.npy'
        
        Returns:
        --------
        np.ndarray: Array of shape [N, 2] containing [lat, lon] coordinates
        """
        # Load HK region dataset
        ds = self.create_wave_dataset(hk_only=True)
        
        # Extract latitude and longitude
        lat = ds.lat.values
        lon = ds.lon.values
        
        # Set default save path if not provided
        if save_path is None:
            save_path = os.path.join(self.base_dir, 'hk_coordinates.npy')
        
        # Use the utility function to export coordinates
        coordinates = export_coordinates(
            lat=lat,
            lon=lon,
            save_path=save_path,
            description="Hong Kong"
        )
        
        return coordinates
    
    def cluster_hk_coordinates(self, n_clusters=256, coordinate_path=None, save_path=None, 
                              vis_save_path=None, method='recursive_bisection'):
        """
        Cluster Hong Kong coordinates into n_clusters groups with balanced node counts.
        
        This method reads the HK coordinates and performs clustering to ensure each cluster
        has approximately equal number of nodes. The clustering considers spatial proximity
        while maintaining balance.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create. Default is 256.
        coordinate_path : str, optional
            Path to load the coordinates .npy file.
            If None, uses './data/hindcast/hk_coordinates.npy'
        save_path : str, optional
            Path to save the clustering results.
            If None, saves to './data/hindcast/hk_clusters.npz'
        vis_save_path : str, optional
            Path to save the visualization plot.
            If None, auto-generates based on save_path (replaces .npz with _visualization.png)
            Set to False to skip visualization
        method : str
            Clustering method to use. Options:
            - 'recursive_bisection': Recursive bisection for strictly balanced, spatially contiguous clusters
            - 'balanced_kmeans': K-means with balanced reassignment
            - 'kmeans': Standard K-means (may be unbalanced)
        
        Returns:
        --------
        dict: Dictionary containing:
            - 'labels': Cluster labels for each node [N,]
            - 'centers': Cluster center coordinates [n_clusters, 2]
            - 'cluster_sizes': Number of nodes in each cluster [n_clusters,]
            - 'coordinates': Original coordinates [N, 2]
            - 'n_clusters': Number of clusters (int)
            - 'method': Clustering method used (str)
            - 'max_points': Maximum number of points in any patch (int)
            - 'local_coords_template': Pre-calculated local coordinates [n_clusters, max_points, 2]
            - 'mask_template': Pre-calculated validity mask [n_clusters, max_points]
        """
        # Set default coordinate path if not provided
        if coordinate_path is None:
            coordinate_path = os.path.join(self.base_dir, 'hk_coordinates.npy')
        
        # Load or generate coordinates
        if not os.path.exists(coordinate_path):
            print(f"Coordinates file not found at {coordinate_path}")
            print("Generating coordinates first...")
            coordinates = self.export_hk_coordinates()
        else:
            coordinates = np.load(coordinate_path)
            print(f"Loaded coordinates from: {coordinate_path}")
            print(f"Shape: {coordinates.shape}")
        
        # Set default save path if not provided
        if save_path is None:
            save_path = os.path.join(self.base_dir, 'hk_clusters.npz')
        
        # Use the utility function to perform clustering
        result = cluster_coordinates(
            coordinates=coordinates,
            n_clusters=n_clusters,
            method=method,
            save_path=save_path,
            vis_save_path=vis_save_path
        )
        
        return result
