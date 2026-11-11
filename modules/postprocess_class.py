"""
Post-processing class for generated wave samples.
Includes visualization and analysis tools.
"""
import numpy as np
import pandas as pd
import xarray as xr
import json
import os
import glob
import random
import concurrent.futures
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tqdm import tqdm
from modules.plot.plot_wave import plot_wave_data, plot_wave_data_global
from modules.utils import HK_REGION


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

class PostProcess:
    """
    Post-processing class for generated wave samples.
    """
    
    def __init__(self, sample_dir, region_mode='hk', gt_file_format='nc'):
        """
        Initialize PostProcess class.
        
        Args:
            sample_dir: Directory containing sample .npz files
            region_mode: 'global' or 'hk' - determines which plotting function to use and coordinate file
            gt_file_format: Ground truth file format - 'nc' for NetCDF or 'npz' for NPZ files
        """
        self.sample_dir = sample_dir
        self.region_mode = region_mode
        self.gt_file_format = gt_file_format.lower()
        
        # Validate file_format
        if self.gt_file_format not in ['nc', 'npz', 'npy']:
            raise ValueError(f"gt_file_format must be 'nc', 'npz' or 'npy', got '{gt_file_format}'")

        # Determine coordinate file and ground truth path based on region_mode
        if region_mode == 'global':
            coord_file = "./data/global_hindcast/global_coordinates.npy"
            # Global data uses different path and filename format
            self.gt_data_dir = "./data/global_hindcast"
            self.gt_filename_prefix = "global_hindcast"
            # New NPZ format uses standard variable keys (hs, mwd, ssh)
            self.gt_var_mapping = {}
            # Per-timestamp NPY GT product (global-only): [N, C] float32 whose
            # column layout is declared by npy_files_4var/wave_columns.json
            # (see modules/preprocess/convert_npz4var_to_npy.py).
            self._npy_columns = None
            if self.gt_file_format == 'npy':
                cols_path = os.path.join(self.gt_data_dir, 'npy_files_4var',
                                         'wave_columns.json')
                with open(cols_path) as fh:
                    self._npy_columns = list(json.load(fh)['columns'])
        else:
            coord_file = "./data/hindcast/hk_coordinates.npy"
            # HK data path — use NPZ dir when gt_file_format is npz
            if self.gt_file_format == 'npz':
                self.gt_data_dir = "./data/hindcast/processed_npz"
            else:
                self.gt_data_dir = "./data/hindcast/processed"
            self.gt_filename_prefix = "wave"
            # HK data uses standard variable names
            self.gt_var_mapping = {}
        self.coord_file = coord_file
        
        # Load coordinates
        print(f"Loading coordinates from {coord_file}...")
        self.coords = np.load(coord_file)
        self.lons = self.coords[:, 1]
        self.lats = self.coords[:, 0]
        self.num_nodes = self.coords.shape[0]
        print(f"Loaded {self.num_nodes} nodes")
        
        # Calculate region from coordinates
        self.region = [
            self.lons.min(),  # min_lon
            self.lons.max(),  # max_lon
            self.lats.min(),  # min_lat
            self.lats.max()   # max_lat
        ]
        print(f"Calculated region: lon [{self.region[0]:.2f}, {self.region[1]:.2f}], lat [{self.region[2]:.2f}, {self.region[3]:.2f}]")
        
        # Find all sample files
        sample_pattern = os.path.join(sample_dir, "sample_*.npz")
        self.sample_files = sorted(glob.glob(sample_pattern))
        
        if len(self.sample_files) == 0:
            raise ValueError(f"No sample files found in {sample_dir}")
        
        print(f"Found {len(self.sample_files)} sample files")
    
    def _get_gt_file_path(self, timestamp, data_dir=None):
        """
        Construct ground truth file path based on region mode and timestamp.
        
        Args:
            timestamp: Timestamp string (e.g., '20210101T09')
            data_dir: Optional override for data directory
            
        Returns:
            str: Full path to ground truth file
        """
        if data_dir is None:
            data_dir = self.gt_data_dir
            
        # Determine subdirectory based on region mode and file format
        if self.region_mode == 'global':
            if self.gt_file_format == 'npz':
                subdir = "npz_files"
            elif self.gt_file_format == 'npy':
                subdir = "npy_files_4var"
            else:
                subdir = "nc_files"
        else:
            subdir = ""
            
        # Format timestamp based on region mode
        if self.region_mode == 'global':
            # Global format: global_hindcast_YYYYMMDD_HH.npz
            # timestamp is '20210101T09', convert to '20210101_09'
            date_part = timestamp[:8]  # YYYYMMDD
            hour_part = timestamp[9:11]  # HH
            formatted_timestamp = f"{date_part}_{hour_part}"
            filename = f"{self.gt_filename_prefix}_{formatted_timestamp}.{self.gt_file_format}"
            year = date_part[:4]
            return os.path.join(data_dir, subdir, year, filename)
        else:
            # HK format: wave_YYYYMMDDTHH.nc
            filename = f"{self.gt_filename_prefix}_{timestamp}.{self.gt_file_format}"

        return os.path.join(data_dir, subdir, filename)
    
    def _load_ground_truth_nc(self, gt_file, variable=None):
        """
        Load ground truth data from NetCDF file.
        
        Args:
            gt_file: Path to the NetCDF file
            variable: Variable name to extract (optional, returns full dataset if None)
            
        Returns:
            xarray.Dataset or numpy array: Ground truth data
        """
        ds_gt = xr.load_dataset(gt_file)
        
        # Filter to HK region to match samples
        if self.region_mode != 'global':
            min_lon, max_lon, min_lat, max_lat = HK_REGION
            ds_gt = ds_gt.where(
                (ds_gt.lon >= min_lon) & (ds_gt.lon <= max_lon) & 
                (ds_gt.lat >= min_lat) & (ds_gt.lat <= max_lat), 
                drop=True
            )
        
        if variable is not None:
            return ds_gt[variable].values.squeeze()
        return ds_gt
    
    def _load_ground_truth_npz(self, gt_file, variable=None):
        """
        Load ground truth data from NPZ file.
        
        Args:
            gt_file: Path to the NPZ file
            variable: Variable name to extract (optional, returns dict if None)
            
        Returns:
            dict or numpy array: Ground truth data
        """
        data = np.load(gt_file, allow_pickle=True)
        
        if variable is not None:
            # Map variable names to NPZ keys using region-specific mapping
            var_key = self.gt_var_mapping.get(variable, variable)
            
            if var_key in data:
                return data[var_key].astype(np.float32)
            else:
                raise KeyError(f"Variable '{variable}' (mapped to '{var_key}') not found in {gt_file}")
        
        # Return all variables as a dict-like object
        return data
    
    def _load_ground_truth_npy(self, gt_file, variable=None):
        """
        Load ground truth from a per-timestamp NPY file (global 4-var product).

        The file holds a single float32 array [N, C]; the column layout is
        self._npy_columns (from npy_files_4var/wave_columns.json).

        Args:
            gt_file: Path to the NPY file
            variable: Variable name to extract (returns full [N, C] array if None)

        Returns:
            numpy array: [N] values for the variable, or [N, C] if variable is None
        """
        arr = np.load(gt_file)
        if variable is None:
            return arr
        var_key = self.gt_var_mapping.get(variable, variable)
        return arr[:, self._npy_columns.index(var_key)].astype(np.float32)

    def _load_ground_truth(self, gt_file, variable=None):
        """
        Load ground truth data based on file format.
        
        Args:
            gt_file: Path to the ground truth file
            variable: Variable name to extract (optional)
            
        Returns:
            Ground truth data (format depends on file type)
        """
        if self.gt_file_format == 'npz':
            return self._load_ground_truth_npz(gt_file, variable)
        elif self.gt_file_format == 'npy':
            return self._load_ground_truth_npy(gt_file, variable)
        else:
            return self._load_ground_truth_nc(gt_file, variable)
    
    def visualize_random_samples(self, num_samples=5, save_dir=None, seed=None,
                                  variable='hs', era5_dir=None):
        """
        Randomly select and visualize wave samples on a geographical map.

        Args:
            num_samples: Number of samples to randomly select and visualize
            save_dir: Directory to save visualization plots (if None, displays only)
            seed: Random seed for reproducibility
            variable: Variable name to plot (default: 'hs')
            era5_dir: Path to ERA5 NC data directory. When provided, also plots
                      the corresponding ERA5 wind and pressure fields.

        Returns:
            List of figure objects
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Randomly select samples
        if num_samples > len(self.sample_files):
            print(f"Warning: Requested {num_samples} samples but only {len(self.sample_files)} available")
            num_samples = len(self.sample_files)
        
        selected_files = random.sample(self.sample_files, num_samples)
        print(f"Randomly selected {num_samples} samples for visualization (region_mode='{self.region_mode}')")
        
        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Saving plots to {save_dir}")
        
        # Visualize each selected sample
        figures = []
        for i, sample_file in enumerate(selected_files):
            # Load sample
            sample_name = os.path.basename(sample_file).replace('.npz', '')
            print(f"\nVisualizing {sample_name}...")
            
            data = np.load(sample_file)
            sample = data['sample'].squeeze()

            # Multi-variable sample: select the requested variable channel
            if sample.ndim == 2 and sample.shape[1] > 1:
                var_names = ['hs', 'tp', 'dm', 'elev']
                idx = var_names.index(variable) if variable in var_names else 0
                sample = sample[:, idx]
            
            if sample.shape[0] != self.num_nodes:
                print(f"Warning: Sample has {sample.shape[0]} nodes but expected {self.num_nodes}")
                continue

            # Set save path if save_dir is provided
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f'{sample_name}_visualization.jpg')
            
            if self.region_mode == 'global':
                # Create xarray Dataset for global plotting
                # plot_wave_data_global expects lon/lat as data variables, not coordinates
                ds = xr.Dataset(
                    {
                        variable: (['node'], sample),
                        'lon': (['node'], self.lons),
                        'lat': (['node'], self.lats)
                    },
                    coords={
                        'time': sample_name
                    }
                )
                
                # Use plot_wave_data_global for global visualization
                fig = plot_wave_data_global(
                    dataset=ds,
                    variable=variable,
                    output_path=save_path,
                    projection='Robinson',
                    dpi=200,
                    show_interpolation=True
                )
            else:
                # Create xarray Dataset for HK region plotting (1D node coordinates)
                ds = xr.Dataset(
                    {
                        variable: (['node'], sample)
                    },
                    coords={
                        'lon': (['node'], self.lons),
                        'lat': (['node'], self.lats),
                        'time': sample_name  # Use sample name as time
                    }
                )
                
                # Use plot_wave_data to visualize HK region
                # Use self.region calculated from coordinates instead of hardcoded HK_REGION
                fig = plot_wave_data(
                    dataset=ds,
                    variable=variable,
                    region=self.region,
                    save_path=save_path,
                    show_interpolation=True,
                    use_high_res_coastline=True,
                    use_satellite_background=True,
                )
            
            figures.append(fig)

            # ---- ERA5 input visualization ----
            if era5_dir is not None:
                ts = self._extract_timestamp_from_name(sample_name)
                if ts is not None:
                    self._save_era5_figure(sample_name, ts, save_dir, era5_dir)

        print(f"\n✓ Visualized {len(figures)} samples")
        return figures

    @staticmethod
    def _extract_timestamp_from_name(sample_name):
        """Extract YYYYMMDDTHH from 'sample_20210101T00_ens003' or 'sample_20210101T00'."""
        prefix = 'sample_'
        if not sample_name.startswith(prefix):
            return None
        rest = sample_name[len(prefix):]
        if '_ens' in rest:
            rest = rest.split('_ens')[0]
        if len(rest) >= 10 and 'T' in rest:
            return rest
        return None

    def _save_era5_figure(self, sample_name, timestamp, save_dir, era5_dir):
        """Save a 4-panel figure of ERA5 wind + pressure input for a sample."""
        import cartopy.feature as cfeature
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        from modules.utils import ERA5_VARIABLE_NAMES

        era5_path = os.path.join(era5_dir, timestamp[:4], f'era5_{timestamp}.nc')
        if not os.path.exists(era5_path):
            print(f"  Warning: ERA5 file not found: {era5_path} — skipping.")
            return

        try:
            ds = xr.load_dataset(era5_path, engine='netcdf4')
        except Exception as e:
            print(f"  Warning: failed to load ERA5: {e} — skipping.")
            return

        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        if ds[lat_name].values[0] < ds[lat_name].values[-1]:
            ds = ds.sortby(lat_name, ascending=False)

        u_data = ds[ERA5_VARIABLE_NAMES[0]].values
        v_data = ds[ERA5_VARIABLE_NAMES[1]].values
        p_data = ds[ERA5_VARIABLE_NAMES[2]].values / 100.0
        wind_speed = np.sqrt(u_data ** 2 + v_data ** 2)

        lat = ds[lat_name].values
        lon_e = ds[lon_name].values
        lon_p = np.where(lon_e > 180, lon_e - 360, lon_e)
        Lon, Lat = np.meshgrid(lon_p, lat)

        map_extent = [float(lon_p.min()), float(lon_p.max()),
                      float(lat.min()), float(lat.max())]
        wind_vmax = max(float(np.nanmax(np.abs(u_data))),
                        float(np.nanmax(np.abs(v_data))))
        p_vmin, p_vmax = float(np.nanmin(p_data)), float(np.nanmax(p_data))
        ws_vmax = float(np.nanmax(wind_speed))

        fig = plt.figure(figsize=(10, 9))
        proj = ccrs.PlateCarree()

        panels = [
            (0, u_data,     'RdBu_r',  -wind_vmax, wind_vmax, 'm/s',
             f'10m U-Wind (East-West)\n{timestamp}'),
            (1, v_data,     'RdBu_r',  -wind_vmax, wind_vmax, 'm/s',
             f'10m V-Wind (North-South)\n{timestamp}'),
            (2, wind_speed, 'viridis',  0,          ws_vmax,   'm/s',
             f'10m Wind Speed\n{timestamp}'),
            (3, p_data,     'RdYlBu_r', p_vmin,     p_vmax,    'hPa',
             f'MSL Pressure\n{timestamp}'),
        ]

        for idx, data_2d, cmap, vmin, vmax, clabel, title in panels:
            ax = fig.add_subplot(2, 2, idx + 1, projection=proj)
            ax.set_extent(map_extent, crs=proj)
            ax.add_feature(cfeature.LAND, color='whitesmoke', alpha=1, zorder=3)
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.4, zorder=4)

            im = ax.pcolormesh(Lon, Lat, data_2d, cmap=cmap,
                               transform=proj, shading='auto', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label=clabel)

            if idx == 2:  # wind vectors overlaid on wind speed
                ss = max(1, min(len(lat), len(lon_p)) // 30)
                ax.quiver(Lon[::ss, ::ss], Lat[::ss, ::ss],
                          u_data[::ss, ::ss], v_data[::ss, ::ss],
                          transform=ccrs.PlateCarree(), alpha=0.6, width=0.002,
                          color='white', scale=500)

            ax.set_title(title, fontsize=16)

        plt.tight_layout()
        sp = os.path.join(save_dir, f'{sample_name}_era5_input.jpg')
        plt.savefig(sp, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved ERA5 input: {sp}")
    def visualize_samples_with_ground_truth(self, 
                                            data_dir=None, 
                                            num_samples=5, 
                                            save_dir=None, 
                                            seed=42, 
                                            variable='hs'):
        """
        Randomly select and visualize wave samples alongside their ground truth values.
        Generates three plots for each sample: ground truth, prediction, and difference.
        
        Args:
            data_dir: Directory containing ground truth NetCDF files (wave_*.nc)
            num_samples: Number of samples to randomly select and visualize
            save_dir: Directory to save visualization plots (if None, displays only)
            seed: Random seed for reproducibility
            variable: Variable name to plot (default: 'hs')
        
        Returns:
            List of tuples (gt_fig, pred_fig, diff_fig) for each sample
        """
        # Use default ground truth data directory if not specified
        if data_dir is None:
            data_dir = self.gt_data_dir
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Filter sample files that have timestamp format (conditional generation)
        # These files should be named sample_YYYYMMDDTHH.npz
        conditional_files = []
        for f in self.sample_files:
            basename = os.path.basename(f)
            # Check if file matches pattern sample_YYYYMMDDTHH.npz
            if basename.startswith('sample_') and 'T' in basename:
                conditional_files.append(f)
        
        if len(conditional_files) == 0:
            raise ValueError(
                f"No conditional sample files found in {self.sample_dir}. "
                "This method requires samples generated with timestamps (conditional generation)."
            )
        
        # Randomly select samples
        if num_samples > len(conditional_files):
            print(f"Warning: Requested {num_samples} samples but only {len(conditional_files)} available")
            num_samples = len(conditional_files)
        
        selected_files = random.sample(conditional_files, num_samples)
        print(f"Randomly selected {num_samples} conditional samples for comparison")
        
        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Saving comparison plots to {save_dir}")
        
        # Visualize each selected sample with ground truth
        figure_groups = []
        
        for i, sample_file in enumerate(selected_files):
            # Load sample and extract timestamp
            sample_name = os.path.basename(sample_file).replace('.npz', '')
            print(f"\nProcessing {sample_name}...")
            
            sample_data = np.load(sample_file)
            sample = sample_data['sample'].squeeze()  # Shape: (num_nodes,)
            
            # Extract timestamp from sample file
            # Format: sample_YYYYMMDDTHH.npz -> YYYYMMDDTHH
            timestamp = sample_name.replace('sample_', '')
            
            # Construct ground truth file path (uses region-specific format)
            gt_file = self._get_gt_file_path(timestamp, data_dir)
            
            # Check if ground truth file exists
            if not os.path.exists(gt_file):
                print(f"Warning: Ground truth file not found: {gt_file}. Skipping...")
                continue
            
            # Load ground truth data
            try:
                gt_values = self._load_ground_truth(gt_file, variable)
                
            except Exception as e:
                print(f"Error loading ground truth: {e}. Skipping...")
                continue
            
            # Validate dimensions match
            if gt_values.shape[0] != self.num_nodes or sample.shape[0] != self.num_nodes:
                print(f"Warning: Dimension mismatch. GT: {gt_values.shape}, Sample: {sample.shape}, Expected: {self.num_nodes}")
                continue
            
            # Calculate difference
            difference = sample - gt_values
            
            # Create xarray Datasets for plotting
            # 1. Ground Truth
            ds_gt_plot = xr.Dataset(
                {variable: (['node'], gt_values)},
                coords={
                    'lon': (['node'], self.lons),
                    'lat': (['node'], self.lats),
                    'time': timestamp
                }
            )
            
            # 2. Prediction
            ds_pred = xr.Dataset(
                {variable: (['node'], sample)},
                coords={
                    'lon': (['node'], self.lons),
                    'lat': (['node'], self.lats),
                    'time': timestamp
                }
            )
            
            # 3. Difference (Prediction - Ground Truth)
            ds_diff = xr.Dataset(
                {variable: (['node'], difference)},
                coords={
                    'lon': (['node'], self.lons),
                    'lat': (['node'], self.lats),
                    'time': timestamp
                }
            )
            
            # Generate plots
            save_path_combined = None
            
            if save_dir:
                save_path_combined = os.path.join(save_dir, f'{sample_name}_combined.jpg')
            
            # Calculate vmin and vmax from ground truth data
            gt_vmin = np.nanmin(gt_values)
            gt_vmax = np.nanmax(gt_values)
            
            # Calculate symmetric vmin and vmax for difference plot (centered at 0)
            diff_abs_max = np.nanmax(np.abs(difference))
            diff_vmin = -diff_abs_max
            diff_vmax = diff_abs_max
            
            print(f"  Plotting combined figure (GT vmin={gt_vmin:.3f}, vmax={gt_vmax:.3f}, Diff vmin={diff_vmin:.3f}, vmax={diff_vmax:.3f})...")
            
            # Import variable metadata
            from modules.utils import WAVE_VARIABLES
            
            # Get variable metadata for title
            var_prop = WAVE_VARIABLES.get(variable, {'plot_title': variable})
            var_title = var_prop.get('plot_title', variable)
            
            # Create a figure with three subplots (1 row, 3 columns)
            fig = plt.figure(figsize=(30, 9))
            
            # Create three axes with PlateCarree projection
            ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
            ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
            ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
            
            # Plot ground truth on first subplot
            _, ax1 = plot_wave_data(
                dataset=ds_gt_plot,
                variable=variable,
                region=HK_REGION,
                vmin=gt_vmin,
                vmax=gt_vmax,
                ax=ax1,
                return_ax=True,
                scatter_size=10,
                custom_title=f'{var_title} (Ground Truth)\nDate: {timestamp} | Nodes: {self.num_nodes}',
                use_high_res_coastline=True,
                use_satellite_background=True,
            )

            # Plot prediction on second subplot (use same vmin/vmax as GT)
            _, ax2 = plot_wave_data(
                dataset=ds_pred,
                variable=variable,
                region=HK_REGION,
                vmin=gt_vmin,
                vmax=gt_vmax,
                ax=ax2,
                return_ax=True,
                scatter_size=10,
                custom_title=f'{var_title} (Prediction)\nDate: {timestamp} | Nodes: {self.num_nodes}',
                use_high_res_coastline=True,
                use_satellite_background=True,
            )

            # Plot difference on third subplot (use symmetric scale centered at 0, with seismic colormap)
            _, ax3 = plot_wave_data(
                dataset=ds_diff,
                variable=variable,
                region=HK_REGION,
                vmin=diff_vmin,
                vmax=diff_vmax,
                cmap='seismic',
                ax=ax3,
                return_ax=True,
                scatter_size=10,
                custom_title=f'{var_title} (Difference)\nDate: {timestamp} | Nodes: {self.num_nodes}',
                use_high_res_coastline=True,
                use_satellite_background=True,
            )
            
            # Adjust layout
            plt.tight_layout()
            
            # Save combined figure
            if save_path_combined:
                os.makedirs(os.path.dirname(save_path_combined), exist_ok=True)
                plt.savefig(save_path_combined, dpi=150, bbox_inches='tight', facecolor='white')
                print(f"  Combined plot saved to: {save_path_combined}")
            
            plt.close(fig)
            
            figure_groups.append(fig)
        
        print(f"\n✓ Generated comparison plots for {len(figure_groups)} samples")
        return figure_groups

    def calculate_rmse_and_export(self, 
                                  data_dir=None, 
                                  save_path="rmse_results.csv", 
                                  variable='hs'):
        """
        Calculate RMSE for each sample against ground truth and export to CSV.
        
        Args:
            data_dir: Directory containing ground truth NetCDF files
            save_path: Path to save the CSV file
            variable: Variable name to calculate RMSE for (default: 'hs')
            
        Returns:
            DataFrame containing the results
        """
        # Use default ground truth data directory if not specified
        if data_dir is None:
            data_dir = self.gt_data_dir
        
        # Filter sample files that have timestamp format (conditional generation)
        conditional_files = []
        for f in self.sample_files:
            basename = os.path.basename(f)
            if basename.startswith('sample_') and 'T' in basename:
                conditional_files.append(f)
        
        if not conditional_files:
            print(f"No conditional sample files found in {self.sample_dir} for RMSE calculation.")
            return None
            
        print(f"Calculating RMSE for {len(conditional_files)} samples...")
        
        results = []
        
        for sample_file in conditional_files:
            # Load sample and extract timestamp
            sample_name = os.path.basename(sample_file).replace('.npz', '')
            timestamp = sample_name.replace('sample_', '')
            
            # Load sample data
            try:
                sample_data = np.load(sample_file)
                sample = sample_data['sample'].squeeze()
            except Exception as e:
                print(f"Error loading sample {sample_name}: {e}")
                continue
            
            # Construct ground truth file path (uses region-specific format)
            gt_file = self._get_gt_file_path(timestamp, data_dir)
            
            if not os.path.exists(gt_file):
                print(f"Warning: Ground truth file not found: {gt_file}. Skipping...")
                continue
            
            # Load ground truth data
            try:
                gt_values = self._load_ground_truth(gt_file, variable)
                
            except Exception as e:
                print(f"Error loading ground truth: {e}. Skipping...")
                continue
            
            # Validate dimensions match
            if gt_values.shape[0] != self.num_nodes or sample.shape[0] != self.num_nodes:
                print(f"Warning: Dimension mismatch for {sample_name}. GT: {gt_values.shape}, Sample: {sample.shape}")
                continue

            # Calculate RMSE
            # Only consider non-NaN values (ground truth might have NaNs for land)
            valid_mask = ~np.isnan(gt_values)
            if np.sum(valid_mask) == 0:
                print(f"Warning: No valid ground truth points for {sample_name}")
                continue

            diff = sample[valid_mask] - gt_values[valid_mask]
            mse = np.mean(diff ** 2)
            rmse = np.sqrt(mse)
            
            results.append({
                'timestamp': timestamp,
                'sample_name': sample_name,
                'rmse': rmse
            })
            
        # Create DataFrame and export
        if results:
            df = pd.DataFrame(results)
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Ensure save directory exists
            save_dir = os.path.dirname(os.path.abspath(save_path))
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            df.to_csv(save_path, index=False)
            print(f"✓ RMSE results exported to {save_path}")
            print(f"  Average RMSE: {df['rmse'].mean():.4f}")
            print(f"  Min RMSE: {df['rmse'].min():.4f}")
            print(f"  Max RMSE: {df['rmse'].max():.4f}")
            
            return df
        else:
            print("No results calculated.")
            return None

    def calculate_ensemble_rmse_and_export(self, 
                                           data_dir=None, 
                                           save_path="ensemble_rmse_results.csv", 
                                           variable='hs',
                                           num_ensemble_samples=None,
                                           max_workers=None):
        """
        Calculate RMSE for ensemble predictions (averaged across multiple samples) against ground truth.
        For each timestamp, averages all ensemble members and calculates RMSE against ground truth.
        Uses ProcessPoolExecutor to parallelize across timestamps.
        
        Args:
            data_dir: Directory containing ground truth NetCDF files
            save_path: Path to save the CSV file
            variable: Variable name to calculate RMSE for (default: 'hs')
            num_ensemble_samples: Number of ensemble samples to use for averaging (default: None, use all available)
            max_workers: Number of processes to use (default: None, uses min(16, cpu_count + 2))
            
        Returns:
            DataFrame containing the results (timestamp, ensemble_size, rmse)
        """
        # Use default ground truth data directory if not specified
        if data_dir is None:
            data_dir = self.gt_data_dir
        
        # Filter sample files that have ensemble format (with _ensXXX suffix)
        ensemble_files = []
        for f in self.sample_files:
            basename = os.path.basename(f)
            if basename.startswith('sample_') and '_ens' in basename and 'T' in basename:
                ensemble_files.append(f)
        
        if not ensemble_files:
            print(f"No ensemble sample files found in {self.sample_dir} for RMSE calculation.")
            print("Expected filename format: sample_<timestamp>_ens<XXX>.npz")
            return None
        
        print(f"Found {len(ensemble_files)} ensemble sample files...")
        
        # Group files by timestamp
        from collections import defaultdict
        timestamp_groups = defaultdict(list)
        
        for sample_file in ensemble_files:
            basename = os.path.basename(sample_file).replace('.npz', '')
            # Extract timestamp (everything between 'sample_' and '_ens')
            parts = basename.split('_ens')
            if len(parts) == 2:
                timestamp = parts[0].replace('sample_', '')
                timestamp_groups[timestamp].append(sample_file)
        
        print(f"Grouped into {len(timestamp_groups)} unique timestamps")
        
        # Use ProcessPoolExecutor for parallel processing
        if max_workers is None:
            max_workers = min(48, (os.cpu_count() or 1) + 2)
        
        print(f"Using {max_workers} processes for processing...")
        
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_timestamp = {
                executor.submit(
                    process_single_timestamp_ensemble_standalone,
                    timestamp,
                    files,
                    data_dir,
                    variable,
                    num_ensemble_samples,
                    self.num_nodes,
                    self.gt_file_format,
                    self.region_mode,
                    self.gt_filename_prefix,
                    self.gt_var_mapping
                ): timestamp
                for timestamp, files in timestamp_groups.items()
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_timestamp),
                                total=len(timestamp_groups),
                                desc="Calculating ensemble RMSE"):
                timestamp = future_to_timestamp[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing timestamp {timestamp}: {e}")
        
        # Create DataFrame and export
        if results:
            df = pd.DataFrame(results)
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Ensure save directory exists
            save_dir_path = os.path.dirname(os.path.abspath(save_path))
            if save_dir_path:
                os.makedirs(save_dir_path, exist_ok=True)
            
            df.to_csv(save_path, index=False)
            print(f"✓ Ensemble RMSE results exported to {save_path}")
            print(f"  Number of timestamps: {len(df)}")
            print(f"  Average ensemble size: {df['ensemble_size'].mean():.1f}")
            print(f"  Average RMSE: {df['rmse'].mean():.4f}")
            print(f"  Min RMSE: {df['rmse'].min():.4f}")
            print(f"  Max RMSE: {df['rmse'].max():.4f}")
            
            return df
        else:
            print("No results calculated.")
            return None

    def calculate_multivar_rmse_and_export(self, 
                                           data_dir=None, 
                                           save_path="multivar_rmse_results.csv", 
                                           variables=['hs', 'tp', 'dm', 'elev'],
                                           num_ensemble_samples=None,
                                           max_workers=None):
        """
        Calculate RMSE for multiple variables (channels) separately and export results.
        Assumes samples have multiple channels corresponding to the provided variables.
        Uses multi-threading to speed up processing.
        
        Args:
            data_dir: Directory containing ground truth NetCDF files
            save_path: Path to save the CSV file
            variables: List of variable names corresponding to the channels in the sample.
            num_ensemble_samples: Number of ensemble samples to use for averaging (default: None, use all available)
            max_workers: Number of threads to use (default: None, uses number of CPUs * 5)
            
        Returns:
            DataFrame containing the results
        """
        # Use default ground truth data directory if not specified
        if data_dir is None:
            data_dir = self.gt_data_dir
        
        # Filter sample files that have ensemble format (with _ensXXX suffix)
        ensemble_files = []
        for f in self.sample_files:
            basename = os.path.basename(f)
            if basename.startswith('sample_') and '_ens' in basename and 'T' in basename:
                ensemble_files.append(f)
        
        if not ensemble_files:
            print(f"No ensemble sample files found in {self.sample_dir} for RMSE calculation.")
            print("Expected filename format: sample_<timestamp>_ens<XXX>.npz")
            return None
        
        print(f"Found {len(ensemble_files)} ensemble sample files...")
        
        # Group files by timestamp
        from collections import defaultdict
        timestamp_groups = defaultdict(list)
        
        for sample_file in ensemble_files:
            basename = os.path.basename(sample_file).replace('.npz', '')
            # Extract timestamp (everything between 'sample_' and '_ens')
            parts = basename.split('_ens')
            if len(parts) == 2:
                timestamp = parts[0].replace('sample_', '')
                timestamp_groups[timestamp].append(sample_file)
        
        print(f"Grouped into {len(timestamp_groups)} unique timestamps")
        print(f"Processing ensemble predictions for variables: {variables}...")
        
        results = []
        
        # Use ProcessPoolExecutor for parallel processing to avoid GIL/Segmentation faults
        # with NetCDF/HDF5 libraries
        if max_workers is None:
            max_workers = min(48, (os.cpu_count() or 1) + 2)
            
        print(f"Using {max_workers} processes for processing...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for each timestamp
            future_to_timestamp = {
                executor.submit(
                    process_single_timestamp_multivar_standalone, 
                    timestamp, 
                    files, 
                    data_dir, 
                    variables, 
                    num_ensemble_samples,
                    self.num_nodes,
                    self.gt_file_format,
                    self.region_mode,
                    self.gt_filename_prefix,
                    self.gt_var_mapping
                ): timestamp 
                for timestamp, files in timestamp_groups.items()
            }
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_timestamp), 
                               total=len(timestamp_groups),
                               desc="Calculating multivar RMSE"):
                timestamp = future_to_timestamp[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing timestamp {timestamp}: {e}")
        
        # Create DataFrame and export
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values('timestamp')
            
            save_dir_path = os.path.dirname(os.path.abspath(save_path))
            if save_dir_path:
                os.makedirs(save_dir_path, exist_ok=True)
            
            df.to_csv(save_path, index=False)
            print(f"✓ Multivar RMSE results exported to {save_path}")
            
            # Print summary stats
            print("\nSummary Statistics:")
            for var_name in variables:
                col = f'{var_name}_rmse'
                if col in df.columns:
                    print(f"  {var_name.upper()} RMSE: Mean={df[col].mean():.4f}, Min={df[col].min():.4f}, Max={df[col].max():.4f}")
            
            return df
        else:
            print("No results calculated.")
            return None



    def visualize_ensemble_samples_with_ground_truth(self,
                                                       data_dir=None,
                                                       num_samples=5,
                                                       save_dir=None,
                                                       seed=42,
                                                       variables=None,
                                                       era5_dir=None):
        """
        Randomly select and visualize ensemble wave samples alongside their ground truth values.
        Generates four plots for each sample: ground truth, ensemble mean (prediction), 
        ensemble standard deviation (uncertainty), and difference (prediction - ground truth).
        
        Args:
            data_dir: Directory containing ground truth NetCDF files (wave_*.nc)
            num_samples: Number of timestamps to randomly select and visualize
            save_dir: Directory to save visualization plots (if None, displays only)
            seed: Random seed for reproducibility
            variables: List of variable names to plot. If None, automatically detected.
        
        Returns:
            List of figures for each sample
        """
        # Use default ground truth data directory if not specified
        if data_dir is None:
            data_dir = self.gt_data_dir
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Filter sample files that have ensemble format (with _ensXXX suffix)
        ensemble_files = []
        for f in self.sample_files:
            basename = os.path.basename(f)
            if basename.startswith('sample_') and '_ens' in basename and 'T' in basename:
                ensemble_files.append(f)
        
        if not ensemble_files:
            raise ValueError(
                f"No ensemble sample files found in {self.sample_dir}. "
                "This method requires samples generated with ensemble format (sample_<timestamp>_ens<XXX>.npz)."
            )
        
        # Group files by timestamp
        from collections import defaultdict
        timestamp_groups = defaultdict(list)
        
        for sample_file in ensemble_files:
            basename = os.path.basename(sample_file).replace('.npz', '')
            # Extract timestamp (everything between 'sample_' and '_ens')
            parts = basename.split('_ens')
            if len(parts) == 2:
                timestamp = parts[0].replace('sample_', '')
                timestamp_groups[timestamp].append(sample_file)
        
        print(f"Found {len(timestamp_groups)} unique timestamps with ensemble samples")
        
        # Randomly select timestamps
        available_timestamps = list(timestamp_groups.keys())
        if num_samples > len(available_timestamps):
            print(f"Warning: Requested {num_samples} samples but only {len(available_timestamps)} available")
            num_samples = len(available_timestamps)
        
        selected_timestamps = random.sample(available_timestamps, num_samples)
        print(f"Randomly selected {num_samples} timestamps for comparison")
        
        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Saving comparison plots to {save_dir}")
        
        # Build task list for parallel processing
        tasks = []
        for i, timestamp in enumerate(selected_timestamps):
            tasks.append((
                timestamp,
                timestamp_groups[timestamp],
                data_dir,
                save_dir,
                self.num_nodes,
                self.lons,
                self.lats,
                self.region_mode,
                self.gt_file_format,
                self.gt_filename_prefix,
                self.gt_var_mapping,
                variables,
                seed + i if seed is not None else None,
                era5_dir,
            ))

        max_workers = min(64, (os.cpu_count() or 1) + 2)
        print(f"Using {max_workers} parallel workers for visualization...")

        all_files = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_timestamp_viz_standalone, task): task[0]
                       for task in tasks}
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures), desc="Visualizing ensemble samples"):
                timestamp = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_files.extend(result)
                except Exception as e:
                    print(f"Error processing timestamp {timestamp}: {e}")

        print(f"\n✓ Generated {len(all_files)} ensemble comparison files across {len(selected_timestamps)} timestamps")
        return all_files

    def visualize_trajectory(self,
                             trajectory_dir=None,
                             num_samples=2,
                             save_dir=None,
                             seed=42,
                             variables=None):
        """
        Visualize intermediate ODE trajectory snapshots saved during sampling
        (files sample_<timestamp>_ens<XXX>_t<t>.npz under <run_dir>/trajectory).

        For each selected (timestamp, ensemble member) sample, exports ONE figure
        per (ODE time t, variable) — variables are saved as separate images.
        Color ranges come from the ground truth of each timestamp, so all
        figures of the same variable share a consistent vmin/vmax.

        Args:
            trajectory_dir: Directory containing trajectory npz files.
                            Defaults to <run_dir>/trajectory inferred from self.sample_dir
                            (self.sample_dir is <run_dir>/samples).
            num_samples: Number of (timestamp, ens) samples to randomly select.
            save_dir: Directory to save figures.
            seed: Random seed for sample selection.
            variables: Variable names per channel; None auto-detects from channel count.

        Returns:
            List of generated file paths.
        """
        import re
        from collections import defaultdict

        if trajectory_dir is None:
            # self.sample_dir is <run_dir>/samples -> trajectory lives alongside it
            trajectory_dir = os.path.join(os.path.dirname(self.sample_dir.rstrip('/')), 'trajectory')

        if not os.path.isdir(trajectory_dir):
            raise ValueError(f"Trajectory directory not found: {trajectory_dir}")

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Group trajectory files by (timestamp, ensemble index)
        pattern = re.compile(r'^sample_(.+)_ens(\d+)_t([\d.]+)\.npz$')
        sample_groups = defaultdict(list)  # (ts, ens_idx) -> [(ode_t, path), ...]
        for fname in os.listdir(trajectory_dir):
            m = pattern.match(fname)
            if not m:
                continue
            ts, ens_str, t_str = m.groups()
            sample_groups[(ts, int(ens_str))].append(
                (float(t_str), os.path.join(trajectory_dir, fname)))

        if not sample_groups:
            raise ValueError(
                f"No trajectory files found in {trajectory_dir}. "
                "Expected files named sample_<timestamp>_ens<XXX>_t<t>.npz "
                "(enable sampling.save_trajectory_times during sampling)."
            )

        # Sort snapshots by ODE time within each sample
        for key in sample_groups:
            sample_groups[key].sort(key=lambda x: x[0])

        n_times = len(next(iter(sample_groups.values())))
        print(f"Found {len(sample_groups)} trajectory samples "
              f"({n_times} ODE times each) in {trajectory_dir}")

        keys = sorted(sample_groups.keys())
        if num_samples is not None and num_samples < len(keys):
            keys = random.sample(keys, num_samples)
        print(f"Selected {len(keys)} samples for trajectory visualization")

        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(trajectory_dir.rstrip('/')),
                                    'visualization', 'trajectory')
        # Separate subfolders for member / ensemble-stats / GT figures
        members_dir = os.path.join(save_dir, 'members')
        ensemble_dir = os.path.join(save_dir, 'ensemble')
        gt_dir = os.path.join(save_dir, 'gt')
        for d in (members_dir, ensemble_dir, gt_dir):
            os.makedirs(d, exist_ok=True)
        print(f"Saving trajectory plots to {save_dir} (members/ ensemble/ gt/)")

        # Detect variables from the first trajectory file (needed for GT ranges)
        if variables is None:
            first_arr = np.load(next(iter(sample_groups.values()))[0][1])['sample']
            n_ch = first_arr.shape[-1] if first_arr.ndim > 1 else 1
            variables = _detect_traj_variables(n_ch)

        # Per-variable color ranges from ground truth, so all figures of the
        # same variable share a consistent vmin/vmax (across t and members).
        gt_ranges = {}
        for ts in sorted({ts for ts, _ in keys}):
            gt_ranges[ts] = self._compute_gt_var_ranges(ts, variables)

        tasks = []
        # t=1 final states live under <run_dir>/samples (unless the user
        # already requested t=1.0 in save_trajectory_times)
        has_t1 = {key: any(abs(t - 1.0) < 1e-6 for t, _ in flist)
                  for key, flist in sample_groups.items()}
        for ts, ens_idx in keys:
            t_files = list(sample_groups[(ts, ens_idx)])
            if not has_t1[(ts, ens_idx)]:
                final_path = os.path.join(
                    self.sample_dir, f'sample_{ts}_ens{ens_idx:03d}.npz')
                if os.path.exists(final_path):
                    t_files.append((1.0, final_path))
            tasks.append((
                ts, ens_idx, t_files, members_dir,
                self.num_nodes, self.lons, self.lats, self.region_mode, variables,
                gt_ranges.get(ts),
            ))

        # Ensemble mean / std tasks: one per selected timestamp, using ALL
        # available members (not just the randomly selected ones)
        ensemble_tasks = []
        selected_ts = sorted({ts for ts, _ in keys})
        for ts in selected_ts:
            t_map = defaultdict(list)  # ode_t -> [member paths]
            for (ts2, ens_idx), flist in sample_groups.items():
                if ts2 != ts:
                    continue
                for ode_t, path in flist:
                    t_map[ode_t].append(path)
                if not has_t1[(ts2, ens_idx)]:
                    final_path = os.path.join(
                        self.sample_dir, f'sample_{ts2}_ens{ens_idx:03d}.npz')
                    if os.path.exists(final_path):
                        t_map[1.0].append(final_path)
            n_members = max((len(v) for v in t_map.values()), default=0)
            if n_members >= 2:
                ensemble_tasks.append((
                    ts, sorted(t_map.items()), ensemble_dir,
                    self.num_nodes, self.lons, self.lats, self.region_mode,
                    variables, gt_ranges.get(ts),
                ))
        if ensemble_tasks:
            print(f"Ensemble mean/std enabled for {len(ensemble_tasks)} timestamps")

        # Ground-truth tasks: one per selected timestamp (skipped if GT missing)
        gt_tasks = []
        for ts in selected_ts:
            gt_file = self._gt_file_path(ts)
            if os.path.exists(gt_file):
                gt_tasks.append((
                    ts, gt_file, self.gt_file_format, self.gt_var_mapping,
                    gt_dir, self.num_nodes, self.lons, self.lats,
                    self.region_mode, variables, gt_ranges.get(ts),
                ))

        max_workers = min(16, (os.cpu_count() or 1) + 2,
                          len(tasks) + len(ensemble_tasks) + len(gt_tasks) or 1)
        all_files = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_trajectory_sample_standalone, task): task[:2]
                       for task in tasks}
            for task in ensemble_tasks:
                futures[executor.submit(_process_trajectory_ensemble_standalone, task)] = (task[0], 'ens_stats')
            for task in gt_tasks:
                futures[executor.submit(_process_trajectory_gt_standalone, task)] = (task[0], 'gt')
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures), desc="Visualizing trajectories"):
                try:
                    result = future.result()
                    if result:
                        all_files.extend(result)
                except Exception as e:
                    print(f"Error processing trajectory sample {futures[future]}: {e}")

        print(f"\n✓ Generated {len(all_files)} trajectory figures "
              f"({len(tasks)} member samples + {len(ensemble_tasks)} ensemble-stats "
              f"+ {len(gt_tasks)} GT tasks)")
        return all_files

    def visualize_node_distributions(self,
                                     num_nodes=4,
                                     seed=42,
                                     timestamps=None,
                                     save_dir=None,
                                     variables=None,
                                     node_indices=None):
        """Visualize the empirical ensemble distribution at random nodes.

        For each timestamp, loads ALL ensemble members, randomly picks
        `num_nodes` nodes, and exports:
          - one map figure showing the selected node locations (labelled 0..N-1)
            on top of the ensemble-mean field of the first variable;
          - one figure per (selected node, variable): KDE curve of the
            member values + rug plot of the individual members + a red
            dashed line for the ground truth (if available). No title or
            axis labels. Each figure is saved as PDF (without legend) and
            JPG (with the legend below the axes); the legend is also
            exported once as a standalone PDF (nodedist_legend.pdf).

        Args:
            num_nodes: Number of random nodes to pick per timestamp.
            seed: Random seed for node selection.
            timestamps: Optional list of timestamps to restrict to; None uses
                        every timestamp found in the samples directory.
            save_dir: Directory to save figures (defaults to
                      <run_dir>/visualization/node_distribution).
            variables: Variable names per channel; None auto-detects.
            node_indices: Optional explicit list of node indices to plot;
                        overrides random selection (num_nodes/seed).

        Returns:
            List of generated file paths.
        """
        import re
        from collections import defaultdict
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        from scipy.stats import gaussian_kde
        from modules.plot.plot_wave import plot_wave_data

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Group sample files by timestamp
        pattern = re.compile(r'^sample_(.+)_ens(\d+)\.npz$')
        groups = defaultdict(list)
        for fname in os.listdir(self.sample_dir):
            m = pattern.match(fname)
            if m:
                groups[m.group(1)].append(
                    os.path.join(self.sample_dir, fname))
        if not groups:
            raise ValueError(f"No sample files found in {self.sample_dir}")

        ts_list = sorted(groups)
        if timestamps:
            wanted = set(timestamps)
            missing = wanted - set(ts_list)
            if missing:
                print(f"  Warning: timestamps not found in samples: {sorted(missing)}")
            ts_list = [ts for ts in ts_list if ts in wanted]
        if not ts_list:
            raise ValueError("No timestamps to visualize")

        if save_dir is None:
            save_dir = os.path.join(
                os.path.dirname(self.sample_dir.rstrip('/')),
                'visualization', 'node_distribution')
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving node-distribution plots to {save_dir}")

        if self.region_mode == 'global':
            region_param = None
        else:
            from modules.utils import HK_REGION
            region_param = list(HK_REGION)

        generated = []
        for ts in ts_list:
            paths = sorted(groups[ts])
            arrs = []
            for p in paths:
                a = np.load(p)['sample']
                if a.ndim == 1:
                    a = a[:, np.newaxis]
                arrs.append(a.astype(np.float32))
            stack = np.stack(arrs)  # (members, nodes, C)
            n_members, n_nodes_total, n_ch = stack.shape

            cur_vars = variables
            if cur_vars is None:
                cur_vars = _detect_traj_variables(n_ch)

            # Random node selection (sorted for stable output ordering)
            if node_indices is not None:
                node_indices = np.sort(np.asarray(node_indices, dtype=int))
            else:
                rng = np.random.default_rng(seed)
                node_indices = np.sort(rng.choice(
                    n_nodes_total, size=min(num_nodes, n_nodes_total),
                    replace=False))
            print(f"[{ts}] {n_members} members; selected nodes "
                  f"{node_indices.tolist()} of {n_nodes_total}")

            # Ground truth values per variable (red dashed line)
            gt_values = {}
            gt_file = self._gt_file_path(ts)
            if os.path.exists(gt_file):
                try:
                    if self.gt_file_format == 'npz':
                        gt_data = np.load(gt_file, allow_pickle=True)
                        for variable in cur_vars:
                            var_key = self.gt_var_mapping.get(variable, variable)
                            if var_key in gt_data:
                                gt_values[variable] = gt_data[var_key].astype(
                                    np.float32).squeeze()
                    else:
                        gt_data = xr.load_dataset(gt_file)
                        if self.region_mode != 'global':
                            from modules.utils import HK_REGION
                            min_lon, max_lon, min_lat, max_lat = HK_REGION
                            gt_data = gt_data.where(
                                (gt_data.lon >= min_lon) & (gt_data.lon <= max_lon) &
                                (gt_data.lat >= min_lat) & (gt_data.lat <= max_lat),
                                drop=True)
                        for variable in cur_vars:
                            var_key = self.gt_var_mapping.get(variable, variable)
                            if var_key in gt_data:
                                gt_values[variable] = gt_data[var_key].values.squeeze()
                except Exception as e:
                    print(f"  Warning: failed to load GT for {ts}: {e}")
                    gt_values = {}

            # --- Map figure: selected node locations on ens-mean background ---
            try:
                var0 = cur_vars[0]
                ds = xr.Dataset(
                    {var0: (['node'], stack.mean(axis=0)[:, 0])},
                    coords={'lon': (['node'], self.lons),
                            'lat': (['node'], self.lats),
                            'time': ts})
                fig = plt.figure(figsize=(5, 3.5))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                plot_wave_data(
                    dataset=ds, variable=var0, region=region_param,
                    vmin=None, vmax=None, cmap='turbo', ax=ax, return_ax=True,
                    scatter_size=8, custom_title='', show_cities=False,
                    show_colorbar=False, show_nodes=False,
                    show_left_labels=False, show_bottom_labels=False,
                    show_top_labels=False,
                    use_high_res_coastline=True,
                    use_satellite_background=(self.region_mode != 'global'),
                )
                for k, idx in enumerate(node_indices):
                    ax.scatter([self.lons[idx]], [self.lats[idx]],
                               marker='*', s=16, c='magenta',
                               transform=ccrs.PlateCarree(), zorder=10)
                    ax.annotate(str(k), (self.lons[idx], self.lats[idx]),
                                xytext=(5, 4), textcoords='offset points',
                                color='magenta', fontsize=13, fontweight='bold',
                                transform=ccrs.PlateCarree(), zorder=11)
                map_path = os.path.join(save_dir, f'nodedist_{ts}_nodemap.pdf')
                plt.tight_layout()
                fig.savefig(map_path, bbox_inches='tight',
                            facecolor='white')
                fig.savefig(map_path[:-4] + '.jpg', dpi=600,
                            bbox_inches='tight', facecolor='white')
                plt.close(fig)
                generated.append(map_path)
            except Exception as e:
                print(f"  Warning: node map failed for {ts}: {e}")

            # --- Per-node distribution figures: one figure per (node, variable) ---
            # Nature-style: single-column width, Arial, small outward ticks,
            # bottom spine only, muted blue/red palette.
            nature_rc = {
                'font.family': 'sans-serif',
                'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                'font.size': 8,
                'axes.linewidth': 0.8,
                'xtick.direction': 'out',
                'xtick.labelsize': 7,
                'xtick.major.size': 3,
                'xtick.major.width': 0.8,
                'legend.fontsize': 7,
            }
            kde_color = '#2166ac'
            gt_color = '#b2182b'
            with plt.rc_context(nature_rc):
                for k, idx in enumerate(node_indices):
                    for v, variable in enumerate(cur_vars):
                        if v >= n_ch:
                            break
                        fig, ax = plt.subplots(figsize=(5.0, 3.0))
                        vals = stack[:, idx, v]
                        vals = vals[np.isfinite(vals)]

                        v_min, v_max = float(vals.min()), float(vals.max())
                        pad = 0.15 * (v_max - v_min) if v_max > v_min else max(abs(v_min) * 0.1, 1e-6)
                        xs = np.linspace(v_min - pad, v_max + pad, 200)
                        kde_drawn = False
                        if v_max > v_min and len(vals) >= 2:
                            try:
                                kde = gaussian_kde(vals)
                                ys = kde(xs)
                                ax.plot(xs, ys, color=kde_color, lw=1.5,
                                        label='KDE')
                                ax.fill_between(xs, ys, alpha=0.25,
                                                color=kde_color, lw=0)
                                kde_drawn = True
                            except np.linalg.LinAlgError:
                                pass
                        if not kde_drawn:
                            ax.hist(vals, bins='auto', color=kde_color,
                                    alpha=0.6)

                        # Rug plot: one tick per ensemble member
                        y_lo, y_hi = ax.get_ylim()
                        ax.plot(vals, np.full_like(vals, y_lo), '|',
                                color=kde_color, markersize=6, alpha=0.5,
                                markeredgewidth=0.8, label='members')
                        # Ground truth
                        if variable in gt_values:
                            gt_v = float(gt_values[variable].ravel()[idx])
                            ax.axvline(gt_v, color=gt_color, ls='--', lw=1.2,
                                       label='GT')
                        # Point estimate (ensemble mean) + interval estimate
                        # (2.5-97.5% percentile -> 95% CI, shaded band)
                        est_color = '#333333'
                        mean_v = float(vals.mean())
                        p_lo, p_hi = (float(np.percentile(vals, 2.5)),
                                      float(np.percentile(vals, 97.5)))
                        ax.axvspan(p_lo, p_hi, color='#999999', alpha=0.15,
                                   lw=0, zorder=0, label='95% CI')
                        ax.axvline(mean_v, color=est_color, ls='-', lw=1.2,
                                   label='Point estimate')
                        ax.set_yticks([])
                        ax.set_xticks([])
                        for spine in ('top', 'right', 'left'):
                            ax.spines[spine].set_visible(False)
                        ax.margins(x=0.02)

                        # PDF: legend-free (legend exported separately below)
                        fig.tight_layout()
                        out_path = os.path.join(
                            save_dir,
                            f'nodedist_{ts}_node{k:02d}_idx{idx}_{variable}.pdf')
                        fig.savefig(out_path, bbox_inches='tight',
                                    facecolor='white')
                        # JPG: with legend below the axes, no title / axis labels
                        ax.legend(loc='upper center', ncol=3,
                                  bbox_to_anchor=(0.5, -0.12),
                                  frameon=False, handlelength=1.6,
                                  columnspacing=1.2)
                        fig.tight_layout()
                        jpg_path = out_path[:-4] + '.jpg'
                        fig.savefig(jpg_path, dpi=600, bbox_inches='tight',
                                    facecolor='white')
                        plt.close(fig)
                        generated.extend([out_path, jpg_path])

        # --- Standalone legend PDF shared by all node-distribution figures ---
        with plt.rc_context(nature_rc):
            legend_path = os.path.join(save_dir, 'nodedist_legend.pdf')
            try:
                fig_leg, ax_leg = plt.subplots(figsize=(5.0, 1.0))
                ax_leg.axis('off')
                handles = [
                    plt.Line2D([0], [0], color=kde_color, lw=1.5, label='KDE'),
                    plt.Line2D([0], [0], color=kde_color, marker='|', ls='None',
                               markersize=6, markeredgewidth=0.8,
                               label='members'),
                    plt.Line2D([0], [0], color=gt_color, ls='--', lw=1.2,
                               label='GT'),
                    plt.Rectangle((0, 0), 1, 1, color='#999999', alpha=0.15,
                                  lw=0, label='95% CI'),
                    plt.Line2D([0], [0], color='#333333', lw=1.2,
                               label='Point estimate'),
                ]
                ax_leg.legend(handles=handles, loc='center', ncol=5,
                              frameon=False, handlelength=1.6,
                              columnspacing=1.2)
                fig_leg.savefig(legend_path, bbox_inches='tight',
                                facecolor='white')
                plt.close(fig_leg)
                generated.append(legend_path)
            except Exception as e:
                print(f"  Warning: standalone legend PDF failed: {e}")

        print(f"\n✓ Generated {len(generated)} node-distribution figures "
              f"({len(ts_list)} timestamps)")
        return generated

    def _gt_file_path(self, timestamp):
        """Ground-truth file path for a timestamp (region/format aware)."""
        if self.region_mode == 'global':
            subdir = "npz_files" if self.gt_file_format == 'npz' else "nc_files"
            date_part = timestamp[:8]
            hour_part = timestamp[9:11]
            year = date_part[:4]
            return os.path.join(
                self.gt_data_dir, subdir, year,
                f"{self.gt_filename_prefix}_{date_part}_{hour_part}.{self.gt_file_format}")
        return os.path.join(
            self.gt_data_dir,
            f"{self.gt_filename_prefix}_{timestamp}.{self.gt_file_format}")

    def _compute_gt_var_ranges(self, timestamp, variables):
        """Load ground truth for a timestamp and compute per-variable color
        ranges {variable: (vmin, vmax)}. Returns None on failure (caller falls
        back to per-figure percentile ranges)."""
        gt_file = self._gt_file_path(timestamp)

        if not os.path.exists(gt_file):
            print(f"  Warning: GT file not found for {timestamp}: {gt_file} "
                  "— falling back to per-figure ranges")
            return None

        try:
            if self.gt_file_format == 'npz':
                gt_data = np.load(gt_file, allow_pickle=True)
            else:
                gt_data = xr.load_dataset(gt_file)
                if self.region_mode != 'global':
                    from modules.utils import HK_REGION
                    min_lon, max_lon, min_lat, max_lat = HK_REGION
                    gt_data = gt_data.where(
                        (gt_data.lon >= min_lon) & (gt_data.lon <= max_lon) &
                        (gt_data.lat >= min_lat) & (gt_data.lat <= max_lat),
                        drop=True)

            ranges = {}
            for variable in variables:
                var_key = self.gt_var_mapping.get(variable, variable)
                try:
                    if self.gt_file_format == 'npz':
                        gt_values = gt_data[var_key].astype(np.float32).squeeze()
                    else:
                        gt_values = gt_data[var_key].values.squeeze()
                except KeyError:
                    continue
                if variable == 'elev':
                    ranges[variable] = (float(np.nanpercentile(gt_values, 0.1)),
                                        float(np.nanpercentile(gt_values, 99.9)))
                else:
                    ranges[variable] = (float(np.nanmin(gt_values)),
                                        float(np.nanmax(gt_values)))
            return ranges if ranges else None
        except Exception as e:
            print(f"  Warning: failed to load GT for {timestamp}: {e}")
            return None

    def visualize_region(self,
                         timestamps,
                         region,
                         data_dir=None,
                         save_dir=None,
                         variables=None,
                         scatter_size=25,
                         show_interpolation=False,
                         era5_dir=None,
                         use_high_res_coastline=True,
                         use_satellite_background=True):
        """
        Visualize ensemble samples for specific timestamps cropped to a lat/lon bounding box.

        Produces a 6-panel figure (GT, Ensemble Mean, Ensemble Std, Sample1, Sample2,
        Difference) for each requested timestamp × variable combination, using
        PlateCarree projection via plot_wave_data().

        When ``show_interpolation=True`` an additional interpolated version of the
        figure is saved alongside the scatter version.

        Colorbar ranges are determined from the **region-local** GT values (only
        nodes inside the bounding box), giving much better contrast for regional views.

        Args:
            timestamps: List of YYYYMMDDTHH strings to visualize.
            region: [min_lon, max_lon, min_lat, max_lat] bounding box.
            data_dir: Override for ground truth directory. Uses self.gt_data_dir if None.
            save_dir: Directory to save output plots.
            variables: List of variable names. Auto-detected from sample shape if None.
            scatter_size: Scatter point size (default 25; larger helps for small regions).
            show_interpolation: If True, also save an interpolated (pcolormesh) version.

        Returns:
            List of figures generated.
        """
        import random as _random
        from collections import defaultdict
        from modules.utils import WAVE_VARIABLES
        import cartopy.feature as cfeature
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

        if data_dir is None:
            data_dir = self.gt_data_dir

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Saving region plots to {save_dir}")

        # Pre-compute region mask (nodes inside the bounding box)
        min_lon, max_lon, min_lat, max_lat = region
        region_mask = (
            (self.lons >= min_lon) & (self.lons <= max_lon) &
            (self.lats >= min_lat) & (self.lats <= max_lat)
        )

        # --- Group ensemble sample files by timestamp ---
        ensemble_files = [
            f for f in self.sample_files
            if os.path.basename(f).startswith('sample_')
            and '_ens' in os.path.basename(f)
            and 'T' in os.path.basename(f)
        ]
        if not ensemble_files:
            raise ValueError(
                f"No ensemble sample files found in {self.sample_dir}. "
                "Expected files named sample_<YYYYMMDDTHH>_ens<NNN>.npz."
            )

        timestamp_groups = defaultdict(list)
        for f in ensemble_files:
            basename = os.path.basename(f).replace('.npz', '')
            parts = basename.split('_ens')
            if len(parts) == 2:
                ts = parts[0].replace('sample_', '')
                timestamp_groups[ts].append(f)

        figures = []

        for timestamp in timestamps:
            if timestamp not in timestamp_groups:
                print(f"Warning: timestamp {timestamp} not found in samples — skipping.")
                continue

            sample_files_list = timestamp_groups[timestamp]
            print(f"\nProcessing {timestamp} (ensemble size: {len(sample_files_list)})...")

            # Load ensemble members
            ensemble_samples = []
            for sf in sample_files_list:
                try:
                    ensemble_samples.append(np.load(sf)['sample'].squeeze())
                except Exception as e:
                    print(f"  Warning: could not load {os.path.basename(sf)}: {e}")
            if not ensemble_samples:
                print(f"  No valid samples for {timestamp} — skipping.")
                continue

            ensemble_arr = np.array(ensemble_samples)  # (ens, nodes) or (ens, nodes, vars)

            # Determine variable list
            current_variables = variables
            if current_variables is None:
                if ensemble_arr.ndim == 3 and ensemble_arr.shape[2] == 4:
                    current_variables = ['hs', 'tp', 'dm', 'elev']
                else:
                    current_variables = ['hs']
            if ensemble_arr.ndim == 2:
                ensemble_arr = ensemble_arr[:, :, np.newaxis]

            # Load ground truth
            gt_file = self._get_gt_file_path(timestamp, data_dir)
            if not os.path.exists(gt_file):
                print(f"  Warning: GT file not found: {gt_file} — skipping.")
                continue
            try:
                gt_data = self._load_ground_truth(gt_file)
            except Exception as e:
                print(f"  Error loading GT: {e} — skipping.")
                continue

            # Pick 2 random members for sample panels
            ens_size = ensemble_arr.shape[0]
            if ens_size >= 2:
                s_idx = _random.sample(range(ens_size), 2)
            else:
                s_idx = [0, 0]

            for var_idx, variable in enumerate(current_variables):
                var_ens = ensemble_arr[:, :, var_idx]  # (ens, nodes)
                if variable == 'dm':
                    # 循环均值/离散度：10° 与 350° 的均值应为 0° 而非 180°
                    rad = np.deg2rad(var_ens.astype(np.float64))
                    ensemble_mean = np.rad2deg(np.arctan2(np.mean(np.sin(rad), axis=0),
                                                          np.mean(np.cos(rad), axis=0))) % 360.0
                    circ_dev = np.arctan2(np.sin(rad - np.deg2rad(ensemble_mean)),
                                          np.cos(rad - np.deg2rad(ensemble_mean)))
                    ensemble_std = np.rad2deg(np.std(circ_dev, axis=0))
                else:
                    ensemble_mean = np.mean(var_ens, axis=0)
                    ensemble_std = np.std(var_ens, axis=0)
                sample1 = var_ens[s_idx[0]]
                sample2 = var_ens[s_idx[1]]

                # Extract GT for this variable
                var_key = self.gt_var_mapping.get(variable, variable)
                try:
                    if self.gt_file_format == 'npz':
                        gt_values = gt_data[var_key].astype(np.float32).squeeze()
                    elif self.gt_file_format == 'npy':
                        gt_values = gt_data[:, self._npy_columns.index(var_key)].astype(np.float32)
                    else:
                        gt_values = gt_data[var_key].values.squeeze()
                except (KeyError, ValueError):
                    print(f"  Warning: variable '{variable}' (key '{var_key}') not in GT — skipping.")
                    continue

                if gt_values.shape[0] != self.num_nodes or ensemble_mean.shape[0] != self.num_nodes:
                    print(f"  Dimension mismatch: GT {gt_values.shape}, mean {ensemble_mean.shape}, expected {self.num_nodes} — skipping.")
                    continue

                difference = ensemble_mean - gt_values
                if variable == 'dm':
                    # 循环差值：359° vs 1° 的真实偏差是 -2° 而非 +358°
                    difference = (difference + 180.0) % 360.0 - 180.0

                # --- Colorbar range from REGION-LOCAL GT values ---
                gt_local = gt_values[region_mask]
                if variable == 'elev':
                    gt_vmin = float(np.nanpercentile(gt_local, 0.1))
                    gt_vmax = float(np.nanpercentile(gt_local, 99.9))
                else:
                    gt_vmin = float(np.nanmin(gt_local))
                    gt_vmax = float(np.nanmax(gt_local))
                std_local = ensemble_std[region_mask]
                std_vmin, std_vmax = 0.0, float(np.nanmax(std_local))
                diff_local = difference[region_mask]
                diff_abs = float(np.nanmax(np.abs(diff_local)))
                diff_vmin, diff_vmax = -diff_abs, diff_abs

                var_prop = WAVE_VARIABLES.get(variable, {})
                var_title = var_prop.get('plot_title', variable)

                # dm 是循环变量（0°=360°）：值面板用循环色图 twilight 并固定
                # 0-360；差值 wrap 到 ±180°，用循环色图 twilight_shifted（两端同色）
                val_cmap = 'twilight' if variable == 'dm' else 'turbo'
                diff_cmap = 'twilight_shifted' if variable == 'dm' else 'seismic'
                if variable == 'dm':
                    gt_vmin, gt_vmax = 0.0, 360.0
                    diff_vmin, diff_vmax = -180.0, 180.0

                print(f"  [{variable}] local GT=[{gt_vmin:.3f},{gt_vmax:.3f}] diff=[{diff_vmin:.3f},{diff_vmax:.3f}]")

                # Build xarray Datasets (lon/lat as coords — plot_wave_data format)
                def _make_ds(values):
                    return xr.Dataset(
                        {variable: (['node'], values)},
                        coords={
                            'lon': (['node'], self.lons),
                            'lat': (['node'], self.lats),
                            'time': timestamp,
                        }
                    )

                ds_gt   = _make_ds(gt_values)
                ds_mean = _make_ds(ensemble_mean)
                ds_std  = _make_ds(ensemble_std)
                ds_s1   = _make_ds(sample1)
                ds_s2   = _make_ds(sample2)
                ds_diff = _make_ds(difference)

                panels = [
                    (ds_gt,   gt_vmin,   gt_vmax,   val_cmap,  f'{var_title} (Ground Truth)\n{timestamp}'),
                    (ds_mean, gt_vmin,   gt_vmax,   val_cmap,  f'{var_title} (Ensemble Mean)\n{timestamp}'),
                    (ds_std,  std_vmin,  std_vmax,  'viridis', f'{var_title} (Ensemble Std)\n{timestamp}'),
                    (ds_s1,   gt_vmin,   gt_vmax,   val_cmap,  f'{var_title} (Sample {s_idx[0]+1})\n{timestamp}'),
                    (ds_s2,   gt_vmin,   gt_vmax,   val_cmap,  f'{var_title} (Sample {s_idx[1]+1})\n{timestamp}'),
                    (ds_diff, diff_vmin, diff_vmax, diff_cmap, f'{var_title} (Mean - GT)\n{timestamp}'),
                ]

                region_tag = f"lon{min_lon:.0f}-{max_lon:.0f}_lat{min_lat:.0f}-{max_lat:.0f}"

                # ---- Scatter version ----
                fig = plt.figure(figsize=(36, 18))
                for i, (ds, vmin, vmax, cmap, title) in enumerate(panels):
                    ax = fig.add_subplot(2, 3, i + 1, projection=ccrs.PlateCarree())
                    plot_wave_data(
                        dataset=ds,
                        variable=variable,
                        region=list(region),
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmap,
                        ax=ax,
                        return_ax=True,
                        scatter_size=scatter_size,
                        custom_title=title,
                        show_cities=False,
                        use_high_res_coastline=use_high_res_coastline,
                        use_satellite_background=use_satellite_background,
                    )
                plt.tight_layout()
                if save_dir:
                    sp = os.path.join(save_dir, f"region_{timestamp}_{variable}_{region_tag}_scatter.jpg")
                    plt.savefig(sp, dpi=150, bbox_inches='tight', facecolor='white')
                    print(f"  Saved scatter: {sp}")
                plt.close(fig)
                figures.append(fig)

                # ---- Interpolated version ----
                if show_interpolation:
                    from scipy.interpolate import griddata
                    from scipy.spatial import cKDTree

                    fig_interp = plt.figure(figsize=(36, 18))
                    grid_res = 400
                    lon_grid = np.linspace(min_lon, max_lon, grid_res)
                    lat_grid = np.linspace(min_lat, max_lat, grid_res)
                    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

                    # Pre-filter nodes inside region for interpolation source
                    lons_r = self.lons[region_mask]
                    lats_r = self.lats[region_mask]

                    for i, (ds, vmin, vmax, cmap, title) in enumerate(panels):
                        ax = fig_interp.add_subplot(2, 3, i + 1, projection=ccrs.PlateCarree())
                        ax.set_extent(list(region), crs=ccrs.PlateCarree())

                        # Get values in region
                        vals_r = ds[variable].values[region_mask]
                        valid = ~np.isnan(vals_r)
                        if valid.sum() == 0:
                            ax.set_title(title, fontsize=16)
                            continue

                        # Linear interpolation; dm 循环面板（值 0-360°、误差 ±180°）
                        # 在 sin/cos 空间插值，避免环绕处线性插值产生断裂带
                        circular_panel = variable == 'dm' and cmap != 'viridis'
                        if circular_panel:
                            rad_src = np.deg2rad(vals_r[valid].astype(np.float64))
                            src_comps = [np.sin(rad_src), np.cos(rad_src)]
                        else:
                            src_comps = [vals_r[valid]]

                        interp_comps = [
                            griddata(
                                (lons_r[valid], lats_r[valid]),
                                comp,
                                (lon_mesh, lat_mesh),
                                method='linear',
                                fill_value=np.nan,
                            )
                            for comp in src_comps
                        ]

                        # Fill nearby NaNs with nearest neighbor
                        nan_mask = np.isnan(interp_comps[0])
                        if np.any(nan_mask):
                            tree = cKDTree(np.column_stack([lons_r[valid], lats_r[valid]]))
                            dists, _ = tree.query(np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()]))
                            dists = dists.reshape(lon_mesh.shape)
                            fill = nan_mask & (dists < 0.5)
                            if np.any(fill):
                                for k, comp in enumerate(src_comps):
                                    nearest_vals = griddata(
                                        (lons_r[valid], lats_r[valid]),
                                        comp,
                                        (lon_mesh, lat_mesh),
                                        method='nearest',
                                    )
                                    interp_comps[k][fill] = nearest_vals[fill]

                        if circular_panel:
                            interp_vals = np.rad2deg(np.arctan2(interp_comps[0], interp_comps[1]))
                            if cmap == val_cmap:
                                interp_vals = interp_vals % 360.0
                        else:
                            interp_vals = interp_comps[0]

                        if variable in ('hs', 'elev'):
                            interp_vals = np.clip(interp_vals, 0, None)

                        masked = np.ma.masked_invalid(interp_vals)
                        im = ax.pcolormesh(lon_mesh, lat_mesh, masked,
                                           cmap=cmap, shading='auto',
                                           transform=ccrs.PlateCarree(),
                                           vmin=vmin, vmax=vmax)
                        ax.add_feature(cfeature.LAND, color='whitesmoke', alpha=1, zorder=3)
                        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, zorder=4)
                        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5, alpha=0.7, zorder=4)
                        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
                        var_label = var_prop.get('plot_label', variable)
                        cbar.set_label(var_label, fontsize=12)
                        # draw_labels=True 在当前 cartopy/shapely 版本组合下会
                        # 触发 gridliner 的 LinearRing 崩溃,标签关闭、网格线保留
                        gl = ax.gridlines(draw_labels=False, alpha=0.3, zorder=5)
                        ax.set_title(title, fontsize=16)

                    plt.tight_layout()
                    if save_dir:
                        sp = os.path.join(save_dir, f"region_{timestamp}_{variable}_{region_tag}_interp.jpg")
                        plt.savefig(sp, dpi=150, bbox_inches='tight', facecolor='white')
                        print(f"  Saved interp: {sp}")
                    plt.close(fig_interp)
                    figures.append(fig_interp)

            # ---- ERA5 input (wind + pressure) visualization ----
            if era5_dir is not None:
                self._save_era5_region_figure(
                    timestamp=timestamp,
                    region=region,
                    region_tag=region_tag,
                    era5_dir=era5_dir,
                    save_dir=save_dir,
                )

        print(f"\n✓ Generated {len(figures)} region plot(s).")
        return figures

    def _save_era5_region_figure(self, timestamp, region, region_tag, era5_dir, save_dir):
        """Generate and save a figure showing ERA5 wind and pressure input for a region."""
        import cartopy.feature as cfeature
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        from modules.utils import ERA5_VARIABLE_NAMES

        year = timestamp[:4]
        era5_path = os.path.join(era5_dir, year, f'era5_{timestamp}.nc')
        if not os.path.exists(era5_path):
            print(f"  Warning: ERA5 file not found: {era5_path} — skipping ERA5 viz.")
            return

        try:
            ds = xr.load_dataset(era5_path, engine='netcdf4')
        except Exception as e:
            print(f"  Warning: failed to load ERA5: {e} — skipping ERA5 viz.")
            return

        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'

        min_lon, max_lon, min_lat, max_lat = region

        # Convert lon from [-180,180] to [0,360] for ERA5
        era5_min_lon = min_lon if min_lon >= 0 else min_lon + 360
        era5_max_lon = max_lon if max_lon >= 0 else max_lon + 360

        # Crop to region
        ds = ds.where(
            (ds[lon_name] >= era5_min_lon) & (ds[lon_name] <= era5_max_lon) &
            (ds[lat_name] >= min_lat) & (ds[lat_name] <= max_lat),
            drop=True
        )
        if ds[lat_name].values[0] < ds[lat_name].values[-1]:
            ds = ds.sortby(lat_name, ascending=False)

        u_var = ERA5_VARIABLE_NAMES[0]
        v_var = ERA5_VARIABLE_NAMES[1]
        p_var = ERA5_VARIABLE_NAMES[2]

        u_data = ds[u_var].values
        v_data = ds[v_var].values
        p_data = ds[p_var].values / 100.0
        wind_speed = np.sqrt(u_data ** 2 + v_data ** 2)

        lat = ds[lat_name].values
        lon_era5 = ds[lon_name].values
        # Convert ERA5 lon back to [-180,180] for plotting
        lon_plot = np.where(lon_era5 > 180, lon_era5 - 360, lon_era5)
        Lon, Lat = np.meshgrid(lon_plot, lat)

        # Symmetric wind colorbar range
        max_abs_u = float(np.nanmax(np.abs(u_data)))
        max_abs_v = float(np.nanmax(np.abs(v_data)))
        wind_vmax = max(max_abs_u, max_abs_v)
        wind_vmin = -wind_vmax

        # Pressure range
        p_vmin = float(np.nanmin(p_data))
        p_vmax = float(np.nanmax(p_data))

        # Wind speed range
        ws_vmin = 0.0
        ws_vmax = float(np.nanmax(wind_speed))

        fig = plt.figure(figsize=(36, 18))
        proj = ccrs.PlateCarree()

        titles = [
            f'10m U-Wind (East-West)\n{timestamp}',
            f'10m V-Wind (North-South)\n{timestamp}',
            f'10m Wind Speed\n{timestamp}',
            f'MSL Pressure (Contour)\n{timestamp}',
            f'MSL Pressure (Color Fill)\n{timestamp}',
            f'10m Wind Vectors\n{timestamp}',
        ]

        for idx in range(6):
            ax = fig.add_subplot(2, 3, idx + 1, projection=proj)
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=proj)
            ax.add_feature(cfeature.LAND, color='whitesmoke', alpha=1, zorder=3)
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, zorder=4)
            ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5, alpha=0.7, zorder=4)

            gl = ax.gridlines(draw_labels=True, alpha=0.3, zorder=5)
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size': 10}

            if idx == 0:
                im = ax.pcolormesh(Lon, Lat, u_data, cmap='RdBu_r',
                                   transform=proj, shading='auto',
                                   vmin=wind_vmin, vmax=wind_vmax)
                plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label='m/s')
            elif idx == 1:
                im = ax.pcolormesh(Lon, Lat, v_data, cmap='RdBu_r',
                                   transform=proj, shading='auto',
                                   vmin=wind_vmin, vmax=wind_vmax)
                plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label='m/s')
            elif idx == 2:
                im = ax.pcolormesh(Lon, Lat, wind_speed, cmap='viridis',
                                   transform=proj, shading='auto',
                                   vmin=ws_vmin, vmax=ws_vmax)
                plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label='m/s')
            elif idx == 3:
                p_levels = np.linspace(p_vmin, p_vmax, 20)
                cf = ax.contourf(Lon, Lat, p_data, levels=p_levels,
                                 cmap='RdYlBu_r', transform=proj, alpha=0.9, extend='both')
                ax.contour(Lon, Lat, p_data, levels=p_levels[::2],
                           colors='black', linewidths=0.4, transform=proj, alpha=0.5)
                plt.colorbar(cf, ax=ax, shrink=0.8, pad=0.02, label='hPa')
            elif idx == 4:
                im = ax.pcolormesh(Lon, Lat, p_data, cmap='RdYlBu_r',
                                   transform=proj, shading='auto',
                                   vmin=p_vmin, vmax=p_vmax)
                plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label='hPa')
            elif idx == 5:
                subsample = max(1, min(len(lat), len(lon_plot)) // 15)
                q = ax.quiver(Lon[::subsample, ::subsample],
                              Lat[::subsample, ::subsample],
                              u_data[::subsample, ::subsample],
                              v_data[::subsample, ::subsample],
                              transform=proj, alpha=0.8, width=0.002)
                ax.quiverkey(q, 0.92, 0.05, wind_vmax,
                             f'{wind_vmax:.1f} m/s', labelpos='E',
                             coordinates='axes')

            ax.set_title(titles[idx], fontsize=16)

        plt.tight_layout()
        era5_save_path = os.path.join(save_dir, f"region_{timestamp}_era5_input_{region_tag}.jpg")
        plt.savefig(era5_save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved ERA5 input: {era5_save_path}")

    def visualize_spatial_rmse(self,
                               data_dir=None,
                               save_path=None,
                               variable='hs',
                               num_ensemble_samples=None,
                               max_workers=None):
        """
        Compute per-node RMSE aggregated over all timestamps and plot both:
            (1) raw per-node RMSE
            (2) normalized per-node RMSE = RMSE_i / std_t(gt_t,i)
        on a map.

        Args:
            data_dir: Directory containing ground truth files (defaults to self.gt_data_dir)
            save_path: Either a file path (used as base; '_raw'/'_norm' suffixes are appended)
                or None (defaults to <sample_dir parent>/spatial_rmse_<var>.jpg).
            variable: Variable name to compute RMSE for (default: 'hs')
            num_ensemble_samples: Number of ensemble members to average per timestamp
            max_workers: Number of parallel processes

        Returns:
            dict with 'rmse_raw', 'rmse_norm', 'gt_std_per_node', 'count'.
        """
        if data_dir is None:
            data_dir = self.gt_data_dir

        ensemble_files = [
            f for f in self.sample_files
            if os.path.basename(f).startswith('sample_')
            and '_ens' in os.path.basename(f)
            and 'T' in os.path.basename(f)
        ]
        if not ensemble_files:
            print(f"No ensemble sample files found in {self.sample_dir}.")
            return None

        from collections import defaultdict
        timestamp_groups = defaultdict(list)
        for sample_file in ensemble_files:
            basename = os.path.basename(sample_file).replace('.npz', '')
            parts = basename.split('_ens')
            if len(parts) == 2:
                timestamp = parts[0].replace('sample_', '')
                timestamp_groups[timestamp].append(sample_file)

        print(f"Computing per-node RMSE across {len(timestamp_groups)} timestamps...")

        if max_workers is None:
            max_workers = min(48, (os.cpu_count() or 1) + 2)

        # Online accumulators
        sum_sq = np.zeros(self.num_nodes, dtype=np.float64)        # for RMSE
        count = np.zeros(self.num_nodes, dtype=np.int64)
        sum_gt = np.zeros(self.num_nodes, dtype=np.float64)        # for GT mean
        sum_gt_sq = np.zeros(self.num_nodes, dtype=np.float64)     # for GT std

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_ts = {
                executor.submit(
                    process_single_timestamp_pernode_se_standalone,
                    timestamp,
                    files,
                    data_dir,
                    variable,
                    num_ensemble_samples,
                    self.num_nodes,
                    self.gt_file_format,
                    self.region_mode,
                    self.gt_filename_prefix,
                    self.gt_var_mapping,
                ): timestamp
                for timestamp, files in timestamp_groups.items()
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_ts),
                               total=len(timestamp_groups),
                               desc="Aggregating spatial RMSE"):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Error: {e}")
                    continue
                if result is None:
                    continue
                vmask = result['valid_mask']
                sum_sq[vmask] += result['sq_err'][vmask]
                gt_vals = result['gt_values'][vmask]
                sum_gt[vmask] += gt_vals
                sum_gt_sq[vmask] += gt_vals ** 2
                count[vmask] += 1

        with np.errstate(invalid='ignore', divide='ignore'):
            mean_sq = np.where(count > 0, sum_sq / np.maximum(count, 1), np.nan)
            per_node_rmse = np.sqrt(mean_sq)

            gt_mean = np.where(count > 0, sum_gt / np.maximum(count, 1), np.nan)
            gt_var = np.where(count > 1,
                              sum_gt_sq / np.maximum(count, 1) - gt_mean ** 2,
                              np.nan)
            gt_var = np.maximum(gt_var, 0.0)
            gt_std = np.sqrt(gt_var)

            # Normalized RMSE per node = RMSE / std_t(gt). Avoid div-by-zero.
            with np.errstate(invalid='ignore', divide='ignore'):
                rmse_norm = np.where(gt_std > 1e-8, per_node_rmse / gt_std, np.nan)

        per_node_rmse[count == 0] = np.nan
        rmse_norm[count == 0] = np.nan

        # Resolve save paths: append _raw and _norm to the base
        if save_path is None:
            parent_dir = os.path.dirname(os.path.abspath(self.sample_dir.rstrip(os.sep)))
            save_path = os.path.join(parent_dir, f'spatial_rmse_{variable}.jpg')
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        base, ext = os.path.splitext(save_path)
        if not ext:
            ext = '.jpg'
        path_raw = f'{base}_raw{ext}'
        path_norm = f'{base}_norm{ext}'

        from modules.utils import WAVE_VARIABLES
        var_prop = WAVE_VARIABLES.get(variable, {'plot_title': variable})
        var_title = var_prop.get('plot_title', variable)

        n_ts = int(count.max()) if count.size else 0

        def _plot_field(values, save_to, title, vmin, vmax, cmap):
            if self.region_mode == 'global':
                from modules.plot.plot_wave import interpolate_wave_data_to_grid
                import cartopy.feature as cfeature
                from numpy import ma

                lon = self.lons.copy()
                lat = self.lats.copy()
                # Wrap to -180..180 for plotting
                lon = np.where(lon > 180, lon - 360, lon)

                # Compact figure: scatter (left) + interpolated (right) + shared colorbar
                proj = ccrs.Robinson(central_longitude=0)
                fig = plt.figure(figsize=(11, 4.2))
                gs = fig.add_gridspec(
                    1, 2,
                    width_ratios=[1, 1],
                    wspace=-0.05,
                )
                ax1 = fig.add_subplot(gs[0, 0], projection=proj)
                ax2 = fig.add_subplot(gs[0, 1], projection=proj)

                # Left: scatter of raw per-node values
                sc = ax1.scatter(
                    lon, lat, c=values, cmap=cmap, s=0.5,
                    edgecolors='none', transform=ccrs.PlateCarree(),
                    vmin=vmin, vmax=vmax,
                )
                ax1.add_feature(cfeature.LAND, color='lightgray', alpha=0.35, zorder=5)
                ax1.add_feature(cfeature.COASTLINE, linewidth=0.4, alpha=0.8, zorder=6)
                ax1.set_global()
                ax1.set_title('Scatter (per node)', fontsize=11)
                gl1 = ax1.gridlines(draw_labels=False, alpha=0.25,
                                    xlocs=range(-180, 181, 60),
                                    ylocs=range(-90, 91, 30))
                gl1.top_labels = False
                gl1.right_labels = False
                gl1.xlabel_style = {'size': 7}
                gl1.ylabel_style = {'size': 7}

                # Right: interpolated field
                lon_mesh, lat_mesh, values_interp = interpolate_wave_data_to_grid(
                    lon, lat, values, grid_resolution=400, max_distance=0.5,
                )
                # Don't clip — RMSE is non-negative already; let cmap handle range
                vmasked = ma.masked_invalid(values_interp)
                ax2.pcolormesh(
                    lon_mesh, lat_mesh, vmasked, cmap=cmap, shading='auto',
                    transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax,
                )
                ax2.add_feature(cfeature.LAND, color='whitesmoke', alpha=1.0, zorder=10)
                ax2.add_feature(cfeature.COASTLINE, linewidth=0.6, alpha=1.0, zorder=11)
                ax2.set_global()
                ax2.set_title('Interpolated', fontsize=11)
                gl2 = ax2.gridlines(draw_labels=False, alpha=0.25,
                                    xlocs=range(-180, 181, 60),
                                    ylocs=range(-90, 91, 30))
                gl2.top_labels = False
                gl2.right_labels = False
                gl2.left_labels = False
                gl2.xlabel_style = {'size': 7}
                gl2.ylabel_style = {'size': 7}

                # Reserve space: top for suptitle, bottom for colorbar
                fig.subplots_adjust(top=0.80, bottom=0.22, left=0.02, right=0.98)

                # Manually place a short, centered horizontal colorbar
                cax = fig.add_axes([0.36, 0.12, 0.30, 0.035])
                from modules.utils import WAVE_VARIABLES as _WV
                _vp = _WV.get(variable, {'plot_label': variable})
                cbar = fig.colorbar(sc, cax=cax, orientation='horizontal')
                cbar.set_label(_vp.get('plot_label', variable), fontsize=10)
                cbar.ax.tick_params(labelsize=9)

                # Suptitle
                fig.suptitle(title, fontsize=12, y=0.99)
                plt.savefig(save_to, dpi=180, facecolor='white')
                plt.close(fig)
                return None
            else:
                ds = xr.Dataset(
                    {variable: (['node'], values)},
                    coords={
                        'lon': (['node'], self.lons),
                        'lat': (['node'], self.lats),
                        'time': 'spatial_rmse',
                    },
                )
                return plot_wave_data(
                    dataset=ds,
                    variable=variable,
                    region=self.region,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    save_path=save_to,
                    use_high_res_coastline=True,
                    use_satellite_background=True,
                    show_interpolation=False,
                    custom_title=title,
                )

        # Raw RMSE plot
        if np.any(~np.isnan(per_node_rmse)):
            raw_vmax = float(np.nanpercentile(per_node_rmse, 99))
            raw_title = (f'{var_title} (Per-node RMSE — raw)\n'
                         f'Timestamps used: {n_ts} | Nodes: {self.num_nodes} | '
                         f'Mean: {np.nanmean(per_node_rmse):.4f}')
            _plot_field(per_node_rmse, path_raw, raw_title, 0.0, raw_vmax, 'magma_r')
            print(f"✓ Spatial RMSE (raw) saved to: {path_raw}")
            print(f"  Raw  — Mean: {np.nanmean(per_node_rmse):.4f} | "
                  f"Median: {np.nanmedian(per_node_rmse):.4f} | "
                  f"Min: {np.nanmin(per_node_rmse):.4f} | "
                  f"Max: {np.nanmax(per_node_rmse):.4f}")

        # Normalized RMSE plot (RMSE / std_t(GT))
        if np.any(~np.isnan(rmse_norm)):
            norm_vmax = float(np.nanpercentile(rmse_norm, 99))
            norm_title = (
                f'{var_title} (Per-node RMSE — normalized by GT temporal std)\n'
                f'Timestamps used: {n_ts} | Nodes: {self.num_nodes} | '
                f'Mean: {np.nanmean(rmse_norm):.4f}'
            )
            _plot_field(rmse_norm, path_norm, norm_title, 0.0, norm_vmax, 'magma_r')
            print(f"✓ Spatial RMSE (normalized) saved to: {path_norm}")
            print(f"  Norm — Mean: {np.nanmean(rmse_norm):.4f} | "
                  f"Median: {np.nanmedian(rmse_norm):.4f} | "
                  f"Min: {np.nanmin(rmse_norm):.4f} | "
                  f"Max: {np.nanmax(rmse_norm):.4f}")

        return {
            'rmse_raw': per_node_rmse,
            'rmse_norm': rmse_norm,
            'gt_std_per_node': gt_std,
            'count': count,
        }

    def visualize_temporal_rmse(self,
                                rmse_csv_path=None,
                                save_path=None,
                                variable='hs',
                                rmse_column='rmse',
                                data_dir=None,
                                max_workers=None):
        """
        Plot the temporal distribution of RMSE in two versions:
            (1) raw RMSE per timestamp
            (2) normalized RMSE per timestamp = RMSE_t / std_x(gt_t,x)
        Each plot includes a time series (with monthly mean overlay) and a histogram.

        Args:
            rmse_csv_path: Path to CSV from calculate_ensemble_rmse_and_export. Must
                contain 'timestamp' and `rmse_column`.
            save_path: Base path; '_raw' / '_norm' suffixes are appended. If None,
                derived from rmse_csv_path.
            variable: Variable name (used for axis labels/title and to load GT).
            rmse_column: RMSE column name in the CSV.
            data_dir: Ground truth directory (defaults to self.gt_data_dir).
            max_workers: Parallel workers for GT-stats computation.

        Returns:
            DataFrame with columns ['timestamp','datetime',rmse_column,'gt_std','rmse_norm']
        """
        if rmse_csv_path is None or not os.path.exists(rmse_csv_path):
            raise FileNotFoundError(
                f"RMSE CSV not found: {rmse_csv_path}. Run calculate_ensemble_rmse_and_export first."
            )
        if data_dir is None:
            data_dir = self.gt_data_dir

        df = pd.read_csv(rmse_csv_path)
        if 'timestamp' not in df.columns:
            raise ValueError(f"CSV missing 'timestamp' column: {rmse_csv_path}")
        if rmse_column not in df.columns:
            raise ValueError(
                f"CSV missing '{rmse_column}' column. Available: {list(df.columns)}"
            )

        df['timestamp'] = df['timestamp'].astype(str)
        df['datetime'] = pd.to_datetime(df['timestamp'], format='%Y%m%dT%H', errors='coerce')
        df = df.dropna(subset=['datetime', rmse_column]).sort_values('datetime').reset_index(drop=True)
        if df.empty:
            print("No valid RMSE rows to plot.")
            return df

        # Compute per-timestamp GT std in parallel
        if max_workers is None:
            max_workers = min(48, (os.cpu_count() or 1) + 2)

        print(f"Computing GT spatial std for {len(df)} timestamps...")
        gt_stats = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_ts = {
                executor.submit(
                    process_single_timestamp_gt_stats_standalone,
                    ts,
                    data_dir,
                    variable,
                    self.num_nodes,
                    self.gt_file_format,
                    self.region_mode,
                    self.gt_filename_prefix,
                    self.gt_var_mapping,
                ): ts
                for ts in df['timestamp'].tolist()
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_ts),
                               total=len(future_to_ts),
                               desc="GT stats"):
                try:
                    res = future.result()
                except Exception:
                    continue
                if res is not None:
                    gt_stats[res['timestamp']] = res

        df['gt_std'] = df['timestamp'].map(lambda t: gt_stats.get(t, {}).get('gt_std', np.nan))
        with np.errstate(invalid='ignore', divide='ignore'):
            df['rmse_norm'] = np.where(df['gt_std'] > 1e-8,
                                        df[rmse_column] / df['gt_std'], np.nan)

        from modules.utils import WAVE_VARIABLES
        var_prop = WAVE_VARIABLES.get(variable, {'plot_title': variable})
        var_title = var_prop.get('plot_title', variable)

        if save_path is None:
            save_path = os.path.join(
                os.path.dirname(os.path.abspath(rmse_csv_path)),
                f'temporal_rmse_{variable}.jpg'
            )
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        base, ext = os.path.splitext(save_path)
        if not ext:
            ext = '.jpg'
        path_raw = f'{base}_raw{ext}'
        path_norm = f'{base}_norm{ext}'

        def _plot_temporal(values_series, save_to, ylabel, header):
            vals = values_series.values
            valid = ~np.isnan(vals)
            if not np.any(valid):
                print(f"No valid values for {save_to}; skipping.")
                return
            mean_v = float(np.nanmean(vals))
            median_v = float(np.nanmedian(vals))
            min_v = float(np.nanmin(vals))
            max_v = float(np.nanmax(vals))

            # Build a temp DataFrame indexed by datetime for resampling
            tmp = pd.DataFrame({'v': vals}, index=df['datetime'])
            monthly = tmp['v'].resample('MS').mean()
            # 7-day rolling stats for a smooth band
            roll = tmp['v'].rolling('7D', min_periods=3)
            roll_mean = roll.mean()
            roll_q25 = roll.quantile(0.25)
            roll_q75 = roll.quantile(0.75)

            # Color palette
            c_pts = '#5B8FF9'      # soft blue dots
            c_band = '#9EC5FE'     # lighter blue band
            c_roll = '#1F4E96'     # deep blue rolling mean
            c_month = '#E84A5F'    # warm red monthly mean
            c_mean = '#444444'     # neutral overall mean
            c_hist = '#5B8FF9'

            # Compact figure (default fonts feel larger)
            fig = plt.figure(figsize=(9.5, 5.6))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.1], hspace=0.32)
            ax = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])

            # Scatter cloud of per-timestamp values (light)
            ax.scatter(df['datetime'].values, vals,
                       s=4, c=c_pts, alpha=0.18, edgecolors='none',
                       label='Per-timestamp', rasterized=True)
            # 7-day IQR band
            ax.fill_between(roll_mean.index, roll_q25.values, roll_q75.values,
                            color=c_band, alpha=0.45, linewidth=0,
                            label='7-day IQR')
            # 7-day rolling mean
            ax.plot(roll_mean.index, roll_mean.values,
                    color=c_roll, lw=1.6, label='7-day mean')
            # Monthly mean
            ax.plot(monthly.index, monthly.values,
                    color=c_month, lw=2.2, marker='o', ms=5,
                    markerfacecolor='white', markeredgewidth=1.5,
                    label='Monthly mean')
            # Overall mean
            ax.axhline(mean_v, color=c_mean, ls='--', lw=1.0, alpha=0.7,
                       label=f'Overall mean = {mean_v:.4f}')

            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(
                f'{header} — {var_title}\n'
                f'N={int(np.sum(valid))} | mean={mean_v:.4f} | '
                f'median={median_v:.4f} | min={min_v:.4f} | max={max_v:.4f}',
                fontsize=11, pad=8,
            )
            ax.grid(True, axis='y', alpha=0.25, linestyle='--')
            ax.set_axisbelow(True)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.spines['left'].set_color('#888')
            ax.spines['bottom'].set_color('#888')
            ax.tick_params(colors='#444', labelsize=9)
            ax.legend(loc='upper right', frameon=False, fontsize=9, ncol=2)
            # Tight y-limits with small padding
            ymin = float(np.nanpercentile(vals, 0.5))
            ymax = float(np.nanpercentile(vals, 99.5))
            pad = (ymax - ymin) * 0.08
            ax.set_ylim(ymin - pad, ymax + pad)

            # Histogram
            n, bins, patches = ax2.hist(vals[valid], bins=40, color=c_hist,
                                        alpha=0.85, edgecolor='white', linewidth=0.6)
            ax2.axvline(mean_v, color=c_month, ls='--', lw=1.4,
                        label=f'mean = {mean_v:.4f}')
            ax2.axvline(median_v, color=c_mean, ls=':', lw=1.4,
                        label=f'median = {median_v:.4f}')
            ax2.set_xlabel(ylabel, fontsize=10)
            ax2.set_ylabel('Count', fontsize=10)
            ax2.grid(True, axis='y', alpha=0.25, linestyle='--')
            ax2.set_axisbelow(True)
            for s in ('top', 'right'):
                ax2.spines[s].set_visible(False)
            ax2.spines['left'].set_color('#888')
            ax2.spines['bottom'].set_color('#888')
            ax2.tick_params(colors='#444', labelsize=9)
            ax2.legend(loc='upper right', frameon=False, fontsize=9)

            plt.savefig(save_to, dpi=170, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"✓ Temporal RMSE saved to: {save_to}")

        # Raw
        _plot_temporal(df[rmse_column], path_raw,
                       ylabel=f'RMSE ({var_title})',
                       header='Temporal RMSE Distribution (raw)')

        # Normalized
        _plot_temporal(df['rmse_norm'], path_norm,
                       ylabel=f'RMSE / GT spatial std ({var_title})',
                       header='Temporal RMSE Distribution (normalized by GT spatial std)')

        return df