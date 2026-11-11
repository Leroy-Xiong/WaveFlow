import xarray as xr
import os
import glob
import pandas as pd
import numpy as np
from modules.plot.plot_wave import plot_wave_data
from multiprocessing import Pool, cpu_count
from functools import partial
from modules.datasets.coordinate_utils import export_coordinates, cluster_coordinates

class GlobalHindcastProcessor():
    def __init__(self):
        self.data_dir = "./data/global_hindcast/raw/"
        
    def load_data(self, start_time, end_time, lat_range, lon_range):
        # Parse start and end time
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        
        # Determine which files need to be loaded based on the year range
        start_year = start_dt.year
        end_year = end_dt.year
        
        # Calculate the decade ranges for file names
        # Files are named like: mentaschi2023swh_1950-1959.nc, mentaschi2023swh_1960-1969.nc, etc.
        start_decade = (start_year // 10) * 10
        end_decade = (end_year // 10) * 10
        
        # Collect all needed file paths
        file_paths = []
        for decade_start in range(start_decade, end_decade + 1, 10):
            decade_end = decade_start + 9
            nc_file_path = os.path.join(self.data_dir, f'mentaschi2023swh_{decade_start}-{decade_end}.nc')
            if os.path.exists(nc_file_path):
                file_paths.append(nc_file_path)
            else:
                print(f"Warning: File not found: {nc_file_path}")
        
        if not file_paths:
            raise FileNotFoundError(f"No data files found for the year range {start_year}-{end_year}")
        
        # Load and concatenate multiple files if needed
        # Use h5netcdf engine as netcdf4 engine may encounter HDF errors with this file
        if len(file_paths) == 1:
            ds = xr.open_dataset(file_paths[0], engine='h5netcdf')
        else:
            datasets = [xr.open_dataset(fp, engine='h5netcdf') for fp in file_paths]
            ds = xr.concat(datasets, dim='time')
            # Close individual datasets to free memory
            for dataset in datasets:
                dataset.close()
        
        # Filter by time with hour precision
        ds = ds.sel(time=slice(start_dt, end_dt))
        
        # Filter by latitude and longitude
        mask = (ds.y >= lat_range[0]) & (ds.y <= lat_range[1]) & \
               (ds.x >= lon_range[0]) & (ds.x <= lon_range[1])
        ds = ds.where(mask, drop=True)
        
        # print(ds)
        return ds
    
    def plot_random_samples_stats(self, start_time='1950-01-01', end_time='2022-12-31', n=1000, variable='swh', 
                                  lat_range=None, lon_range=None, save_dir=None, dpi=150, cmap='turbo', seed=None):
        """
        Randomly sample n timesteps, calculate mean and standard deviation maps, and plot them.
        
        Args:
            start_time (str): Start time in format 'YYYY-MM-DD'
            end_time (str): End time in format 'YYYY-MM-DD'
            n (int): Number of random timesteps to sample
            variable (str): Variable name to plot
            lat_range (list): [lat_min, lat_max]. If None, uses global range [-90, 90]
            lon_range (list): [lon_min, lon_max]. If None, uses global range [-180, 180]
            save_dir (str): Directory to save figures
            dpi (int): DPI for figures
            cmap (str): Colormap
            seed (int): Random seed
            
        Returns:
            list: [fig_mean, fig_std]
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Set default global ranges if not specified
        if lat_range is None:
            lat_range = [-90, 90]
        if lon_range is None:
            lon_range = [-180, 180]
            
        # Get available time points
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        start_year = start_dt.year
        end_year = end_dt.year
        start_decade = (start_year // 10) * 10
        end_decade = (end_year // 10) * 10
        
        file_paths = []
        for decade_start in range(start_decade, end_decade + 1, 10):
            decade_end = decade_start + 9
            nc_file_path = os.path.join(self.data_dir, f'mentaschi2023swh_{decade_start}-{decade_end}.nc')
            if os.path.exists(nc_file_path):
                file_paths.append(nc_file_path)
        
        if not file_paths:
            raise FileNotFoundError(f"No data files found for the year range {start_year}-{end_year}")
            
        # Load only time coordinate
        if len(file_paths) == 1:
            ds_time_only = xr.open_dataset(file_paths[0], engine='h5netcdf')[['time']]
        else:
            time_datasets = [xr.open_dataset(fp, engine='h5netcdf')[['time']] for fp in file_paths]
            ds_time_only = xr.concat(time_datasets, dim='time')
            for dataset in time_datasets:
                dataset.close()
                
        ds_time_only = ds_time_only.sel(time=slice(start_dt, end_dt))
        available_times = ds_time_only.time.values
        n_times = len(available_times)
        ds_time_only.close()
        
        if n_times == 0:
            print(f"No data available in the specified time range.")
            return []
            
        # Random sample
        n_sample = min(n, n_times)
        random_indices = np.random.choice(n_times, size=n_sample, replace=False)
        selected_times = available_times[random_indices]
        
        print(f"Sampling {n_sample} timesteps for statistics...")
        
        # Load data
        datasets = []
        for i, time_point in enumerate(selected_times):
            time_str = pd.Timestamp(time_point).strftime('%Y-%m-%dT%H:%M:%S')
            try:
                ds = self.load_data(time_str, time_str, lat_range=lat_range, lon_range=lon_range)
                datasets.append(ds)
            except Exception as e:
                print(f"Error loading time {time_str}: {e}")
                
        if not datasets:
            return []
            
        # Concatenate
        combined = xr.concat(datasets, dim='time')
        
        # Prepare variable name - rename coordinates to lon/lat if needed
        # For unstructured grid data, x and y are data variables, not coordinates
        # We need to set them as coordinates first
        if 'x' in combined and 'y' in combined:
            combined = combined.set_coords(['x', 'y'])
        
        rename_dict = {}
        if 'x' in combined.coords:
            rename_dict['x'] = 'lon'
        if 'y' in combined.coords:
            rename_dict['y'] = 'lat'
        
        ds_plot = combined.rename(rename_dict) if rename_dict else combined
        
        if variable == 'swh':
            if 'swh' in ds_plot:
                ds_plot = ds_plot.rename({'swh': 'hs'})
                plot_var = 'hs'
            elif 'WWM_1' in ds_plot:
                ds_plot = ds_plot.rename({'WWM_1': 'hs'})
                plot_var = 'hs'
            else:
                plot_var = variable
        else:
            plot_var = variable
            
        if plot_var not in ds_plot:
            data_vars = list(ds_plot.data_vars)
            if data_vars:
                plot_var = data_vars[0]
                print(f"Variable not found, using {plot_var}")
            else:
                raise KeyError("No variable found to plot")

        # Calculate stats
        print("Calculating mean and standard deviation...")
        
        # Save lon/lat coordinates before aggregation (they will be lost during mean/std)
        # Take the mean of lon/lat along time dimension if they vary with time
        lon_coord = ds_plot['lon'].mean(dim='time') if 'time' in ds_plot['lon'].dims else ds_plot['lon']
        lat_coord = ds_plot['lat'].mean(dim='time') if 'time' in ds_plot['lat'].dims else ds_plot['lat']
        
        mean_ds = ds_plot.mean(dim='time', keep_attrs=True)
        std_ds = ds_plot.std(dim='time', keep_attrs=True)
        
        # Restore lon/lat as coordinates
        mean_ds = mean_ds.assign_coords({'lon': lon_coord, 'lat': lat_coord})
        std_ds = std_ds.assign_coords({'lon': lon_coord, 'lat': lat_coord})
        
        figures = []
        
        # Plot Mean
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path_mean = os.path.join(save_dir, f'global_hindcast_{plot_var}_mean_n{n_sample}.png')
        else:
            save_path_mean = None
            
        title_mean = f'Global Hindcast - {plot_var.upper()} Mean\n(N={n_sample} samples)'
        fig_mean = plot_wave_data(
            mean_ds,
            variable=plot_var,
            region=(lon_range[0], lon_range[1], lat_range[0], lat_range[1]),
            save_path=save_path_mean,
            dpi=dpi,
            cmap=cmap,
            custom_title=title_mean,
            show_interpolation=True,
            scatter_size=1
        )
        figures.append(fig_mean)
        print(f"Mean plot created.")
        
        # Plot Std
        if save_dir:
            save_path_std = os.path.join(save_dir, f'global_hindcast_{plot_var}_std_n{n_sample}.png')
        else:
            save_path_std = None
            
        title_std = f'Global Hindcast - {plot_var.upper()} Std Dev\n(N={n_sample} samples)'
        fig_std = plot_wave_data(
            std_ds,
            variable=plot_var,
            region=(lon_range[0], lon_range[1], lat_range[0], lat_range[1]),
            save_path=save_path_std,
            dpi=dpi,
            cmap=cmap,
            custom_title=title_std,
            show_interpolation=True,
            scatter_size=1
        )
        figures.append(fig_std)
        print(f"Std plot created.")
        
        # Cleanup
        combined.close()
        for ds in datasets:
            ds.close()
            
        return figures
    
    def visualize_random_timesteps(self, start_time='1950-01-01', end_time='2022-12-31', n=10, variable='swh', 
                                   lat_range=None, lon_range=None, save_dir=None, dpi=150, vmin=None, vmax=None, 
                                   cmap='turbo', scatter_size=5, seed=None):
        """
        Randomly visualize n timesteps of global hindcast data.
        
        Args:
            start_time (str): Start time in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' (default: '1950-01-01')
            end_time (str): End time in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' (default: '2022-12-31')
            n (int): Number of random timesteps to visualize (default: 5)
            variable (str): Variable name to plot (default: 'swh' for significant wave height)
            lat_range (list or tuple, optional): [lat_min, lat_max]. If None, uses global range [-90, 90]
            lon_range (list or tuple, optional): [lon_min, lon_max]. If None, uses global range [-180, 180]
            save_dir (str, optional): Directory to save the figures. If None, displays plots
            dpi (int): Resolution of the output figures (default: 150)
            vmin (float, optional): Minimum value for color scale
            vmax (float, optional): Maximum value for color scale
            cmap (str): Colormap name (default: 'turbo')
            scatter_size (float or int): Size of scatter points (default: 15)
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            list: List of figure objects created
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Set default global ranges if not specified
        if lat_range is None:
            lat_range = [-90, 90]
        if lon_range is None:
            lon_range = [-180, 180]
        
        # First, get available time points without loading all data
        # We only need to load time coordinate to sample timesteps
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        start_year = start_dt.year
        end_year = end_dt.year
        start_decade = (start_year // 10) * 10
        end_decade = (end_year // 10) * 10
        
        # Collect all needed file paths
        file_paths = []
        for decade_start in range(start_decade, end_decade + 1, 10):
            decade_end = decade_start + 9
            nc_file_path = os.path.join(self.data_dir, f'mentaschi2023swh_{decade_start}-{decade_end}.nc')
            if os.path.exists(nc_file_path):
                file_paths.append(nc_file_path)
        
        if not file_paths:
            raise FileNotFoundError(f"No data files found for the year range {start_year}-{end_year}")
        
        # Load only time coordinate to get available timesteps
        if len(file_paths) == 1:
            ds_time_only = xr.open_dataset(file_paths[0], engine='h5netcdf')[['time']]
        else:
            time_datasets = [xr.open_dataset(fp, engine='h5netcdf')[['time']] for fp in file_paths]
            ds_time_only = xr.concat(time_datasets, dim='time')
            for dataset in time_datasets:
                dataset.close()
        
        # Filter by time range
        ds_time_only = ds_time_only.sel(time=slice(start_dt, end_dt))
        available_times = ds_time_only.time.values
        n_times = len(available_times)
        ds_time_only.close()
        
        if n_times == 0:
            print(f"No data available in the specified time range: {start_time} to {end_time}")
            return []
        
        # Randomly select n timesteps (or all if n > available)
        n_sample = min(n, n_times)
        random_indices = np.random.choice(n_times, size=n_sample, replace=False)
        random_indices = np.sort(random_indices)  # Sort for better visualization order
        selected_times = available_times[random_indices]
        
        print(f"Visualizing {n_sample} random timesteps from {n_times} available timesteps...")
        
        # Create visualizations by loading data one timestep at a time
        figures = []
        for idx, time_point in enumerate(selected_times):
            # Convert numpy datetime to pandas Timestamp for better formatting
            time_pd = pd.Timestamp(time_point)
            time_str = time_pd.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Load data for this single timestep only
            ds_single = self.load_data(time_str, time_str, lat_range=lat_range, lon_range=lon_range)
            
            # Get the data for this timestep
            ds_time = ds_single.isel(time=0)
            
            # Prepare dataset for plot_wave_data
            # Rename coordinates to match expected format (lon, lat)
            ds_plot = ds_time.rename({'x': 'lon', 'y': 'lat'})
            
            # Rename variable if needed (swh and WWM_1 are common names for significant wave height)
            if variable == 'swh':
                if 'swh' in ds_plot:
                    ds_plot = ds_plot.rename({'swh': 'hs'})
                    plot_var = 'hs'
                elif 'WWM_1' in ds_plot:
                    ds_plot = ds_plot.rename({'WWM_1': 'hs'})
                    plot_var = 'hs'
                else:
                    plot_var = variable
            else:
                plot_var = variable
            
            # Final check: if plot_var not in ds_plot, use the first available data variable
            if plot_var not in ds_plot:
                data_vars = list(ds_plot.data_vars)
                if data_vars:
                    old_var = plot_var
                    plot_var = data_vars[0]
                    print(f"Warning: Variable '{old_var}' not found in dataset. Using '{plot_var}' instead.")
                else:
                    raise KeyError(f"Variable '{plot_var}' not found and no other data variables available. "
                                   f"Available: {list(ds_plot.variables)}")
            
            # Create custom title
            time_str = pd.Timestamp(time_point).strftime('%Y-%m-%d %H:%M')
            custom_title = f'Global Hindcast - {plot_var.upper()}\nDate: {time_str}'
            
            # Set save path if save_dir is provided
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'global_hindcast_{plot_var}_{idx:04d}.png')
            else:
                save_path = None
            
            # Call plot_wave_data
            fig = plot_wave_data(
                ds_plot, 
                variable=plot_var,
                region=(lon_range[0], lon_range[1], lat_range[0], lat_range[1]),
                save_path=save_path,
                dpi=dpi,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                scatter_size=scatter_size,
                custom_title=custom_title,
                show_interpolation=True,
            )
            
            figures.append(fig)
            print(f"  - Visualized timestep {idx+1}/{n_sample}: {time_pd.strftime('%Y-%m-%d %H:%M')}")
            
            # Close dataset to free memory
            ds_single.close()
        
        print(f"\nCompleted visualization of {n_sample} timesteps.")
        if save_dir:
            print(f"Figures saved to: {save_dir}")
        
        return figures
    
    @staticmethod
    def _save_single_timestep(args):
        """
        Helper function to save a single timestep to a netCDF file.
        This function is designed to be used with multiprocessing.
        
        Args:
            args: Tuple of (ds_single, output_file, compression_level)
        """
        ds_single, output_file, compression_level = args
        
        # Remove any inherited encoding that might cause issues
        ds_single.encoding = {}
        for var in ds_single.data_vars:
            ds_single[var].encoding = {}
        for coord in ds_single.coords:
            ds_single[coord].encoding = {}
        
        # Configure compression encoding for all variables
        encoding = {}
        for var in ds_single.data_vars:
            encoding[var] = {'zlib': True, 'complevel': compression_level}
        
        # Also encode coordinates if needed (only actual dimensions, not scalar coordinates)
        for coord in ds_single.coords:
            # Only encode if it's an actual dimension in the dataset
            if coord in ds_single.dims and coord in ['x', 'y', 'lon', 'lat']:
                encoding[coord] = {'zlib': True, 'complevel': compression_level}
        
        # Save to netCDF file
        ds_single.to_netcdf(output_file, encoding=encoding, engine='h5netcdf')
        
        return output_file
    
    def split_to_daily_files(self, output_dir, compression_level=4, verbose=True, n_workers=None):
        """
        Split all data files into separate netCDF files, one per timestep.
        Processes all decade files in the data directory sequentially without loading all data at once.
        Uses multiprocessing to speed up file writing.
        
        Args:
            output_dir (str): Directory to save the individual timestep nc files
            compression_level (int): Compression level for output files (0-9, default: 4)
            verbose (bool): Print progress messages (default: True)
            n_workers (int, optional): Number of worker processes. If None, uses CPU count
            
        Returns:
            list: List of output file paths created
        """
        # Set number of workers
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)  # Leave one CPU free
        
        if verbose:
            print(f"Using {n_workers} worker processes")
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all available data files
        file_pattern = os.path.join(self.data_dir, 'mentaschi2023swh_*.nc')
        file_paths = sorted(glob.glob(file_pattern))
        
        if not file_paths:
            raise FileNotFoundError(f"No data files found in {self.data_dir}")
        
        if verbose:
            print(f"Found {len(file_paths)} data files to process")
            print(f"Output directory: {output_dir}")
            print("="*60)
        
        output_files = []
        total_timesteps_processed = 0
        
        # Process each file sequentially
        for file_idx, file_path in enumerate(file_paths):
            file_name = os.path.basename(file_path)
            
            if verbose:
                print(f"\n[File {file_idx+1}/{len(file_paths)}] Processing: {file_name}")
            
            # Load the current file and read all data into memory
            ds = xr.open_dataset(file_path, engine='h5netcdf')
            ds = ds.load()  # Load all data into memory
            
            # Get total number of timesteps
            n_timesteps = len(ds.time)
            
            if verbose:
                print(f"  Contains {n_timesteps} timesteps")
            
            # Prepare all tasks for this file
            tasks = []
            for time_idx in range(n_timesteps):
                # Select single timestep
                ds_single = ds.isel(time=time_idx)
                time_value = pd.Timestamp(ds_single.time.values)
                
                # Format datetime as string for filename (YYYYMMDD_HH)
                time_str = time_value.strftime('%Y%m%d_%H')
                output_file = os.path.join(output_dir, f'global_hindcast_{time_str}.nc')
                
                # Add task to list
                tasks.append((ds_single, output_file, compression_level))
            
            # Process tasks in parallel using multiprocessing
            if verbose:
                print(f"  Writing {n_timesteps} files using {n_workers} workers...")
            
            with Pool(processes=n_workers) as pool:
                if verbose:
                    from tqdm import tqdm
                    results = list(tqdm(pool.imap(self._save_single_timestep, tasks), 
                                       total=n_timesteps, 
                                       desc=f"  Processing {file_name}", 
                                       unit="file"))
                else:
                    results = pool.map(self._save_single_timestep, tasks)
            
            output_files.extend(results)
            total_timesteps_processed += n_timesteps
            
            # Close the current file to free memory
            ds.close()
            
            if verbose:
                print(f"  ✓ Completed {n_timesteps} timesteps from {file_name}")
        
        if verbose:
            print("="*60)
            print(f"\n✓ All files processed successfully!")
            print(f"  Total timesteps processed: {total_timesteps_processed}")
            print(f"  Output directory: {output_dir}")
        
        return output_files
    
    def compute_statistics_from_nc_files(self, nc_dir, n_samples=None, variable='swh', 
                                         output_file=None, seed=None, verbose=True):
        """
        Randomly sample netCDF files and compute statistics (min, max, mean, std) for a variable.
        
        Args:
            nc_dir (str): Directory containing the individual timestep nc files
            n_samples (int, optional): Number of files to randomly sample. If None, uses all files
            variable (str): Variable name to compute statistics for (default: 'swh')
            output_file (str, optional): Path to save statistics as JSON. If None, only returns dict
            seed (int, optional): Random seed for reproducibility
            verbose (bool): Print progress messages (default: True)
            
        Returns:
            dict: Dictionary containing statistics:
                  {
                      'variable': variable name,
                      'n_files_sampled': number of files used,
                      'n_total_points': total number of data points,
                      'min': minimum value,
                      'max': maximum value,
                      'mean': mean value,
                      'std': standard deviation,
                      'sampled_files': list of sampled file names
                  }
        """
        import json
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Find all nc files in the directory
        file_pattern = os.path.join(nc_dir, '*.nc')
        all_files = sorted(glob.glob(file_pattern))
        
        if not all_files:
            raise FileNotFoundError(f"No netCDF files found in {nc_dir}")
        
        # Sample files if n_samples is specified
        if n_samples is None or n_samples >= len(all_files):
            sampled_files = all_files
            if verbose:
                print(f"Using all {len(all_files)} files for statistics computation")
        else:
            sampled_indices = np.random.choice(len(all_files), size=n_samples, replace=False)
            sampled_files = [all_files[i] for i in sorted(sampled_indices)]
            if verbose:
                print(f"Randomly sampled {n_samples} files from {len(all_files)} total files")
        
        if verbose:
            print(f"Computing statistics for variable: {variable}")
            print("="*60)
        
        # Initialize arrays to collect all data
        all_values = []
        n_total_points = 0
        
        # Load each file and collect data
        if verbose:
            from tqdm import tqdm
            file_iterator = tqdm(sampled_files, desc="Processing files", unit="file")
        else:
            file_iterator = sampled_files
        
        for file_path in file_iterator:
            # Load dataset
            ds = xr.open_dataset(file_path, engine='h5netcdf')
            
            # Get the variable data
            if variable in ds:
                var_data = ds[variable].values
            elif variable == 'swh' and 'WWM_1' in ds:
                var_data = ds['WWM_1'].values
            elif variable == 'hs' and 'swh' in ds:
                var_data = ds['swh'].values
            else:
                ds.close()
                raise KeyError(f"Variable '{variable}' not found in {os.path.basename(file_path)}. "
                              f"Available variables: {list(ds.data_vars)}")
            
            # Flatten and remove NaN values
            var_data_flat = var_data.flatten()
            var_data_valid = var_data_flat[~np.isnan(var_data_flat)]
            
            all_values.append(var_data_valid)
            n_total_points += len(var_data_valid)
            
            ds.close()
        
        # Concatenate all values
        all_values = np.concatenate(all_values)
        
        # Compute basic statistics
        stats = {
            'variable': variable,
            'n_files_sampled': len(sampled_files),
            'n_total_files': len(all_files),
            'n_total_points': n_total_points,
            'min': float(np.min(all_values)),
            'max': float(np.max(all_values)),
            'mean': float(np.mean(all_values)),
            'std': float(np.std(all_values)),
            'sampled_files': [os.path.basename(f) for f in sampled_files]
        }
        
        # Compute log-transformed statistics for hs/swh variable
        # These are needed for NormalizeWaveData transform
        if variable in ['swh', 'hs']:
            log_values = np.log1p(all_values)  # log(1 + x)
            var_name = 'hs' if variable == 'swh' else variable
            stats[f'{var_name}_log_mean'] = float(np.mean(log_values))
            stats[f'{var_name}_log_std'] = float(np.std(log_values))
        
        if verbose:
            print("="*60)
            print(f"\nStatistics for variable '{variable}':")
            print(f"  Files sampled: {stats['n_files_sampled']} / {stats['n_total_files']}")
            print(f"  Total data points: {stats['n_total_points']:,}")
            print(f"  Min: {stats['min']:.6f}")
            print(f"  Max: {stats['max']:.6f}")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std: {stats['std']:.6f}")
            if variable in ['swh', 'hs']:
                var_name = 'hs' if variable == 'swh' else variable
                print(f"  {var_name}_log_mean: {stats[f'{var_name}_log_mean']:.6f}")
                print(f"  {var_name}_log_std: {stats[f'{var_name}_log_std']:.6f}")
        
        # Save to file if requested
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=4)
            if verbose:
                print(f"\nStatistics saved to: {output_file}")
        
        return stats
    
    def export_global_coordinates(self, save_path=None, sample_file=None):
        """
        Export global hindcast coordinates (lat, lon) as numpy array.
        
        The output shape is [N, 2] where N is the number of nodes in global grid,
        and the 2 columns are [latitude, longitude].
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the .npy file. 
            If None, saves to './data/global_hindcast/global_coordinates.npy'
        sample_file : str, optional
            Path to a sample netCDF file to extract coordinates from.
            If None, uses the first file found in data_dir.
        
        Returns:
        --------
        np.ndarray: Array of shape [N, 2] containing [lat, lon] coordinates
        """
        # Find a sample file if not provided
        if sample_file is None:
            file_pattern = os.path.join(self.data_dir, 'mentaschi2023swh_*.nc')
            files = sorted(glob.glob(file_pattern))
            if not files:
                raise FileNotFoundError(f"No data files found in {self.data_dir}")
            sample_file = files[0]
        
        print(f"Loading coordinates from: {sample_file}")
        
        # Load a single timestep to get coordinates
        ds = xr.open_dataset(sample_file, engine='h5netcdf')
        
        # Extract latitude and longitude
        # Global hindcast uses 'y' for lat and 'x' for lon
        lat = ds.y.values
        lon = ds.x.values
        
        ds.close()
        
        # Set default save path if not provided
        if save_path is None:
            save_path = './data/global_hindcast/global_coordinates.npy'
        
        # Use the utility function to export coordinates
        coordinates = export_coordinates(
            lat=lat,
            lon=lon,
            save_path=save_path,
            description="global"
        )
        
        return coordinates
    
    def cluster_global_coordinates(self, n_clusters=256, coordinate_path=None, 
                                   save_path=None, vis_save_path=None, 
                                   method='recursive_bisection',
                                   use_geographic_distance=True,
                                   use_spherical_view=True):
        """
        Cluster global coordinates into n_clusters groups with balanced node counts.
        
        This method reads the global coordinates and performs clustering to ensure 
        each cluster has approximately equal number of nodes. The clustering 
        considers spatial proximity while maintaining balance.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create. Default is 256.
        coordinate_path : str, optional
            Path to load the coordinates .npy file.
            If None, uses './data/global_hindcast/global_coordinates.npy'
        save_path : str, optional
            Path to save the clustering results.
            If None, saves to './data/global_hindcast/global_clusters.npz'
        vis_save_path : str, optional
            Path to save the visualization plot.
            If None, auto-generates based on save_path (replaces .npz with _visualization.png)
            Set to False to skip visualization
        method : str
            Clustering method to use. Options:
            - 'recursive_bisection': Recursive bisection for strictly balanced, spatially contiguous clusters
            - 'balanced_kmeans': K-means with balanced reassignment
            - 'kmeans': Standard K-means (may be unbalanced)
        use_geographic_distance : bool
            If True, use Haversine formula for geographic distance calculation.
            If False, use Euclidean distance on raw coordinates.
            Default is True.
        use_spherical_view : bool
            If True, also generate 3D spherical view visualizations from multiple angles.
            Default is True for global data.
        
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
            coordinate_path = './data/global_hindcast/global_coordinates.npy'
        
        # Load or generate coordinates
        if not os.path.exists(coordinate_path):
            print(f"Coordinates file not found at {coordinate_path}")
            print("Generating coordinates first...")
            coordinates = self.export_global_coordinates()
        else:
            coordinates = np.load(coordinate_path)
            print(f"Loaded coordinates from: {coordinate_path}")
            print(f"Shape: {coordinates.shape}")

        # Set default save path if not provided
        if save_path is None:
            save_path = './data/global_hindcast/global_clusters.npz'
        
        # Use the utility function to perform clustering
        result = cluster_coordinates(
            coordinates=coordinates,
            n_clusters=n_clusters,
            method=method,
            save_path=save_path,
            vis_save_path=vis_save_path,
            use_geographic_distance=use_geographic_distance,
            use_spherical_view=use_spherical_view
        )
        
        return result
    
    def convert_nc_to_npz(self, nc_dir, output_dir=None, verbose=True, delete_after_convert=True):
        """
        Convert nc files to npz compressed format, only saving WWM_1 variable (float32).
        Delete original nc files after conversion to save disk space.
        
        Args:
            nc_dir (str): Directory path containing nc files
            output_dir (str, optional): Output directory for npz files. If None, uses nc_dir
            verbose (bool): Whether to print progress information (default True)
            delete_after_convert (bool): Whether to delete original nc files after conversion (default True)
            
        Returns:
            list: List of successfully converted npz file paths
        """
        import os
        import glob
        import numpy as np
        import xarray as xr
        from tqdm import tqdm
        
        # Set output directory
        if output_dir is None:
            output_dir = nc_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all nc files
        nc_pattern = os.path.join(nc_dir, '*.nc')
        nc_files = sorted(glob.glob(nc_pattern))
        
        if not nc_files:
            print(f"No nc files found in {nc_dir}")
            return []
        
        if verbose:
            print(f"Found {len(nc_files)} nc files to convert")
            print(f"Output directory: {output_dir}")
            print("=" * 60)
        
        converted_files = []
        failed_files = []
        
        # Process each nc file
        for nc_file in tqdm(nc_files, desc="Conversion progress", disable=not verbose):
            # Generate output filename
            base_name = os.path.basename(nc_file)
            name_without_ext = os.path.splitext(base_name)[0]
            npz_file = os.path.join(output_dir, f"{name_without_ext}.npz")
            
            # Load nc file
            ds = xr.open_dataset(nc_file, engine='netcdf4')
            
            # Check if WWM_1 variable exists
            if 'WWM_1' not in ds:
                ds.close()
                failed_files.append((nc_file, "WWM_1 variable not found"))
                continue
            
            # Extract WWM_1 data and convert to float32
            wwm1_data = ds['WWM_1'].values.astype(np.float32)
            
            # Extract time information for potential future validation
            time_value = ds['time'].values if 'time' in ds else None
            
            # Save as npz format with compression
            np.savez_compressed(
                npz_file,
                WWM_1=wwm1_data,
                time=time_value
            )
            
            ds.close()
            
            # Verify npz file was saved correctly
            try:
                test_load = np.load(npz_file)
                _ = test_load['WWM_1']
                test_load.close()
            except Exception as e:
                failed_files.append((nc_file, f"npz verification failed: {str(e)}"))
                continue
            
            converted_files.append(npz_file)
            
            # Delete original nc file
            if delete_after_convert:
                os.remove(nc_file)
        
        if verbose:
            print("=" * 60)
            print(f"\nConversion completed!")
            print(f"  Successfully converted: {len(converted_files)} files")
            print(f"  Failed: {len(failed_files)} files")
            
            if failed_files:
                print("\nFailed files:")
                for f, reason in failed_files:
                    print(f"  - {os.path.basename(f)}: {reason}")
        
        return converted_files
