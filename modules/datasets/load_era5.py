import os
import sys
import time
import logging
import xarray as xr
import pandas as pd
from tqdm import tqdm
import numpy as np
import glob
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from modules.plot.plot_wave import plot_era5_wind, plot_era5_pressure, plot_era5_variable_distributions
from modules.utils import ERA5_HK_LON_RANGE_1, ERA5_HK_LAT_RANGE_1, ERA5_VARIABLE_NAMES


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class ERA5DataProcessor:
    def __init__(self):
        self.era5_data = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr')
    
    def load_data(self, variables=None, lon_range=(70, 140), lat_range=(50, 10), time_range=None, time_point=None):
        """
        Extract subset data for specified variables, geographical region and time period.
        
        Parameters
        ----------
        variables : list of str, optional
            List of variable names to extract. If None, returns all variables.
        lon_range : tuple of float, optional
            Longitude range as (min_longitude, max_longitude). Default: (70, 140)
        lat_range : tuple of float, optional  
            Latitude range as (max_latitude, min_latitude). Note: latitude should be 
            ordered from north to south. Default: (50, 10)
        time_range : tuple, optional
            Time period as (start_time, end_time). Should be in format 'YYYYMMDDTHH'.
            If both time_range and time_point are provided, time_range takes precedence. Default: None
        time_point : str, optional
            Specific time point to extract in format 'YYYYMMDDTHH'. 
            If both time_range and time_point are None, returns all times. Default: None
            
        Returns
        -------
        xarray.Dataset
            Subset dataset containing selected variables, geographical region and time period.
            
        Examples
        --------
        >>> # Extract data for specific time range with hourly precision
        >>> subset = load_data(variables=['2m_temperature'],
        ...                    time_range=('20230601T00', '20230601T12'))
        
        >>> # Extract data for specific time point
        >>> subset = load_data(time_point='20230615T06')
        """
        # If no variables specified, use all variables
        if variables is None:
            data_subset = self.era5_data
        else:
            data_subset = self.era5_data[variables]
        
        # Normalize ranges to ensure min <= max
        lon_min, lon_max = sorted(lon_range)
        lat_min, lat_max = sorted(lat_range)
        
        # Extract data for specified geographical region
        # Use inclusive boundaries to be more robust
        lon_mask = (data_subset.longitude >= lon_min) & (data_subset.longitude <= lon_max)
        lat_mask = (data_subset.latitude >= lat_min) & (data_subset.latitude <= lat_max)
        
        subset = data_subset.sel(
            longitude=data_subset.longitude[lon_mask],
            latitude=data_subset.latitude[lat_mask]
        )
        
        # Extract data based on time selection (time_range takes precedence over time_point)
        if time_range is not None:
            # Use pandas to parse compact time format
            start_time = pd.to_datetime(time_range[0], format='%Y%m%dT%H')
            end_time = pd.to_datetime(time_range[1], format='%Y%m%dT%H')
            # Time range selection
            subset = subset.sel(time=slice(start_time, end_time))
        elif time_point is not None:
            # Use pandas to parse compact time format
            time_point_dt = pd.to_datetime(time_point, format='%Y%m%dT%H')
            # Single time point selection
            subset = subset.sel(time=time_point_dt, method='nearest')
        
        return subset

    def export_hourly_era5(self, output_dir='./data/era5', start_time='19691216T00', end_time='20230115T23'):
        """
        Export hourly ERA5 data for wind components and pressure from start_time to end_time.
        Each hour is saved as a separate netCDF file with compressed encoding.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save the exported ERA5 files. Default: './data/era5'
        start_time : str, optional
            Start time for data extraction in format 'YYYYMMDDTHH'. Default: '19691216T00'
        end_time : str, optional
            End time for data extraction in format 'YYYYMMDDTHH'. Default: '20230115T23'
            
        Returns
        -------
        None
            Files are saved to disk in the specified output directory.
        """
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define variables to extract
        variables = ERA5_VARIABLE_NAMES
        
        # Use load_data method to get the full time series
        data_subset = self.load_data(
            variables=variables,
            lon_range=ERA5_HK_LON_RANGE_1,
            lat_range=ERA5_HK_LAT_RANGE_1,
            time_range=(start_time, end_time)
        )
        
        # Get all time points
        all_times = data_subset.time.values
        total_files = len(all_times)
        
        print(f"Starting export of {total_files} hourly ERA5 files from {start_time} to {end_time}")
        print(f"Output directory: {output_dir}")
        
        # Process in chunks to avoid memory issues
        chunk_size = 24 * 30  # 30 days at a time
        
        for i in tqdm(range(0, len(all_times), chunk_size), desc="Processing chunks"):
            # Get time chunk
            time_chunk = all_times[i:i+chunk_size]
            
            # Load chunk into memory
            ds_chunk = data_subset.sel(time=time_chunk).compute()
            
            # Process each time point individually
            for time_point in time_chunk:
                # Select data for single time point
                ds_single_time = ds_chunk.sel(time=time_point)
                
                # Convert time to filename-friendly format (hour precision)
                # Format: era5_YYYYMMDDTHH.nc to match wave file naming convention
                ts = pd.to_datetime(time_point)
                year_dir = os.path.join(output_dir, ts.strftime('%Y'))
                os.makedirs(year_dir, exist_ok=True)
                
                time_str = ts.strftime('%Y%m%dT%H')
                filename = f"era5_{time_str}.nc"
                filepath = os.path.join(year_dir, filename)
                
                # Set compression encoding for all variables
                encoding = {
                    var: {'zlib': True, 'complevel': 4}
                    for var in ERA5_VARIABLE_NAMES
                }
                
                # Save to netCDF file
                ds_single_time.to_netcdf(filepath, encoding=encoding, engine='h5netcdf')
        
        print(f"\nExport completed! {total_files} files saved to {output_dir}")

    def export_3hourly_era5_global(self, output_dir='./data/era5_global_3h', start_time='19491216T00', end_time='20230115T23', max_workers=4):
        """
        Export 3-hourly ERA5 data globally for wind components and pressure from start_time to end_time.
        Only exports data at 00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00 UTC each day.
        Each time point is saved as a separate netCDF file with compressed encoding.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save the exported ERA5 files. Default: './data/era5_global_3h'
        start_time : str, optional
            Start time for data extraction in format 'YYYYMMDDTHH'. Default: '19491216T00'
        end_time : str, optional
            End time for data extraction in format 'YYYYMMDDTHH'. Default: '20230115T23'
        max_workers : int, optional
            Maximum number of threads for parallel file writing. Default: 4
            
        Returns
        -------
        None
            Files are saved to disk in the specified output directory.
        """
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define variables to extract
        variables = ERA5_VARIABLE_NAMES
        
        # Use load_data method to get the full time series (global range)
        # Global range: full longitude and latitude coverage
        data_subset = self.load_data(
            variables=variables,
            lon_range=(0, 360),
            lat_range=(90, -90),
            time_range=(start_time, end_time)
        )
        
        # Get all time points and filter for 3-hourly intervals (0, 3, 6, 9, 12, 15, 18, 21 UTC)
        all_times = data_subset.time.values
        
        # Filter times to keep only 3-hourly intervals
        times_3hourly = []
        for time_point in all_times:
            ts = pd.to_datetime(time_point)
            # Keep only times at 0, 3, 6, 9, 12, 15, 18, 21 hours
            if ts.hour in [0, 3, 6, 9, 12, 15, 18, 21]:
                times_3hourly.append(time_point)
        
        total_files = len(times_3hourly)
        
        print(f"Starting export of {total_files} 3-hourly global ERA5 files from {start_time} to {end_time}")
        print(f"Output directory: {output_dir}")
        print(f"Time intervals: 00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00 UTC")
        print(f"Using {max_workers} threads for parallel processing")
        
        # Thread lock for directory creation
        dir_lock = threading.Lock()
        
        def save_single_time(ds_chunk, time_point):
            """
            Helper function to save a single time point to netCDF file.
            """
            # Select data for single time point
            ds_single_time = ds_chunk.sel(time=time_point)
            
            # Convert time to filename-friendly format (hour precision)
            # Format: era5_YYYYMMDDTHH.nc to match wave file naming convention
            ts = pd.to_datetime(time_point)
            year_dir = os.path.join(output_dir, ts.strftime('%Y'))
            
            # Thread-safe directory creation
            with dir_lock:
                os.makedirs(year_dir, exist_ok=True)
            
            time_str = ts.strftime('%Y%m%dT%H')
            filename = f"era5_{time_str}.nc"
            filepath = os.path.join(year_dir, filename)
            
            # Set compression encoding for all variables
            encoding = {
                var: {'zlib': True, 'complevel': 6}
                for var in ERA5_VARIABLE_NAMES
            }
            
            # Save to netCDF file
            ds_single_time.to_netcdf(filepath, encoding=encoding, engine='h5netcdf')
            return filepath
        
        # Process in chunks to avoid memory issues
        # Smaller chunk size for global data due to larger spatial coverage
        chunk_size = 24 * 7  # 7 days at a time (56 time points)
        
        # Progress bar for overall progress
        with tqdm(total=total_files, desc="Exporting files") as pbar:
            for i in range(0, len(times_3hourly), chunk_size):
                # Get time chunk
                time_chunk = times_3hourly[i:i+chunk_size]
                
                # Load chunk into memory
                ds_chunk = data_subset.sel(time=time_chunk).compute()
                
                # Process time points in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks for this chunk
                    futures = {
                        executor.submit(save_single_time, ds_chunk, time_point): time_point
                        for time_point in time_chunk
                    }
                    
                    # Process completed tasks
                    for future in as_completed(futures):
                        try:
                            filepath = future.result()
                            pbar.update(1)
                        except Exception as exc:
                            time_point = futures[future]
                            ts = pd.to_datetime(time_point)
                            print(f"\nError processing {ts}: {exc}")
        
        print(f"\nExport completed! {total_files} files saved to {output_dir}")

    def compute_variable_statistics(self, output_file='./data/era5_statistics.json', 
                                     start_time='19691216T00', end_time='20230115T23'):
        """
        Compute min and max values for wind components and pressure over specified time and region.
        Results are exported to a JSON file.
        
        Parameters
        ----------
        output_file : str, optional
            Path to save the statistics JSON file. Default: './data/era5_statistics.json'
        start_time : str, optional
            Start time in format 'YYYYMMDDTHH'. Default: '19691216T00'
        end_time : str, optional
            End time in format 'YYYYMMDDTHH'. Default: '20230115T23'
            
        Returns
        -------
        dict
            Dictionary containing min and max values for each variable.
        """
        print(f"Computing statistics for ERA5 data from {start_time} to {end_time}")
        print(f"Region: lon_range={ERA5_HK_LON_RANGE_1}, lat_range={ERA5_HK_LAT_RANGE_1}")
        
        # Define variables to analyze
        variables = ERA5_VARIABLE_NAMES
        
        # Parse time range
        start_time_dt = pd.to_datetime(start_time, format='%Y%m%dT%H')
        end_time_dt = pd.to_datetime(end_time, format='%Y%m%dT%H')
        
        # Select data subset directly from self.era5_data
        print("Filtering spatial and temporal data...")
        data_subset = self.era5_data[variables]
        
        # Filter geographical region
        lon_mask = (data_subset.longitude >= ERA5_HK_LON_RANGE_1[0]) & (data_subset.longitude < ERA5_HK_LON_RANGE_1[1])
        lat_mask = (data_subset.latitude >= ERA5_HK_LAT_RANGE_1[0]) & (data_subset.latitude < ERA5_HK_LAT_RANGE_1[1])
        
        subset = data_subset.sel(
            longitude=data_subset.longitude[lon_mask],
            latitude=data_subset.latitude[lat_mask]
        )
        
        # Filter time range
        subset = subset.sel(time=slice(start_time_dt, end_time_dt))
        
        print(f"Data shape: {subset.dims}")
        print("Computing min and max values for each variable...")
        
        # Compute statistics for each variable
        statistics = []
        for var in variables:
            print(f"  Processing {var}...")
            var_data = subset[var]
            
            # Compute min and max values
            min_val = float(var_data.min().compute().values)
            max_val = float(var_data.max().compute().values)
            
            statistics.append({
                'variable': var,
                'min_value': min_val,
                'max_value': max_val
            })
            
            print(f"    {var}: min={min_val:.4f}, max={max_val:.4f}")
        
        # Create statistics dictionary
        stats_dict = {
            'time_range': {
                'start': start_time,
                'end': end_time
            },
            'spatial_range': {
                'lon_range': ERA5_HK_LON_RANGE_1,
                'lat_range': ERA5_HK_LAT_RANGE_1
            },
            'variables': statistics
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to JSON
        import json
        with open(output_file, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"\nStatistics saved to {output_file}")
        
        return stats_dict

    def compute_variable_statistics_from_nc(self, data_dir='./data/era5/nc_files', 
                                            output_file='./data/era5_statistics_from_nc.json',
                                            start_time='19691216T00', end_time='20230115T23',
                                            sample_ratio=1.0, random_seed=42,
                                            global_range=False):
        """
        Compute min and max values for wind components and pressure from processed nc files.
        Supports random sampling of files to reduce computation time.
        
        Parameters
        ----------
        data_dir : str, optional
            Base directory containing the processed ERA5 nc files.
            Files should be organized as {data_dir}/{year}/era5_{time}.nc
            Default: './data/era5/nc_files'
        output_file : str, optional
            Path to save the statistics JSON file. Default: './data/era5_statistics_from_nc.json'
        start_time : str, optional
            Start time in format 'YYYYMMDDTHH'. Default: '19691216T00'
        end_time : str, optional
            End time in format 'YYYYMMDDTHH'. Default: '20230115T23'
        sample_ratio : float, optional
            Ratio of files to sample (0.0 to 1.0). Default: 1.0 (use all files)
            Example: 0.1 means randomly sample 10% of files
        random_seed : int, optional
            Random seed for reproducible sampling. Default: 42
        global_range : bool, optional
            If True, use global spatial range (lon: 0-360, lat: -90 to 90).
            If False, use Hong Kong regional range. Default: False
            
        Returns
        -------
        dict
            Dictionary containing min and max values for each variable, along with sampling info.
        """
        import numpy as np
        import glob
        
        print(f"Computing statistics from nc files from {start_time} to {end_time}")
        print(f"Data directory: {data_dir}")
        print(f"Sample ratio: {sample_ratio}")
        
        # Parse time range
        start_time_dt = pd.to_datetime(start_time, format='%Y%m%dT%H')
        end_time_dt = pd.to_datetime(end_time, format='%Y%m%dT%H')
        
        # Define variables to analyze
        variables = ERA5_VARIABLE_NAMES
        
        # Collect all nc files in the time range
        all_files = []
        for year in range(start_time_dt.year, end_time_dt.year + 1):
            year_dir = os.path.join(data_dir, str(year))
            if os.path.exists(year_dir):
                year_files = glob.glob(os.path.join(year_dir, 'era5_*.nc'))
                all_files.extend(year_files)
        
        # Filter files by time range
        filtered_files = []
        for filepath in all_files:
            # Extract time from filename: era5_YYYYMMDDTHH.nc
            filename = os.path.basename(filepath)
            time_str = filename.replace('era5_', '').replace('.nc', '')
            try:
                file_time = pd.to_datetime(time_str, format='%Y%m%dT%H')
                if start_time_dt <= file_time <= end_time_dt:
                    filtered_files.append(filepath)
            except:
                continue
        
        total_files = len(filtered_files)
        print(f"Found {total_files} nc files in the specified time range")
        
        if total_files == 0:
            print("Warning: No nc files found!")
            return None
        
        # Random sampling if sample_ratio < 1.0
        if sample_ratio < 1.0:
            np.random.seed(random_seed)
            sample_size = max(1, int(total_files * sample_ratio))
            sampled_files = np.random.choice(filtered_files, size=sample_size, replace=False).tolist()
            print(f"Randomly sampled {sample_size} files ({sample_ratio*100:.1f}%)")
        else:
            sampled_files = filtered_files
            print(f"Processing all {total_files} files")
        
        # Initialize statistics tracking for each variable
        # Using Welford's algorithm for online mean and variance computation
        var_stats = {
            var: {
                'min': float('inf'), 
                'max': float('-inf'),
                'count': 0,
                'mean': 0.0,
                'm2': 0.0  # Sum of squares of differences from the mean
            } 
            for var in ERA5_VARIABLE_NAMES
        }
        
        # Process files and update statistics
        successful_files = 0
        failed_files = []
        
        for filepath in tqdm(sampled_files, desc="Processing nc files"):
            try:
                # Try multiple engines for compatibility
                ds = None
                for engine in ['h5netcdf', 'netcdf4', 'scipy']:
                    try:
                        ds = xr.load_dataset(filepath, engine=engine)
                        break
                    except:
                        continue
                
                if ds is None:
                    failed_files.append(filepath)
                    continue
                
                # Update statistics for each variable using Welford's algorithm
                for var in variables:
                    if var in ds:
                        var_data = ds[var]
                        
                        # Update min/max
                        var_min = float(var_data.min().values)
                        var_max = float(var_data.max().values)
                        var_stats[var]['min'] = min(var_stats[var]['min'], var_min)
                        var_stats[var]['max'] = max(var_stats[var]['max'], var_max)
                        
                        # Update mean and variance using Welford's online algorithm
                        # Flatten data for processing
                        flat_data = var_data.values.flatten()
                        
                        for x in flat_data:
                            var_stats[var]['count'] += 1
                            delta = x - var_stats[var]['mean']
                            var_stats[var]['mean'] += delta / var_stats[var]['count']
                            delta2 = x - var_stats[var]['mean']
                            var_stats[var]['m2'] += delta * delta2
                
                successful_files += 1
                ds.close()
                
            except Exception as e:
                failed_files.append(filepath)
                continue
        
        print(f"\nSuccessfully processed: {successful_files}/{len(sampled_files)} files")
        if failed_files:
            print(f"Failed files: {len(failed_files)}")
        
        # Prepare statistics results
        statistics = []
        for var in variables:
            min_val = float(var_stats[var]['min'])
            max_val = float(var_stats[var]['max'])
            mean_val = float(var_stats[var]['mean'])
            
            # Compute standard deviation
            if var_stats[var]['count'] > 1:
                std_val = float(np.sqrt(var_stats[var]['m2'] / var_stats[var]['count']))
            else:
                std_val = 0.0
            
            statistics.append({
                'variable': var,
                'min_value': min_val,
                'max_value': max_val,
                'mean_value': mean_val,
                'std_value': std_val,
                'count': int(var_stats[var]['count'])
            })
            
            print(f"  {var}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}")
        
        # Determine spatial range based on global_range parameter
        if global_range:
            lon_range = [0, 360]
            lat_range = [-90, 90]
        else:
            lon_range = ERA5_HK_LON_RANGE_1
            lat_range = ERA5_HK_LAT_RANGE_1
        
        # Create statistics dictionary with metadata
        stats_dict = {
            'time_range': {
                'start': start_time,
                'end': end_time
            },
            'spatial_range': {
                'lon_range': lon_range,
                'lat_range': lat_range,
                'global': global_range
            },
            'sampling_info': {
                'total_files_in_range': total_files,
                'sampled_files': len(sampled_files),
                'sample_ratio': sample_ratio,
                'successful_files': successful_files,
                'failed_files': len(failed_files),
                'random_seed': random_seed
            },
            'variables': statistics
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to JSON
        import json
        with open(output_file, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"\nStatistics saved to {output_file}")
        
        return stats_dict

    def visualize_selected_samples(self, selected_times, save_path):
        for time in selected_times:
            # Load wind field data (using ERA5_VARIABLE_NAMES)
            ds_wind = self.load_data(
                variables=[ERA5_VARIABLE_NAMES[0], ERA5_VARIABLE_NAMES[1]], 
                lon_range=ERA5_HK_LON_RANGE_1, 
                lat_range=ERA5_HK_LAT_RANGE_1, 
                time_point=time
            )
            plot_era5_wind(ds_wind, output_path=os.path.join(save_path, f"era5_wind_{time}_hk.png"))
            
            # Load pressure field data (using ERA5_VARIABLE_NAMES)
            ds_pressure = self.load_data(
                variables=[ERA5_VARIABLE_NAMES[2]], 
                lon_range=ERA5_HK_LON_RANGE_1, 
                lat_range=ERA5_HK_LAT_RANGE_1, 
                time_point=time
            )
            plot_era5_pressure(ds_pressure, output_path=os.path.join(save_path, f"era5_pressure_{time}_hk.png"))

    def visualize_selected_samples_from_nc(self, selected_times, save_path, data_dir='./data/era5'):
        """
        Visualize selected ERA5 samples by loading data from processed nc files.
        
        Parameters
        ----------
        selected_times : list of str
            List of time points in format 'YYYYMMDDTHH' to visualize.
        save_path : str
            Directory path to save the generated plots.
        data_dir : str, optional
            Base directory containing the processed ERA5 nc files.
            Files should be organized as {data_dir}/{year}/era5_{time}.nc
            Default: './data/era5'
        
        Returns
        -------
        None
            Plots are saved to the specified save_path directory.
        """
        os.makedirs(save_path, exist_ok=True)
        
        for time in selected_times:
            # Parse time string to get year and construct file path
            ts = pd.to_datetime(time, format='%Y%m%dT%H')
            year = ts.strftime('%Y')
            filename = f"era5_{time}.nc"
            filepath = os.path.join(data_dir, 'nc_files', year, filename)
            
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"Warning: File not found: {filepath}")
                continue
            
            # Load the entire dataset from nc file (use h5netcdf engine to match export format)
            try:
                ds = xr.load_dataset(filepath, engine='h5netcdf')
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
                continue
            
            # Extract wind field data (u and v components) (using ERA5_VARIABLE_NAMES)
            ds_wind = ds[[ERA5_VARIABLE_NAMES[0], ERA5_VARIABLE_NAMES[1]]]
            plot_era5_wind(ds_wind, output_path=os.path.join(save_path, f"era5_wind_{time}_hk.png"))
            
            # Extract pressure field data (using ERA5_VARIABLE_NAMES)
            ds_pressure = ds[[ERA5_VARIABLE_NAMES[2]]]
            plot_era5_pressure(ds_pressure, output_path=os.path.join(save_path, f"era5_pressure_{time}_hk.png"))

    def plot_variable_distributions_from_nc(self, data_dir='./data/era5', save_path='./plots/era5_distributions.png', 
                                             start_time='19691216T00', end_time='20230115T23', 
                                             sample_ratio=0.02, random_seed=42):
        """
        Plot distributions for ERA5 variables by sampling from processed nc files.
        
        This method randomly samples nc files from the specified time range and creates
        distribution plots (histograms) for the three ERA5 variables: u-component of wind,
        v-component of wind, and mean sea level pressure.
        
        Parameters
        ----------
        data_dir : str, optional
            Base directory containing the processed ERA5 nc files.
            Files should be organized as {data_dir}/nc_files/{year}/era5_{time}.nc
            Default: './data/era5'
        save_path : str, optional
            Path to save the distribution plot. Default: './plots/era5_distributions.png'
        start_time : str, optional
            Start time in format 'YYYYMMDDTHH'. Default: '19691216T00'
        end_time : str, optional
            End time in format 'YYYYMMDDTHH'. Default: '20230115T23'
        sample_ratio : float, optional
            Ratio of files to sample (0.0 to 1.0). Default: 0.1 (sample 10% of files)
            Example: 0.1 means randomly sample 10% of files
        random_seed : int, optional
            Random seed for reproducible sampling. Default: 42
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure object
        """
        
        print(f"Sampling nc files from {start_time} to {end_time}")
        print(f"Data directory: {data_dir}")
        print(f"Sample ratio: {sample_ratio}")
        
        # Parse time range
        start_time_dt = pd.to_datetime(start_time, format='%Y%m%dT%H')
        end_time_dt = pd.to_datetime(end_time, format='%Y%m%dT%H')
        
        # Collect all nc files in the time range
        all_files = []
        nc_dir = os.path.join(data_dir, 'nc_files')
        for year in range(start_time_dt.year, end_time_dt.year + 1):
            year_dir = os.path.join(nc_dir, str(year))
            if os.path.exists(year_dir):
                year_files = glob.glob(os.path.join(year_dir, 'era5_*.nc'))
                all_files.extend(year_files)
        
        # Filter files by time range
        filtered_files = []
        for filepath in all_files:
            # Extract time from filename: era5_YYYYMMDDTHH.nc
            filename = os.path.basename(filepath)
            time_str = filename.replace('era5_', '').replace('.nc', '')
            try:
                file_time = pd.to_datetime(time_str, format='%Y%m%dT%H')
                if start_time_dt <= file_time <= end_time_dt:
                    filtered_files.append(filepath)
            except:
                continue
        
        total_files = len(filtered_files)
        print(f"Found {total_files} nc files in the specified time range")
        
        if total_files == 0:
            print("Warning: No nc files found!")
            return None
        
        # Random sampling
        np.random.seed(random_seed)
        sample_size = max(1, int(total_files * sample_ratio))
        sampled_files = np.random.choice(filtered_files, size=sample_size, replace=False).tolist()
        print(f"Randomly sampled {sample_size} files ({sample_ratio*100:.1f}%)")
        
        # Collect data from sampled files
        all_data = {var: [] for var in ERA5_VARIABLE_NAMES}
        successful_files = 0
        
        for filepath in tqdm(sampled_files, desc="Loading sampled nc files"):
            try:
                # Try multiple engines for compatibility
                ds = None
                for engine in ['h5netcdf', 'netcdf4', 'scipy']:
                    try:
                        ds = xr.load_dataset(filepath, engine=engine)
                        break
                    except:
                        continue
                
                if ds is None:
                    continue
                
                # Collect data for each variable
                for var in ERA5_VARIABLE_NAMES:
                    if var in ds:
                        var_data = ds[var].values.flatten()
                        all_data[var].extend(var_data[~np.isnan(var_data)])
                
                successful_files += 1
                ds.close()
                
            except Exception as e:
                continue
        
        print(f"Successfully loaded: {successful_files}/{sample_size} files")
        
        # Create a combined dataset for plotting
        combined_ds = xr.Dataset()
        for var in ERA5_VARIABLE_NAMES:
            if len(all_data[var]) > 0:
                # Reshape to 1D array for plotting
                combined_ds[var] = xr.DataArray(
                    np.array(all_data[var]),
                    dims=['sample']
                )
        
        # Call the plotting function
        fig = plot_era5_variable_distributions(combined_ds, save_path=save_path)
        
        return fig

    def convert_nc_to_npz(self, nc_dir, output_dir=None, verbose=True, delete_after_convert=False, num_workers=4):
        """
        Convert ERA5 nc files to npz compressed format, saving all ERA5 variables (float32).
        
        Args:
            nc_dir (str): Directory path containing nc files, organized as {nc_dir}/{year}/era5_YYYYMMDDTHH.nc
            output_dir (str, optional): Output directory for npz files. If None, uses nc_dir
            verbose (bool): Whether to print progress information (default True)
            delete_after_convert (bool): Whether to delete original nc files after conversion (default False)
            num_workers (int): Number of parallel workers for conversion (default 4)
            
        Returns:
            list: List of successfully converted npz file paths
        """
        import os
        import glob
        import numpy as np
        import xarray as xr
        from tqdm import tqdm
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        # Set output directory
        if output_dir is None:
            output_dir = nc_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all nc files recursively (including year subdirectories)
        nc_pattern = os.path.join(nc_dir, '**', 'era5_*.nc')
        nc_files = sorted(glob.glob(nc_pattern, recursive=True))
        
        if not nc_files:
            print(f"No nc files found in {nc_dir}")
            return []
        
        if verbose:
            print(f"Found {len(nc_files)} nc files to convert")
            print(f"Output directory: {output_dir}")
            print(f"Using {num_workers} parallel workers")
            print("=" * 60)
        
        # Filter out already converted files (skip only if npz > 8MB)
        files_to_convert = []
        already_converted = []
        overwritten_files = []
        MIN_FILE_SIZE = 8 * 1024 * 1024  # 8MB in bytes
        
        for nc_file in nc_files:
            rel_path = os.path.relpath(nc_file, nc_dir)
            base_name = os.path.basename(nc_file)
            name_without_ext = os.path.splitext(base_name)[0]
            year_dir = os.path.dirname(rel_path)
            if year_dir and year_dir != '.':
                npz_output_dir = os.path.join(output_dir, year_dir)
            else:
                npz_output_dir = output_dir
            npz_file = os.path.join(npz_output_dir, f"{name_without_ext}.npz")
            
            if os.path.exists(npz_file):
                file_size = os.path.getsize(npz_file)
                if file_size > MIN_FILE_SIZE:
                    # File exists and is large enough, skip it
                    already_converted.append(npz_file)
                else:
                    # File exists but too small, will overwrite
                    overwritten_files.append((npz_file, file_size))
                    files_to_convert.append((nc_file, npz_file, nc_dir, output_dir, delete_after_convert))
            else:
                files_to_convert.append((nc_file, npz_file, nc_dir, output_dir, delete_after_convert))
        
        if verbose and already_converted:
            print(f"Skipping {len(already_converted)} already converted files (>8MB)")
        if verbose and overwritten_files:
            print(f"Overwriting {len(overwritten_files)} small files (<8MB)")
        
        converted_files = already_converted.copy()
        failed_files = []
        
        if not files_to_convert:
            if verbose:
                print("All files already converted!")
            return converted_files
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._convert_single_file, args): args[0] for args in files_to_convert}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Conversion progress", disable=not verbose):
                nc_file, result, error = future.result()
                if result:
                    converted_files.append(result)
                else:
                    failed_files.append((nc_file, error))
        
        if verbose:
            print("=" * 60)
            print(f"\nConversion completed!")
            print(f"  Successfully converted: {len(converted_files)} files")
            print(f"  Failed: {len(failed_files)} files")
            
            if failed_files:
                print("\nFailed files:")
                for f, reason in failed_files[:10]:  # Show first 10 failures
                    print(f"  - {os.path.basename(f)}: {reason}")
                if len(failed_files) > 10:
                    print(f"  ... and {len(failed_files) - 10} more")
        
        return converted_files

    @staticmethod
    def _convert_single_file(args):
        """
        Convert a single nc file to npz format (static method for multiprocessing).
        
        Args:
            args: Tuple of (nc_file, npz_file, nc_dir, output_dir, delete_after_convert)
            
        Returns:
            tuple: (nc_file, npz_file or None, error_message or None)
        """
        import os
        import numpy as np
        import xarray as xr
        from modules.utils import ERA5_VARIABLE_NAMES
        
        nc_file, npz_file, nc_dir, output_dir, delete_after_convert = args
        
        try:
            # Ensure output directory exists
            npz_output_dir = os.path.dirname(npz_file)
            if npz_output_dir:
                os.makedirs(npz_output_dir, exist_ok=True)
            
            # Load nc file
            ds = xr.open_dataset(nc_file, engine='h5netcdf')
            
            # Prepare data dictionary for npz
            npz_data = {}
            
            # Extract all ERA5 variables and convert to float32
            for var in ERA5_VARIABLE_NAMES:
                if var in ds:
                    npz_data[var] = ds[var].values.astype(np.float32)
            
            # Save as npz format with compression
            np.savez_compressed(npz_file, **npz_data)
            
            ds.close()
            
            # Verify npz file was saved correctly
            try:
                test_load = np.load(npz_file)
                # Verify at least one ERA5 variable exists
                var_found = False
                for var in ERA5_VARIABLE_NAMES:
                    if var in test_load:
                        var_found = True
                        break
                if not var_found:
                    raise ValueError("No ERA5 variables found in saved npz file")
                test_load.close()
            except Exception as e:
                return (nc_file, None, f"npz verification failed: {str(e)}")
            
            # Delete original nc file if requested
            if delete_after_convert:
                os.remove(nc_file)
            
            return (nc_file, npz_file, None)
            
        except Exception as e:
            return (nc_file, None, str(e))

    def export_coordinates(self, nc_dir, output_file=None):
        """
        Export latitude and longitude coordinates from a sample ERA5 nc file.
        The coordinates are shared across all ERA5 files, so only one copy is needed.
        
        Args:
            nc_dir (str): Directory path containing nc files, will search recursively
            output_file (str, optional): Path to save the coordinates npz file.
                If None, saves to './data/era5/era5_coordinates.npz'
                
        Returns:
            dict: Dictionary containing 'latitude' and 'longitude' arrays
        """
        import os
        import glob
        import numpy as np
        import xarray as xr
        
        # Set default output path
        if output_file is None:
            output_file = './data/era5/era5_coordinates.npz'
        
        # Find all nc files recursively
        nc_pattern = os.path.join(nc_dir, '**', 'era5_*.nc')
        nc_files = sorted(glob.glob(nc_pattern, recursive=True))
        
        if not nc_files:
            print(f"No nc files found in {nc_dir}")
            return None
        
        # Use the first file as sample to extract coordinates
        sample_file = nc_files[0]
        print(f"Extracting coordinates from: {sample_file}")
        
        # Load sample file
        ds = xr.open_dataset(sample_file, engine='h5netcdf')
        
        # Extract coordinates
        coordinates = {}
        if 'latitude' in ds:
            coordinates['latitude'] = ds['latitude'].values.astype(np.float32)
        if 'longitude' in ds:
            coordinates['longitude'] = ds['longitude'].values.astype(np.float32)
        
        ds.close()
        
        if not coordinates:
            print("No latitude/longitude coordinates found in the sample file")
            return None
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save coordinates to npz file
        np.savez_compressed(output_file, **coordinates)
        
        print(f"Coordinates saved to: {output_file}")
        print(f"  Latitude shape: {coordinates['latitude'].shape}")
        print(f"  Longitude shape: {coordinates['longitude'].shape}")
        
        return coordinates

    def verify_npz_integrity(self, npz_dir, start_time=None, end_time=None, verbose=True, num_workers=32):
        """
        Verify the integrity of converted NPZ files using multi-threading.
        
        Checks include:
        1. Whether each time point has a corresponding NPZ file
        2. Whether each NPZ file contains all three ERA5 variables (u-wind, v-wind, pressure)
        3. Whether the data shape of each variable is correct
        
        Args:
            npz_dir (str): Directory path for NPZ files, organized as {npz_dir}/{year}/era5_YYYYMMDDTHH.npz
            start_time (str, optional): Start time in format 'YYYYMMDDTHH'. If None, checks all files
            end_time (str, optional): End time in format 'YYYYMMDDTHH'. If None, checks all files
            verbose (bool): Whether to print detailed information (default True)
            num_workers (int): Number of parallel workers for verification (default 32)
            
        Returns:
            dict: Verification result dictionary containing the following fields:
                - 'is_complete': Whether complete (bool)
                - 'expected_count': Expected number of files
                - 'actual_count': Actual number of files
                - 'missing_files': List of missing files
                - 'corrupted_files': List of corrupted files (missing variables or unreadable)
                - 'incomplete_variables': List of files with incomplete variables
                - 'variable_stats': Statistics for each variable
        """
        import os
        import glob
        import numpy as np
        from datetime import datetime, timedelta
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        if verbose:
            print("=" * 60)
            print("Starting NPZ file integrity verification...")
            print(f"NPZ directory: {npz_dir}")
            print(f"Using {num_workers} parallel workers")
            print("=" * 60)
        
        # Find all NPZ files
        npz_pattern = os.path.join(npz_dir, '**', 'era5_*.npz')
        npz_files = sorted(glob.glob(npz_pattern, recursive=True))
        
        if not npz_files:
            if verbose:
                print(f"Error: No NPZ files found in {npz_dir}")
            return {
                'is_complete': False,
                'expected_count': 0,
                'actual_count': 0,
                'missing_files': [],
                'corrupted_files': [],
                'incomplete_variables': [],
                'variable_stats': {},
                'error': 'No NPZ files found'
            }
        
        if verbose:
            print(f"Found {len(npz_files)} NPZ files")
        
        # Parse time range
        file_times = []
        for npz_file in npz_files:
            filename = os.path.basename(npz_file)
            # Extract time from filename: era5_YYYYMMDDTHH.npz
            try:
                time_str = filename.replace('era5_', '').replace('.npz', '')
                file_time = pd.to_datetime(time_str, format='%Y%m%dT%H')
                file_times.append((file_time, npz_file))
            except:
                continue
        
        # Filter by time range
        if start_time is not None:
            start_dt = pd.to_datetime(start_time, format='%Y%m%dT%H')
            file_times = [(t, f) for t, f in file_times if t >= start_dt]
        if end_time is not None:
            end_dt = pd.to_datetime(end_time, format='%Y%m%dT%H')
            file_times = [(t, f) for t, f in file_times if t <= end_dt]
        
        file_times.sort(key=lambda x: x[0])
        actual_files = [f for t, f in file_times]
        
        if not actual_files:
            if verbose:
                print("Error: No NPZ files found in the specified time range")
            return {
                'is_complete': False,
                'expected_count': 0,
                'actual_count': 0,
                'missing_files': [],
                'corrupted_files': [],
                'incomplete_variables': [],
                'variable_stats': {},
                'error': 'No NPZ files found in time range'
            }
        
        # Calculate expected time points
        if file_times:
            min_time = min(t for t, f in file_times)
            max_time = max(t for t, f in file_times)
            
            # Generate all expected time points (3-hour intervals)
            expected_times = []
            current_time = min_time
            while current_time <= max_time:
                expected_times.append(current_time)
                current_time += timedelta(hours=3)
            
            expected_count = len(expected_times)
        else:
            expected_count = 0
            expected_times = []
        
        # Check for missing files
        actual_times_set = set(t for t, f in file_times)
        missing_files = []
        for expected_time in expected_times:
            if expected_time not in actual_times_set:
                missing_filename = f"era5_{expected_time.strftime('%Y%m%dT%H')}.npz"
                missing_year_dir = expected_time.strftime('%Y')
                missing_path = os.path.join(npz_dir, missing_year_dir, missing_filename)
                missing_files.append(missing_path)
        
        # Check variable integrity for each file using multi-threading
        corrupted_files = []
        incomplete_variables = []
        variable_shapes = {var: [] for var in ERA5_VARIABLE_NAMES}
        
        # Thread-safe data structures
        corrupted_lock = threading.Lock()
        incomplete_lock = threading.Lock()
        shapes_lock = threading.Lock()
        
        if verbose:
            print(f"\nStarting variable integrity check for {len(actual_files)} files using {num_workers} workers...")
        
        def verify_single_file(npz_file):
            """Verify a single NPZ file (for parallel execution)."""
            try:
                # Load NPZ file
                data = np.load(npz_file)
                
                # Check if all required variables are present
                missing_vars = []
                local_shapes = {var: None for var in ERA5_VARIABLE_NAMES}
                
                for var in ERA5_VARIABLE_NAMES:
                    if var not in data:
                        missing_vars.append(var)
                    else:
                        # Record variable shape
                        local_shapes[var] = data[var].shape
                
                if missing_vars:
                    with incomplete_lock:
                        incomplete_variables.append({
                            'file': npz_file,
                            'missing_variables': missing_vars,
                            'available_variables': list(data.keys())
                        })
                else:
                    # Only record shapes if all variables are present
                    with shapes_lock:
                        for var in ERA5_VARIABLE_NAMES:
                            if local_shapes[var] is not None:
                                variable_shapes[var].append(local_shapes[var])
                
                data.close()
                
            except Exception as e:
                with corrupted_lock:
                    corrupted_files.append({
                        'file': npz_file,
                        'error': str(e)
                    })
        
        # Process files in parallel using ThreadPoolExecutor with tqdm progress bar
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(verify_single_file, npz_file): npz_file for npz_file in actual_files}
            
            # Use tqdm to show progress
            if verbose:
                for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying files"):
                    try:
                        future.result()
                    except Exception as e:
                        npz_file = futures[future]
                        with corrupted_lock:
                            corrupted_files.append({
                                'file': npz_file,
                                'error': f"Thread execution error: {str(e)}"
                            })
            else:
                # Non-verbose mode: just wait for completion
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        npz_file = futures[future]
                        with corrupted_lock:
                            corrupted_files.append({
                                'file': npz_file,
                                'error': f"Thread execution error: {str(e)}"
                            })
        
        # Statistics for variable shape consistency
        variable_stats = {}
        for var in ERA5_VARIABLE_NAMES:
            shapes = variable_shapes[var]
            if shapes:
                unique_shapes = list(set(shapes))
                variable_stats[var] = {
                    'count': len(shapes),
                    'shapes': unique_shapes,
                    'shape_consistent': len(unique_shapes) == 1
                }
            else:
                variable_stats[var] = {
                    'count': 0,
                    'shapes': [],
                    'shape_consistent': False
                }
        
        # Determine completeness
        is_complete = (
            len(missing_files) == 0 and 
            len(corrupted_files) == 0 and 
            len(incomplete_variables) == 0
        )
        
        # Print results
        if verbose:
            print("\n" + "=" * 60)
            print("Verification Summary:")
            print("=" * 60)
            print(f"Expected files: {expected_count}")
            print(f"Actual files: {len(actual_files)}")
            print(f"Missing files: {len(missing_files)}")
            print(f"Corrupted files: {len(corrupted_files)}")
            print(f"Incomplete variable files: {len(incomplete_variables)}")
            print(f"\nOverall integrity: {'Complete' if is_complete else 'Incomplete'}")
            
            if missing_files:
                print(f"\nMissing files (first 10):")
                for f in missing_files[:10]:
                    print(f"  - {os.path.basename(f)}")
                if len(missing_files) > 10:
                    print(f"  ... and {len(missing_files) - 10} more")
            
            if corrupted_files:
                print(f"\nCorrupted files:")
                for item in corrupted_files[:10]:
                    print(f"  - {os.path.basename(item['file'])}: {item['error']}")
                if len(corrupted_files) > 10:
                    print(f"  ... and {len(corrupted_files) - 10} more")
            
            if incomplete_variables:
                print(f"\nFiles with incomplete variables:")
                for item in incomplete_variables[:10]:
                    print(f"  - {os.path.basename(item['file'])}")
                    print(f"    Missing variables: {item['missing_variables']}")
                    print(f"    Available variables: {item['available_variables']}")
                if len(incomplete_variables) > 10:
                    print(f"  ... and {len(incomplete_variables) - 10} more")
            
            print(f"\nVariable Statistics:")
            for var, stats in variable_stats.items():
                status = "OK" if stats['shape_consistent'] else "INCONSISTENT"
                print(f"  [{status}] {var}:")
                print(f"      File count: {stats['count']}")
                if stats['shapes']:
                    print(f"      Data shape: {stats['shapes'][0]}")
            print("=" * 60)
        
        return {
            'is_complete': is_complete,
            'expected_count': expected_count,
            'actual_count': len(actual_files),
            'missing_files': missing_files,
            'corrupted_files': corrupted_files,
            'incomplete_variables': incomplete_variables,
            'variable_stats': variable_stats
        }

    # ── CDS sequential download ─────────────────────────────────────────────

    def download_cds_sequential(self, start_date, end_date,
                                  output_dir='./data/era5/global_nc_files',
                                  temp_dir=None):
        """
        Download ERA5 global 3-hourly surface data from CDS day-by-day.

        Saves individual NC files matching the WeatherBench2 format:
        {output_dir}/{year}/era5_{YYYYMMDDTHH}.nc

        Uses cdsapi with long retry timeouts – CDS queues can take hours.
        Already-downloaded files are skipped (resumable).

        Parameters
        ----------
        start_date : str  –  'YYYY-MM-DD'
        end_date   : str  –  'YYYY-MM-DD' (inclusive)
        output_dir : str  –  root output directory
        temp_dir   : str  –  scratch directory for raw CDS files
        """
        import cdsapi

        if temp_dir is None:
            temp_dir = os.path.join(os.path.dirname(output_dir.rstrip('/')), '_cds_downloads')

        # ── constants matching WeatherBench2 format ──────────────────────
        HOURS = ['00:00', '03:00', '06:00', '09:00',
                 '12:00', '15:00', '18:00', '21:00']
        CDS_VARS = ['10m_u_component_of_wind', '10m_v_component_of_wind',
                    'mean_sea_level_pressure']
        VAR_MAP = {'u10': '10m_u_component_of_wind',
                   'v10': '10m_v_component_of_wind',
                   'msl': 'mean_sea_level_pressure'}
        OUTPUT_VARS = ['10m_u_component_of_wind', '10m_v_component_of_wind',
                       'mean_sea_level_pressure']
        VAR_ATTRS = {
            '10m_u_component_of_wind': {'long_name': '10 metre U wind component',
                                        'short_name': 'u10', 'units': 'm s**-1'},
            '10m_v_component_of_wind': {'long_name': '10 metre V wind component',
                                        'short_name': 'v10', 'units': 'm s**-1'},
            'mean_sea_level_pressure': {'long_name': 'Mean sea level pressure',
                                        'short_name': 'msl',
                                        'standard_name': 'air_pressure_at_mean_sea_level',
                                        'units': 'Pa'},
        }
        COMPRESSION = {'zlib': True, 'complevel': 4}

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        log = logging.getLogger(__name__)

        # ── helpers ──────────────────────────────────────────────────────
        def _normalize_ds(ds):
            if 'valid_time' in ds.dims:
                ds = ds.rename({'valid_time': 'time'})
            rename = {k: v for k, v in VAR_MAP.items() if k in ds.data_vars}
            if rename:
                ds = ds.rename(rename)
            for dim in ['number', 'expver']:
                if dim in ds.dims:
                    ds = ds.isel({dim: 0})
                if dim in ds.coords:
                    ds = ds.drop_vars(dim)
                if dim in ds.dims:
                    ds = ds.squeeze(dim, drop=True)
            for v in OUTPUT_VARS:
                if v in ds:
                    ds[v] = ds[v].astype(np.float32)
                    ds[v].attrs = VAR_ATTRS.get(v, {})
            if 'latitude' in ds.coords and ds.latitude.values[0] < ds.latitude.values[-1]:
                ds = ds.sortby('latitude', ascending=False)
            return ds

        def _day_exists(year_dir, date_str):
            y, m, d = date_str.split('-')
            for h in HOURS:
                hh = h.replace(':00', '')
                fpath = os.path.join(year_dir, f'era5_{y}{m}{d}T{hh}.nc')
                if not os.path.exists(fpath) or os.path.getsize(fpath) < 1_000_000:
                    return False
            return True

        # ── main loop ───────────────────────────────────────────────────
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        total_days = (end_dt - start_dt).days + 1

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        log.info('=' * 60)
        log.info('ERA5 CDS download: %s → %s  (%d days)', start_date, end_date, total_days)
        log.info('Output: %s', output_dir)
        log.info('=' * 60)

        client = cdsapi.Client(timeout=300, sleep_max=1200, retry_max=2000)

        success = failed = skipped = total_files = 0
        current = start_dt

        while current <= end_dt:
            date_str = current.strftime('%Y-%m-%d')
            y, m, d = date_str.split('-')
            year_dir = os.path.join(output_dir, y)
            os.makedirs(year_dir, exist_ok=True)

            if _day_exists(year_dir, date_str):
                skipped += 1
                current += timedelta(days=1)
                continue

            temp_file = os.path.join(temp_dir, f'era5_{y}{m}{d}.nc')
            if os.path.exists(temp_file):
                os.remove(temp_file)

            log.info('Requesting %s ...', date_str)

            try:
                # CDS retrieve (blocks until complete, cdsapi auto-retries)
                client.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': ['reanalysis'],
                        'variable': CDS_VARS,
                        'year': [y],
                        'month': [m],
                        'day': [d],
                        'time': HOURS,
                        'data_format': 'netcdf',
                        'grid': [0.25, 0.25],
                        'area': [90, 0, -90, 360],
                    },
                    target=temp_file,
                )

                if not os.path.exists(temp_file) or os.path.getsize(temp_file) < 1000:
                    log.error('%s: downloaded file empty/missing', date_str)
                    failed += 1
                    current += timedelta(days=1)
                    continue

                # Normalize & split into individual timestep NC files
                ds = xr.open_dataset(temp_file, engine='h5netcdf')
                ds = _normalize_ds(ds)
                saved = 0
                for t in ds.time.values:
                    ts_str = pd.to_datetime(t).strftime('%Y%m%dT%H')
                    fpath = os.path.join(year_dir, f'era5_{ts_str}.nc')
                    if os.path.exists(fpath) and os.path.getsize(fpath) > 1_000_000:
                        continue
                    ds_1 = ds.sel(time=t)
                    enc = {v: COMPRESSION for v in OUTPUT_VARS if v in ds_1}
                    ds_1.to_netcdf(fpath, encoding=enc, engine='h5netcdf')
                    saved += 1
                ds.close()
                log.info('%s: saved %d/8 files', date_str, saved)
                success += 1
                total_files += saved
            except KeyboardInterrupt:
                log.info('Interrupted. Progress saved.')
                break
            except Exception as e:
                log.error('%s: %s', date_str, e)
                failed += 1

            # Small delay between requests (CDS-friendly)
            if current < end_dt:
                time.sleep(30)

            # Progress summary every 10 days
            elapsed = (current - start_dt).days + 1
            if elapsed % 10 == 0:
                log.info('--- progress %d/%d days | S:%d F:%d K:%d | files:%d ---',
                         elapsed, total_days, success, failed, skipped, total_files)

            current += timedelta(days=1)

        # Clean up temp files
        try:
            os.remove(temp_file)
        except OSError:
            pass

        log.info('=' * 60)
        log.info('COMPLETE: %d ok, %d failed, %d skipped, %d files saved',
                 success, failed, skipped, total_files)
        log.info('=' * 60)