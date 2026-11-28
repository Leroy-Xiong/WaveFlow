import os
import random
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import torch
import shutil


HK_REGION = (113.6, 114.6, 22.0, 22.7)

# Global region range (lon_range, lat_range) - full global coverage
GLOBAL_REGION = (-180, 180, -90, 90)

# ERA5 data Hong Kong region range (lon_range, lat_range)
ERA5_HK_LON_RANGE_1 = (110, 120)
ERA5_HK_LAT_RANGE_1 = (15, 25)
ERA5_HK_LON_RANGE_2 = (111, 117)
ERA5_HK_LAT_RANGE_2 = (19, 25)
ERA5_HK_LON_RANGE_3 = (112.5, 115.5)
ERA5_HK_LAT_RANGE_3 = (21, 24)

# ERA5 global region range (full coverage)
ERA5_GLOBAL_LON_RANGE = (-180, 180)
ERA5_GLOBAL_LAT_RANGE = (-90, 90)

# ERA5 variable names definition
ERA5_VARIABLE_NAMES = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind', 
    'mean_sea_level_pressure'
]

# Wave variable metadata definition
WAVE_VARIABLES = {
    'hs': {
        'units': 'm',
        'long_name': 'Significant Wave Height',
        'plot_label': 'Significant Wave Height (m)',
        'plot_title': 'Significant Wave Height Distribution'
    },
    'tp': {
        'units': 's',
        'long_name': 'Peak Wave Period',
        'plot_label': 'Peak Wave Period (s)',
        'plot_title': 'Peak Wave Period Distribution'
    },
    'dm': {
        'units': 'degree',
        'long_name': 'Mean Wave Direction',
        'plot_label': 'Mean Wave Direction (°)',
        'plot_title': 'Mean Wave Direction Distribution'
    },
    'elev': {
        'units': 'm',
        'long_name': 'Water Surface Elevation',
        'plot_label': 'Water Surface Elevation (m)',
        'plot_title': 'Water Surface Elevation Distribution'
    }
}

# Major cities coordinates for map visualization
MAJOR_CITIES = {
    # Asia Pacific
    'Tokyo': (139.69, 35.68),
    'Shanghai': (121.47, 31.23),
    'Hong Kong': (114.17, 22.28),
    'Singapore': (103.85, 1.29),
    'Manila': (120.98, 14.60),
    'Jakarta': (106.85, -6.21),
    'Mumbai': (72.88, 19.08),
    'Sydney': (151.21, -33.87),
    'Melbourne': (144.96, -37.81),
    'Auckland': (174.76, -36.85),
    # Europe
    'London': (-0.13, 51.51),
    'Paris': (2.35, 48.86),
    'Amsterdam': (4.90, 52.37),
    'Hamburg': (10.00, 53.55),
    'Barcelona': (2.17, 41.39),
    'Rome': (12.50, 41.90),
    'Athens': (23.73, 37.98),
    'Istanbul': (28.98, 41.02),
    # Americas
    'New York': (-74.01, 40.71),
    'Miami': (-80.19, 25.76),
    'Houston': (-95.37, 29.76),
    'New Orleans': (-90.07, 29.95),
    'Los Angeles': (-118.24, 34.05),
    'San Francisco': (-122.42, 37.77),
    'Vancouver': (-123.12, 49.28),
    'Rio de Janeiro': (-43.17, -22.91),
    'Buenos Aires': (-58.38, -34.60),
    # Middle East & Africa
    'Dubai': (55.30, 25.26),
    'Cape Town': (18.42, -33.92),
}


def get_random_files(directory_path, n=1, seed=42):
    """
    Randomly select and return paths of n files from a given directory.
    
    Args:
        directory_path (str): Path to the directory to search for files
        n (int): Number of random files to return (default: 1)
        seed (int, optional): Random seed for reproducible results. If provided, 
                             sets the random number generator state.
    
    Returns:
        list: List of full file paths. Returns empty list if directory doesn't exist 
              or doesn't contain enough files.
    
    Raises:
        ValueError: If n is not a positive integer
    """    
    # Validate that n is a positive integer
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Check if directory exists
    if not os.path.isdir(directory_path):
        return []
    
    # Get all files in the directory (excluding subdirectories)
    try:
        all_files = []
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                all_files.append(item_path)
    except PermissionError:
        # Return empty list if permission denied to read directory
        return []
    
    # Check if we have enough files
    if len(all_files) < n:
        return []
    
    # Randomly select n files without replacement
    selected_files = random.sample(all_files, n)
    
    return selected_files


def generate_hourly_timestamps(start_time, end_time):
    """
    Generate a list of hourly timestamps between start_time and end_time (inclusive).
    
    Args:
        start_time (str): Start time in format 'YYYYMMDDTHH'
        end_time (str): End time in format 'YYYYMMDDTHH'
    
    Returns:
        list: List of hourly timestamps in format 'YYYYMMDDTHH'
    
    Raises:
        ValueError: If start_time is after end_time or time format is invalid
    """
    
    # Parse input strings to datetime objects
    try:
        start_dt = datetime.strptime(start_time, '%Y%m%dT%H')
        end_dt = datetime.strptime(end_time, '%Y%m%dT%H')
    except ValueError as e:
        raise ValueError(f"Invalid time format. Expected 'YYYYMMDDTHH'. Error: {e}")
    
    # Validate that start time is not after end time
    if start_dt > end_dt:
        raise ValueError("Start time cannot be after end time")
    
    hourly_timestamps = []
    current_dt = start_dt
    
    # Generate hourly timestamps
    while current_dt <= end_dt:
        # Format datetime to hour-level precision: YYYYMMDDTHH
        timestamp = current_dt.strftime('%Y%m%dT%H')
        hourly_timestamps.append(timestamp)
        
        # Move to next hour
        current_dt += timedelta(hours=1)
    
    return hourly_timestamps


def visualize_wave_tensor(tensor_data, save_path, variable='hs', region=None, 
                          coord_file='./data/hindcast/hk_coordinates.npy'):
    """
    Debug utility function to visualize wave data tensor.
    
    Args:
        tensor_data (torch.Tensor or numpy.ndarray): Wave data tensor with shape [num_nodes,] or [num_nodes].
                                                     Supports PyTorch tensor or numpy array
        save_path (str): Path to save the visualization figure
        variable (str): Variable name, default is 'hs'. Options: 'hs', 'tp', 'dm', 'elev'
        region (tuple, optional): Map extent [min_lon, max_lon, min_lat, max_lat].
                                 Defaults to HK_REGION if not specified
        coord_file (str): Path to coordinate file, default is './data/hindcast/hk_coordinates.npy'
    
    Returns:
        matplotlib.figure.Figure: Generated figure object
    
    Example:
        >>> # Debug in training code
        >>> from modules.utils import visualize_wave_tensor
        >>> visualize_wave_tensor(x1[0], './debug_ground_truth.png')
        >>> visualize_wave_tensor(x1_pred[0], './debug_prediction.png')
    """
    from modules.plot.plot_demo import plot_wave_data
    
    # Use default HK_REGION if region is not specified
    if region is None:
        region = HK_REGION
    
    # Convert tensor to numpy array
    if isinstance(tensor_data, torch.Tensor):
        data_np = tensor_data.cpu().detach().numpy()
    else:
        data_np = np.array(tensor_data)
    
    # Ensure data is 1-dimensional
    if data_np.ndim != 1:
        raise ValueError(f"Expected 1D tensor with shape [num_nodes], got shape {data_np.shape}")
    
    # Load coordinate data
    if not os.path.exists(coord_file):
        raise FileNotFoundError(f"Coordinate file not found: {coord_file}")
    
    coords = np.load(coord_file)
    lats = coords[:, 0]
    lons = coords[:, 1]
    
    # Validate data shape matches coordinates
    if len(data_np) != len(lats):
        raise ValueError(
            f"Data length {len(data_np)} does not match coordinate length {len(lats)}"
        )
    
    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars={
            variable: (("node",), data_np)
        },
        coords={
            "lat": (("node",), lats),
            "lon": (("node",), lons),
            "time": np.datetime64('2000-01-01T00:00:00')  # Use dummy timestamp
        }
    )
    
    # Call plot_wave_data for visualization
    fig = plot_wave_data(
        ds,
        variable=variable,
        region=region,
        save_path=save_path
    )
    
    return fig


def backup_python_scripts(work_dir, project_root=None):
    """
    Backup all Python scripts from the project to the work directory.
    Maintains the original directory structure.
    
    Args:
        work_dir (str): Path to the work directory where code will be backed up
        project_root (str, optional): Root directory of the project to backup.
                                     If None, uses the parent directory of the modules folder.
    
    Returns:
        str: Path to the code backup directory
    
    Example:
        >>> from modules.utils import backup_python_scripts
        >>> backup_dir = backup_python_scripts('/path/to/work_dir')
        >>> print(f"Code backed up to: {backup_dir}")
    """
    # Determine project root if not provided
    if project_root is None:
        # Get the parent directory of the modules folder (project root)
        current_file = os.path.abspath(__file__)
        modules_dir = os.path.dirname(current_file)
        project_root = os.path.dirname(modules_dir)
    
    # Create code backup directory
    code_backup_dir = os.path.join(work_dir, "code_backup")
    os.makedirs(code_backup_dir, exist_ok=True)
    
    # Walk through project directory and copy all .py files
    for root, dirs, files in os.walk(project_root):
        # Skip output directories, data directories, external repro projects,
        # and hidden directories
        dirs[:] = [d for d in dirs
                   if not d.startswith('output') and not d.startswith('data')
                   and not d.startswith('.') and d != 'baseline_repros']
        
        for file in files:
            if file.endswith('.py'):
                src_path = os.path.join(root, file)
                # Compute relative path from project root
                rel_path = os.path.relpath(src_path, project_root)
                dst_path = os.path.join(code_backup_dir, rel_path)
                
                # Create destination directory if needed
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                # Copy the file
                shutil.copy2(src_path, dst_path)
    
    return code_backup_dir