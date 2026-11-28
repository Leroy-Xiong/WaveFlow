import os
import json
import numpy as np
from typing import List, Union

from modules.utils import ERA5_VARIABLE_NAMES


class NormalizeWaveData:
    """
    Transform class to normalize wave data (hs, tp, dm, elev, mwd, ssh) and ERA5 data.

    Normalization methods:
    - hs, tp: Log transform + Standardization: (log(x+1) - mu_log) / sigma_log
    - dm, mwd: Sine-Cosine decomposition: sin(rad), cos(rad) where rad = deg * pi / 180
    - elev, ssh: Standardization (Z-score): (x - mu) / sigma
    - ERA5: Min-Max normalization to [-1, 1] (kept as is for now unless specified)
    """
    
    def __init__(self, stats_file: str, variables: List[str] = None, era5_stats_file: str = None,
                 normalize_era5: bool = True, use_tidal_elev: bool = False, tidal_elev_scale: float = 0.3,
                 log_power: float = 1.0):
        """
        Initialize the normalization transform.

        Args:
            stats_file (str): Path to JSON file containing stats for wave variables.
            variables (List[str], optional): List of wave variables to normalize.
                                            Defaults to ['hs', 'tp', 'dm', 'elev'].
            era5_stats_file (str, optional): Path to JSON file containing min/max values for ERA5 variables.
                                             Defaults to './data/era5/era5_statistics.json'.
            normalize_era5 (bool): Whether to normalize ERA5 data. Defaults to True.
            use_tidal_elev (bool): Whether to add tidal sin signal to elev channel.
            tidal_elev_scale (float): Amplitude of tidal signal added to z-scored elev.
            log_power (float): Exponent applied after log1p for hs/tp:
                               y = (log1p(x))^log_power. 1.0 = legacy behavior.
                               <1.0 compresses the high-value tail closer to the
                               mainstream distribution (extreme-aware training).
        """
        if stats_file is None:
            raise ValueError("stats_file must be provided")
        
        # Set default variables
        if variables is None:
            self.variables = ['hs', 'tp', 'dm', 'elev']
        else:
            self.variables = variables
            
        # Read wave statistics from file
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        # Store stats for each wave variable
        self.stats = stats
        self.stats_file = stats_file
        self.log_power = float(log_power if log_power and log_power > 0 else 1.0)

        # Tidal elevation encoding
        self.use_tidal_elev = use_tidal_elev
        self.tidal_elev_scale = tidal_elev_scale
        
        # Validate that required stats are present
        for var in self.variables:
            if var in ['hs', 'tp']:
                if f'{var}_log_mean' not in stats or f'{var}_log_std' not in stats:
                    pass
            elif var in ['elev', 'ssh']:
                if f'{var}_mean' not in stats or f'{var}_std' not in stats:
                    pass
            elif var in ['dm', 'mwd']:
                pass  # No stats needed for sin/cos decomposition
        
        # Handle ERA5 normalization (Min-Max)
        self.normalize_era5 = normalize_era5
        self.era5_min_values = {}
        self.era5_max_values = {}
        self.era5_ranges = {}
        
        if normalize_era5:
            # Set default ERA5 stats file path
            if era5_stats_file is None:
                era5_stats_file = './data/era5/era5_statistics.json'
            
            # Read ERA5 statistics from file
            with open(era5_stats_file, 'r') as f:
                era5_stats = json.load(f)
            
            # Define ERA5 variable names (use full names throughout)
            era5_var_names = ERA5_VARIABLE_NAMES
            
            # Extract min/max values for each ERA5 variable
            for var_info in era5_stats['variables']:
                var_name = var_info['variable']
                if var_name in era5_var_names:
                    self.era5_min_values[var_name] = var_info['min_value']
                    self.era5_max_values[var_name] = var_info['max_value']
                    self.era5_ranges[var_name] = self.era5_max_values[var_name] - self.era5_min_values[var_name]
                    
                    if self.era5_ranges[var_name] == 0:
                        raise ValueError(f"Min and max values are equal for ERA5 {var_name}, cannot normalize")
    
    def __call__(self, sample: dict) -> dict:
        """
        Apply normalization to the sample.
        
        Args:
            sample (dict): Input sample with 'wave_data' tensor [N, C] and 'era5' or 'era5_on_nodes'
            
        Returns:
            dict: Normalized sample. 
                  - wave_data: [N, C_new] where C_new >= C (dm splits into sin/cos)
                  - era5: Min-Max normalized to [-1, 1] (if present)
                  - era5_on_nodes: Min-Max normalized to [-1, 1] (if present)
        """
        # Normalize wave variables
        if 'wave_data' in sample:
            data = sample['wave_data'] # [N, C]
            channels = []
            
            for i, var in enumerate(self.variables):
                # Extract channel i
                vals = data[..., i]
                
                if var in ['hs', 'tp']:
                    # Log Transform + Standardization: (log(x+1) - mu_log) / sigma_log
                    log_mean = self.stats.get(f'{var}_log_mean')
                    log_std = self.stats.get(f'{var}_log_std')
                    
                    if log_mean is None or log_std is None:
                        raise ValueError(f"Missing log statistics for {var}")
                    
                    # Apply transform: (log1p(x))^p, then z-score (p=1 -> legacy log1p)
                    log_val = np.log1p(vals)
                    if self.log_power != 1.0:
                        log_val = np.power(np.maximum(log_val, 0.0), self.log_power)
                    normalized = (log_val - log_mean) / log_std
                    channels.append(normalized)

                elif var in ['elev', 'ssh']:
                    # Z-score: (x - mean) / std
                    mean = self.stats.get(f'{var}_mean')
                    std = self.stats.get(f'{var}_std')
                    
                    if mean is None or std is None:
                        raise ValueError(f"Missing statistics for {var}")
                    
                    normalized = (vals - mean) / std
                    channels.append(normalized)
                    
                elif var in ['dm', 'mwd']:
                    # Sine-Cosine Decomposition
                    # Convert degrees to radians
                    deg = vals
                    rad = deg * np.pi / 180.0
                    
                    sin_val = np.sin(rad)
                    cos_val = np.cos(rad)
                    
                    channels.append(sin_val)
                    channels.append(cos_val)
            
            # Stack all channels back into a tensor
            sample['wave_data'] = np.stack(channels, axis=-1)

            # Add tidal encoding to elev channel (channel index -1 = elev)
            if self.use_tidal_elev and 'tod' in sample:
                hour = sample['tod'] * 24.0  # tod in [0,1) → hour in [0, 24)
                tide = self.tidal_elev_scale * np.sin(2 * np.pi * hour / 12.42)
                sample['wave_data'][:, -1] += tide  # elev is the last channel

        # Normalize ERA5 data on regular grid if present and normalization is enabled
        if self.normalize_era5 and 'era5' in sample:
            era5_data = sample['era5']  # Shape: [3, H, W] or [T, 3, H, W]
            
            # Normalize each channel separately
            era5_var_names = ERA5_VARIABLE_NAMES
            
            is_sequence = (era5_data.ndim == 4)
            
            for i, var_name in enumerate(era5_var_names):
                # Extract values for this variable
                if is_sequence:
                    # era5_data: [T, 3, H, W], so era5_data[:, i] is [T, H, W]
                    vals = era5_data[:, i]
                else:
                    # era5_data: [3, H, W], so era5_data[i] is [H, W]
                    vals = era5_data[i]
                
                # Normalize to [0, 1] range first
                normalized = (vals - self.era5_min_values[var_name]) / self.era5_ranges[var_name]
                
                # Scale to [-1, 1] range
                normalized = normalized * 2 - 1
                
                # Update the data
                if is_sequence:
                    era5_data[:, i] = normalized
                else:
                    era5_data[i] = normalized
            
            sample['era5'] = era5_data
        
        # Normalize ERA5 data on wave nodes if present and normalization is enabled
        if self.normalize_era5 and 'era5_on_nodes' in sample:
            era5_on_nodes = sample['era5_on_nodes']  # Shape: [N, 3*T]
            
            # era5_on_nodes has shape [N, 3*T] where channels are [u_t0, v_t0, p_t0, u_t1, v_t1, p_t1, ...]
            # Normalize each ERA5 variable across all time frames
            era5_var_names = ERA5_VARIABLE_NAMES
            num_era5_vars = len(era5_var_names)
            
            # Reshape to [N, T, 3] to process each variable type
            N = era5_on_nodes.shape[0]
            total_channels = era5_on_nodes.shape[1]
            T = total_channels // num_era5_vars
            
            era5_reshaped = era5_on_nodes.reshape(N, T, num_era5_vars)
            
            for i, var_name in enumerate(era5_var_names):
                # Extract all time frames for this variable: [N, T]
                vals = era5_reshaped[:, :, i]
                
                # Normalize to [0, 1] range first
                normalized = (vals - self.era5_min_values[var_name]) / self.era5_ranges[var_name]
                
                # Scale to [-1, 1] range
                normalized = normalized * 2 - 1
                
                # Update the data
                era5_reshaped[:, :, i] = normalized
            
            # Reshape back to [N, 3*T]
            sample['era5_on_nodes'] = era5_reshaped.reshape(N, total_channels)
        
        return sample
    
    def denormalize(self, data: Union[np.ndarray, dict], variable: str = None) -> Union[np.ndarray, dict]:
        """
        Apply inverse normalization to convert data back to original scale.
        
        Args:
            data (Union[np.ndarray, dict]): Normalized data (array [..., C_new]) or sample dictionary.
            variable (str, optional): Variable name. If provided, assumes data is that single variable's data (legacy).
                                      If None and data is array, assumes data is the full tensor [..., C_new].
            
        Returns:
            Union[np.ndarray, dict]: Denormalized data in original scale.
            
        Raises:
            ValueError: If data shape does not match expected channels for the variables.
        """
        # Handle dictionary input (full sample denormalization)
        if isinstance(data, dict):
            sample = data.copy()
            if 'wave_data' in sample:
                sample['wave_data'] = self.denormalize(sample['wave_data'])
            return sample

        # Handle array input
        if variable is None:
            # Validate input data shape
            if not isinstance(data, np.ndarray):
                raise TypeError(f"Expected numpy array, got {type(data).__name__}")
            
            if data.ndim < 1:
                raise ValueError(f"Data must have at least 1 dimension, got shape {data.shape}")
            
            # Calculate expected number of channels
            expected_channels = sum(2 if var in ['dm', 'mwd'] else 1 for var in self.variables)
            actual_channels = data.shape[-1]
            
            if actual_channels != expected_channels:
                raise ValueError(
                    f"Data has {actual_channels} channels but expected {expected_channels} "
                    f"for variables {self.variables} (dm expands to sin/cos)"
                )
            
            # Assume data is [..., C_new] and we need to reconstruct [..., C]
            # We iterate through self.variables to know what we expect in output
            # and track index in input data.
            
            out_channels = []
            current_idx = 0
            
            for var in self.variables:
                if var in ['dm', 'mwd']:
                    # Expect sin and cos at current_idx and current_idx+1
                    sin_val = data[..., current_idx]
                    cos_val = data[..., current_idx + 1]
                    current_idx += 2
                    
                    # Calculate angle
                    rad = np.arctan2(sin_val, cos_val)
                    deg = rad * 180.0 / np.pi
                    deg = (deg + 360.0) % 360.0
                    out_channels.append(deg)
                    
                elif var in ['hs', 'tp']:
                    val = data[..., current_idx]
                    current_idx += 1
                    
                    log_mean = self.stats.get(f'{var}_log_mean')
                    log_std = self.stats.get(f'{var}_log_std')
                    
                    if log_mean is None or log_std is None:
                        raise ValueError(f"Missing log statistics for {var} in stats file")
                    
                    log_val = val * log_std + log_mean
                    if self.log_power != 1.0:
                        log_val = np.sign(log_val) * np.abs(log_val) ** (1.0 / self.log_power)
                    denorm = np.expm1(log_val)
                    denorm = np.maximum(denorm, 0)
                    out_channels.append(denorm)
                    
                elif var == 'elev' or var == 'ssh':
                    val = data[..., current_idx]
                    current_idx += 1
                    
                    mean = self.stats.get(f'{var}_mean')
                    std = self.stats.get(f'{var}_std')
                    
                    if mean is None or std is None:
                        raise ValueError(f"Missing statistics for {var} in stats file")
                    
                    denorm = val * std + mean
                    out_channels.append(denorm)
            
            return np.stack(out_channels, axis=-1)

        # Handle legacy single variable denormalization (if data is passed for a specific variable)
        if variable in ['hs', 'tp']:
            log_mean = self.stats.get(f'{variable}_log_mean')
            log_std = self.stats.get(f'{variable}_log_std')
            
            log_val = data * log_std + log_mean
            if self.log_power != 1.0:
                log_val = np.sign(log_val) * np.abs(log_val) ** (1.0 / self.log_power)
            denormalized = np.expm1(log_val)
            denormalized = np.maximum(denormalized, 0)
            return denormalized
            
        elif variable == 'elev':
            mean = self.stats.get(f'{variable}_mean')
            std = self.stats.get(f'{variable}_std')
            
            denormalized = data * std + mean
            return denormalized
            
        elif variable in ['dm', 'mwd']:
             if isinstance(data, np.ndarray) and data.shape[-1] == 2:
                 sin_val = data[..., 0]
                 cos_val = data[..., 1]
                 rad = np.arctan2(sin_val, cos_val)
                 deg = rad * 180.0 / np.pi
                 deg = (deg + 360.0) % 360.0
                 return deg
             else:
                 raise NotImplementedError("For 'dm'/'mwd', input data must have shape [..., 2] containing (sin, cos) in last dimension.")
        
        elif variable in ['dm_sin', 'dm_cos']:
             return data
             
        else:
             raise ValueError(f"Variable '{variable}' not found or not supported for denormalization")


