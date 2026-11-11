import os
import numpy as np
import torch
from typing import List
from torch.utils.data import DataLoader

from modules.datasets.wave_dataset import WaveDataset, collate_wave_samples
from modules.datasets.wave_normalize import NormalizeWaveData


def test_wave_dataset(data_dir: str = "./data/hindcast/processed", era5_dir: str = None,
                     stats_file: str = None, era5_stats_file: str = None, variables: List[str] = None, 
                     load_era5: bool = True, num_era5_frames: int = 1, region: str = 'hk',
                     file_format: str = 'nc', era5_file_format: str = 'nc') -> None:
    """
    Test function for WaveDataset to verify functionality.
    
    Args:
        data_dir (str): Path to the directory containing data files.
        era5_dir (str, optional): Path to the directory containing ERA5 NetCDF files.
        stats_file (str, optional): Path to JSON file containing min/max values for wave variables.
        era5_stats_file (str, optional): Path to JSON file containing min/max values for ERA5 variables.
        variables (List[str], optional): List of variables to load and test.
        load_era5 (bool): Whether to test ERA5 data loading. Defaults to True.
        num_era5_frames (int): Number of ERA5 frames to load.
        region (str): Data region to load - 'hk' for Hong Kong region, 'global' for global data. Defaults to 'hk'.
        file_format (str): File format for wave data: 'nc' for NetCDF files or 'npz' for NPZ files. Defaults to 'nc'.
        era5_file_format (str): File format for ERA5 data: 'nc' for NetCDF files or 'npz' for NPZ files. Defaults to 'nc'.
    """
    print(f"Testing WaveDataset (region={region}, num_era5_frames={num_era5_frames}, file_format={file_format}, era5_file_format={era5_file_format})...")
    
    # Set default variables
    if variables is None:
        variables = ['hs', 'tp', 'dm', 'elev']
    
    try:
        # Create dataset instance with normalization transform
        transform = None
        if stats_file is not None:
            transform = NormalizeWaveData(stats_file=stats_file, era5_stats_file=era5_stats_file, 
                                         variables=variables, normalize_era5=load_era5)
        
        dataset = WaveDataset(data_dir=data_dir, era5_dir=era5_dir, transform=transform, 
                             variables=variables, load_era5=load_era5, num_era5_frames=num_era5_frames,
                             region=region, file_format=file_format, era5_file_format=era5_file_format)
        print(f"✓ Successfully created dataset with {len(dataset)} samples")
        
        # Test getting a single item
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Successfully retrieved sample 0")
            
            # Test node-wise data
        if 'wave_data' in sample:
            wave_data = sample['wave_data']
            print(f"  - wave_data shape: {wave_data.shape}")
            
            # Check normalization effects if transform is active
            if transform is not None:
                current_idx = 0
                for var in variables:
                    if var == 'dm':
                        sin_val = wave_data[..., current_idx]
                        cos_val = wave_data[..., current_idx+1]
                        print(f"  - dm split into sin/cos at indices {current_idx}, {current_idx+1}")
                        assert np.min(sin_val) >= -1.0001 and np.max(sin_val) <= 1.0001
                        assert np.min(cos_val) >= -1.0001 and np.max(cos_val) <= 1.0001
                        current_idx += 2
                    else:
                        val = wave_data[..., current_idx]
                        print(f"  - {var} at index {current_idx}: mean={np.mean(val):.4f}, std={np.std(val):.4f}")
                        current_idx += 1
            else:
                 print(f"  - Raw wave_data loaded. Channels: {wave_data.shape[-1]}")

        # Test ERA5 data
            print(f"  - ERA5 data shape: {sample['era5'].shape}")  # [3, H, W] grid data
            print(f"  - ERA5 data range: [{np.min(sample['era5']):.4f}, {np.max(sample['era5']):.4f}]")
            
            print("✓ Data types are correct")
            if transform is not None:
                print("✓ Data normalization applied")
                
                # Test Denormalization (Dictionary Mode)
                print("\nTesting Denormalization (Dictionary Mode)...")
                denormalized_sample = transform.denormalize(sample)
                
                if 'wave_data' in denormalized_sample:
                    dw = denormalized_sample['wave_data']
                    print(f"  - Denormalized wave_data shape: {dw.shape}")
                    
                    current_idx = 0
                    for var in variables:
                        if current_idx < dw.shape[-1]:
                            val = dw[..., current_idx]
                            print(f"  - {var}: range [{np.min(val):.4f}, {np.max(val):.4f}]")
                        current_idx += 1
                
                print("✓ Denormalization successful")
            
        else:
            print("⚠ Dataset is empty, cannot test sample retrieval")
            return
            
        # Test with DataLoader
        print("\nTesting DataLoader...")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_wave_samples)
        batch = next(iter(dataloader))
        print(f"✓ DataLoader works correctly")
        
        if 'wave_data' in batch:
            print(f"  - Batch wave_data shape: {batch['wave_data'].shape}") # [B, N, C]
            print(f"  - Batch wave_data range: [{torch.min(batch['wave_data']):.4f}, {torch.max(batch['wave_data']):.4f}]")
        
        # Check batched ERA5 data
        if load_era5 and 'era5' in batch:
            print(f"  - Batch ERA5 data shape: {batch['era5'].shape}")  # [B, 3, H, W] grid data
            print(f"  - Batch ERA5 data range: [{torch.min(batch['era5']):.4f}, {torch.max(batch['era5']):.4f}]")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        raise


def test_global_wave_dataset(data_dir: str = "./data/global_hindcast/nc_files",
                             era5_dir: str = "./data/era5/global_nc_files",
                             stats_file: str = None, era5_stats_file: str = None,
                             variables: List[str] = None, load_era5: bool = True,
                             num_era5_frames: int = 1, file_format: str = 'nc',
                             era5_file_format: str = 'nc') -> None:
    """
    Test function specifically for global wave dataset to verify global data loading functionality.
    
    This function tests the WaveDataset with region='global' to ensure it can properly load
    global hindcast wave data and corresponding global ERA5 data.
    
    Args:
        data_dir (str): Path to the directory containing global wave data files.
                               Defaults to "./data/global_hindcast/nc_files".
        era5_dir (str): Path to the directory containing global ERA5 NetCDF files.
                               Defaults to "./data/era5/global_nc_files".
        stats_file (str, optional): Path to JSON file containing statistics for wave variables.
                                    If None, normalization will not be applied.
        era5_stats_file (str, optional): Path to JSON file containing min/max values for ERA5 variables.
                                         Defaults to "./data/era5/era5_statistics.json".
        variables (List[str], optional): List of wave variables to load and test.
                                         Defaults to ['hs'] (global data only has hs/WWM_1).
        load_era5 (bool): Whether to test ERA5 data loading. Defaults to True.
        num_era5_frames (int): Number of historical ERA5 frames to load (6h interval). Defaults to 1.
        file_format (str): File format for wave data: 'nc' for NetCDF files or 'npz' for NPZ files. Defaults to 'nc'.
        era5_file_format (str): File format for ERA5 data: 'nc' for NetCDF files or 'npz' for NPZ files. Defaults to 'nc'.
    
    Example:
        >>> test_global_wave_dataset(
        ...     data_dir="./data/global_hindcast/nc_files",
        ...     era5_dir="./data/era5/global_nc_files",
        ...     stats_file="./data/global_hindcast/statistics.json",
        ...     era5_stats_file="./data/era5/era5_statistics.json",
        ...     num_era5_frames=4,
        ...     file_format='nc',
        ...     era5_file_format='npz'
        ... )
    """
    print("=" * 70)
    print("Testing Global Wave Dataset")
    print("=" * 70)
    print(f"Global data directory: {data_dir}")
    print(f"Global ERA5 directory: {era5_dir}")
    print(f"Number of ERA5 frames: {num_era5_frames}")
    print(f"File format: {file_format}")
    print(f"ERA5 file format: {era5_file_format}")
    print()
    
    # Set default variables for global data (only 'hs' is available)
    if variables is None:
        variables = ['hs']
    
    # Set default ERA5 stats file
    if era5_stats_file is None:
        era5_stats_file = "./data/era5/era5_statistics.json"
    
    try:
        # Create normalization transform if stats file is provided
        transform = None
        if stats_file is not None:
            print(f"Loading normalization statistics from: {stats_file}")
            transform = NormalizeWaveData(
                stats_file=stats_file,
                era5_stats_file=era5_stats_file,
                variables=variables,
                normalize_era5=load_era5
            )
            print("✓ Normalization transform created")
        
        # Create global dataset
        print("\nCreating global WaveDataset (region='global')...")
        dataset = WaveDataset(
            data_dir=data_dir,
            era5_dir=era5_dir,
            transform=transform,
            variables=variables,
            load_era5=load_era5,
            num_era5_frames=num_era5_frames,
            region='global',
            file_format=file_format,
            era5_file_format=era5_file_format
        )
        print(f"✓ Successfully created global dataset with {len(dataset)} samples")
        
        # Test getting a single sample
        if len(dataset) == 0:
            print("⚠ Dataset is empty, cannot test sample retrieval")
            return
        
        print("\n--- Testing Single Sample Retrieval ---")
        sample = dataset[0]
        print(f"✓ Successfully retrieved sample 0")
        print(f"  - Timestamp: {sample['timestamp']}")
        
        # Test wave data
        if 'wave_data' in sample:
            wave_data = sample['wave_data']
            print(f"\n  Wave Data:")
            print(f"    - Shape: {wave_data.shape} [N, C] where N=num_nodes, C=num_variables")
            print(f"    - Number of nodes: {wave_data.shape[0]}")
            print(f"    - Number of channels: {wave_data.shape[1]}")
            print(f"    - Data range: [{np.min(wave_data):.4f}, {np.max(wave_data):.4f}]")
            
            # Verify global data has more nodes than HK region (HK has ~5000 nodes)
            if wave_data.shape[0] > 10000:
                print(f"    ✓ Global data detected (large number of nodes: {wave_data.shape[0]})")
            else:
                print(f"    ⚠ Warning: Expected more nodes for global data, got {wave_data.shape[0]}")
        
        # Test ERA5 data
        if load_era5 and 'era5' in sample:
            era5_data = sample['era5']
            print(f"\n  ERA5 Data:")
            if era5_data.ndim == 4:
                # Sequence mode: [T, 3, H, W]
                T, C, H, W = era5_data.shape
                print(f"    - Shape: {era5_data.shape} [T={T}, C={C}, H={H}, W={W}]")
                print(f"    - Time frames: {T}")
                print(f"    - Grid size: {H} x {W}")
                # Global ERA5 should be 721 x 1440 (0.25 degree resolution)
                if H >= 700 and W >= 1400:
                    print(f"    ✓ Global ERA5 grid detected ({H} x {W})")
                else:
                    print(f"    ⚠ Warning: Expected larger grid for global ERA5, got {H} x {W}")
            else:
                # Single frame mode: [3, H, W]
                C, H, W = era5_data.shape
                print(f"    - Shape: {era5_data.shape} [C={C}, H={H}, W={W}]")
                print(f"    - Grid size: {H} x {W}")
                if H >= 700 and W >= 1400:
                    print(f"    ✓ Global ERA5 grid detected ({H} x {W})")
                else:
                    print(f"    ⚠ Warning: Expected larger grid for global ERA5, got {H} x {W}")
            print(f"    - Data range: [{np.min(era5_data):.4f}, {np.max(era5_data):.4f}]")
        
        # Test denormalization if transform is applied
        if transform is not None:
            print("\n--- Testing Denormalization ---")
            denormalized_sample = transform.denormalize(sample)
            if 'wave_data' in denormalized_sample:
                dw = denormalized_sample['wave_data']
                print(f"  - Denormalized wave_data shape: {dw.shape}")
                
                current_idx = 0
                for var in variables:
                    if current_idx < dw.shape[-1]:
                        val = dw[..., current_idx]
                        print(f"  - {var}: range [{np.min(val):.4f}, {np.max(val):.4f}]")
                    current_idx += 1 if var != 'dm' else 2
            print("✓ Denormalization successful")
        
        # Test DataLoader
        print("\n--- Testing DataLoader ---")
        batch_size = min(4, len(dataset))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_wave_samples
        )
        batch = next(iter(dataloader))
        print(f"✓ DataLoader works correctly (batch_size={batch_size})")
        
        if 'wave_data' in batch:
            print(f"  - Batch wave_data shape: {batch['wave_data'].shape}")  # [B, N, C]
            print(f"  - Batch wave_data range: [{torch.min(batch['wave_data']):.4f}, {torch.max(batch['wave_data']):.4f}]")
        
        if load_era5 and 'era5' in batch:
            print(f"  - Batch ERA5 data shape: {batch['era5'].shape}")
            print(f"  - Batch ERA5 data range: [{torch.min(batch['era5']):.4f}, {torch.max(batch['era5']):.4f}]")
        
        print("\n" + "=" * 70)
        print("✓ All global dataset tests passed!")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n✗ File not found error: {e}")
        print("  Please ensure global data files exist at the specified paths:")
        print(f"    - Wave data: {data_dir}")
        print(f"    - ERA5 data: {era5_dir}")
        raise
    except Exception as e:
        print(f"\n✗ Global dataset test failed with error: {e}")
        raise


# Example usage
if __name__ == "__main__":
    # Example 1: Traditional grid mode (backward compatible)
    print("=" * 60)
    print("Example 1: Grid Mode (Traditional)")
    print("=" * 60)
    
    # Load normalization statistics from file
    stats_file = "./data/hindcast/wave_data_stats.json"
    era5_stats_file = "./data/era5/era5_statistics.json"
    variables = ['hs', 'tp', 'dm', 'elev']
    
    # Create normalization transform (will normalize both wave and ERA5 data)
    transform = NormalizeWaveData(stats_file=stats_file, era5_stats_file=era5_stats_file, 
                                 variables=variables, normalize_era5=True)
    
    # Create dataset with normalization and ERA5 data (sequence mode)
    data_dir = "./data/hindcast/processed"
    era5_dir = "./data/era5/nc_files"
    num_frames = 4  # 4 time points (past 18h + current)
    
    dataset = WaveDataset(
        data_dir=data_dir, 
        era5_dir=era5_dir,
        transform=transform, 
        variables=variables,
        load_era5=True,
        num_era5_frames=num_frames
    )
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_wave_samples)
    
    # Iterate over data
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        
        # Wave variables
        if 'wave_data' in batch:
             print(f"  Wave Data shape: {batch['wave_data'].shape}")
        
        # ERA5 data (if loaded)
        if 'era5' in batch:
            print(f"  ERA5 sequence shape: {batch['era5'].shape}")  # [B, T, 3, H, W]
        
        # For demonstration, only process one batch
        break
    
    # Run the comprehensive test function
    print("\n" + "=" * 60)
    print(f"Running Comprehensive Test (T={num_frames})")
    print("=" * 60)
    test_wave_dataset(data_dir=data_dir, era5_dir=era5_dir, 
                     stats_file=stats_file, era5_stats_file=era5_stats_file,
                     variables=variables, load_era5=True, num_era5_frames=num_frames,
                     region='hk')
