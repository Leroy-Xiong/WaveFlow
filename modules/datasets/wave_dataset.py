import os
import ast
import struct
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
from typing import List, Tuple, Optional, Union, Sequence
from collections import OrderedDict
from glob import glob
import json
from tqdm import tqdm
from datetime import datetime, timedelta

from modules.utils import (HK_REGION, GLOBAL_REGION, WAVE_VARIABLES, ERA5_VARIABLE_NAMES, 
                            ERA5_HK_LON_RANGE_1, ERA5_HK_LAT_RANGE_1, ERA5_HK_LON_RANGE_2, ERA5_HK_LAT_RANGE_2, 
                            ERA5_HK_LON_RANGE_3, ERA5_HK_LAT_RANGE_3, ERA5_GLOBAL_LON_RANGE, ERA5_GLOBAL_LAT_RANGE)


# --- Absolute-time feature encoding (tidal conditioning, v3) ---
# Hour-of-day alone carries almost no tidal information: the dominant M2 tide
# (12.42 h) drifts ~0.84 h/day against the clock, so E[tide | hour] ~= 0 over
# multi-year data. The features below are sin/cos of ABSOLUTE time at the major
# tidal-constituent periods (+ fortnightly/monthly/annual cycles), which makes
# the astronomical tide a deterministic, learnable function of the inputs.
# A harmonic-regression probe (modules/tod_harmonic_probe.py) using exactly
# these periods reaches elev RMSE ~0.26 m on the 2021-2022 test set, vs ~0.60 m
# for hour-of-day-only features.
TIME_FEATURE_PERIODS_H = {
    "M2": 12.4206,
    "S2": 12.0000,
    "N2": 12.6583,
    "K2": 11.9672,
    "K1": 23.9345,
    "O1": 25.8193,
    "P1": 24.0659,
    "S1": 24.0000,    # solar day (hour-of-day, periodic)
    "M4": 6.2103,
    "Mf": 327.86,     # fortnightly
    "Mm": 661.31,     # monthly
    "Sa": 8765.82,    # annual
    "Ssa": 4382.91,   # semi-annual
}
TIME_FEAT_DIM = 2 * len(TIME_FEATURE_PERIODS_H)  # sin + cos per period
_TIME_FEATURE_EPOCH = datetime(1970, 1, 1, 0)


def compute_time_features(timestamp: str) -> np.ndarray:
    """Compute absolute-time sin/cos features from a 'YYYYMMDDTHH' timestamp.

    Returns:
        np.ndarray: float32 array of shape [TIME_FEAT_DIM] = [26].
    """
    dt = datetime.strptime(timestamp, "%Y%m%dT%H")
    t_h = (dt - _TIME_FEATURE_EPOCH).total_seconds() / 3600.0
    feats = np.empty(TIME_FEAT_DIM, dtype=np.float32)
    for i, T in enumerate(TIME_FEATURE_PERIODS_H.values()):
        phase = 2.0 * np.pi * t_h / T
        feats[2 * i] = np.sin(phase)
        feats[2 * i + 1] = np.cos(phase)
    return feats


class WaveDataset(Dataset):
    """
    A PyTorch Dataset for loading wave data (hs, tp, dm, elev) and ERA5 data (wind and pressure).
    
    This dataset loads all wave variables from NetCDF files or NPZ files that were processed from 
    the original MATLAB hindcast data, and corresponding ERA5 data for the same time.
    
    The dataset returns data in node-wise format [N,] where N is the number of nodes.
    Patch organization is handled by the model (PatchedGraphTransformer) if needed.
    """

    def __init__(self, data_dir: str, era5_dir: str = None, transform=None,
                 variables: List[str] = None, split: str = 'train', load_era5: bool = True, era5_region: int = 1,
                 num_era5_frames: int = 1, region: str = 'hk', file_format: str = 'nc',
                 train_max_year: int = None,
                 era5_file_format: str = 'nc', coord_file_path: str = None,
                 interpolate_era5_to_nodes: bool = False, use_preinterpolated_era5: bool = False,
                 atm_source: str = 'era5', ens_lead_time: int = 24, ens_dir: str = None,
                 gencast_dir: str = None, atm_members: int = 1, ens_init: str = None,
                 use_mmap: bool = True, daily_whole_day_cache: bool = False,
                 oversample_alpha: float = 0.0, oversample_tau: float = 3.0,
                 oversample_weights_file: str = None):
        """
        Initialize the WaveDataset.
        
        Args:
            data_dir (str): Path to the directory containing wave data files
            era5_dir (str, optional): Path to the directory containing ERA5 NetCDF files.
                                     Defaults to './data/era5/nc_files'.
            transform (callable, optional): Optional transform to be applied on a sample
            variables (List[str], optional): List of wave variables to load. 
                                            Defaults to ['hs', 'tp', 'dm', 'elev'].
            split (str): 'train', 'val' or 'test'. Test set includes 2021 and 2022 data,
                        train set includes all other years. Defaults to 'train'.
            load_era5 (bool): Whether to load ERA5 data. Defaults to True.
            era5_region (int): Preset index for ERA5 region (1, 2, or 3). Defaults to 1.
            num_era5_frames (int): Number of historical ERA5 frames to load (6h interval). Defaults to 1.
            region (str): Data region to load - 'hk' for Hong Kong region, 'global' for global data. Defaults to 'hk'.
            file_format (str): File format for wave data - 'nc', 'npz', 'npy' (uncompressed)
                or 'npy_daily' (per-day packed [8, N, 3], read via pread + fd cache). Defaults to 'nc'.
            era5_file_format (str): File format for ERA5 data - 'nc', 'npz', 'npy' (uncompressed f32),
                'npy16' (uncompressed f16, msl stored as offset msl-1e5) or 'npy_daily'
                (per-day packed [8, N, 3] f32). Defaults to 'nc'.
            coord_file_path (str, optional): Path to coordinate file for wave node positions. 
                                            Required when interpolate_era5_to_nodes=True.
            interpolate_era5_to_nodes (bool): Whether to interpolate ERA5 data to wave node positions.
                                             If True, returns 'era5_on_nodes' in sample. Defaults to False.
            use_preinterpolated_era5 (bool): Whether to use pre-interpolated ERA5 data.
                                            If True, checks for pre-interpolated files in {era5_dir}_preinterpolated.
                                            Defaults to False.
            atm_source (str): Atmospheric data source - 'era5' for ERA5 reanalysis or 'ens' for 
                             IFS ENS forecast data. Defaults to 'era5'.
            ens_lead_time (int): ENS forecast lead time in hours (e.g., 24 means 24h forecast).
                                Only used when atm_source='ens'. Defaults to 24.
            ens_dir (str, optional): Path to ENS data directory. Defaults to './data/ifs/ens'.
            ens_init (str, optional): Fixed forecast issue time ('YYYYMMDDTHH'). When
                                set, every valid time uses this same init and the lead
                                is derived per timestamp (one operational cycle along
                                its trajectory). Defaults to None (= init = valid -
                                ens_lead_time, the legacy per-lead behaviour).
            atm_members (int): Number of atmospheric forcing ensemble members on disk.
                1 (default) = single forcing file per (init, lead), legacy ens-mean /
                fc0 behaviour. >1 = per-member files `{prefix}_m{mm}_{init}.nc`
                (member index 0..atm_members-1); __getitem__ then loads member 0 by
                default and the sampler can request any member via
                load_member_condition(ts, member_idx). Only valid with
                atm_source='ens'/'gencast'.
        """
        self.data_dir = data_dir
        self.era5_dir = era5_dir if era5_dir is not None else './data/era5/nc_files'
        self.transform = transform
        self.split = split
        self.load_era5 = load_era5
        self.era5_region = era5_region
        self.num_era5_frames = num_era5_frames
        self.region = region.lower()
        self.file_format = file_format.lower()
        self.era5_file_format = era5_file_format.lower()
        self.interpolate_era5_to_nodes = interpolate_era5_to_nodes
        self.use_preinterpolated_era5 = use_preinterpolated_era5

        # Cache for pre-computed KD-tree interpolation weights (shared across all timestamps)
        self._interp_indices = None       # [N, k] nearest neighbor indices
        self._interp_weights = None       # [N, k] normalized IDW weights
        self._interp_exact_mask = None    # [N,] mask for exact-match nodes

        # ENS / GenCast forecast settings
        self.atm_source = atm_source.lower()
        if self.atm_source not in ['era5', 'ens', 'gencast']:
            raise ValueError(f"atm_source must be 'era5', 'ens', or 'gencast', got '{atm_source}'")
        self.ens_lead_time = ens_lead_time
        self.ens_dir = os.path.abspath(os.path.expanduser(ens_dir)) if ens_dir is not None else os.path.abspath('./data/ifs/ens')
        # Optional fixed issue time: one forecast cycle, lead derived per timestamp
        self.ens_init = str(ens_init).strip() if ens_init else None
        self._ens_init_dt = (datetime.strptime(self.ens_init, "%Y%m%dT%H")
                             if self.ens_init else None)
        # GenCast: uses same directory layout and frame-planning as ENS, just different prefix
        self.gencast_dir = os.path.abspath(os.path.expanduser(gencast_dir)) if gencast_dir is not None else os.path.abspath('./data/gencast_global')
        # Per-member forecast forcing (WaveFlow sample k is conditioned on member k).
        # atm_members > 1 requires per-member files; the plain DataLoader path then
        # serves member 0 and the sampler fetches each member explicitly.
        self.atm_members = int(atm_members)
        if self.atm_members > 1 and self.atm_source == 'era5':
            raise ValueError("atm_members > 1 requires atm_source='ens' or 'gencast'")
        self.atm_default_member = 0 if self.atm_members > 1 else None
        self.use_mmap = use_mmap

        # Load wave node coordinates for ERA5 interpolation
        if self.interpolate_era5_to_nodes:
            if coord_file_path is None:
                raise ValueError("coord_file_path must be provided when interpolate_era5_to_nodes=True")
            self.wave_coords = np.load(coord_file_path)  # [N, 2] with [lat, lon]
            self.num_wave_nodes = self.wave_coords.shape[0]
        else:
            self.wave_coords = None
        
        # Validate file_format
        if self.file_format not in ['nc', 'npz', 'npy', 'npy_daily']:
            raise ValueError(f"file_format must be 'nc', 'npz', 'npy' or 'npy_daily', got '{file_format}'")
        
        # Validate era5_file_format
        if self.era5_file_format not in ['nc', 'npz', 'npy', 'npy16', 'npy_daily']:
            raise ValueError(f"era5_file_format must be 'nc', 'npz', 'npy', 'npy16' or 'npy_daily', got '{era5_file_format}'")
        
        if (self.file_format == 'npy_daily' or self.era5_file_format == 'npy_daily') and self.region != 'global':
            raise ValueError("npy_daily format is only supported for region='global'")
        
        # Set file_pattern based on region and file_format
        if self.region == 'global':
            if self.file_format == 'npz':
                self.file_pattern = "global_hindcast_*.npz"
            elif self.file_format in ('npy', 'npy_daily'):
                self.file_pattern = "global_hindcast_*.npy"
            else:
                self.file_pattern = "global_hindcast_*.nc"
        else:
            if self.file_format == 'npz':
                self.file_pattern = "wave_*.npz"
            elif self.file_format == 'npy':
                self.file_pattern = "wave_*.npy"
            else:
                self.file_pattern = "wave_*.nc"
        
        # Set default variables based on region
        if variables is None:
            if self.region == 'global':
                # Global data: hs (SWH), mwd (MWD), ssh (SSH)
                self.variables = ['hs', 'mwd', 'ssh']
            else:
                # HK data has all 4 variables
                self.variables = ['hs', 'tp', 'dm', 'elev']
        else:
            self.variables = variables
        
        # Daily-packed NPY mode: file_list comes from the daily index (timestamps),
        # not from globbing per-timestamp files.
        self._daily_fd_cache = OrderedDict()  # day_file_path -> (fd, data_offset); per-worker
        self._daily_fd_cache_max = 512
        self._wave_daily_index = None  # ts -> (day, slot)
        self._era5_daily_index = None  # ts -> (day, slot)
        # Column names of per-day packed wave files; default is the legacy 3-var
        # layout, overridden by wave_daily_columns.json when present (e.g. 4-var).
        self._daily_wave_columns = list(self.NPY_WAVE_COLUMNS)
        # Same convention for per-timestamp NPY files: wave_columns.json in
        # data_dir overrides the legacy 3-var column layout (e.g. 4-var).
        self._npy_columns = list(self.NPY_WAVE_COLUMNS)
        if self.file_format == 'npy':
            npy_cols_meta = os.path.join(self.data_dir, 'wave_columns.json')
            if os.path.exists(npy_cols_meta):
                with open(npy_cols_meta) as fh:
                    self._npy_columns = list(json.load(fh)['columns'])
        # Whole-day RAM cache: only use with day-locality sampling (block shuffle
        # with sequential days within a block), otherwise it amplifies wave reads 8x.
        self.daily_whole_day_cache = daily_whole_day_cache
        self._daily_ram_cache = OrderedDict()  # day_file_path -> [8, N, C] float32
        self._daily_ram_cache_max = 32  # ~2 GB per worker

        if self.file_format == 'npy_daily':
            daily_dir = self._wave_daily_dir()
            idx_data = np.load(os.path.join(daily_dir, 'wave_daily_ts_index.npz'), allow_pickle=True)
            entries = {str(ts): (str(d), int(s)) for ts, d, s in
                       zip(idx_data['timestamps'], idx_data['days'], idx_data['slots'])}
            idx_data.close()
            cols_meta = os.path.join(daily_dir, 'wave_daily_columns.json')
            if os.path.exists(cols_meta):
                with open(cols_meta) as fh:
                    self._daily_wave_columns = list(json.load(fh)['columns'])
            if split == 'test':
                self._wave_daily_index = {ts: v for ts, v in entries.items() if ts[:4] in ('2021', '2022')}
            elif split == 'val':
                self._wave_daily_index = {ts: v for ts, v in entries.items() if ts[:4] in ('2019', '2020')}
            elif split == 'train':
                if train_max_year is not None:
                    self._wave_daily_index = {ts: v for ts, v in entries.items() if int(ts[:4]) <= train_max_year}
                else:
                    self._wave_daily_index = {ts: v for ts, v in entries.items() if 1959 <= int(ts[:4]) <= 2020}
            else:
                raise ValueError(f"split must be 'train', 'val' or 'test', got '{split}'")
            # file_list holds timestamp strings (YYYYMMDDTHH) in daily mode
            self.file_list = sorted(self._wave_daily_index.keys())
            all_files = self.file_list  # for the empty-check below
        else:
            # Find all NetCDF files matching the pattern
            # Use recursive glob to search subdirectories (e.g., data/era5/global_nc_files/1959/*.nc)
            search_pattern = os.path.join(data_dir, '**', self.file_pattern)
            all_files = sorted(glob(search_pattern, recursive=True))
        
        if len(all_files) == 0:
            raise ValueError(f"No files found in {data_dir} matching pattern {self.file_pattern}")
        
        # Split files based on year in filename
        # Test set: 2021 and 2022; Val set: 2019 and 2020; Train set depends on region
        test_years = ['2021', '2022']
        val_years = ['2019', '2020']
        
        def extract_year_from_filename(filepath):
            """Extract 4-digit year from filename like wave_19700105T01.nc or global_hindcast_19500101_03.nc"""
            filename = os.path.basename(filepath)
            # Handle different filename formats
            if filename.startswith('wave_'):
                # HK format: wave_YYYYMMDDTHH.nc
                year_str = filename.replace('wave_', '')[:4]
            elif filename.startswith('global_hindcast_'):
                # Global format: global_hindcast_YYYYMMDD_HH.nc
                year_str = filename.replace('global_hindcast_', '')[:4]
            else:
                # Fallback: try to extract first 4 digits
                import re
                match = re.search(r'\d{4}', filename)
                if match:
                    year_str = match.group()
                else:
                    raise ValueError(f"Cannot extract year from filename: {filename}")
            return year_str
        
        if self.file_format == 'npy_daily':
            pass  # file_list already built from the daily index above
        elif split == 'test':
            # Keep only files from 2021 and 2022
            self.file_list = [f for f in all_files if extract_year_from_filename(f) in test_years]
        elif split == 'val':
            # Validation set: 2019 and 2020 (used for post-hoc calibration fits)
            self.file_list = [f for f in all_files if extract_year_from_filename(f) in val_years]
        elif split == 'train':
            # Train set logic depends on region
            if train_max_year is not None:
                # Optional cap from config (e.g. 2018 -> train on <=2018, keeping
                # 2019-2020 as val and 2021-2022 as test)
                self.file_list = [
                    f for f in all_files
                    if int(extract_year_from_filename(f)) <= train_max_year
                ]
            elif self.region == 'global':
                # For global data: train set is 1959-2020
                self.file_list = [
                    f for f in all_files
                    if 1959 <= int(extract_year_from_filename(f)) <= 2020
                ]
            else:
                # For HK data: keep files that are NOT from 2021 or 2022
                self.file_list = [f for f in all_files if extract_year_from_filename(f) not in test_years]
        else:
            raise ValueError(f"split must be 'train', 'val' or 'test', got '{split}'")
        
        if len(self.file_list) == 0:
            raise ValueError(f"No files found for {split} split in {data_dir}")
        
        # Filter file_list to only include samples with existing atmospheric data
        if self.load_era5:
            self.era5_dir = os.path.abspath(os.path.expanduser(self.era5_dir))
            if self.era5_file_format == 'npy_daily':
                idx_data = np.load(os.path.join(self._era5_daily_dir(), 'era5_daily_ts_index.npz'),
                                   allow_pickle=True)
                self._era5_daily_index = {str(ts): (str(d), int(s)) for ts, d, s in
                                          zip(idx_data['timestamps'], idx_data['days'], idx_data['slots'])}
                idx_data.close()
            self.file_list = self._filter_files_with_atm_data(self.file_list)
        
        if len(self.file_list) == 0:
            raise ValueError(f"No files with corresponding atmospheric data found for {split} split in {data_dir}")

        # Storm oversampling: duplicate high-intensity timestamps so the model
        # sees extreme sea states more often. Only applied to the train split.
        self._sample_indices = None
        self.oversample_alpha = float(oversample_alpha or 0.0)
        self.oversample_tau = float(oversample_tau or 3.0)
        self.oversample_weights_file = oversample_weights_file
        self._build_oversample_indices()

        # Per-worker ERA5 data cache to avoid redundant NFS reads.        # ERA5 files are tiny (~36 KB each) and many timestamps are reused
        # across different wave samples (same ERA5 timestamp for different lead times).
        # Since DataLoader workers are separate processes, each worker maintains its
        # own cache — no shared-memory issues.
        # Cache size: 10000 entries × ~36 KB ≈ 360 MB max per worker for HK region.
        self._era5_cache = {}
        self._era5_cache_max_size = 10000  # Covers typical HK training epoch working set

        # Pre-compute KD-tree interpolation weights once for node-interpolation runs.
        # Verified byte-identical (0.0 max diff) to the per-timestamp KD-tree slow path
        # for both ens/GenCast NC frames and ERA5 NPZ frames; makes per-frame
        # interpolation ~10-20x faster. try/except: fall back to the slow path if the
        # coordinate sources are unavailable at init (e.g. era5_dir not yet synced).
        if self.interpolate_era5_to_nodes:
            try:
                self._precompute_interpolation_weights()
            except Exception as e:
                import warnings
                warnings.warn(f"[WaveDataset] interpolation weight pre-compute failed "
                              f"({e!r}); falling back to per-timestamp KD-tree")

        # Mmap-based cache: single large file for O(1) random access to ALL timestamps.
        # Eliminates NFS random-read bottleneck of 464k individual NPZ files.
        # File: (num_unique_ts, 7788, 3) float32, ~43 GB.
        self._era5_mmap = None
        self._era5_ts_index = {}  # timestamp -> row index in mmap
        if self.use_mmap and self.use_preinterpolated_era5 and self.interpolate_era5_to_nodes:
            # Region-aware mmap cache: HK uses hk7788, global uses global prefix
            if self.region == 'global':
                mmap_basename = 'era5_global_all_frames.npy'
                idx_basename = 'era5_global_ts_index.npz'
            else:
                mmap_basename = 'era5_hk7788_all_frames.npy'
                idx_basename = 'era5_hk7788_ts_index.npz'
            mmap_path = os.path.join(os.path.dirname(self.era5_dir), mmap_basename)
            idx_path = os.path.join(os.path.dirname(self.era5_dir), idx_basename)
            if os.path.exists(mmap_path) and os.path.exists(idx_path):
                self._era5_mmap = np.load(mmap_path, mmap_mode='r')
                idx_data = np.load(idx_path, allow_pickle=True)
                self._era5_ts_index = {ts: int(i) for ts, i in
                    zip(idx_data['timestamps'], idx_data['indices'])}
                idx_data.close()
                print(f"Loaded ERA5 mmap cache: {self._era5_mmap.shape}, "
                      f"{len(self._era5_ts_index)} timestamps indexed")

        # Pre-compute ERA5 crop slice for NPZ data (matching the era5_region setting).
        # NPZ files contain the full 40×40 grid (region 1). For regions 2/3 we need to crop.
        self._era5_npz_slice = None  # (lat_slice, lon_slice) or None for no crop
        if self.load_era5 and self.era5_file_format == 'npz' and self.region == 'hk':
            from modules.utils import ERA5_HK_LON_RANGE_1, ERA5_HK_LAT_RANGE_1, \
                ERA5_HK_LON_RANGE_2, ERA5_HK_LAT_RANGE_2, \
                ERA5_HK_LON_RANGE_3, ERA5_HK_LAT_RANGE_3
            region_ranges = {
                1: (ERA5_HK_LON_RANGE_1, ERA5_HK_LAT_RANGE_1),
                2: (ERA5_HK_LON_RANGE_2, ERA5_HK_LAT_RANGE_2),
                3: (ERA5_HK_LON_RANGE_3, ERA5_HK_LAT_RANGE_3),
            }
            (min_lon, max_lon), (min_lat, max_lat) = region_ranges.get(self.era5_region, region_ranges[1])
            # NPZ grid: lon 110→119.75 step 0.25 (40 pts), lat 24.75→15 step 0.25 (40 pts, N→S)
            npz_lats = np.arange(24.75, 14.75, -0.25, dtype=np.float32)  # 40 points, N→S
            npz_lons = np.arange(110.0, 120.0, 0.25, dtype=np.float32)   # 40 points, W→E
            lat_mask = (npz_lats >= min_lat) & (npz_lats <= max_lat - 0.25)
            lon_mask = (npz_lons >= min_lon) & (npz_lons <= max_lon - 0.25)
            if lat_mask.sum() < 40 or lon_mask.sum() < 40:
                lat_slice = slice(np.where(lat_mask)[0][0], np.where(lat_mask)[0][-1] + 1)
                lon_slice = slice(np.where(lon_mask)[0][0], np.where(lon_mask)[0][-1] + 1)
                self._era5_npz_slice = (lat_slice, lon_slice)
                cropped_shape = (lat_mask.sum(), lon_mask.sum())
                print(f"ERA5 NPZ crop region {self.era5_region}: 40×40 → {cropped_shape[0]}×{cropped_shape[1]}")
            else:
                print(f"ERA5 NPZ region {self.era5_region}: full 40×40 (no crop needed)")

        # Wave data mmap cache: O(1) random access, eliminates NFS random-read bottleneck
        self._wave_mmap = None
        self._wave_ts_index = {}
        if self.use_mmap and self.region == 'global' and self.file_format == 'npz' and self.split == 'train':
            mmap_path = os.path.join(os.path.dirname(self.data_dir),
                                     'global_wave_all_frames.npy')
            idx_path = os.path.join(os.path.dirname(self.data_dir),
                                    'wave_mmap_ts_index.npz')
            if os.path.exists(mmap_path) and os.path.exists(idx_path):
                self._wave_mmap = np.load(mmap_path, mmap_mode='r')
                idx_data = np.load(idx_path, allow_pickle=True)
                self._wave_ts_index = {ts: int(i) for ts, i in
                    zip(idx_data['timestamps'], idx_data['indices'])}
                idx_data.close()
                print(f"Loaded wave mmap cache: {self._wave_mmap.shape}, "
                      f"{len(self._wave_ts_index)} timestamps indexed")
                # Filter file_list to only include timestamps present in the mmap.
                # Corrupted NPZ files on NFS may still appear in the glob but are
                # missing from the mmap; this filter ensures consistency.
                before = len(self.file_list)
                self.file_list = [
                    f for f in self.file_list
                    if self._extract_ts(f) in self._wave_ts_index
                ]
                dropped = before - len(self.file_list)
                if dropped > 0:
                    print(f"  Dropped {dropped} files not found in wave mmap "
                          f"({len(self.file_list)} remaining)")

    def _filter_files_with_atm_data(self, file_list: List[str]) -> List[str]:
        """
        Filter file list to only include files that have corresponding atmospheric data.
        
        This ensures the training/testing set only contains samples where both
        wave data and atmospheric (ERA5 or ENS) data exist.
        
        Args:
            file_list (List[str]): List of wave data file paths.
            
        Returns:
            List[str]: Filtered list containing only files with existing atmospheric data.
        """
        filtered_files = []
        missing_count = 0
        
        # Determine which check to use based on configuration
        check_preinterpolated = self.use_preinterpolated_era5 and self.interpolate_era5_to_nodes

        # Fast path for direct_concat mode: when the node-level ERA5 mmap cache
        # (era5_hk7788_all_frames.npy / era5_global_all_frames.npy) is present,
        # check its timestamp index instead of stat-ing per-timestamp preinterp
        # files — the per-file dir may be absent after storage cleanups while
        # the mmap is the authoritative cache actually used at load time.
        preinterp_ts_set = None
        if check_preinterpolated:
            era5_base = os.path.dirname(os.path.abspath(os.path.expanduser(self.era5_dir)))
            if self.region == 'global':
                idx_basename = 'era5_global_ts_index.npz'
            else:
                idx_basename = 'era5_hk7788_ts_index.npz'
            idx_path = os.path.join(era5_base, idx_basename)
            if os.path.exists(idx_path):
                idx_data = np.load(idx_path, allow_pickle=True)
                preinterp_ts_set = set(str(t) for t in idx_data['timestamps'])
                idx_data.close()
                print(f"Preinterp availability check: using mmap ts index "
                      f"({len(preinterp_ts_set)} timestamps) from {idx_path}")
        
        _init_note = (f"init={self.ens_init}" if self.ens_init
                      else f"lead_time={self.ens_lead_time}h")
        if self.atm_source == 'ens':
            desc = f"Checking IFS ENS data availability ({_init_note})"
        elif self.atm_source == 'gencast':
            desc = f"Checking GenCast data availability ({_init_note})"
        elif check_preinterpolated:
            desc = "Checking pre-interpolated ERA5 data availability"
        else:
            desc = "Checking ERA5 data availability"
        
        for file_path in tqdm(file_list, desc=desc):
            # Extract timestamp from filename
            filename = os.path.basename(file_path)
            if self.file_format == 'npy_daily':
                # Daily mode: file_list entries are already timestamps (YYYYMMDDTHH)
                timestamp = filename
            elif filename.startswith('wave_'):
                # HK format: wave_YYYYMMDDTHH.nc or wave_YYYYMMDDTHH.npz
                timestamp = filename.replace('wave_', '').replace('.nc', '').replace('.npz', '').replace('.npy', '')
            elif filename.startswith('global_hindcast_'):
                # Global format: global_hindcast_YYYYMMDD_HH.nc or global_hindcast_YYYYMMDD_HH.npz
                parts = filename.replace('global_hindcast_', '').replace('.nc', '').replace('.npz', '').replace('.npy', '').split('_')
                if len(parts) == 2:
                    timestamp = f"{parts[0]}T{parts[1]}"
                else:
                    continue
            else:
                continue
            
            # Check if all required atmospheric data frames exist
            all_frames_exist = True
            
            if self.atm_source == 'ens':
                # ENS: use frame plan to check all required files
                # (per-member forcing: every member file must exist)
                for m in range(max(1, self.atm_members)):
                    frame_paths = self._build_ens_frame_plan(timestamp, m if self.atm_members > 1 else None)
                    for path in frame_paths:
                        if not os.path.exists(path):
                            all_frames_exist = False
                            missing_count += 1
                            break
                    if not all_frames_exist:
                        break
            elif self.atm_source == 'gencast':
                # GenCast: same frame plan logic, different file prefix
                for m in range(max(1, self.atm_members)):
                    frame_paths = self._build_gencast_frame_plan(timestamp, m if self.atm_members > 1 else None)
                    for path in frame_paths:
                        if not os.path.exists(path):
                            all_frames_exist = False
                            missing_count += 1
                            break
                    if not all_frames_exist:
                        break
            else:
                # ERA5: check historical frames (6h intervals backwards)
                dt_base = datetime.strptime(timestamp, "%Y%m%dT%H")
                for i in range(self.num_era5_frames):
                    dt_offset = dt_base - timedelta(hours=i * 6)
                    offset_time_str = dt_offset.strftime("%Y%m%dT%H")
                    
                    if self.era5_file_format == 'npy_daily':
                        # Index membership instead of per-file stat (fast, and
                        # correctly excludes NaN-filled missing slots)
                        frame_exists = offset_time_str in self._era5_daily_index
                    elif check_preinterpolated:
                        if preinterp_ts_set is not None:
                            frame_exists = offset_time_str in preinterp_ts_set
                        else:
                            frame_exists = os.path.exists(self._get_preinterpolated_path(offset_time_str))
                    else:
                        frame_exists = os.path.exists(self._get_era5_path_for_time(offset_time_str))
                    
                    if not frame_exists:
                        all_frames_exist = False
                        missing_count += 1
                        break
            
            if all_frames_exist:
                filtered_files.append(file_path)
        
        if self.atm_source == 'ens':
            source_name = "IFS ENS"
        elif self.atm_source == 'gencast':
            source_name = "GenCast"
        elif check_preinterpolated:
            source_name = "pre-interpolated ERA5"
        else:
            source_name = "ERA5"
        print(f"Filtered dataset: {len(filtered_files)}/{len(file_list)} samples have complete {source_name} data "
              f"({missing_count} missing)")
        
        return filtered_files

    def _build_oversample_indices(self):
        """
        Build an extended sample-index list that duplicates storm timestamps.

        Weight per train timestamp:
            w = 1 + alpha * clip((region_max_hs - tau) / (max_hs_ref - tau), 0, None)
        where max_hs_ref is the dataset-wide maximum region-max hs. Each file is
        repeated round(w) times (>= 1). Deterministic (no RNG), so all DDP
        ranks / workers build identical indices.
        """
        if self.split != 'train' or self.oversample_alpha <= 0:
            return
        if not self.oversample_weights_file or not os.path.exists(self.oversample_weights_file):
            print(f"[WaveDataset] oversample enabled but weights file not found: "
                  f"{self.oversample_weights_file} — skipping oversampling")
            return

        wdata = np.load(self.oversample_weights_file)
        max_by_ts = dict(zip(wdata['timestamps'].astype(str), wdata['max_hs'].astype(np.float64)))
        ref_max = float(np.nanmax(wdata['max_hs'])) if wdata['max_hs'].size else 12.0

        import re as _re
        weights = np.ones(len(self.file_list), dtype=np.float64)
        for i, path in enumerate(self.file_list):
            m = _re.search(r'(\d{8}T\d{2})', os.path.basename(path))
            if not m:
                continue
            mx = max_by_ts.get(m.group(1), float('nan'))
            if np.isnan(mx):
                continue
            weights[i] = 1.0 + self.oversample_alpha * np.clip(
                (mx - self.oversample_tau) / (ref_max - self.oversample_tau), 0.0, None)

        repeats = np.rint(weights).astype(np.int64)
        repeats = np.maximum(repeats, 1)
        self._sample_indices = np.repeat(np.arange(len(self.file_list)), repeats)
        frac_storm = (weights > 1).sum() / len(weights)
        n_dup = int((repeats - 1).sum())
        print(f"[WaveDataset] oversampling: {len(self.file_list)} -> "
              f"{len(self._sample_indices)} samples (+{n_dup} storm dups, "
              f"{frac_storm:.2%} of timestamps weighted, tau={self.oversample_tau}m, "
              f"alpha={self.oversample_alpha})")

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self._sample_indices) if self._sample_indices is not None else len(self.file_list)

    def _atm_member_filename(self, prefix: str, init_str: str,
                             member: int = None) -> str:
        """
        Forecast forcing filename for a given member index.

        member=None keeps the legacy single-forcing name (ens-mean / fc0 layout);
        member=k (0-based) selects the per-member file `{prefix}_m{kk}_{init}.nc`
        as exported by build_gencast_members.py / build_ens_members.py.
        """
        if member is None:
            return f"{prefix}_{init_str}.nc"
        return f"{prefix}_m{member:02d}_{init_str}.nc"

    def _build_ens_frame_plan(self, base_timestamp: str, member: int = None) -> list:
        """
        Build a plan of ENS files to load for multi-frame input.
        
        For each frame (oldest first, 6h apart), determines the most recent
        ENS init_time (00Z or 12Z) and the corresponding lead_time.
        
        Example: ens_lead_time=24h, num_frames=8, valid_time=Jan02 00Z
          Frame 1: valid=Dec31 06Z → init=Dec31 00Z, lead_time=6h
          Frame 2: valid=Dec31 12Z → init=Dec31 12Z, lead_time=0h
          Frame 3: valid=Dec31 18Z → init=Dec31 12Z, lead_time=6h
          Frame 4: valid=Jan01 00Z → init=Jan01 00Z, lead_time=0h
          Frame 5: valid=Jan01 06Z → init=Jan01 00Z, lead_time=6h
          Frame 6: valid=Jan01 12Z → init=Jan01 00Z, lead_time=12h
          Frame 7: valid=Jan01 18Z → init=Jan01 00Z, lead_time=18h
          Frame 8: valid=Jan02 00Z → init=Jan01 00Z, lead_time=24h
        
        Args:
            base_timestamp (str): Target valid time in format YYYYMMDDTHH.
            
        Returns:
            list of str: File paths for each frame, oldest first.
        """
        dt_base = datetime.strptime(base_timestamp, "%Y%m%dT%H")
        member = member if member is not None else self.atm_default_member
        # The init_time for the main forecast chain: a fixed issue time when
        # ens_init is set (one cycle, varying lead), else valid - ens_lead_time.
        dt_main_init = (self._ens_init_dt if self._ens_init_dt is not None
                        else dt_base - timedelta(hours=self.ens_lead_time))

        frame_paths = []
        for i in range(self.num_era5_frames - 1, -1, -1):
            dt_valid_frame = dt_base - timedelta(hours=i * 6)

            if dt_valid_frame >= dt_main_init:
                # Within the main forecast chain
                dt_init = dt_main_init
            else:
                # Pad region: find the most recent init_time (00Z or 12Z) <= dt_valid_frame
                hour = dt_valid_frame.hour
                if hour >= 12:
                    dt_init = dt_valid_frame.replace(hour=12, minute=0, second=0)
                else:
                    dt_init = dt_valid_frame.replace(hour=0, minute=0, second=0)

            lead_hours = int((dt_valid_frame - dt_init).total_seconds() // 3600)
            init_str = dt_init.strftime("%Y%m%dT%H")
            init_year = dt_init.strftime("%Y")

            path = os.path.join(
                self.ens_dir,
                f"lead_time_{lead_hours}h",
                init_year,
                self._atm_member_filename("ifs_ens", init_str, member)
            )
            frame_paths.append(path)

        return frame_paths

    def _get_era5_path_for_time(self, time_str: str) -> str:
        """
        Get the ERA5 file path for a specific timestamp.
        
        Args:
            time_str (str): Timestamp string in format YYYYMMDDTHH.
            
        Returns:
            str: Path to corresponding ERA5 file
        """
        # Extract year from time string
        year = time_str[:4]
        
        # Construct ERA5 file path based on era5_file_format
        # {era5_dir}/{year}/era5_{time}.nc or {era5_dir}/{year}/era5_{time}.npz
        extension = 'npz' if self.era5_file_format == 'npz' else 'nc'
        era5_path = os.path.join(self.era5_dir, year, f'era5_{time_str}.{extension}')
        
        return era5_path

    def _get_ens_path_for_time(self, valid_time_str: str, lead_time_hours: int,
                               member: int = None) -> str:
        """
        Get the IFS ENS forecast file path for a given valid time and lead time.

        Computes init_time = valid_time - lead_time, then constructs the path:
        {ens_dir}/lead_time_{lead_time}h/{init_year}/ifs_ens[_m{kk}]_{init_time}.nc

        Args:
            valid_time_str (str): Valid time string in format YYYYMMDDTHH.
            lead_time_hours (int): Forecast lead time in hours.
            member (int, optional): Forcing member index (see _atm_member_filename).

        Returns:
            str: Path to corresponding ENS file.
        """
        dt_valid = datetime.strptime(valid_time_str, "%Y%m%dT%H")
        dt_init = dt_valid - timedelta(hours=lead_time_hours)
        init_time_str = dt_init.strftime("%Y%m%dT%H")
        init_year = dt_init.strftime("%Y")
        member = member if member is not None else self.atm_default_member

        ens_path = os.path.join(
            self.ens_dir,
            f"lead_time_{lead_time_hours}h",
            init_year,
            self._atm_member_filename("ifs_ens", init_time_str, member)
        )
        return ens_path

    def _get_gencast_path_for_time(self, valid_time_str: str, lead_time_hours: int,
                                   member: int = None) -> str:
        """
        Get the GenCast forecast file path for a given valid time and lead time.

        Mirrors _get_ens_path_for_time but uses gencast_ prefix and gencast_dir.

        Args:
            valid_time_str (str): Valid time string in format YYYYMMDDTHH.
            lead_time_hours (int): Forecast lead time in hours.
            member (int, optional): Forcing member index (see _atm_member_filename).

        Returns:
            str: Path to corresponding GenCast file.
        """
        dt_valid = datetime.strptime(valid_time_str, "%Y%m%dT%H")
        dt_init = dt_valid - timedelta(hours=lead_time_hours)
        init_time_str = dt_init.strftime("%Y%m%dT%H")
        init_year = dt_init.strftime("%Y")
        member = member if member is not None else self.atm_default_member

        gencast_path = os.path.join(
            self.gencast_dir,
            f"lead_time_{lead_time_hours}h",
            init_year,
            self._atm_member_filename("gencast", init_time_str, member)
        )
        return gencast_path

    def _build_gencast_frame_plan(self, base_timestamp: str, member: int = None) -> list:
        """
        Build a plan of GenCast files to load for multi-frame input.

        Same logic as _build_ens_frame_plan but uses GenCast file naming.

        Args:
            base_timestamp (str): Target valid time in format YYYYMMDDTHH.
            member (int, optional): Forcing member index (see _atm_member_filename).

        Returns:
            list of str: File paths for each frame, oldest first.
        """
        dt_base = datetime.strptime(base_timestamp, "%Y%m%dT%H")
        member = member if member is not None else self.atm_default_member
        dt_main_init = (self._ens_init_dt if self._ens_init_dt is not None
                        else dt_base - timedelta(hours=self.ens_lead_time))

        frame_paths = []
        for i in range(self.num_era5_frames - 1, -1, -1):
            dt_valid_frame = dt_base - timedelta(hours=i * 6)

            if dt_valid_frame >= dt_main_init:
                dt_init = dt_main_init
            else:
                hour = dt_valid_frame.hour
                if hour >= 12:
                    dt_init = dt_valid_frame.replace(hour=12, minute=0, second=0)
                else:
                    dt_init = dt_valid_frame.replace(hour=0, minute=0, second=0)

            lead_hours = int((dt_valid_frame - dt_init).total_seconds() // 3600)
            init_str = dt_init.strftime("%Y%m%dT%H")
            init_year = dt_init.strftime("%Y")

            path = os.path.join(
                self.gencast_dir,
                f"lead_time_{lead_hours}h",
                init_year,
                self._atm_member_filename("gencast", init_str, member)
            )
            frame_paths.append(path)

        return frame_paths
    
    def _load_era5_npz_data(self, era5_path: str) -> np.ndarray:
        """
        Load and process ERA5 data from an NPZ file.
        
        Args:
            era5_path (str): Path to the ERA5 NPZ file.
            
        Returns:
            np.ndarray: ERA5 data with shape [3, H, W] containing u_wind, v_wind, and pressure.
        """
        # Load NPZ file
        data = np.load(era5_path, allow_pickle=True)
        
        # Extract wind components and pressure using ERA5_VARIABLE_NAMES
        u_wind = data[ERA5_VARIABLE_NAMES[0]].astype(np.float32)  # [H, W] grid data
        v_wind = data[ERA5_VARIABLE_NAMES[1]].astype(np.float32)  # [H, W] grid data
        pressure = data[ERA5_VARIABLE_NAMES[2]].astype(np.float32)  # [H, W] grid data

        data.close()

        # Stack into [3, H, W] tensor (ERA5 is grid-based)
        era5_data = np.stack([u_wind, v_wind, pressure], axis=0)  # [3, H, W]

        # Crop to the configured ERA5 region (e.g., region 2: 40×40 → 24×24)
        if self._era5_npz_slice is not None:
            lat_slice, lon_slice = self._era5_npz_slice
            era5_data = era5_data[:, lat_slice, lon_slice]

        # For global data, crop latitude from 721 to 720 by removing the southernmost row
        # This ensures the height dimension is divisible by patch_size (30)
        if self.region == 'global' and era5_data.shape[1] == 721:
            era5_data = era5_data[:, :-1, :]  # Remove the last row (southernmost, lat=-90)

        # Reorder data to match sorted longitude coordinates if needed
        if hasattr(self, '_era5_lon_sort_idx'):
            era5_data = era5_data[:, :, self._era5_lon_sort_idx]

        return era5_data

    def _load_era5_nc_data(self, era5_path: str) -> np.ndarray:
        """
        Load and process ERA5 data from a NetCDF file.
        
        Args:
            era5_path (str): Path to the ERA5 NetCDF file.
            
        Returns:
            np.ndarray: ERA5 data with shape [3, H, W] containing u_wind, v_wind, and pressure.
        """
        ds_era5 = xr.load_dataset(era5_path, engine='netcdf4')
        
        # Select ERA5 region based on region and preset
        if self.region == 'global':
            # For global data, use full global region
            lon_range = ERA5_GLOBAL_LON_RANGE
            lat_range = ERA5_GLOBAL_LAT_RANGE
        elif self.era5_region == 1:
            lon_range = ERA5_HK_LON_RANGE_1
            lat_range = ERA5_HK_LAT_RANGE_1
        elif self.era5_region == 2:
            lon_range = ERA5_HK_LON_RANGE_2
            lat_range = ERA5_HK_LAT_RANGE_2
        elif self.era5_region == 3:
            lon_range = ERA5_HK_LON_RANGE_3
            lat_range = ERA5_HK_LAT_RANGE_3
        else:
            lon_range = ERA5_HK_LON_RANGE_1
            lat_range = ERA5_HK_LAT_RANGE_1
            
        min_lon, max_lon = lon_range
        min_lat, max_lat = lat_range
        
        # Filter ERA5 data (left closed, right open)
        if 'longitude' in ds_era5:
            lon_name = 'longitude'
            lat_name = 'latitude'
        else:
            lon_name = 'lon'
            lat_name = 'lat'
        
        # Convert longitude from 0-360 to -180-180 if needed
        # ERA5 global data uses 0-360 range, but utils.py defines ERA5_GLOBAL_LON_RANGE as (-180, 180)
        if self.region == 'global' and ds_era5[lon_name].max() > 180:
            # Convert longitude coordinates: 180-360 becomes -180-0
            ds_era5[lon_name] = ((ds_era5[lon_name] + 180) % 360) - 180
            # Sort by longitude to maintain consistent order (-180 to 180)
            ds_era5 = ds_era5.sortby(lon_name)
            
        ds_era5 = ds_era5.where(
            (ds_era5[lon_name] >= min_lon) & (ds_era5[lon_name] < max_lon) & 
            (ds_era5[lat_name] >= min_lat) & (ds_era5[lat_name] < max_lat), 
            drop=True
        )
        
        # Ensure latitude is in descending order (ERA5 convention: north to south)
        # ENS data has ascending latitude which would cause a vertical flip
        if ds_era5[lat_name].values[0] < ds_era5[lat_name].values[-1]:
            ds_era5 = ds_era5.sortby(lat_name, ascending=False)

        # GenCast 导出的预测帧带 batch=1 维（lead0 分析场文件没有），squeeze
        # 掉单例非空间维，保证多帧 stack 形状一致（2026-10-19 gc 采样修复；
        # ENS/ERA5 树无此维，no-op）
        for _ax in ('batch', 'sample', 'level'):
            if _ax in ds_era5.dims and ds_era5.sizes[_ax] == 1:
                ds_era5 = ds_era5.isel({_ax: 0})

        # Extract wind components and pressure (using ERA5_VARIABLE_NAMES)
        u_wind = ds_era5[ERA5_VARIABLE_NAMES[0]].values.astype(np.float32)  # [H, W] grid data
        v_wind = ds_era5[ERA5_VARIABLE_NAMES[1]].values.astype(np.float32)  # [H, W] grid data
        pressure = ds_era5[ERA5_VARIABLE_NAMES[2]].values.astype(np.float32)  # [H, W] grid data
        
        # Stack into [3, H, W] tensor (ERA5 is grid-based)
        era5_data = np.stack([u_wind, v_wind, pressure], axis=0)  # [3, H, W]
        
        # For global data, crop latitude from 721 to 720 by removing the southernmost row
        # This ensures the height dimension is divisible by patch_size (30)
        if self.region == 'global' and era5_data.shape[1] == 721:
            era5_data = era5_data[:, :-1, :]  # Remove the last row (southernmost, lat=-90)
        
        return era5_data

    def _load_era5_data(self, era5_path: str) -> np.ndarray:
        """
        Load and process ERA5 data from the given path.
        
        Automatically selects the appropriate loader based on era5_file_format.
        
        Args:
            era5_path (str): Path to the ERA5 file.
            
        Returns:
            np.ndarray: ERA5 data with shape [3, H, W] containing u_wind, v_wind, and pressure.
        """
        if self.era5_file_format == 'npz':
            return self._load_era5_npz_data(era5_path)
        else:
            return self._load_era5_nc_data(era5_path)

    def _get_preinterpolated_path(self, time_str: str) -> str:
        """
        Get the file path for pre-interpolated ERA5 data.
        
        The pre-interpolated files are stored in a separate directory alongside
        the original ERA5 data directory, with year-based subdirectories.
        Each file contains interpolated data for a single timestamp.
        
        Args:
            time_str (str): Timestamp string in format YYYYMMDDTHH.
            
        Returns:
            str: Path to the pre-interpolated ERA5 file.
        """
        # Extract year from time string
        year = time_str[:4]
        
        # Create pre-interpolated data directory alongside the era5_dir
        # e.g., if era5_dir is './data/era5/global_npz_files',
        # pre-interpolated files go to './data/era5/global_npz_files_preinterpolated/{year}'
        # era5_file_format='npy':   './data/era5/global_npy_files_preinterpolated/{year}'
        # era5_file_format='npy16': './data/era5/global_npy_files_preinterpolated_f16/{year}'
        #                           (float16; msl stored as offset msl-1e5 to avoid f16 overflow)
        if self.era5_file_format == 'npy16':
            preinterp_base = f"{self.era5_dir.replace('npz_files', 'npy_files')}_preinterpolated_f16"
        elif self.era5_file_format == 'npy':
            preinterp_base = f"{self.era5_dir.replace('npz_files', 'npy_files')}_preinterpolated"
        else:
            preinterp_base = f"{self.era5_dir}_preinterpolated"
        preinterp_dir = os.path.join(preinterp_base, year)
        
        # Construct filename (timestamp only, region is implied by directory)
        extension = 'npy' if self.era5_file_format in ('npy', 'npy16') else 'npz'
        filename = f'era5_preinterp_{time_str}.{extension}'
        
        return os.path.join(preinterp_dir, filename)
    
    def _save_preinterpolated_era5(self, time_str: str, era5_on_nodes: np.ndarray) -> str:
        """
        Save pre-interpolated ERA5 data to file.
        
        Args:
            time_str (str): Timestamp string in format YYYYMMDDTHH.
            era5_on_nodes (np.ndarray): Interpolated ERA5 data with shape [N, 3].
            
        Returns:
            str: Path to the saved file.
        """
        save_path = self._get_preinterpolated_path(time_str)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save data with metadata (timestamp and num_nodes only)
        np.savez_compressed(
            save_path,
            era5_on_nodes=era5_on_nodes.astype(np.float32),
            timestamp=time_str,
            num_nodes=self.num_wave_nodes if hasattr(self, 'num_wave_nodes') else era5_on_nodes.shape[0]
        )
        
        return save_path
    
    def _load_preinterpolated_era5(self, time_str: str) -> Optional[np.ndarray]:
        """
        Load pre-interpolated ERA5 data from file if it exists.
        
        Args:
            time_str (str): Timestamp string in format YYYYMMDDTHH.
            
        Returns:
            Optional[np.ndarray]: Interpolated ERA5 data with shape [N, 3] if file exists,
                                  None otherwise.
        """
        if self.era5_file_format == 'npy_daily':
            # Per-day packed files: one pread() per frame, no open() in steady state
            entry = self._era5_daily_index.get(time_str)
            if entry is None:
                return None
            day, slot = entry
            day_path = os.path.join(self._era5_daily_dir(), day[:4], f'era5_preinterp_{day}.npy')
            try:
                return self._daily_slot(day_path, slot).astype(np.float32)
            except Exception as e:
                print(f"Warning: Failed to load daily ERA5 slot {time_str} from {day_path}: {e}")
                return None
        
        load_path = self._get_preinterpolated_path(time_str)
        
        if not os.path.exists(load_path):
            return None
        
        try:
            if self.era5_file_format == 'npy16':
                # Float16 NPY: [N, 3], msl stored as (msl - 1e5) offset (raw msl
                # overflows float16); restore offset here. ~50% less IO than f32.
                arr = np.load(load_path).astype(np.float32)
                arr[:, 2] += 100000.0
                return arr
            if self.era5_file_format == 'npy':
                # Uncompressed NPY: [N, 3] float32, no zip/deflate overhead
                return np.load(load_path).astype(np.float32)
            data = np.load(load_path, allow_pickle=True)
            era5_on_nodes = data['era5_on_nodes'].astype(np.float32)
            data.close()
            return era5_on_nodes
            
        except Exception as e:
            print(f"Warning: Failed to load pre-interpolated file {load_path}: {e}. Recomputing...")
            return None
    
    def _process_single_timestamp(self, timestamp: str, force_recompute: bool,
                                   delete_original: bool) -> Tuple[str, Optional[str]]:
        """
        Process a single ERA5 timestamp: interpolate and save.
        
        Args:
            timestamp (str): Timestamp string in format YYYYMMDDTHH.
            force_recompute (bool): Whether to recompute if file exists.
            delete_original (bool): Whether to delete original file after processing.
            
        Returns:
            Tuple[str, Optional[str]]: (status, result)
                - status: 'skipped', 'computed', 'error'
                - result: save_path if computed, None otherwise
        """
        # Check if pre-interpolated file already exists
        if not force_recompute:
            existing_data = self._load_preinterpolated_era5(timestamp)
            if existing_data is not None:
                return 'skipped', None
        
        try:
            # Load single ERA5 frame
            era5_path = self._get_era5_path_for_time(timestamp)
            
            if not os.path.exists(era5_path):
                return 'error', f"ERA5 file not found: {era5_path}"
            
            era5_data = self._load_era5_data(era5_path)  # [3, H, W]
            
            # Load coordinates (skip if pre-computed weights are available)
            if self._interp_indices is not None:
                # Fast path: coordinates already cached in _precompute_interpolation_weights
                era5_lats = self._cached_era5_lats
                era5_lons = self._cached_era5_lons
            elif self.era5_file_format == 'nc':
                ds_era5 = xr.load_dataset(era5_path, engine='netcdf4')
                if 'latitude' in ds_era5:
                    era5_lats = ds_era5['latitude'].values.astype(np.float32)
                    era5_lons = ds_era5['longitude'].values.astype(np.float32)
                else:
                    era5_lats = ds_era5['lat'].values.astype(np.float32)
                    era5_lons = ds_era5['lon'].values.astype(np.float32)
                ds_era5.close()
            else:
                coord_file = os.path.join(self.era5_dir, 'era5_coordinates.npz')
                if not os.path.exists(coord_file):
                    return 'error', f"ERA5 coordinates file not found: {coord_file}"
                coord_data = np.load(coord_file)
                era5_lats = coord_data['latitude'].astype(np.float32)
                era5_lons = coord_data['longitude'].astype(np.float32)
                coord_data.close()
                
                if self.region == 'global' and era5_lats.shape[0] == 721:
                    era5_lats = era5_lats[:-1]
            
            # Handle global data longitude conversion
            if self.region == 'global' and era5_lons.max() > 180:
                era5_lons = ((era5_lons + 180) % 360) - 180
                lon_sort_idx = np.argsort(era5_lons)
                era5_lons = era5_lons[lon_sort_idx]
                era5_data = era5_data[:, :, lon_sort_idx]
            
            # Interpolate single frame to wave nodes
            era5_on_nodes = self._interpolate_era5_to_wave_nodes(era5_data, era5_lats, era5_lons)
            
            # Save pre-interpolated data (single timestamp)
            save_path = self._save_preinterpolated_era5(timestamp, era5_on_nodes)
            
            # Delete original ERA5 file if requested
            if delete_original and os.path.exists(era5_path):
                try:
                    os.remove(era5_path)
                except Exception:
                    pass  # Ignore deletion errors
            
            return 'computed', save_path
            
        except Exception as e:
            return 'error', str(e)
    
    def precompute_era5_interpolation(self, num_workers: int = 1, force_recompute: bool = False,
                                      delete_original: bool = False) -> List[str]:
        """
        Pre-compute ERA5 interpolation for all unique ERA5 timestamps and save to files.
        
        Each ERA5 timestamp is interpolated and stored separately. When loading,
        multiple timestamps can be loaded and concatenated as needed.
        
        Args:
            num_workers (int): Number of parallel workers for processing. Defaults to 1.
            force_recompute (bool): If True, recompute even if pre-interpolated files exist.
                                   Defaults to False.
            delete_original (bool): If True, delete original ERA5 files after successful
                                   interpolation to save disk space. Defaults to False.
        
        Returns:
            List[str]: List of paths to saved pre-interpolated files.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing

        # Pre-compute KD-tree weights once for all timestamps (massive speedup)
        self._precompute_interpolation_weights()

        if not self.interpolate_era5_to_nodes:
            raise ValueError("interpolate_era5_to_nodes must be True to precompute interpolation. "
                           "Please initialize dataset with interpolate_era5_to_nodes=True and "
                           "provide coord_file_path.")
        
        if self.wave_coords is None:
            raise ValueError("wave_coords must be loaded to precompute interpolation. "
                           "Please provide coord_file_path when initializing the dataset.")
        
        # Collect all unique ERA5 timestamps needed
        all_era5_timestamps = set()
        for idx in range(len(self.file_list)):
            file_path = self.file_list[idx]
            filename = os.path.basename(file_path)
            
            # Extract base timestamp from wave file
            if filename.startswith('wave_'):
                timestamp = filename.replace('wave_', '').replace('.nc', '').replace('.npz', '')
            elif filename.startswith('global_hindcast_'):
                parts = filename.replace('global_hindcast_', '').replace('.nc', '').replace('.npz', '').split('_')
                if len(parts) == 2:
                    timestamp = f"{parts[0]}T{parts[1]}"
                else:
                    continue
            else:
                continue
            
            # Add all historical timestamps (6h intervals)
            dt_base = datetime.strptime(timestamp, "%Y%m%dT%H")
            for i in range(self.num_era5_frames):
                dt_offset = dt_base - timedelta(hours=i * 6)
                offset_time_str = dt_offset.strftime("%Y%m%dT%H")
                all_era5_timestamps.add(offset_time_str)
        
        # Filter out already existing files if not force_recompute
        timestamps_to_process = []
        skipped_count = 0
        for timestamp in sorted(all_era5_timestamps):
            if not force_recompute:
                existing_data = self._load_preinterpolated_era5(timestamp)
                if existing_data is not None:
                    skipped_count += 1
                    continue
            timestamps_to_process.append(timestamp)
        
        print(f"Pre-computing ERA5 interpolation...")
        print(f"  Region: {self.region}")
        print(f"  Number of unique ERA5 timestamps: {len(all_era5_timestamps)}")
        print(f"  Already exists (will skip): {skipped_count}")
        print(f"  To process: {len(timestamps_to_process)}")
        print(f"  Number of wave nodes: {self.num_wave_nodes}")
        print(f"  Force recompute: {force_recompute}")
        print(f"  Delete original: {delete_original}")
        print(f"  Num workers: {num_workers}")
        print()
        
        saved_files = []
        computed_count = 0
        error_count = 0
        
        if len(timestamps_to_process) == 0:
            print("All files already exist, nothing to process.")
            return saved_files
        
        # Process timestamps with multi-threading
        if num_workers > 1:
            # Use ProcessPoolExecutor for CPU-bound interpolation
            # Limit workers to available CPUs
            max_workers = min(num_workers, multiprocessing.cpu_count())
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_timestamp = {
                    executor.submit(self._process_single_timestamp, ts, force_recompute, delete_original): ts
                    for ts in timestamps_to_process
                }
                
                # Process completed tasks with progress bar
                with tqdm(total=len(timestamps_to_process), desc="Pre-computing ERA5 interpolation") as pbar:
                    for future in as_completed(future_to_timestamp):
                        timestamp = future_to_timestamp[future]
                        try:
                            status, result = future.result()
                            if status == 'computed':
                                saved_files.append(result)
                                computed_count += 1
                            elif status == 'error':
                                print(f"Error processing {timestamp}: {result}")
                                error_count += 1
                        except Exception as e:
                            print(f"Error processing {timestamp}: {e}")
                            error_count += 1
                        pbar.update(1)
        else:
            # Single-threaded processing
            for timestamp in tqdm(timestamps_to_process, desc="Pre-computing ERA5 interpolation"):
                status, result = self._process_single_timestamp(timestamp, force_recompute, delete_original)
                if status == 'computed':
                    saved_files.append(result)
                    computed_count += 1
                elif status == 'error':
                    print(f"Error processing {timestamp}: {result}")
                    error_count += 1
        
        print()
        print(f"Pre-computation complete!")
        print(f"  Computed: {computed_count}")
        print(f"  Skipped (already exists): {skipped_count}")
        print(f"  Errors: {error_count}")
        print(f"  Deleted original files: {delete_original}")
        
        return saved_files

    def _precompute_interpolation_weights(self):
        """
        Pre-compute KD-tree nearest neighbor indices and IDW weights for ERA5→wave-node interpolation.

        The ERA5 grid coordinates are static, so the KD-tree, nearest neighbor indices, and
        interpolation weights are identical for every timestamp. Pre-computing them once and
        reusing across all timestamps eliminates ~90% of the per-timestamp work.

        Caches self._interp_indices [N, k], self._interp_weights [N, k], self._interp_exact_mask [N].
        Also caches the lon_sort_idx for global data.
        """
        from scipy.spatial import cKDTree

        if self._interp_indices is not None:
            return  # Already computed

        # Load ERA5 coordinates (once)
        if self.era5_file_format == 'nc':
            # For NC we need a sample file to get coords; pick the first available
            sample_path = self._get_era5_path_for_time('19700101T00')
            if not os.path.exists(sample_path):
                raise FileNotFoundError(f"Cannot precompute weights: sample ERA5 file not found: {sample_path}")
            ds_era5 = xr.load_dataset(sample_path, engine='netcdf4')
            if 'latitude' in ds_era5:
                era5_lats = ds_era5['latitude'].values.astype(np.float32)
                era5_lons = ds_era5['longitude'].values.astype(np.float32)
            else:
                era5_lats = ds_era5['lat'].values.astype(np.float32)
                era5_lons = ds_era5['lon'].values.astype(np.float32)
            ds_era5.close()
        else:
            coord_file = os.path.join(self.era5_dir, 'era5_coordinates.npz')
            if not os.path.exists(coord_file):
                raise FileNotFoundError(f"ERA5 coordinates file not found: {coord_file}")
            coord_data = np.load(coord_file)
            era5_lats = coord_data['latitude'].astype(np.float32)
            era5_lons = coord_data['longitude'].astype(np.float32)
            coord_data.close()

        # Crop latitude for global data
        if self.region == 'global' and era5_lats.shape[0] == 721:
            era5_lats = era5_lats[:-1]

        # Handle global longitude conversion
        self._era5_lon_sort_idx = None
        if self.region == 'global' and era5_lons.max() > 180:
            era5_lons = ((era5_lons + 180) % 360) - 180
            lon_sort_idx = np.argsort(era5_lons)
            era5_lons = era5_lons[lon_sort_idx]
            self._era5_lon_sort_idx = lon_sort_idx

        # Cache the coordinates for use by _interpolate_era5_to_wave_nodes fast path
        self._cached_era5_lats = era5_lats
        self._cached_era5_lons = era5_lons

        # Build meshgrid and convert to 3D Cartesian
        era5_lat_grid, era5_lon_grid = np.meshgrid(era5_lats, era5_lons, indexing='ij')
        era5_lat_flat = era5_lat_grid.flatten()
        era5_lon_flat = era5_lon_grid.flatten()

        wave_lats = self.wave_coords[:, 0]
        wave_lons = self.wave_coords[:, 1]
        N = self.num_wave_nodes

        def latlon_to_cartesian(lats, lons):
            lats_rad = np.radians(lats)
            lons_rad = np.radians(lons)
            x = np.cos(lats_rad) * np.cos(lons_rad)
            y = np.cos(lats_rad) * np.sin(lons_rad)
            z = np.sin(lats_rad)
            return np.stack([x, y, z], axis=-1)

        era5_cartesian = latlon_to_cartesian(era5_lat_flat, era5_lon_flat)
        wave_cartesian = latlon_to_cartesian(wave_lats, wave_lons)

        k = 4
        tree = cKDTree(era5_cartesian)
        distances, indices = tree.query(wave_cartesian, k=k)

        min_distances = distances[:, 0]
        exact_match_mask = min_distances < 1e-10

        weights = 1.0 / (distances + 1e-10)
        weights_sum = weights.sum(axis=1, keepdims=True)
        normalized_weights = weights / weights_sum

        self._interp_indices = indices
        self._interp_weights = normalized_weights
        self._interp_exact_mask = exact_match_mask

    def _interpolate_era5_to_wave_nodes(self, era5_data: np.ndarray, era5_lats: np.ndarray, era5_lons: np.ndarray) -> np.ndarray:
        """
        Interpolate ERA5 grid data to wave node positions using spherical distance-weighted interpolation.

        If pre-computed KD-tree weights are available (via _precompute_interpolation_weights),
        uses a fast path that skips coordinate conversion and tree queries entirely.

        Args:
            era5_data: ERA5 data with shape [3, H, W] or [T, 3, H, W]
            era5_lats: Latitude coordinates of ERA5 grid [H,]
            era5_lons: Longitude coordinates of ERA5 grid [W,]

        Returns:
            np.ndarray: Interpolated ERA5 data at wave node positions with shape [N, 3] or [T, N, 3]
        """
        # --- Fast path: use pre-computed KD-tree weights ---
        if self._interp_indices is not None:
            # Defensive: squeeze any singleton dimensions (e.g., (3, 40, 1, 40) -> (3, 40, 40))
            era5_data = np.squeeze(era5_data)

            is_sequence = era5_data.ndim == 4
            if is_sequence:
                T, C, H, W = era5_data.shape
            else:
                C, H, W = era5_data.shape
                T = 1
                era5_data = era5_data[np.newaxis, ...]

            N = self.num_wave_nodes

            # Apply longitude reorder if needed (for global data)
            if hasattr(self, '_era5_lon_sort_idx') and self._era5_lon_sort_idx is not None:
                era5_data = era5_data[:, :, :, self._era5_lon_sort_idx]

            indices = self._interp_indices      # [N, k]
            normalized_weights = self._interp_weights  # [N, k]
            exact_match_mask = self._interp_exact_mask  # [N,]

            interpolated_frames = []
            for t in range(T):
                era5_flat = era5_data[t].reshape(C, -1)  # [C, H*W]
                interpolated_t = np.zeros((N, C), dtype=np.float32)
                for c in range(C):
                    nearest_values = era5_flat[c, indices]  # [N, k]
                    interpolated_values = (nearest_values * normalized_weights).sum(axis=1)
                    exact_values = era5_flat[c, indices[:, 0]]
                    interpolated_values[exact_match_mask] = exact_values[exact_match_mask]
                    interpolated_t[:, c] = interpolated_values
                interpolated_frames.append(interpolated_t)

            if is_sequence:
                result = np.stack(interpolated_frames, axis=0)
                result = result.transpose(1, 0, 2)
                result = result.reshape(N, T * C)
            else:
                result = interpolated_frames[0]
            return result.astype(np.float32)

        # --- Original slow path (fallback when weights not pre-computed) ---
        from scipy.spatial import cKDTree

        # Handle both single frame and sequence
        is_sequence = era5_data.ndim == 4

        if is_sequence:
            T, C, H, W = era5_data.shape
        else:
            C, H, W = era5_data.shape
            T = 1
            era5_data = era5_data[np.newaxis, ...]  # Add time dimension for uniform handling

        # Get wave node coordinates
        wave_lats = self.wave_coords[:, 0]  # [N,]
        wave_lons = self.wave_coords[:, 1]  # [N,]
        N = self.num_wave_nodes

        # Create ERA5 grid coordinate arrays
        era5_lat_grid, era5_lon_grid = np.meshgrid(era5_lats, era5_lons, indexing='ij')
        era5_lat_flat = era5_lat_grid.flatten()  # [H*W,]
        era5_lon_flat = era5_lon_grid.flatten()  # [H*W,]

        # Convert lat/lon to 3D Cartesian coordinates on unit sphere
        def latlon_to_cartesian(lats, lons):
            lats_rad = np.radians(lats)
            lons_rad = np.radians(lons)
            x = np.cos(lats_rad) * np.cos(lons_rad)
            y = np.cos(lats_rad) * np.sin(lons_rad)
            z = np.sin(lats_rad)
            return np.stack([x, y, z], axis=-1)

        era5_cartesian = latlon_to_cartesian(era5_lat_flat, era5_lon_flat)  # [H*W, 3]
        wave_cartesian = latlon_to_cartesian(wave_lats, wave_lons)  # [N, 3]

        k = 4
        tree = cKDTree(era5_cartesian)
        distances, indices = tree.query(wave_cartesian, k=k)  # distances: [N, k], indices: [N, k]

        min_distances = distances[:, 0]  # [N,]
        exact_match_mask = min_distances < 1e-10

        weights = 1.0 / (distances + 1e-10)  # [N, k]
        weights_sum = weights.sum(axis=1, keepdims=True)  # [N, 1]
        normalized_weights = weights / weights_sum  # [N, k]

        interpolated_frames = []

        for t in range(T):
            era5_flat = era5_data[t].reshape(C, -1)  # [C, H*W]
            interpolated_t = np.zeros((N, C), dtype=np.float32)
            for c in range(C):
                nearest_values = era5_flat[c, indices]  # [N, k]
                interpolated_values = (nearest_values * normalized_weights).sum(axis=1)  # [N,]
                exact_values = era5_flat[c, indices[:, 0]]
                interpolated_values[exact_match_mask] = exact_values[exact_match_mask]
                interpolated_t[:, c] = interpolated_values
            interpolated_frames.append(interpolated_t)

        # Stack results
        if is_sequence:
            # [T, N, C] -> [N, T*C]
            result = np.stack(interpolated_frames, axis=0)  # [T, N, C]
            result = result.transpose(1, 0, 2)  # [N, T, C]
            result = result.reshape(N, T * C)  # [N, T*C]
        else:
            # Single frame: [N, C]
            result = interpolated_frames[0]

        return result.astype(np.float32)

    def _load_npz_data(self, file_path: str) -> Tuple[np.ndarray, str]:
        """
        Load wave data from an NPZ file.
        
        Args:
            file_path (str): Path to the NPZ file.
            
        Returns:
            Tuple[np.ndarray, str]: Wave data array and timestamp string.
        """
        data = np.load(file_path, allow_pickle=True)
        
        # Extract timestamp from filename
        filename = os.path.basename(file_path)
        if filename.startswith('wave_'):
            # HK format: wave_YYYYMMDDTHH.npz
            timestamp = filename.replace('wave_', '').replace('.npz', '')
        elif filename.startswith('global_hindcast_'):
            # Global format: global_hindcast_YYYYMMDD_HH.npz
            parts = filename.replace('global_hindcast_', '').replace('.npz', '').split('_')
            if len(parts) == 2:
                timestamp = f"{parts[0]}T{parts[1]}"
            else:
                raise ValueError(f"Unexpected global hindcast filename format: {filename}")
        else:
            raise ValueError(f"Unknown filename format: {filename}. Expected 'wave_*.npz' or 'global_hindcast_*.npz'")
        
        # Extract wave variables
        wave_vars_list = []
        for var in self.variables:
            if var in data:
                wave_vars_list.append(data[var].astype(np.float32))
            else:
                raise KeyError(f"Variable '{var}' not found in dataset file {file_path}")
        
        # Stack variables: [N, C]
        wave_data = np.stack(wave_vars_list, axis=-1)
        
        return wave_data, timestamp
    
    # Column layout of converted NPY wave files (see modules/preprocess/convert_mmap_to_npy.py)
    NPY_WAVE_COLUMNS = ['hs', 'mwd', 'ssh']

    def _load_npy_data(self, file_path: str) -> Tuple[np.ndarray, str]:
        """
        Load wave data from an uncompressed NPY file.

        The file holds a single float32 array of shape [N, C] whose column
        layout is self._npy_columns: legacy (hs, mwd, ssh) by default, or the
        layout recorded in data_dir/wave_columns.json when present (e.g. the
        4-var hs/tp/dm/elev product from convert_npz4var_to_npy.py). Only the
        columns listed in self.variables are returned.

        Args:
            file_path (str): Path to the NPY file.

        Returns:
            Tuple[np.ndarray, str]: Wave data array [N, C] and timestamp string.
        """
        arr = np.load(file_path)  # [N, C] float32, uncompressed
        timestamp = self._extract_ts(file_path)
        col_idx = [self._npy_columns.index(v) for v in self.variables]
        wave_data = arr[:, col_idx].astype(np.float32)
        return wave_data, timestamp
    
    # ---- Daily-packed NPY support (one [8, N, C] f32 file per day) ----
    DAILY_SLOTS_PER_DAY = 8        # 3h spacing, slot = hour // 3
    # Legacy 3-var slot size; kept for reference — actual slot size is parsed
    # from each file's NPY header in _daily_fd (supports 4-var files).
    DAILY_SLOT_NBYTES = 651652 * 3 * 4

    def _wave_daily_dir(self) -> str:
        """Directory of per-day packed wave files (derived from data_dir)."""
        if 'npy_files' in self.data_dir:
            return self.data_dir.replace('npy_files', 'npy_files_daily')
        return self.data_dir.replace('npz_files', 'npy_files_daily')

    def _era5_daily_dir(self) -> str:
        """Directory of per-day packed ERA5 files (derived from era5_dir)."""
        return f"{self.era5_dir.replace('npz_files', 'npy_files')}_preinterpolated_daily"

    def _daily_fd(self, day_path: str) -> Tuple[int, int, int, int]:
        """
        Return (fd, data_offset, slot_nbytes, n_cols) for a per-day NPY file,
        using a per-worker LRU cache so steady-state sampling issues zero
        open() syscalls — only pread(). Shape is parsed from the NPY header
        ([8, N, C] float32), so both legacy 3-var and 4-var files work.
        """
        entry = self._daily_fd_cache.get(day_path)
        if entry is not None:
            self._daily_fd_cache.move_to_end(day_path)
            return entry
        fd = os.open(day_path, os.O_RDONLY)
        header = os.pread(fd, 10, 0)
        if header[:6] != b'\x93NUMPY':
            os.close(fd)
            raise ValueError(f"Not an NPY file: {day_path}")
        major = header[6]
        if major == 1:
            hlen = struct.unpack('<H', header[8:10])[0]  # v1.0 header length
            hoff = 10
        else:
            hlen = struct.unpack('<I', os.pread(fd, 4, 8))[0]  # v2.0+ uint32
            hoff = 12
        info = ast.literal_eval(os.pread(fd, hlen, hoff).decode('latin1'))
        if info['descr'] not in ('<f4', '|f4') or info['fortran_order']:
            os.close(fd)
            raise ValueError(f"Unsupported NPY layout in {day_path}: {info}")
        shape = info['shape']  # (slots, N, C)
        data_offset = hoff + hlen
        slot_nbytes = int(np.prod(shape[1:])) * 4
        n_cols = shape[-1]
        if len(self._daily_fd_cache) >= self._daily_fd_cache_max:
            _, (old_fd, *_) = self._daily_fd_cache.popitem(last=False)
            os.close(old_fd)
        self._daily_fd_cache[day_path] = (fd, data_offset, slot_nbytes, n_cols)
        return fd, data_offset, slot_nbytes, n_cols

    def _pread_daily_slot(self, day_path: str, slot: int) -> np.ndarray:
        """Read one [N, C] float32 slot from a per-day packed NPY file."""
        fd, data_offset, nbytes, n_cols = self._daily_fd(day_path)
        buf = os.pread(fd, nbytes, data_offset + slot * nbytes)
        if len(buf) != nbytes:
            raise IOError(f"Short read on {day_path} slot {slot}: got {len(buf)}/{nbytes} bytes")
        return np.frombuffer(buf, dtype=np.float32).reshape(-1, n_cols)

    def _daily_slot_cached(self, day_path: str, slot: int) -> np.ndarray:
        """
        Read one slot via a per-worker whole-day RAM cache: the first access to a
        day reads the entire day file (~62 MB), subsequent slots of the same day
        are served from RAM. Only effective when the sampler provides day-level
        locality (block-shuffled sampler with sequential days within a block);
        with fully random sampling it reads 8x the needed bytes for wave data.
        """
        day_arr = self._daily_ram_cache.get(day_path)
        if day_arr is None:
            day_arr = np.load(day_path)  # [8, N, 3] float32, one sequential read
            if len(self._daily_ram_cache) >= self._daily_ram_cache_max:
                self._daily_ram_cache.popitem(last=False)
            self._daily_ram_cache[day_path] = day_arr
        else:
            self._daily_ram_cache.move_to_end(day_path)
        return day_arr[slot]

    def _daily_slot(self, day_path: str, slot: int) -> np.ndarray:
        if self.daily_whole_day_cache:
            return self._daily_slot_cached(day_path, slot)
        return self._pread_daily_slot(day_path, slot)

    def _load_npy_daily_data(self, timestamp: str) -> Tuple[np.ndarray, str]:
        """
        Load wave data for one timestamp from a per-day packed NPY file.

        Args:
            timestamp (str): Timestamp in format YYYYMMDDTHH (file_list entry).

        Returns:
            Tuple[np.ndarray, str]: Wave data array [N, C] and timestamp string.
        """
        day, slot = self._wave_daily_index[timestamp]
        day_path = os.path.join(self._wave_daily_dir(), day[:4], f'global_hindcast_{day}.npy')
        arr = self._daily_slot(day_path, slot)  # [N, C]
        col_idx = [self._daily_wave_columns.index(v) for v in self.variables]
        wave_data = arr[:, col_idx].astype(np.float32)
        return wave_data, timestamp
    
    def _load_nc_data(self, file_path: str) -> Tuple[np.ndarray, str]:
        """
        Load wave data from a NetCDF file.
        
        Args:
            file_path (str): Path to the NetCDF file.
            
        Returns:
            Tuple[np.ndarray, str]: Wave data array and timestamp string.
        """
        ds = xr.load_dataset(file_path)
        
        # Extract timestamp from filename
        filename = os.path.basename(file_path)
        if filename.startswith('wave_'):
            # HK format: wave_YYYYMMDDTHH.nc
            timestamp = filename.replace('wave_', '').replace('.nc', '')
        elif filename.startswith('global_hindcast_'):
            # Global format: global_hindcast_YYYYMMDD_HH.nc
            parts = filename.replace('global_hindcast_', '').replace('.nc', '').split('_')
            if len(parts) == 2:
                timestamp = f"{parts[0]}T{parts[1]}"
            else:
                raise ValueError(f"Unexpected global hindcast filename format: {filename}")
        else:
            raise ValueError(f"Unknown filename format: {filename}. Expected 'wave_*.nc' or 'global_hindcast_*.nc'")
        
        # Filter data based on region
        if self.region == 'hk':
            min_lon, max_lon, min_lat, max_lat = HK_REGION
            ds_filtered = ds.where(
                (ds.lon >= min_lon) & (ds.lon <= max_lon) & 
                (ds.lat >= min_lat) & (ds.lat <= max_lat), 
                drop=True
            )
        elif self.region == 'global':
            # For global data, use full global region
            # Global data uses 'x' and 'y' for coordinates instead of 'lon' and 'lat'
            min_lon, max_lon, min_lat, max_lat = GLOBAL_REGION
            ds_filtered = ds.where(
                (ds.x >= min_lon) & (ds.x <= max_lon) & 
                (ds.y >= min_lat) & (ds.y <= max_lat), 
                drop=True
            )
        else:
            raise ValueError(f"Unknown region: {self.region}. Supported regions: 'hk', 'global'")
        
        # Extract all wave variables and stack them
        wave_vars_list = []
        for var in self.variables:
            if var in ds_filtered:
                wave_vars_list.append(ds_filtered[var].values.astype(np.float32))
            else:
                raise KeyError(f"Variable '{var}' not found in dataset file {file_path}")
        
        # Stack variables: [N, C]
        wave_data = np.stack(wave_vars_list, axis=-1)
        
        return wave_data, timestamp

    def _extract_timestamp_from_file(self, file_path: str) -> str:
        """
        Extract timestamp from wave data filename.
        
        Supports both HK format (wave_YYYYMMDDTHH.*) and global format (global_hindcast_YYYYMMDD_HH.*).
        
        Args:
            file_path (str): Path to the wave data file.
            
        Returns:
            str: Timestamp string in format YYYYMMDDTHH.
            
        Raises:
            ValueError: If filename format is not recognized.
        """
        filename = os.path.basename(file_path)
        
        if filename.startswith('wave_'):
            # HK format: wave_YYYYMMDDTHH.nc or wave_YYYYMMDDTHH.npz
            return filename.replace('wave_', '').replace('.nc', '').replace('.npz', '')
        elif filename.startswith('global_hindcast_'):
            # Global format: global_hindcast_YYYYMMDD_HH.nc or global_hindcast_YYYYMMDD_HH.npz
            parts = filename.replace('global_hindcast_', '').replace('.nc', '').replace('.npz', '').split('_')
            if len(parts) == 2:
                return f"{parts[0]}T{parts[1]}"
            else:
                raise ValueError(f"Unexpected global hindcast filename format: {filename}")
        else:
            raise ValueError(f"Unknown filename format: {filename}. Expected 'wave_*' or 'global_hindcast_*'")
    
    def _load_wave_data(self, file_path: str) -> Tuple[np.ndarray, str]:
        """
        Load wave data from file based on file_format.

        Args:
            file_path (str): Path to the wave data file.

        Returns:
            Tuple[np.ndarray, str]: Wave data array and timestamp string.
        """
        # Fast path: mmap cache (global wave data only)
        if self._wave_mmap is not None:
            ts = self._extract_ts(file_path)
            idx = self._wave_ts_index.get(ts)
            if idx is not None:
                return self._wave_mmap[idx].copy(), ts
            # Mmap loaded but timestamp not found — should not happen for global training data.
            # Raise clear error instead of falling back to potentially corrupted NPZ files.
            raise KeyError(
                f"Timestamp '{ts}' (file: {file_path}) not found in wave mmap cache. "
                f"Rebuild the mmap or check data consistency."
            )

        if self.file_format == 'npz':
            return self._load_npz_data(file_path)
        elif self.file_format == 'npy':
            return self._load_npy_data(file_path)
        elif self.file_format == 'npy_daily':
            return self._load_npy_daily_data(file_path)  # file_path is the timestamp
        else:
            return self._load_nc_data(file_path)

    @staticmethod
    def _extract_ts(file_path: str) -> str:
        """Extract timestamp string (YYYYMMDDTHH) from wave data filename."""
        fname = os.path.basename(file_path)
        if fname.startswith('wave_'):
            return fname.replace('wave_', '').replace('.npz', '').replace('.nc', '').replace('.npy', '')
        elif fname.startswith('global_hindcast_'):
            # global_hindcast_YYYYMMDD_HH.npz -> YYYYMMDDTHH
            parts = fname.replace('global_hindcast_', '').replace('.npz', '').replace('.nc', '').replace('.npy', '').split('_')
            if len(parts) == 2:
                return f"{parts[0]}T{parts[1]}"
        raise ValueError(f"Unknown filename format: {fname}")
    
    def _load_era5_coordinates(self, era5_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ERA5 coordinate arrays (latitude and longitude).
        
        Args:
            era5_path (str): Path to the ERA5 file (used for NC format).
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: ERA5 latitude and longitude arrays.
            
        Raises:
            FileNotFoundError: If coordinate file is not found for NPZ format.
        """
        if self.era5_file_format == 'nc':
            ds_era5 = xr.load_dataset(era5_path, engine='netcdf4')
            if 'latitude' in ds_era5:
                era5_lats = ds_era5['latitude'].values.astype(np.float32)
                era5_lons = ds_era5['longitude'].values.astype(np.float32)
            else:
                era5_lats = ds_era5['lat'].values.astype(np.float32)
                era5_lons = ds_era5['lon'].values.astype(np.float32)
            ds_era5.close()
            # Match _load_era5_nc_data's row order (descending latitude), so
            # that the global crop below drops the southernmost row in both.
            if era5_lats[0] < era5_lats[-1]:
                era5_lats = era5_lats[::-1]
        else:
            # For NPZ, load from separate coordinate file
            coord_file = os.path.join(self.era5_dir, 'era5_coordinates.npz')
            if not os.path.exists(coord_file):
                raise FileNotFoundError(
                    f"ERA5 coordinates file not found: {coord_file}. "
                    f"Required for interpolate_era5_to_nodes=True with NPZ format."
                )
            coord_data = np.load(coord_file)
            era5_lats = coord_data['latitude'].astype(np.float32)
            era5_lons = coord_data['longitude'].astype(np.float32)
            coord_data.close()
        
        # For global data, crop latitude from 721 to 720 to match cropped data
        if self.region == 'global' and era5_lats.shape[0] == 721:
            era5_lats = era5_lats[:-1]
        
        # Handle global data longitude conversion to [-180, 180] range
        if self.region == 'global' and era5_lons.max() > 180:
            era5_lons = ((era5_lons + 180) % 360) - 180
            lon_sort_idx = np.argsort(era5_lons)
            era5_lons = era5_lons[lon_sort_idx]
            # Store sort index for potential data reordering
            self._era5_lon_sort_idx = lon_sort_idx
        
        return era5_lats, era5_lons
    
    def _compute_era5_interpolation(self, timestamp: str, member: int = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute ERA5/ENS data loading (and optional interpolation) for a single timestamp.

        Args:
            timestamp (str): Base timestamp in format YYYYMMDDTHH.
            member (int, optional): Forcing member index (see _atm_member_filename).

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                - If interpolate_era5_to_nodes=True: interpolated data [N, 3] (single frame)
                - Otherwise: raw data [3, H, W] (single frame)
        """
        # Get file path based on atmospheric data source
        if self.atm_source == 'ens':
            atm_path = self._get_ens_path_for_time(timestamp, self.ens_lead_time, member)
        elif self.atm_source == 'gencast':
            atm_path = self._get_gencast_path_for_time(timestamp, self.ens_lead_time, member)
        else:
            atm_path = self._get_era5_path_for_time(timestamp)

        # Check per-worker memory cache first (avoids redundant NFS reads)
        # Cache stores raw grid data [3, H, W] keyed by file path.
        # If interpolation is needed, we still compute it from cached raw data.
        if atm_path in self._era5_cache:
            era5_data = self._era5_cache[atm_path].copy()
        else:
            try:
                # ENS and GenCast files are always NC format; ERA5 uses configured format
                if self.atm_source in ('ens', 'gencast'):
                    era5_data = self._load_era5_nc_data(atm_path)  # [3, H, W]
                else:
                    era5_data = self._load_era5_data(atm_path)  # [3, H, W]

                # Cache the raw grid data (FIFO eviction when full)
                if len(self._era5_cache) >= self._era5_cache_max_size:
                    self._era5_cache.pop(next(iter(self._era5_cache)))
                self._era5_cache[atm_path] = era5_data.copy()
            except Exception as e:
                raise FileNotFoundError(f"Atmospheric data file not found: {atm_path}") from e

        if self.interpolate_era5_to_nodes:
            return self._interpolate_era5_frame_to_nodes(era5_data, atm_path)
        else:
            return era5_data

    def _interpolate_era5_frame_to_nodes(self, era5_data: np.ndarray, atm_path: str) -> np.ndarray:
        """
        Interpolate one raw atmospheric grid [3, H, W] to wave nodes [N, 3].

        Shared by _compute_era5_interpolation (single frame) and _load_era5_sequence
        (ens/GenCast frame plans with interpolate_era5_to_nodes, where each frame
        must be mapped to nodes BEFORE the sequence concat).

        Args:
            era5_data (np.ndarray): Raw grid [3, H, W] already in NC-load order.
            atm_path (str): Path of the atmospheric file (for coordinate lookup).

        Returns:
            np.ndarray: Interpolated data [N, 3].
        """
        # Load coordinates and interpolate from (possibly cached) raw data
        era5_lats, era5_lons = self._load_era5_coordinates(atm_path)

        # _load_era5_nc_data already sorts global longitudes to ascending
        # [-180, 180); applying _era5_lon_sort_idx to such data again
        # (here or in the interpolation fast path) would roll the field
        # by 180°. Only NPZ-loaded data (file order) need the reorder.
        data_nc_sorted = (self.atm_source in ('ens', 'gencast')
                          or self.era5_file_format == 'nc')
        saved_idx = getattr(self, '_era5_lon_sort_idx', None)
        if data_nc_sorted:
            self._era5_lon_sort_idx = None
        elif saved_idx is not None:
            # Reorder data if longitude was sorted
            era5_data = era5_data[:, :, saved_idx]
        try:
            return self._interpolate_era5_to_wave_nodes(
                era5_data, era5_lats, era5_lons)
        finally:
            self._era5_lon_sort_idx = saved_idx
    
    def _load_era5_data_for_sample(self, timestamp: str, member: int = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Load ERA5 data for a single timestamp, using pre-interpolated data if available.

        Args:
            timestamp (str): Timestamp in format YYYYMMDDTHH.
            member (int, optional): Forcing member index (see _atm_member_filename).

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: ERA5 data for single timestamp,
                either interpolated to nodes [N, 3] or raw grid data [3, H, W].
        """
        # ENS/GenCast forecast forcing has no node-interpolated mmap/NPZ cache
        # (those caches are ERA5-only) — always load the forecast grid file and
        # interpolate on-the-fly. _compute_era5_interpolation resolves the path
        # from atm_source/ens_lead_time.
        if self.atm_source != 'era5':
            return self._compute_era5_interpolation(timestamp, member)

        # Try node-interpolated mmap cache (direct_concat mode: [N, 3] from memory-mapped file)
        if self._era5_mmap is not None:
            idx = self._era5_ts_index.get(timestamp)
            if idx is not None:
                return self._era5_mmap[idx].copy()
            # Mmap loaded but timestamp not found — skip NPZ fallback to avoid
            # hitting corrupted files. Raise clear error.
            raise KeyError(
                f"Timestamp '{timestamp}' not found in ERA5 mmap cache. "
                f"Rebuild the mmap or check data consistency."
            )

        # Try to load pre-interpolated data from per-timestamp NPZ
        if self.interpolate_era5_to_nodes:
            preinterp_data = self._load_preinterpolated_era5(timestamp)
            if preinterp_data is not None:
                return preinterp_data

        # Fall back to computing on-the-fly
        return self._compute_era5_interpolation(timestamp)
    
    def _load_era5_interp_frame(self, path: str) -> np.ndarray:
        """
        Load one raw atmospheric frame, interpolating to wave nodes if configured.

        Ens/GenCast frame plans load raw [3, H, W] NC grids; when
        interpolate_era5_to_nodes is set each frame must be mapped to [N, 3]
        BEFORE the sequence concat so the model receives [N, 3*T] — concatenating
        the raw grids would give [3*T, H, W] and break the reshape at the
        normalizer (see wave_normalize.reshape(N, T, 3)).

        Args:
            path (str): Path of the atmospheric file (NC format for ens/gencast).

        Returns:
            np.ndarray: [N, 3] if interpolate_era5_to_nodes else raw [3, H, W].
        """
        era5_data = self._load_era5_nc_data(path)
        if self.interpolate_era5_to_nodes:
            return self._interpolate_era5_frame_to_nodes(era5_data, path)
        return era5_data

    def _load_era5_sequence(self, base_timestamp: str, member: int = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Load a sequence of atmospheric data for multiple timestamps.

        For ERA5: loads num_era5_frames historical timestamps (6h intervals backwards).
        For ENS: loads num_era5_frames consecutive lead times from the same init time,
                 ending at ens_lead_time (e.g., lead_time 6h, 12h, 18h, 24h).

        Args:
            base_timestamp (str): Base timestamp (valid time) in format YYYYMMDDTHH.
            member (int, optional): Forcing member index; the same member is used
                for every frame of the sequence (see _atm_member_filename).

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                - If interpolate_era5_to_nodes=True: concatenated data [N, 3*T]
                - Otherwise: stacked data [T, 3, H, W]
        """
        dt_base = datetime.strptime(base_timestamp, "%Y%m%dT%H")
        era5_frames = []

        if self.atm_source == 'ens':
            # Use frame plan: each frame picks the best init_time/lead_time
            # to maintain 6h uniform spacing (see _build_ens_frame_plan)
            frame_paths = self._build_ens_frame_plan(base_timestamp, member)
            for path in frame_paths:
                era5_frames.append(self._load_era5_interp_frame(path))
        elif self.atm_source == 'gencast':
            # Same frame-planning logic, GenCast file naming
            frame_paths = self._build_gencast_frame_plan(base_timestamp, member)
            for path in frame_paths:
                era5_frames.append(self._load_era5_interp_frame(path))
        else:
            # ERA5 multi-frame: load frames in reverse order (oldest first)

            # Fast path: use node-interpolated mmap cache with vectorized gather (direct_concat mode)
            if self._era5_mmap is not None and self.interpolate_era5_to_nodes:
                frame_indices = []
                for i in range(self.num_era5_frames - 1, -1, -1):
                    dt_offset = dt_base - timedelta(hours=i * 6)
                    offset_time_str = dt_offset.strftime("%Y%m%dT%H")
                    idx = self._era5_ts_index.get(offset_time_str)
                    if idx is None:
                        # Fall back to per-frame loading for this frame
                        era5_frames.append(self._load_era5_data_for_sample(offset_time_str))
                    else:
                        frame_indices.append(idx)
                if frame_indices:
                    # Vectorized gather: [num_found, 7788, 3]
                    mmap_frames = self._era5_mmap[frame_indices]
                    # Insert at correct positions (oldest first)
                    mmap_idx = 0
                    full_frames = []
                    for i in range(self.num_era5_frames - 1, -1, -1):
                        dt_offset = dt_base - timedelta(hours=i * 6)
                        offset_time_str = dt_offset.strftime("%Y%m%dT%H")
                        if self._era5_ts_index.get(offset_time_str) is not None:
                            full_frames.append(mmap_frames[mmap_idx])
                            mmap_idx += 1
                        else:
                            full_frames.append(era5_frames[len(era5_frames) - (self.num_era5_frames - i)])
                    era5_frames = full_frames
            else:
                for i in range(self.num_era5_frames - 1, -1, -1):
                    dt_offset = dt_base - timedelta(hours=i * 6)
                    offset_time_str = dt_offset.strftime("%Y%m%dT%H")
                    era5_frame = self._load_era5_data_for_sample(offset_time_str)
                    era5_frames.append(era5_frame)

        if self.interpolate_era5_to_nodes:
            # Concatenate along feature dimension: [N, 3] * T -> [N, 3*T]
            return np.concatenate(era5_frames, axis=-1)
        else:
            # Stack along time dimension: [3, H, W] * T -> [T, 3, H, W]
            return np.stack(era5_frames, axis=0)
    
    def load_member_condition(self, timestamp: str, member_idx: int):
        """
        Fetch the atmospheric forcing condition of one ensemble member for a valid
        time (member-forced sampling: WaveFlow sample k is conditioned on member k).

        Returns the raw (un-normalized) array with the same shape/layout the
        DataLoader path produces for this dataset — [N, 3*T] when
        interpolate_era5_to_nodes=True, else [T, 3, H, W] ([3, H, W] for
        single-frame models). Callers normalize via self.transform, e.g.
        transform({'era5': cond})['era5'].

        Args:
            timestamp (str): Valid time in format YYYYMMDDTHH.
            member_idx (int): 0-based forcing member index.

        Returns:
            np.ndarray: Raw forcing condition for this member.
        """
        if self.atm_source == 'era5':
            raise ValueError("load_member_condition requires atm_source='ens'/'gencast'")
        if not (0 <= int(member_idx) < max(1, self.atm_members)):
            raise ValueError(f"member_idx {member_idx} out of range (atm_members={self.atm_members})")
        if self.num_era5_frames > 1:
            return self._load_era5_sequence(timestamp, int(member_idx))
        return self._load_era5_data_for_sample(timestamp, int(member_idx))

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            dict: Dictionary containing:
                - 'wave_data': shape [N, C] where C is number of variables.
                - 'era5' (if load_era5=True and interpolate_era5_to_nodes=False): 
                  shape [T, 3, H, W] containing [u_wind, v_wind, pressure].
                - 'era5_on_nodes' (if load_era5=True and interpolate_era5_to_nodes=True): 
                  shape [N, 3*T] containing interpolated ERA5 values at wave node positions.
                - 'timestamp': time string in format YYYYMMDDTHH (e.g., '20210101T00').
        """
        # Load wave data
        if self._sample_indices is not None:
            idx = int(self._sample_indices[idx])
            file_path = self.file_list[idx]
        else:
            file_path = self.file_list[idx]
        wave_data, timestamp = self._load_wave_data(file_path)
        
        # Extract time-of-day (hour normalized to [0, 1)) for tidal conditioning
        hour = int(timestamp[9:11])
        tod = np.float32(hour / 24.0)

        sample = {
            'wave_data': wave_data,
            'timestamp': timestamp,
            'tod': tod,
            # Absolute-time tidal features (sin/cos at constituent periods)
            'time_feats': compute_time_features(timestamp),
        }

        # Load ERA5 data if requested
        if self.load_era5:
            if self.num_era5_frames > 1:
                # Load multiple frames and concatenate/stack
                era5_result = self._load_era5_sequence(timestamp)
            else:
                # Load single frame
                era5_result = self._load_era5_data_for_sample(timestamp)
            
            if self.interpolate_era5_to_nodes:
                sample['era5_on_nodes'] = era5_result  # [N, 3*T]
            else:
                sample['era5'] = era5_result  # [T, 3, H, W] or [3, H, W]
        
        # Apply transforms if any
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def visualize_timestamps(self, timestamps, save_dir='./output/plot/data/global_hindcast/single_timestep_samples',
                              variables=None, visualize_era5=True, cmap='turbo'):
        """
        Visualize specific timestamps.

        Parameters
        ----------
        timestamps : list of str
            Timestamps in YYYYMMDDTHH format, e.g. ['19840530T12', '19660717T06'].
        save_dir : str
            Output directory.
        variables : list of str, optional
            Wave variables to plot, defaults to ['hs'].
        visualize_era5 : bool
            Whether to also plot ERA5 wind and pressure.
        cmap : str
            Colormap for wave plots.
        """
        import matplotlib.pyplot as plt
        from modules.plot.plot_wave import (plot_wave_data_global,
                                             plot_era5_wind_global, plot_era5_pressure_global)

        if variables is None:
            variables = ['hs']

        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []

        for ts in timestamps:
            ts_underscore = ts.replace('T', '_')
            year = ts[:4]
            npy_path = os.path.join(self.data_dir, year, f'global_hindcast_{ts_underscore}.npy')

            if not os.path.exists(npy_path):
                print(f"Warning: file not found: {npy_path}, skipping")
                continue

            # Load wave data
            arr = np.load(npy_path)
            coord_path = os.path.join(os.path.dirname(self.data_dir), 'global_coordinates.npy')
            coords = np.load(coord_path)
            npy_cols = ['hs', 'mwd', 'ssh']
            ds = xr.Dataset(
                {npy_cols[c]: (['node'], arr[:, c].astype(np.float32))
                 for c in range(arr.shape[1])},
                coords={'lon': (['node'], coords[:, 1]),
                        'lat': (['node'], coords[:, 0]),
                        'time': f'{year}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:00:00'},
            )
            ds = ds.rename({'lon': 'x', 'lat': 'y'})
            ds_plot = ds.rename({'x': 'lon', 'y': 'lat'})
            if 'WWM_1' in ds_plot:
                ds_plot = ds_plot.rename({'WWM_1': 'hs'})

            # Plot wave variables
            for var in variables:
                plot_var = 'hs' if var == 'hs' and 'hs' in ds_plot else var
                if plot_var not in ds_plot:
                    print(f"Warning: variable '{var}' not found, skipping")
                    continue
                filename = f'{ts_underscore}_{var}.png'
                save_path = os.path.join(save_dir, filename)
                try:
                    fig = plot_wave_data_global(
                        dataset=ds_plot, variable=plot_var,
                        output_path=save_path, cmap=cmap,
                        use_satellite_background=True,
                        contour_var='ssh' if 'ssh' in ds_plot else None,
                        direction_var='mwd' if 'mwd' in ds_plot else None,
                        custom_title=f'{var.upper()} - global_hindcast_{ts_underscore}',
                        transparent=True,
                    )
                    saved_paths.append(save_path)
                    print(f'Saved: {save_path}')
                    plt.close(fig)
                except Exception as e:
                    print(f'Error plotting {var}: {e}')

            # Plot ERA5
            if visualize_era5 and self.load_era5:
                try:
                    era5_path = self._get_era5_path_for_time(ts)
                    if os.path.exists(era5_path):
                        if era5_path.endswith('.npz'):
                            d = np.load(era5_path, allow_pickle=True)
                            lat_vals = d['_latitude'].astype(np.float32)
                            lon_vals = d['_longitude'].astype(np.float32)
                            ds_era5 = xr.Dataset(
                                {ERA5_VARIABLE_NAMES[0]: (['latitude', 'longitude'], d[ERA5_VARIABLE_NAMES[0]]),
                                 ERA5_VARIABLE_NAMES[1]: (['latitude', 'longitude'], d[ERA5_VARIABLE_NAMES[1]]),
                                 ERA5_VARIABLE_NAMES[2]: (['latitude', 'longitude'], d[ERA5_VARIABLE_NAMES[2]])},
                                coords={'latitude': lat_vals, 'longitude': lon_vals},
                            )
                            d.close()
                        else:
                            ds_era5 = xr.load_dataset(era5_path, engine='netcdf4')

                        # Wind plot (transparent PNG for use in flowcharts etc.)
                        era5_wind_path = os.path.join(save_dir, f'{ts}_era5_wind.png')
                        try:
                            fig_w = plot_era5_wind_global(ds_era5, output_path=era5_wind_path, use_satellite_background=True, transparent=True)
                            saved_paths.append(era5_wind_path)
                            print(f'Saved: {era5_wind_path}')
                            plt.close(fig_w)
                        except Exception as e:
                            print(f'Error plotting ERA5 wind: {e}')

                        # Pressure plot (transparent PNG for use in flowcharts etc.)
                        era5_pres_path = os.path.join(save_dir, f'{ts}_era5_pressure.png')
                        try:
                            fig_p = plot_era5_pressure_global(ds_era5, output_path=era5_pres_path, use_satellite_background=True, transparent=True)
                            saved_paths.append(era5_pres_path)
                            print(f'Saved: {era5_pres_path}')
                            plt.close(fig_p)
                        except Exception as e:
                            print(f'Error plotting ERA5 pressure: {e}')
                    else:
                        print(f'Warning: ERA5 file not found: {era5_path}')
                except Exception as e:
                    print(f'Error visualizing ERA5: {e}')

        print(f'\nCompleted! Visualized {len(timestamps)} timestamps')
        return saved_paths

    def visualize_random_samples(self, n_samples: int = 5, variables: List[str] = None,
                                  region: Sequence[float] = None, save_dir: str = './output/plot/data/hk/random_samples',
                                  show_interpolation: bool = False, cmap: str = 'turbo',
                                  visualize_era5: bool = True):
        """
        Randomly visualize samples from the dataset.
        
        Uses plot_wave_data from plot_wave.py to plot wave data,
        and plot_era5_wind/plot_era5_pressure from plot_wave.py to plot ERA5 data.
        
        Args:
            n_samples (int): Number of samples to visualize, default is 5
            variables (List[str], optional): List of variables to plot, 
                                             defaults to ['hs', 'tp', 'dm', 'elev']
            region (List[float], optional): Map extent [min_lon, max_lon, min_lat, max_lat],
                                            default is None (auto-calculated from data)
            save_dir (str, optional): Directory to save images, 
                                      default is None (display without saving)
            show_interpolation (bool): Whether to show interpolated plot, default is False
            cmap (str): Colormap name, default is 'turbo'
            visualize_era5 (bool): Whether to visualize ERA5 data, default is True
        
        Returns:
            List[str]: List of saved image paths (empty list if save_dir is None)
        """
        import matplotlib.pyplot as plt
        from modules.plot.plot_wave import (plot_wave_data, plot_wave_data_global, 
                                             plot_era5_wind, plot_era5_pressure,
                                             plot_era5_wind_global, plot_era5_pressure_global)
        
        # Set default variables
        if variables is None:
            variables = self.variables if hasattr(self, 'variables') else ['hs', 'tp', 'dm', 'elev']
        
        # Ensure n_samples does not exceed dataset size
        n_samples = min(n_samples, len(self))
        
        # Randomly select sample indices
        random_indices = np.random.choice(len(self), size=n_samples, replace=False)
        
        saved_paths = []
        
        for idx in random_indices:
            # Get file path for this sample
            file_path = self.file_list[idx]
            
            # Load file (support NC, NPZ, NPY formats)
            if file_path.endswith('.npy'):
                # Uncompressed NPY: [N, C] array + separate coordinate file
                arr = np.load(file_path)
                # Coordinates: same parent dir as data_dir
                coord_path = os.path.join(os.path.dirname(self.data_dir),
                                          'global_coordinates.npy')
                coords = np.load(coord_path)
                lons_arr = coords[:, 1]
                lats_arr = coords[:, 0]
                npy_cols = ['hs', 'mwd', 'ssh']
                ds = xr.Dataset(
                    {npy_cols[c]: (['node'], arr[:, c].astype(np.float32))
                     for c in range(arr.shape[1])},
                    coords={'lon': (['node'], lons_arr),
                            'lat': (['node'], lats_arr),
                            'time': self._extract_ts(file_path)},
                )
                if self.region == 'global':
                    ds = ds.rename({'lon': 'x', 'lat': 'y'})
            else:
                ds = xr.load_dataset(file_path)
            
            # Filter data based on region
            if self.region == 'hk':
                min_lon, max_lon, min_lat, max_lat = HK_REGION
                ds_filtered = ds.where(
                    (ds.lon >= min_lon) & (ds.lon <= max_lon) & 
                    (ds.lat >= min_lat) & (ds.lat <= max_lat), 
                    drop=True
                )
            elif self.region == 'global':
                min_lon, max_lon, min_lat, max_lat = GLOBAL_REGION
                ds_filtered = ds.where(
                    (ds.x >= min_lon) & (ds.x <= max_lon) & 
                    (ds.y >= min_lat) & (ds.y <= max_lat), 
                    drop=True
                )
            else:
                ds_filtered = ds
            
            # Calculate actual data extent from filtered dataset for accurate plotting
            if self.region == 'hk':
                data_lons = ds_filtered.lon.values
                data_lats = ds_filtered.lat.values
            elif self.region == 'global':
                data_lons = ds_filtered.x.values
                data_lats = ds_filtered.y.values
            else:
                # Fallback to detect coordinate names
                if 'lon' in ds_filtered:
                    data_lons = ds_filtered.lon.values
                    data_lats = ds_filtered.lat.values
                elif 'x' in ds_filtered:
                    data_lons = ds_filtered.x.values
                    data_lats = ds_filtered.y.values
                else:
                    data_lons = None
                    data_lats = None
            
            # Compute data extent if coordinates are available
            if data_lons is not None and data_lats is not None:
                # Filter out NaN values
                valid_lons = data_lons[~np.isnan(data_lons)]
                valid_lats = data_lats[~np.isnan(data_lats)]
                
                if len(valid_lons) > 0 and len(valid_lats) > 0:
                    # Calculate exact data bounds
                    data_extent = [
                        float(np.min(valid_lons)),  # min_lon
                        float(np.max(valid_lons)),  # max_lon
                        float(np.min(valid_lats)),  # min_lat
                        float(np.max(valid_lats))   # max_lat
                    ]
                else:
                    data_extent = None
            else:
                data_extent = None
            
            # Use provided region, or calculated data extent, or let plot_wave_data auto-calculate
            plot_region = region if region is not None else data_extent
            
            # Create visualization for each wave variable
            for var in variables:
                # Check if variable exists in dataset
                actual_var = var
                if self.region == 'global' and var == 'hs' and 'WWM_1' in ds_filtered:
                    actual_var = 'WWM_1'
                
                if actual_var not in ds_filtered:
                    print(f"Warning: Variable '{var}' not found in dataset {file_path}, skipping")
                    continue
                
                # Build save path
                save_path = None
                if save_dir is not None:
                    timestamp = os.path.basename(file_path).replace('.npy', '').replace('.nc', '').replace('wave_', '').replace('global_hindcast_', '')
                    filename = f"{timestamp}_{var}.jpg"
                    save_path = os.path.join(save_dir, filename)
                    os.makedirs(save_dir, exist_ok=True)
                
                # Prepare dataset for plotting
                # For global data, rename coordinates from (x, y) to (lon, lat) to match plot_wave_data expectations
                if self.region == 'global':
                    ds_plot = ds_filtered.rename({'x': 'lon', 'y': 'lat'})
                    # Also rename variable if it's WWM_1 to hs for consistent plotting
                    if actual_var == 'WWM_1':
                        ds_plot = ds_plot.rename({'WWM_1': 'hs'})
                        plot_var = 'hs'
                    else:
                        plot_var = actual_var
                else:
                    ds_plot = ds_filtered
                    plot_var = actual_var
                
                # Call plot_wave_data for plotting
                try:
                    # Use global plotting function for global data
                    if self.region == 'global':
                        fig = plot_wave_data_global(
                            dataset=ds_plot,
                            variable=plot_var,
                            output_path=save_path,
                            cmap=cmap,
                            show_interpolation=show_interpolation,
                            custom_title=f"{var.upper()} - {os.path.basename(file_path)}",
                            use_satellite_background=True,
                            contour_var='ssh' if 'ssh' in ds_plot else None,
                            direction_var='mwd' if 'mwd' in ds_plot else None,
                        )
                    else:
                        fig = plot_wave_data(
                            dataset=ds_plot,
                            variable=plot_var,
                            region=plot_region,
                            save_path=save_path,
                            cmap=cmap,
                            show_interpolation=show_interpolation,
                            custom_title=f"{var.upper()} - {os.path.basename(file_path)}"
                        )
                    
                    if save_path:
                        saved_paths.append(save_path)
                        print(f"Saved: {save_path}")
                    else:
                        plt.show()
                    
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"Error plotting {var}: {e}")
                    continue
            
            # Visualize ERA5 data if enabled and dataset loads ERA5
            if visualize_era5 and self.load_era5:
                try:
                    # Extract timestamp from filename
                    filename = os.path.basename(file_path)
                    if filename.startswith('wave_'):
                        timestamp = filename.replace('wave_', '').replace('.nc', '').replace('.npy', '')
                    elif filename.startswith('global_hindcast_'):
                        parts = filename.replace('global_hindcast_', '').replace('.nc', '').replace('.npy', '').split('_')
                        timestamp = f"{parts[0]}T{parts[1]}"
                    else:
                        timestamp = None
                    
                    if timestamp:
                        # Load ERA5 data for this timestamp
                        era5_path = self._get_era5_path_for_time(timestamp)
                        
                        if os.path.exists(era5_path):
                            if era5_path.endswith('.npz'):
                                d = np.load(era5_path, allow_pickle=True)
                                lat_vals = d['_latitude'].astype(np.float32)
                                lon_vals = d['_longitude'].astype(np.float32)
                                ds_era5 = xr.Dataset(
                                    {ERA5_VARIABLE_NAMES[0]: (['latitude', 'longitude'], d[ERA5_VARIABLE_NAMES[0]]),
                                     ERA5_VARIABLE_NAMES[1]: (['latitude', 'longitude'], d[ERA5_VARIABLE_NAMES[1]]),
                                     ERA5_VARIABLE_NAMES[2]: (['latitude', 'longitude'], d[ERA5_VARIABLE_NAMES[2]])},
                                    coords={'latitude': lat_vals, 'longitude': lon_vals},
                                )
                                d.close()
                            else:
                                ds_era5 = xr.load_dataset(era5_path, engine='netcdf4')
                            
                            # Select ERA5 region based on configuration
                            if self.region == 'global':
                                lon_range = ERA5_GLOBAL_LON_RANGE
                                lat_range = ERA5_GLOBAL_LAT_RANGE
                            elif self.era5_region == 1:
                                lon_range = ERA5_HK_LON_RANGE_1
                                lat_range = ERA5_HK_LAT_RANGE_1
                            elif self.era5_region == 2:
                                lon_range = ERA5_HK_LON_RANGE_2
                                lat_range = ERA5_HK_LAT_RANGE_2
                            elif self.era5_region == 3:
                                lon_range = ERA5_HK_LON_RANGE_3
                                lat_range = ERA5_HK_LAT_RANGE_3
                            else:
                                lon_range = ERA5_HK_LON_RANGE_1
                                lat_range = ERA5_HK_LAT_RANGE_1
                            
                            min_lon, max_lon = lon_range
                            min_lat, max_lat = lat_range
                            
                            # Filter ERA5 data
                            if 'longitude' in ds_era5:
                                lon_name = 'longitude'
                                lat_name = 'latitude'
                            else:
                                lon_name = 'lon'
                                lat_name = 'lat'
                            
                            # Convert longitude from 0-360 to -180-180 if needed for global data
                            if self.region == 'global' and ds_era5[lon_name].max() > 180:
                                ds_era5[lon_name] = ((ds_era5[lon_name] + 180) % 360) - 180
                                ds_era5 = ds_era5.sortby(lon_name)
                            
                            ds_era5_filtered = ds_era5.where(
                                (ds_era5[lon_name] >= min_lon) & (ds_era5[lon_name] < max_lon) & 
                                (ds_era5[lat_name] >= min_lat) & (ds_era5[lat_name] < max_lat), 
                                drop=True
                            )
                            
                            # Build save path for ERA5 wind
                            era5_wind_save_path = None
                            era5_pressure_save_path = None
                            if save_dir is not None:
                                era5_wind_filename = f"{timestamp}_era5_wind.jpg"
                                era5_pressure_filename = f"{timestamp}_era5_pressure.jpg"
                                era5_wind_save_path = os.path.join(save_dir, era5_wind_filename)
                                era5_pressure_save_path = os.path.join(save_dir, era5_pressure_filename)
                            
                            # Plot ERA5 wind components
                            try:
                                # Use global plotting functions for global data
                                if self.region == 'global':
                                    fig_wind = plot_era5_wind_global(
                                        dataset=ds_era5_filtered,
                                        output_path=era5_wind_save_path if era5_wind_save_path else 'temp_wind.jpg',
                                        use_satellite_background=True
                                    )
                                else:
                                    fig_wind = plot_era5_wind(
                                        dataset=ds_era5_filtered,
                                        output_path=era5_wind_save_path if era5_wind_save_path else 'temp_wind.jpg'
                                    )
                                if era5_wind_save_path:
                                    saved_paths.append(era5_wind_save_path)
                                    print(f"Saved: {era5_wind_save_path}")
                                plt.close(fig_wind)
                            except Exception as e:
                                print(f"Error plotting ERA5 wind: {e}")
                            
                            # Plot ERA5 pressure
                            try:
                                # Use global plotting functions for global data
                                if self.region == 'global':
                                    fig_pressure = plot_era5_pressure_global(
                                        dataset=ds_era5_filtered,
                                        output_path=era5_pressure_save_path if era5_pressure_save_path else 'temp_pressure.jpg',
                                        use_satellite_background=True
                                    )
                                else:
                                    fig_pressure = plot_era5_pressure(
                                        dataset=ds_era5_filtered,
                                        output_path=era5_pressure_save_path if era5_pressure_save_path else 'temp_pressure.jpg'
                                    )
                                if era5_pressure_save_path:
                                    saved_paths.append(era5_pressure_save_path)
                                    print(f"Saved: {era5_pressure_save_path}")
                                plt.close(fig_pressure)
                            except Exception as e:
                                print(f"Error plotting ERA5 pressure: {e}")
                        else:
                            print(f"Warning: ERA5 file not found: {era5_path}")
                            
                except Exception as e:
                    print(f"Error visualizing ERA5 data: {e}")
        
        print(f"\nCompleted! Visualized {n_samples} samples with {len(variables)} wave variables")
        if visualize_era5 and self.load_era5:
            print("ERA5 wind and pressure plots also generated")
        return saved_paths


def collate_wave_samples(batch: List[dict]) -> dict:
    """
    Custom collate function for batching wave data samples.
    
    This function handles the batching of wave data samples in node-wise format.
    It stacks arrays for wave_data and ERA5 data.
    
    Timestamp is kept as a list of strings.
    
    Args:
        batch (List[dict]): List of samples from WaveDataset
        
    Returns:
        dict: Batched data as a dictionary:
            - wave_data: [B, N, C]
            - era5 (if present): [B, T, 3, H, W] or [B, 3, H, W] (ERA5 on regular grid)
            - era5_on_nodes (if present): [B, N, 3*T] (ERA5 interpolated to wave nodes)
            - timestamp: List[str] of length B
            - tod (if present): [B] tensor of normalized hour values [0, 1)
            - time_feats (if present): [B, TIME_FEAT_DIM] tensor of absolute-time sin/cos features
    """
    # Extract timestamps
    timestamps = [sample['timestamp'] for sample in batch]
    
    # Batch wave data
    # Each sample['wave_data'] is [N, C]
    wave_data_list = [sample['wave_data'] for sample in batch]
    batched_wave = torch.stack([torch.from_numpy(d) for d in wave_data_list]) # [B, N, C]
    
    result = {
        'wave_data': batched_wave,
        'timestamp': timestamps
    }

    # Batch time-of-day for tidal conditioning
    if 'tod' in batch[0]:
        tod_list = [sample['tod'] for sample in batch]
        result['tod'] = torch.tensor(tod_list, dtype=torch.float32)  # [B]

    # Batch absolute-time tidal features
    if 'time_feats' in batch[0]:
        time_feats_list = [sample['time_feats'] for sample in batch]
        result['time_feats'] = torch.from_numpy(np.stack(time_feats_list))  # [B, TIME_FEAT_DIM]

    # Batch ERA5 data on regular grid if present
    if 'era5' in batch[0]:
        era5_list = [sample['era5'] for sample in batch]
        batched_era5 = torch.stack([torch.from_numpy(d) for d in era5_list]) # [B, 3, H, W] or [B, T, 3, H, W]
        result['era5'] = batched_era5
    
    # Batch ERA5 data on wave nodes if present
    if 'era5_on_nodes' in batch[0]:
        era5_on_nodes_list = [sample['era5_on_nodes'] for sample in batch]
        batched_era5_on_nodes = torch.stack([torch.from_numpy(d) for d in era5_on_nodes_list]) # [B, N, 3*T]
        result['era5_on_nodes'] = batched_era5_on_nodes
        
    return result



from modules.datasets.wave_normalize import NormalizeWaveData  # noqa: E402  (re-export for backward compat)
from modules.datasets.wave_dataset_selftest import (  # noqa: E402
    test_wave_dataset,
    test_global_wave_dataset,
)
