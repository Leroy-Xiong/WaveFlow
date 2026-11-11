"""
GridWaveDataset: PyTorch Dataset for regular-grid wave data (HK zoom 75×107).

Loads preinterpolated .npy files from ``preinterpolated_hk_zoom/`` and
applies the same normalisation as the existing node-based WaveDataset
(hs/tp log1p+zscore, dm sin/cos decomposition, elev zscore).

Output tensor shape: (C, H, W) = (5, 75, 107).
Land pixels are kept as NaN and masked during loss computation.
"""

import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from modules.datasets.wave_dataset import compute_time_features
from modules.utils import ERA5_VARIABLE_NAMES


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_year(ts: str) -> str:
    """'20220101T00' -> '2022'."""
    return ts[:4]


def _ts_from_filename(fname: str) -> str:
    """'wave_20220101T00.npy' -> '20220101T00'."""
    return os.path.basename(fname).replace('wave_', '').replace('.npy', '')


# ---------------------------------------------------------------------------
# GridWaveDataset
# ---------------------------------------------------------------------------

class GridWaveDataset(Dataset):
    """PyTorch Dataset for HK zoom regular-grid wave data.

    Parameters
    ----------
    grid_dir : str
        Root directory with per-year subdirectories of ``wave_*.npy`` files.
    era5_dir : str or None
        Directory containing ERA5 .nc files. If None, ERA5 is not loaded.
    stats_file : str
        Path to ``wave_data_stats.json`` with normalisation constants.
    variables : list of str
        Wave variables to load. Default ['hs', 'tp', 'dm', 'elev'] → 5 output channels.
    split : str
        'train' (1970–2020) or 'test' (2021–2022).
    load_era5 : bool
        Whether to load ERA5 atmospheric data.
    era5_region : int
        ERA5 region preset (1, 2, or 3). Default 2.
    num_era5_frames : int
        Number of historical ERA5 frames (6-h interval). Default 8.
    """

    def __init__(
        self,
        grid_dir: str = './data/hindcast/preinterpolated_hk_zoom',
        era5_dir: str = './data/era5/nc_files',
        stats_file: str = './data/hindcast/wave_data_stats.json',
        era5_stats_file: str = './data/era5/era5_statistics.json',
        variables: list = None,
        split: str = 'train',
        load_era5: bool = True,
        era5_region: int = 2,
        num_era5_frames: int = 8,
        era5_file_format: str = 'nc',
    ):
        super().__init__()
        self.grid_dir = grid_dir
        self.era5_dir = era5_dir
        self.split = split
        self.load_era5 = load_era5
        self.era5_region = era5_region
        self.num_era5_frames = num_era5_frames
        self.era5_file_format = era5_file_format.lower()

        self.variables = variables or ['hs', 'tp', 'dm', 'elev']
        # dm expands to sin+cos → output channels = n_vars + 1
        self.out_channels = len(self.variables) + (1 if 'dm' in self.variables else 0)

        # Load normalisation stats
        with open(stats_file, 'r') as f:
            self.stats = json.load(f)

        # Load ERA5 Min-Max normalisation stats (same as NormalizeWaveData)
        self.era5_min = {}
        self.era5_range = {}
        if load_era5:
            with open(era5_stats_file, 'r') as f:
                era5_stats = json.load(f)
            from modules.utils import ERA5_VARIABLE_NAMES
            for var_info in era5_stats['variables']:
                name = var_info['variable']
                if name in ERA5_VARIABLE_NAMES:
                    self.era5_min[name] = var_info['min_value']
                    self.era5_range[name] = var_info['max_value'] - var_info['min_value']

        # Gather all .npy files
        self._samples = self._build_index()

    # ------------------------------------------------------------------
    # File index
    # ------------------------------------------------------------------
    def _build_index(self) -> list:
        """Return sorted list of (timestamp, npy_path) tuples for this split."""
        test_years = {'2021', '2022'}
        samples = []
        year_dirs = sorted(
            d for d in os.listdir(self.grid_dir)
            if os.path.isdir(os.path.join(self.grid_dir, d)) and d.isdigit()
        )
        for year in year_dirs:
            if (self.split == 'test') != (year in test_years):
                continue
            year_dir = os.path.join(self.grid_dir, year)
            for fname in sorted(os.listdir(year_dir)):
                if fname.startswith('wave_') and fname.endswith('.npy'):
                    ts = _ts_from_filename(fname)
                    samples.append((ts, os.path.join(year_dir, fname)))
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    # ------------------------------------------------------------------
    # ERA5 loading (matching WaveDataset._load_era5_sequence)
    # ------------------------------------------------------------------
    def _load_era5(self, ts: str) -> np.ndarray:
        """Load ERA5 frames for a timestamp.  Returns (T, 3, H_era, W_era) or None.

        Supports both NC (xarray) and NPZ formats selected via ``era5_file_format``.
        """
        from datetime import datetime, timedelta

        dt_base = datetime.strptime(ts, '%Y%m%dT%H')
        use_npz = (self.era5_file_format == 'npz')
        ext = 'npz' if use_npz else 'nc'

        # Region bounds for cropping (ERA5 NPZ is full Region-1 40×40)
        from modules.utils import (
            ERA5_HK_LON_RANGE_1, ERA5_HK_LAT_RANGE_1,
            ERA5_HK_LON_RANGE_2, ERA5_HK_LAT_RANGE_2,
            ERA5_HK_LON_RANGE_3, ERA5_HK_LAT_RANGE_3,
        )
        region_map = {
            1: (ERA5_HK_LON_RANGE_1, ERA5_HK_LAT_RANGE_1),
            2: (ERA5_HK_LON_RANGE_2, ERA5_HK_LAT_RANGE_2),
            3: (ERA5_HK_LON_RANGE_3, ERA5_HK_LAT_RANGE_3),
        }
        (min_lon, max_lon), (min_lat, max_lat) = region_map.get(
            self.era5_region, region_map[1]
        )

        frames = []
        for i in range(self.num_era5_frames - 1, -1, -1):
            dt_offset = dt_base - timedelta(hours=i * 6)
            era5_ts = dt_offset.strftime('%Y%m%dT%H')
            year = era5_ts[:4]
            path = os.path.join(self.era5_dir, year, f'era5_{era5_ts}.{ext}')

            if not os.path.exists(path):
                return None

            if use_npz:
                data = np.load(path)
                # Use coordinate arrays to crop to target region
                lons = data['_longitude']
                lats = data['_latitude']
                lon_mask = (lons >= min_lon) & (lons < max_lon)
                lat_mask = (lats >= min_lat) & (lats < max_lat)
                frame = np.stack([
                    data[ERA5_VARIABLE_NAMES[0]][lat_mask, :][:, lon_mask].astype(np.float32),
                    data[ERA5_VARIABLE_NAMES[1]][lat_mask, :][:, lon_mask].astype(np.float32),
                    data[ERA5_VARIABLE_NAMES[2]][lat_mask, :][:, lon_mask].astype(np.float32),
                ], axis=0)  # (3, H, W)
                data.close()
            else:
                import xarray as xr
                ds = xr.load_dataset(path, engine='netcdf4')
                lon_name = 'longitude' if 'longitude' in ds else 'lon'
                lat_name = 'latitude' if 'latitude' in ds else 'lat'
                ds = ds.where(
                    (ds[lon_name] >= min_lon) & (ds[lon_name] < max_lon) &
                    (ds[lat_name] >= min_lat) & (ds[lat_name] < max_lat),
                    drop=True,
                )
                frame = np.stack([
                    ds[ERA5_VARIABLE_NAMES[0]].values.astype(np.float32),
                    ds[ERA5_VARIABLE_NAMES[1]].values.astype(np.float32),
                    ds[ERA5_VARIABLE_NAMES[2]].values.astype(np.float32),
                ], axis=0)  # (3, H, W)
                ds.close()

            frames.append(frame)

        if len(frames) != self.num_era5_frames:
            return None

        era5 = np.stack(frames, axis=0)  # (T, 3, H, W)

        # Min-Max normalise to [-1, 1] (matching NormalizeWaveData)
        for i, var_name in enumerate(ERA5_VARIABLE_NAMES):
            vals = era5[:, i]  # (T, H, W)
            norm = (vals - self.era5_min[var_name]) / self.era5_range[var_name]
            era5[:, i] = norm * 2.0 - 1.0

        return era5

    # ------------------------------------------------------------------
    # Normalisation (same as NormalizeWaveData but on grid format)
    # ------------------------------------------------------------------
    def _normalize(self, wave_grid: np.ndarray) -> np.ndarray:
        """Normalize raw grid data.

        Input:  (n_vars, H, W) with raw values [hs, tp, dm, elev].
        Output: (out_channels, H, W) with [hs_norm, tp_norm, dm_sin, dm_cos, elev_norm].
        NaN pixels (land) are preserved.
        """
        channels = []
        for i, var in enumerate(self.variables):
            vals = wave_grid[i]  # (H, W)

            if var in ('hs', 'tp'):
                log_mean = self.stats[f'{var}_log_mean']
                log_std = self.stats[f'{var}_log_std']
                valid = ~np.isnan(vals)
                result = np.full_like(vals, np.nan, dtype=np.float32)
                if valid.any():
                    clipped = np.maximum(vals[valid], 0.0)
                    result[valid] = ((np.log1p(clipped) - log_mean) / log_std).astype(np.float32)
                channels.append(result)

            elif var == 'dm':
                rad = np.deg2rad(vals)
                # sin channel
                result_sin = np.full_like(vals, np.nan, dtype=np.float32)
                valid = ~np.isnan(vals)
                if valid.any():
                    result_sin[valid] = np.sin(rad[valid]).astype(np.float32)
                channels.append(result_sin)
                # cos channel
                result_cos = np.full_like(vals, np.nan, dtype=np.float32)
                if valid.any():
                    result_cos[valid] = np.cos(rad[valid]).astype(np.float32)
                channels.append(result_cos)

            elif var == 'elev':
                mean = self.stats['elev_mean']
                std = self.stats['elev_std']
                valid = ~np.isnan(vals)
                result = np.full_like(vals, np.nan, dtype=np.float32)
                if valid.any():
                    result[valid] = ((vals[valid] - mean) / std).astype(np.float32)
                channels.append(result)

        return np.stack(channels, axis=0)  # (C_out, H, W)

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        ts, npy_path = self._samples[idx]
        raw = np.load(npy_path).astype(np.float32)  # (n_vars, H, W)

        # Handle both 3-ch (hs,tp,dm) and 4-ch (hs,tp,dm,elev)
        n_loaded = raw.shape[0]
        n_expected = len(self.variables)
        if n_loaded < n_expected:
            # Missing channels (e.g., elev) — pad with NaN
            n_missing = n_expected - n_loaded
            pad = np.full((n_missing, *raw.shape[1:]), np.nan, dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=0)
        elif n_loaded > n_expected:
            # Extra channels — truncate
            raw = raw[:n_expected]

        wave_data = self._normalize(raw)  # (C_out, H, W)

        # Validity mask: True where NOT NaN (ocean pixels)
        valid_mask = ~np.isnan(wave_data[0])  # (H, W), any channel works

        # Zero-fill NaN for model input
        wave_data = np.nan_to_num(wave_data, nan=0.0)

        sample = {
            'wave_data': torch.from_numpy(wave_data),
            'valid_mask': torch.from_numpy(valid_mask),
            'timestamp': ts,
            # Time-of-day (hour normalized to [0, 1)) — deprecated, kept for parity
            'tod': np.float32(int(ts[9:11]) / 24.0),
            # Absolute-time tidal features (sin/cos at constituent periods), same
            # as WaveDataset — used by use_tod_conditioning=True models
            'time_feats': compute_time_features(ts),
        }

        if self.load_era5:
            era5 = self._load_era5(ts)
            if era5 is not None:
                sample['era5'] = torch.from_numpy(era5)

        return sample


# ---------------------------------------------------------------------------
# collate function
# ---------------------------------------------------------------------------

def collate_grid_samples(batch: list) -> dict:
    """Collate a list of GridWaveDataset samples into a batch.

    - wave_data:  (B, C, H, W)
    - era5:       (B, T, 3, H_era5, W_era5) or None
    - timestamp:  list of str
    """
    wave = torch.stack([s['wave_data'] for s in batch], dim=0)  # (B, C, H, W)
    mask = torch.stack([s['valid_mask'] for s in batch], dim=0)  # (B, H, W)

    # Expand mask to match wave channels: (B, H, W) → (B, C, H, W)
    # Clone after expand to avoid shared-memory issues with pin_memory
    mask_4d = mask.unsqueeze(1).expand_as(wave).clone()

    result = {
        'wave_data': wave,
        'valid_mask': mask_4d,
        'timestamp': [s['timestamp'] for s in batch],
    }

    if 'tod' in batch[0]:
        result['tod'] = torch.tensor([s['tod'] for s in batch], dtype=torch.float32)  # [B]

    if 'time_feats' in batch[0]:
        result['time_feats'] = torch.from_numpy(
            np.stack([s['time_feats'] for s in batch])
        )  # [B, TIME_FEAT_DIM]

    if 'era5' in batch[0]:
        result['era5'] = torch.stack([s['era5'] for s in batch], dim=0)

    return result
