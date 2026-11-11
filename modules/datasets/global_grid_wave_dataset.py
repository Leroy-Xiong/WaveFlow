"""
GlobalGridWaveDataset: PyTorch Dataset for global 0.25° regular-grid wave data.

Supports two file formats:
- 'npy': Per-sample NPY files (fast, uncompressed). Wave shape (3,720,1440), ERA5 shape (3,720,1440).
- 'npz': Per-sample NPZ files (compressed, slower). Wave key 'data' (3,721,1440), ERA5 per-variable keys.

Grid: 720×1440 (south pole duplicate row already dropped in NPY; cropped on-the-fly for NPZ).
"""

import os
import json
import glob
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import Dataset


TRAIN_YEARS = range(1959, 2021)
TEST_YEARS = range(2021, 2024)

ERA5_VARIABLES = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'mean_sea_level_pressure',
]


class GlobalGridWaveDataset(Dataset):
    def __init__(
        self,
        wave_dir,
        era5_dir,
        stats_file,
        era5_stats_file,
        land_mask_file,
        split='train',
        num_era5_frames=4,
        grid_shape=(720, 1440),
        file_format='npy',
    ):
        super().__init__()
        self.wave_dir = wave_dir
        self.era5_dir = era5_dir
        self.num_era5_frames = num_era5_frames
        self.grid_shape = grid_shape
        self.file_format = file_format

        with open(stats_file) as f:
            stats = json.load(f)
        self.hs_log_mean = stats['hs_log_mean']
        self.hs_log_std = stats['hs_log_std']

        with open(era5_stats_file) as f:
            era5_stats = json.load(f)
        self.era5_min = []
        self.era5_max = []
        for var_info in era5_stats['variables']:
            self.era5_min.append(var_info['min_value'])
            self.era5_max.append(var_info['max_value'])
        self.era5_min = np.array(self.era5_min, dtype=np.float32)
        self.era5_max = np.array(self.era5_max, dtype=np.float32)

        land_mask_raw = np.load(land_mask_file)
        if land_mask_raw.ndim == 1:
            land_mask_raw = land_mask_raw.reshape(721, 1440)
        self.land_mask = land_mask_raw[:grid_shape[0], :grid_shape[1]]

        years = TRAIN_YEARS if split == 'train' else TEST_YEARS
        self.file_list = self._build_index(years)

    def _build_index(self, years):
        files = []
        if self.file_format == 'daily':
            for year in years:
                year_dir = os.path.join(self.wave_dir, str(year))
                if not os.path.isdir(year_dir):
                    continue
                for path in sorted(glob.glob(os.path.join(year_dir, 'wave_*.npy'))):
                    basename = os.path.basename(path)  # wave_YYYYMMDD.npy
                    date_str = basename.replace('wave_', '').replace('.npy', '')
                    for h in [0, 3, 6, 9, 12, 15, 18, 21]:
                        ts = datetime.strptime(f"{date_str}{h:02d}", "%Y%m%d%H")
                        files.append((ts, path))
        else:
            ext = 'npy' if self.file_format in ('npy', 'packed') else 'npz'
            for year in years:
                year_dir = os.path.join(self.wave_dir, str(year))
                if not os.path.isdir(year_dir):
                    continue
                pattern = os.path.join(year_dir, f'global_hindcast_*.{ext}')
                for path in sorted(glob.glob(pattern)):
                    basename = os.path.basename(path)
                    parts = basename.replace(f'.{ext}', '').split('_')
                    date_str = parts[2]
                    hour_str = parts[3]
                    ts = datetime.strptime(f"{date_str}{hour_str}", "%Y%m%d%H")
                    files.append((ts, path))
        return files

    def __len__(self):
        return len(self.file_list)

    def _load_era5_frame(self, ts):
        year_str = ts.strftime('%Y')
        if self.file_format == 'npy':
            fname = ts.strftime('era5_%Y%m%dT%H.npy')
            path = os.path.join(self.era5_dir, year_str, fname)
            if not os.path.exists(path):
                return None
            return np.load(path)  # (3, H, W) sequential read
        else:
            fname = ts.strftime('era5_%Y%m%dT%H.npz')
            path = os.path.join(self.era5_dir, year_str, fname)
            if not os.path.exists(path):
                return None
            data = np.load(path)
            frames = [data[v][:self.grid_shape[0], :self.grid_shape[1]] for v in ERA5_VARIABLES]
            return np.stack(frames, axis=0)

    def __getitem__(self, idx):
        ts, wave_path = self.file_list[idx]

        if self.file_format == 'daily':
            daily_data = np.load(wave_path)  # (8, 720, 1440) sequential read
            hs = daily_data[ts.hour // 3]
        elif self.file_format == 'packed':
            hs = np.load(wave_path)  # (720, 1440) hs-only
        elif self.file_format == 'npy':
            wave_data = np.load(wave_path)  # (3, 720, 1440)
            hs = wave_data[0]
        else:
            wave_data = np.load(wave_path)['data']
            hs = wave_data[0, :self.grid_shape[0], :self.grid_shape[1]]

        valid_mask = ~self.land_mask & ~np.isnan(hs)

        hs = np.nan_to_num(hs, nan=0.0)
        hs = np.log1p(np.maximum(hs, 0.0))
        hs = (hs - self.hs_log_mean) / self.hs_log_std
        hs[~valid_mask] = 0.0

        if self.file_format == 'daily':
            era5_frames = []
            for i in range(self.num_era5_frames - 1, -1, -1):
                frame_ts = ts - timedelta(hours=6 * i)
                era5_path = os.path.join(
                    self.era5_dir, frame_ts.strftime('%Y'),
                    frame_ts.strftime('era5_%Y%m%d.npy'))
                if not os.path.exists(era5_path):
                    return None
                era5_day = np.load(era5_path)  # (8, 3, 720, 1440)
                era5_frames.append(era5_day[frame_ts.hour // 3])
            era5 = np.stack(era5_frames, axis=0)  # (4, 3, H, W)
        elif self.file_format == 'packed':
            year_str = ts.strftime('%Y')
            era5_path = os.path.join(self.era5_dir, year_str,
                                     ts.strftime('era5_%Y%m%dT%H.npy'))
            if not os.path.exists(era5_path):
                return None
            era5 = np.load(era5_path)  # (4, 3, H, W) pre-packed
        else:
            era5_frames = []
            for i in range(self.num_era5_frames - 1, -1, -1):
                frame_ts = ts - timedelta(hours=6 * i)
                frame = self._load_era5_frame(frame_ts)
                if frame is None:
                    return None
                era5_frames.append(frame)
            era5 = np.stack(era5_frames, axis=0)  # (T, 3, H, W)

        era5_min = self.era5_min[:, None, None]
        era5_max = self.era5_max[:, None, None]
        era5 = (era5 - era5_min) / (era5_max - era5_min + 1e-8) * 2.0 - 1.0

        return {
            'wave_data': torch.from_numpy(hs[None].astype(np.float32)),  # (1, H, W)
            'era5': torch.from_numpy(era5.astype(np.float32)),  # (T, 3, H, W)
            'valid_mask': torch.from_numpy(valid_mask[None]),  # (1, H, W) bool
            'timestamp': ts.strftime('%Y%m%dT%H'),
        }


def collate_global_grid_samples(batch):
    batch = [s for s in batch if s is not None]
    if len(batch) == 0:
        return None
    return {
        'wave_data': torch.stack([s['wave_data'] for s in batch]),
        'era5': torch.stack([s['era5'] for s in batch]),
        'valid_mask': torch.stack([s['valid_mask'] for s in batch]),
        'timestamp': [s['timestamp'] for s in batch],
    }
