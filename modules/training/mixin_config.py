import numpy as np
import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.optim.lr_scheduler as lr_scheduler
import logging
import sys
from datetime import datetime
import time
import yaml
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import random

from modules.flow_and_diffusion import FlowDiffusion
from modules.graph_transformer import (
    GraphTransformer_Base, 
    PatchedGraphTransformer_Base,
    PatchedGraphTransformer_Small,
    PatchedGraphTransformer_Large,
    PatchedGraphTransformer_Huge,
    PatchedGraphTransformer_Giant
)
from modules.datasets.wave_dataset import WaveDataset, collate_wave_samples, NormalizeWaveData
from modules.utils import backup_python_scripts
from modules.profiling import TrainingProfiler, enable_gpu_optimizations
from modules.ema import EMA
from modules.config import load_config, apply_overrides
import copy

# ── torch.compile stability ──
# Prevent aggressive recompilation: raise the cache limit so inductor keeps
# more FX graphs cached, and use 'high' matmul precision to avoid retuning.
torch._dynamo.config.cache_size_limit = 256
torch.set_float32_matmul_precision('high')




class TrainerConfigMixin:
    def _extract_config_parameters(self, config):
        """
        Extract and store configuration parameters from config dict.
        
        Args:
            config: Configuration dictionary loaded from YAML file
        """
        # Data config
        self.data_dir = config['data']['data_dir']
        self.batch_size = int(config['training']['batch_size'])
        self.coord_file_path = config['data']['coord_file_path']
        
        # Calculate learning rate with relative learning rate as priority
        # If base_learning_rate is set (non-null), use relative learning rate calculation
        # Otherwise, fall back to absolute learning_rate
        config_base_lr = config['training'].get('base_learning_rate')
        
        if config_base_lr is not None and config_base_lr != '':
            # Calculate relative learning rate (priority)
            # Formula: effective_lr = base_learning_rate * (batch_size * num_gpus) / base_batch_size
            base_learning_rate = float(config_base_lr)
            base_batch_size = float(config['training'].get('base_batch_size', self.batch_size))
            
            # Validate base_batch_size to prevent division by zero
            if base_batch_size <= 0:
                raise ValueError(
                    f"base_batch_size must be positive, got {base_batch_size}. "
                    f"Please check training.base_batch_size in config."
                )
            
            # Get world_size (number of GPUs) from environment variable or default to 1
            # Note: _setup_ddp() is called after this method, so we read from env directly
            if 'WORLD_SIZE' in os.environ:
                num_gpus = int(os.environ['WORLD_SIZE'])
            else:
                num_gpus = 1
            
            # Calculate effective learning rate
            global_batch_size = self.batch_size * num_gpus
            self.learning_rate = base_learning_rate * (global_batch_size / base_batch_size)
            
            # Store learning rate calculation info for later logging
            self._lr_calculation_info = {
                'mode': 'relative',
                'base_learning_rate': base_learning_rate,
                'base_batch_size': base_batch_size,
                'batch_size_per_gpu': self.batch_size,
                'num_gpus': num_gpus,
                'global_batch_size': global_batch_size,
                'effective_learning_rate': self.learning_rate
            }
        else:
            # Use absolute learning rate (fallback)
            config_lr = config['training'].get('learning_rate', 1e-4)
            self.learning_rate = float(config_lr)
            self._lr_calculation_info = {
                'mode': 'absolute',
                'learning_rate': self.learning_rate
            }
        
        self.stats_file = config['data']['stats_file']
        self.era5_stats_file = config['data']['era5_stats_file']
        self.era5_dir = config['data'].get('era5_dir', './data/era5/global_nc_files')
        
        self.variables = config['data']['variables']
        self.era5_region = config['data'].get('era5_region', 1)
        self.cluster_file = config['data'].get('cluster_file', './data/hindcast/hk_clusters.npz')
        self.region = config['data'].get('region', 'hk')
        self.file_format = config['data'].get('file_format', 'nc')
        self.era5_file_format = config['data'].get('era5_file_format', 'nc')
        self.use_preinterpolated_era5 = config['data'].get('use_preinterpolated_era5', False)
        
        # Training config
        self.epochs = int(config['training']['epochs'])
        self.log_interval = int(config['training']['log_interval'])
        self.checkpoint_interval = int(config['training'].get('checkpoint_interval', 1))  # Default: save every epoch
        self.seed = config['training'].get('seed', 42)  # Random seed for reproducibility
        
        # Learning rate warmup configuration
        self.warmup_epochs = int(config['training'].get('warmup_epochs', 5))  # Default: 5 epochs (0 = disabled)
        self.scheduler_type = config['training'].get('scheduler_type', 'cosine').lower()  # Default: cosine
        # LR schedule horizon in epochs. Default = training.epochs (previous behavior).
        # Set it larger than epochs to keep the schedule of a longer run and stop early.
        self.scheduler_epochs = int(config['training'].get('scheduler_epochs', self.epochs))
        
        # Gradient clipping configuration
        grad_clip = config['training'].get('grad_clip_norm', 1.0)
        self.grad_clip_norm = float(grad_clip) if grad_clip else None  # Default: 1.0
        
        # EMA (Exponential Moving Average) configuration
        self.use_ema = config['training'].get('use_ema', True)  # Default: enabled
        self.ema_decay = float(config['training'].get('ema_decay', 0.9999))  # Default: 0.9999
        self.ema_update_after_step = int(config['training'].get('ema_update_after_step', 100))  # Default: 100
        
        # ProRes (Progressive Residual Warmup) configuration
        # Reference: https://arxiv.org/abs/2603.05369
        self.use_prores = config['training'].get('use_prores', False)  # Default: disabled
        self.prores_warmup_epochs = float(config['training'].get('prores_warmup_epochs', 1))  # T parameter, default: 1 epoch
        
        # Dataset config
        self.num_workers = int(config['dataset']['num_workers'])
        self.pin_memory = config['dataset']['pin_memory']
        self.use_patches = config['dataset'].get('use_patches', False)
        self.prefetch_factor = int(config['dataset'].get('prefetch_factor', 4))
        # Grid-specific (region='hk_grid')
        self.grid_shape = config['data'].get('grid_shape', None)
        self.land_mask_file = config['data'].get('land_mask_file', None)
        self.persistent_workers = config['dataset'].get('persistent_workers', True)
        self.use_mmap = config['dataset'].get('use_mmap', True)

        # Model config
        self.in_channels = int(config['model']['in_channels'])
        self.out_channels = int(config['model'].get('out_channels', self.in_channels))
        self.conditional_generation = config['model']['conditional_generation']
        self.model_type = config['model'].get('model_type', 'base').lower()  # Default to 'base'
        self.era5_encoder_type = config['model'].get('era5_encoder_type', 'cnn').lower()
        self.era5_patch_size = int(config['model'].get('era5_patch_size', 4))
        self.era5_num_frames = int(config['model'].get('era5_num_frames', 1))
        self.coord_encoding_type = config['model'].get('coord_encoding_type', 'fourier').lower()  # Default to 'fourier'
        self.local_coord_type = config['model'].get('local_coord_type', '2d').lower()  # Default to '2d'
        self.use_decoder_coord_encoding = config['model'].get('use_decoder_coord_encoding', False)  # Default to false for backward compatibility
        self.decoder_mlp_depth = int(config['model'].get('decoder_mlp_depth', 2))  # Default to 2 for backward compatibility
        self.prediction_type = config['model'].get('prediction_type', 'v_pred').lower()  # Default to 'v_pred'
        self.t_eps = float(config['model'].get('t_eps', 1e-6))  # Epsilon for x-pred mode
        self.use_amp = config['model'].get('use_amp', False)  # Enable automatic mixed precision (BF16)
        self.compile_model = config['model'].get('compile', False)  # Enable torch.compile() for optimization
        self.multimodal_fusion_type = config['model'].get('multimodal_fusion_type', 'cross_attn').lower()
        self.use_era5_temporal_encoder = config['model'].get('use_era5_temporal_encoder', False)
        self.era5_temporal_out_channels = int(config['model'].get('era5_temporal_out_channels', 8))
        
        # Large model stability features for BF16 training
        self.qk_norm = config['model'].get('qk_norm', True)  # QK Layer Normalization (default: enabled for stability)
        self.use_layer_scale = config['model'].get('use_layer_scale', True)  # LayerScale (default: enabled for stability)
        self.layer_scale_init = float(config['model'].get('layer_scale_init', 1e-5))  # LayerScale initial value

        # Time-of-Day (tidal) conditioning (v1: adaLN embedding)
        self.use_tod_conditioning = config['model'].get('use_tod_conditioning', False)
        # Injection point for tidal time features: 'trunk' (all adaLN blocks, v3)
        # or 'decoder' (trunk stays time-free; only the output head sees time).
        self.tod_injection = config['model'].get('tod_injection', 'trunk').lower()
        if self.tod_injection not in ('trunk', 'decoder'):
            raise ValueError(f"tod_injection must be 'trunk' or 'decoder', got '{self.tod_injection}'")

        # ERA5 storm-intensity conditioning (extreme_feats injected with t_emb)
        self.use_extreme_conditioning = config['model'].get('use_extreme_conditioning', False)
        self.extreme_feats_dim = int(config['model'].get('extreme_feats_dim', 6))

        # Tidal elev encoding (v2: sin signal added to elev input channel)
        self.use_tidal_elev = config['model'].get('use_tidal_elev', False)

        # Per-channel loss weights (e.g. [hs, tp, dm_sin, dm_cos, elev] for 5ch).
        # None / empty = uniform weighting (previous behavior).
        self.channel_weights = config['training'].get('channel_weights', None)
        if self.channel_weights is not None:
            self.channel_weights = [float(w) for w in self.channel_weights]
            if len(self.channel_weights) != self.in_channels:
                raise ValueError(
                    f"channel_weights length {len(self.channel_weights)} != in_channels {self.in_channels}"
                )

        # Extreme sea-state loss up-weighting (training side):
        #   se *= 1 + extreme_loss_weight * relu(hs_norm - extreme_loss_thresh)
        self.extreme_loss_weight = float(config['training'].get('extreme_loss_weight', 0.0))
        self.extreme_loss_thresh = float(config['training'].get('extreme_loss_thresh', 2.5))
        # Asymmetric (Exloss-style) extra penalty on extreme-hs velocity
        # underestimates (0.0 = symmetric, legacy behavior). Scalar -> hs channel
        # only; list of len(in_channels) -> per-channel asymmetric factors.
        _asym = config['training'].get('extreme_asym', 0.0)
        self.extreme_asym = ([float(a) for a in _asym] if isinstance(_asym, (list, tuple))
                             else float(_asym))
        # Channels (0-based) to which the extreme gate weight applies (default: [0]).
        self.extreme_weight_channels = config['training'].get('extreme_weight_channels', None)
        if self.extreme_weight_channels is not None:
            self.extreme_weight_channels = [int(c) for c in self.extreme_weight_channels]
        # Student-t prior noise (t-Flow, arXiv:2410.14171): degrees of freedom nu.
        # None/absent = Gaussian prior (legacy). Scalar float (yaml `.inf` = Gaussian)
        # or per-channel list of len(in_channels). Applies to BOTH training loss
        # noise and sampling-time initial noise (kept consistent inside FlowDiffusion).
        _tdf = config['training'].get('t_noise_df', None)
        self.t_noise_df = ([float(nu) for nu in _tdf] if isinstance(_tdf, (list, tuple))
                           else (float(_tdf) if _tdf is not None else None))
        # Condition-gated heavy tail: dict {s_lo, s_hi} (ERA5 max-wind thresholds,
        # normalized [0,1] scale) or None = ungated (uniform t-prior everywhere).
        self.t_noise_gate = config['training'].get('t_noise_gate', None)
        if self.t_noise_gate is not None:
            self.t_noise_gate = {'s_lo': float(self.t_noise_gate['s_lo']),
                                 's_hi': float(self.t_noise_gate['s_hi'])}
        # ERA5 condition augmentation (training side): with probability
        # condition_aug_prob, add N(0, condition_aug_noise) to the ERA5 condition.
        self.condition_aug_noise = float(config['training'].get('condition_aug_noise', 0.0))
        self.condition_aug_prob = float(config['training'].get('condition_aug_prob', 0.0))

        # Sampling config
        self.total_samples = int(config['sampling']['total_samples'])
        self.sampling_steps = int(config['sampling']['steps'])
        self.sampling_batch_size = int(config['sampling'].get('batch_size', 10))  # Default to 10 if not specified
        self.sample_save_dir_name = config['sampling']['save_dir']  # Will be joined with work_dir later
        self.test_subset_ratio = float(config['sampling'].get('test_subset_ratio', 1.0))  # Default to 1.0 (use entire test set)
        self.sample_split = config['sampling'].get('sample_split', 'test')  # Default to 'test', can be 'train' or 'test'
        self.ensemble_size = int(config['sampling'].get('ensemble_size', 1))  # Number of samples per condition for empirical distribution
        self.resume_sample_path = config['sampling'].get('resume_sample_path', None)  # Path to resume sampling from
        self.use_ema_for_sampling = config['sampling'].get('use_ema_for_sampling', True)  # Whether to use EMA for sampling
        self.atm_source = config['sampling'].get('atm_source', 'era5')  # 'era5' or 'ens'
        self.ens_lead_time = int(config['sampling'].get('ens_lead_time', 24))  # IFS ENS lead time in hours
        self.ens_dir = config['sampling'].get('ens_dir', './data/ifs/ens')  # ENS data directory
        # Optional fixed forecast issue time ('YYYYMMDDTHH'): when set, every target
        # valid time uses the SAME init (lead = valid - init varies per timestamp),
        # i.e. one operational forecast cycle evaluated along its 0-60 h trajectory.
        # None (default) keeps the per-timestamp `valid - ens_lead_time` behaviour.
        self.ens_init = config['sampling'].get('ens_init', None)
        if self.ens_init is not None:
            self.ens_init = str(self.ens_init).strip()
            from datetime import datetime as _dt
            try:
                _dt.strptime(self.ens_init, '%Y%m%dT%H')
            except ValueError as exc:
                raise ValueError(
                    f"sampling.ens_init must look like '20250922T12', got "
                    f"{self.ens_init!r}") from exc
        self.gencast_dir = None  # GenCast is not part of this release
        # Member-forced ensemble sampling: generative sample k is conditioned on
        # atmospheric forecast member k (1:1 pairing) instead of all samples
        # sharing the ens-mean condition. Requires atm_source='ens' and
        # per-member forcing files (atm_members=ensemble_size on the dataset).
        self.member_forced = bool(config['sampling'].get('member_forced', False))
        # Number of atmospheric forcing members ON DISK. Defaults to ensemble_size
        # (1:1 member->sample). Set smaller to draw several generative samples per
        # atmospheric member (sample j uses member j % atm_members): e.g.
        # ensemble_size=100 + atm_members=10 = 10 draws per IFS ENS member.
        _am = config['sampling'].get('atm_members', None)
        self.atm_members = (int(_am) if _am not in (None, '', 0) else None)
        if self.member_forced and self.atm_source == 'era5':
            raise ValueError("sampling.member_forced requires sampling.atm_source='ens'")
        # Only sample these timestamps (e.g. ['20210101T00', ...]); None/empty = sample all.
        # Accepts a yaml list or a comma-separated string (for -o overrides).
        self.target_timestamps = self._parse_str_list(config['sampling'].get('target_timestamps', None))
        # Optionally snapshot intermediate ODE states at these times t in [0,1] (t=0 noise, t=1 sample)
        self.save_trajectory_times = self._parse_float_list(config['sampling'].get('save_trajectory_times', None))
        # Stochastic (SDE-style) sampling: noise amplitude per step; 0.0 = deterministic Euler ODE
        self.stochastic_gamma = float(config['sampling'].get('stochastic_gamma', 0.0))
        # sigma(t) schedule for stochastic sampling: 'const' or 'linear'
        self.stochastic_schedule = config['sampling'].get('stochastic_schedule', 'const')
        # Per-member Gaussian noise added to the ERA5 condition at sampling time
        # (simulates reanalysis uncertainty; 0.0 = all members share the same condition)
        self.condition_noise = float(config['sampling'].get('condition_noise', 0.0))
        # UQ baseline: ensemble members share ONE prior-noise draw per condition, so
        # all member-to-member spread comes from condition perturbation only
        # (input-perturbation ensemble). Requires condition_noise > 0 to have any effect.
        self.fix_prior_noise = bool(config['sampling'].get('fix_prior_noise', False))
        # MC-dropout sampling: keep dropout active during sampling (regression
        # baselines trained with model.dropout > 0)
        self.mc_dropout = bool(config['sampling'].get('mc_dropout', False))

        # Training objective (UQ method comparison):
        #   'cfm' - conditional flow matching (default)
        #   'mse' - deterministic regression (deep-ensemble members / MC dropout)
        #   'nll' - heteroscedastic Gaussian regression (mu, log_sigma heads)
        self.objective = config['training'].get('objective', 'cfm')
        # Model options for UQ baselines
        self.model_dropout = float(config['model'].get('dropout', 0.0))
        self.model_out_channels = config['model'].get('out_channels', None)
        
        # Checkpoint config
        self.resume_path = config['checkpoint'].get('resume_path', None)
        self.resume_training = config['checkpoint'].get('resume_training', False)
        self.load_optimizer = config['checkpoint'].get('load_optimizer', True)
        
        # Work directory
        self.work_dir = config['work_dir']['path']
        self.work_dir_base = config['work_dir'].get('base_dir', './')
        # Set prefix based on mode: 'train' for training, 'sample' for sampling
        default_prefix = 'train' if self.mode == 'train' else 'sample'
        prefix_from_config = config['work_dir'].get('prefix', None)
        self.work_dir_prefix = prefix_from_config if prefix_from_config else default_prefix
    
    @staticmethod
    def _parse_str_list(value):
        """Parse a yaml list or comma-separated string into a list of strings. None/empty -> None."""
        if value is None or value == '':
            return None
        if isinstance(value, str):
            items = [s.strip() for s in value.split(',') if s.strip()]
        else:
            items = [str(s).strip() for s in value if str(s).strip()]
        return items if items else None
    
    @classmethod
    def _parse_float_list(cls, value):
        """Parse a yaml list or comma-separated string into a list of floats in [0,1]. None/empty -> None."""
        items = cls._parse_str_list(value)
        if items is None:
            return None
        floats = [float(s) for s in items]
        for v in floats:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"save_trajectory_times values must be in [0, 1], got {v}")
        return floats
    
