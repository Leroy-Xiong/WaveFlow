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
    PatchedGraphTransformer_Tiny,
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




class TrainerModelMixin:
    def _initialize_model(self):
        """
        Initialize the GraphTransformer or PatchedGraphTransformer model based on configuration.
        Also initializes the Flow Diffusion Framework.
        
        Model selection:
        - If use_patches=False: Uses GraphTransformer_Base with node coordinates
        - If use_patches=True: Uses PatchedGraphTransformer_Base with patch centroids
        """
        if self.region == 'global_grid':
            # ---- ViT on global 0.25° regular grid ----
            from modules.vit_global_wave import ViTGlobalWave_Huge, ViTGlobalWave_Large, ViTGlobalWave_Base

            vit_map = {
                'vit_huge': ViTGlobalWave_Huge,
                'vit_large': ViTGlobalWave_Large,
                'vit_base': ViTGlobalWave_Base,
            }
            vit_fn = vit_map.get(self.model_type, ViTGlobalWave_Huge)
            img_size = tuple(self.grid_shape) if self.grid_shape else (720, 1440)
            patch_size = int(self.config['model'].get('patch_size', 16))

            self.model = vit_fn(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                era5_num_frames=self.era5_num_frames,
                qk_norm=self.qk_norm,
                use_layer_scale=self.use_layer_scale,
                layer_scale_init=self.layer_scale_init,
            )

            if self.global_rank == 0:
                n_patches = ((img_size[0] + patch_size - 1) // patch_size) * ((img_size[1] + patch_size - 1) // patch_size)
                self.logger.info(f"Initialized ViTGlobalWave_{self.model_type} on {img_size} grid")
                self.logger.info(f"  - Patch size {patch_size} → {n_patches} patches")
                self.logger.info(f"  - Total params: {sum(p.numel() for p in self.model.parameters()):,}")
        elif self.region == 'hk_grid':
            # ---- ViT on regular grid ----
            from modules.vit_wave import ViTWave_Base, ViTWave_Small, ViTWave_Large

            vit_map = {
                'vit_small': ViTWave_Small,
                'vit_base': ViTWave_Base,
                'vit_large': ViTWave_Large,
            }
            vit_fn = vit_map.get(self.model_type, ViTWave_Base)
            img_size = tuple(self.grid_shape) if self.grid_shape else (75, 107)
            patch_size = int(self.config['model'].get('patch_size', 4))

            self.model = vit_fn(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                era5_num_frames=self.era5_num_frames,
                era5_encoder_type=self.era5_encoder_type,
                qk_norm=self.qk_norm,
                use_layer_scale=self.use_layer_scale,
                use_tod_conditioning=self.use_tod_conditioning,
                tod_injection=self.tod_injection,
            )

            if self.global_rank == 0:
                n_patches = ((img_size[0] + patch_size - 1) // patch_size) * ((img_size[1] + patch_size - 1) // patch_size)
                self.logger.info(f"Initialized ViTWave_{self.model_type} on {img_size} grid")
                self.logger.info(f"  - Patch size {patch_size} → {n_patches} patches")
                self.logger.info(f"  - Total params: {sum(p.numel() for p in self.model.parameters()):,}")
        elif self.use_patches:
            # For patch mode, we need patch centroids, local_coords_template, and mask_template
            if not hasattr(self, 'patch_centroids'):
                raise ValueError(
                    "Cluster data not loaded. "
                    "Make sure use_patches=True and cluster_file is specified in config."
                )
            
            patch_centroids = self.patch_centroids  # [n_clusters, 2]
            local_coords_template = self.local_coords_template  # [n_clusters, max_points, 2]
            mask_template = self.mask_template  # [n_clusters, max_points]
            cluster_labels = self.cluster_labels  # [num_nodes]
            
            # Select model based on config
            model_map = {
                'tiny': PatchedGraphTransformer_Tiny,
                'small': PatchedGraphTransformer_Small,
                'base': PatchedGraphTransformer_Base,
                'large': PatchedGraphTransformer_Large,
                'huge': PatchedGraphTransformer_Huge,
                'giant': PatchedGraphTransformer_Giant
            }
            
            if self.model_type not in model_map:
                raise ValueError(
                    f"Invalid model_type '{self.model_type}'. "
                    f"Must be one of: {list(model_map.keys())}"
                )
            
            model_fn = model_map[self.model_type]
            
            # Initialize PatchedGraphTransformer model
            self.model = model_fn(
                patch_centroids=patch_centroids,
                local_coords_template=local_coords_template,
                mask_template=mask_template,
                cluster_labels=cluster_labels,
                in_channels=self.in_channels,
                use_cross_attn=self.conditional_generation,
                era5_encoder_type=self.era5_encoder_type,
                era5_patch_size=self.era5_patch_size,
                era5_num_frames=self.era5_num_frames,
                coord_encoding_type=self.coord_encoding_type,
                use_decoder_coord_encoding=self.use_decoder_coord_encoding,
                decoder_mlp_depth=self.decoder_mlp_depth,
                multimodal_fusion_type=self.multimodal_fusion_type,
                use_era5_temporal_encoder=self.use_era5_temporal_encoder,
                era5_temporal_out_channels=self.era5_temporal_out_channels,
                use_prores=self.use_prores,
                prores_warmup_epochs=self.prores_warmup_epochs,
                use_tod_conditioning=self.use_tod_conditioning,
                tod_injection=self.tod_injection,
                use_extreme_conditioning=self.use_extreme_conditioning,
                extreme_feats_dim=self.extreme_feats_dim,
                # Large model stability features
                qk_norm=self.qk_norm,
                use_layer_scale=self.use_layer_scale,
                layer_scale_init=self.layer_scale_init,
                # UQ baselines: dropout (MC dropout) and out_channels (NLL mu/log_sigma)
                dropout=self.model_dropout,
                out_channels=self.model_out_channels,
            )
            
            if self.global_rank == 0:
                self.logger.info(f"Initialized PatchedGraphTransformer_{self.model_type.capitalize()} with {len(patch_centroids)} patches")
                self.logger.info(f"  - Local coords template shape: {local_coords_template.shape}")
                self.logger.info(f"  - Mask template shape: {mask_template.shape}")
                self.logger.info(f"  - Multimodal fusion type: {self.multimodal_fusion_type}")
                self.logger.info(f"  - Conditional generation: {self.conditional_generation}")
                if self.use_prores:
                    self.logger.info(f"  - ProRes enabled: warmup_epochs={self.prores_warmup_epochs}")
                if self.use_tod_conditioning:
                    self.logger.info(f"  - Time-of-Day conditioning: enabled")
                if self.use_extreme_conditioning:
                    self.logger.info(f"  - Extreme conditioning: enabled")
                # Log stability features
                self.logger.info(f"  - QK Normalization: {self.qk_norm}")
                self.logger.info(f"  - LayerScale: {self.use_layer_scale} (init={self.layer_scale_init})")
        else:
            # For traditional mode, use node coordinates
            self.model = GraphTransformer_Base(
                node_coords=self.coords,
                in_channels=self.in_channels,
                use_cross_attn=self.conditional_generation,
                era5_encoder_type=self.era5_encoder_type,
                era5_patch_size=self.era5_patch_size,
                era5_num_frames=self.era5_num_frames,
                coord_encoding_type=self.coord_encoding_type,
                use_prores=self.use_prores,
                prores_warmup_epochs=self.prores_warmup_epochs,
                use_tod_conditioning=self.use_tod_conditioning,
                tod_injection=self.tod_injection,
                use_extreme_conditioning=self.use_extreme_conditioning,
                extreme_feats_dim=self.extreme_feats_dim,
                qk_norm=self.qk_norm,
                use_layer_scale=self.use_layer_scale,
                layer_scale_init=self.layer_scale_init
            )
            
            if self.global_rank == 0:
                self.logger.info(f"Initialized GraphTransformer_Base with {len(self.coords)} nodes")
                self.logger.info(f"  - QK Normalization: {self.qk_norm}")
                self.logger.info(f"  - LayerScale: {self.use_layer_scale} (init={self.layer_scale_init})")
                if self.use_prores:
                    self.logger.info(f"  - ProRes enabled: warmup_epochs={self.prores_warmup_epochs}")
                if self.use_tod_conditioning:
                    self.logger.info(f"  - Time-of-Day conditioning: enabled")

        # ── Parameter logging (BEFORE compile & DDP, for clean module structure) ──
        if self.global_rank == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            # Per-module parameter breakdown
            self.logger.info("Parameter breakdown by module:")
            module_params = {}
            for name, param in self.model.named_parameters():
                top_module = name.split('.')[0]
                if top_module not in module_params:
                    module_params[top_module] = {'total': 0, 'trainable': 0}
                module_params[top_module]['total'] += param.numel()
                if param.requires_grad:
                    module_params[top_module]['trainable'] += param.numel()
            for mod_name, counts in sorted(module_params.items()):
                self.logger.info(f"  {mod_name}: {counts['total']:,} params ({counts['trainable']:,} trainable)")
            # Log key scalar learnable parameters (e.g. tod_scale)
            self.logger.info("Key learnable scalars/parameters:")
            scalar_count = 0
            for name, param in self.model.named_parameters():
                if param.numel() == 1:
                    self.logger.info(f"  {name}: shape={list(param.shape)}, value={param.data.item():.6f}, requires_grad={param.requires_grad}")
                    scalar_count += 1
            if scalar_count == 0:
                self.logger.info("  (no scalar parameters found)")

        # Compile model with torch.compile() if enabled (before DDP wrapping)
        if self.compile_model:
            # compile_mode is configurable: 'default' compiles in ~3 min;
            # 'max-autotune-no-cudagraphs' tunes GEMMs (first compile 20-60 min,
            # then cached) for extra GPU-side throughput.
            # CUDA graphs (mode='reduce-overhead') cause "backward through graph
            # a second time" errors with this model's autograd patterns.
            compile_mode = self.config['model'].get('compile_mode', 'default')
            if self.global_rank == 0:
                self.logger.info(f"Compiling model with torch.compile(mode='{compile_mode}')...")
            self.model = torch.compile(self.model, mode=compile_mode)
        
        # Wrap model with DDP
        # gradient_as_bucket_view=False prevents "Grad strides do not match bucket
        # view strides" warnings when torch.compile changes tensor strides.
        self.model = DDP(self.model.to(self.device), device_ids=[self.local_rank],
                         gradient_as_bucket_view=False, broadcast_buffers=False)

        # Initialize the Flow Diffusion Framework
        self.fd = FlowDiffusion(model=self.model, device=self.device,
                                prediction_type=self.prediction_type,
                                t_eps=self.t_eps,
                                use_amp=self.use_amp,
                                grad_clip_norm=self.grad_clip_norm,
                                channel_weights=self.channel_weights,
                                extreme_loss_weight=self.extreme_loss_weight,
                                extreme_loss_thresh=self.extreme_loss_thresh,
                                extreme_asym=self.extreme_asym,
                                extreme_weight_channels=self.extreme_weight_channels,
                                t_noise_df=self.t_noise_df,
                                t_noise_gate=self.t_noise_gate,
                                objective=self.objective,
                                mc_dropout=self.mc_dropout)
        
        # Initialize EMA if enabled
        self.ema = None
        if self.use_ema:
            self.ema = EMA(
                model=self.model,
                decay=self.ema_decay,
                update_after_step=self.ema_update_after_step,
                device=self.device
            )
            if self.global_rank == 0:
                self.logger.info(f"EMA initialized with decay={self.ema_decay}, update_after_step={self.ema_update_after_step}")
    
    def _initialize_optimizer(self):
        """
        Initialize the optimizer for model training.
        """
        # fused=True: single multi-tensor kernel per step instead of one kernel
        # per parameter — significantly fewer kernel launches for large models.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            fused=torch.cuda.is_available()
        )
    
    def _initialize_lr_scheduler(self):
        """
        Initialize the learning rate scheduler with optional warmup.
        
        Supports:
        - Linear warmup + Cosine annealing (default)
        - Linear warmup + Step decay
        - Linear warmup only (no decay)
        
        The warmup can be specified in epochs or steps.
        """
        # Calculate total training steps
        # NOTE: the LR schedule horizon is decoupled from `training.epochs` via
        # `training.scheduler_epochs` (default: same as epochs, i.e. previous
        # behavior). Setting scheduler_epochs > epochs makes training use the
        # long-run schedule and simply stop early — runs with fewer epochs stay
        # LR-comparable with full-length runs.
        total_batches = len(self.dataloader)
        total_steps = total_batches * self.scheduler_epochs
        
        # Calculate warmup steps from warmup_epochs
        warmup_steps = self.warmup_epochs * total_batches
        
        # Ensure warmup doesn't exceed total steps
        warmup_steps = min(warmup_steps, total_steps)
        
        # Cosine annealing minimum LR ratio (e.g., 0.01 means LR won't drop below 1% of initial LR)
        min_lr_ratio = 0.01
        
        if self.scheduler_type == 'cosine':
            if warmup_steps == 0:
                # No warmup: use CosineAnnealingLR directly from the beginning
                self.lr_scheduler = lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps,  # Number of steps for cosine annealing
                    eta_min=self.learning_rate * min_lr_ratio  # Minimum learning rate
                )
            else:
                # Use SequentialLR to combine LinearLR warmup with CosineAnnealingLR
                # Warmup phase: LinearLR from 0 to initial_lr
                warmup_scheduler = lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.01,  # Start from 1% of initial_lr
                    end_factor=1.0,      # End at 100% of initial_lr
                    total_iters=warmup_steps
                )
                
                # Cosine annealing phase: decay from initial_lr to eta_min
                cosine_scheduler = lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps - warmup_steps,  # Number of steps for cosine annealing
                    eta_min=self.learning_rate * min_lr_ratio  # Minimum learning rate
                )
                
                self.lr_scheduler = lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps]
                )
        else:
            # Fallback: use LambdaLR for other scheduler types
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # Linear warmup: lr = current_step / warmup_steps * base_lr
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    # After warmup: apply main scheduler
                    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    if self.scheduler_type == 'step':
                        # Step decay: reduce by 0.1 every 30% of remaining steps
                        decay_factor = 0.1 ** (int(progress * 10) // 3)
                        return decay_factor
                    else:
                        # No decay (constant after warmup)
                        return 1.0
            
            self.lr_scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        if self.global_rank == 0:
            self.logger.info(f"Learning rate scheduler initialized:")
            if warmup_steps > 0:
                self.logger.info(f"  - Warmup: enabled ({warmup_steps} steps, {self.warmup_epochs} epochs)")
            else:
                self.logger.info(f"  - Warmup: disabled")
            self.logger.info(f"  - Total steps: {total_steps}")
            self.logger.info(f"  - Scheduler type: {self.scheduler_type}")
            self.logger.info(f"  - Initial learning rate: {self.learning_rate}")
            if self.scheduler_type == 'cosine':
                self.logger.info(f"  - Min LR ratio: {min_lr_ratio} (eta_min={self.learning_rate * min_lr_ratio:.2e})")
    
    def _initialize_dataset_and_dataloader(self):
        """
        Initialize the dataset, distributed sampler, and dataloader.
        This method combines:
        1. Dataset initialization with normalization transform
        2. Distributed sampler creation for multi-GPU training/sampling
        3. DataLoader creation with proper configuration
        
        Behavior depends on self.mode:
        - 'train': Initialize training dataset with shuffle=True
        - 'sample': Initialize dataset based on self.sample_split (default: 'test')
                    with shuffle=False for reproducibility
        
        ERA5 data loading is controlled by self.conditional_generation:
        - True: Load ERA5 data for conditional generation
        - False: Don't load ERA5 data for unconditional generation
        
        Patch mode is controlled by self.use_patches (from config):
        - True: Organize data into patches for PatchedGraphTransformer
        - False: Use traditional grid/node format (default)
        
        For sampling mode, you can set sample_split='train' in config to test
        model performance on the training set.
        """
        # Determine dataset split and batch size based on mode
        if self.mode == 'sample':
            split = self.sample_split  # Use configured split ('train' or 'test')
            batch_size = self.sampling_batch_size
            shuffle = False  # Don't shuffle for reproducibility in sampling
        else:  # train mode
            split = 'train'
            batch_size = self.batch_size
            shuffle = True
        
        # Initialize the dataset with normalization transform
        transform = NormalizeWaveData(
            stats_file=self.stats_file,
            variables=self.variables,
            era5_stats_file=self.era5_stats_file,
            normalize_era5=self.conditional_generation,
            use_tidal_elev=self.use_tidal_elev,
            log_power=self.config['data'].get('log_power', 1.0)
        )
        
        # Store transform for denormalization during sampling
        self.transform = transform

        # Determine if we need to interpolate ERA5 to wave nodes
        interpolate_era5_to_nodes = (self.conditional_generation and
                                     self.multimodal_fusion_type == 'direct_concat')

        # ENS/GenCast source only applies during sampling, training always uses ERA5
        atm_source = self.atm_source if self.mode == 'sample' else 'era5'

        if self.region == 'global_grid':
            # ---- Global 0.25° regular grid dataset ----
            from modules.datasets.global_grid_wave_dataset import GlobalGridWaveDataset, collate_global_grid_samples
            self.dataset = GlobalGridWaveDataset(
                wave_dir=self.data_dir,
                era5_dir=self.era5_dir,
                stats_file=self.stats_file,
                era5_stats_file=self.era5_stats_file,
                land_mask_file=self.land_mask_file,
                split=split,
                num_era5_frames=self.era5_num_frames,
                grid_shape=tuple(self.grid_shape),
                file_format=self.config['data'].get('file_format', 'npy'),
            )
            self._collate_fn = collate_global_grid_samples
        elif self.region == 'hk_grid':
            # ---- Grid dataset (regular grid, preinterpolated .npy files) ----
            from modules.datasets.grid_wave_dataset import GridWaveDataset, collate_grid_samples
            self.dataset = GridWaveDataset(
                grid_dir=self.data_dir,
                era5_dir=self.era5_dir,
                stats_file=self.stats_file,
                era5_stats_file=self.era5_stats_file,
                variables=self.variables,
                split=split,
                load_era5=self.conditional_generation,
                era5_region=self.era5_region,
                num_era5_frames=self.era5_num_frames,
                era5_file_format=self.era5_file_format,
            )
            self._collate_fn = collate_grid_samples
        else:
            self.dataset = WaveDataset(
                data_dir=self.data_dir,
                era5_dir=self.era5_dir,
                variables=self.variables,
                load_era5=self.conditional_generation,
                split=split,
                transform=transform,
                era5_region=self.era5_region,
                num_era5_frames=self.era5_num_frames,
                region=self.region,
                file_format=self.file_format,
                train_max_year=self.config['data'].get('train_max_year'),
                era5_file_format=self.era5_file_format,
                coord_file_path=self.coord_file_path if interpolate_era5_to_nodes else None,
                interpolate_era5_to_nodes=interpolate_era5_to_nodes,
                use_preinterpolated_era5=self.use_preinterpolated_era5,
                atm_source=atm_source,
                ens_lead_time=self.ens_lead_time,
                ens_dir=self.ens_dir,
                gencast_dir=self.gencast_dir,
                ens_init=self.ens_init,
                atm_members=((self.atm_members or self.ensemble_size)
                             if (self.member_forced and self.mode == 'sample') else 1),
                use_mmap=self.use_mmap,
                oversample_alpha=self.config['dataset'].get('oversample_alpha', 0.0),
                oversample_tau=self.config['dataset'].get('oversample_tau', 3.0),
                oversample_weights_file=self.config['dataset'].get(
                    'oversample_weights_file', './data/hindcast/train_ts_max_hs.npz'),
            )
        
        # If in sampling mode, create a subset based on test_subset_ratio
        if self.mode == 'sample' and self.test_subset_ratio < 1.0:
            total_samples = len(self.dataset)
            subset_size = int(total_samples * self.test_subset_ratio)
            if subset_size < 1:
                subset_size = 1  # At least one sample
            
            # Randomly sample subset_size samples from the dataset
            np.random.seed(42)  # Set seed for reproducibility
            random_indices = np.random.choice(total_samples, subset_size, replace=False)
            random_indices = sorted(random_indices)  # Sort for better file access patterns
            self.dataset.file_list = [self.dataset.file_list[i] for i in random_indices]
            
            if self.global_rank == 0:
                self.logger.info(f"Randomly sampled {subset_size} samples from {split} set (ratio: {self.test_subset_ratio:.2%})")
        if self.global_rank == 0:
            self.logger.info(f"Dataset initialized with {len(self.dataset)} samples (mode: {self.mode}, split: {split})")
            self.logger.info(f"Patch mode: {self.use_patches}")
            if self.use_patches:
                self.logger.info(f"Cluster file: {self.cluster_file}")
                self.logger.info(f"Number of patches: {self.n_clusters}")
                self.logger.info(f"Max points per patch: {self.local_coords_template.shape[1]}")
        
        # Initialize the distributed sampler for multi-GPU training/sampling
        # This ensures each GPU sees a different subset of the data
        self.sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=shuffle
        )
        
        # Create data loader with the DistributedSampler
        # Configure prefetch_factor and persistent_workers for better data loading performance
        dataloader_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': False,  # Shuffle must be False when using DistributedSampler
            'sampler': self.sampler,
            'collate_fn': getattr(self, '_collate_fn', collate_wave_samples),
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
        }
        
        # prefetch_factor requires num_workers > 0
        if self.num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = self.prefetch_factor
        
        self.dataloader = DataLoader(**dataloader_kwargs)
        
        if self.global_rank == 0:
            self.logger.info(f"DataLoader configured: num_workers={self.num_workers}, "
                           f"prefetch_factor={dataloader_kwargs.get('prefetch_factor', 'N/A')}, "
                           f"persistent_workers={self.persistent_workers}")
    
    def _load_checkpoint(self):
        """
        Load model checkpoint for resuming training or sampling.
        
        Loads:
        - Model state dict
        - Optimizer state dict (if resume_training is True and load_optimizer is True)
        - Training progress (epoch)
        - EMA state dict (if available and EMA is enabled)
        """
        if not os.path.exists(self.resume_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.resume_path}")
        
        if self.global_rank == 0:
            self.logger.info(f"Loading checkpoint from: {self.resume_path}")
        
        # Load checkpoint to CPU first to avoid GPU memory issues
        # weights_only=False for compatibility with PyTorch 2.6+ and numpy scalar types
        checkpoint = torch.load(self.resume_path, map_location='cpu', weights_only=False)
        
        # Load model state dict, handling _orig_mod. prefix from torch.compile
        model_state = checkpoint['model_state_dict']
        model_keys = set(self.model.module.state_dict().keys())
        ckpt_keys = set(model_state.keys())
        if not ckpt_keys & model_keys and all(k.startswith('_orig_mod.') for k in ckpt_keys):
            model_state = {k.removeprefix('_orig_mod.'): v for k, v in model_state.items()}
        elif not ckpt_keys & model_keys and all(not k.startswith('_orig_mod.') for k in ckpt_keys):
            model_state = {f'_orig_mod.{k}': v for k, v in model_state.items()}
        # Shape-adaptive loading: keep parameters whose shape matches the current
        # model (e.g. warm-starting an experiment that changed extreme_feats_dim);
        # mismatched keys are skipped and keep their random initialization.
        cur_state = self.model.module.state_dict()
        mismatched = [k for k, v in model_state.items() if k in cur_state and cur_state[k].shape != v.shape]
        if mismatched:
            model_state = {k: v for k, v in model_state.items() if k not in mismatched}
            if self.global_rank == 0:
                self.logger.info(f"Shape mismatch, skipping {len(mismatched)} param(s): {mismatched[:8]}")
        # strict=False: mismatched keys were dropped above (warm-start with a
        # changed architecture keeps random init for the new parameters).
        self.model.module.load_state_dict(model_state, strict=False)
        
        if self.global_rank == 0:
            self.logger.info("Model state loaded successfully")
        
        # Load EMA state if available and EMA is enabled
        if self.ema is not None and 'ema_state_dict' in checkpoint:
            ema_state = checkpoint['ema_state_dict']
            # Handle _orig_mod. prefix mismatch in EMA shadow keys
            if 'shadow' in ema_state:
                model_param_names = set(n for n, _ in self.ema.model.named_parameters())
                shadow_keys = set(ema_state['shadow'].keys())
                if not shadow_keys & model_param_names and all(k.startswith('_orig_mod.') for k in shadow_keys):
                    ema_state['shadow'] = {k.removeprefix('_orig_mod.'): v for k, v in ema_state['shadow'].items()}
                # Shape-adaptive EMA load: drop shadow entries whose shape no
                # longer matches the current model (warm-start with changed
                # architecture); they are re-synced from the loaded weights below.
                cur_params = dict(self.ema.model.named_parameters())
                ema_state['shadow'] = {k: v for k, v in ema_state['shadow'].items()
                                       if k in cur_params and cur_params[k].shape == v.shape}
                if len(ema_state['shadow']) < len(cur_params):
                    if self.global_rank == 0:
                        self.logger.info(
                            f"EMA shadow shape mismatch, re-syncing "
                            f"{len(cur_params) - len(ema_state['shadow'])} param(s) from loaded model")
            self.ema.load_state_dict(ema_state)
            # Re-sync any shadow entries that were dropped (shape mismatch) from
            # the freshly-loaded model weights so EMA == online model for those.
            for name, param in self.ema.model.named_parameters():
                if param.requires_grad and name not in self.ema.shadow:
                    self.ema.shadow[name] = param.data.clone().detach()
            if self.global_rank == 0:
                self.logger.info(f"EMA state loaded successfully (step={self.ema.step})")
        elif self.ema is not None:
            # EMA enabled but no state in checkpoint.
            # The EMA shadow was created in _initialize_model() from random-init weights
            # BEFORE the checkpoint was loaded, so it would overwrite the trained weights
            # with random noise on apply_shadow(). Re-sync shadow to the freshly-loaded
            # model weights so EMA == online model for inference.
            ema_model = self.ema.model
            for name, param in ema_model.named_parameters():
                if param.requires_grad and name in self.ema.shadow:
                    self.ema.shadow[name] = param.data.clone().detach().to(self.ema.shadow[name].device)
            if self.global_rank == 0:
                self.logger.info("No EMA state in checkpoint, re-synced EMA shadow from loaded model weights")
        
        # Load optimizer state and training progress if resuming training
        if self.resume_training:
            resumed_epoch = checkpoint.get('epoch', 0) + 1
            
            if self.load_optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.global_rank == 0:
                    self.logger.info("Optimizer state loaded successfully")
                
                # Load scheduler state if available (only when loading optimizer)
                if self.lr_scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    if self.global_rank == 0:
                        self.logger.info("Scheduler state loaded successfully")
                
                # Set starting epoch from checkpoint
                self.start_epoch = resumed_epoch
                if self.global_rank == 0:
                    self.logger.info(f"Resuming training from epoch {self.start_epoch} (with optimizer state)")
            else:
                # Not loading optimizer: restart LR schedule from beginning
                # but keep track of resumed epoch for logging
                self.start_epoch = 0
                self.resumed_from_epoch = resumed_epoch
                if self.global_rank == 0:
                    self.logger.info(f"Resuming training from epoch {resumed_epoch} (LR schedule restarted, warmup will be from epoch {resumed_epoch} to {resumed_epoch + self.warmup_epochs})")
        else:
            if self.global_rank == 0:
                self.logger.info("Checkpoint loaded for sampling/inference (not resuming training)")
    
    def _save_checkpoint(self, epoch):
        """
        Save model checkpoint at the end of an epoch. Only saves on rank 0.
        
        Args:
            epoch: Current epoch number
        """
        if self.global_rank != 0:
            return
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.work_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create checkpoint filename with epoch+1
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),  # Use .module for DDP model
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        # Save scheduler state if available
        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        # Save EMA state if enabled
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

