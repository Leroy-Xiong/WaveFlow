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




from modules.training.mixin_config import TrainerConfigMixin
from modules.training.mixin_env import TrainerEnvMixin
from modules.training.mixin_saving import TrainerSavingMixin
from modules.training.mixin_sampling import TrainerSamplingMixin
from modules.training.mixin_model import TrainerModelMixin
from modules.training.main_utils import set_seed


class TrainerSampler(TrainerConfigMixin, TrainerEnvMixin, TrainerSavingMixin,
                 TrainerSamplingMixin, TrainerModelMixin):
    def __init__(self, config_path='./configs/configs.yaml', mode='train', config=None):
        """
        Initializes the trainer with GraphTransformer model, flow diffusion framework, and optimizer.
        
        Args:
            config_path: Path to the configuration file (default: './configs/configs.yaml')
            mode: Operating mode, either 'train' or 'sample' (default: 'train')
            config: Optional pre-loaded config dict (overrides config_path if provided)
        """
        # Store mode and config path
        self.mode = mode
        self.config_path = config_path
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)
            config = self.config
        
        # Extract configuration parameters
        self._extract_config_parameters(config)
        
        # Setup DDP environment and device
        self._setup_ddp()
        
        # Set random seed for reproducibility
        set_seed(self.seed)

        # Create work directory and setup logging (only on rank 0)
        if self.global_rank == 0:
            self._create_work_dir()
            self._setup_logging()
            
            # Log the resolved configuration (reflects -o overrides applied to self.config)
            try:
                config_content = yaml.safe_dump(self.config, default_flow_style=False, sort_keys=False)
                self.logger.info(f"Resolved configuration (source: {self.config_path}):\n{config_content}")
            except Exception as e:
                self.logger.error(f"Failed to dump configuration: {e}")
        
        # Broadcast work_dir from rank 0 to all other ranks
        if self.world_size > 1:
            # Create a list to hold the work_dir (only rank 0 has the correct value)
            work_dir_list = [self.work_dir] if self.global_rank == 0 else [None]
            dist.broadcast_object_list(work_dir_list, src=0)
            self.work_dir = work_dir_list[0]
        
        # Set sample save directory for all ranks
        self.sample_save_dir = os.path.join(self.work_dir, self.sample_save_dir_name)
        # Directory for intermediate ODE trajectory snapshots (kept separate from
        # sample_save_dir so post_process globbing of sample_*.npz is unaffected)
        self.trajectory_save_dir = os.path.join(self.work_dir, 'trajectory')
        
        # Load coordinates
        self._load_coordinates()
        
        # Initialize dataset first (needed for patch mode to get centroids)
        self._initialize_dataset_and_dataloader()
        
        # Initialize model and framework
        self._initialize_model()
        self._initialize_optimizer()
        
        # Initialize lr scheduler before loading checkpoint (needed for checkpoint loading)
        self._initialize_lr_scheduler()
        
        # Load checkpoint if specified
        self.start_epoch = 0
        self.start_batch = 0
        if self.resume_path:
            self._load_checkpoint()
    
    def __del__(self):
        """
        Destructor to clean up resources.
        Closes TensorBoard writer if it exists.
        """
        self.close()
    
    def close(self):
        """
        Explicitly close resources (TensorBoard writer, etc.).
        Safe to call multiple times.
        """
        try:
            if hasattr(self, 'tb_writer') and self.tb_writer is not None:
                self.tb_writer.close()
                self.tb_writer = None  # Prevent double close
                if hasattr(self, 'global_rank') and self.global_rank == 0:
                    if hasattr(self, 'logger') and self.logger is not None:
                        self.logger.info("TensorBoard writer closed.")
        except Exception:
            # Suppress exceptions during cleanup to prevent masking original errors
            pass
    
    def train(self):
        """
        Executes the training loop for unconditional generation.
        Uses epochs and log_interval from config.
        Saves checkpoints at the end of every N epochs (configurable via checkpoint_interval).
        Supports resuming from checkpoint if resume_training is True.
        """
        # Get checkpoint interval (in epochs) from config
        total_batches = len(self.dataloader)
        
        if self.global_rank == 0:
            self.logger.info(f"Total batches per epoch: {total_batches}")
            self.logger.info(f"Checkpoint will be saved every {self.checkpoint_interval} epoch(s)")
            if self.resume_training:
                self.logger.info(f"Resuming from epoch {self.start_epoch}")
        
        # Prevent CUDA cache purge at epoch boundaries.
        # When all batch tensors go out of scope between epochs, PyTorch's
        # CUDA caching allocator sees low memory pressure and releases its
        # entire cache (130 GB → 6 GB). By keeping the last batch alive as
        # a class attribute, we anchor the memory pool.
        self._batch_anchor = None
        
        # train_step no longer toggles module states every step (that walk
        # over all submodules costs measurable CPU per step) — set once here.
        self.model.train()

        for epoch in range(self.start_epoch, self.epochs):
            # Important: Set epoch for sampler to shuffle data differently each epoch
            self.sampler.set_epoch(epoch)

            # Create new iterator for this epoch's shuffled data
            dataloader_iter = iter(self.dataloader)

            epoch_loss = 0.0
            num_batches = 0
            step_start_time = time.time()
            log_loss_sum = 0.0  # Accumulate loss for logging
            log_grad_norm_sum = 0.0  # Accumulate grad norm for logging
            log_batch_count = 0  # Count batches for averaging

            # Initialize profiler for this epoch
            profiler = TrainingProfiler()
            profiler.start_epoch()

            # Inner loop to iterate through all batches in the dataset
            for batch_idx in range(total_batches):
                try:
                    batch_data = next(dataloader_iter)
                except StopIteration:
                    break

                # Profile: time spent waiting for dataloader
                profiler.record_dataloader_wait()

                # data is in batch_data['wave_data'] with shape [B, N, C]
                # non_blocking=True + pin_memory: H2D copies overlap with compute
                x = batch_data['wave_data'].to(self.device, non_blocking=True)  # [B, N, C]
                
                # Prepare ERA5 data based on fusion type
                condition_kwargs = {}
                # Land mask for grid model (True = valid ocean pixel)
                if 'valid_mask' in batch_data:
                    condition_kwargs['mask'] = batch_data['valid_mask'].to(self.device, non_blocking=True)
                if self.conditional_generation:
                    if self.region == 'global_grid':
                        # Global grid: model does pixel-level concat internally
                        era5_data = batch_data['era5'].to(self.device, non_blocking=True)
                        condition_kwargs['era5_data'] = era5_data
                    elif self.multimodal_fusion_type == 'direct_concat':
                        # Direct concat: use ERA5 data interpolated to wave nodes
                        era5_on_nodes = batch_data['era5_on_nodes'].to(self.device, non_blocking=True)
                        condition_kwargs['era5_on_nodes'] = era5_on_nodes
                    else:
                        # Cross attention: use ERA5 data on regular grid
                        era5_data = batch_data['era5'].to(self.device, non_blocking=True)
                        condition_kwargs['era5_data'] = era5_data

                # ERA5 condition augmentation: teach the model reanalysis uncertainty.
                # ERA5 is min-max normalized to [0,1]; clamp keeps conditions in-range.
                if self.conditional_generation and self.condition_aug_noise > 0.0 and \
                        self.condition_aug_prob > 0.0 and torch.rand(1).item() < self.condition_aug_prob:
                    if 'era5_data' in condition_kwargs:
                        condition_kwargs['era5_data'] = (condition_kwargs['era5_data'] + \
                            torch.randn_like(condition_kwargs['era5_data']) * self.condition_aug_noise).clamp(0.0, 1.0)
                    if 'era5_on_nodes' in condition_kwargs:
                        condition_kwargs['era5_on_nodes'] = (condition_kwargs['era5_on_nodes'] + \
                            torch.randn_like(condition_kwargs['era5_on_nodes']) * self.condition_aug_noise).clamp(0.0, 1.0)

                # Add time-of-day for tidal conditioning
                if self.use_tod_conditioning and 'tod' in batch_data:
                    condition_kwargs['tod'] = batch_data['tod'].to(self.device, non_blocking=True)

                # Add absolute-time tidal features (v3)
                if self.use_tod_conditioning and 'time_feats' in batch_data:
                    condition_kwargs['time_feats'] = batch_data['time_feats'].to(self.device, non_blocking=True)

                # Calculate training progress for ProRes (in epochs)
                # training_progress = current_epoch + fraction_of_current_epoch
                if self.use_prores:
                    training_progress = epoch + (batch_idx / total_batches)
                    # Pre-compute ProRes alpha values for all layers to avoid recompilation
                    # ProRes formula: alpha(l, t) = min(t / (T * l), 1)
                    # l is 1-indexed layer number, t is training progress in epochs
                    num_layers = len(self.model.blocks) if hasattr(self.model, 'blocks') else len(self.model.module.blocks)
                    # Vectorized computation: alpha(l, t) = min(t / (T * l), 1)
                    layer_nums = torch.arange(1, num_layers + 1, dtype=torch.float32, device=self.device)
                    prores_alphas = torch.clamp(training_progress / (self.prores_warmup_epochs * layer_nums), max=1.0)
                    condition_kwargs['prores_alphas'] = prores_alphas

                # Forward pass through the model with/without ERA5 condition
                # debug_grad=True to enable per-layer gradient monitoring
                profiler.record_gpu_start()
                loss, grad_norm, _ = self.fd.train_step(x, self.optimizer, debug_grad=False, **condition_kwargs)
                profiler.record_gpu_end()

                # Update EMA after each optimizer step
                if self.ema is not None:
                    self.ema.update()
                
                # Update learning rate scheduler if enabled
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                epoch_loss += loss
                num_batches += 1
                log_loss_sum += loss
                log_grad_norm_sum += grad_norm
                log_batch_count += 1
                
                # Log only on Rank 0
                if self.global_rank == 0 and (batch_idx + 1) % self.log_interval == 0:
                    elapsed_time = time.time() - step_start_time
                    steps_per_sec = self.log_interval / elapsed_time if elapsed_time > 0 else 0
                    # train_step returns detached GPU tensors; float() syncs only
                    # here (once per log_interval) instead of every step.
                    avg_loss = float(log_loss_sum / log_batch_count) if log_batch_count > 0 else 0
                    avg_grad_norm = float(log_grad_norm_sum / log_batch_count) if log_batch_count > 0 else 0
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(f"Epoch {epoch}, Step {batch_idx + 1}: Avg Loss = {avg_loss:.4f}, Grad Norm = {avg_grad_norm:.4f}, LR = {current_lr:.6f}, Steps/Sec = {steps_per_sec:.2f}")
                    
                    # Write loss to TensorBoard
                    global_step = epoch * 1000 + int((batch_idx + 1) / total_batches * 1000)
                    self.tb_writer.add_scalar('Loss/Train', avg_loss, global_step)
                    self.tb_writer.add_scalar('Gradient/Norm', avg_grad_norm, global_step)
                    self.tb_writer.add_scalar('Steps_per_sec', steps_per_sec, global_step)
                    self.tb_writer.add_scalar('Learning_Rate', current_lr, global_step)
                    
                    step_start_time = time.time()
                    # Reset accumulators for next logging interval
                    log_loss_sum = 0.0
                    log_grad_norm_sum = 0.0
                    log_batch_count = 0

                # Keep last batch tensors alive to anchor CUDA memory pool.
                # Prevents the caching allocator from purging 130 GB → 6 GB
                # at epoch boundaries when all batch data goes out of scope.
                self._batch_anchor = (x, condition_kwargs)

                # Track when this batch finished for dataloader wait time measurement
                profiler.mark_batch_end()

            # Synchronize processes (optional, but good for timing/logging)
            dist.barrier()
            
            # Logging average epoch loss (Rank 0 only)
            if self.global_rank == 0:
                avg_epoch_loss = float(epoch_loss / num_batches) if num_batches > 0 else float(epoch_loss)
                self.logger.info(f"Epoch {epoch}: Average Loss (Rank 0) = {avg_epoch_loss:.4f}, Batches processed = {num_batches}")

                # Profiling: report dataloader wait vs GPU compute ratio
                profiler.log_epoch_stats(self.logger, epoch, tb_writer=self.tb_writer)
                
                # Write epoch average loss to TensorBoard
                self.tb_writer.add_scalar('Loss/Train_Epoch_Avg', avg_epoch_loss, epoch)
            
            # Save checkpoint at the end of epoch if it matches the interval
            if (epoch + 1) % self.checkpoint_interval == 0:
                # Synchronize before saving
                dist.barrier()
                self._save_checkpoint(epoch)
                dist.barrier()
    
    def sample(self):
        """
        Generates samples from the trained model using DDP parallel sampling with batch control.
        Each GPU generates a portion of the total samples in batches to control GPU memory usage.
        Each individual sample is saved as a separate .npz file.
        
        By default, uses EMA model weights if EMA is enabled. This provides more stable
        and higher quality samples compared to the online model weights.
        
        Batch size is read from config: sampling.batch_size
        Adjust this in configs.yaml based on available GPU memory
        
        Sample naming: sample_{global_idx}.npz where global_idx is the global sample index
        
        For conditional generation: Iterates through the entire test dataset, generating one sample per test data
        For unconditional generation: Generates the number of samples specified in config (sampling.total_samples)
        """
        # Apply EMA weights for sampling if enabled and configured to use EMA
        ema_backup = None
        use_ema = (self.ema is not None and self.use_ema_for_sampling)
        
        if use_ema:
            ema_backup = self.ema.apply_shadow()
            if self.global_rank == 0:
                self.logger.info("Using EMA model weights for sampling")
        else:
            if self.global_rank == 0:
                if self.ema is None:
                    self.logger.info("EMA not enabled, using online model weights for sampling")
                elif not self.use_ema_for_sampling:
                    self.logger.info("use_ema_for_sampling is false, using online model weights for sampling")
        
        try:
            self._sample_impl()
        finally:
            # Restore original weights after sampling (important for continued training)
            if ema_backup is not None:
                self.ema.restore(ema_backup)
                if self.global_rank == 0:
                    self.logger.info("Restored original model weights after sampling")
    
