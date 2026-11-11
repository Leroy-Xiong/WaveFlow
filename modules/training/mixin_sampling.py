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




class TrainerSamplingMixin:
    def _load_member_conditions(self, needed_meta):
        """
        Member-forced ensemble conditions (sampling.member_forced=True).

        needed_meta = [(ts, j), ...] in needed_indices order; generative sample j
        of timestamp ts is conditioned on atmospheric forecast member j. Fetches
        each member's forcing through the dataset (same frame plan and
        normalization as the DataLoader path) and stacks to the subset layout
        [len(needed_meta), ...] that era5_data_repeated[needed_indices] would give.

        Returns:
            torch.Tensor: Condition batch on self.device.
        """
        key = ('era5_on_nodes'
               if self.multimodal_fusion_type == 'direct_concat' else 'era5')
        conds = []
        n_mem = max(1, int(self.dataset.atm_members))
        for ts, j in needed_meta:
            # ensemble_size may exceed the number of atmospheric members: rotate
            # through the members so every sample still sees a real forecast member
            member = j % n_mem
            try:
                cond = self.dataset.load_member_condition(ts, member)
            except Exception as sub_e:
                try:
                    paths = self.dataset._build_ens_frame_plan(ts, member)
                except Exception:
                    paths = None
                raise RuntimeError(
                    f'load_member_condition failed: ts={ts} member={j} paths={paths}: {sub_e!r}'
                ) from sub_e
            if self.dataset.transform is not None:
                cond = self.dataset.transform({key: np.asarray(cond)})[key]
            conds.append(torch.from_numpy(np.ascontiguousarray(cond)))
        return torch.stack(conds, dim=0).to(self.device, non_blocking=True)

    def _run_conditional_sampling(self, num_nodes, samples_per_gpu):
        """
        Run conditional sampling by iterating through test dataset.
        Handles both traditional mode and patch mode.
        For each condition, generate ensemble_size samples for computing empirical distribution.
        Uses asynchronous CPU saving to avoid blocking GPU sampling.
        
        Args:
            num_nodes: Number of nodes in the graph (for traditional mode)
            samples_per_gpu: Number of batches this GPU will process
            
        Returns:
            total_saved: Total number of samples saved
        """
        total_saved = 0
        total_batches_processed = 0
        total_samples_generated = 0
        save_lock = threading.Lock()
        target_set = set(self.target_timestamps) if self.target_timestamps else None
        matched_targets = set()
        
        if self.global_rank == 0:
            self.logger.info(f"[Rank {self.global_rank}] Starting conditional sampling with {samples_per_gpu} batches")
            self.logger.info(f"[Rank {self.global_rank}] Ensemble size: {self.ensemble_size}")
            self.logger.info(f"[Rank {self.global_rank}] Save directory: {self.sample_save_dir}")
            if target_set is not None:
                self.logger.info(f"[Rank {self.global_rank}] Target timestamps filter enabled: {len(target_set)} timestamps requested")
            if self.save_trajectory_times:
                self.logger.info(f"[Rank {self.global_rank}] Saving ODE trajectory at t={self.save_trajectory_times} to {self.trajectory_save_dir}")
        
        # Verify save directory exists
        if not os.path.exists(self.sample_save_dir):
            os.makedirs(self.sample_save_dir, exist_ok=True)
            if self.global_rank == 0:
                self.logger.info(f"[Rank {self.global_rank}] Created save directory: {self.sample_save_dir}")
        
        if self.save_trajectory_times:
            os.makedirs(self.trajectory_save_dir, exist_ok=True)
        
        # Use ThreadPoolExecutor for asynchronous saving
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending_futures = []
            
            for batch_idx, batch_data in enumerate(self.dataloader):
                try:
                    # Get ERA5 data and timestamps based on fusion type
                    if 'timestamp' not in batch_data:
                        raise KeyError(f"[Rank {self.global_rank}] 'timestamp' not found in batch_data. Available keys: {list(batch_data.keys())}")
                    
                    batch_timestamps = batch_data['timestamp']  # List of timestamp strings
                    
                    if not batch_timestamps:
                        self.logger.warning(f"[Rank {self.global_rank}] Batch {batch_idx} has empty timestamps!")
                        continue
                    
                    if self.multimodal_fusion_type == 'direct_concat':
                        if 'era5_on_nodes' not in batch_data:
                            raise KeyError(f"[Rank {self.global_rank}] 'era5_on_nodes' not found in batch_data for direct_concat mode")
                        era5_data = batch_data['era5_on_nodes'].to(self.device)
                        current_batch_size = era5_data.shape[0]
                    else:
                        if 'era5' not in batch_data:
                            raise KeyError(f"[Rank {self.global_rank}] 'era5' not found in batch_data")
                        era5_data = batch_data['era5'].to(self.device)
                        current_batch_size = era5_data.shape[0]
                    
                    if current_batch_size == 0:
                        self.logger.warning(f"[Rank {self.global_rank}] Batch {batch_idx} has zero batch size!")
                        continue
                    
                    if self.global_rank == 0 and batch_idx % 10 == 0:
                        self.logger.info(f"[Rank {self.global_rank}] Processing batch {batch_idx}, size: {current_batch_size}, timestamps: {batch_timestamps[:3]}...")

                    # Check which samples are missing
                    needed_indices = []
                    needed_meta = []
                    existing_count = 0
                    
                    for i in range(current_batch_size):
                        ts = batch_timestamps[i]
                        # Skip timestamps not in the requested target set (if configured)
                        if target_set is not None and ts not in target_set:
                            continue
                        matched_targets.add(ts)
                        for j in range(self.ensemble_size):
                            fname = f"sample_{ts}_ens{j:03d}.npz"
                            file_path = os.path.join(self.sample_save_dir, fname)
                            is_valid = False
                            
                            if os.path.exists(file_path):
                                try:
                                    if os.path.getsize(file_path) > 50 * 1024:
                                        is_valid = True
                                        existing_count += 1
                                except OSError as e:
                                    self.logger.warning(f"[Rank {self.global_rank}] Error checking file {file_path}: {e}")
                                    is_valid = False
                            
                            if not is_valid:
                                needed_indices.append(i * self.ensemble_size + j)
                                needed_meta.append((ts, j))
                    
                    if not needed_indices:
                        if self.global_rank == 0 and batch_idx % 100 == 0:
                            self.logger.info(f"[Rank {self.global_rank}] Batch {batch_idx}/{samples_per_gpu}: all {current_batch_size * self.ensemble_size} samples exist, skipping")
                        continue
                    
                    if self.global_rank == 0 and batch_idx % 10 == 0:
                        self.logger.info(f"[Rank {self.global_rank}] Batch {batch_idx}: {len(needed_indices)}/{current_batch_size * self.ensemble_size} samples need generation")
                    
                    # Build the [n_needed, ...] condition batch.
                    # Member-forced mode: sample j of timestamp ts is conditioned on
                    # atmospheric forecast member j (per-(ts, j) fetch; the batch
                    # condition from the DataLoader is only used for shape checks).
                    # Default mode: all members share the batch condition.
                    if self.member_forced:
                        era5_data_subset = self._load_member_conditions(needed_meta)
                    else:
                        # Generate ensemble_size samples for each condition
                        era5_data_repeated = era5_data.repeat_interleave(self.ensemble_size, dim=0)
                        era5_data_subset = era5_data_repeated[needed_indices]

                    # Repeat time-of-day for ensemble members (same tod applies to all ensemble members)
                    if self.use_tod_conditioning and 'tod' in batch_data:
                        tod_data = batch_data['tod'].to(self.device)
                        tod_repeated = tod_data.repeat_interleave(self.ensemble_size, dim=0)

                    # Repeat absolute-time tidal features for ensemble members
                    if self.use_tod_conditioning and 'time_feats' in batch_data:
                        time_feats_data = batch_data['time_feats'].to(self.device)
                        time_feats_repeated = time_feats_data.repeat_interleave(self.ensemble_size, dim=0)

                    # Per-member condition perturbation: each ensemble member sees a
                    # slightly different ERA5 condition (simulates reanalysis uncertainty).
                    # ERA5 is min-max normalized to [-1, 1] (see wave_normalize.py);
                    # clip keeps perturbed conditions in-range.
                    if self.condition_noise > 0.0:
                        era5_data_subset = (era5_data_subset + \
                            torch.randn_like(era5_data_subset) * self.condition_noise).clamp(-1.0, 1.0)
                    
                    # Prepare condition kwargs based on fusion type
                    condition_kwargs = {}
                    if self.multimodal_fusion_type == 'direct_concat':
                        condition_kwargs['era5_on_nodes'] = era5_data_subset
                    else:
                        condition_kwargs['era5_data'] = era5_data_subset

                    # Add time-of-day for tidal conditioning (subset to needed indices)
                    if self.use_tod_conditioning and 'tod' in batch_data:
                        condition_kwargs['tod'] = tod_repeated[needed_indices]

                    # Add absolute-time tidal features (subset to needed indices)
                    if self.use_tod_conditioning and 'time_feats' in batch_data:
                        condition_kwargs['time_feats'] = time_feats_repeated[needed_indices]

                    # Generate samples conditioned on repeated ERA5 data
                    try:
                        # Input-perturbation baseline: one prior-noise draw per
                        # condition, shared by its ensemble members (needed_indices
                        # are grouped as i*E+j for condition i, member j).
                        prior_noise = None
                        if self.fix_prior_noise:
                            n_needed = len(needed_indices)
                            base_noise = torch.randn(
                                current_batch_size, num_nodes, self.in_channels,
                                device=self.device, dtype=torch.float32)
                            cond_idx = torch.tensor(
                                [idx // self.ensemble_size for idx in needed_indices],
                                device=self.device, dtype=torch.long)
                            prior_noise = base_noise[cond_idx]

                        sample_result = self.fd.sample(
                            batch_size=len(needed_indices),
                            data_shape=(num_nodes, self.in_channels),
                            steps=self.sampling_steps,
                            save_times=self.save_trajectory_times,
                            stochastic_gamma=self.stochastic_gamma,
                            stochastic_schedule=self.stochastic_schedule,
                            prior_noise=prior_noise,
                            **condition_kwargs
                        )
                    except Exception as e:
                        # _setup_logging() only runs on rank 0 (see trainer.py),
                        # so non-rank-0 workers must not rely on self.logger here.
                        logger = getattr(self, 'logger', None)
                        if logger is not None:
                            logger.error(f"[Rank {self.global_rank}] Sampling failed for batch {batch_idx}: {e}")
                        else:
                            import logging
                            logging.getLogger(__name__).error(
                                f"[Rank {self.global_rank}] Sampling failed for batch {batch_idx}: {e}")
                        raise
                    
                    # Unpack optional ODE trajectory snapshots {t_val: tensor}
                    if self.save_trajectory_times:
                        batch_samples, batch_trajectory = sample_result
                    else:
                        batch_samples, batch_trajectory = sample_result, None
                    
                    # Validate generated samples
                    if batch_samples is None:
                        raise ValueError(f"[Rank {self.global_rank}] fd.sample returned None for batch {batch_idx}")
                    
                    if not isinstance(batch_samples, torch.Tensor):
                        raise TypeError(f"[Rank {self.global_rank}] fd.sample returned {type(batch_samples)}, expected tensor")
                    
                    if batch_samples.numel() == 0:
                        raise ValueError(f"[Rank {self.global_rank}] fd.sample returned empty tensor for batch {batch_idx}")
                    
                    if torch.isnan(batch_samples).any():
                        self.logger.warning(f"[Rank {self.global_rank}] Batch {batch_idx} contains NaN values in generated samples!")
                    
                    total_samples_generated += len(needed_indices)
                    
                    # Asynchronously save samples while GPU continues to next batch
                    batch_samples_cpu = batch_samples.cpu()
                    batch_trajectory_cpu = (
                        {t_val: traj.cpu() for t_val, traj in batch_trajectory.items()}
                        if batch_trajectory is not None else None
                    )
                    
                    # Submit save task to thread pool
                    future = executor.submit(
                        self._save_samples_subset_sync,
                        batch_samples_cpu,
                        needed_meta,
                        save_lock,
                        batch_trajectory_cpu
                    )
                    pending_futures.append(future)
                    total_batches_processed += 1
                    
                    # Log progress
                    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == samples_per_gpu:
                        if self.global_rank == 0:
                            self.logger.info(
                                f"[Rank {self.global_rank}] Progress: {batch_idx + 1}/{samples_per_gpu} batches, "
                                f"{total_samples_generated} samples generated, "
                                f"{len(pending_futures)} save tasks pending"
                            )
                    
                    # Clean up completed futures periodically to avoid memory buildup
                    if len(pending_futures) >= 10:
                        completed = [f for f in pending_futures if f.done()]
                        for f in completed:
                            try:
                                result = f.result()
                                total_saved += result
                            except Exception as e:
                                (getattr(self, 'logger', None) or logging.getLogger(__name__)).error(
                                    f"[Rank {self.global_rank}] Save task failed: {e}")
                            pending_futures.remove(f)

                except Exception as e:
                    (getattr(self, 'logger', None) or logging.getLogger(__name__)).error(
                        f"[Rank {self.global_rank}] Error processing batch {batch_idx}: {e}")
                    raise
            
            # Wait for all save tasks to complete
            if self.global_rank == 0:
                self.logger.info(f"[Rank {self.global_rank}] Waiting for {len(pending_futures)} pending save tasks to complete...")
            
            for i, future in enumerate(pending_futures):
                try:
                    result = future.result(timeout=3600)  # 1h timeout per save task (slow NFS home)
                    total_saved += result
                except Exception as e:
                    self.logger.error(f"[Rank {self.global_rank}] Save task {i}/{len(pending_futures)} failed: {e}")
        
        if self.global_rank == 0:
            self.logger.info(
                f"[Rank {self.global_rank}] Conditional sampling completed: "
                f"{total_saved} samples saved, "
                f"{total_batches_processed} batches processed, "
                f"{total_samples_generated} samples generated"
            )
            if target_set is not None:
                self.logger.info(
                    f"[Rank {self.global_rank}] Target timestamps matched: "
                    f"{len(matched_targets)}/{len(target_set)}"
                )
                missing = sorted(target_set - matched_targets)
                if missing:
                    self.logger.warning(
                        f"[Rank {self.global_rank}] {len(missing)} requested timestamps not found in dataset, "
                        f"e.g. {missing[:5]}"
                    )
        
        return total_saved

    def _run_conditional_sampling_grid(self, samples_per_gpu):
        """
        Grid-mode conditional sampling: generates ensemble samples on the
        regular (75x107) grid, denormalises, masks land pixels to NaN, and
        saves as grid-format .npz files.

        Post-processing can later interpolate these to irregular nodes for
        RMSE comparison against the node-based model.
        """
        import concurrent.futures
        num_nodes = self.coords.shape[0]
        data_shape = (self.in_channels, self.grid_shape[0], self.grid_shape[1])

        if self.global_rank == 0:
            self.logger.info(f"[Rank {self.global_rank}] Starting grid conditional sampling...")
            self.logger.info(f"  data_shape={data_shape}, ensemble_size={self.ensemble_size}")

        os.makedirs(self.sample_save_dir, exist_ok=True)

        total_saved = 0
        total_batches_processed = 0
        total_samples_generated = 0
        save_lock = threading.Lock()
        target_set = set(self.target_timestamps) if self.target_timestamps else None
        matched_targets = set()

        if target_set is not None and self.global_rank == 0:
            self.logger.info(f"[Rank {self.global_rank}] Target timestamps filter enabled: {len(target_set)} timestamps requested")
        if self.save_trajectory_times:
            os.makedirs(self.trajectory_save_dir, exist_ok=True)
            if self.global_rank == 0:
                self.logger.info(f"[Rank {self.global_rank}] Saving ODE trajectory at t={self.save_trajectory_times} to {self.trajectory_save_dir}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            pending_futures = []

            for batch_idx, batch_data in enumerate(self.dataloader):
                try:
                    if 'timestamp' not in batch_data:
                        raise KeyError(f"[Rank {self.global_rank}] 'timestamp' not found in batch_data. "
                                       f"Available keys: {list(batch_data.keys())}")
                    batch_timestamps = batch_data['timestamp']

                    if not batch_timestamps:
                        continue

                    if 'era5' not in batch_data:
                        raise KeyError(f"[Rank {self.global_rank}] 'era5' not found in batch_data")
                    era5_data = batch_data['era5'].to(self.device)
                    current_batch_size = era5_data.shape[0]

                    if current_batch_size == 0:
                        continue

                    era5_data_repeated = era5_data.repeat_interleave(self.ensemble_size, dim=0)

                    # Repeat time-of-day for ensemble members (same tod applies to all ensemble members)
                    if self.use_tod_conditioning and 'tod' in batch_data:
                        tod_data = batch_data['tod'].to(self.device)
                        tod_repeated = tod_data.repeat_interleave(self.ensemble_size, dim=0)

                    # Repeat absolute-time tidal features for ensemble members
                    if self.use_tod_conditioning and 'time_feats' in batch_data:
                        time_feats_data = batch_data['time_feats'].to(self.device)
                        time_feats_repeated = time_feats_data.repeat_interleave(self.ensemble_size, dim=0)

                    # Skip already-existing samples
                    needed_indices = []
                    needed_meta = []
                    for i in range(current_batch_size):
                        ts = batch_timestamps[i]
                        # Skip timestamps not in the requested target set (if configured)
                        if target_set is not None and ts not in target_set:
                            continue
                        matched_targets.add(ts)
                        for j in range(self.ensemble_size):
                            fname = f"sample_{ts}_ens{j:03d}.npz"
                            file_path = os.path.join(self.sample_save_dir, fname)
                            if not (os.path.exists(file_path) and os.path.getsize(file_path) > 50 * 1024):
                                needed_indices.append(i * self.ensemble_size + j)
                                needed_meta.append((ts, j))

                    if not needed_indices:
                        continue

                    era5_data_subset = era5_data_repeated[needed_indices]

                    # Per-member condition perturbation (same as HK sampling path)
                    if self.condition_noise > 0.0:
                        era5_data_subset = (era5_data_subset + \
                            torch.randn_like(era5_data_subset) * self.condition_noise).clamp(0.0, 1.0)

                    # Prepare condition kwargs based on fusion type
                    condition_kwargs = {'era5_data': era5_data_subset}

                    # Add time-of-day for tidal conditioning (subset to needed indices)
                    if self.use_tod_conditioning and 'tod' in batch_data:
                        condition_kwargs['tod'] = tod_repeated[needed_indices]

                    # Add absolute-time tidal features (subset to needed indices)
                    if self.use_tod_conditioning and 'time_feats' in batch_data:
                        condition_kwargs['time_feats'] = time_feats_repeated[needed_indices]

                    sample_result = self.fd.sample(
                        batch_size=len(needed_indices),
                        data_shape=data_shape,
                        steps=self.sampling_steps,
                        save_times=self.save_trajectory_times,
                        stochastic_gamma=self.stochastic_gamma,
                        stochastic_schedule=self.stochastic_schedule,
                        **condition_kwargs,
                    )
                    if self.save_trajectory_times:
                        batch_samples, batch_trajectory = sample_result
                    else:
                        batch_samples, batch_trajectory = sample_result, None
                    total_samples_generated += len(needed_indices)

                    batch_samples_cpu = batch_samples.cpu()
                    batch_trajectory_cpu = (
                        {t_val: traj.cpu() for t_val, traj in batch_trajectory.items()}
                        if batch_trajectory is not None else None
                    )
                    future = executor.submit(
                        self._save_samples_grid_sync,
                        batch_samples_cpu,
                        needed_meta,
                        save_lock,
                        batch_trajectory_cpu
                    )
                    pending_futures.append(future)
                    total_batches_processed += 1

                    # Clean up completed futures
                    if len(pending_futures) >= 10:
                        completed = [f for f in pending_futures if f.done()]
                        for f in completed:
                            try:
                                total_saved += f.result()
                            except Exception as e:
                                self.logger.error(f"[Rank {self.global_rank}] Save task failed: {e}")
                            pending_futures.remove(f)

                except Exception as e:
                    self.logger.error(f"[Rank {self.global_rank}] Error processing batch {batch_idx}: {e}")
                    raise

            # Wait for remaining save tasks
            for future in pending_futures:
                try:
                    total_saved += future.result(timeout=3600)
                except Exception as e:
                    self.logger.error(f"[Rank {self.global_rank}] Save task failed: {e}")

        if self.global_rank == 0:
            self.logger.info(f"[Rank {self.global_rank}] Grid conditional sampling completed: "
                             f"{total_saved} samples saved")
            if target_set is not None:
                self.logger.info(f"[Rank {self.global_rank}] Target timestamps matched: "
                                 f"{len(matched_targets)}/{len(target_set)}")
                missing = sorted(target_set - matched_targets)
                if missing:
                    self.logger.warning(f"[Rank {self.global_rank}] {len(missing)} requested timestamps "
                                        f"not found in dataset, e.g. {missing[:5]}")

        return total_saved

    def _run_unconditional_sampling(self, num_nodes, my_samples, batch_size, start_idx):
        """
        Run unconditional sampling to generate specified number of samples.
        Handles both traditional mode and patch mode.
        
        For patch mode: Generates dummy local_coords and mask since we don't have real data.
        
        Args:
            num_nodes: Number of nodes in the graph (for traditional mode)
            my_samples: Number of samples this GPU should generate
            batch_size: Number of samples per batch
            start_idx: Starting global index for sample numbering
            
        Returns:
            total_saved: Total number of samples saved
        """
        total_saved = 0
        num_batches = (my_samples + batch_size - 1) // batch_size  # Ceiling division
        failed_batches = []
        
        if self.global_rank == 0:
            self.logger.info(f"[Rank {self.global_rank}] Starting unconditional sampling: {my_samples} samples, {num_batches} batches")
            self.logger.info(f"[Rank {self.global_rank}] Batch size: {batch_size}, Start index: {start_idx}")
        
        # Verify save directory exists
        if not os.path.exists(self.sample_save_dir):
            os.makedirs(self.sample_save_dir, exist_ok=True)
            if self.global_rank == 0:
                self.logger.info(f"[Rank {self.global_rank}] Created save directory: {self.sample_save_dir}")
        
        for batch_idx in range(num_batches):
            try:
                # Calculate batch size for this iteration
                current_batch_size = min(batch_size, my_samples - batch_idx * batch_size)
                
                if current_batch_size <= 0:
                    self.logger.warning(f"[Rank {self.global_rank}] Batch {batch_idx}: current_batch_size is {current_batch_size}, skipping")
                    continue
                
                if self.global_rank == 0 and batch_idx % 10 == 0:
                    self.logger.info(f"[Rank {self.global_rank}] Generating batch {batch_idx}/{num_batches}, size: {current_batch_size}")
                
                # Generate batch of samples (no conditions)
                try:
                    batch_samples = self.fd.sample(
                        batch_size=current_batch_size,
                        data_shape=(num_nodes, self.in_channels),
                        steps=self.sampling_steps,
                        stochastic_gamma=self.stochastic_gamma,
                        stochastic_schedule=self.stochastic_schedule,
                    )
                except Exception as e:
                    error_msg = f"[Rank {self.global_rank}] Sampling failed for batch {batch_idx}: {e}"
                    self.logger.error(error_msg)
                    failed_batches.append((batch_idx, str(e)))
                    continue
                
                # Validate generated samples
                if batch_samples is None:
                    self.logger.error(f"[Rank {self.global_rank}] fd.sample returned None for batch {batch_idx}")
                    failed_batches.append((batch_idx, "fd.sample returned None"))
                    continue
                
                if not isinstance(batch_samples, torch.Tensor):
                    self.logger.error(f"[Rank {self.global_rank}] fd.sample returned {type(batch_samples)}, expected tensor for batch {batch_idx}")
                    failed_batches.append((batch_idx, f"Unexpected return type: {type(batch_samples)}"))
                    continue
                
                if batch_samples.numel() == 0:
                    self.logger.error(f"[Rank {self.global_rank}] fd.sample returned empty tensor for batch {batch_idx}")
                    failed_batches.append((batch_idx, "Empty tensor returned"))
                    continue
                
                # Check for NaN/Inf in generated samples
                if torch.isnan(batch_samples).any():
                    nan_count = torch.isnan(batch_samples).sum().item()
                    self.logger.warning(f"[Rank {self.global_rank}] Batch {batch_idx} contains {nan_count} NaN values in generated samples!")
                
                if torch.isinf(batch_samples).any():
                    inf_count = torch.isinf(batch_samples).sum().item()
                    self.logger.warning(f"[Rank {self.global_rank}] Batch {batch_idx} contains {inf_count} Inf values in generated samples!")
                
                # Save samples using the encapsulated method
                try:
                    saved_count = self._save_samples_batch(
                        batch_samples, batch_idx, batch_size, start_idx
                    )
                    total_saved += saved_count
                    
                    if saved_count != current_batch_size:
                        self.logger.warning(f"[Rank {self.global_rank}] Batch {batch_idx}: saved {saved_count}/{current_batch_size} samples")
                
                except Exception as e:
                    error_msg = f"[Rank {self.global_rank}] Failed to save batch {batch_idx}: {e}"
                    self.logger.error(error_msg)
                    failed_batches.append((batch_idx, str(e)))
                    continue
                
                # Log progress
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                    if self.global_rank == 0:
                        self.logger.info(
                            f"[Rank {self.global_rank}] Progress: {total_saved}/{my_samples} samples saved "
                            f"(Batch {batch_idx + 1}/{num_batches})"
                        )
            
            except Exception as e:
                error_msg = f"[Rank {self.global_rank}] Unexpected error in batch {batch_idx}: {e}"
                self.logger.error(error_msg)
                failed_batches.append((batch_idx, str(e)))
                continue
        
        # Log summary
        if self.global_rank == 0:
            self.logger.info(
                f"[Rank {self.global_rank}] Unconditional sampling completed: "
                f"{total_saved}/{my_samples} samples saved, "
                f"{num_batches - len(failed_batches)}/{num_batches} batches successful"
            )
            if failed_batches:
                self.logger.error(f"[Rank {self.global_rank}] Failed batches: {len(failed_batches)}")
                for batch_idx, error in failed_batches[:10]:  # Show first 10
                    self.logger.error(f"  - Batch {batch_idx}: {error}")
        
        return total_saved
    
    def _sample_impl(self):
        """
        Internal implementation of sampling logic.
        """
        batch_size = self.sampling_batch_size
        
        # Validate configuration
        if self.coords is None:
            raise ValueError(f"[Rank {self.global_rank}] coords is None! Cannot determine num_nodes.")
        
        num_nodes = self.coords.shape[0]
        if num_nodes == 0:
            raise ValueError(f"[Rank {self.global_rank}] num_nodes is 0! Check coord_file_path.")
        
        if self.global_rank == 0:
            self.logger.info(f"[Rank {self.global_rank}] Sampling configuration validated: num_nodes={num_nodes}")
        
        # Determine total samples based on conditional vs unconditional generation
        if self.conditional_generation:
            # Conditional: total samples = size of test dataset
            total_samples = len(self.dataset)
            samples_per_gpu = len(self.dataloader)  # Number of batches this GPU will process
            
            if total_samples == 0:
                raise ValueError(f"[Rank {self.global_rank}] Dataset is empty!")
            
            if self.global_rank == 0:
                self.logger.info(f"[Rank {self.global_rank}] Conditional mode: dataset size={total_samples}, dataloader batches={samples_per_gpu}")
        else:
            # Unconditional: use configured total_samples
            total_samples = self.total_samples
            samples_per_gpu = self.total_samples // self.world_size
            remainder = self.total_samples % self.world_size
            
            # Distribute remainder samples to first few GPUs
            if self.global_rank < remainder:
                my_samples = samples_per_gpu + 1
            else:
                my_samples = samples_per_gpu
            
            if self.global_rank == 0:
                self.logger.info(f"[Rank {self.global_rank}] Unconditional mode: total={total_samples}, per_gpu={samples_per_gpu}, remainder={remainder}")
        
        # Calculate the starting global index for this GPU (for unconditional sampling)
        if not self.conditional_generation:
            start_idx = self.global_rank * samples_per_gpu + min(self.global_rank, remainder)
        
        if self.global_rank == 0:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Starting parallel sampling with batch control...")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Model mode: {'Patch-based' if self.use_patches else 'Traditional (node-based)'}")
            if self.conditional_generation:
                self.logger.info(f"Conditional generation: Iterating through entire test dataset")
                self.logger.info(f"Total test samples: {total_samples}")
                self.logger.info(f"Each GPU processes ~{samples_per_gpu} batches")
                self.logger.info(f"Ensemble size: {self.ensemble_size}")
                self.logger.info(f"Atmospheric source: {self.atm_source}")
                if self.atm_source == 'ens':
                    self.logger.info(f"ENS lead time: {self.ens_lead_time}h")
                    self.logger.info(f"ENS data dir: {self.ens_dir}")
            else:
                self.logger.info(f"Unconditional generation: No ERA5 conditions")
                self.logger.info(f"Total samples to generate: {total_samples}")
                self.logger.info(f"Samples per GPU: {samples_per_gpu} (+ {remainder} GPUs will generate 1 extra)")
            self.logger.info(f"Number of GPUs: {self.world_size}")
            self.logger.info(f"Batch size per GPU: {batch_size}")
            self.logger.info(f"Sampling steps: {self.sampling_steps}")
            self.logger.info(f"Use EMA for sampling: {self.use_ema_for_sampling} (EMA enabled: {self.ema is not None})")
            self.logger.info(f"Save directory: {self.sample_save_dir}")
            self.logger.info(f"{'='*60}")
        
        # Move finished samples from previous run if specified
        if self.global_rank == 0 and self.resume_sample_path:
            if os.path.exists(self.resume_sample_path):
                self.logger.info(f"Moving entire directory from {self.resume_sample_path} to {self.sample_save_dir}...")
                try:
                    # Avoid moving to itself
                    if os.path.abspath(self.resume_sample_path) != os.path.abspath(self.sample_save_dir):
                        if os.path.exists(self.sample_save_dir):
                            shutil.rmtree(self.sample_save_dir)
                        shutil.move(self.resume_sample_path, self.sample_save_dir)
                        self.logger.info("Move completed.")
                except Exception as e:
                    self.logger.error(f"Error moving directory: {e}")
            else:
                self.logger.warning(f"resume_sample_path {self.resume_sample_path} does not exist.")
        
        # Create save directory (all ranks create to avoid race condition)
        try:
            os.makedirs(self.sample_save_dir, exist_ok=True)
            if self.global_rank == 0:
                self.logger.info(f"[Rank {self.global_rank}] Save directory ready: {self.sample_save_dir}")
        except Exception as e:
            raise RuntimeError(f"[Rank {self.global_rank}] Failed to create save directory {self.sample_save_dir}: {e}") from e
        
        # Wait for all GPUs to finish creating directory and copying
        dist.barrier()
        
        # Generate samples
        start_time = time.time()
        
        try:
            if self.conditional_generation:
                # Conditional generation: iterate through test dataset
                if self.region in ('hk_grid', 'global_grid'):
                    if self.member_forced:
                        raise NotImplementedError(
                            "member_forced sampling requires node-based datasets "
                            "(region='hk'/'global'); grid datasets have no "
                            "atm_source support")
                    total_saved = self._run_conditional_sampling_grid(samples_per_gpu)
                else:
                    total_saved = self._run_conditional_sampling(num_nodes, samples_per_gpu)
            else:
                # Unconditional generation: generate specified number of samples
                total_saved = self._run_unconditional_sampling(num_nodes, my_samples, batch_size, start_idx)
        except Exception as e:
            (getattr(self, 'logger', None) or logging.getLogger(__name__)).error(
                f"[Rank {self.global_rank}] Sampling failed with error: {e}")
            raise
        
        elapsed_time = time.time() - start_time
        
        # Synchronize all GPUs to ensure all saves are complete
        dist.barrier()
        
        if self.global_rank == 0:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"All GPUs finished sampling!")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Samples saved in directory: {self.sample_save_dir}")
            if self.conditional_generation:
                self.logger.info(f"Total samples generated: {total_samples} (one per test dataset sample)")
            else:
                self.logger.info(f"Total samples generated: {total_samples}")
            self.logger.info(f"Each sample saved as individual .npz file")
            self.logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
            self.logger.info(f"Throughput: {total_saved / elapsed_time:.2f} samples/second")
            self.logger.info(f"{'='*60}")
        
            self.logger.info(f"[Rank {self.global_rank}] Completed: {total_saved} samples saved")
        
        # Verify saved samples count matches expected
        if self.global_rank == 0:
            try:
                saved_files = [f for f in os.listdir(self.sample_save_dir) if f.endswith('.npz')]
                self.logger.info(f"[Rank {self.global_rank}] Verification: {len(saved_files)} .npz files in save directory")
            except Exception as e:
                self.logger.warning(f"[Rank {self.global_rank}] Could not verify saved files: {e}")


