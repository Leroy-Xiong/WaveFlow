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




class TrainerSavingMixin:
    def _save_samples_batch(self, batch_samples, batch_idx, batch_size, start_idx):
        """
        Save a batch of samples to individual npz files.
        
        Args:
            batch_samples: Tensor of samples with shape (batch_size, num_nodes, 1)
            batch_idx: Current batch index
            batch_size: Number of samples in the batch
            start_idx: Starting index for global sample numbering
            
        Returns:
            total_saved: Number of samples saved
        """
        # Validate input
        if batch_samples is None:
            raise ValueError(f"[Rank {self.global_rank}] batch_samples is None!")
        
        if not isinstance(batch_samples, torch.Tensor):
            raise TypeError(f"[Rank {self.global_rank}] batch_samples must be a tensor, got {type(batch_samples)}")
        
        if batch_samples.numel() == 0:
            raise ValueError(f"[Rank {self.global_rank}] batch_samples is empty!")
        
        # Move to CPU and convert to numpy to free GPU memory
        try:
            batch_samples_np = batch_samples.cpu().numpy()
        except Exception as e:
            raise RuntimeError(f"[Rank {self.global_rank}] Failed to convert batch_samples to numpy: {e}") from e
        
        current_batch_size = batch_samples_np.shape[0]
        total_saved = 0
        failed_saves = []
        
        if self.global_rank == 0 and batch_idx % 10 == 0:
            self.logger.info(f"[Rank {self.global_rank}] Saving batch {batch_idx}, shape: {batch_samples_np.shape}")
        
        # Save each sample individually as npz file
        for i in range(current_batch_size):
            global_idx = start_idx + batch_idx * batch_size + i
            
            try:
                # Denormalize this sample
                sample_normalized = batch_samples_np[i]  # Shape: (num_nodes, C)
                
                if self.transform is None:
                    raise ValueError(f"[Rank {self.global_rank}] transform is None!")
                
                sample_denormalized = self.transform.denormalize(sample_normalized, variable=None)
                
                # Validate denormalized sample
                if sample_denormalized is None:
                    raise ValueError(f"[Rank {self.global_rank}] Denormalization returned None for sample {global_idx}")
                
                if np.isnan(sample_denormalized).any():
                    self.logger.warning(f"[Rank {self.global_rank}] Sample {global_idx} contains NaN values!")
                
                if np.isinf(sample_denormalized).any():
                    self.logger.warning(f"[Rank {self.global_rank}] Sample {global_idx} contains Inf values!")
                
                # Squeeze last dimension if C_orig == 1 for cleaner output
                if sample_denormalized.shape[-1] == 1:
                    sample_denormalized = sample_denormalized.squeeze(-1)
                
                save_path = os.path.join(
                    self.sample_save_dir, 
                    f"sample_{global_idx:06d}.npz"
                )
                
                # Save as compressed npz file
                np.savez_compressed(
                    save_path,
                    sample=sample_denormalized,
                    sampling_steps=self.sampling_steps,
                    global_idx=global_idx,
                    rank=self.global_rank
                )
                
                # Verify the file was saved correctly
                if not os.path.exists(save_path):
                    raise IOError(f"[Rank {self.global_rank}] File was not created: {save_path}")
                
                file_size = os.path.getsize(save_path)
                if file_size < 100:  # Less than 100 bytes is suspicious
                    self.logger.warning(f"[Rank {self.global_rank}] Sample {global_idx} has suspiciously small file size: {file_size} bytes")
                
                # Try to reload and verify
                try:
                    loaded = np.load(save_path)
                    loaded_sample = loaded['sample']
                    if loaded_sample.shape != sample_denormalized.shape:
                        raise ValueError(f"[Rank {self.global_rank}] Shape mismatch: saved {sample_denormalized.shape}, loaded {loaded_sample.shape}")
                    loaded.close()
                except Exception as e:
                    self.logger.error(f"[Rank {self.global_rank}] Failed to verify saved file {save_path}: {e}")
                    failed_saves.append((global_idx, str(e)))
                    continue
                
                total_saved += 1
                
            except Exception as e:
                error_msg = f"[Rank {self.global_rank}] Failed to save sample {global_idx}: {e}"
                self.logger.error(error_msg)
                failed_saves.append((global_idx, str(e)))
                continue
        
        if failed_saves:
            self.logger.error(f"[Rank {self.global_rank}] Batch {batch_idx}: {len(failed_saves)}/{current_batch_size} samples failed to save")
            for idx, err in failed_saves[:5]:  # Show first 5 errors
                self.logger.error(f"  - Sample {idx}: {err}")
        
        return total_saved
    
    def _save_samples_batch_conditional(self, batch_samples, batch_timestamps):
        """
        Save a batch of conditional samples to individual npz files.
        Used when iterating through test dataset for conditional generation.
        This is a wrapper that converts tensor to numpy if needed.
        
        Args:
            batch_samples: Tensor of samples with shape (batch_size, ensemble_size, num_nodes, 1)
            batch_timestamps: List of timestamp strings (e.g., ['20210101T00', '20210101T01', ...])
            
        Returns:
            total_saved: Number of samples saved
        """
        # Convert to numpy if it's a tensor
        if torch.is_tensor(batch_samples):
            batch_samples_np = batch_samples.cpu().numpy()
        else:
            batch_samples_np = batch_samples
        
        return self._save_samples_batch_conditional_sync(
            batch_samples_np, batch_timestamps, None
        )
    
    def _save_samples_batch_conditional_sync(self, batch_samples_np, batch_timestamps, lock):
        """
        Thread-safe method to save a batch of conditional samples to individual npz files.
        This method is designed to be called from a thread pool.
        
        Args:
            batch_samples_np: Numpy array of samples with shape (batch_size, ensemble_size, num_nodes, 1)
            batch_timestamps: List of timestamp strings (e.g., ['20210101T00', '20210101T01', ...])
            lock: Threading lock for thread-safe logging (can be None if single-threaded)
            
        Returns:
            total_saved: Number of samples saved
        """
        # Validate inputs
        if batch_samples_np is None:
            raise ValueError(f"[Rank {self.global_rank}] batch_samples_np is None!")
        
        if batch_timestamps is None:
            raise ValueError(f"[Rank {self.global_rank}] batch_timestamps is None!")
        
        if not isinstance(batch_samples_np, np.ndarray):
            raise TypeError(f"[Rank {self.global_rank}] batch_samples_np must be numpy array, got {type(batch_samples_np)}")
        
        if batch_samples_np.size == 0:
            raise ValueError(f"[Rank {self.global_rank}] batch_samples_np is empty!")
        
        current_batch_size = batch_samples_np.shape[0]
        ensemble_size = batch_samples_np.shape[1]
        total_saved = 0
        failed_saves = []
        
        if len(batch_timestamps) != current_batch_size:
            raise ValueError(f"[Rank {self.global_rank}] Timestamp count ({len(batch_timestamps)}) doesn't match batch size ({current_batch_size})")
        
        # Save each sample individually as npz file with its corresponding timestamp and ensemble index
        for i in range(current_batch_size):
            timestamp = batch_timestamps[i]
            
            # Process each ensemble member for this timestamp
            for j in range(ensemble_size):
                try:
                    sample_normalized = batch_samples_np[i, j]  # Shape: (num_nodes, C)
                    
                    if self.transform is None:
                        raise ValueError(f"[Rank {self.global_rank}] transform is None!")
                    
                    sample_denormalized = self.transform.denormalize(sample_normalized, variable=None)
                    
                    # Validate denormalized sample
                    if sample_denormalized is None:
                        raise ValueError(f"[Rank {self.global_rank}] Denormalization returned None for {timestamp}_ens{j:03d}")
                    
                    if np.isnan(sample_denormalized).any():
                        if lock:
                            with lock:
                                self.logger.warning(f"[Rank {self.global_rank}] Sample {timestamp}_ens{j:03d} contains NaN values!")
                        else:
                            self.logger.warning(f"[Rank {self.global_rank}] Sample {timestamp}_ens{j:03d} contains NaN values!")
                    
                    if np.isinf(sample_denormalized).any():
                        if lock:
                            with lock:
                                self.logger.warning(f"[Rank {self.global_rank}] Sample {timestamp}_ens{j:03d} contains Inf values!")
                        else:
                            self.logger.warning(f"[Rank {self.global_rank}] Sample {timestamp}_ens{j:03d} contains Inf values!")
                    
                    # Squeeze last dimension if C_orig == 1 for cleaner output
                    if sample_denormalized.shape[-1] == 1:
                        sample_denormalized = sample_denormalized.squeeze(-1)
                    
                    save_path = os.path.join(
                        self.sample_save_dir, 
                        f"sample_{timestamp}_ens{j:03d}.npz"
                    )
                    
                    # Save as compressed npz file
                    np.savez_compressed(
                        save_path,
                        sample=sample_denormalized,
                        sampling_steps=self.sampling_steps,
                        timestamp=timestamp,
                        ensemble_index=j,
                        rank=self.global_rank
                    )
                    
                    # Verify the file was saved correctly
                    if not os.path.exists(save_path):
                        raise IOError(f"[Rank {self.global_rank}] File was not created: {save_path}")
                    
                    file_size = os.path.getsize(save_path)
                    if file_size < 100:  # Less than 100 bytes is suspicious
                        if lock:
                            with lock:
                                self.logger.warning(f"[Rank {self.global_rank}] Sample {timestamp}_ens{j:03d} has suspiciously small file size: {file_size} bytes")
                        else:
                            self.logger.warning(f"[Rank {self.global_rank}] Sample {timestamp}_ens{j:03d} has suspiciously small file size: {file_size} bytes")
                    
                    # Try to reload and verify
                    try:
                        loaded = np.load(save_path)
                        loaded_sample = loaded['sample']
                        if loaded_sample.shape != sample_denormalized.shape:
                            raise ValueError(f"[Rank {self.global_rank}] Shape mismatch: saved {sample_denormalized.shape}, loaded {loaded_sample.shape}")
                        loaded.close()
                    except Exception as e:
                        error_msg = f"[Rank {self.global_rank}] Failed to verify saved file {save_path}: {e}"
                        if lock:
                            with lock:
                                self.logger.error(error_msg)
                        else:
                            self.logger.error(error_msg)
                        failed_saves.append((timestamp, j, str(e)))
                        continue
                    
                    total_saved += 1
                    
                except Exception as e:
                    error_msg = f"[Rank {self.global_rank}] Failed to save sample {timestamp}_ens{j:03d}: {e}"
                    if lock:
                        with lock:
                            self.logger.error(error_msg)
                    else:
                        self.logger.error(error_msg)
                    failed_saves.append((timestamp, j, str(e)))
                    continue
        
        if failed_saves:
            error_summary = f"[Rank {self.global_rank}] Conditional batch: {len(failed_saves)}/{current_batch_size * ensemble_size} samples failed to save"
            if lock:
                with lock:
                    self.logger.error(error_summary)
                    for ts, ens_idx, err in failed_saves[:5]:
                        self.logger.error(f"  - {ts}_ens{ens_idx:03d}: {err}")
            else:
                self.logger.error(error_summary)
                for ts, ens_idx, err in failed_saves[:5]:
                    self.logger.error(f"  - {ts}_ens{ens_idx:03d}: {err}")
        
        return total_saved
    
    def _save_samples_subset_sync(self, batch_samples_np, batch_metadata, lock, trajectory_np=None):
        """
        Thread-safe method to save a subset of samples.
        
        Args:
            batch_samples_np: Numpy array or Tensor of samples with shape (N, num_nodes, C)
            batch_metadata: List of (timestamp, ensemble_index) tuples corresponding to samples
            lock: Threading lock
            trajectory_np: Optional dict {ode_t: np.ndarray of shape (N, num_nodes, C)}
                           with intermediate ODE states to save alongside the final
                           samples (saved under self.trajectory_save_dir).
            
        Returns:
            total_saved: Number of samples saved
        """
        # Validate inputs
        if batch_samples_np is None:
            raise ValueError(f"[Rank {self.global_rank}] batch_samples_np is None!")
        
        if batch_metadata is None:
            raise ValueError(f"[Rank {self.global_rank}] batch_metadata is None!")
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(batch_samples_np):
            batch_samples_np = batch_samples_np.cpu().numpy()
        
        if not isinstance(batch_samples_np, np.ndarray):
            raise TypeError(f"[Rank {self.global_rank}] batch_samples_np must be numpy array or tensor, got {type(batch_samples_np)}")
        
        if batch_samples_np.size == 0:
            raise ValueError(f"[Rank {self.global_rank}] batch_samples_np is empty!")
        
        total_saved = 0
        N = batch_samples_np.shape[0]
        failed_saves = []
        
        if len(batch_metadata) != N:
            raise ValueError(f"[Rank {self.global_rank}] Metadata count ({len(batch_metadata)}) doesn't match sample count ({N})")
        
        for i in range(N):
            try:
                sample_normalized = batch_samples_np[i]  # Shape: (num_nodes, C)
                timestamp, ensemble_index = batch_metadata[i]
                
                if self.transform is None:
                    raise ValueError(f"[Rank {self.global_rank}] transform is None!")
                
                sample_denormalized = self.transform.denormalize(sample_normalized, variable=None)
                
                # Validate denormalized sample
                if sample_denormalized is None:
                    raise ValueError(f"[Rank {self.global_rank}] Denormalization returned None for {timestamp}_ens{ensemble_index:03d}")
                
                if np.isnan(sample_denormalized).any():
                    if lock:
                        with lock:
                            self.logger.warning(f"[Rank {self.global_rank}] Sample {timestamp}_ens{ensemble_index:03d} contains NaN values!")
                    else:
                        self.logger.warning(f"[Rank {self.global_rank}] Sample {timestamp}_ens{ensemble_index:03d} contains NaN values!")
                
                if np.isinf(sample_denormalized).any():
                    if lock:
                        with lock:
                            self.logger.warning(f"[Rank {self.global_rank}] Sample {timestamp}_ens{ensemble_index:03d} contains Inf values!")
                    else:
                        self.logger.warning(f"[Rank {self.global_rank}] Sample {timestamp}_ens{ensemble_index:03d} contains Inf values!")
                
                # Squeeze last dimension if C_orig == 1 for cleaner output
                if sample_denormalized.shape[-1] == 1:
                    sample_denormalized = sample_denormalized.squeeze(-1)
                
                save_path = os.path.join(
                    self.sample_save_dir,
                    f"sample_{timestamp}_ens{ensemble_index:03d}.npz"
                )
                
                np.savez_compressed(
                    save_path,
                    sample=sample_denormalized,
                    sampling_steps=self.sampling_steps,
                    timestamp=timestamp,
                    ensemble_index=ensemble_index,
                    rank=self.global_rank
                )
                
                # Verify the file was saved correctly
                if not os.path.exists(save_path):
                    raise IOError(f"[Rank {self.global_rank}] File was not created: {save_path}")
                
                file_size = os.path.getsize(save_path)
                if file_size < 100:  # Less than 100 bytes is suspicious
                    if lock:
                        with lock:
                            self.logger.warning(f"[Rank {self.global_rank}] Sample {timestamp}_ens{ensemble_index:03d} has suspiciously small file size: {file_size} bytes")
                    else:
                        self.logger.warning(f"[Rank {self.global_rank}] Sample {timestamp}_ens{ensemble_index:03d} has suspiciously small file size: {file_size} bytes")
                
                # Try to reload and verify
                try:
                    loaded = np.load(save_path)
                    loaded_sample = loaded['sample']
                    if loaded_sample.shape != sample_denormalized.shape:
                        raise ValueError(f"[Rank {self.global_rank}] Shape mismatch: saved {sample_denormalized.shape}, loaded {loaded_sample.shape}")
                    loaded.close()
                except Exception as e:
                    error_msg = f"[Rank {self.global_rank}] Failed to verify saved file {save_path}: {e}"
                    if lock:
                        with lock:
                            self.logger.error(error_msg)
                    else:
                        self.logger.error(error_msg)
                    failed_saves.append((timestamp, ensemble_index, str(e)))
                    continue
                
                total_saved += 1
                
                # Save intermediate ODE trajectory snapshots (optional)
                if trajectory_np is not None:
                    try:
                        os.makedirs(self.trajectory_save_dir, exist_ok=True)
                        for ode_t, traj_arr in trajectory_np.items():
                            traj_i = traj_arr[i]
                            if torch.is_tensor(traj_i):
                                traj_i = traj_i.cpu().numpy()
                            traj_denormalized = self.transform.denormalize(traj_i, variable=None)
                            if traj_denormalized is None:
                                raise ValueError(f"Denormalization returned None for trajectory t={ode_t}")
                            if traj_denormalized.shape[-1] == 1:
                                traj_denormalized = traj_denormalized.squeeze(-1)
                            traj_path = os.path.join(
                                self.trajectory_save_dir,
                                f"sample_{timestamp}_ens{ensemble_index:03d}_t{float(ode_t):.2f}.npz"
                            )
                            np.savez_compressed(
                                traj_path,
                                sample=traj_denormalized,
                                ode_time=float(ode_t),
                                sampling_steps=self.sampling_steps,
                                timestamp=timestamp,
                                ensemble_index=ensemble_index,
                                rank=self.global_rank
                            )
                    except Exception as traj_e:
                        traj_err = f"[Rank {self.global_rank}] Failed to save trajectory for {timestamp}_ens{ensemble_index:03d}: {traj_e}"
                        if lock:
                            with lock:
                                self.logger.error(traj_err)
                        else:
                            self.logger.error(traj_err)
                
            except Exception as e:
                timestamp, ensemble_index = batch_metadata[i] if i < len(batch_metadata) else ("UNKNOWN", -1)
                error_msg = f"[Rank {self.global_rank}] Failed to save sample {timestamp}_ens{ensemble_index:03d}: {e}"
                if lock:
                    with lock:
                        self.logger.error(error_msg)
                else:
                    self.logger.error(error_msg)
                failed_saves.append((timestamp, ensemble_index, str(e)))
                continue
        
        if failed_saves:
            error_summary = f"[Rank {self.global_rank}] Subset batch: {len(failed_saves)}/{N} samples failed to save"
            if lock:
                with lock:
                    self.logger.error(error_summary)
                    for ts, ens_idx, err in failed_saves[:5]:
                        self.logger.error(f"  - {ts}_ens{ens_idx:03d}: {err}")
            else:
                self.logger.error(error_summary)
                for ts, ens_idx, err in failed_saves[:5]:
                    self.logger.error(f"  - {ts}_ens{ens_idx:03d}: {err}")
        
        return total_saved

    def _save_samples_grid_sync(self, batch_samples_np, batch_metadata, lock, trajectory_np=None):
        """
        Thread-safe method to save grid-format samples.

        Unlike _save_samples_subset_sync which saves (N, C) node arrays,
        this saves (C, H, W) grid arrays with land pixels set to NaN.

        Args:
            batch_samples_np: Numpy array of shape (N, C, H, W) — normalized grid samples.
            batch_metadata: List of (timestamp, ensemble_index) tuples.
            lock: Threading lock.
            trajectory_np: Optional dict {ode_t: np.ndarray of shape (N, C, H, W)}
                           with intermediate ODE states (saved under
                           self.trajectory_save_dir).

        Returns:
            total_saved: Number of samples saved.
        """
        if batch_samples_np is None:
            raise ValueError(f"[Rank {self.global_rank}] batch_samples_np is None!")
        if batch_metadata is None:
            raise ValueError(f"[Rank {self.global_rank}] batch_metadata is None!")

        if torch.is_tensor(batch_samples_np):
            batch_samples_np = batch_samples_np.cpu().numpy()

        if not isinstance(batch_samples_np, np.ndarray):
            raise TypeError(f"[Rank {self.global_rank}] batch_samples_np must be numpy array, got {type(batch_samples_np)}")

        total_saved = 0
        N = batch_samples_np.shape[0]
        failed_saves = []

        if len(batch_metadata) != N:
            raise ValueError(f"[Rank {self.global_rank}] Metadata count ({len(batch_metadata)}) doesn't match sample count ({N})")

        for i in range(N):
            try:
                sample_grid = batch_samples_np[i]  # (C, H, W) normalized
                timestamp, ensemble_index = batch_metadata[i]

                if self.transform is None:
                    raise ValueError(f"[Rank {self.global_rank}] transform is None!")

                # Denormalize: transpose to (H, W, C) for denormalize(), then back to (C_out, H, W)
                sample_hwc = np.transpose(sample_grid, (1, 2, 0))  # (H, W, C)
                sample_denorm_hwc = self.transform.denormalize(sample_hwc, variable=None)  # (H, W, C_out)
                sample_denorm = np.transpose(sample_denorm_hwc, (2, 0, 1))  # (C_out, H, W)

                # Set land pixels to NaN using land mask.
                # File convention: True = land, False = ocean — need to invert.
                if self.land_mask_file and os.path.exists(self.land_mask_file):
                    land_mask = np.load(self.land_mask_file).astype(bool)  # True=land
                    for c in range(sample_denorm.shape[0]):
                        sample_denorm[c, land_mask] = np.nan

                save_path = os.path.join(
                    self.sample_save_dir,
                    f"sample_{timestamp}_ens{ensemble_index:03d}.npz"
                )

                np.savez_compressed(
                    save_path,
                    sample=sample_denorm,
                    sampling_steps=self.sampling_steps,
                    timestamp=timestamp,
                    ensemble_index=ensemble_index,
                    rank=self.global_rank
                )

                if not os.path.exists(save_path):
                    raise IOError(f"[Rank {self.global_rank}] File was not created: {save_path}")

                total_saved += 1

                # Save intermediate ODE trajectory snapshots (optional)
                if trajectory_np is not None:
                    try:
                        os.makedirs(self.trajectory_save_dir, exist_ok=True)
                        for ode_t, traj_arr in trajectory_np.items():
                            traj_i = traj_arr[i]
                            if torch.is_tensor(traj_i):
                                traj_i = traj_i.cpu().numpy()
                            traj_hwc = np.transpose(traj_i, (1, 2, 0))  # (H, W, C)
                            traj_denorm_hwc = self.transform.denormalize(traj_hwc, variable=None)
                            traj_denorm = np.transpose(traj_denorm_hwc, (2, 0, 1))  # (C_out, H, W)
                            if self.land_mask_file and os.path.exists(self.land_mask_file):
                                land_mask = np.load(self.land_mask_file).astype(bool)  # True=land
                                for c in range(traj_denorm.shape[0]):
                                    traj_denorm[c, land_mask] = np.nan
                            traj_path = os.path.join(
                                self.trajectory_save_dir,
                                f"sample_{timestamp}_ens{ensemble_index:03d}_t{float(ode_t):.2f}.npz"
                            )
                            np.savez_compressed(
                                traj_path,
                                sample=traj_denorm,
                                ode_time=float(ode_t),
                                sampling_steps=self.sampling_steps,
                                timestamp=timestamp,
                                ensemble_index=ensemble_index,
                                rank=self.global_rank
                            )
                    except Exception as traj_e:
                        traj_err = f"[Rank {self.global_rank}] Failed to save grid trajectory for {timestamp}_ens{ensemble_index:03d}: {traj_e}"
                        if lock:
                            with lock:
                                self.logger.error(traj_err)
                        else:
                            self.logger.error(traj_err)

            except Exception as e:
                ts, ens_idx = batch_metadata[i] if i < len(batch_metadata) else ("UNKNOWN", -1)
                error_msg = f"[Rank {self.global_rank}] Failed to save grid sample {ts}_ens{ens_idx:03d}: {e}"
                if lock:
                    with lock:
                        self.logger.error(error_msg)
                else:
                    self.logger.error(error_msg)
                failed_saves.append((ts, ens_idx, str(e)))
                continue

        if failed_saves:
            error_summary = f"[Rank {self.global_rank}] Grid subset: {len(failed_saves)}/{N} samples failed"
            if lock:
                with lock:
                    self.logger.error(error_summary)
            else:
                self.logger.error(error_summary)

        return total_saved

