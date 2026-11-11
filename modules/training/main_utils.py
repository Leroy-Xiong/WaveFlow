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




def setup_ddp():
    """Initializes the DDP process group."""
    from datetime import timedelta
    # backend="nccl" is the best for NVIDIA GPUs
    # Long timeout to tolerate torch.compile recompilation on one rank.
    # 3h also covers eval-sampling shard imbalance: one rank may finish its
    # (already-sampled) shard early and wait at the final barrier while the
    # other rank grinds through a partially-completed shard alone.
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=3))

def cleanup_ddp():
    """Destroys the process group."""
    dist.destroy_process_group()

def validate_checkpoint_for_sample_mode(args, config):
    """
    Validate checkpoint configuration for sample mode.
    
    Args:
        args: Command line arguments containing mode
        config: Configuration dictionary
    
    Raises:
        SystemExit: If checkpoint validation fails
    """
    if args.mode == 'sample':
        global_rank = int(os.environ["RANK"])
        resume_path = config['checkpoint'].get('resume_path', None)
        if resume_path is None:
            if global_rank == 0:
                print("ERROR: For sample mode, you must set 'checkpoint.resume_path' in configs.yaml")
            cleanup_ddp()
            sys.exit(1)
        if not os.path.exists(resume_path):
            if global_rank == 0:
                print(f"ERROR: Checkpoint file not found: {resume_path}")
            cleanup_ddp()
            sys.exit(1)

def set_seed(seed=42, deterministic=False):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42)
        deterministic: If True, enable deterministic cuDNN operations (slower but reproducible).
                       If False, enable cuDNN benchmark for faster performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN settings: deterministic=False + benchmark=True for performance
    # deterministic=True only when reproducibility is critical
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


