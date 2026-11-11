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




class TrainerEnvMixin:
    def _setup_ddp(self):
        """
        Setup DDP (Distributed Data Parallel) configuration.
        This method initializes local rank, global rank, world size,
        and sets the appropriate CUDA device for this process.

        Raises:
            RuntimeError: If required DDP environment variables are not set.
        """
        # Check for required DDP environment variables
        required_env_vars = ["LOCAL_RANK", "RANK", "WORLD_SIZE"]
        missing_vars = [var for var in required_env_vars if var not in os.environ]

        if missing_vars:
            raise RuntimeError(
                f"DDP environment variables not set: {missing_vars}. "
                f"Please launch with torchrun or set these variables manually. "
                f"For single-GPU training, use: LOCAL_RANK=0 RANK=0 WORLD_SIZE=1"
            )

        # Get local rank (GPU ID on the specific node) and global rank
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        # Set the device for this process
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        # Enable GPU performance optimizations (TF32, cuDNN benchmark)
        enable_gpu_optimizations()
    
    def _create_work_dir(self):
        """
        Create a unique work directory for this training run.
        Directory name format: {base_dir}/{prefix}_YYYYMMDD_HHMMSS
        Also copies the configuration file and all Python scripts to the work directory for reference.
        """
        if self.work_dir is None:
            # Use Beijing time (UTC+8) for work directory timestamp
            from datetime import timezone, timedelta
            beijing_now = datetime.now(tz=timezone(timedelta(hours=8)))
            timestamp = beijing_now.strftime("%Y%m%d_%H%M%S")
            self.work_dir = os.path.join(self.work_dir_base, f"{self.work_dir_prefix}_{timestamp}")
        
        os.makedirs(self.work_dir, exist_ok=True)
        print(f"Work directory created: {self.work_dir}")
        
        # Save resolved configuration (reflects -o overrides) to work directory
        config_backup_path = os.path.join(self.work_dir, "config.yaml")
        with open(config_backup_path, 'w') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False)
        print(f"Resolved configuration saved to: {config_backup_path}")
        
        # Backup all Python scripts to work directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        code_backup_dir = backup_python_scripts(self.work_dir, project_root)
        print(f"All Python scripts backed up to: {code_backup_dir}")
    
    def _setup_logging(self):
        """
        Setup logging to file and console.
        All output will be written to a log file in the work directory.
        Timestamps are in Beijing time (UTC+8).
        """
        # Set log file name based on mode: 'training.log' for train, 'sampling.log' for sample
        log_filename = "training.log" if self.mode == 'train' else "sampling.log"
        log_file = os.path.join(self.work_dir, log_filename)

        # Use Beijing time (UTC+8) for all log timestamps
        beijing_tz = datetime.now().astimezone().tzinfo.__class__
        try:
            from datetime import timezone, timedelta
            beijing_tz = timezone(timedelta(hours=8), name='CST')
        except Exception:
            pass

        class BeijingFormatter(logging.Formatter):
            """Formatter that converts timestamps to Beijing time (UTC+8)."""
            def formatTime(self, record, datefmt=None):
                from datetime import datetime, timezone, timedelta
                ct = datetime.fromtimestamp(record.created, tz=timezone(timedelta(hours=8)))
                if datefmt:
                    return ct.strftime(datefmt)
                return ct.strftime('%Y-%m-%d %H:%M:%S')

        formatter = BeijingFormatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, stream_handler]
        )

        self.logger = logging.getLogger(__name__)

        # Capture warnings from the warnings module
        logging.captureWarnings(True)
        
        # Define an exception handler to log uncaught exceptions
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            self.logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
            
        # Set the exception hook
        sys.excepthook = handle_exception

        self.logger.info(f"Logging initialized. Log file: {log_file}")
        self.logger.info(f"Work directory: {self.work_dir}")
        
        # Log learning rate calculation details
        if hasattr(self, '_lr_calculation_info'):
            lr_info = self._lr_calculation_info
            if lr_info['mode'] == 'absolute':
                self.logger.info(f"Learning rate (absolute): {lr_info['learning_rate']}")
            else:
                self.logger.info(f"Learning rate (relative calculation):")
                self.logger.info(f"  Base learning rate: {lr_info['base_learning_rate']}")
                self.logger.info(f"  Base batch size: {lr_info['base_batch_size']}")
                self.logger.info(f"  Current batch size (per GPU): {lr_info['batch_size_per_gpu']}")
                self.logger.info(f"  Number of GPUs: {lr_info['num_gpus']}")
                self.logger.info(f"  Global batch size: {lr_info['global_batch_size']}")
                self.logger.info(f"  Effective learning rate: {lr_info['effective_learning_rate']}")
        
        # Initialize TensorBoard writer only in training mode
        if self.mode == 'train':
            self.tb_writer = SummaryWriter(log_dir=self.work_dir)
        else:
            self.tb_writer = None
    
    def _load_coordinates(self):
        """
        Load node coordinates from the specified npy file.
        Also load cluster information if use_patches is enabled.
        """
        # Grid model needs node coordinates for sampling (num_nodes calculation)
        # and post-processing interpolation reference
        if self.region == 'hk_grid':
            self.coords = np.load(self.coord_file_path)
            if self.global_rank == 0:
                self.logger.info(f"Grid model: loaded {self.coords.shape[0]} node coordinates for sampling reference")
            return
        self.coords = np.load(self.coord_file_path)
        
        # Load cluster information if patch mode is enabled
        if self.use_patches:
            if not os.path.exists(self.cluster_file):
                raise FileNotFoundError(
                    f"Cluster file not found: {self.cluster_file}\n"
                    "Please run clustering first or disable use_patches."
                )
            
            cluster_data = np.load(self.cluster_file)
            self.patch_centroids = cluster_data['centers']  # [n_clusters, 2]
            self.mask_template = cluster_data['mask_template']  # [n_clusters, max_points]
            self.cluster_labels = cluster_data['labels']  # [num_nodes] - cluster assignment for each node
            self.n_clusters = int(cluster_data['n_clusters'])
            
            # Load local coords based on local_coord_type config
            if self.local_coord_type == '3d':
                if 'local_coords_3d_template' not in cluster_data:
                    raise ValueError(
                        f"3D local coordinates not found in cluster file: {self.cluster_file}\n"
                        f"Please re-run clustering with the updated coordinate_utils.py to generate 3D coordinates."
                    )
                self.local_coords_template = cluster_data['local_coords_3d_template']  # [n_clusters, max_points, 3]
            elif self.local_coord_type == 'none':
                # Create dummy local coords with 0 channels (will be handled by model)
                max_points = cluster_data['mask_template'].shape[1]
                self.local_coords_template = np.zeros((self.n_clusters, max_points, 0), dtype=np.float32)
            else:  # '2d' (default)
                self.local_coords_template = cluster_data['local_coords_template']  # [n_clusters, max_points, 2]
            
            if self.global_rank == 0:
                print(f"Loaded cluster data from: {self.cluster_file}")
                print(f"Number of clusters: {self.n_clusters}")
                print(f"Local coord type: {self.local_coord_type}")
                print(f"Patch centroids shape: {self.patch_centroids.shape}")
                print(f"Local coords template shape: {self.local_coords_template.shape}")
                print(f"Mask template shape: {self.mask_template.shape}")
                print(f"Cluster labels shape: {self.cluster_labels.shape}")
    
