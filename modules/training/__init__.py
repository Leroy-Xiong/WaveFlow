"""
Training package: TrainerSampler and DDP/sampling helpers (split from main.py).

  - trainer.py:     TrainerSampler class (init, train, sample, close)
  - mixin_config:   config parameter extraction
  - mixin_env:      DDP setup, work dir, logging, coordinates
  - mixin_saving:   sample save routines (npz writers)
  - mixin_sampling: conditional/unconditional sampling loops
  - mixin_model:    model/optimizer/scheduler/dataset/checkpoint init
  - main_utils:     setup_ddp, cleanup_ddp, set_seed, checkpoint validation
"""
from modules.training.trainer import TrainerSampler
from modules.training.main_utils import (
    setup_ddp,
    cleanup_ddp,
    validate_checkpoint_for_sample_mode,
    set_seed,
)

__all__ = [
    "TrainerSampler",
    "setup_ddp",
    "cleanup_ddp",
    "validate_checkpoint_for_sample_mode",
    "set_seed",
]
