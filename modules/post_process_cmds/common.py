"""Shared helpers for post_process subcommands."""
from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=None)
def _get_postprocess(sample_dir: str, region_mode: str, gt_file_format: str):
    """Lazy / cached PostProcess factory so subcommands share an instance."""
    from modules.postprocess_samples import PostProcess
    return PostProcess(
        sample_dir=os.path.join(sample_dir, "samples"),
        region_mode=region_mode,
        gt_file_format=gt_file_format,
    )
