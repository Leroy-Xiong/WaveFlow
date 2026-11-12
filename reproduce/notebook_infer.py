"""Notebook helper: run the released weights on one GPU, then plot.

The heavy lifting stays in ``reproduce.py infer`` (stubs -> torchrun sampling
-> render). This module only selects a GPU, checks the downloaded checkpoint,
runs the sampler with ``--no-render``, points the plotting code at the fresh
samples, and returns the standalone figure.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "weights" / "waveflow-global-hs-ep99.pt"


def pick_gpu() -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. The released sampler expects a GPU; "
            "run this cell on a CUDA machine.")
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        try:
            free, _total = torch.cuda.mem_get_info(i)
        except Exception:
            free = 0
        if free > best_free:
            best, best_free = i, free
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best)
    return best


def fig3f_from_weights(weights: Path | None = None):
    """Infer the 10-member global Hs ensemble at 2022-09-24 00Z and plot Fig. 3f."""
    weights = Path(weights) if weights else WEIGHTS
    if not weights.exists():
        raise FileNotFoundError(
            f"missing {weights}; run `python reproduce.py setup --weights` first")
    gpu = pick_gpu()
    print(f"using GPU {gpu} for inference; weights: {weights}")
    cmd = [sys.executable, "reproduce.py", "infer", "fig3f", "--no-render"]
    subprocess.run(cmd, cwd=str(REPO), check=True)
    samples = REPO / "output" / "repro" / "fig3f" / "samples"
    if not samples.exists():
        raise FileNotFoundError(f"inference finished but {samples} is missing")
    os.environ["WF_GLOBAL_HS_SAMPLES"] = str(samples)
    sys.path.insert(0, str(REPO / "reproduce"))
    import notebook_standalone
    return notebook_standalone.fig3f()
