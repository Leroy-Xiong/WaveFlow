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
    """Restrict this process to one GPU and return its physical index.

    Users usually have one GPU, so do not enumerate every device (querying
    memory on all GPUs can create contexts on all of them). If the caller set
    ``CUDA_VISIBLE_DEVICES``, keep only its first entry; otherwise use GPU 0.
    """
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        first = visible.split(",")[0].strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = first
        physical = int(first) if first.isdigit() else 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        physical = 0
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. The released sampler expects a GPU; "
            "run this cell on a CUDA machine.")
    return physical


def fig3f_from_weights(weights: Path | None = None):
    """Infer the 10-member global Hs ensemble at 2022-09-24 00Z and plot Fig. 3f."""
    weights = Path(weights) if weights else WEIGHTS
    if not weights.exists():
        raise FileNotFoundError(
            f"missing {weights}; run `python reproduce.py setup --weights` first")
    from wb2_era5 import ensure_wb2_era5
    ensure_wb2_era5(
        ["20220924T00"], region="global",
        out_dir=REPO / "data" / "era5" / "wb2", frames=4,
    )
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
