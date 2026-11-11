#!/usr/bin/env python
"""
Interpolate global grid model output (720×1440) to irregular nodes (~652k).

Precomputes KDTree + IDW weights once, then apples them to all sample
.npz files in batch.  Uses multiprocessing for speed.

Usage
-----
    # Precompute weights once (K=4 default):
    python -m modules.preprocess.interpolate_global_grid_to_nodes precompute

    # Interpolate all samples from a run:
    python -m modules.preprocess.interpolate_global_grid_to_nodes interpolate \\
        --sample-dir output/results/global/new_data_hs_grid/p16_vit_huge_xpred/sample_epoch99_20260801_000131

    # Or do both in one shot:
    python -m modules.preprocess.interpolate_global_grid_to_nodes all \\
        --sample-dir output/results/global/new_data_hs_grid/p16_vit_huge_xpred/sample_epoch99_20260801_000131
"""

import os
import sys
import argparse
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np

# Add project root so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.grid_interpolation import GlobalGridToNodeInterpolator

# Default weights output path
DEFAULT_WEIGHTS_DIR = './data/global_hindcast/interpolation_weights'
DEFAULT_K = 4


# ---------------------------------------------------------------------------
# Subcommand: precompute
# ---------------------------------------------------------------------------
def cmd_precompute(args):
    interp = GlobalGridToNodeInterpolator(k_neighbors=args.k)
    interp.precompute_weights(
        weights_dir=args.weights_dir,
        force=args.force,
    )
    print("Precomputation done.")


# ---------------------------------------------------------------------------
# Worker for multiprocessing
# ---------------------------------------------------------------------------
_worker_interp = None  # module-level singleton per worker process


def _init_worker(weights_dir, k):
    global _worker_interp
    _worker_interp = GlobalGridToNodeInterpolator(k_neighbors=k)
    _worker_interp.precompute_weights(weights_dir=weights_dir)


def _process_one(args_tuple):
    """Process a single file.  Returns (fname, success)."""
    global _worker_interp
    npz_path, output_dir, overwrite = args_tuple
    fname = os.path.basename(npz_path)
    out_path = os.path.join(output_dir, fname)

    if os.path.exists(out_path) and not overwrite:
        return (fname, 0)  # skipped

    ok = _worker_interp.interpolate_file(npz_path, out_path)
    return (fname, 1 if ok else -1)


# ---------------------------------------------------------------------------
# Subcommand: interpolate
# ---------------------------------------------------------------------------
def cmd_interpolate(args):
    import glob
    from tqdm import tqdm

    sample_dir = os.path.join(args.sample_dir, "samples")
    if not os.path.isdir(sample_dir):
        sample_dir = args.sample_dir  # fallback
    if not os.path.isdir(sample_dir):
        print(f"ERROR: samples directory not found: {sample_dir}")
        sys.exit(1)

    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(args.sample_dir, "samples_nodes")
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(sample_dir, "sample_*.npz")))
    if not files:
        print(f"No sample_*.npz files found in {sample_dir}")
        return

    total = len(files)
    print(f"Found {total:,} grid-format sample files")
    print(f"Output: {output_dir}")

    n_workers = min(args.num_workers or cpu_count(), 32)
    overwrite = args.overwrite

    task_args = [(f, output_dir, overwrite) for f in files]

    print(f"Using {n_workers} workers, K={args.k} ...")
    t0 = time.time()

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(args.weights_dir, args.k),
    ) as pool:
        results = list(tqdm(
            pool.imap_unordered(_process_one, task_args, chunksize=100),
            total=total,
            desc="Grid → nodes",
        ))

    ok = sum(r[1] == 1 for r in results)
    fail = sum(r[1] == -1 for r in results)
    skipped = sum(r[1] == 0 for r in results)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed/60:.1f} min ({total/elapsed:.0f} files/s)")
    print(f"  OK: {ok:,}  Failed: {fail:,}  Skipped: {skipped:,}")
    print(f"  Output: {output_dir}/")
    print(f"\nNext: python post_process.py rmse-ensemble "
          f"--sample-dir {os.path.dirname(output_dir)}/samples_nodes "
          f"--gt-format npz --num-ensemble 10")


# ---------------------------------------------------------------------------
# Subcommand: all (precompute + interpolate)
# ---------------------------------------------------------------------------
def cmd_all(args):
    print("=" * 60)
    print("[1/2] Precomputing interpolation weights ...")
    print("=" * 60)
    interp = GlobalGridToNodeInterpolator(k_neighbors=args.k)
    interp.precompute_weights(
        weights_dir=args.weights_dir,
        force=args.force,
    )
    print("\n" + "=" * 60)
    print("[2/2] Interpolating all grid samples to nodes ...")
    print("=" * 60)
    cmd_interpolate(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Global grid (720×1440) → irregular nodes interpolation",
    )
    sub = parser.add_subparsers(dest='cmd')

    # ---- precompute ----
    p_pc = sub.add_parser('precompute', help='Precompute KDTree+IDW weights')
    p_pc.add_argument('--k', type=int, default=DEFAULT_K,
                      help='Number of nearest grid cells per node')
    p_pc.add_argument('--weights-dir', default=DEFAULT_WEIGHTS_DIR,
                      help='Directory to save interp weights')
    p_pc.add_argument('--force', action='store_true',
                      help='Recompute even if cached')
    p_pc.set_defaults(func=cmd_precompute)

    # ---- interpolate ----
    p_int = sub.add_parser('interpolate', help='Interpolate sample files to nodes')
    p_int.add_argument('--sample-dir', required=True,
                       help='Run directory containing samples/ subfolder')
    p_int.add_argument('--output-dir', default=None,
                       help='Output directory (default: <sample-dir>/samples_nodes/)')
    p_int.add_argument('--k', type=int, default=DEFAULT_K)
    p_int.add_argument('--weights-dir', default=DEFAULT_WEIGHTS_DIR)
    p_int.add_argument('--num-workers', type=int, default=None,
                       help='Parallel processes (default: cpu_count)')
    p_int.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output files')
    p_int.set_defaults(func=cmd_interpolate)

    # ---- all ----
    p_all = sub.add_parser('all', help='Precompute + interpolate')
    p_all.add_argument('--sample-dir', required=True)
    p_all.add_argument('--output-dir', default=None)
    p_all.add_argument('--k', type=int, default=DEFAULT_K)
    p_all.add_argument('--weights-dir', default=DEFAULT_WEIGHTS_DIR)
    p_all.add_argument('--num-workers', type=int, default=None)
    p_all.add_argument('--force', action='store_true')
    p_all.add_argument('--overwrite', action='store_true')
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == '__main__':
    main()
