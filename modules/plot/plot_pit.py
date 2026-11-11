#!/usr/bin/env python
"""PIT (Probability Integral Transform) calibration diagnostics for ensemble samples.

Streams over all test timestamps; for every node-hour pair and each wave
variable computes PIT = F_pred(y_obs) from the M-member ensemble, then plots
histograms per variable (hs, tp, dm, elev):

  1. overall PIT histogram with uniform reference line
  2. PIT histograms binned by the variable's own GT value (dm: binned by GT hs)
  3. overlaid PIT density per bin

Interpretation:
  U-shape (mass at both edges) -> under-dispersive (over-confident)
  peaked center               -> over-dispersive
  mass shifted toward 1       -> systematic under-prediction
  mass shifted toward 0       -> systematic over-prediction

Supported sample formats (auto-detected):
  - hk:     sample (N, 4) per member [hs, tp, dm, elev]; GT {data_dir}/wave_{ts}.npz
  - global: sample (N,)   per member (single-variable hs);
            GT {data_dir}/{year}/global_hindcast_{YYYYMMDD_HH}.npz
            -> pass --data-dir ./data/global_hindcast/npz_files

Outputs -> <sample_dir>/extreme_eval/pit/
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import re
import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

N_PIT_BINS = 20  # fine resolution; randomised PIT is continuous (10 bins only resolved M=10 ranks)

VAR_ORDER = ["hs", "tp", "dm", "elev"]
VAR_CFG = {
    "hs":   dict(bins=[0.0, 1.0, 2.0, 3.0, 4.0, np.inf],
                 labels=["0-1m", "1-2m", "2-3m", "3-4m", ">4m"], bin_by="hs"),
    "tp":   dict(bins=[0.0, 4.0, 6.0, 8.0, 10.0, np.inf],
                 labels=["0-4s", "4-6s", "6-8s", "8-10s", ">10s"], bin_by="tp"),
    "elev": dict(bins=[-np.inf, -0.5, 0.0, 0.5, np.inf],
                 labels=["<-0.5m", "-0.5-0m", "0-0.5m", ">0.5m"], bin_by="elev"),
    "dm":   dict(bins=[0.0, 1.0, 2.0, 3.0, 4.0, np.inf],
                 labels=["hs 0-1m", "hs 1-2m", "hs 2-3m", "hs 3-4m", "hs >4m"], bin_by="hs"),
}
GLOBAL_VAR_CFG = {
    "hs":   dict(bins=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, np.inf],
                 labels=["0-1m", "1-2m", "2-3m", "3-4m", "4-6m", ">6m"], bin_by="hs"),
}


def detect_format(sample_dir):
    """Auto-detect sample layout: 'hk' (multi-var) vs 'global' (single-var hs)."""
    for f in os.listdir(sample_dir):
        if re.match(r"sample_\d{8}T\d{2}_ens\d+\.npz$", f):
            s = np.load(os.path.join(sample_dir, f))["sample"].squeeze()
            return "hk" if s.ndim == 2 else "global"
    raise SystemExit(f"no sample files found in {sample_dir}")


def load_gt(data_dir, ts, fmt):
    if fmt == "hk":
        path = os.path.join(data_dir, f"wave_{ts}.npz")
        return path, np.load(path)
    path = os.path.join(data_dir, ts[:4], f"global_hindcast_{ts[:8]}_{ts[9:11]}.npz")
    return path, np.load(path)


def process_timestamp(args):
    ts, sample_dir, data_dir, num_ens, fmt = args
    sample_files = [
        os.path.join(sample_dir, f"sample_{ts}_ens{i:03d}.npz") for i in range(num_ens)
    ]
    ens = []
    for f in sample_files:
        if not os.path.exists(f):
            return None
        ens.append(np.load(f)["sample"].squeeze())
    if len(ens) < 2:
        return None
    ens = np.stack(ens, axis=0)  # hk: (M, N, 4); global: (M, N)
    gt_path, gt = load_gt(data_dir, ts, fmt)
    if not os.path.exists(gt_path):
        return None
    M = ens.shape[0]
    rng = np.random.default_rng(int(ts.replace("T", "")) % 2**32)
    var_order = VAR_ORDER if fmt == "hk" else ["hs"]
    var_cfg = VAR_CFG if fmt == "hk" else GLOBAL_VAR_CFG

    result = {}
    for var in var_order:
        cfg = var_cfg[var]
        gv = gt[var].astype(np.float32).squeeze()
        valid = ~np.isnan(gv)
        if fmt == "hk":
            ev = ens[:, :, var_order.index(var)][:, valid]  # (M, n)
        else:
            ev = ens[:, valid]
        g = gv[valid]
        less = (ev < g[None, :]).sum(axis=0)
        eq = (ev == g[None, :]).sum(axis=0)
        # Randomised PIT: jitter within each discrete rank (unbiased)
        pit = (less + 0.5 * eq + rng.random(less.shape) * 0.9999) / M
        pit = np.clip(pit, 0, 1)
        h_all = np.histogram(pit, bins=N_PIT_BINS, range=(0, 1))[0]
        # bin by the configured variable's GT value
        bv = gt[cfg["bin_by"]].astype(np.float32).squeeze()[valid]
        h_by_bin = []
        for b in range(len(cfg["labels"])):
            m = (bv >= cfg["bins"][b]) & (bv < cfg["bins"][b + 1])
            h_by_bin.append(np.histogram(pit[m], bins=N_PIT_BINS, range=(0, 1))[0]
                            if m.sum() else np.zeros(N_PIT_BINS, dtype=np.int64))
        result[var] = (h_all, np.stack(h_by_bin), int(valid.sum()))
    return result


def parse_args(argv=None):
    """Parse CLI arguments (shared with load_data.py / post_process.py wrappers)."""
    return _build_parser().parse_args(argv)


def _build_parser():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True, help="run dir containing samples/")
    ap.add_argument("--num-ensemble", type=int, default=10)
    ap.add_argument("--data-dir", default="./data/hindcast/processed_npz",
                    help="GT data dir (HK: processed_npz; global: global_hindcast/npz_files)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--variables", default="all",
                    help="comma-separated vars or 'all' (default: all)")
    ap.add_argument("--format", choices=["auto", "hk", "global"], default="auto",
                    help="sample layout; auto-detected from first sample file (default: auto)")
    ap.add_argument("--compare-dir", default=None,
                    help="optional second sample dir (e.g. raw/uncalibrated) to add a "
                         "before-vs-after calibration comparison figure")
    return ap


def _list_timestamps(sample_dir):
    return sorted(
        {re.match(r"sample_(\d{8}T\d{2})_ens\d+\.npz$", f).group(1)
         for f in os.listdir(sample_dir)
         if re.match(r"sample_\d{8}T\d{2}_ens\d+\.npz$", f)}
    )


def _aggregate(sample_dir, data_dir, num_ens, fmt, variables, var_cfg, workers):
    ts_list = _list_timestamps(sample_dir)
    print(f"timestamps: {len(ts_list)}  ({sample_dir})")
    tasks = [(ts, sample_dir, data_dir, num_ens, fmt) for ts in ts_list]
    agg = {var: (np.zeros(N_PIT_BINS, dtype=np.int64),
                 np.zeros((len(var_cfg[var]["labels"]), N_PIT_BINS), dtype=np.int64),
                 0) for var in variables}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(process_timestamp, tasks, chunksize=8):
            if res is None:
                continue
            for var in variables:
                ha, hb, n = res[var]
                a0, a1, a2 = agg[var]
                agg[var] = (a0 + ha, a1 + hb, a2 + int(n))
            done += 1
            if done % 2000 == 0:
                print(f"  processed {done}/{len(ts_list)}", flush=True)
    return agg


def run(args):
    fmt = args.format
    sample_dir = os.path.join(args.sample_dir, "samples")
    if fmt == "auto":
        fmt = detect_format(sample_dir)
        print(f"detected sample format: {fmt}")
    var_order = VAR_ORDER if fmt == "hk" else ["hs"]
    var_cfg = VAR_CFG if fmt == "hk" else GLOBAL_VAR_CFG

    variables = var_order if args.variables == "all" else [v.strip() for v in args.variables.split(",")]

    agg = _aggregate(sample_dir, args.data_dir, args.num_ensemble, fmt,
                     variables, var_cfg, args.workers)
    agg_cmp = None
    if getattr(args, "compare_dir", None):
        cmp_dir = os.path.join(args.compare_dir, "samples")
        agg_cmp = _aggregate(cmp_dir, args.data_dir, args.num_ensemble, fmt,
                             variables, var_cfg, args.workers)

    bin_edges = np.linspace(0, 1, N_PIT_BINS + 1)
    bin_c = (bin_edges[:-1] + bin_edges[1:]) / 2
    out_dir = os.path.join(args.sample_dir, "extreme_eval", "pit")
    os.makedirs(out_dir, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(args.sample_dir))

    def coverage(h, lo=0.1, hi=0.9):
        inside = h[(bin_edges[:-1] >= lo) & (bin_edges[1:] <= hi)].sum()
        return 100.0 * inside / h.sum()

    width = bin_edges[1] - bin_edges[0]
    overall = {}
    for var in variables:
        h_all, h_by_bin, n_total = agg[var]
        cfg = var_cfg[var]
        if n_total == 0:
            continue
        # CSV
        with open(os.path.join(out_dir, f"pit_histograms_{var}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bucket_center", "overall"] + [f"{var}_{lb}" for lb in cfg["labels"]])
            for i in range(N_PIT_BINS):
                w.writerow([f"{bin_c[i]:.3f}"] + [h_all[i]]
                           + [h_by_bin[b, i] for b in range(len(cfg["labels"]))])
        # summary
        print(f"\n[{var}] coverage (PIT in [0.1,0.9]) | low(<0.05) / high(>0.95)")
        print(f"  overall: {coverage(h_all):5.1f}%   "
              f"low={100*h_all[bin_edges[:-1] < 0.05].sum()/h_all.sum():4.1f}%  "
              f"high={100*h_all[bin_edges[1:] > 0.95].sum()/h_all.sum():4.1f}%")
        for b in range(len(cfg["labels"])):
            h = h_by_bin[b]
            if h.sum() == 0:
                continue
            print(f"  {cfg['labels'][b]:>8s}: {coverage(h):5.1f}%   "
                  f"low={100*h[bin_edges[:-1] < 0.05].sum()/h.sum():4.1f}%  "
                  f"high={100*h[bin_edges[1:] > 0.95].sum()/h.sum():4.1f}%")
        overall[var] = (h_all, h_by_bin, n_total)
        # Fig: per-bin panels (density axis, uniform = 1)
        ncol = 3
        nrow = int(np.ceil((len(cfg["labels"]) + 1) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(14, 4.2 * nrow), squeeze=False)
        panels = [(None, h_all, f"overall\nn={h_all.sum():,}")] + [
            (cfg["labels"][b], h_by_bin[b], f"{cfg['labels'][b]}\nn={h_by_bin[b].sum():,}")
            for b in range(len(cfg["labels"]))]
        for ax, (label, h, title) in zip(axes.ravel(), panels):
            den = h / (h.sum() * width) if h.sum() else np.zeros_like(h, dtype=float)
            ax.bar(bin_c, den, width=width, color="#4C72B0", alpha=0.85)
            ax.axhline(1.0, color="crimson", ls="--", lw=1.2)
            ax.set_title(title, fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 2)
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        fig.suptitle(f"[{var}] PIT by GT value", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"pit_by_bin_{var}.png"))
        plt.close(fig)

    plot_vars = list(overall.keys())
    if not plot_vars:
        print(f"\nfigures -> {out_dir}")
        return
    ncol = 2
    nrow = int(np.ceil(len(plot_vars) / ncol))

    # Fig: combined overall histograms, all variables in one figure
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 4.8 * nrow), squeeze=False)
    for ax, var in zip(axes.ravel(), plot_vars):
        h_all, _, n_total = overall[var]
        ax.bar(bin_c, h_all / (h_all.sum() * width), width=width, color="#4C72B0", alpha=0.85)
        ax.axhline(1.0, color="crimson", ls="--", lw=1.5, label="uniform reference")
        ax.set_xlabel(f"PIT  $F_{{\\mathrm{{pred}}}}({var}_{{\\mathrm{{obs}}}})$")
        ax.set_ylabel("density")
        ax.set_title(f"[{var}] overall  (n={n_total/1e6:.1f}M)", fontsize=12)
        ax.set_ylim(0, 2)
        ax.legend()
    for ax in axes.ravel()[len(plot_vars):]:
        ax.axis("off")
    fig.suptitle(f"Overall PIT — {run_name} (M={args.num_ensemble})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pit_overall_all.png"))
    plt.close(fig)

    # Fig: combined overlaid densities, all variables in one figure
    fig, axes = plt.subplots(nrow, ncol, figsize=(8 * ncol, 5.5 * nrow), squeeze=False)
    for ax, var in zip(axes.ravel(), plot_vars):
        h_all, h_by_bin, _ = overall[var]
        cfg = var_cfg[var]
        cmap = plt.cm.viridis(np.linspace(0, 0.9, len(cfg["labels"])))
        ax.plot(bin_c, h_all / (h_all.sum() * width), "-o", ms=4, lw=2,
                color="#E69F00", label=f"overall (n={h_all.sum():,})")
        for b in range(len(cfg["labels"])):
            h = h_by_bin[b]
            if h.sum() == 0:
                continue
            ax.plot(bin_c, h / (h.sum() * width), "o-", ms=3, color=cmap[b],
                    label=f"{cfg['labels'][b]} (n={h.sum():,})")
        ax.plot(bin_c, np.ones(N_PIT_BINS), "--", color="crimson", lw=1.5, label="uniform")
        ax.set_xlabel(f"PIT  $F_{{\\mathrm{{pred}}}}({var}_{{\\mathrm{{obs}}}})$")
        ax.set_ylabel("density")
        ax.set_title(f"[{var}] PIT density by bin", fontsize=12)
        ax.set_ylim(0, 2)
        ax.legend(fontsize=9)
    for ax in axes.ravel()[len(plot_vars):]:
        ax.axis("off")
    fig.suptitle(f"PIT density by GT bin — {run_name} (M={args.num_ensemble})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pit_by_bin_overlay_all.png"))
    plt.close(fig)

    # Fig: before-vs-after calibration comparison (--compare-dir, e.g. raw samples)
    if agg_cmp is not None:
        print("\nPIT coverage (PIT in [0.1,0.9]) before -> after calibration")
        for var in plot_vars:
            h_raw = agg_cmp[var][0]
            print(f"  [{var}] {coverage(h_raw):5.1f}%  ->  {coverage(overall[var][0]):5.1f}%")
        fig, axes = plt.subplots(nrow, ncol, figsize=(8 * ncol, 5.2 * nrow), squeeze=False)
        for ax, var in zip(axes.ravel(), plot_vars):
            h_cal, _, n_total = overall[var]
            h_raw = agg_cmp[var][0]
            if h_raw.sum() == 0:
                continue
            ax.plot(bin_c, h_raw / (h_raw.sum() * width), "o--", ms=4, lw=1.6, color="gray",
                    label=f"before (raw, n={h_raw.sum():,})")
            ax.plot(bin_c, h_cal / (h_cal.sum() * width), "o-", ms=4, lw=2, color="#4C72B0",
                    label=f"after (calibrated, n={h_cal.sum():,})")
            ax.plot(bin_c, np.ones(N_PIT_BINS), "--", color="crimson", lw=1.5, label="uniform")
            ax.set_xlabel(f"PIT  $F_{{\\mathrm{{pred}}}}({var}_{{\\mathrm{{obs}}}})$")
            ax.set_ylabel("density")
            ax.set_title(f"[{var}] PIT before vs after calibration", fontsize=12)
            ax.set_ylim(0, 2)
            ax.legend(fontsize=9)
        for ax in axes.ravel()[len(plot_vars):]:
            ax.axis("off")
        fig.suptitle(f"PIT before vs after calibration — {run_name} (M={args.num_ensemble})",
                     fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pit_before_after.png"))
        plt.close(fig)

    print(f"\nfigures -> {out_dir}")


def main():
    run(parse_args())

if __name__ == "__main__":
    main()
