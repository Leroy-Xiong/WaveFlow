#!/usr/bin/env python
"""Extreme-weather evaluation of ensemble samples vs hindcast GT.

For each test timestamp:
  - loads GT (hs, tp) and all ensemble members
  - bins node-hour pairs by GT hs and accumulates RMSE / bias / MAE / CRPS /
    ensemble spread / 90th-percentile coverage
  - records region-max statistics to rank extreme (storm) events

Outputs (under <sample_dir>/extreme_eval/):
  metrics_by_hs_bin.csv   per GT-hs-bin metrics
  top_events.csv          top-N storm events ranked by GT region-max hs
  summary.txt             console summary (also printed)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import re
import sys
import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HS_BINS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, np.inf]
BIN_LABELS = ["0-1m", "1-2m", "2-3m", "3-4m", "4-6m", ">6m"]
VAR_ORDER = ["hs", "tp", "dm", "elev"]


def _hs_bin_idx(v):
    for i, (lo, hi) in enumerate(zip(HS_BINS[:-1], HS_BINS[1:])):
        if lo <= v < hi:
            return i
    return len(HS_BINS) - 2


def process_timestamp(args):
    ts, sample_dir, data_dir, num_ens = args
    sample_files = [
        os.path.join(sample_dir, f"sample_{ts}_ens{i:03d}.npz") for i in range(num_ens)
    ]
    ens = []
    for f in sample_files:
        if not os.path.exists(f):
            return None
        d = np.load(f)
        ens.append(d["sample"].squeeze())
    if not ens:
        return None
    ens = np.stack(ens, axis=0)  # (M, N, C)

    gt_file = os.path.join(data_dir, f"wave_{ts}.npz")
    if not os.path.exists(gt_file):
        return None
    gt = np.load(gt_file)
    gt_hs = gt["hs"].astype(np.float32).squeeze()
    gt_tp = gt["tp"].astype(np.float32).squeeze()

    pred_hs = ens[:, :, 0]  # (M, N)
    pred_tp = ens[:, :, 1]
    mean_hs = pred_hs.mean(axis=0)
    mean_tp = pred_tp.mean(axis=0)

    nb = len(HS_BINS) - 1
    bins = [dict(n=0, ss=0.0, sb=0.0, sae=0.0, crps=0.0, spread=0.0, cov=0,
                 tp_ss=0.0, tp_sae=0.0, tp_crps=0.0, under=0)
            for _ in range(nb)]

    valid = ~np.isnan(gt_hs)
    for b in range(nb):
        m = valid & (gt_hs >= HS_BINS[b]) & (gt_hs < HS_BINS[b + 1])
        n = int(m.sum())
        if n == 0:
            continue
        g = gt_hs[m]
        pm = mean_hs[m]
        pe = pred_hs[:, m]  # (M, n)
        bi = _hs_bin_idx
        bins[b]["n"] = n
        diff = pm - g
        bins[b]["ss"] = float((diff ** 2).sum())
        bins[b]["sb"] = float(diff.sum())
        bins[b]["sae"] = float(np.abs(diff).sum())
        bins[b]["crps"] = float(
            np.abs(pe - g[None, :]).sum() / pe.shape[0]
            - np.abs(pe[:, None, :] - pe[None, :, :]).sum() / (2 * pe.shape[0] ** 2)
        )
        bins[b]["spread"] = float(pe.std(axis=0).sum())
        q05 = np.quantile(pe, 0.05, axis=0)
        q95 = np.quantile(pe, 0.95, axis=0)
        bins[b]["cov"] = int(((g >= q05) & (g <= q95)).sum())
        bins[b]["under"] = int((pm < g).sum())

        pt = mean_tp[m]
        gt2 = gt_tp[m]
        if not np.isnan(gt2).all():
            v = ~np.isnan(gt2)
            bins[b]["tp_ss"] = float(((pt[v] - gt2[v]) ** 2).sum())
            bins[b]["tp_sae"] = float(np.abs(pt[v] - gt2[v]).sum())
            te = pred_tp[:, m]
            bins[b]["tp_crps"] = float(
                np.abs(te - gt2[None, :]).sum() / te.shape[0]
                - np.abs(te[:, None, :] - te[None, :, :]).sum() / (2 * te.shape[0] ** 2)
            )

    ev = {
        "ts": ts,
        "gt_max": float(np.nanmax(gt_hs)),
        "mean_max": float(np.nanmax(mean_hs)),
        "p90_max": float(np.nanmax(np.quantile(pred_hs, 0.90, axis=0))),
        "gt_mean": float(np.nanmean(gt_hs)),
        "mean_rmse": float(np.sqrt(np.nanmean((mean_hs - gt_hs) ** 2))),
    }
    return bins, ev


def parse_args(argv=None):
    """Parse CLI arguments (shared with load_data.py / post_process.py wrappers)."""
    return _build_parser().parse_args(argv)


def _build_parser():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True, help="run dir containing samples/")
    ap.add_argument("--num-ensemble", type=int, default=10)
    ap.add_argument("--data-dir", default="./data/hindcast/processed_npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--top-events", type=int, default=40)
    return ap


def run(args):
    sample_dir = os.path.join(args.sample_dir, "samples")
    ts_list = sorted(
        {re.match(r"sample_(\d{8}T\d{2})_ens\d+\.npz$", f).group(1)
         for f in os.listdir(sample_dir)
         if re.match(r"sample_\d{8}T\d{2}_ens\d+\.npz$", f)}
    )
    print(f"timestamps: {len(ts_list)}")

    tasks = [(ts, sample_dir, args.data_dir, args.num_ensemble) for ts in ts_list]
    nb = len(HS_BINS) - 1
    agg = [dict(n=0, ss=0.0, sb=0.0, sae=0.0, crps=0.0, spread=0.0, cov=0,
                tp_ss=0.0, tp_sae=0.0, tp_crps=0.0, under=0) for _ in range(nb)]
    events = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(process_timestamp, tasks, chunksize=8):
            if res is None:
                continue
            bins, ev = res
            for b in range(nb):
                if bins[b]["n"] == 0:
                    continue
                for k in agg[b]:
                    agg[b][k] += bins[b][k]
            events.append(ev)
            done += 1
            if done % 2000 == 0:
                print(f"  processed {done}/{len(ts_list)}", flush=True)

    events.sort(key=lambda e: e["gt_max"], reverse=True)

    out_dir = os.path.join(args.sample_dir, "extreme_eval")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for b in range(nb):
        a = agg[b]
        n = a["n"]
        r = {
            "bin": BIN_LABELS[b],
            "n_pairs": n,
            "rmse": np.sqrt(a["ss"] / n) if n else np.nan,
            "bias": a["sb"] / n if n else np.nan,
            "mae": a["sae"] / n if n else np.nan,
            "crps": a["crps"] / n if n else np.nan,
            "spread": a["spread"] / n if n else np.nan,
            "cov90_pct": 100.0 * a["cov"] / n if n else np.nan,
            "under_frac": a["under"] / n if n else np.nan,
            "tp_rmse": np.sqrt(a["tp_ss"] / n) if n else np.nan,
            "tp_mae": a["tp_sae"] / n if n else np.nan,
            "tp_crps": a["tp_crps"] / n if n else np.nan,
        }
        rows.append(r)
        print(
            f"{r['bin']:>6s} n={n:>10d} rmse={r['rmse']:.3f} bias={r['bias']:+.3f} "
            f"mae={r['mae']:.3f} crps={r['crps']:.3f} spread={r['spread']:.3f} "
            f"cov90={r['cov90_pct']:5.1f}% under={r['under_frac']:.2f} "
            f"tp_rmse={r['tp_rmse']:.2f}"
        )

    with open(os.path.join(out_dir, "metrics_by_hs_bin.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(out_dir, "top_events.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "gt_max", "mean_max", "p90_max",
                                          "gt_mean", "mean_rmse"])
        w.writeheader()
        for e in events[: args.top_events]:
            w.writerow(e)

    print("\n=== top 15 storm events (by GT region-max hs) ===")
    print(f"{'ts':<12} {'gt_max':>7} {'mean_max':>8} {'p90_max':>8} {'gt_mean':>7} {'rmse':>6}")
    for e in events[:15]:
        print(f"{e['ts']:<12} {e['gt_max']:7.2f} {e['mean_max']:8.2f} "
              f"{e['p90_max']:8.2f} {e['gt_mean']:7.2f} {e['mean_rmse']:6.3f}")

    ev_max = np.array([e["gt_max"] for e in events])
    pr_max = np.array([e["mean_max"] for e in events])
    m = ev_max >= 2.0
    slope, intercept = np.polyfit(ev_max[m], pr_max[m], 1)
    bias = (pr_max[m] - ev_max[m]).mean()
    corr = np.corrcoef(ev_max[m], pr_max[m])[0, 1]
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"timestamps: {len(ts_list)} processed: {done}\n")
        f.write(f"peak (region-max hs >= 2m) n={m.sum()}: slope={slope:.3f} "
                f"intercept={intercept:.3f} bias={bias:+.3f} corr={corr:.3f}\n")
    print(f"\npeak events (gt_max>=2m): n={m.sum()}")
    print(f"  pred vs gt peak regression: slope={slope:.3f} intercept={intercept:.3f}")
    print(f"  peak bias (mean_max - gt_max) = {bias:+.3f} m, corr = {corr:.3f}")
    print(f"\noutputs -> {out_dir}")


def main():
    run(parse_args())

if __name__ == "__main__":
    main()
