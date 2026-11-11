#!/usr/bin/env python
"""Error-vs-value boxplots for ensemble samples vs hindcast GT.

For every test timestamp and each wave variable (hs, tp, dm, elev), bins
node-hour pairs by the variable's own GT value and collects the signed
prediction error of the ensemble mean (dm: shortest angular error in deg):

  - hs   bins: 0-1, 1-2, 2-3, 3-4, 4-6, >6 m
  - tp   bins: 0-4, 4-6, 6-8, 8-10, >10 s
  - dm   bins: 0-90, 90-180, 180-270, 270-360 deg
  - elev bins: <-0.5, -0.5-0, 0-0.5, >0.5 m

Per-bin error distributions are subsampled (capped) for tractable boxplots;
totals / RMSE / MAE / bias are exact.

Outputs -> <sample_dir>/extreme_eval/error_box/:
  error_box_by_value_error.png  2x2 boxplots of signed node error
  error_box_by_value_rmse.png   2x2 boxplots of per-timestamp RMSE
  error_box_by_value_mae.png    2x2 boxplots of per-timestamp MAE
  error_box_by_value_bias.png   2x2 boxplots of per-timestamp bias
  error_box_<var>.png           per-variable figure (signed + absolute error)
  error_box_stats.csv           per-bin stats (n, rmse, mae, bias, p5/p50/p95,
                                n_ts, per-timestamp median of rmse/mae/bias)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

VAR_ORDER = ["hs", "tp", "dm", "elev"]
# node mode: bin each node-hour pair by its own GT value
VAR_CFG = {
    "hs":   dict(bins=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, np.inf],
                 labels=["0-1m", "1-2m", "2-3m", "3-4m", "4-6m", ">6m"]),
    "tp":   dict(bins=[0.0, 4.0, 6.0, 8.0, 10.0, np.inf],
                 labels=["0-4s", "4-6s", "6-8s", "8-10s", ">10s"]),
    "dm":   dict(bins=[0.0, 90.0, 180.0, 270.0, 360.0],
                 labels=["0-90d", "90-180d", "180-270d", "270-360d"]),
    "elev": dict(bins=[-np.inf, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, np.inf],
                 labels=["<-1.0m", "-1~-0.5m", "-0.5~-0.25m", "-0.25~0m",
                         "0~0.25m", "0.25~0.5m", "0.5~1m", ">1.0m"]),
}
GLOBAL_VAR_CFG = {
    "hs": dict(bins=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, np.inf],
               labels=["0-1m", "1-2m", "2-3m", "3-4m", "4-6m", ">6m"]),
}
# timestamp mode: bin each whole timestamp by the region-mean GT value of the
# variable (dm: binned by region-mean hs, like the PIT convention)
TS_VAR_CFG = {
    "hs":   dict(bins=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, np.inf],
                 labels=["0-0.25m", "0.25-0.5m", "0.5-0.75m", "0.75-1m",
                         "1-1.5m", "1.5-2m", ">2m"]),
    "tp":   dict(bins=[0.0, 4.0, 5.0, 6.0, 7.0, 8.0, np.inf],
                 labels=["0-4s", "4-5s", "5-6s", "6-7s", "7-8s", ">8s"]),
    "dm":   dict(bins=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, np.inf],
                 labels=["hs 0-0.25m", "hs 0.25-0.5m", "hs 0.5-0.75m",
                         "hs 0.75-1m", "hs 1-1.5m", "hs 1.5-2m", "hs >2m"],
                 bin_by="hs"),
    "elev": dict(bins=[-np.inf, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, np.inf],
                 labels=["<-1.0m", "-1~-0.5m", "-0.5~-0.25m", "-0.25~0m",
                         "0~0.25m", "0.25~0.5m", "0.5~1m", ">1.0m"]),
}

CAP_PER_BIN = 50000  # max error samples kept per (var, bin) for the boxplot


def detect_format(sample_dir):
    for f in os.listdir(sample_dir):
        if re.match(r"sample_\d{8}T\d{2}_ens\d+\.npz$", f):
            s = np.load(os.path.join(sample_dir, f))["sample"].squeeze()
            return "hk" if s.ndim == 2 else "global"
    raise SystemExit(f"no sample files found in {sample_dir}")


def _seeded_rng(ts, var, b):
    return np.random.default_rng((int(ts.replace("T", "")) * 7 + hash(var) * 31 + b) % 2**32)


def process_timestamp(args):
    ts, sample_dir, data_dir, num_ens, fmt, group_by = args
    sample_files = [
        os.path.join(sample_dir, f"sample_{ts}_ens{i:03d}.npz") for i in range(num_ens)
    ]
    ens = []
    for f in sample_files:
        if not os.path.exists(f):
            return None
        ens.append(np.load(f)["sample"].squeeze())
    if not ens:
        return None
    ens = np.stack(ens, axis=0)  # hk: (M, N, 4); global: (M, N)
    mean = ens.mean(axis=0)      # (N, C) or (N,)

    var_order = VAR_ORDER if fmt == "hk" else ["hs"]
    var_cfg = (TS_VAR_CFG if group_by == "timestamp" else VAR_CFG) if fmt == "hk" \
        else GLOBAL_VAR_CFG
    if fmt == "hk":
        gt_path = os.path.join(data_dir, f"wave_{ts}.npz")
    else:
        gt_path = os.path.join(data_dir, ts[:4], f"global_hindcast_{ts[:8]}_{ts[9:11]}.npz")
    if not os.path.exists(gt_path):
        return None
    gt = np.load(gt_path)

    # timestamp-mode: region-mean GT value of each variable (dm: mean hs)
    ts_key = {}
    if group_by == "timestamp":
        for var in var_order:
            cfg = var_cfg[var]
            bv = gt[cfg.get("bin_by", var)].astype(np.float32).squeeze()
            bv = bv[~np.isnan(bv)]
            if bv.size == 0:
                return None
            if cfg.get("bin_by") == "hs" or var == "hs":
                ts_key[var] = float(bv.mean())
            elif var == "dm":
                r = np.deg2rad(bv.astype(np.float64))
                ts_key[var] = float(np.rad2deg(np.arctan2(np.sin(r).mean(),
                                                          np.cos(r).mean())))
            else:
                ts_key[var] = float(bv.mean())

    result = {}
    for i, var in enumerate(var_order):
        cfg = var_cfg[var]
        gv = gt[var].astype(np.float32).squeeze()
        valid = ~np.isnan(gv)
        p = mean[:, i] if fmt == "hk" else mean
        err = (p - gv)[valid].astype(np.float32)
        g = gv[valid]
        if var == "dm":
            err = (err + 180.0) % 360.0 - 180.0  # shortest angular error in deg
        nb = len(cfg["labels"])
        counts = np.zeros(nb, dtype=np.int64)
        picks = [[] for _ in range(nb)]
        ts_r = [[] for _ in range(nb)]  # per-timestamp RMSE/MAE/bias within each bin
        ts_a = [[] for _ in range(nb)]
        ts_b = [[] for _ in range(nb)]
        if group_by == "timestamp":
            b = int(np.searchsorted(cfg["bins"], ts_key[var], side="right") - 1)
            b = max(0, min(b, nb - 1))
            counts[b] = int(valid.sum())
            if counts[b]:
                idx = np.nonzero(valid)[0]
                take = min(len(idx), CAP_PER_BIN)
                if take < len(idx):
                    idx = _seeded_rng(ts, var, b).choice(idx, size=take, replace=False)
                picks[b] = err[idx]
                ts_r[b] = [float(np.sqrt(np.mean(err ** 2)))]
                ts_a[b] = [float(np.mean(np.abs(err)))]
                ts_b[b] = [float(np.mean(err))]
        else:
            for b in range(nb):
                lo, hi = cfg["bins"][b], cfg["bins"][b + 1]
                m = (g >= lo) & (g < hi)
                counts[b] = int(m.sum())
                if counts[b] == 0:
                    continue
                eb = err[m]
                idx = np.nonzero(m)[0]
                take = min(len(idx), CAP_PER_BIN)
                if take < len(idx):
                    idx = _seeded_rng(ts, var, b).choice(idx, size=take, replace=False)
                picks[b] = err[idx]
                ts_r[b] = [float(np.sqrt(np.mean(eb ** 2)))]
                ts_a[b] = [float(np.mean(np.abs(eb)))]
                ts_b[b] = [float(np.mean(eb))]
        result[var] = (counts, picks, ts_r, ts_a, ts_b)
    return result


def parse_args(argv=None):
    return _build_parser().parse_args(argv)


def _build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True, help="run dir containing samples/")
    ap.add_argument("--num-ensemble", type=int, default=10)
    ap.add_argument("--data-dir", default="./data/hindcast/processed_npz")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--group-by", choices=["node", "timestamp"], default="node",
                    help="classification unit: 'node' bins each node-hour pair by its "
                         "own GT value; 'timestamp' bins each whole timestamp by the "
                         "region-mean GT value (default: node)")
    return ap


def run(args):
    fmt = args.format
    sample_dir = os.path.join(args.sample_dir, "samples")
    ts_list = sorted(
        {re.match(r"sample_(\d{8}T\d{2})_ens\d+\.npz$", f).group(1)
         for f in os.listdir(sample_dir)
         if re.match(r"sample_\d{8}T\d{2}_ens\d+\.npz$", f)}
    )
    print(f"timestamps: {len(ts_list)}")
    if fmt == "auto":
        fmt = detect_format(sample_dir)
        print(f"detected sample format: {fmt}")
    var_order = VAR_ORDER if fmt == "hk" else ["hs"]
    var_cfg = (TS_VAR_CFG if args.group_by == "timestamp" else VAR_CFG) \
        if fmt == "hk" else GLOBAL_VAR_CFG
    group_txt = "per-timestamp mean GT" if args.group_by == "timestamp" else "per-node GT"

    tasks = [(ts, sample_dir, args.data_dir, args.num_ensemble, fmt, args.group_by)
             for ts in ts_list]
    nb = {var: len(var_cfg[var]["labels"]) for var in var_order}
    agg = {var: (np.zeros(nb[var], dtype=np.int64),
                 [[] for _ in range(nb[var])],
                 [[] for _ in range(nb[var])],
                 [[] for _ in range(nb[var])],
                 [[] for _ in range(nb[var])]) for var in var_order}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(process_timestamp, tasks, chunksize=8):
            if res is None:
                continue
            for var in var_order:
                counts, picks, ts_r, ts_a, ts_b = res[var]
                a_counts, a_picks, a_r, a_a, a_b = agg[var]
                for b in range(len(picks)):
                    if len(picks[b]):
                        a_picks[b].append(picks[b])
                        a_r[b].extend(ts_r[b])
                        a_a[b].extend(ts_a[b])
                        a_b[b].extend(ts_b[b])
                a_counts += counts
            done += 1
            if done % 2000 == 0:
                print(f"  processed {done}/{len(ts_list)}", flush=True)

    out_dir = os.path.join(args.sample_dir, "extreme_eval", "error_box")
    os.makedirs(out_dir, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(args.sample_dir))

    box_cmap = {"hs": "#4C72B0", "tp": "#55A868", "dm": "#C44E52", "elev": "#8172B3"}

    def _fmt_count(n):
        return f"{n/1e6:.1f}M" if n >= 1e6 else f"{n/1e3:.0f}k"

    def annot_stats(ax, bin_labels, stats, show=("rmse", "mae", "bias")):
        """Append n/RMSE/... as extra lines under each bin label (tick labels)."""
        ticks = []
        for lb, s in zip(bin_labels, stats):
            lines = [lb]
            if s["n"] == 0:
                lines.append("n=0")
            else:
                lines.append(f"n={_fmt_count(s['n'])}")
                for k in show:
                    v = s[k]
                    if not np.isnan(v):
                        lines.append(f"{k.upper()} {v:.2f}")
            ticks.append("\n".join(lines))
        ax.set_xticks(range(1, len(ticks) + 1), labels=ticks, fontsize=8)

    def boxplot_err(ax, data, labels, color, ylabel, title):
        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        widths=0.6, medianprops=dict(color="black", lw=1.4))
        ax.set_xticks(range(1, len(labels) + 1), labels=labels)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.axhline(0, color="crimson", ls="--", lw=1.2)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)

    csv_rows = []
    var_stats = {}
    var_ts = {}  # per-timestamp metric distributions per bin
    for var in var_order:
        counts, picks, ts_r, ts_a, ts_b = agg[var]
        cfg = var_cfg[var]
        stats = []
        ts_dist = []
        for b in range(len(cfg["labels"])):
            e = np.concatenate(picks[b]) if picks[b] else np.zeros(0, dtype=np.float32)
            s = {"n": int(counts[b]), "rmse": np.nan, "mae": np.nan, "bias": np.nan,
                 "p5": np.nan, "p50": np.nan, "p95": np.nan,
                 "n_ts": len(ts_r[b]), "rmse_p50": np.nan, "mae_p50": np.nan,
                 "bias_p50": np.nan}
            if s["n"] > 0:
                s["rmse"] = float(np.sqrt(np.mean(e ** 2)))
                s["mae"] = float(np.mean(np.abs(e)))
                s["bias"] = float(np.mean(e))
                if len(e):
                    q = np.quantile(e, [0.05, 0.5, 0.95])
                    s["p5"], s["p50"], s["p95"] = map(float, q)
            if ts_r[b]:
                s["rmse_p50"] = float(np.median(ts_r[b]))
                s["mae_p50"] = float(np.median(ts_a[b]))
                s["bias_p50"] = float(np.median(ts_b[b]))
            stats.append(s)
            ts_dist.append((np.array(ts_r[b]), np.array(ts_a[b]), np.array(ts_b[b])))
            csv_rows.append({"variable": var, "bin": cfg["labels"][b], **s})
        var_stats[var] = stats
        var_ts[var] = ts_dist
        data = [np.concatenate(picks[b]) if picks[b] else np.zeros(0, dtype=np.float32)
                for b in range(len(cfg["labels"]))]
        unit = {"hs": "error (m)", "tp": "error (s)", "dm": "error (deg)", "elev": "error (m)"}[var]
        # per-variable figure: signed + absolute error
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
        boxplot_err(axes[0], data, cfg["labels"], box_cmap[var], unit,
                    f"[{var}] signed error by {group_txt}")
        annot_stats(axes[0], cfg["labels"], stats)
        abs_data = [np.abs(d) for d in data]
        boxplot_err(axes[1], abs_data, cfg["labels"], box_cmap[var], f"|{unit}|",
                    f"[{var}] absolute error by {group_txt}")
        annot_stats(axes[1], cfg["labels"], stats, show=("mae",))
        fig.suptitle(f"{run_name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"error_box_{var}.png"))
        plt.close(fig)
        print(f"[{var}] " + "  ".join(
            f"{cfg['labels'][b]}: n={s['n']:,} rmse={s['rmse']:.2f} "
            f"mae={s['mae']:.2f} bias={s['bias']:+.2f} (#ts={s['n_ts']})"
            for b, s in enumerate(stats)))

    def annot_ts_stats(ax, bin_labels, stats):
        ticks = []
        for lb, s in zip(bin_labels, stats):
            lines = [lb]
            if s["n_ts"] == 0:
                lines.append("#ts=0")
            else:
                lines.append(f"#ts={s['n_ts']}")
                if not np.isnan(s["rmse_p50"]):
                    lines.append(f"median {s['rmse_p50']:.2f}")
            ticks.append("\n".join(lines))
        ax.set_xticks(range(1, len(ticks) + 1), labels=ticks, fontsize=8)

    def overview_panel(ax, var, kind):
        cfg = var_cfg[var]
        unit = {"hs": "err (m)", "tp": "err (s)", "dm": "err (deg)", "elev": "err (m)"}[var]
        ylbl = {"error": unit, "rmse": "RMSE " + unit, "mae": "MAE " + unit,
                "bias": "bias " + unit}[kind]
        if kind == "error":
            counts, picks = agg[var][:2]
            data = [np.concatenate(picks[b]) if picks[b] else np.zeros(0, dtype=np.float32)
                    for b in range(len(cfg["labels"]))]
            boxplot_err(ax, data, cfg["labels"], box_cmap[var], ylbl,
                        f"[{var}] error vs {group_txt}")
            annot_stats(ax, cfg["labels"], var_stats[var], show=("rmse", "bias"))
        else:
            metric = {"rmse": 0, "mae": 1, "bias": 2}[kind]
            dists = var_ts[var]
            data = [dists[b][metric] if len(dists[b][metric]) else np.zeros(0, dtype=np.float32)
                    for b in range(len(cfg["labels"]))]
            boxplot_err(ax, data, cfg["labels"], box_cmap[var], ylbl,
                        f"[{var}] per-timestamp {kind} by {group_txt}")
            annot_ts_stats(ax, cfg["labels"], var_stats[var])

    for kind in ["error", "rmse", "mae", "bias"]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        for ax, var in zip(axes.ravel(), var_order):
            overview_panel(ax, var, kind)
        fig.suptitle(f"{kind.upper()} by {group_txt} — {run_name}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"error_box_by_value_{kind}.png"))
        plt.close(fig)

    with open(os.path.join(out_dir, "error_box_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group_by", "variable", "bin", "n", "rmse",
                                          "mae", "bias", "p5", "p50", "p95",
                                          "n_ts", "rmse_p50", "mae_p50", "bias_p50"])
        w.writeheader()
        for r in csv_rows:
            w.writerow({"group_by": args.group_by, **r})

    print(f"\nfigures -> {out_dir}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
