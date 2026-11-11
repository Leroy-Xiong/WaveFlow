"""Post-hoc PIT/rank calibration for ensemble samples (inference-only, no retrain).

Post-processes an existing sample run: fits a calibration map on one year
(fit-year) and applies it to another (apply-year), per wave variable.
Two modes (see baseline_repros/competitor_analysis/PIT_calibration.md):

- global:  single inverse-CDF map per variable (H^{-1}) over all nodes/times
- layered: per-layer maps binned by ensemble-mean hs of the sample, so the
           map can differ by sea state (storm layers get their own fit)

Calibration map:
  PIT = (rank of GT within ensemble + jitter) / M   (same as plot_pit)
  H   = smoothed empirical CDF of PIT (B=200 bins, 5-bin moving average)
  F_cal(y) = H(F_ens(y));  members z_(k) = interp(H^{-1}(u_k), u_k, y_(k)),
  u_k = k/(M+1).
"""
from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

VAR_ORDER = ["hs", "tp", "dm", "elev"]
N_BINS = 200
SMOOTH = 5
M_DEFAULT = 10
DEFAULT_LAYER_BINS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, np.inf]


# ---------------------------------------------------------------- utils


def list_ts(sample_dir):
    pat = re.compile(r"sample_(\d{8}T\d{2})_ens\d+\.npz$")
    return sorted({m.group(1) for f in os.listdir(sample_dir)
                   if (m := pat.match(f))})


def ens_mean_hs(ens):
    """Ensemble-mean hs per node; used to assign each node-hour to a layer."""
    return ens[:, :, 0].mean(axis=0)


def layer_index(values, bins):
    return np.clip(np.searchsorted(bins, values, side="right") - 1, 0, len(bins) - 2)


def randomized_pit(ev, gg, rng):
    """PIT = (less + 0.5*eq + jitter) / M, seeded per timestamp (reproducible)."""
    less = (ev < gg[None, :]).sum(axis=0)
    eq = (ev == gg[None, :]).sum(axis=0)
    return np.clip((less + 0.5 * eq + rng.random(less.shape) * 0.9999) / ev.shape[0], 0, 1)


def interp_vec(x, xp, fp):
    """Vectorised np.interp for 2-D fp (M, N): interpolate each column."""
    idx = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, len(xp) - 2)
    frac = np.clip((x - xp[idx]) / (xp[idx + 1] - xp[idx]), 0, 1)
    return fp[idx] + frac[:, None] * (fp[idx + 1] - fp[idx])


# ---------------------------------------------------------------- fit


class PitFitter:
    """Aggregate PIT histograms over timestamps and build inverse-CDF maps."""

    def __init__(self, sample_dir, gt_dir, num_ens, mode, layer_bins, workers):
        self.sample_dir = sample_dir
        self.gt_dir = gt_dir
        self.num_ens = num_ens
        self.mode = mode
        self.layer_bins = layer_bins
        self.n_layers = len(layer_bins) - 1 if mode == "layered" else 1
        self.workers = workers

    def _ts_hist(self, ts):
        ens = np.stack([np.load(os.path.join(self.sample_dir, f"sample_{ts}_ens{i:03d}.npz"))["sample"]
                        for i in range(self.num_ens)], axis=0)
        g = np.load(os.path.join(self.gt_dir, f"wave_{ts}.npz"))
        rng = np.random.default_rng(int(ts.replace("T", "")) % 2**32)
        hist = np.zeros((len(VAR_ORDER), self.n_layers, N_BINS), dtype=np.int64)
        layers = None
        if self.mode == "layered":
            layers = layer_index(ens_mean_hs(ens), self.layer_bins)
        for v, var in enumerate(VAR_ORDER):
            gv = g[var].astype(np.float32)
            valid = ~np.isnan(gv)
            ev = ens[:, valid, v]
            gg = gv[valid]
            pit = randomized_pit(ev, gg, rng)
            if layers is None:
                hist[v, 0] += np.histogram(pit, bins=N_BINS, range=(0, 1))[0]
            else:
                ll = layers[valid]
                for L in range(self.n_layers):
                    m = ll == L
                    if m.sum():
                        hist[v, L] += np.histogram(pit[m], bins=N_BINS, range=(0, 1))[0]
        return hist

    def fit(self, ts_list):
        hist = np.zeros((len(VAR_ORDER), self.n_layers, N_BINS), dtype=np.int64)
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            for i, h in enumerate(ex.map(self._ts_hist, ts_list, chunksize=8)):
                hist += h
                if (i + 1) % 1000 == 0:
                    print(f"  fit {i+1}/{len(ts_list)}", flush=True)
        return hist

    @staticmethod
    def build_inverse_cdfs(hist):
        """Smoothed PIT empirical CDF H and its inverse on a u-grid, per (var, layer)."""
        t = (np.arange(N_BINS) + 0.5) / N_BINS
        u_full = (np.arange(N_BINS) + 0.5) / N_BINS
        inv = np.zeros_like(hist, dtype=np.float64)  # (4, n_layers, N_BINS) = H^{-1}(u_full)
        for v in range(hist.shape[0]):
            for L in range(hist.shape[1]):
                h = np.convolve(hist[v, L].astype(np.float64), np.ones(SMOOTH) / SMOOTH, mode="same")
                total = h.sum()
                if total == 0:
                    inv[v, L] = u_full
                    continue
                h = h / total * N_BINS
                Hc = np.cumsum(h)
                Hc = Hc / Hc.max()
                inv[v, L] = np.interp(u_full, Hc, t, left=t[0], right=t[-1])
        return inv


# ---------------------------------------------------------------- apply

u_full = (np.arange(N_BINS) + 0.5) / N_BINS


def _apply_one(args):
    ts, sample_dir, out_dir, inv_cdfs, num_ens, mode, layer_bins = args
    u_grid = (np.arange(num_ens) + 1) / (num_ens + 1)  # k/(M+1), k=1..M
    ens = np.stack([np.load(os.path.join(sample_dir, f"sample_{ts}_ens{i:03d}.npz"))["sample"]
                    for i in range(num_ens)], axis=0)  # (M, N, 4)
    yc = np.zeros_like(ens)
    for v in range(len(VAR_ORDER)):
        srt = np.sort(ens[:, :, v], axis=0)  # (M, N) ascending
        if mode == "layered":
            layers = layer_index(ens_mean_hs(ens), layer_bins)
            for L in range(inv_cdfs.shape[1]):
                mask = layers == L
                if mask.sum() == 0:
                    continue
                vk = np.interp(u_grid, u_full, inv_cdfs[v, L])  # (M,)
                yc[:, mask, v] = interp_vec(vk, u_grid, srt[:, mask])
        else:
            vk = np.interp(u_grid, u_full, inv_cdfs[v, 0])
            yc[:, :, v] = interp_vec(vk, u_grid, srt)
    for i in range(num_ens):
        f = os.path.join(sample_dir, f"sample_{ts}_ens{i:03d}.npz")
        data = np.load(f)
        out = {k: val for k, val in data.items()}
        out["sample"] = yc[i]
        np.savez(os.path.join(out_dir, f"sample_{ts}_ens{i:03d}.npz"), **out)
    return ts


def apply_calibration(ts_list, sample_dir, out_dir, inv_cdfs, num_ens,
                      mode, layer_bins, workers):
    """Rank-calibrate members of each timestamp; write calibrated npz files."""
    os.makedirs(out_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max(1, workers // 2)) as ex:
        tasks = [(ts, sample_dir, out_dir, inv_cdfs, num_ens, mode, layer_bins)
                 for ts in ts_list]
        for i, _ in enumerate(ex.map(_apply_one, tasks, chunksize=4)):
            if (i + 1) % 500 == 0:
                print(f"  apply {i+1}/{len(ts_list)}", flush=True)


# ---------------------------------------------------------------- CLI


def run(args):
    sample_dir = os.path.join(args.sample_dir, "samples")
    out_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)
    ts_all = list_ts(sample_dir)
    ts_apply = [t for t in ts_all if t.startswith(str(args.apply_year))]
    if args.fit_sample_dir:
        # Cross-directory protocol: fit on ALL timestamps of a separate run
        # (e.g. a val-split sampling run), apply to apply-year of this run.
        fit_sample_dir = os.path.join(args.fit_sample_dir, "samples")
        ts_fit = list_ts(fit_sample_dir)
        fit_desc = f"{args.fit_sample_dir} (all {len(ts_fit)} ts)"
    else:
        fit_sample_dir = sample_dir
        ts_fit = [t for t in ts_all if t.startswith(str(args.fit_year))]
        fit_desc = f"fit-year {args.fit_year}"
    if not ts_fit or not ts_apply:
        raise SystemExit(f"need fit timestamps ({fit_desc}) and apply-year "
                         f"{args.apply_year} timestamps (found {len(ts_all)} ts "
                         f"in sample dir)")
    print(f"fit on {len(ts_fit)} ts ({fit_desc}), "
          f"apply on {len(ts_apply)} ts ({args.apply_year})", flush=True)

    layer_bins = [np.inf if b == "inf" else float(b) for b in args.layer_bins]
    fitter = PitFitter(fit_sample_dir, args.data_dir, args.num_ensemble,
                       args.mode, layer_bins, args.workers)
    hist = fitter.fit(ts_fit)
    inv_cdfs = PitFitter.build_inverse_cdfs(hist)
    for v in range(len(VAR_ORDER)):
        for L in range(hist.shape[1]):
            print(f"  {VAR_ORDER[v]} L{L}: n={hist[v, L].sum()/1e6:.1f}M", flush=True)

    apply_calibration(ts_apply, sample_dir, out_dir, inv_cdfs,
                      args.num_ensemble, args.mode, layer_bins, args.workers)
    print("done ->", args.out_dir)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-dir", required=True,
                   help="Run directory containing a 'samples/' subfolder.")
    p.add_argument("--out-dir", default=None,
                   help="Output sample directory (default: <sample-dir>_calib/).")
    p.add_argument("--fit-year", type=int, default=2021,
                   help="Year used to fit the calibration map (default: 2021).")
    p.add_argument("--fit-sample-dir", default=None,
                   help="Optional separate run directory whose samples (all "
                        "timestamps) are used to FIT the map — e.g. a val-split "
                        "sampling run (2019-2020) while applying to the test run. "
                        "Overrides --fit-year.")
    p.add_argument("--apply-year", type=int, default=2022,
                   help="Year whose samples get calibrated (default: 2022).")
    p.add_argument("--data-dir", default="./data/hindcast/processed_npz",
                   help="GT wave data dir (default: ./data/hindcast/processed_npz).")
    p.add_argument("--num-ensemble", type=int, default=M_DEFAULT)
    p.add_argument("--mode", choices=["global", "layered"], default="global",
                   help="'global': one inverse-CDF map per variable; "
                        "'layered': per-layer maps binned by ensemble-mean hs.")
    p.add_argument("--layer-bins", nargs="+", default=[str(b) for b in DEFAULT_LAYER_BINS],
                   help="Layer boundaries in m (ascending, last may be 'inf'). "
                        "Default: 0 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5 inf (9 layers).")
    p.add_argument("--workers", type=int, default=16)
    return p


def cmd_pit_calib(args):
    """Post-hoc PIT/rank calibration of ensemble samples (see calibrate.py)."""
    if args.out_dir is None:
        args.out_dir = args.sample_dir.rstrip("/") + "_calib"
    run(args)
