#!/usr/bin/env python
"""Stream the full test set into 2D histograms for the joint-distribution figure.

Each worker loads one timestamp (hindcast + ten ensemble members), returns the
two (Hs, T) histograms and the sums needed for the Hs-water-level correlation
below a set of thresholds.  The main process reduces them, so memory stays
flat regardless of the number of timestamps.

USAGE
  python compute_joint_full.py hk 32
  python compute_joint_full.py global 10
"""
from __future__ import annotations

import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
OUT_DIR = os.path.join(ROOT, "output/plot/figure_ed_joint")

HK_S = ("output/results/hk/all_var/cond_patch512_allVar_hk3/"
        "sample_epoch_49_20260128_183959/samples")
HK_GT = "data/hindcast/processed_npz/wave_{ts}.npz"
GL_S = ("output/results/global/new_data_allvar/patch2048_4frames_concat_xpred_giant_tod_ext/"
        "sample_allvar_e199_full_v2_20261018_133956/samples")
GL_GT = "data/global_hindcast/npy_files_daily_4var/{y}/global_hindcast_{yyyymmdd}.npy"

# variable order in the sample arrays / hindcast files: hs, tp, dm, elev
PAIRS = [("hs", "tp"), ("hs", "elev"), ("tp", "elev"), ("hs", "dm"), ("tp", "dm")]
SUFFIX = ""
if os.environ.get("JOINT_PAIRS"):
    PAIRS = [tuple(x.split("-")) for x in os.environ["JOINT_PAIRS"].split(",")]
    SUFFIX = "_extra"
IDX = {"hs": 0, "tp": 1, "dm": 2, "elev": 3}

CFG = {
    "hk": dict(root=HK_S, thresh=(0.0, 0.5, 1.0, 1.5, 2.0),
               ranges={"hs": (0.0, 3.0), "tp": (0.0, 13.0),
                       "elev": (-1.5, 1.5), "dm": (0.0, 360.0)},
               bins=54),
    "global": dict(root=GL_S, thresh=(0.0, 2.0, 4.0, 6.0, 8.0),
                   ranges={"hs": (0.0, 12.0), "tp": (0.0, 16.0),
                           "elev": (-1.5, 1.5), "dm": (0.0, 360.0)},
                   bins=54),
}


def hist2d_fast(x, y, xr, yr, nb):
    """Two-dimensional histogram via bincount; several times faster than
    numpy.histogram2d on the 6.5-million-point global fields."""
    ix = np.floor((x - xr[0]) / (xr[1] - xr[0]) * nb).astype(np.int64)
    iy = np.floor((y - yr[0]) / (yr[1] - yr[0]) * nb).astype(np.int64)
    ok = (ix >= 0) & (ix < nb) & (iy >= 0) & (iy < nb)
    idx = ix[ok] * nb + iy[ok]
    return np.bincount(idx, minlength=nb * nb).reshape(nb, nb).astype(np.float64)


def moment_sums(h, e, thresh):
    out = []
    for lo in thresh:
        m = (h > lo)
        hh, ee = h[m], e[m]
        out.append(np.array([hh.size, hh.sum(), ee.sum(), (hh * ee).sum(),
                             (hh * hh).sum(), (ee * ee).sum()], dtype=np.float64))
    return np.stack(out)


def worker(ts, cfg, domain):
    if domain == "hk":
        raw = np.load(HK_GT.format(ts=ts))
        g = {"hs": raw["hs"], "tp": raw["tp"], "dm": raw["dm"],
             "elev": raw["elev"]}
    else:
        gt = np.load(GL_GT.format(y=ts[:4], yyyymmdd=ts[:8]),
                     mmap_mode="r")[int(ts[9:11]) // 3]
        g = {"hs": gt[:, 0], "tp": gt[:, 1], "dm": gt[:, 2], "elev": gt[:, 3]}
    members = [np.load(os.path.join(cfg["root"], f"sample_{ts}_ens{k:03d}.npz"))["sample"]
               for k in range(10)]
    mo = {"hs": np.concatenate([m[:, 0] for m in members]),
          "tp": np.concatenate([m[:, 1] for m in members]),
          "dm": np.concatenate([m[:, 2] for m in members]),
          "elev": np.concatenate([m[:, 3] for m in members])}
    del members
    nb = cfg["bins"]
    H_gt, H_mo = [], []
    for a, b in PAIRS:
        lo_a, hi_a = cfg["ranges"][a]
        lo_b, hi_b = cfg["ranges"][b]
        ok = np.isfinite(g[a]) & np.isfinite(g[b])
        H_gt.append(hist2d_fast(g[a][ok], g[b][ok], (lo_a, hi_a),
                                (lo_b, hi_b), nb))
        ok = np.isfinite(mo[a]) & np.isfinite(mo[b])
        H_mo.append(hist2d_fast(mo[a][ok], mo[b][ok], (lo_a, hi_a),
                                (lo_b, hi_b), nb))
    ok = np.isfinite(g["hs"]) & np.isfinite(g["elev"])
    s_gt = moment_sums(g["hs"][ok], g["elev"][ok], cfg["thresh"])
    ok = np.isfinite(mo["hs"]) & np.isfinite(mo["elev"])
    s_mo = moment_sums(mo["hs"][ok], mo["elev"][ok], cfg["thresh"])
    return np.stack(H_gt), np.stack(H_mo), s_gt, s_mo


def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "hk"
    nthread = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    cfg = CFG[domain]
    ts = sorted(os.path.basename(f).replace("sample_", "").replace("_ens000.npz", "")
                for f in glob.glob(os.path.join(cfg["root"], "*_ens000.npz")))
    print(f"{domain}: {len(ts):,} timestamps, {nthread} threads", flush=True)
    nb = cfg["bins"]
    H_gt = np.zeros((len(PAIRS), nb, nb))
    H_mo = np.zeros((len(PAIRS), nb, nb))
    s_gt = np.zeros((len(cfg["thresh"]), 6))
    s_mo = np.zeros((len(cfg["thresh"]), 6))
    t0 = time.time()
    pool_cls = ThreadPoolExecutor
    with pool_cls(max_workers=nthread) as ex:
        for i, (a, b, c, d) in enumerate(ex.map(worker, ts,
                                                [cfg] * len(ts), [domain] * len(ts))):
            H_gt += a
            H_mo += b
            s_gt += c
            s_mo += d
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f"  {i+1:,}/{len(ts):,}  {el/60:.1f} min  "
                      f"eta {el/(i+1)*(len(ts)-i-1)/60:.1f} min", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, f"pairs_full_{domain}{SUFFIX}.npz"),
                        H_gt=H_gt, H_mo=H_mo, s_gt=s_gt, s_mo=s_mo,
                        thresh=np.array(cfg["thresh"]),
                        pairs=np.array([f"{a}-{b}" for a, b in PAIRS]),
                        n_ts=len(ts))
    print(f"saved pairs_full_{domain}{SUFFIX}.npz in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
