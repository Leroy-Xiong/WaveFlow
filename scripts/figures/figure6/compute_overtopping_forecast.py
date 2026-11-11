#!/usr/bin/env python
"""EurOtop overtopping q for the ENS-driven forecast samples (Fig. 6 panel d).

Maps the forecast ensemble (per-member forced WaveFlow samples, hs/tp/dm/elev)
to per-segment mean overtopping discharge q [L/s/m] and P(q > 1 L/s/m), using
exactly the same segment table, crest freeboard and toe depth as the analysis
(perfect-forcing) product `ragasa_q_m100_hybridspot_typerev_gex.npz`, so the
two curves plotted in panel d are directly comparable.

Alignment strategy (why the segment order comes from the analysis npz):
  * seg order / per-segment parameters are taken from the analysis npz, so the
    forecast product is guaranteed index-aligned with it;
  * crest_proxy is the stored (clamped, hybrid+spot+design+imputed) freeboard;
  * h_toe is recomputed from the SCHISM grid exactly as the analysis pipeline
    did (load_node_depths + H_TOE_FLOOR);
  * coast_type / gamma_f / normal_deg / node_idx come from the segment table
    with the imagery-review type overrides applied.

Usage:
    python scripts/figures/figure6/compute_overtopping_forecast.py \
        --samples-dir <run>/forecast_ensmem_ragasa_init20250922T12/test/samples
    # -> data/hk_coastline/overtopping/ragasa_q_forecast_init20250922T12.npz

    # regression check: recompute from the analysis samples and compare with
    # the analysis npz (should match to float32 precision)
    python scripts/figures/figure6/compute_overtopping_forecast.py \
        --samples-dir <run>/case_ragasa2025_m100/test/samples \
        --timestamps 20250923T00,20250923T12,20250924T00 --out /tmp/q_chk.npz
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))
os.chdir(_REPO_ROOT)
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import plot_figure6_common as C  # noqa: E402  (CASE_SPECS = single source of truth)
from compute_overtopping_saola import (  # noqa: E402
    H_TOE_FLOOR, Q_CRIT_LPSM, apply_type_overrides, compute_q, load_node_depths)

DEFAULT_INIT = "20250922T12"
# 3-hourly valid times across the analysis window (lead 12-60 h)
DEFAULT_TS = ",".join(
    f"2025092{d}T{h:02d}"
    for d, hours in ((3, range(0, 24, 3)), (4, range(0, 24, 3)), (5, [0]))
    for h in hours)
PANEL_SEGS = [6760, 8196, 8032, 6569]


def parse_ts(text):
    return [s.strip() for s in str(text).split(",") if s.strip()]


def lead_hours(ts_list, init):
    """Lead of every valid time relative to the forecast issue time."""
    base = pd.Timestamp(init)
    return [int((pd.Timestamp(t) - base) / pd.Timedelta(hours=1)) for t in ts_list]


def build_segment_table(case, ref_npz):
    """Segment table ordered exactly like the analysis q npz."""
    spec = C.CASE_SPECS[case]
    seg = pd.read_csv(C.SEG_CSV)
    if spec.get("type_overrides"):
        seg = apply_type_overrides(seg, spec["type_overrides"])
    if spec.get("exclude_segs") and os.path.exists(spec["exclude_segs"]):
        ex = set(pd.read_csv(spec["exclude_segs"]).seg_id.astype(int))
        seg = seg[~seg.seg_id.isin(ex)]
    ref = np.load(ref_npz, allow_pickle=True)
    seg_ids = ref["seg_ids"]
    tab = seg.set_index("seg_id").reindex(seg_ids)
    if tab.node_idx.isna().any():
        miss = seg_ids[tab.node_idx.isna().to_numpy()]
        raise SystemExit(f"{len(miss)} reference segments missing from the "
                         f"segment table (e.g. {miss[:5].tolist()})")
    return tab.reset_index(drop=True), ref


def load_waves(samples_dir, ts_list, members, node_idx):
    """(T, M, S, 4) physical samples at the segment driving nodes."""
    out = np.full((len(ts_list), members, len(node_idx), 4), np.nan,
                  dtype=np.float32)
    missing = 0
    for i, ts in enumerate(ts_list):
        for k in range(members):
            path = os.path.join(samples_dir, f"sample_{ts}_ens{k:03d}.npz")
            if not os.path.exists(path):
                missing += 1
                continue
            with np.load(path) as d:
                out[i, k] = d["sample"][node_idx]
    if missing:
        print(f"  WARNING: {missing}/{len(ts_list) * members} sample files missing")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="ragasa", choices=sorted(C.CASE_SPECS))
    ap.add_argument("--samples-dir", required=True,
                    help="dir with sample_{ts}_ens{k}.npz (forecast or analysis)")
    ap.add_argument("--init", default=DEFAULT_INIT,
                    help=f"forecast issue time (default {DEFAULT_INIT})")
    ap.add_argument("--timestamps", default=DEFAULT_TS,
                    help="comma-separated valid times (3-hourly default)")
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--ref-npz", default=None,
                    help="analysis q npz defining segment order/crests "
                         "(default: the case's npz_m100)")
    ap.add_argument("--out", default=None,
                    help="output npz (default data/hk_coastline/overtopping/"
                         "<case>_q_forecast_init<init>.npz)")
    args = ap.parse_args(argv)

    spec = C.CASE_SPECS[args.case]
    ref_npz = args.ref_npz or spec["npz_m100"]
    out = args.out or os.path.join(
        "data/hk_coastline/overtopping",
        f"{args.case}_q_forecast_init{args.init}.npz")
    ts_list = parse_ts(args.timestamps)
    if not ts_list:
        raise SystemExit("no timestamps given")

    tab, ref = build_segment_table(args.case, ref_npz)
    seg_ids = ref["seg_ids"]
    dep = load_node_depths()
    h_toe = np.maximum(dep[tab.node_idx.values], H_TOE_FLOOR)
    rc_land = ref["crest_proxy"].astype(np.float64)
    leads = lead_hours(ts_list, args.init)

    print(f"forecast q: {args.samples_dir}")
    print(f"  segments/order from {ref_npz}: {len(seg_ids)}  "
          f"({dict(tab.coast_type.value_counts())})")
    print(f"  init {args.init}; {len(ts_list)} valid times, lead "
          f"{min(leads)}-{max(leads)} h; {args.members} members")

    waves = load_waves(args.samples_dir, ts_list, args.members,
                       tab.node_idx.values)
    q, n_weir = compute_q(waves, tab, h_toe, rc_land)      # (S, T, M)
    valid = np.isfinite(q)
    n_valid = valid.sum(axis=2)
    p_gt1 = np.where(
        n_valid > 0,
        ((q > Q_CRIT_LPSM) & valid).sum(axis=2) / np.maximum(n_valid, 1),
        np.nan)
    finite = np.where(valid, q, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        q_med = np.nanmedian(finite, axis=2)
        q_p10 = np.nanpercentile(finite, 10, axis=2)
        q_p90 = np.nanpercentile(finite, 90, axis=2)

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        out, q_lpsm=q.astype(np.float32), seg_ids=seg_ids,
        timestamps=np.asarray(ts_list), init=np.asarray(args.init),
        leads=np.asarray(leads, dtype=np.int32),
        p_gt1_lpsm=p_gt1.astype(np.float32),
        q_med_lpsm=q_med.astype(np.float32),
        q_p10_lpsm=q_p10.astype(np.float32),
        q_p90_lpsm=q_p90.astype(np.float32),
        n_weir_fallback=n_weir.astype(np.int32),
        h_toe=h_toe.astype(np.float32),
        crest_proxy=rc_land.astype(np.float32))
    print(f"  wrote {out}")

    print("\npanel d segments (forecast):")
    for sid in PANEL_SEGS:
        if sid not in seg_ids:
            continue
        i = int(np.where(seg_ids == sid)[0][0])
        med = q_med[i]
        k = int(np.nanargmax(med))
        print(f"  seg {sid:5d}  max median q {med[k]:8.1f} L/s/m at "
              f"{ts_list[k]} (lead {leads[k]:2d} h)   maxP {np.nanmax(p_gt1[i]):.2f}"
              f"   weir {int(n_weir[i])}")
    print(f"\nNaN in q: {int(np.isnan(q).sum())}/{q.size}   "
          f"weir-fallback cells: {int(n_weir.sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
