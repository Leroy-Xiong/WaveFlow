#!/usr/bin/env python
"""Check the hindcast reference frames needed by the reference-bearing panels.

The hindcast itself is a third-party product and is not redistributed. The
release needs only four small slices:

    Fig. 5g   data/hindcast/processed_npz/wave_20211013T00.npz
    Fig. 5h/i data/global_hindcast/npz_files/2022/global_hindcast_20220928_12.npz
    ED 5      wave_20220701T20.npz, wave_20210125T17.npz
    ED 6      npy_files_daily_4var/2022/global_hindcast_{20220924,20220115}.npy

Redirect them with --gt-hk / --gt-global / --gt-global-4var (or the
WF_GT_HK / WF_GT_GLOBAL / WF_GT_GLOBAL_4VAR environment variables).

    python check_hindcast.py
    python check_hindcast.py --gt-hk /data/hk_hindcast/processed_npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: E402

REPO = bootstrap.REPO
PANELS = bootstrap.load_panels()


def resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    s = str(p)
    if s.startswith("data/hindcast/processed_npz/"):
        base = os.environ.get("WF_GT_HK")
        if base:
            return Path(base) / p.name
    elif s.startswith("data/global_hindcast/npz_files/"):
        base = os.environ.get("WF_GT_GLOBAL")
        if base:
            return Path(base) / p.parent.name / p.name
    elif s.startswith("data/global_hindcast/npy_files_daily_4var/"):
        base = os.environ.get("WF_GT_GLOBAL_4VAR")
        if base:
            return Path(base) / p.parent.name / p.name
    return REPO / p


def inspect(path: Path):
    """Return (ok, message) after a light shape/key check."""
    if not path.exists():
        return False, "missing"
    try:
        if path.suffix == ".npz":
            with np.load(path) as d:
                keys = set(d.keys())
                if "hs" not in keys:
                    return False, f"keys={sorted(keys)} (need hs)"
                n = d["hs"].shape
                if len(keys & {"hs", "tp", "dm", "elev"}) == 4:
                    return True, f"4-var npz, hs{n}"
                return True, f"npz, hs{n}"
        if path.suffix == ".npy":
            a = np.load(path, mmap_mode="r")
            if a.ndim != 3 or a.shape[0] != 8:
                return False, f"shape {a.shape}, expected (8, N, 4)"
            return True, f"daily pack {a.shape}"
    except Exception as exc:
        return False, f"unreadable: {exc}"
    return True, "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-hk", default=None)
    ap.add_argument("--gt-global", default=None)
    ap.add_argument("--gt-global-4var", default=None)
    args = ap.parse_args(argv)
    if args.gt_hk:
        os.environ["WF_GT_HK"] = args.gt_hk
    if args.gt_global:
        os.environ["WF_GT_GLOBAL"] = args.gt_global
    if args.gt_global_4var:
        os.environ["WF_GT_GLOBAL_4VAR"] = args.gt_global_4var

    all_ok = True
    for key, spec in PANELS.items():
        gts = spec.get("requires", {}).get("gt") or []
        if not gts:
            print(f"{key:10s} no hindcast needed")
            continue
        print(f"{key:10s} needs {len(gts)} hindcast file(s)")
        for rel in gts:
            path = resolve(rel)
            ok, msg = inspect(path)
            all_ok &= ok
            print(f"  {'OK  ' if ok else 'MISS'} {path}  [{msg}]")
    print("\nRESULT:", "all reference frames available" if all_ok else
          "some hindcast frames are missing (see DATA.md)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
