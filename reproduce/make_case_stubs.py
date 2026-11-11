#!/usr/bin/env python
"""Create zero-valued wave "anchor" files so inference can run without the
hindcast targets.

Sampling only consumes the timestamp and the atmospheric condition; the wave
target is loaded but never used. Writing zeros therefore unlocks inference on
arbitrary times from the released weights and public forcing alone.

Three layouts are supported (matching the released configs):

    hk           per-timestamp ``wave_<ts>.npz``  (7788, 4)
    global_hs    daily-packed ``npy_files_daily`` (8, 651652, 3) + index
    global_4var  per-timestamp ``global_hindcast_<ts>.npy`` (651652, 4)

USAGE
-----
    python make_case_stubs.py --region hk --timestamps 20250924T03 \
        --out data/hindcast/case_ragasa_2025/processed_npz
    python make_case_stubs.py --region global_4var --timestamps 20220924T06,20220115T15 \
        --out data/global_hindcast/case_stubs/npy_files_4var
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: E402

REPO = bootstrap.REPO
HK_KEYS = ["hs", "tp", "dm", "elev"]
LEGACY_COLUMNS = ["hs", "mwd", "ssh"]
VAR4_COLUMNS = ["hs", "tp", "dm", "elev"]


def parse_timestamps(timestamps: str | None, start: str | None, end: str | None,
                     step_hours: int) -> list[str]:
    if timestamps:
        return [t.strip() for t in timestamps.replace("\n", ",").split(",")
                if t.strip()]
    if not (start and end):
        raise SystemExit("give --timestamps or --start/--end")
    t = datetime.strptime(start, "%Y%m%dT%H")
    stop = datetime.strptime(end, "%Y%m%dT%H")
    out = []
    while t <= stop:
        out.append(t.strftime("%Y%m%dT%H"))
        t += timedelta(hours=step_hours)
    return out


def n_nodes(region: str, coords: Path | None) -> int:
    if region == "hk":
        p = Path(coords) if coords else REPO / "data/hindcast/hk_coordinates.npy"
    else:
        p = Path(coords) if coords else REPO / "data/global_hindcast/global_coordinates.npy"
    return int(np.load(p, mmap_mode="r").shape[0])


def write_hk(timestamps, out: Path, n: int):
    out.mkdir(parents=True, exist_ok=True)
    for ts in timestamps:
        path = out / f"wave_{ts}.npz"
        if not path.exists():
            np.savez(path, **{k: np.zeros(n, dtype=np.float32) for k in HK_KEYS})
    print(f"hk stubs: {len(timestamps)} timestamps in {out}")


def write_global_hs(timestamps, out: Path, n: int):
    daily = Path(str(out).replace("npy_files", "npy_files_daily"))
    if daily == out:
        daily = out / "npy_files_daily"
    stamps, days, slots = [], [], []
    for ts in timestamps:
        dt = datetime.strptime(ts, "%Y%m%dT%H")
        day = dt.strftime("%Y%m%d")
        slot = dt.hour // 3
        path = daily / day[:4] / f"global_hindcast_{day}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            np.save(path, np.zeros((8, n, len(LEGACY_COLUMNS)), dtype=np.float32))
        stamps.append(ts)
        days.append(day)
        slots.append(slot)
    daily.mkdir(parents=True, exist_ok=True)
    np.savez(daily / "wave_daily_ts_index.npz",
             timestamps=np.asarray(stamps), days=np.asarray(days),
             slots=np.asarray(slots, dtype=np.int64))
    with open(daily / "wave_daily_columns.json", "w") as fh:
        json.dump({"columns": LEGACY_COLUMNS}, fh)
    print(f"global_hs stubs: {len(timestamps)} timestamps in {daily}")


def write_global_4var(timestamps, out: Path, n: int):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "wave_columns.json", "w") as fh:
        json.dump({"columns": VAR4_COLUMNS}, fh)
    for ts in timestamps:
        dt = datetime.strptime(ts, "%Y%m%dT%H")
        path = out / dt.strftime("%Y") / f"global_hindcast_{dt:%Y%m%d}_{dt:%H}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            np.save(path, np.zeros((n, len(VAR4_COLUMNS)), dtype=np.float32))
    print(f"global_4var stubs: {len(timestamps)} timestamps in {out}")


def check_forcing(region: str, timestamps):
    """Warn about missing ERA5 frames (cheap existence checks only)."""
    missing = 0
    if region == "hk":
        for ts in timestamps:
            dt = datetime.strptime(ts, "%Y%m%dT%H")
            for i in range(8):
                t = dt - timedelta(hours=6 * i)
                p = REPO / "data/era5/npz_files" / t.strftime("%Y") / \
                    f"era5_{t:%Y%m%dT%H}.npz"
                if not p.exists():
                    missing += 1
                    if missing <= 5:
                        print(f"  WARNING missing ERA5 frame: {p}")
    elif region in ("global_hs", "global_4var"):
        days = set()
        for ts in timestamps:
            dt = datetime.strptime(ts, "%Y%m%dT%H")
            for i in (0, 6, 12, 18):
                t = dt - timedelta(hours=i)
                days.add(t.strftime("%Y%m%d"))
        for day in sorted(days):
            p = REPO / "data/era5/global_npy_files_preinterpolated_daily" / \
                day[:4] / f"era5_preinterp_{day}.npy"
            if not p.exists():
                missing += 1
                print(f"  WARNING missing pre-interpolated ERA5 day: {p}")
    if missing:
        print(f"  {missing} forcing file(s) missing — sampling will skip those "
              "timestamps (see DATA.md)")
    return missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True,
                    choices=["hk", "global_hs", "global_4var"])
    ap.add_argument("--timestamps", default=None,
                    help="comma-separated valid times, e.g. 20220924T06,20220115T15")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--step-hours", type=int, default=1)
    ap.add_argument("--out", required=True,
                    help="wave data directory for the selected region/layout")
    ap.add_argument("--coords", default=None,
                    help="optional coordinates file (node count source)")
    ap.add_argument("--no-forcing-check", action="store_true")
    args = ap.parse_args(argv)

    timestamps = parse_timestamps(args.timestamps, args.start, args.end,
                                  args.step_hours)
    if not timestamps:
        raise SystemExit("no timestamps selected")
    n = n_nodes(args.region, Path(args.coords) if args.coords else None)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    {
        "hk": write_hk,
        "global_hs": write_global_hs,
        "global_4var": write_global_4var,
    }[args.region](timestamps, out, n)
    if not args.no_forcing_check:
        check_forcing(args.region, timestamps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
