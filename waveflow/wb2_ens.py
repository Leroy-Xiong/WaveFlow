"""Fetch per-member IFS ENS forcing frames from WeatherBench2.

Fig. 5g--i are member-forced ensembles: generative sample *k* is conditioned on
IFS ENS forecast member *k*.  The paper used ten members (``m00..m09``); this
module rebuilds the same forcing tree from the public WeatherBench2 archive so
the notebook never needs the authors' case-forcing archive.

Source
------
``gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr``
  ``time``                 forecast initialisation, 00/12 UTC, 2018-01-01..2022-12-31
  ``number``               50 members; member ``m<k>`` = ``number k+1`` (k=0..9)
  ``prediction_timedelta`` 61 steps of 6 h (0..360 h)
  ``latitude``/``longitude`` 721 x 1440 (90..-90, 0..359.75)
  variables               10m u/v wind, mean sea level pressure

Output (layout identical to the paper tree consumed by
``WaveDataset._build_ens_frame_plan``)::

    data/ifs/wb2_ens_members/lead_time_<L>h/<YYYY>/ifs_ens_m<MM>_<init>.nc
    data/ifs/wb2_ens_members_hk/lead_time_<L>h/<YYYY>/ifs_ens_m<MM>_<init>.nc

The ``_hk`` mirror is the 40 x 40 box lon [110, 120), lat [15, 25).

Note on transfer volume: the Zarr chunks contain all 50 members (about 200 MB
per (init, lead) frame), so a member slice does not save bandwidth.  The
figures need roughly 8 (init, lead) frames per lead time; the CLI prints the
plan with ``--dry-run`` before downloading.

CLI
---
    python reproduce/wb2_ens.py --valid 20211013T00 --leads 12,24,48,72 \
        --frames 8 --members 10 --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from wb2_era5 import check_wb2_client, open_wb2  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WB2_ENS_URL = "gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr"
WB2_ENS_START = datetime(2018, 1, 1, 0)
WB2_ENS_END = datetime(2022, 12, 31, 12)
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
FRAME_STEP_HOURS = 6
HK_BOX = (110.0, 120.0, 15.0, 25.0)   # left-closed, right-open
GLOBAL_ROOT = REPO / "data" / "ifs" / "wb2_ens_members"
HK_ROOT = REPO / "data" / "ifs" / "wb2_ens_members_hk"


# ---------------------------------------------------------------------------
# forcing plan (mirrors WaveDataset._build_ens_frame_plan)
# ---------------------------------------------------------------------------
def frame_plan(valid_ts: str, lead: int, frames: int) -> list[tuple[str, int]]:
    """Return ``[(init, lead_hours), ...]`` for one target valid time.

    The main forecast chain starts at ``valid - lead``; frames older than that
    fall back to the most recent 00/12 UTC initialisation at a 0/6/12/18 h
    lead, exactly like the dataset loader.
    """
    dt_base = datetime.strptime(valid_ts, "%Y%m%dT%H")
    dt_main_init = dt_base - timedelta(hours=int(lead))
    plan = []
    for i in range(int(frames) - 1, -1, -1):
        dt_valid = dt_base - timedelta(hours=i * FRAME_STEP_HOURS)
        if dt_valid >= dt_main_init:
            dt_init = dt_main_init
        else:
            dt_init = dt_valid.replace(
                hour=12 if dt_valid.hour >= 12 else 0, minute=0, second=0)
        lead_h = int((dt_valid - dt_init).total_seconds() // 3600)
        plan.append((dt_init.strftime("%Y%m%dT%H"), lead_h))
    # oldest first, unique, keep order
    seen, out = set(), []
    for item in plan:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def required_frames(valid_ts: str, leads, frames: int) -> list[tuple[str, int]]:
    out, seen = [], set()
    for lead in leads:
        for item in frame_plan(valid_ts, int(lead), frames):
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def _paths(init: str, lead_h: int, members: int, trees) -> list[Path]:
    roots = {"global": GLOBAL_ROOT, "hk": HK_ROOT}
    paths = []
    for tree in trees:
        root = roots[tree]
        for m in range(int(members)):
            paths.append(Path(root) / f"lead_time_{lead_h}h" / init[:4] /
                         f"ifs_ens_m{m:02d}_{init}.nc")
    return paths


def _check_coverage(frames) -> None:
    late = [(i, l) for i, l in frames
            if datetime.strptime(i, "%Y%m%dT%H") > WB2_ENS_END]
    if late:
        raise RuntimeError(
            "WeatherBench2 IFS ENS covers 2018-01-01 .. 2022-12-31 12 UTC; "
            f"requested init {late[0][0]}. Later cases need ECMWF open data "
            "(https://data.ecmwf.int/forecasts/) instead.")


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def _crop_hk(ds: xr.Dataset) -> xr.Dataset:
    lon0, lon1, lat0, lat1 = HK_BOX
    lat = ds["latitude"]
    lon = ds["longitude"]
    keep = ((lat >= lat0) & (lat < lat1) & (lon >= lon0) & (lon < lon1))
    return ds.where(keep, drop=True)


def _atomic_save(ds: xr.Dataset, fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    ds = ds.reset_coords(drop=True)
    used = {d for v in ds.data_vars.values() for d in v.dims}
    drop = [c for c, da in ds.coords.items() if set(da.dims) - used]
    if drop:
        ds = ds.drop_vars(drop)
    tmp = Path(str(fp) + ".tmp")
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    try:
        ds.to_netcdf(tmp, encoding=encoding, engine="netcdf4")
    except Exception:
        ds.to_netcdf(tmp, encoding=encoding, engine="h5netcdf")
    os.replace(tmp, fp)


def _member_dataset(sel, m: int, ascending_lat: bool = True) -> xr.Dataset:
    ds = xr.Dataset({v: sel[v].isel(number=m).astype(np.float32)
                     for v in VARIABLES})
    if ascending_lat:
        ds = ds.sortby("latitude")
    return ds


def ensure_wb2_ens(
    valid_ts: str,
    leads,
    frames: int,
    members: int = 10,
    trees=("global", "hk"),
    dry_run: bool = False,
) -> int:
    """Download/cache the IFS ENS frames needed by one member-forced case.

    ``trees`` selects the outputs: ``"global"`` (1440 x 721) and/or ``"hk"``
    (40 x 40 crop).  Only ask for the tree the model consumes -- the Hong Kong
    model reads ``_hk``; the global model reads the full grid.

    Returns the number of files written (0 when everything was cached).
    """
    check_wb2_client()
    trees = tuple(trees)
    for tree in trees:
        if tree not in ("global", "hk"):
            raise ValueError(f"unknown ENS tree {tree!r}; use 'global' and/or 'hk'")
    frames_needed = required_frames(valid_ts, leads, frames)
    _check_coverage(frames_needed)

    roots = {"global": GLOBAL_ROOT, "hk": HK_ROOT}
    missing = []
    for init, lead_h in frames_needed:
        if any(not p.exists() or p.stat().st_size == 0
               for p in _paths(init, lead_h, members, trees)):
            missing.append((init, lead_h))
    est_gb = len(missing) * 0.2
    print(f"[wb2-ens] valid {valid_ts}: {len(frames_needed)} (init, lead) frame(s) "
          f"needed, {len(missing)} to fetch, trees={','.join(trees)}")
    for init, lead_h in frames_needed:
        mark = "*" if (init, lead_h) in missing else " "
        print(f"[wb2-ens]  {mark} init {init}  lead {lead_h:3d} h")
    if not missing:
        print("[wb2-ens] cache hit under "
              + " / ".join(str(roots[t]) for t in trees))
        return 0
    print(f"[wb2-ens] about {est_gb:.1f} GB from {WB2_ENS_URL} (50-member chunks)")
    if dry_run:
        return 0

    ds = open_wb2(WB2_ENS_URL)
    written = 0
    try:
        for init, lead_h in missing:
            if lead_h % FRAME_STEP_HOURS:
                raise ValueError(f"lead {lead_h} h is not a multiple of 6 h")
            index = lead_h // FRAME_STEP_HOURS
            try:
                sel = ds[VARIABLES].sel(
                    time=np.datetime64(datetime.strptime(init, "%Y%m%dT%H")),
                ).isel(prediction_timedelta=index,
                       number=slice(1, int(members) + 1)).compute()
            except Exception as exc:
                raise RuntimeError(
                    f"failed to read WB2 IFS ENS init {init} lead {lead_h}: "
                    f"{exc}") from exc
            for m in range(int(members)):
                one = _member_dataset(sel, m)
                if not np.isfinite(one.to_array().values).all():
                    raise ValueError(
                        f"non-finite IFS ENS values at init {init} "
                        f"lead {lead_h} member {m}")
                fname = f"ifs_ens_m{m:02d}_{init}.nc"
                sub = Path(f"lead_time_{lead_h}h") / init[:4] / fname
                if "global" in trees:
                    _atomic_save(one, GLOBAL_ROOT / sub)
                    written += 1
                if "hk" in trees:
                    _atomic_save(_crop_hk(one), HK_ROOT / sub)
                    written += 1
            print(f"[wb2-ens] wrote init {init} lead {lead_h}h "
                  f"({members} members x {len(trees)} tree(s))")
    finally:
        ds.close()
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--valid", required=True,
                    help="target valid time (YYYYMMDDTHH), e.g. 20211013T00")
    ap.add_argument("--leads", default="12,24,48,72",
                    help="comma-separated forecast leads in hours")
    ap.add_argument("--frames", type=int, default=8,
                    help="6-hourly history frames of the model (8 HK, 4 global)")
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--trees", default="global,hk",
                    help="which outputs to write: global, hk, or global,hk")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the (init, lead) plan without downloading")
    args = ap.parse_args(argv)
    leads = [int(x) for x in args.leads.replace(" ", "").split(",") if x]
    trees = tuple(t.strip() for t in args.trees.split(",") if t.strip())
    ensure_wb2_ens(args.valid, leads, frames=args.frames,
                   members=args.members, trees=trees, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
