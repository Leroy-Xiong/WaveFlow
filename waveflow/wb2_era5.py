"""Fetch the ERA5 frames needed by an inference case from WeatherBench2.

The public release does not ship ERA5.  Instead, the inference helpers fetch
the required surface variables for a case window (target plus the 6-hourly
history frames) from the public WeatherBench2 ERA5 Zarr store and write one
NetCDF file per timestamp.  No daily packing and no timestamp index is needed.

The resulting layout is::

    data/era5/wb2/global/<YYYY>/era5_<YYYYMMDDTHH>.nc   # 0-360, 90..-90
    data/era5/wb2/hk/<YYYY>/era5_<YYYYMMDDTHH>.nc       # 111-117E, 19-25N

which is what ``WaveDataset`` expects for ``era5_file_format: nc`` with
``use_preinterpolated_era5: false``.

Access is anonymous (the bucket is public); install ``zarr`` + ``gcsfs`` first
(they are part of ``environment.yml``).  Set ``HTTPS_PROXY`` when the machine
needs a proxy to reach Google Cloud Storage.

CLI (optional prefetch, before running any inference)::

    python reproduce/wb2_era5.py --region hk --times 20211013T00 --frames 8
    python reproduce/wb2_era5.py --region global --start 2022-09-25T12 \
        --end 2022-09-28T12
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
WB2_ERA5_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
)
WB2_ERA5_END = datetime(2023, 1, 10, 23)
# Public GCS bucket: always use anonymous credentials so a machine without a
# Google Cloud login can still read it.
WB2_STORAGE_OPTIONS = {"token": "anon"}
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
REGIONS = {
    "global": (0.0, 360.0, -90.0, 90.0),
    "hk": (111.0, 117.0, 19.0, 25.0),
}


# ---------------------------------------------------------------------------
# shared WeatherBench2 access helpers (also used by wb2_ens.py)
# ---------------------------------------------------------------------------
def check_wb2_client(modules=("zarr", "gcsfs")) -> None:
    """Raise a helpful error when the WeatherBench2 client stack is absent."""
    missing = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise ImportError(
            "WeatherBench2 access needs " + " + ".join(missing) +
            ". Install them with\n"
            "  conda install -c conda-forge " + " ".join(missing) + "\n"
            "or\n"
            "  pip install " + " ".join(missing) + "\n"
            "See environment.yml / INFERENCE.md.")


def open_wb2(url: str) -> xr.Dataset:
    """Open one public WeatherBench2 Zarr store anonymously."""
    check_wb2_client()
    try:
        return xr.open_zarr(url, consolidated=True,
                            storage_options=dict(WB2_STORAGE_OPTIONS),
                            decode_timedelta=False)
    except Exception as exc:  # network / auth / missing store
        raise RuntimeError(
            f"cannot open WeatherBench2 store {url}: {exc}\n"
            "Check the network path to Google Cloud Storage (HTTPS_PROXY), "
            "or point the tools at a pre-downloaded cache.") from exc


def _str_to_dt(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y%m%dT%H")


def _check_coverage(times) -> None:
    too_late = [t for t in times if t > WB2_ERA5_END]
    if too_late:
        raise RuntimeError(
            "WeatherBench2 ERA5 ends at 2023-01-10 23:00; requested "
            f"{too_late[0]:%Y%m%dT%H}. Use another ERA5 source (e.g. the "
            "Copernicus Climate Data Store) for this case.")


# ---------------------------------------------------------------------------
# ERA5
# ---------------------------------------------------------------------------
def required_times(timestamps, frames: int = 4, step_hours: int = 6) -> list[str]:
    """Return the sorted, unique ERA5 times needed for ``timestamps``."""
    out = set()
    for stamp in timestamps:
        t = _str_to_dt(stamp)
        for i in range(int(frames)):
            out.add((t - timedelta(hours=step_hours * i)).strftime("%Y%m%dT%H"))
    return sorted(out)


def _region_bounds(region: str):
    key = "global" if str(region).lower().startswith("global") else str(region).lower()
    if key not in REGIONS:
        raise ValueError(f"unknown ERA5 region {region!r}; choose from {sorted(REGIONS)}")
    return key, REGIONS[key]


def default_out_dir(region: str = "global") -> Path:
    """Per-domain cache root: ``data/era5/wb2/<global|hk>``."""
    key, _ = _region_bounds(region)
    return REPO / "data" / "era5" / "wb2" / key


def _paths_for(times, out_dir: Path) -> list[Path]:
    return [out_dir / t[:4] / f"era5_{t}.nc" for t in times]


def _write_frames(ds, missing, region: str) -> None:
    key, (lon0, lon1, lat0, lat1) = _region_bounds(region)
    times = [t for t, _ in missing]
    select_times = np.array([np.datetime64(_str_to_dt(t)) for t in times])
    try:
        subset = ds[VARIABLES].sel(
            time=select_times,
            longitude=slice(lon0, lon1),
            latitude=slice(lat1, lat0),
        ).load()
    except Exception as exc:
        raise RuntimeError(
            f"failed to read WB2 ERA5 for {times}: {exc}") from exc
    for stamp, path in missing:
        one = subset.sel(time=np.datetime64(_str_to_dt(stamp)))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        try:
            one.to_netcdf(tmp, engine="h5netcdf")
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        print(f"[wb2] wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def ensure_wb2_era5(
    timestamps,
    region: str = "global",
    out_dir: Path | None = None,
    frames: int = 4,
    step_hours: int = 6,
    batch: int = 8,
) -> list[Path]:
    """Download/cache the ERA5 frames for ``timestamps`` from WeatherBench2.

    Parameters
    ----------
    timestamps:
        Target valid times in ``YYYYMMDDTHH`` format.
    region:
        ``global`` (full 0-360 grid) or ``hk`` (111-117E, 19-25N).
    out_dir:
        Destination; defaults to ``data/era5/wb2/<region>``.
    frames:
        Number of 6-hourly frames including the target (4 for the global
        models, 8 for the Hong Kong four-variable model).
    step_hours:
        Spacing between frames; the released models were trained with 6 h.
    batch:
        Frames fetched per Zarr read (bounds peak memory).
    """
    key, _ = _region_bounds(region)
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir(key)
    times = required_times(timestamps, frames=frames, step_hours=step_hours)
    if not times:
        return []
    _check_coverage([_str_to_dt(t) for t in times])

    paths = _paths_for(times, out_dir)
    missing = [(t, p) for t, p in zip(times, paths)
               if not p.exists() or p.stat().st_size == 0]
    if not missing:
        print(f"[wb2] ERA5 cache hit: {len(paths)} frame(s) under {out_dir}")
        return paths

    print(f"[wb2] downloading {len(missing)} ERA5 frame(s) from WeatherBench2 "
          f"({key})")
    print(f"[wb2] {WB2_ERA5_URL}")
    ds = open_wb2(WB2_ERA5_URL)
    try:
        for i in range(0, len(missing), max(1, int(batch))):
            _write_frames(ds, missing[i:i + max(1, int(batch))], key)
    finally:
        ds.close()
    return paths


def ensure_wb2_era5_range(
    start: str,
    end: str,
    region: str = "hk",
    step_hours: int = 6,
    out_dir: Path | None = None,
    batch: int = 96,
) -> list[Path]:
    """Fetch every frame of a contiguous window (used by the ED 3 edge histograms).

    ``start``/``end`` are ``YYYYMMDDTHH`` valid times.  For the Hong Kong
    domain a two-year 6-hourly window is only a few tens of MB, so one call is
    much cheaper than thousands of single-timestamp reads.
    """
    key, _ = _region_bounds(region)
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir(key)
    t0, t1 = _str_to_dt(start), _str_to_dt(end)
    if t1 < t0:
        raise ValueError(f"end {end} is before start {start}")
    times, t = [], t0
    while t <= t1:
        times.append(t.strftime("%Y%m%dT%H"))
        t += timedelta(hours=int(step_hours))
    _check_coverage([_str_to_dt(x) for x in times])

    paths = _paths_for(times, out_dir)
    missing = [(stamp, p) for stamp, p in zip(times, paths)
               if not p.exists() or p.stat().st_size == 0]
    if not missing:
        print(f"[wb2] ERA5 range cache hit: {len(paths)} frame(s) under {out_dir}")
        return paths
    print(f"[wb2] downloading {len(missing)} ERA5 frame(s) "
          f"({start}..{end}, {step_hours} h, {key}) from WeatherBench2")
    ds = open_wb2(WB2_ERA5_URL)
    try:
        for i in range(0, len(missing), max(1, int(batch))):
            _write_frames(ds, missing[i:i + max(1, int(batch))], key)
    finally:
        ds.close()
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="global", choices=sorted(REGIONS))
    ap.add_argument("--times", default=None,
                    help="comma-separated target valid times (YYYYMMDDTHH)")
    ap.add_argument("--start", default=None, help="window start (YYYYMMDDTHH)")
    ap.add_argument("--end", default=None, help="window end (YYYYMMDDTHH)")
    ap.add_argument("--frames", type=int, default=4,
                    help="6-hourly history frames per target (4 global, 8 HK)")
    ap.add_argument("--step-hours", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else None
    if args.start and args.end:
        paths = ensure_wb2_era5_range(args.start, args.end, region=args.region,
                                      step_hours=args.step_hours, out_dir=out)
    elif args.times:
        stamps = [t.strip() for t in args.times.split(",") if t.strip()]
        paths = ensure_wb2_era5(stamps, region=args.region, out_dir=out,
                                frames=args.frames, step_hours=args.step_hours)
    else:
        raise SystemExit("give --times, or --start with --end")
    print(f"[wb2] {len(paths)} ERA5 frame(s) ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
