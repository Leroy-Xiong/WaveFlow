"""Download just the ERA5 frames needed by an inference case from WeatherBench2.

The public release no longer ships ERA5. Instead, the inference helper fetches
the required surface variables for a case window (target minus 6-hourly history
frames) directly from the public WeatherBench2 ERA5 Zarr store and writes one
NetCDF file per timestamp. No daily packing and no timestamp index are needed.

The resulting layout is::

    data/era5/wb2/<YYYY>/era5_<YYYYMMDDTHH>.nc

which is what ``WaveDataset`` expects for ``era5_file_format: nc`` with
``use_preinterpolated_era5: false``.
"""
from __future__ import annotations

import os
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
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
REGIONS = {
    "global": (0.0, 360.0, -90.0, 90.0),
    "hk": (111.0, 117.0, 19.0, 25.0),
}


def required_times(timestamps, frames: int = 4, step_hours: int = 6) -> list[str]:
    """Return the sorted, unique ERA5 times needed for ``timestamps``."""
    out = set()
    for stamp in timestamps:
        t = datetime.strptime(stamp, "%Y%m%dT%H")
        for i in range(int(frames)):
            out.add((t - timedelta(hours=step_hours * i)).strftime("%Y%m%dT%H"))
    return sorted(out)


def _region_bounds(region: str):
    key = "global" if region.lower().startswith("global") else region.lower()
    if key not in REGIONS:
        raise ValueError(f"unknown ERA5 region {region!r}; choose from {sorted(REGIONS)}")
    return REGIONS[key]


def ensure_wb2_era5(
    timestamps,
    region: str = "global",
    out_dir: Path | None = None,
    frames: int = 4,
    step_hours: int = 6,
) -> list[Path]:
    """Download/cache the ERA5 frames for ``timestamps`` from WeatherBench2.

    Parameters
    ----------
    timestamps:
        Target valid times in ``YYYYMMDDTHH`` format.
    region:
        ``global`` (full 0-360 grid) or ``hk`` (111-117E, 19-25N).
    out_dir:
        Destination; defaults to ``data/era5/wb2``.
    frames:
        Number of 6-hourly frames including the target (the model uses 4).
    step_hours:
        Spacing between frames; the released models were trained with 6 h.
    """
    out_dir = Path(out_dir) if out_dir is not None else REPO / "data" / "era5" / "wb2"
    lon0, lon1, lat0, lat1 = _region_bounds(region)
    times = required_times(timestamps, frames=frames, step_hours=step_hours)
    if not times:
        return []

    requested = [datetime.strptime(t, "%Y%m%dT%H") for t in times]
    too_late = [t for t in requested if t > WB2_ERA5_END]
    if too_late:
        raise RuntimeError(
            "WeatherBench2 ERA5 ends at 2023-01-10 23:00; requested "
            f"{too_late[0]:%Y%m%dT%H}. Use another ERA5 source for this case."
        )

    paths = [out_dir / t[:4] / f"era5_{t}.nc" for t in times]
    missing = [(t, p) for t, p in zip(times, paths)
               if not p.exists() or p.stat().st_size == 0]
    if not missing:
        print(f"[wb2] ERA5 cache hit: {len(paths)} frames under {out_dir}")
        return paths

    print(f"[wb2] downloading {len(missing)} ERA5 frame(s) from WeatherBench2")
    print(f"[wb2] {WB2_ERA5_URL}")
    try:
        ds = xr.open_zarr(WB2_ERA5_URL, consolidated=True)
    except Exception as exc:
        raise RuntimeError(
            f"cannot open WeatherBench2 ERA5 at {WB2_ERA5_URL}: {exc}"
        ) from exc

    try:
        select_times = np.array([np.datetime64(datetime.strptime(t, "%Y%m%dT%H"))
                                for t, _ in missing])
        try:
            subset = ds[VARIABLES].sel(
                time=select_times,
                longitude=slice(lon0, lon1),
                latitude=slice(lat1, lat0),
            ).load()
        except Exception as exc:
            raise RuntimeError(
                f"failed to read WB2 ERA5 for {[t for t, _ in missing]}: {exc}"
            ) from exc

        for stamp, path in missing:
            one = subset.sel(time=np.datetime64(datetime.strptime(stamp, "%Y%m%dT%H")))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + f".tmp{os.getpid()}")
            try:
                one.to_netcdf(tmp, engine="h5netcdf")
                os.replace(tmp, path)
            finally:
                if tmp.exists():
                    tmp.unlink()
            print(f"[wb2] wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    finally:
        ds.close()

    return paths
