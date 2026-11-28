"""Real-time IFS ENS forcing from ECMWF open data.

The released checkpoints are trained on ERA5 reanalysis. To run them as an
operational forecast, the atmosphere has to come from a forecast instead: this
module pulls the IFS ensemble (stream ``enfo``, 0.25 deg) from
`ECMWF open data <https://data.ecmwf.int/forecasts/>`_ and writes it in the
layout ``WaveDataset`` expects for ``atm_source='ens'``::

    <ens_dir>/lead_time_<L>h/<init_year>/ifs_ens_<init>.nc

Each file holds the ensemble-mean of the three surface fields the model
consumes (10 m u/v wind, mean sea level pressure) on the Hong Kong window
(lat 15.0-24.75, lon 110.0-119.75, 0.25 deg, lat ascending), which is exactly
what the training-era WeatherBench2 export used.

Open data publishes one GRIB2 file per step plus an ``.index`` listing the byte
offset of every field, so a single member/parameter costs one small HTTP Range
request instead of the whole multi-gigabyte step file.

GRIB decoding needs ``eccodes`` (or ``cfgrib``); they are optional:

    conda install -c conda-forge eccodes
    # or: pip install --target .deps eccodes cfgrib
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT_URL = "https://data.ecmwf.int/forecasts"
AWS_ROOT_URL = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
STREAM = "enfo"
RESOL = "0p25"

PARAMS = {"10u": "10m_u_component_of_wind",
          "10v": "10m_v_component_of_wind",
          "msl": "mean_sea_level_pressure"}
VAR_ORDER = ["10m_u_component_of_wind", "10m_v_component_of_wind",
             "mean_sea_level_pressure"]

LAT_RANGE = (15.0, 24.75)
LON_RANGE = (110.0, 119.75)
FRAME_STEP_H = 6          # the Hong Kong config uses 8 frames, 6 h apart
N_FRAMES = 8


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _deps_path() -> None:
    """Use a local ``.deps`` install when the environment has no eccodes."""
    deps = Path(__file__).resolve().parents[1] / ".deps"
    if deps.is_dir() and str(deps) not in sys.path:
        sys.path.insert(0, str(deps))


_SESSION = None
_SESSION_LOCK = __import__("threading").Lock()


def _session():
    """One pooled ``requests`` session: the server answers in ~2 s per request,
    so connection reuse matters more than anything else here."""
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                import requests
                from requests.adapters import HTTPAdapter

                s = requests.Session()
                s.mount("https://", HTTPAdapter(pool_connections=32,
                                                pool_maxsize=32, max_retries=0))
                s.headers["User-Agent"] = "waveflow-forecast/1.0"
                _SESSION = s
    return _SESSION


def _request(url: str, byte_range: tuple[int, int] | None = None,
             retries: int = 6, timeout: int = 60) -> bytes:
    headers = {}
    if byte_range is not None:
        start, length = byte_range
        headers["Range"] = f"bytes={start}-{start + length - 1}"
    last = None
    for attempt in range(retries):
        try:
            resp = _session().get(url, headers=headers, timeout=timeout)
            if resp.status_code == 404:
                raise urllib.error.HTTPError(url, 404, "not found", None, None)
            resp.raise_for_status()
            return resp.content
        except urllib.error.HTTPError:
            raise
        except Exception as exc:  # noqa: BLE001 - network hiccups
            last = exc
        time.sleep(min(2.0, 0.3 * (attempt + 1)))
    raise RuntimeError(f"GET {url} failed: {last}")


def _cycle_base(init: str, step: int, root: str) -> str:
    day, hh = init[:8], init[9:11]
    return (f"{root}/{day}/{hh}z/ifs/{RESOL}/{STREAM}/"
            f"{day}{hh}0000-{step}h-{STREAM}-ef")


_ROOT_CHOICE: dict[tuple[str, int], str] = {}


def _index_for(init: str, step: int, cache: Path) -> tuple[list[dict], str]:
    """Index records plus the data root that served them.

    ``data.ecmwf.int`` keeps a rolling window of a few days; older cycles live
    on the AWS mirror, so the root that worked is remembered per step.
    """
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{init}_{step:03d}h.index"
    if path.exists():
        root = _ROOT_CHOICE.get((init, step), ROOT_URL)
        return [json.loads(line) for line in path.read_text().splitlines() if line], root
    last = None
    for root in (ROOT_URL, AWS_ROOT_URL):
        try:
            raw = _request(_cycle_base(init, step, root) + ".index")
        except Exception as exc:  # noqa: BLE001
            last = exc
            continue
        path.write_bytes(raw)
        _ROOT_CHOICE[(init, step)] = root
        return ([json.loads(line) for line in raw.decode().splitlines() if line.strip()],
                root)
    raise RuntimeError(f"no index for {init}+{step}h ({last})")


def _record(index: list[dict], param: str, member: int) -> dict:
    for rec in index:
        if rec.get("param") == param and int(rec.get("number", -1)) == member:
            return rec
    raise KeyError(f"{param} member {member} absent from the index")


def _decode(blob: bytes) -> np.ndarray:
    _deps_path()
    import eccodes

    gid = eccodes.codes_new_from_message(blob)
    try:
        ni = eccodes.codes_get(gid, "Ni")
        nj = eccodes.codes_get(gid, "Nj")
        return eccodes.codes_get_array(gid, "values").reshape(nj, ni).astype(np.float32)
    finally:
        eccodes.codes_release(gid)


def _crop(field: np.ndarray) -> np.ndarray:
    lats = np.arange(90.0, -90.0 - 1e-9, -0.25)
    lons = np.arange(-180.0, 180.0 - 1e-9, 0.25)
    if field.shape != (len(lats), len(lons)):
        raise RuntimeError(f"unexpected global shape {field.shape}")
    li = np.where((lats >= LAT_RANGE[0] - 1e-6) & (lats <= LAT_RANGE[1] + 1e-6))[0]
    lj = np.where((lons >= LON_RANGE[0] - 1e-6) & (lons <= LON_RANGE[1] + 1e-6))[0]
    return np.ascontiguousarray(field[np.ix_(li, lj)][::-1])


# ---------------------------------------------------------------------------
# cycles
# ---------------------------------------------------------------------------
def latest_cycle(now: datetime | None = None, back: int = 8) -> str:
    """Most recent cycle whose step-0 index is online (cycles land ~7 h late)."""
    now = now or datetime.utcnow()
    t = now.replace(minute=0, second=0, microsecond=0)
    t -= timedelta(hours=t.hour % 6)
    for _ in range(back):
        init = t.strftime("%Y%m%dT%H")
        try:
            _request(_cycle_base(init, 0, ROOT_URL) + ".index", timeout=20)
            return init
        except Exception:  # noqa: BLE001
            t -= timedelta(hours=6)
    raise RuntimeError("no recent IFS ENS cycle found on ECMWF open data")


def frame_plan(init: str, lead_hours: int, frames: int = N_FRAMES,
               stride: int = FRAME_STEP_H) -> list[tuple[int, str]]:
    """``(step, frame_init)`` pairs the dataset will ask for at this lead.

    Mirrors ``WaveDataset._build_ens_frame_plan``: the frames inside the main
    chain come from ``init``, the padding frames from the most recent 00/12Z
    cycle before them.
    """
    dt_main = datetime.strptime(init, "%Y%m%dT%H")
    dt_valid = dt_main + timedelta(hours=lead_hours)
    plan = []
    for i in range(frames - 1, -1, -1):
        dt_frame = dt_valid - timedelta(hours=i * stride)
        if dt_frame >= dt_main:
            dt_init = dt_main
        else:
            hour = 12 if dt_frame.hour >= 12 else 0
            dt_init = dt_frame.replace(hour=hour, minute=0, second=0)
        step = int((dt_frame - dt_init).total_seconds() // 3600)
        plan.append((step, dt_init.strftime("%Y%m%dT%H")))
    return plan


def _fetch_root(index, init: str, step: int, members: list[int],
                root: str) -> dict[str, np.ndarray]:
    base = _cycle_base(init, step, root)
    jobs = [(name, m, _record(index, name, m)) for m in members for name in PARAMS]
    acc = {name: None for name in VAR_ORDER}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_request, base + ".grib2",
                               (int(rec["_offset"]), int(rec["_length"]))): name
                   for name, _, rec in jobs}
        for fut in futures:
            name = futures[fut]
            field = _decode(fut.result())            # decode serially
            key = PARAMS[name]
            acc[key] = field if acc[key] is None else acc[key] + field
    return {k: v / len(members) for k, v in acc.items()}


def fetch_mean(init: str, step: int, members: list[int], cache: Path,
               crop: bool = True) -> dict[str, np.ndarray]:
    """Ensemble mean of the three surface fields for one (init, step).

    ``crop`` selects the Hong Kong window; ``crop=False`` keeps the full
    0.25 deg globe (lat 90 -> -90, lon -180 -> 179.75) for the global model.
    """
    index, root = _index_for(init, step, cache)
    other = AWS_ROOT_URL if root == ROOT_URL else ROOT_URL
    fields = None
    for attempt in (root, other):
        try:
            fields = _fetch_root(index, init, step, members, attempt)
            _ROOT_CHOICE[(init, step)] = attempt
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 404 or attempt == other:
                raise
    if not crop:
        return fields
    return {k: _crop(v) for k, v in fields.items()}


def write_hk(ens_dir: Path, init: str, step: int,
             fields: dict[str, np.ndarray]) -> Path:
    """Write one forcing file in the layout ``atm_source='ens'`` expects."""
    import xarray as xr

    lats = np.arange(LAT_RANGE[0], LAT_RANGE[1] + 1e-9, 0.25, dtype=np.float32)
    lons = np.arange(LON_RANGE[0], LON_RANGE[1] + 1e-9, 0.25, dtype=np.float32)
    ds = xr.Dataset(
        {name: (("latitude", "longitude"), fields[name]) for name in VAR_ORDER},
        coords={"latitude": lats, "longitude": lons},
        attrs={"init_time": init, "lead_time_hours": step,
               "source": "ECMWF open data, IFS ENS (enfo), ensemble mean"})
    path = ens_dir / f"lead_time_{step}h" / init[:4] / f"ifs_ens_{init}.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp)
    tmp.replace(path)
    return path

# ---------------------------------------------------------------------------
# global layout (ERA5-format npz the global checkpoint consumes)
# ---------------------------------------------------------------------------
def global_coords() -> tuple[np.ndarray, np.ndarray]:
    """Grid of the ECMWF open-data fields: lat 90 -> -90, lon -180 -> 179.75."""
    return (np.arange(90.0, -90.0 - 1e-9, -0.25, dtype=np.float32),
            np.arange(-180.0, 180.0 - 1e-9, 0.25, dtype=np.float32))


def write_global(era5_dir: Path, valid: str,
                 fields: dict[str, np.ndarray]) -> Path:
    """Write one global forcing file in the ERA5 ``npz`` layout.

    The longitude axis is stored as -180..180 so it matches the field ordering
    that ``eccodes`` returns; ``WaveDataset`` only re-sorts when the stored
    longitudes exceed 180, and a mismatch there shifts the field by 180 deg.
    """
    lats, lons = global_coords()
    for name in VAR_ORDER:
        if fields[name].shape != (len(lats), len(lons)):
            raise RuntimeError(f"{name}: unexpected shape {fields[name].shape}")
    year = valid[:4]
    path = era5_dir / year / f"era5_{valid}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = era5_dir / "era5_coordinates.npz"
    if not coords.exists():
        np.savez(coords, latitude=lats, longitude=lons)
    tmp = path.with_name(path.stem + ".part.npz")   # savez keeps a .npz suffix
    np.savez_compressed(tmp, **{n: fields[n].astype(np.float32) for n in VAR_ORDER},
                        _latitude=lats, _longitude=lons)
    tmp.replace(path)
    return path
