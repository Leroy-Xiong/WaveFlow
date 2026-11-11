#!/usr/bin/env python
"""Fetch LandsD HK 5 m DTM LERC tiles (ArcGIS ImageServer, level 15) and mosaic.

Service: https://tiles.arcgis.com/tiles/6j1KwZfY2fZrfNMR/arcgis/rest/services/HK_DTM/ImageServer
("Whole_HK_DTM_5M_tilepkg", Survey and Mapping Office, Lands Department;
2020 LiDAR survey, 5 m grid, accuracy +/-5 m; heights above HK Principal
Datum, HKPD; note HKPD = MSL - 1.23 m, HKO mCD = MSL + 1.38 m).

Tiles: LERC float32, 257x257 px payload on a 256 px grid (1 px overlap,
first 256x256 used), Web-Mercator (wkid 3857), origin top-left
(-20037508.342787, 20037508.342787), level 15 res = 4.77731426794937 m/px.

Findings on sea/NoData (2026-09-02 inspection):
- No LERC validity mask; no -3.4e38 NoData sentinels; no NaN.
- Sea areas are void-filled by the cache builder: open sea (e.g. Mirs Bay)
  is a uniform 2.0 m; near-coast sea is smoothly interpolated from nearby
  land (Victoria Harbour centre reads ~10 m). Sea pixels are therefore NOT
  identifiable from the tile values alone; they act as a low floor for
  landward-max crest proxies.
- Tiles outside the service extent return HTTP 404 -> stored as NaN.

Usage:
    python scripts/fetch_landsd_dtm.py            # fetch + mosaic
    python scripts/fetch_landsd_dtm.py --mosaic-only
"""
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), '.geo_deps'))
import imagecodecs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILE_DIR = os.path.join(ROOT, 'data/hk_coastline/dtm_landsd/tiles/L15')
OUT_NPZ = os.path.join(ROOT, 'data/hk_coastline/dtm_landsd/hk_dtm_l15.npz')

BASE = ('https://tiles.arcgis.com/tiles/6j1KwZfY2fZrfNMR/arcgis/rest/'
        'services/HK_DTM/ImageServer')
LEVEL = 15
RES = 4.77731426794937          # m/px at level 15
ORIGIN_X = -20037508.342787
ORIGIN_Y = 20037508.342787
R_EARTH = 6378137.0
LON0, LON1 = 113.80, 114.48     # task bbox with margin
LAT0, LAT1 = 22.13, 22.60
UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) fetch_landsd_dtm/1.0'}


def lonlat_to_merc(lon, lat):
    return (R_EARTH * math.radians(lon),
            R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def tile_range():
    x0, y1 = lonlat_to_merc(LON0, LAT1)   # NW
    x1, y0 = lonlat_to_merc(LON1, LAT0)   # SE
    c0 = int(math.floor((x0 - ORIGIN_X) / (RES * 256)))
    c1 = int(math.floor((x1 - ORIGIN_X) / (RES * 256)))
    r0 = int(math.floor((ORIGIN_Y - y1) / (RES * 256)))
    r1 = int(math.floor((ORIGIN_Y - y0) / (RES * 256)))
    return r0, r1, c0, c1


def fetch_tile(job):
    row, col = job
    path = os.path.join(TILE_DIR, f'{row}_{col}.lerc')
    if os.path.exists(path):
        return 'cached'
    url = f'{BASE}/tile/{LEVEL}/{row}/{col}'
    for k in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 404:
                return 'missing'        # outside service extent (open sea)
            r.raise_for_status()
            tmp = path + '.tmp'
            with open(tmp, 'wb') as f:
                f.write(r.content)
            os.replace(tmp, path)
            return 'fetched'
        except Exception:
            if k == 2:
                return 'failed'
            time.sleep(2 * (k + 1))


def fetch_all():
    r0, r1, c0, c1 = tile_range()
    jobs = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    print(f'level {LEVEL}: rows {r0}..{r1}, cols {c0}..{c1} -> {len(jobs)} tiles')
    os.makedirs(TILE_DIR, exist_ok=True)
    t0 = time.time()
    from collections import Counter
    stats = Counter()
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, res in enumerate(ex.map(fetch_tile, jobs)):
            stats[res] += 1
            if (i + 1) % 500 == 0:
                print(f'  {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s) '
                      f'{dict(stats)}')
    print(f'done in {time.time() - t0:.0f}s: {dict(stats)}')
    if stats['failed']:
        raise SystemExit('some tiles failed; rerun to retry')


def mosaic():
    r0, r1, c0, c1 = tile_range()
    nrow, ncol = (r1 - r0 + 1) * 256, (c1 - c0 + 1) * 256
    elev = np.full((nrow, ncol), np.nan, dtype=np.float32)
    n_ok = n_missing = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            path = os.path.join(TILE_DIR, f'{r}_{c}.lerc')
            if not os.path.exists(path):
                n_missing += 1
                continue
            with open(path, 'rb') as f:
                arr = imagecodecs.lerc_decode(f.read())
            if isinstance(arr, tuple):
                arr = arr[0]
            iy, ix = (r - r0) * 256, (c - c0) * 256
            elev[iy:iy + 256, ix:ix + 256] = arr[:256, :256]
            n_ok += 1
    # sentinel safety (none seen in practice, but Esri convention)
    elev = np.where(elev < -1e30, np.nan, elev).astype(np.float32)
    origin_x = ORIGIN_X + c0 * 256 * RES   # left edge of mosaic
    origin_y = ORIGIN_Y - r0 * 256 * RES   # top edge of mosaic
    np.savez_compressed(OUT_NPZ, elev=elev, origin_x=origin_x,
                        origin_y=origin_y, res=RES, level=LEVEL,
                        row0=r0, col0=c0)
    print(f'mosaic {elev.shape}, {n_ok} tiles used, {n_missing} missing (sea)')
    print(f'origin_x={origin_x:.3f} origin_y={origin_y:.3f} res={RES}')
    finite = np.isfinite(elev)
    print(f'finite {finite.mean() * 100:.1f}%, elev range '
          f'[{np.nanmin(elev):.2f}, {np.nanmax(elev):.2f}] m')
    print(f'wrote {OUT_NPZ}')


if __name__ == '__main__':
    if '--mosaic-only' not in sys.argv:
        fetch_all()
    mosaic()
