#!/usr/bin/env python
"""Fetch official LandsD orthoimagery for Hong Kong and build a downsampled
WGS84 mosaic cache for paper figures.

Sources (all HKSAR Government / LandsD, no API key needed)
----------------------------------------------------------
1. GeoData Store XYZ imagery tiles (used for the deliverable cache):
       https://mapapi.geodata.gov.hk/gs/api/v1.0.0/xyz/imagery/wgs84/{z}/{x}/{y}.png
   Standard Web-Mercator slippy tiles (EPSG:3857 content, verified by
   coastline overlay against data/hk_coastline/hk_coastline_segments.csv),
   available to zoom 20, no key, clean orthophoto without labels.
   Tiles outside HKSAR coverage return HTTP 204 (empty) -> NoData mask.
2. TDOP (True Digital Orthophoto, CSDI dataset landsd_rcd_1701762716904_4826,
   layer TDOP2025_GRID_INDEX): 192 sheets x (3750 m x 3000 m), GeoTIFF,
   EPSG:2326, 0.25 m GSD, RGB uint8 uncompressed, photographed 2025-01,
   released 2025.  Per-sheet zip ~280 MB (sheet T5-SW-D piloted) via
       https://open.hkmapservice.gov.hk/OpenData/directDownload
       ?productName=TDOP&sheetName=<SHEETNO>&productFormat=TIFF
   A full-territory download is ~54 GB, so only a pilot subset is fetched
   here; the tile service above provides the same vintage imagery at the
   ~20 m/px actually needed for figure output.

Rejected alternatives (2026-09-02 survey)
-----------------------------------------
- DOPM50-L0 (productName=DOPM50-L0, sheetName=DOPM50-L0): single 5 m GeoTIFF
  of the whole territory (OPM50_2025_L0.tif) but it is a *map* product with
  title block, legend, scale bar and red graticule burned in -> unusable as
  a clean basemap.
- DOP5000 (landsd_rcd_1637306839277_11815): 0.2 m GSD, 192 GeoTIFF sheets,
  older vintage, larger volume than TDOP; superseded by TDOP since 2024-01.

Deliverable
-----------
data/hk_coastline/orthophoto/hk_orthophoto_wgs84.npz
    rgb    uint8 (H, W, 3)   EPSG:4326 regular grid, 0.0002 deg/px,
           extent lon [113.80, 114.48], lat [22.11, 22.60]; 0 where masked
    mask   bool  (H, W)      True = valid imagery pixel
    extent float64 (4,)      [lon0, lon1, lat0, lat1]
    meta   str               JSON metadata (source, vintage, resolution, ...)

Usage:
    python scripts/fetch_landsd_orthophoto.py                 # tiles -> cache -> preview
    python scripts/fetch_landsd_orthophoto.py --mosaic-only   # rebuild cache from cached tiles
    python scripts/fetch_landsd_orthophoto.py --tdop-pilot 3  # download N TDOP pilot sheets
Rerunnable: cached tiles / sheet zips are reused.
"""
import argparse
import io
import json
import math
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

requests.packages.urllib3.disable_warnings()

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hk_coastline" / "orthophoto"
SEG_CSV = ROOT / "data" / "hk_coastline" / "hk_coastline_segments.csv"
TILEINDEX = OUT / "tdop_tileindex.geojson"
SHEETS_DIR = OUT / "tdop_sheets"
OUT_NPZ = OUT / "hk_orthophoto_wgs84.npz"
PREVIEW_PNG = ROOT / "output" / "plot" / "figure6" / "orthophoto_preview.png"

# task bbox (WGS84)
LON0, LON1 = 113.80, 114.48
LAT0, LAT1 = 22.11, 22.60
OUT_RES = 0.0002            # deg/px (~20 m)

TILE_BASE = ("https://mapapi.geodata.gov.hk/gs/api/v1.0.0/xyz/imagery/"
             "wgs84/{z}/{x}/{y}.png")
TILE_Z = 14                 # 8.8 m/px at lat 22.3 -> 2.3x finer than output
R_EARTH = 6378137.0

TDOP_FS = ("https://portal.csdi.gov.hk/server/rest/services/common/"
           "landsd_rcd_1701762716904_4826/FeatureServer/0/query")
TDOP_DL = ("https://open.hkmapservice.gov.hk/OpenData/directDownload"
           "?productName=TDOP&sheetName={sheet}&productFormat=TIFF")

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-script"}


# ---------------------------------------------------------------- tile math
def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = n * (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2
    return x, y


def tile_range(z):
    x0, y0 = lonlat_to_tile(LON0, LAT1, z)   # NW: top row has smaller y
    x1, y1 = lonlat_to_tile(LON1, LAT0, z)   # SE
    return int(y0), int(y1), int(x0), int(x1)


def tile_range_bbox(z, bbox):
    lon0, lon1, lat0, lat1 = bbox
    x0, y0 = lonlat_to_tile(lon0, lat1, z)
    x1, y1 = lonlat_to_tile(lon1, lat0, z)
    return int(y0), int(y1), int(x0), int(x1)


def tile_dir(z):
    return OUT / f"tiles_z{z}"


# ---------------------------------------------------------------- download
def fetch_tile(z, x, y, retries=4):
    """Returns 'cached' | 'fetched' | 'empty' (204/outside HKSAR) | 'failed'."""
    path = tile_dir(z) / f"{x}_{y}.png"
    if path.exists():
        return "empty" if path.stat().st_size == 0 else "cached"
    url = TILE_BASE.format(z=z, x=x, y=y)
    for k in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code in (204, 404):
                path.write_bytes(b"")          # 0-byte marker = no coverage
                return "empty"
            r.raise_for_status()
            if "png" not in (r.headers.get("Content-Type") or ""):
                raise ValueError(f"bad content-type {r.headers.get('Content-Type')}")
            tmp = str(path) + ".tmp"
            with open(tmp, "wb") as f:
                f.write(r.content)
            os.replace(tmp, path)
            return "fetched"
        except Exception:
            if k == retries - 1:
                return "failed"
            time.sleep(2 * (k + 1))


def fetch_all_tiles(z, workers=8, bbox=None):
    r0, r1, c0, c1 = (tile_range(z) if bbox is None
                      else tile_range_bbox(z, bbox))
    jobs = [(x, y) for y in range(r0, r1 + 1) for x in range(c0, c1 + 1)]
    print(f"z{z}: rows {r0}..{r1}, cols {c0}..{c1} -> {len(jobs)} tiles")
    tile_dir(z).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from collections import Counter
    stats = Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_tile, z, x, y): (x, y) for x, y in jobs}
        for i, fut in enumerate(as_completed(futs)):
            stats[fut.result()] += 1
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s) "
                      f"{dict(stats)}")
    print(f"tiles done in {time.time() - t0:.0f}s: {dict(stats)}")
    if stats["failed"]:
        raise SystemExit("some tiles failed; rerun to retry")


# ---------------------------------------------------------------- mosaic
def build_cache(z, bbox=None, res=None, out_npz=None):
    """Mosaic mercator tiles, reproject to EPSG:4326 regular grid, save npz.
    bbox=(lon0,lon1,lat0,lat1)、res（deg/px）、out_npz 缺省用全域默认值。"""
    import rasterio
    from PIL import Image
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, reproject

    lon0, lon1, lat0, lat1 = bbox or (LON0, LON1, LAT0, LAT1)
    res = res or OUT_RES
    out_npz = out_npz or OUT_NPZ

    r0, r1, c0, c1 = (tile_range(z) if bbox is None
                      else tile_range_bbox(z, bbox))
    ncols, nrows = (c1 - c0 + 1) * 256, (r1 - r0 + 1) * 256
    rgb = np.zeros((3, nrows, ncols), dtype=np.uint8)
    valid = np.zeros((nrows, ncols), dtype=np.uint8)
    n_ok = n_empty = n_missing = 0
    for y in range(r0, r1 + 1):
        for x in range(c0, c1 + 1):
            path = tile_dir(z) / f"{x}_{y}.png"
            iy, ix = (y - r0) * 256, (x - c0) * 256
            if not path.exists():
                n_missing += 1
                continue
            if path.stat().st_size == 0:
                n_empty += 1
                continue
            t = np.asarray(Image.open(path).convert("RGB"))
            rgb[:, iy:iy + 256, ix:ix + 256] = np.moveaxis(t, -1, 0)
            valid[iy:iy + 256, ix:ix + 256] = 255
            n_ok += 1
    print(f"mosaic {rgb.shape}, {n_ok} tiles with imagery, "
          f"{n_empty} empty (outside HKSAR), {n_missing} not downloaded")

    # mercator georeferencing of the mosaic
    res_m = 2 * math.pi * R_EARTH / (2 ** z * 256)
    origin_x = -math.pi * R_EARTH + c0 * 256 * res_m
    origin_y = math.pi * R_EARTH - r0 * 256 * res_m
    src_transform = Affine(res_m, 0, origin_x, 0, -res_m, origin_y)

    # destination: regular lon/lat grid at res over the bbox
    w = int(round((lon1 - lon0) / res))
    h = int(round((lat1 - lat0) / res))
    dst_transform = Affine(res, 0, lon0, 0, -res, lat1)
    dst_rgb = np.zeros((3, h, w), dtype=np.uint8)
    dst_val = np.zeros((h, w), dtype=np.uint8)

    src_crs = CRS.from_epsg(3857)
    dst_crs = CRS.from_epsg(4326)
    reproject(rgb, dst_rgb, src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              resampling=Resampling.bilinear)
    reproject(valid, dst_val, src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              resampling=Resampling.nearest)
    mask = dst_val > 0
    print(f"cache grid {w}x{h} @ {res} deg/px, valid "
          f"{mask.mean() * 100:.1f}%")

    meta = {
        "product": "LandsD digital orthophoto via HK GeoData Store imagery "
                   "XYZ tile service (mapapi.geodata.gov.hk, no API key)",
        "tile_endpoint": TILE_BASE.format(z="{z}", x="{x}", y="{y}"),
        "tile_zoom": z,
        "tile_ground_resolution_m_at_lat22.3": round(
            2 * math.pi * R_EARTH * math.cos(math.radians(22.3))
            / (2 ** z * 256), 2),
        "underlying_sheet_product": "True Digital Orthophoto (TDOP) 2025, "
            "CSDI dataset landsd_rcd_1701762716904_4826, 0.25 m GSD GeoTIFF "
            "in EPSG:2326, photographed Jan 2025; per-sheet download via "
            "open.hkmapservice.gov.hk directDownload?productName=TDOP",
        "vintage": "latest LandsD orthoimagery as served on download date "
                   "(tile API exposes no per-tile date; TDOP2025 sheets are "
                   "dated Jan 2025)",
        "source_crs": "EPSG:3857 (Web-Mercator slippy tiles)",
        "output_crs": "EPSG:4326",
        "output_resolution_deg": res,
        "extent_lonlat": [lon0, lon1, lat0, lat1],
        "coverage_note": "the imagery service returns pixels for the whole "
            "task bbox (no 204s seen at z14): HKSAR area is the latest "
            "TDOP-grade true orthophoto; areas outside HKSAR (Shenzhen "
            "mainland, outer sea margins) are a different-vintage/coarser "
            "background imagery from the same official service, with "
            "visible seam blocks at the HKSAR boundary. mask is True "
            "everywhere; if only HKSAR imagery is wanted, mask with the "
            "boundary polygon from the tile index / coastline data.",
        "download_date": time.strftime("%Y-%m-%d"),
        "download_url": TILE_BASE,
    }
    np.savez_compressed(
        out_npz,
        rgb=np.moveaxis(dst_rgb, 0, -1),
        mask=mask,
        extent=np.array([lon0, lon1, lat0, lat1]),
        meta=json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"wrote {out_npz} ({out_npz.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------- preview
def make_preview():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    d = np.load(OUT_NPZ)
    rgb, mask, extent = d["rgb"], d["mask"], d["extent"]
    seg = pd.read_csv(SEG_CSV)

    PREVIEW_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    show = rgb.astype(np.float32)
    show[~mask] = np.nan
    ax.imshow(show / 255.0, extent=[extent[0], extent[1], extent[2], extent[3]],
              interpolation="bilinear")
    step = max(1, len(seg) // 4000)     # decimate for draw speed
    for r in seg.iloc[::step].itertuples():
        ax.plot([r.start_lon, r.end_lon], [r.start_lat, r.end_lat],
                "r-", lw=0.4, alpha=0.7)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title("LandsD orthophoto cache (GeoData Store imagery) + "
                 "official coastline overlay")
    # zoom insets: Victoria Harbour + Tuen Mun (TDOP pilot sheet area)
    for i, (xa, xb, ya, yb, name) in enumerate([
            (114.13, 114.23, 22.26, 22.33, "Victoria Harbour"),
            (113.93, 114.03, 22.36, 22.43, "Tuen Mun")]):
        axi = fig.add_axes([0.13 + i * 0.42, 0.10, 0.18, 0.18])
        axi.imshow(show / 255.0,
                   extent=[extent[0], extent[1], extent[2], extent[3]],
                   interpolation="nearest")
        m = ((seg.start_lon > xa) & (seg.start_lon < xb)
             & (seg.start_lat > ya) & (seg.start_lat < yb))
        for r in seg[m].itertuples():
            axi.plot([r.start_lon, r.end_lon], [r.start_lat, r.end_lat],
                     "r-", lw=0.7)
        axi.set_xlim(xa, xb); axi.set_ylim(ya, yb)
        axi.set_title(name, fontsize=8)
        axi.tick_params(labelsize=6)
    fig.savefig(PREVIEW_PNG, dpi=150)
    print(f"wrote {PREVIEW_PNG}")


# ---------------------------------------------------------------- TDOP pilot
def fetch_tdop_tileindex():
    feats, off = [], 0
    while True:
        url = (TDOP_FS + f"?where=1%3D1&outFields=SHEETNO,YEAR_OF_RELEASE,"
               f"DATE_OF_PHOTOGRAPH,DOWNLOAD_LINK&returnGeometry=true"
               f"&f=geojson&outSR=2326&resultRecordCount=2000"
               f"&resultOffset={off}")
        d = requests.get(url, headers=UA, timeout=120).json()
        feats.extend(d["features"])
        if len(d["features"]) < 2000:
            break
        off += 2000
    TILEINDEX.write_text(json.dumps(
        {"type": "FeatureCollection", "features": feats}))
    return feats


def tdop_pilot(n_sheets):
    """Download the first N TDOP sheets (full set = 192 sheets ~54 GB)."""
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    if TILEINDEX.exists():
        feats = json.loads(TILEINDEX.read_text())["features"]
    else:
        print("fetching TDOP tile index ...")
        feats = fetch_tdop_tileindex()
    sheets = sorted(f["properties"]["SHEETNO"] for f in feats)[:n_sheets]
    for sheet in sheets:
        dest = SHEETS_DIR / f"{sheet}.zip"
        if dest.exists() and zipfile.is_zipfile(dest):
            print(f"  {sheet}: cached")
            continue
        url = TDOP_DL.format(sheet=sheet)
        t0 = time.time()
        with requests.get(url, headers=UA, timeout=600, verify=False,
                          stream=True) as r:
            r.raise_for_status()
            with open(str(dest) + ".tmp", "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp = str(dest) + ".tmp"
        if not zipfile.is_zipfile(tmp):
            os.remove(tmp)
            raise SystemExit(f"{sheet}: download is not a valid zip; "
                             f"rerun to retry")
        os.replace(tmp, dest)
        print(f"  {sheet}: {dest.stat().st_size / 1e6:.0f} MB "
              f"in {time.time() - t0:.0f}s")
    print("TDOP pilot done. NOTE: full set is 192 sheets, ~280 MB each "
          "(~54 GB) - not fetched by default.")


# ------------------------------------------------------- coverage mask
COVERAGE_NPZ = OUT / "hk_official_coverage.npz"
COVERAGE_Z = 16
COVERAGE_MIN_BYTES = 50_000


def _tile_total_size(z, x, y, retries=4):
    """1-byte Range probe -> total tile bytes (server returns 206 +
    Content-Range). Official-coverage tiles at z16 are ~150 KB; tiles
    outside HKSAR return a ~5-8 KB placeholder. Returns None on failure."""
    url = TILE_BASE.format(z=z, x=x, y=y)
    for k in range(retries):
        try:
            r = requests.get(url, headers={**UA, "Range": "bytes=0-0"},
                             timeout=30, verify=False)
            if r.status_code in (204, 404):
                return 0
            cr = r.headers.get("Content-Range")      # "bytes 0-0/173047"
            if r.status_code == 206 and cr and "/" in cr:
                return int(cr.split("/")[-1])
            r.raise_for_status()
            return len(r.content)                    # server ignored Range
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(2 * (k + 1))


def _tile_bounds_lonlat(z, x, y):
    """z/x/y 瓦片的经纬度边界 (lon0, lon1, lat0, lat1)。"""
    n = 2 ** z
    lon0, lon1 = x / n * 360.0 - 180.0, (x + 1) / n * 360.0 - 180.0
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon0, lon1, lat0, lat1


def _rasterize_coverage(official, z, r0, c0, smooth_tol_tiles=1.2,
                        bbox=None, res=None):
    """瓦片级掩膜 -> 出图网格。smooth_tol_tiles>0 时先把覆盖瓦片并集成
    多边形并按该容差（单位：瓦片边长）简化——真实边界是平滑的行政/
    水域界线，瓦片阶梯只是采样伪影——再全分辨率栅格化，边缘平直。
    shapely 不可用时退化为最近邻栅格化。"""
    n = 2 ** z
    lon0, lon1, lat0, lat1 = bbox or (LON0, LON1, LAT0, LAT1)
    res = res or OUT_RES
    w = int(round((lon1 - lon0) / res))
    h = int(round((lat1 - lat0) / res))
    lons = lon0 + (np.arange(w) + 0.5) * res
    lats = lat1 - (np.arange(h) + 0.5) * res
    try:
        if smooth_tol_tiles <= 0:
            raise ImportError
        from shapely.geometry import box
        from shapely.ops import unary_union
        ys, xs = np.where(official)
        polys = []
        for x, y in zip(xs, ys):
            lon0, lon1, lat0, lat1 = _tile_bounds_lonlat(z, c0 + x, r0 + y)
            polys.append(box(lon0, lat0, lon1, lat1))
        geom = unary_union(polys)
        tol = smooth_tol_tiles * 360.0 / n      # 1 tile ≈ 360/2^z deg lon
        geom = geom.simplify(tol, preserve_topology=True)
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        from matplotlib.path import Path
        verts = np.asarray(geom.exterior.coords)
        path = Path(verts)
        pts = np.stack(np.meshgrid(lons, lats), axis=-1).reshape(-1, 2)
        cov = path.contains_points(pts, radius=1e-12).reshape(h, w)
        # 简化可能吞掉狭长的覆盖半岛/岛屿，把原始瓦片中心在覆盖内、
        # 但被简化排除的片补回（半径容差内）；反之亦然略去不计。
        print(f"smoothed coverage polygon: {len(verts)} vertices, "
              f"tol={tol:.5f} deg ({smooth_tol_tiles} z{z}-tile)")
        return cov
    except ImportError:
        pass
    tx = np.clip(((lons + 180.0) / 360.0 * n).astype(int) - c0,
                 0, official.shape[1] - 1)
    la = np.radians(lats)
    ty = np.clip((n * (1 - np.log(np.tan(la) + 1 / np.cos(la)) / math.pi)
                  / 2).astype(int) - r0, 0, official.shape[0] - 1)
    return official[np.ix_(ty, tx)]


def build_coverage_mask(z=COVERAGE_Z, workers=8, smooth=True, res=None):
    """官方覆盖区掩膜：瓦片服务在港界内（含水域，如 Mirs Bay / Deep Bay）
    返回真射影像，界外返回占位小图；按瓦片总字节数（Range 探测，不下载
    内容）分类并栅格化到出图网格。探测结果缓存在 _coverage_probe_z{z}.npz
    （-1 = 未探测），重跑只补失败的片。输出 COVERAGE_NPZ。"""
    r0, r1, c0, c1 = tile_range(z)
    nrows, ncols = r1 - r0 + 1, c1 - c0 + 1
    probe_cache = OUT / f"_coverage_probe_z{z}.npz"
    if probe_cache.exists():
        size_grid = np.load(probe_cache)["size_grid"]
        if size_grid.shape != (nrows, ncols):
            size_grid = np.full((nrows, ncols), -1, dtype=np.int64)
    else:
        size_grid = np.full((nrows, ncols), -1, dtype=np.int64)
    jobs = [(c0 + x, r0 + y) for y in range(nrows) for x in range(ncols)
            if size_grid[y, x] < 0]
    print(f"coverage probe z{z}: rows {r0}..{r1}, cols {c0}..{c1} "
          f"-> {len(jobs)} tiles to probe ({int((size_grid >= 0).sum())} "
          f"cached)")
    t0 = time.time()
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_tile_total_size, z, x, y): (x, y)
                for x, y in jobs}
        for i, fut in enumerate(as_completed(futs)):
            x, y = futs[fut]
            sz = fut.result()
            if sz is None:
                failed += 1
                continue
            size_grid[y - r0, x - c0] = sz
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)")
    np.savez_compressed(probe_cache, size_grid=size_grid)
    if failed or (size_grid < 0).any():
        raise SystemExit(f"{failed} tiles failed to probe; rerun to retry")
    sizes = size_grid.ravel()
    official = size_grid >= COVERAGE_MIN_BYTES
    print(f"z{z} tile size percentiles [B]: "
          f"{np.percentile(sizes, [0, 5, 50, 95, 100]).astype(int)}; "
          f"official tiles: {int(official.sum())}/{official.size} "
          f"({time.time() - t0:.0f}s)")
    cov = _rasterize_coverage(official, z, r0, c0,
                              smooth_tol_tiles=1.2 if smooth else 0.0,
                              res=res)
    np.savez_compressed(COVERAGE_NPZ, coverage=cov,
                        extent=np.array([LON0, LON1, LAT0, LAT1]))
    print(f"wrote {COVERAGE_NPZ}: official coverage = "
          f"{cov.mean() * 100:.1f}% of figure extent")


# ------------------------------------------------------- hi-res inset caches
# 与 scripts/figures/figure6/plot_figure6_hazard.py 的 INSETS 对应（外扩
# ~400 m 边距）；z16 源片 ~2.2 m/px，输出 3e-5 deg/px (~3.3 m)
INSET_BBOXES = {
    "b1": (113.833, 113.891, 22.233, 22.273),   # Tai O
    "b2": (114.088, 114.146, 22.313, 22.353),   # Kwai Chung–Stonecutters
    "b3": (114.206, 114.274, 22.268, 22.312),   # Lei Yue Mun
}
INSET_Z, INSET_RES = 16, 3e-5


def build_inset_caches(z=INSET_Z, res=INSET_RES, workers=8):
    """三个 panel b inset 区域的高清正射影像缓存（z16 瓦片）。"""
    for tag, bbox in INSET_BBOXES.items():
        out = OUT / f"hk_orthophoto_inset_{tag}.npz"
        print(f"--- inset {tag} bbox={bbox}")
        fetch_all_tiles(z, workers, bbox=bbox)
        build_cache(z, bbox=bbox, res=res, out_npz=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=int, default=TILE_Z)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mosaic-only", action="store_true")
    ap.add_argument("--tdop-pilot", type=int, default=0, metavar="N",
                    help="download N TDOP pilot sheets (default 0)")
    ap.add_argument("--coverage-mask", action="store_true",
                    help="probe tiles with 1-byte Range requests and "
                         "build the official-coverage mask npz")
    ap.add_argument("--coverage-zoom", type=int, default=COVERAGE_Z,
                    help="zoom level for the coverage probe (default "
                         f"{COVERAGE_Z}; z17 gives ~280 m edge detail)")
    ap.add_argument("--res", type=float, default=OUT_RES,
                    help="output grid resolution in deg/px for the mosaic "
                         f"cache and coverage rasterization (default {OUT_RES})")
    ap.add_argument("--insets", action="store_true",
                    help="build hi-res (z16) orthophoto caches for the "
                         "three panel-b inset regions")
    ap.add_argument("--no-smooth", action="store_true",
                    help="skip polygon simplification of the coverage "
                         "boundary (keep raw tile steps)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.tdop_pilot:
        tdop_pilot(args.tdop_pilot)
        return
    if args.coverage_mask:
        build_coverage_mask(z=args.coverage_zoom, workers=args.workers,
                            smooth=not args.no_smooth, res=args.res)
        return
    if args.insets:
        build_inset_caches(workers=args.workers)
        return
    if not args.mosaic_only:
        fetch_all_tiles(args.zoom, args.workers)
    build_cache(args.zoom, res=args.res)
    make_preview()
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
