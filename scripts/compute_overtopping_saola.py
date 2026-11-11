#!/usr/bin/env python
"""EurOtop mean wave-overtopping discharge per HK coastline segment, Saola 2023 case.

Figure 6 (docs/figure6_plan.md, section 4 step 4): maps the WaveFlow 4-variable
ensemble (hs, tp, dm, elev) to per-segment overtopping discharge q [L/s/m] and
exceedance probability P(q > 1 L/s/m).

Formulae and parameter choices (documented per task spec)
---------------------------------------------------------
* EurOtop (2018), Ch. 5, mean overtopping discharge, dimensionless form
  Q = q / sqrt(g * Hs^3):
  - breaking waves  (Eq. 5.10):  Q = (0.023/sqrt(tan_a)) * gamma_b * xi
        * exp( -(2.7 * Rc / (xi * Hs * gamma_b * gamma_f * gamma_beta * gamma_v))**1.3 ),
        xi = tan_a / sqrt(2*pi*Hs / (g*Tm10^2))   (breaker parameter, Tm10 = Tp/1.1)
  - non-breaking    (Eq. 5.11):  Q = 0.09 * exp( -(1.5 * Rc / (Hs * gamma_f * gamma_beta * gamma_star))**1.3 )
        (gamma_v = gamma_star = 1.0: no wall / storm wall on the slope)
  - saturation cap:  Q <= 0.09 (asymptote of Eq. 5.11 at Rc = 0; also guards
        exp overflow); non-binding for Eq. 7.1 (asymptote 0.047).
  - sloping revetments: q = min(breaking, non-breaking); EurOtop (2018) p. 115
        gives Eq. 5.10 "with a maximum of" Eq. 5.11, i.e. the smaller of the two;
        tan_a = 0.5 (1:2 slope).
  NOTE (validity): Eqs. 5.10-5.13 are "roughly valid for breaker parameters
  xi_m-1,0 < 5" (manual p. 118); cells beyond that keep the same formula here
  and this limitation is reported with the results.
  Version note: the previous implementation used the EurOtop (2007) mean-value
  formulae (0.067/0.2, exp(-4.75...)/(-2.6...)) which the 2018 manual reproduces
  in section 4.2.1; those were replaced by the 2018 Eqs. 5.10/5.11 above, and the
  combination rule was corrected from max() to min().
  gamma_b in Eq. 5.10 is the EurOtop *berm* reduction factor, set to 1.0
  (no berm information per segment). It is unrelated to the depth-limited
  breaker index gamma_b = 0.55 used for the nearshore transformation below.
* Vertical seawalls: EurOtop (2018) Ch. 7, Eq. 7.1 (vertical wall, foreshore
  without influence):
        Q = 0.047 * exp( -(2.35 * Rc / (Hs * gamma_f * gamma_beta))**1.3 )
  NOTE: the task brief gave this without the 1.3 power; the published Eq. 7.1
  includes the **1.3 exponent, which is used here (omitting it overpredicts q
  for Rc/Hs >~ 0.4). Reliability sigma(0.047)=0.007.
* Nearshore transformation (simplified, no SWAN): depth-limited breaking,
  Hs_toe = min(Hs_node, 0.55 * h_toe); Tp unchanged.
* Toe depth h_toe: depth of the segment's nearest model ocean node
  (node_idx) taken from the SCHISM grid data/hindcast/raw_new/hgrid.gr3
  (positive down below datum; negative = above datum/land), floored at
  h_toe >= 1.5 m for very shallow / above-datum nodes. node_idx indexes the
  7788 HK ocean nodes in data/hindcast/hk_coordinates.npy; the hgrid node is
  matched by coordinates (cKDTree, exact match), NOT by row number, because
  hgrid node order diverges from hk_coordinates row order beyond row ~6000.
* Still-water freeboard: Rc = Rc_land - elev (model water level, this
  member/ts/node). If Rc < 0 the wall is submerged; the EurOtop formulae are
  not valid, so a weir-like overflow fallback is applied:
        q = max(q, 0.5 * (-Rc) * sqrt(g * max(Hs_toe, 0.1)))   [m3/s/m]
  (broad-crested-weir scaling q ~ C * h * sqrt(g*H); coefficient 0.5 chosen
  ad hoc for a demonstration figure) and the occurrence is counted
  (`n_weir_fallback` per segment).
* Wave obliquity: beta = angular difference between dm and the segment
  seaward normal (wrapped to [0,180], capped at 80 deg);
  gamma_beta = 1 - 0.0033 * beta.  dm is the direction waves come FROM
  (nautical/meteorological convention, as documented in
  modules/datasets/load_hindcast.py and modules/plot/plot_wave_hk.py).
  Verified from the data itself: at Ragasa closest approach (23 Sep 2025
  21Z, typhoon ~100 km south of HK) 9 m waves in the open waters SE of HK
  carried dm ~ 114 deg (arriving from the ESE storm; a going-to reading
  would send them offshore towards the Pacific).  Since normal_deg points
  seaward, beta = |dm - normal| gives beta = 0 for perpendicular onshore
  attack, exactly the EurOtop definition.  NOTE: if dm were a going-to
  (oceanographic) convention instead, the correct expression would be
  beta = |dm - (normal + 180)| — do NOT "fix" the formula above.
* Crest freeboard proxy Rc_land (--crest-source terrarium, legacy): Terrarium
  DEM (AWS elevation-tiles-prod, z=13, ~19 m/px at HK latitude); max terrain
  elevation within a 200 m LANDWARD half-disk around the segment midpoint,
  clamped to [1.5, 12] m, dem_flag records which clamp bound was hit
  (0=none, 1=min, 2=max). KNOWN FLAW: SRTM predates post-2000 reclamation
  (HKIA, West Kowloon, Tung Chung read ~sea level -> 232/894 segments hit
  the 1.5 m floor, crest_flag=1, unreliable).
* Crest freeboard proxy (--crest-source official, DEFAULT): the canonical
  LandsD 5 m DTM download from data.gov.hk ("Digital Terrain Model (DTM)",
  https://data.gov.hk/en-data/dataset/hk-landsd-openmap-5m-grid-dtm ->
  Whole_HK_DTM_5m.zip, ArcInfo ASCII grid on the HK1980 grid, 5 m cells,
  integer-metre heights, stated accuracy +/-5 m, file vintage 2019-03;
  fetched to data/hk_coastline/dtm_landsd_official/Whole_HK_DTM_5m.asc).
  Same 200 m landward half-disk max, same [1.5, 12] m clamp and crest_flag
  semantics. DATUM: LandsD DTM heights are above HK Principal Datum (HKPD),
  HKPD = MSL - 1.23 m, while the model water level `elev` is relative to
  MSL, so Rc = (DTM - 1.23) - elev_model and the clamp is applied to the
  MSL-referenced value. (For reference: HKO tide-gauge mCD = MSL + 1.38 m.)
  The unclamped MSL-referenced value is additionally recorded as
  crest_proxy_raw in the npz/CSV outputs. CAVEATS: sea pixels are 0 (act as
  a low floor in the half-disk max); the DTM retains some non-ground
  objects (elevated roads/bridges/buildings, and vegetation height where
  vegetated, per the official description), which the 12 m clamp bounds;
  heights are rounded to integer metres.
* Crest freeboard proxy (--crest-source landsd): the SAME LandsD 5 m DTM
  product, but the Esri China (Hong Kong) re-published copy
  ("Whole_HK_DTM_5M_tilepkg" ArcGIS ImageServer, level-15 LERC tiles
  ~4.78 m/px, mosaicked by scripts/fetch_landsd_dtm.py to
  data/hk_coastline/dtm_landsd/hk_dtm_l15.npz, Web-Mercator grid). Same
  underlying official data (service description: source is the data.gov.hk
  ArcInfo ASCII grid; copyright "Esri China (Hong Kong) Ltd.,
  DATA.GOV.HK"), but tile-cache resampling smooths the integer heights into
  fractional values and the tile builder void-fills sea pixels (~2 m), so
  the two snapshots differ by ~1 m median per segment (see
  data/hk_coastline/overtopping/dtm_official_vs_esri.txt). Kept as the
  appendix cross-check.
  Official outputs go to *_official files, Esri-copy outputs to *_landsd
  files; Terrarium outputs stay untouched.
* Crest freeboard proxy (--crest-source hybrid): official ASC first, then
  replace crests for the segments where the DTM is known-bad using LandsD
  iB1000 SpotHeight survey points (297,758 pts,
  data/hk_coastline/landsd_official/ib1000_spotheights.csv, per-segment
  50 m landward-band p90 in
  spotheight_crest_persegment.csv, MSL-referenced; see
  spotheight_crest_report.md): dem_flag==1 (new-reclamation voids; 47
  covered segments gain real crests, median ~5.8 m MSL) and dem_flag==2
  (DTM contaminated by buildings/hillsides; 289 covered) get the spot p90
  where n_spot_50m >= 1; dem_flag==0 segments keep the DTM value (spot
  version reserved for an appendix sensitivity). crest_flag is recomputed
  after replacement; provenance is recorded in crest_src
  ('dtm'/'spot') and crest_flag_dtm columns of the summary CSV.
  Final step: segments with a published DESIGN crest level from EIA /
  project documents (design_crest_overrides.csv, e.g. 3RS seawall, Tung
  Chung East eco-shoreline, HKBCF island — post-2019 reclamation where the
  DTM saw only sea) get the design value (crest_src='design'), after which
  crest_flag is recomputed again.
  Last step: any still-unresolved crest_flag==1 segments are imputed —
  median of valid engineered neighbours within 800 m
  (crest_src='imputed_nb'), or, where no valid neighbour exists (HKBCF
  island), the reclamation formation level +5.5 mPD
  (crest_src='imputed_def'); imputed segments are treated as valid with
  provenance kept in crest_src.
  Outputs go to *_hybrid files.

Usage:
    python scripts/compute_overtopping_saola.py [--which m10|m100|both]
        [--crest-source terrarium|landsd|official|hybrid]
"""
import argparse
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG_CSV = os.path.join(ROOT, 'data/hk_coastline/hk_coastline_segments.csv')
HGRID = os.path.join(ROOT, 'data/hindcast/raw_new/hgrid.gr3')
HK_COORDS = os.path.join(ROOT, 'data/hindcast/hk_coordinates.npy')
TILE_DIR = os.path.join(ROOT, 'data/hk_coastline/dem_tiles')
DTM_NPZ = os.path.join(ROOT, 'data/hk_coastline/dtm_landsd/hk_dtm_l15.npz')
OFFICIAL_ASC = os.path.join(
    ROOT, 'data/hk_coastline/dtm_landsd_official/Whole_HK_DTM_5m.asc')
SPOT_CSV = os.path.join(
    ROOT, 'data/hk_coastline/landsd_official/spotheight_crest_persegment.csv')
DESIGN_CSV = os.path.join(
    ROOT, 'data/hk_coastline/overtopping/design_crest_overrides.csv')
# Release override: redirect the sample root / output dir for re-inferred
# ensembles without editing the script.
OUT_DIR = os.environ.get('WF_FIG6_OVERTOPPING_DIR',
                         os.path.join(ROOT, 'data/hk_coastline/overtopping'))
SAMPLES_BASE = os.environ.get(
    'WF_HK_RUN',
    os.path.join(
        ROOT,
        'output/results/hk_cleaned/all_var/f8_blr5e-5_amp_stable_tod/'
        'size_sweep/base_v1_20260821_025917'))

G = 9.81
BREAKER_IDX = 0.55      # depth-limit: Hs_toe <= gamma_b * h_toe
H_TOE_FLOOR = 1.5       # m, floor for very shallow / above-datum nodes
RC_MIN, RC_MAX = 1.5, 12.0   # m, crest-proxy clamp bounds
HKPD_TO_MSL = -1.23          # HKPD = MSL - 1.23 m -> add to convert to MSL
HS_FLOOR = 0.05         # m, guard against division by ~zero Hs
Q_CAP = 0.09            # dimensionless saturation cap (Eq. 5.11 asymptote at Rc=0; overflow guard)
TAN_A_REVETMENT = 0.5   # 1:2 slope
Q_CRIT_LPSM = 1.0       # exceedance threshold for P maps

M10_TS = pd.date_range('2023-08-25 00:00', periods=288, freq='h')
M100_TS = pd.date_range('2023-08-31 12:00', periods=49, freq='h')

# 案例开关：--case ragasa 用 Ragasa 2025 的样本目录和时间窗重跑同一管线，
# 输出文件名以案例名做前缀（saola_* / ragasa_*），两套结果互不覆盖。
# ragasa m100 窗口按 m10 领域 mean P 峰值（20250924T03 UTC）定：
# 2025-09-23T00Z 起 49 h（23T00Z..25T00Z），含最接近时刻 23T22Z。
CASES = {
    'saola': dict(
        m10_dir='case_saola2023', m100_dir='case_saola2023_m100',
        m10_ts=M10_TS, m100_ts=M100_TS,
        closest_approach='2023-09-01 13 UTC',
        # 验证块打印标记窗口（peak 附近）
        verif_day='20230901', verif_hours=(6, 15)),
    'ragasa': dict(
        m10_dir='case_ragasa2025', m100_dir='case_ragasa2025_m100',
        m10_ts=pd.date_range('2025-09-19 00:00', periods=240, freq='h'),
        m100_ts=pd.date_range('2025-09-23 00:00', periods=49, freq='h'),
        closest_approach='2025-09-23 22 UTC (06:00 HKT 24 Sep)',
        verif_day='20250923', verif_hours=(12, 23)),
}

# HK bounding box for DEM tiles (task spec)
DEM_LAT0, DEM_LAT1 = 22.15, 22.58
DEM_LON0, DEM_LON1 = 113.82, 114.45
DEM_Z = 13
DEM_RADIUS_M = 200.0

# Named reference areas for the verification report (name -> lat, lon)
NAMED_AREAS = {
    'Tai O (Lantau west)': (22.254, 113.862),
    'HK Intl Airport / Chek Lap Kok': (22.315, 113.920),
    'Tung Chung': (22.290, 113.940),
    'Lantau south coast': (22.210, 113.920),
    'Cheung Chau': (22.205, 114.025),
    'Tuen Mun / Castle Peak': (22.390, 113.970),
    'Yuen Long / Deep Bay': (22.470, 114.020),
    'Lau Fau Shan': (22.470, 113.990),
    'Ma Wan': (22.350, 114.060),
    'Tsuen Wan': (22.370, 114.110),
    'Kwai Chung / Stonecutters': (22.340, 114.120),
    'West Kowloon / TST': (22.305, 114.165),
    'Kennedy Town (HK Isl west)': (22.282, 114.128),
    'Central / Wan Chai (HK Isl north)': (22.285, 114.165),
    'Lei Yue Mun (harbour east)': (22.285, 114.235),
    'Chai Wan (HK Isl east)': (22.265, 114.240),
    'Aberdeen (HK Isl south)': (22.245, 114.155),
    'Repulse Bay / Stanley (HK Isl south)': (22.230, 114.200),
    'Shek O (HK Isl SE)': (22.225, 114.250),
    'Po Toi Island': (22.170, 114.250),
    'Waglan Island': (22.182, 114.303),
    'Lamma Island': (22.210, 114.110),
    'Tseung Kwan O / Junk Bay': (22.290, 114.270),
    'Clearwater Bay': (22.270, 114.290),
    'Sai Kung': (22.380, 114.270),
    'Sha Tin / Shing Mun River': (22.385, 114.200),
    'Tolo Harbour / Tai Po': (22.450, 114.200),
    'Ma Liu Shui / Tolo entrance': (22.420, 114.220),
    'Tap Mun / Mirs Bay north': (22.470, 114.360),
    'Mirs Bay': (22.520, 114.350),
    'Sha Tau Kok / Starling Inlet': (22.540, 114.220),
}


# --------------------------------------------------------------------------
# DEM: Terrarium tiles
# --------------------------------------------------------------------------
def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def fetch_tile(z, x, y, retries=3):
    path = os.path.join(TILE_DIR, f'z{z}', f'{x}_{y}.png')
    if os.path.exists(path):
        return path
    url = (f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/'
           f'{z}/{x}/{y}.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            tmp = path + '.tmp'
            with open(tmp, 'wb') as f:
                f.write(data)
            os.replace(tmp, path)
            return path
        except Exception as e:
            if k == retries - 1:
                raise RuntimeError(f'failed to fetch {url}: {e}')
            time.sleep(2 * (k + 1))


def load_dem_mosaic():
    """Fetch and assemble the z=13 Terrarium mosaic over the HK box.

    Returns (elev, lat_axis, lon_axis): elev (npy, npx) float32 metres
    (<0 treated as 0 later), lat_axis per row, lon_axis per column.
    """
    from PIL import Image
    x0f, y0f = lonlat_to_tile(DEM_LON0, DEM_LAT1, DEM_Z)  # NW corner
    x1f, y1f = lonlat_to_tile(DEM_LON1, DEM_LAT0, DEM_Z)  # SE corner
    x0, x1 = int(math.floor(x0f)), int(math.floor(x1f))
    y0, y1 = int(math.floor(y0f)), int(math.floor(y1f))
    xs = list(range(x0, x1 + 1))
    ys = list(range(y0, y1 + 1))
    print(f'DEM tiles: z{DEM_Z} x {xs[0]}..{xs[-1]}, y {ys[0]}..{ys[-1]} '
          f'({len(xs) * len(ys)} tiles)')
    jobs = [(DEM_Z, x, y) for x in xs for y in ys]
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(lambda t: fetch_tile(*t), jobs))
    nrow, ncol = len(ys) * 256, len(xs) * 256
    elev = np.full((nrow, ncol), np.nan, dtype=np.float32)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            path = os.path.join(TILE_DIR, f'z{DEM_Z}', f'{x}_{y}.png')
            with Image.open(path) as im:
                rgb = np.asarray(im.convert('RGB'), dtype=np.float32)
            e = rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0
            elev[iy * 256:(iy + 1) * 256, ix * 256:(ix + 1) * 256] = e
    # pixel-centre coordinates (Web Mercator tiling scheme)
    n = 2.0 ** DEM_Z
    cols = (np.arange(ncol) + 0.5) / 256.0 + xs[0]
    rows = (np.arange(nrow) + 0.5) / 256.0 + ys[0]
    lon_axis = cols / n * 360.0 - 180.0
    lat_axis = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * rows / n))))
    return elev, lat_axis.astype(np.float64), lon_axis.astype(np.float64)


def crest_proxies(seg_eng, elev, lat_axis, lon_axis):
    """Max terrain elevation within a 200 m landward half-disk per segment."""
    elev = np.where(np.isfinite(elev), elev, 0.0)
    elev = np.maximum(elev, 0.0)  # sea/voids (Terrarium ~ -82) -> 0
    nrow, ncol = elev.shape
    m_per_deg_lat = 110540.0
    # local pixel size ~ m: use midpoint latitude
    out_rc = np.full(len(seg_eng), np.nan)
    for i, (lat, lon, nrm) in enumerate(zip(
            seg_eng.mid_lat.values, seg_eng.mid_lon.values,
            seg_eng.normal_deg.values)):
        row_f = np.interp(lat, lat_axis[::-1], np.arange(nrow)[::-1])
        col_f = np.interp(lon, lon_axis, np.arange(ncol))
        m_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
        m_per_px_lat = abs(lat_axis[1] - lat_axis[0]) * m_per_deg_lat
        m_per_px_lon = abs(lon_axis[1] - lon_axis[0]) * m_per_deg_lon
        r_px = int(math.ceil(DEM_RADIUS_M / min(m_per_px_lat, m_per_px_lon))) + 1
        r0, r1 = max(0, int(row_f) - r_px), min(nrow, int(row_f) + r_px + 1)
        c0, c1 = max(0, int(col_f) - r_px), min(ncol, int(col_f) + r_px + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        win = elev[r0:r1, c0:c1]
        plat = lat_axis[r0:r1][:, None]
        plon = lon_axis[c0:c1][None, :]
        dn = (plat - lat) * m_per_deg_lat       # north offset [m]
        de = (plon - lon) * m_per_deg_lon       # east offset [m]
        dist = np.hypot(dn, de)
        # landward unit vector = opposite of seaward normal bearing
        th = math.radians(nrm + 180.0)
        land_dot = de * math.sin(th) + dn * math.cos(th)
        mask = (dist <= DEM_RADIUS_M) & (land_dot > 0.0)
        if not mask.any():
            continue
        out_rc[i] = win[mask].max()
    rc = np.clip(np.nan_to_num(out_rc, nan=RC_MIN), RC_MIN, RC_MAX)
    dem_flag = np.where(~np.isfinite(out_rc), 1,
                        np.where(out_rc < RC_MIN, 1,
                                 np.where(out_rc > RC_MAX, 2, 0))).astype(np.int8)
    return rc, dem_flag, out_rc


# --------------------------------------------------------------------------
# DTM: LandsD 5 m (Web-Mercator mosaic, heights above HKPD)
# --------------------------------------------------------------------------
def lonlat_to_merc(lon, lat):
    lon_r, lat_r = math.radians(lon), math.radians(lat)
    return (6378137.0 * lon_r,
            6378137.0 * math.log(math.tan(math.pi / 4.0 + lat_r / 2.0)))


def load_dtm_landsd():
    """Load the L15 LandsD mosaic. Returns (elev, origin_x, origin_y, res):
    elev float32 (npy, npx), NaN outside the service extent (open sea);
    row 0 is the TOP edge (max y)."""
    d = np.load(DTM_NPZ)
    elev = d['elev']
    res = float(d['res'])
    print(f'LandsD DTM mosaic {elev.shape}, res {res:.4f} m/px, '
          f'finite {np.isfinite(elev).mean() * 100:.1f}%')
    return elev, float(d['origin_x']), float(d['origin_y']), res


def crest_proxies_landsd(seg_eng, elev, origin_x, origin_y, res):
    """Max DTM elevation within a 200 m landward half-disk per segment.

    Returns (rc, dem_flag, rc_raw): rc is clamped to [RC_MIN, RC_MAX] on the
    MSL-referenced value (DTM_HKPD - 1.23); rc_raw is the unclamped
    MSL-referenced value; dem_flag as for the Terrarium path."""
    nrow, ncol = elev.shape
    r_px = int(math.ceil(DEM_RADIUS_M / res)) + 1
    # pixel offset grids (constant): dx east, dy north, in metres
    off = np.arange(-r_px, r_px + 1)
    dy_grid = -off[:, None] * res            # row increases southward
    dx_grid = off[None, :] * res
    dist_grid = np.hypot(dx_grid, dy_grid)
    out_raw = np.full(len(seg_eng), np.nan)
    for i, (lat, lon, nrm) in enumerate(zip(
            seg_eng.mid_lat.values, seg_eng.mid_lon.values,
            seg_eng.normal_deg.values)):
        x, y = lonlat_to_merc(lon, lat)
        col_f = (x - origin_x) / res
        row_f = (origin_y - y) / res
        r0, r1 = int(round(row_f)) - r_px, int(round(row_f)) + r_px + 1
        c0, c1 = int(round(col_f)) - r_px, int(round(col_f)) + r_px + 1
        if r1 <= 0 or c1 <= 0 or r0 >= nrow or c0 >= ncol:
            continue
        sr0, sr1 = max(0, r0), min(nrow, r1)
        sc0, sc1 = max(0, c0), min(ncol, c1)
        win = elev[sr0:sr1, sc0:sc1]
        sub = np.s_[sr0 - r0:sr1 - r0, sc0 - c0:c1 - c0]
        dist = dist_grid[sub]
        dx = dx_grid[sub]
        dy = dy_grid[sub]
        th = math.radians(nrm + 180.0)       # landward unit vector
        land_dot = dx * math.sin(th) + dy * math.cos(th)
        mask = (dist <= DEM_RADIUS_M) & (land_dot > 0.0) & np.isfinite(win)
        if not mask.any():
            continue
        out_raw[i] = win[mask].max()
    out_msl = out_raw + HKPD_TO_MSL          # HKPD -> MSL datum shift
    rc = np.clip(np.nan_to_num(out_msl, nan=RC_MIN), RC_MIN, RC_MAX)
    dem_flag = np.where(~np.isfinite(out_msl), 1,
                        np.where(out_msl < RC_MIN, 1,
                                 np.where(out_msl > RC_MAX, 2, 0))).astype(np.int8)
    return rc, dem_flag, out_msl


# --------------------------------------------------------------------------
# DTM: official LandsD 5 m ASC download (HK1980 grid, heights above HKPD)
# --------------------------------------------------------------------------
def load_dtm_official():
    """Load the canonical data.gov.hk ASC. Returns (elev, xll, yll, cs,
    nrow, ncol): elev float32 with NaN for NODATA (-9999); sea pixels are 0;
    row 0 is the TOP edge (max y)."""
    with open(OFFICIAL_ASC) as f:
        hdr = {}
        for _ in range(6):
            k, v = f.readline().split()
            hdr[k.lower()] = float(v)
    ncols, nrows = int(hdr['ncols']), int(hdr['nrows'])
    xll, yll, cs = hdr['xllcorner'], hdr['yllcorner'], hdr['cellsize']
    elev = np.loadtxt(OFFICIAL_ASC, skiprows=6, dtype=np.float32)
    assert elev.shape == (nrows, ncols), elev.shape
    elev = np.where(elev == -9999.0, np.nan, elev)
    print(f'official LandsD DTM {elev.shape}, cellsize {cs} m, '
          f'finite {np.isfinite(elev).mean() * 100:.1f}%')
    return elev, xll, yll, cs, nrows, ncols


def crest_proxies_official(seg_eng, elev, xll, yll, cs, nrow, ncol):
    """Max DTM elevation within a 200 m landward half-disk per segment,
    on the official HK1980-grid ASC. Same semantics as
    crest_proxies_landsd; returns (rc, dem_flag, rc_raw) with MSL-referenced
    values (HKPD - 1.23)."""
    from pyproj import Transformer
    tf = Transformer.from_crs('EPSG:4326', 'EPSG:2326', always_xy=True)
    r_px = int(math.ceil(DEM_RADIUS_M / cs)) + 1
    off = np.arange(-r_px, r_px + 1)
    dy_grid = -off[:, None] * cs            # row increases southward
    dx_grid = off[None, :] * cs
    dist_grid = np.hypot(dx_grid, dy_grid)
    out_raw = np.full(len(seg_eng), np.nan)
    sx, sy = tf.transform(seg_eng.mid_lon.values, seg_eng.mid_lat.values)
    for i, (x, y, nrm) in enumerate(zip(sx, sy, seg_eng.normal_deg.values)):
        col_f = (x - xll) / cs - 0.5
        row_f = (yll + nrow * cs - y) / cs - 0.5   # row 0 = top (max y)
        r0, r1 = int(round(row_f)) - r_px, int(round(row_f)) + r_px + 1
        c0, c1 = int(round(col_f)) - r_px, int(round(col_f)) + r_px + 1
        if r1 <= 0 or c1 <= 0 or r0 >= nrow or c0 >= ncol:
            continue
        sr0, sr1 = max(0, r0), min(nrow, r1)
        sc0, sc1 = max(0, c0), min(ncol, c1)
        win = elev[sr0:sr1, sc0:sc1]
        sub = np.s_[sr0 - r0:sr1 - r0, sc0 - c0:c1 - c0]
        dist = dist_grid[sub]
        dx = dx_grid[sub]
        dy = dy_grid[sub]
        th = math.radians(nrm + 180.0)       # landward unit vector
        land_dot = dx * math.sin(th) + dy * math.cos(th)
        mask = (dist <= DEM_RADIUS_M) & (land_dot > 0.0) & np.isfinite(win)
        if not mask.any():
            continue
        out_raw[i] = win[mask].max()
    out_msl = out_raw + HKPD_TO_MSL          # HKPD -> MSL datum shift
    rc = np.clip(np.nan_to_num(out_msl, nan=RC_MIN), RC_MIN, RC_MAX)
    dem_flag = np.where(~np.isfinite(out_msl), 1,
                        np.where(out_msl < RC_MIN, 1,
                                 np.where(out_msl > RC_MAX, 2, 0))).astype(np.int8)
    return rc, dem_flag, out_msl


# Coast-type parameters, same mapping as CLASS_STYLE in
# build_hk_coastline_segments.py: (gamma_f, eurotop_applicable)
TYPE_PARAMS = {'seawall_vertical': (1.0, True),
               'revetment_slope': (0.8, True),
               'natural_rocky': (1.0, False),
               'natural_sandy': (1.0, False),
               'lowlying_open': (1.0, False)}


def apply_type_overrides(seg, path):
    """Apply imagery-reviewed coast-type corrections (CSV: seg_id,new_type).

    Updates coast_type / gamma_f / eurotop_applicable on the full segment
    table before the engineering filter, so natural->engineered promotions
    (e.g. an OSM-untagged promenade found in the orthophoto review) enter
    the q computation and engineered->natural demotions leave it."""
    d = pd.read_csv(path)
    bad = set(d.new_type) - set(TYPE_PARAMS)
    if bad:
        raise SystemExit(f'{path}: unknown coast_type(s) {sorted(bad)}; '
                         f'allowed: {sorted(TYPE_PARAMS)}')
    seg = seg.set_index('seg_id')
    n_change = 0
    for _, r in d.iterrows():
        sid = int(r.seg_id)
        if sid not in seg.index:
            print(f'  WARNING: override seg {sid} not in segment table')
            continue
        if seg.loc[sid, 'coast_type'] == r.new_type:
            continue
        gf, applic = TYPE_PARAMS[r.new_type]
        seg.loc[sid, ['coast_type', 'gamma_f', 'eurotop_applicable']] = \
            [r.new_type, gf, applic]
        n_change += 1
    print(f'type overrides from {path}: {n_change} segments reclassified '
          f'(of {len(d)} listed)')
    return seg.reset_index()


def apply_spot_crests(seg_eng, rc, dem_flag, rc_raw, include_flag0=False):
    """Hybrid crest: replace known-bad DTM crests (dem_flag 1/2) with
    iB1000 SpotHeight 50 m-band p90 where covered. Returns
    (rc2, dem_flag2, rc_raw2, crest_src) with crest_src in {'dtm','spot'}.

    include_flag0=True additionally replaces dem_flag==0 segments where
    coverage is good (n_spot>=2, p90 within [RC_MIN, 15] m MSL), per the
    recommendation in spotheight_crest_report.md §6: the DTM 200 m half-disk
    max is biased ~2.5-3 m high even in its 'trusted' class, so the
    spot-replacement run is the less-conservative sensitivity bound."""
    spot = pd.read_csv(SPOT_CSV).set_index('seg_id')
    sp = spot.reindex(seg_eng.seg_id.values)
    p90 = sp['spot_p90_msl'].values.astype(np.float64)
    n_spot = sp['n_spot_50m'].fillna(0).values
    repl = np.isin(dem_flag, (1, 2)) & (n_spot >= 1) & np.isfinite(p90)
    n_repl12 = int(repl.sum())
    if include_flag0:
        repl0 = ((dem_flag == 0) & (n_spot >= 2) & np.isfinite(p90)
                 & (p90 >= RC_MIN) & (p90 <= 15.0))
        repl = repl | repl0
        print(f'spot replacement flag0 (sensitivity): +{int(repl0.sum())} '
              f'segments (median shift '
              f'{np.nanmedian(p90[repl0] - rc_raw[repl0]):.2f} m)')
    rc_raw2 = np.where(repl, p90, rc_raw)
    rc2 = np.clip(np.nan_to_num(rc_raw2, nan=RC_MIN), RC_MIN, RC_MAX)
    dem_flag2 = np.where(~np.isfinite(rc_raw2), 1,
                         np.where(rc_raw2 < RC_MIN, 1,
                                  np.where(rc_raw2 > RC_MAX, 2,
                                           0))).astype(np.int8)
    crest_src = np.where(repl, 'spot', 'dtm')
    n1 = int(((dem_flag == 1) & repl).sum())
    n2 = int(((dem_flag == 2) & repl).sum())
    print(f'spot replacement: {int(repl.sum())} segments '
          f'(flag0 {int(((dem_flag == 0) & repl).sum())}, '
          f'flag1 {n1}/{int((dem_flag == 1).sum())}, '
          f'flag2 {n2}/{int((dem_flag == 2).sum())}); '
          f'new flag counts: '
          f'{dict(zip(*np.unique(dem_flag2, return_counts=True)))}')
    return rc2, dem_flag2, rc_raw2, crest_src


def apply_design_crests(seg_eng, rc, dem_flag, rc_raw, crest_src):
    """Design-crest override（hybrid 最后一步）：对有公开设计堤顶高程的段
    （EIA/项目文件，如 3RS 海堤 +6.5/+7.0 mPD，见
    data/hk_coastline/overtopping/design_crest_overrides.csv 及
    data/hk_coastline/design_docs/ 原文档），用设计值替换钳位代理，
    crest_src='design'。文件不存在则跳过。"""
    if not os.path.exists(DESIGN_CSV):
        print('no design_crest_overrides.csv; skip design override')
        return rc, dem_flag, rc_raw, crest_src
    d = pd.read_csv(DESIGN_CSV).set_index('seg_id').reindex(
        seg_eng.seg_id.values)
    dv = d['crest_msl'].values.astype(np.float64)
    repl = np.isfinite(dv)
    rc_raw2 = np.where(repl, dv, rc_raw)
    rc2 = np.clip(np.nan_to_num(rc_raw2, nan=RC_MIN), RC_MIN, RC_MAX)
    dem_flag2 = np.where(~np.isfinite(rc_raw2), 1,
                         np.where(rc_raw2 < RC_MIN, 1,
                                  np.where(rc_raw2 > RC_MAX, 2,
                                           0))).astype(np.int8)
    crest_src2 = np.where(repl, 'design', crest_src)
    print(f'design override: {int(repl.sum())} segments '
          f'({dict(zip(*np.unique(d.loc[repl, "project"], return_counts=True)))}); '
          f'new flag counts: '
          f'{dict(zip(*np.unique(dem_flag2, return_counts=True)))}')
    return rc2, dem_flag2, rc_raw2, crest_src2


# Default crest for segments with no DTM/spot/design value and no valid
# neighbour: +5.5 mPD reclamation formation level (same basis as the Tung
# Chung East design crest), MSL-referenced.
IMPUTE_DEFAULT_MSL = 5.5 + HKPD_TO_MSL
IMPUTE_NB_RADIUS_M = 800.0
IMPUTE_NB_MIN = 2


def impute_missing_crests(seg_eng, rc, dem_flag, rc_raw, crest_src):
    """Final hybrid step: impute the remaining crest_flag==1 segments.

    Rule 1 (crest_src='imputed_nb'): median clamped crest of the >=
    IMPUTE_NB_MIN nearest engineered segments with a valid crest within
    IMPUTE_NB_RADIUS_M (same structure usually shares one cross-section).
    Rule 2 (crest_src='imputed_def'): segments with no valid neighbour —
    essentially the HKBCF island, ringed entirely by flag-1 segments — get
    the reclamation formation level +5.5 mPD (IMPUTE_DEFAULT_MSL), a
    conservative lower bound for the actual seawall coping.
    Imputed segments are treated as valid downstream (crest_flag cleared to
    0); provenance stays in crest_src."""
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    miss = dem_flag == 1
    if not miss.any():
        return rc, dem_flag, rc_raw, crest_src
    tf = Transformer.from_crs('EPSG:4326', 'EPSG:2326', always_xy=True)
    xs, ys = tf.transform(seg_eng.mid_lon.values, seg_eng.mid_lat.values)
    valid = ~miss
    tree = cKDTree(np.column_stack([xs[valid], ys[valid]]))
    rc2, raw2 = rc.copy(), rc_raw.copy()
    flag2 = dem_flag.copy()
    src2 = crest_src.astype(object).copy()
    n_nb, n_def = 0, 0
    for i in np.where(miss)[0]:
        idx = tree.query_ball_point([xs[i], ys[i]], r=IMPUTE_NB_RADIUS_M)
        if len(idx) >= IMPUTE_NB_MIN:
            val = float(np.median(rc[valid][idx]))
            src2[i] = 'imputed_nb'
            n_nb += 1
        else:
            val = IMPUTE_DEFAULT_MSL
            src2[i] = 'imputed_def'
            n_def += 1
        rc2[i], raw2[i], flag2[i] = val, val, 0
    print(f'imputation: {int(miss.sum())} flag-1 segments -> '
          f'{n_nb} neighbour-median (r<{IMPUTE_NB_RADIUS_M:.0f} m), '
          f'{n_def} default {IMPUTE_DEFAULT_MSL:.2f} m MSL; '
          f'new flag counts: '
          f'{dict(zip(*np.unique(flag2, return_counts=True)))}')
    return rc2, flag2, raw2, src2


# --------------------------------------------------------------------------
# Toe depth from SCHISM grid
# --------------------------------------------------------------------------
def load_node_depths():
    """Depth (m below datum, positive down) for each of the 7788 HK nodes."""
    from scipy.spatial import cKDTree
    with open(HGRID) as f:
        f.readline()
        ne, nn = map(int, f.readline().split())
        nodes = np.loadtxt(f, max_rows=nn)
    assert nodes.shape == (nn, 4), nodes.shape
    lon, lat, dep = nodes[:, 1], nodes[:, 2], nodes[:, 3]
    print(f'hgrid: ne={ne} nn={nn}, depth range [{dep.min():.1f}, {dep.max():.1f}] m')
    hk = np.load(HK_COORDS)  # (7788, 2) lat, lon
    assert hk.shape[0] == 7788
    dist, idx = cKDTree(np.c_[lon, lat]).query(np.c_[hk[:, 1], hk[:, 0]])
    assert dist.max() < 1e-8, f'hgrid<->hk_coordinates mismatch {dist.max()}'
    return dep[idx]


# --------------------------------------------------------------------------
# Wave ensemble loading
# --------------------------------------------------------------------------
def load_samples(samples_dir, ts_list, n_members, node_rows):
    """Load (n_ts, n_members, n_nodes, 4) array of hs, tp, dm, elev."""
    n_ts, n_nodes = len(ts_list), len(node_rows)
    out = np.full((n_ts, n_members, n_nodes, 4), np.nan, dtype=np.float32)

    def one(job):
        i, j, path = job
        with np.load(path) as d:
            return i, j, d['sample'][node_rows, :]

    jobs = []
    missing = 0
    for i, ts in enumerate(ts_list):
        tss = ts.strftime('%Y%m%dT%H')
        for j in range(n_members):
            path = os.path.join(samples_dir, f'sample_{tss}_ens{j:03d}.npz')
            if os.path.exists(path):
                jobs.append((i, j, path))
            else:
                missing += 1
    if missing:
        print(f'  WARNING: {missing} sample files missing in {samples_dir}')
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, j, arr in ex.map(one, jobs, chunksize=8):
            out[i, j] = arr
    print(f'  loaded {len(jobs)} sample files in {time.time() - t0:.1f}s')
    return out


# --------------------------------------------------------------------------
# EurOtop computation (vectorized over segment x ts x member)
# --------------------------------------------------------------------------
def compute_q(waves, seg_eng, h_toe, rc_land):
    """waves: (T, M, S, 4); returns q_lpsm (S, T, M), n_weir (S,)."""
    hs = waves[:, :, :, 0].transpose(2, 0, 1)   # (S,T,M)
    tp = waves[:, :, :, 1].transpose(2, 0, 1)
    dm = waves[:, :, :, 2].transpose(2, 0, 1)
    el = waves[:, :, :, 3].transpose(2, 0, 1)

    S = len(seg_eng)
    hs_toe = np.minimum(hs, BREAKER_IDX * h_toe[:, None, None])
    hs_safe = np.maximum(hs_toe, HS_FLOOR)
    tm10 = np.maximum(tp / 1.1, 1.0)  # floor guards calm generated members (Tp=0)
    rc = rc_land[:, None, None] - el
    gf = seg_eng.gamma_f.values[:, None, None]
    normal = seg_eng.normal_deg.values[:, None, None]

    beta = np.abs((dm - normal + 180.0) % 360.0 - 180.0)  # [0,180]
    beta = np.minimum(beta, 80.0)
    g_beta = 1.0 - 0.0033 * beta

    norm = np.sqrt(G * hs_safe ** 3)          # sqrt(g Hs^3)
    Q = np.zeros_like(rc)

    is_rev = (seg_eng.coast_type == 'revetment_slope').values[:, None, None]
    is_vert = (seg_eng.coast_type == 'seawall_vertical').values[:, None, None]

    # breaking (EurOtop 2018 Eq. 5.10), berm factor gamma_b = 1; base clipped
    # at 0 so Rc<0 does not yield NaN (weir fallback below handles Rc<0)
    xi = TAN_A_REVETMENT / np.sqrt(2.0 * np.pi * hs_safe / (G * tm10 ** 2))
    Q_br = ((0.023 / math.sqrt(TAN_A_REVETMENT)) * xi
            * np.exp(-np.power(np.maximum(
                2.7 * rc / (xi * hs_safe * gf * g_beta), 0.0), 1.3)))
    # non-breaking (Eq. 5.11; gamma_star = 1, no storm wall)
    Q_nb = 0.09 * np.exp(-np.power(np.maximum(
        1.5 * rc / (hs_safe * gf * g_beta), 0.0), 1.3))
    # vertical wall (Eq. 7.1, note the 1.3 power); base clipped at 0: for
    # Rc<0 the fractional power would yield NaN (weir fallback handles it)
    Q_vt = 0.047 * np.exp(
        -np.power(np.maximum(2.35 * rc / (hs_safe * gf * g_beta), 0.0), 1.3))

    Q = np.where(is_rev, np.minimum(Q_br, Q_nb), Q)
    Q = np.where(is_vert, Q_vt, Q)
    Q = np.minimum(Q, Q_CAP)                   # saturation cap (Eq. 5.11 asymptote)
    q = Q * norm                               # m3/s/m

    # submerged-wall weir fallback for Rc < 0
    weir = rc < 0.0
    if weir.any():
        q_weir = 0.5 * (-rc) * np.sqrt(G * np.maximum(hs_toe, 0.1))
        q = np.where(weir, np.maximum(q, q_weir), q)
    n_weir = weir.sum(axis=(1, 2))

    assert not (is_rev & is_vert).any()
    covered = (is_rev | is_vert).all()
    assert covered, 'unexpected coast_type among engineered segments'
    q_lpsm = (q * 1000.0).astype(np.float32)
    return q_lpsm, n_weir.astype(np.int32)


def nearest_area(lat, lon):
    best, best_d = None, np.inf
    for name, (la, lo) in NAMED_AREAS.items():
        d = (la - lat) ** 2 + ((lo - lon) * math.cos(math.radians(lat))) ** 2
        if d < best_d:
            best, best_d = name, d
    return best


# --------------------------------------------------------------------------
def run_ensemble(tag, samples_dir, ts_list, n_members, seg_eng, h_toe, rc_land,
                 waves=None, rc_raw=None, case='saola'):
    print(f'--- ensemble {tag}: {samples_dir} ({len(ts_list)} ts x {n_members} members)')
    node_idx = seg_eng.node_idx.values
    if waves is None:
        waves = load_samples(samples_dir, ts_list, n_members, node_idx)
    q_lpsm, n_weir = compute_q(waves, seg_eng, h_toe, rc_land)       # (S,T,M)
    valid = np.isfinite(q_lpsm)
    # NaN members (missing/corrupt sample files) are excluded from statistics
    n_valid = valid.sum(axis=2)
    p_gt1 = np.where(n_valid > 0,
                     ((q_lpsm > Q_CRIT_LPSM) & valid).sum(axis=2)
                     / np.maximum(n_valid, 1),
                     np.nan).astype(np.float32)                      # (S,T)
    q_mean = np.nanmean(q_lpsm, axis=2).astype(np.float32)
    q_p90 = np.nanquantile(q_lpsm, 0.9, axis=2).astype(np.float32)
    ts_str = np.array([t.strftime('%Y%m%dT%H') for t in ts_list])
    out_npz = os.path.join(OUT_DIR, f'{case}_q_{tag}.npz')
    np.savez_compressed(
        out_npz, q_lpsm=q_lpsm, seg_ids=seg_eng.seg_id.values,
        timestamps=ts_str, p_gt1_lpsm=p_gt1, q_mean_lpsm=q_mean,
        q_p90_lpsm=q_p90, n_weir_fallback=n_weir,
        h_toe=h_toe.astype(np.float32), crest_proxy=rc_land.astype(np.float32),
        crest_proxy_raw=(rc_raw.astype(np.float32)
                         if rc_raw is not None else np.full_like(
                             rc_land, np.nan, dtype=np.float32)))
    print(f'  wrote {out_npz}')
    return q_lpsm, p_gt1, q_mean, ts_str, waves, n_weir


def shuffled_waves(samples_dir, ts_list, n_members, node_idx, col_idx, seed):
    """Load samples and permute one column across members per (ts, segment).

    The marginal distribution of the shuffled column is preserved exactly at
    every (ts, segment); only its pairing with the other variables is broken.
    """
    waves = load_samples(samples_dir, ts_list, n_members, node_idx)
    before = np.sort(waves[:, :, :, col_idx], axis=1)  # per (ts, seg) marginal
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random(waves.shape[:3]), axis=1)  # (T,M,S)
    waves[:, :, :, col_idx] = np.take_along_axis(
        waves[:, :, :, col_idx], perm, axis=1)
    # sanity: the permutation preserves the marginal at every (ts, segment)
    assert np.array_equal(before, np.sort(waves[:, :, :, col_idx], axis=1))
    return waves


def shuffle_tp_control(seg_eng, h_toe, rc_land, dem_flag):
    """Fig 6 panel d control: destroy the hs-tp joint dependence.

    The tp column is independently permuted across the 10 members at each
    (timestamp, segment) with a fixed-seed RNG (seed 42); hs/dm/elev stay
    attached to their original members. Both marginal distributions are
    preserved exactly at every (ts, segment); only the hs-tp pairing is
    destroyed. Writes saola_q_m10_shuffletp.npz (same layout as m10) and
    prints a joint-vs-shuffled comparison.
    """
    tag = 'm10_shuffletp'
    samples_dir = os.path.join(SAMPLES_BASE, 'case_saola2023/test/samples')
    print(f'--- control {tag}: independent tp permutation across members, seed 42')
    waves = shuffled_waves(samples_dir, M10_TS, 10, seg_eng.node_idx.values,
                           col_idx=1, seed=42)
    q_lpsm, p_gt1, q_mean, ts_str, _, n_weir = run_ensemble(
        tag, samples_dir, M10_TS, 10, seg_eng, h_toe, rc_land, waves=waves)

    # ---- comparison joint (recomputed from the saved m10 npz) vs shuffled ----
    d0 = np.load(os.path.join(OUT_DIR, 'saola_q_m10.npz'), allow_pickle=True)
    p_joint = d0['p_gt1_lpsm']
    assert (d0['seg_ids'] == seg_eng.seg_id.values).all()
    assert list(d0['timestamps']) == list(ts_str)
    keep = dem_flag != 1  # exclude unreliable crest-proxy segments
    sum_joint = np.nansum(p_joint[keep], axis=0)
    sum_shuf = np.nansum(p_gt1[keep], axis=0)
    top3 = np.argsort(sum_joint)[-3:][::-1]
    print('\nterritory-wide sum of P(q>1) over engineered segments '
          f'(crest_flag!=1, n={int(keep.sum())}), top-3 timestamps by joint P:')
    print(f'  {"timestamp":12s} {"joint":>8s} {"shuffled":>9s} {"shuf-joint":>10s}')
    for i in top3:
        print(f'  {ts_str[i]:12s} {sum_joint[i]:8.2f} {sum_shuf[i]:9.2f} '
              f'{sum_shuf[i] - sum_joint[i]:+10.2f}')
    maxp_j = np.nanmax(p_joint, axis=1)
    maxp_s = np.nanmax(p_gt1, axis=1)
    delta = maxp_s - maxp_j
    order = np.argsort(-np.abs(delta))[:10]
    print('\ntop 10 segments by |delta max P(q>1)| (shuffled - joint):')
    for i in order:
        r = seg_eng.iloc[i]
        area = nearest_area(r.mid_lat, r.mid_lon)
        print(f'  seg {int(r.seg_id):5d}  joint={maxp_j[i]:.2f}  '
              f'shuffled={maxp_s[i]:.2f}  delta={delta[i]:+.2f}  '
              f'{r.coast_type:16s} {area:38s} '
              f'({r.mid_lat:.3f},{r.mid_lon:.3f}) cf={dem_flag[i]}')
    print(f'overall: mean delta maxP = {np.nanmean(delta):+.4f}, '
          f'segments with |delta|>0.1: {int((np.abs(delta) > 0.1).sum())}')
    return q_lpsm


def m100_controls(seg_eng, h_toe, rc_land, dem_flag, do_tp, do_elev):
    """M=100 peak-window controls (finer P resolution, 0.01 steps).

    --shuffle-tp   (col 1, seed 42): destroys the hs-tp joint dependence.
    --shuffle-elev (col 3, seed 43): destroys the storm-driven hs-elev coupling.
    Baseline joint statistics are read from the existing saola_q_m100.npz
    (recomputed by main() first if absent).
    """
    samples_dir = os.path.join(SAMPLES_BASE, 'case_saola2023_m100/test/samples')
    variants = {}
    if do_tp:
        waves = shuffled_waves(samples_dir, M100_TS, 100,
                               seg_eng.node_idx.values, col_idx=1, seed=42)
        variants['tp-shuffled'] = run_ensemble(
            'm100_shuffletp', samples_dir, M100_TS, 100, seg_eng, h_toe,
            rc_land, waves=waves)
    if do_elev:
        waves = shuffled_waves(samples_dir, M100_TS, 100,
                               seg_eng.node_idx.values, col_idx=3, seed=43)
        variants['elev-shuffled'] = run_ensemble(
            'm100_shuffletelev', samples_dir, M100_TS, 100, seg_eng, h_toe,
            rc_land, waves=waves)
    if not variants:
        return

    d0 = np.load(os.path.join(OUT_DIR, 'saola_q_m100.npz'), allow_pickle=True)
    p_joint, qm_joint = d0['p_gt1_lpsm'], d0['q_mean_lpsm']
    ts_str = d0['timestamps']
    assert (d0['seg_ids'] == seg_eng.seg_id.values).all()
    for name, (_, p, qm, ts, _, _) in variants.items():
        assert list(ts) == list(ts_str)
    keep = dem_flag != 1
    n_keep = int(keep.sum())

    sum_joint = np.nansum(p_joint[keep], axis=0)
    top5 = np.argsort(sum_joint)[-5:][::-1]
    print('\n================ M=100 CONTROLS (peak window) ================')
    print(f'engineered segments with crest_flag!=1: n={n_keep}')
    print('\nterritory-wide sum of P(q>1), top-5 timestamps by joint P:')
    hdr = f'  {"timestamp":12s} {"joint":>8s}' + ''.join(
        f' {name:>14s} {"delta":>8s}' for name in variants)
    print(hdr)
    sums = {name: np.nansum(res[1][keep], axis=0)
            for name, res in variants.items()}
    for i in top5:
        line = f'  {ts_str[i]:12s} {sum_joint[i]:8.2f}'
        for name in variants:
            line += f' {sums[name][i]:14.2f} {sums[name][i] - sum_joint[i]:+8.2f}'
        print(line)

    maxp_j = np.nanmax(np.where(keep[:, None], p_joint, np.nan), axis=1)
    deltas = {}
    for name, res in variants.items():
        maxp_v = np.nanmax(np.where(keep[:, None], res[1], np.nan), axis=1)
        d = maxp_v - maxp_j
        deltas[name] = d
        kd = d[np.isfinite(d)]
        print(f'\nper-segment delta max P(q>1), {name} minus joint '
              f'(crest_flag!=1):')
        print(f'  |delta|>=0.05: {int((np.abs(kd) >= 0.05).sum())} segments; '
              f'|delta|>=0.2: {int((np.abs(kd) >= 0.2).sum())}; '
              f'mean={kd.mean():+.4f}; min={kd.min():+.2f}; max={kd.max():+.2f}')

    if 'elev-shuffled' in deltas:
        d = deltas['elev-shuffled']
        p_elev = variants['elev-shuffled'][1]
        order = np.argsort(-np.abs(np.nan_to_num(d)))[:10]
        print('\ntop 10 segments by |delta max P(q>1)|, elev-shuffled - joint:')
        for i in order:
            r = seg_eng.iloc[i]
            area = nearest_area(r.mid_lat, r.mid_lon)
            print(f'  seg {int(r.seg_id):5d}  joint={maxp_j[i]:.2f}  '
                  f'shuffled={np.nanmax(p_elev[i]):.2f}  '
                  f'delta={d[i]:+.2f}  {r.coast_type:16s} {area:36s} '
                  f'({r.mid_lat:.3f},{r.mid_lon:.3f}) cf={dem_flag[i]}')

    # ensemble-mean q at exceedance-heavy cells (joint mean q in top 1%)
    qm_keep = np.where(keep[:, None], qm_joint, np.nan)
    thr = np.nanpercentile(qm_keep, 99.0)
    hot = qm_keep >= thr
    print(f'\nensemble-mean q at exceedance-heavy cells '
          f'(joint mean q >= P99 = {thr:.1f} L/s/m; {int(hot.sum())} cells):')
    line_j = qm_joint[hot]
    print(f'  {"joint":16s} mean={np.nanmean(line_j):9.2f}  '
          f'max={np.nanmax(line_j):9.2f}')
    for name, res in variants.items():
        v = res[2][hot]
        print(f'  {name:16s} mean={np.nanmean(v):9.2f}  '
              f'max={np.nanmax(v):9.2f}  (delta mean '
              f'{np.nanmean(v) - np.nanmean(line_j):+.2f})')


QCRIT_SWEEP = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]  # L/s/m


def qcrit_sweep():
    """Pure re-analysis of the three M=100 npz files (no new sampling).

    For each exceedance threshold qcrit, compares joint / tp-shuffled /
    elev-shuffled on: segments with event-max P(q>qcrit) >= 0.5, and
    territory-wide sum of P at the peak timestamp 20230901T04. Also compares
    event-max ensemble mean/P90 q (joint vs elev-shuffled) as ratios.
    Engineered segments with crest_flag==1 are excluded (unreliable crest).
    """
    names = {'joint': 'saola_q_m100.npz',
             'tp-shuffled': 'saola_q_m100_shuffletp.npz',
             'elev-shuffled': 'saola_q_m100_shuffletelev.npz'}
    data = {}
    for name, fn in names.items():
        path = os.path.join(OUT_DIR, fn)
        if not os.path.exists(path):
            raise SystemExit(f'missing {path} — run the corresponding mode first')
        data[name] = np.load(path, allow_pickle=True)
    seg_ids = data['joint']['seg_ids']
    ts_str = list(data['joint']['timestamps'])
    for name in names:
        assert (data[name]['seg_ids'] == seg_ids).all()
        assert list(data[name]['timestamps']) == ts_str
    i_peak = ts_str.index('20230901T04')

    summ = pd.read_csv(os.path.join(OUT_DIR, 'saola_segment_summary.csv'))
    summ = summ.set_index('seg_id').loc[seg_ids]  # align to npz order
    keep = summ.crest_flag.values != 1
    n_keep = int(keep.sum())
    print('================ QCRIT SWEEP (M=100, pure re-analysis) ================')
    print(f'segments: {len(seg_ids)} engineered, {n_keep} with crest_flag!=1; '
          f'peak ts = {ts_str[i_peak]}')

    # ---- qcrit-independent high-quantile ratios (joint vs elev-shuffled) ----
    mq_j = np.nanmax(data['joint']['q_mean_lpsm'], axis=1)
    p90_j = np.nanmax(data['joint']['q_p90_lpsm'], axis=1)
    mq_e = np.nanmax(data['elev-shuffled']['q_mean_lpsm'], axis=1)
    p90_e = np.nanmax(data['elev-shuffled']['q_p90_lpsm'], axis=1)
    sel = keep & (mq_j > 1.0)
    r_mq = mq_e[sel] / mq_j[sel]
    r_p90 = p90_e[sel] / p90_j[sel]
    ratio_stats = {
        'median_ratio_meanq': float(np.median(r_mq)),
        'p90_ratio_meanq': float(np.percentile(r_mq, 90)),
        'median_ratio_p90q': float(np.median(r_p90)),
        'p90_ratio_p90q': float(np.percentile(r_p90, 90)),
    }
    print(f'\nevent-max q ratios (elev-shuffled / joint), segments with joint '
          f'max mean-q > 1 L/s/m (n={int(sel.sum())}):')
    print(f'  mean-q: median ratio={ratio_stats["median_ratio_meanq"]:.4f}, '
          f'P90 ratio={ratio_stats["p90_ratio_meanq"]:.4f}')
    print(f'  P90-q : median ratio={ratio_stats["median_ratio_p90q"]:.4f}, '
          f'P90 ratio={ratio_stats["p90_ratio_p90q"]:.4f}')

    # ---- threshold sweep ----
    rows = []
    print(f'\n{"qcrit":>7s} | ' + ' | '.join(
        f'{name:>24s}' for name in names))
    print(f'{"(L/s/m)":>7s} | ' + ' | '.join(
        f'{"nSeg":>6s} {"sumP@peak":>10s}     ' for _ in names))
    for qc in QCRIT_SWEEP:
        line = f'{qc:>7.1f} | '
        for name, d in data.items():
            q = d['q_lpsm']
            valid = np.isfinite(q)
            n_valid = valid.sum(axis=2)
            p = np.where(n_valid > 0,
                         ((q > qc) & valid).sum(axis=2) / np.maximum(n_valid, 1),
                         np.nan)
            maxp = np.nanmax(p, axis=1)
            n_seg = int(((maxp >= 0.5) & keep).sum())
            sum_peak = float(np.nansum(p[keep, i_peak]))
            rows.append({'qcrit': qc, 'variant': name,
                         'n_seg_maxP_ge_50': n_seg,
                         'sumP_at_peak_ts': round(sum_peak, 3),
                         **{k: round(v, 4) for k, v in ratio_stats.items()}})
            line += f'{n_seg:>6d} {sum_peak:>10.2f}      | '
        print(line.rstrip(' |'))
    out_csv = os.path.join(OUT_DIR, 'qcrit_sweep.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'\nwrote {out_csv}')

    # ---- divergence verdict ----
    print('\ndivergence check (criterion: >5% relative difference in segment '
          'counts vs joint, or |median ratio - 1| > 0.05):')
    any_div = abs(ratio_stats['median_ratio_meanq'] - 1) > 0.05 or \
        abs(ratio_stats['median_ratio_p90q'] - 1) > 0.05
    df = pd.DataFrame(rows)
    for qc in QCRIT_SWEEP:
        sub = df[df.qcrit == qc].set_index('variant')
        nj = sub.loc['joint', 'n_seg_maxP_ge_50']
        for name in ('tp-shuffled', 'elev-shuffled'):
            nv = sub.loc[name, 'n_seg_maxP_ge_50']
            denom = max(nj, 1)
            rel = (nv - nj) / denom
            flag = '  <-- DIVERGES' if abs(rel) > 0.05 and nj > 0 else ''
            if abs(rel) > 0.05 and nj > 0:
                any_div = True
            print(f'  qcrit={qc:>6.1f}  {name:>14s} nSeg={nv:>4d} vs joint '
                  f'{nj:>4d}  ({rel:+.1%}){flag}')
    print(f'\nverdict: {"DIVERGENCE at >=1 threshold" if any_div else "no material divergence at any threshold"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', default='both', choices=['m10', 'm100', 'both'])
    ap.add_argument('--crest-source', default='official',
                    choices=['terrarium', 'landsd', 'official', 'hybrid',
                             'hybridspot'],
                    help='DEM for the crest freeboard proxy (default official, '
                         'the canonical data.gov.hk ASC, writes *_official '
                         'outputs; landsd = Esri re-published copy, writes '
                         '*_landsd; terrarium writes the legacy un-suffixed '
                         'files; hybridspot = hybrid + SpotHeight replacement '
                         'of flag-0 segments too, the less-conservative '
                         'sensitivity bound from spotheight_crest_report.md '
                         'section 6, writes *_hybridspot)')
    ap.add_argument('--shuffle-tp', action='store_true',
                    help='run tp-shuffled control experiments (m10 and m100)')
    ap.add_argument('--shuffle-elev', action='store_true',
                    help='run elev-shuffled control experiment (m100)')
    ap.add_argument('--qcrit-sweep', action='store_true',
                    help='pure re-analysis of the M=100 npz files over '
                         'exceedance thresholds (no sampling)')
    ap.add_argument('--case', default='saola', choices=sorted(CASES),
                    help='saola (2023, default) or ragasa (2025): switches '
                         'sample dirs, time windows, output prefixes')
    ap.add_argument('--type-overrides', default=None, metavar='CSV',
                    help='imagery-reviewed coast-type corrections '
                    '(columns seg_id,new_type); applied to the full segment '
                    'table BEFORE the eurotop_applicable filter, so '
                    'natural->engineered promotions enter the computation. '
                    'Appends _typerev to output suffixes.')
    ap.add_argument('--exclude-segs', default=None, metavar='CSV',
                    help='segments with broken geometry (line on open water / '
                         'bridge deck; column seg_id): deactivated from the q '
                         'computation (eurotop_applicable=False, node_idx=-1). '
                         'Appends _gex to output suffixes.')
    args = ap.parse_args()
    case = CASES[args.case]
    m10_ts, m100_ts = case['m10_ts'], case['m100_ts']
    if args.qcrit_sweep:
        qcrit_sweep()
        return
    sfx = {'official': '_official', 'landsd': '_landsd',
           'hybrid': '_hybrid', 'hybridspot': '_hybridspot'}.get(
        args.crest_source, '')
    if args.type_overrides:
        sfx += '_typerev'
    if args.exclude_segs:
        sfx += '_gex'
    if (args.shuffle_tp or args.shuffle_elev) and sfx:
        raise SystemExit('shuffle controls are defined against the legacy '
                         'Terrarium baseline; use --crest-source terrarium')
    os.makedirs(OUT_DIR, exist_ok=True)

    seg = pd.read_csv(SEG_CSV)
    if args.type_overrides:
        seg = apply_type_overrides(seg, args.type_overrides)
    if args.exclude_segs:
        excl = pd.read_csv(args.exclude_segs).seg_id.astype(int).tolist()
        seg = seg.set_index('seg_id')
        hit = [s for s in excl if s in seg.index]
        seg.loc[hit, ['eurotop_applicable', 'node_idx']] = [False, -1]
        seg = seg.reset_index()
        print(f'exclude-segs from {args.exclude_segs}: {len(hit)} segments '
              f'deactivated (broken geometry: line on water/bridge deck)')
    eng = seg[(seg.eurotop_applicable == True) & (seg.node_idx >= 0)].copy()
    eng = eng.reset_index(drop=True)
    n_eng_all = int((seg.eurotop_applicable == True).sum())
    print(f'segments total {len(seg)}; eurotop_applicable {n_eng_all}; '
          f'with node_idx>=0: {len(eng)} (rest get q=NaN / no output row)')
    print('coast types (engineered):', dict(eng.coast_type.value_counts()))

    dep_hk = load_node_depths()
    h_toe = np.maximum(dep_hk[eng.node_idx.values], H_TOE_FLOOR)
    n_floored = int((dep_hk[eng.node_idx.values] < H_TOE_FLOOR).sum())
    print(f'h_toe percentiles [m]: '
          f'{np.percentile(h_toe, [0, 25, 50, 75, 100]).round(2)}; '
          f'floored at {H_TOE_FLOOR} m for {n_floored} segments')

    crest_src = None
    if args.crest_source in ('official', 'hybrid', 'hybridspot'):
        dtm, xll, yll, cs, nrow, ncol = load_dtm_official()
        rc_land, dem_flag, rc_raw = crest_proxies_official(
            eng, dtm, xll, yll, cs, nrow, ncol)
        print(f'official LandsD crest proxy (MSL-referenced, clamped) '
              f'percentiles [m]: '
              f'{np.percentile(rc_land, [0, 25, 50, 75, 100]).round(2)}; '
              f'raw percentiles: {np.nanpercentile(rc_raw, [0, 25, 50, 75, 100]).round(2)}; '
              f'clamp flags: min={int((dem_flag == 1).sum())}, '
              f'max={int((dem_flag == 2).sum())}')
        if args.crest_source in ('hybrid', 'hybridspot'):
            dem_flag_dtm = dem_flag.copy()
            rc_land, dem_flag, rc_raw, crest_src = apply_spot_crests(
                eng, rc_land, dem_flag, rc_raw,
                include_flag0=(args.crest_source == 'hybridspot'))
            rc_land, dem_flag, rc_raw, crest_src = apply_design_crests(
                eng, rc_land, dem_flag, rc_raw, crest_src)
            rc_land, dem_flag, rc_raw, crest_src = impute_missing_crests(
                eng, rc_land, dem_flag, rc_raw, crest_src)
    elif args.crest_source == 'landsd':
        dtm, dtm_ox, dtm_oy, dtm_res = load_dtm_landsd()
        rc_land, dem_flag, rc_raw = crest_proxies_landsd(
            eng, dtm, dtm_ox, dtm_oy, dtm_res)
        print(f'LandsD crest proxy (MSL-referenced, clamped) percentiles [m]: '
              f'{np.percentile(rc_land, [0, 25, 50, 75, 100]).round(2)}; '
              f'raw percentiles: {np.nanpercentile(rc_raw, [0, 25, 50, 75, 100]).round(2)}; '
              f'clamp flags: min={int((dem_flag == 1).sum())}, '
              f'max={int((dem_flag == 2).sum())}')
    else:
        elev, lat_axis, lon_axis = load_dem_mosaic()
        rc_land, dem_flag, rc_raw = crest_proxies(eng, elev, lat_axis, lon_axis)
        print(f'crest proxy percentiles [m]: '
              f'{np.percentile(rc_land, [0, 25, 50, 75, 100]).round(2)}; '
              f'clamp flags: min={int((dem_flag == 1).sum())}, '
              f'max={int((dem_flag == 2).sum())}')

    results = {}
    if args.which in ('m10', 'both') or args.shuffle_tp:
        results['m10'] = run_ensemble(
            'm10' + sfx,
            os.path.join(SAMPLES_BASE, case['m10_dir'], 'test/samples'),
            m10_ts, 10, eng, h_toe, rc_land, rc_raw=rc_raw, case=args.case)
    m100_dir = os.path.join(SAMPLES_BASE, case['m100_dir'], 'test/samples')
    m100_npz = os.path.join(OUT_DIR, f'{args.case}_q_m100{sfx}.npz')
    need_m100 = (args.which in ('m100', 'both')
                 or ((args.shuffle_tp or args.shuffle_elev)
                     and not os.path.exists(m100_npz)))
    if need_m100:
        n_files = (len([f for f in os.listdir(m100_dir) if f.endswith('.npz')])
                   if os.path.isdir(m100_dir) else 0)
        if n_files < len(m100_ts) * 100:
            print(f'--- ensemble m100 SKIPPED: only {n_files}/'
                  f'{len(m100_ts) * 100} sample files present')
        else:
            results['m100'] = run_ensemble('m100' + sfx, m100_dir, m100_ts, 100,
                                           eng, h_toe, rc_land, rc_raw=rc_raw,
                                           case=args.case)
    elif (args.shuffle_tp or args.shuffle_elev) and os.path.exists(m100_npz):
        print(f'--- reusing joint baseline {m100_npz}')
    if args.shuffle_tp:
        shuffle_tp_control(eng, h_toe, rc_land, dem_flag)
    if args.shuffle_tp or args.shuffle_elev:
        if not os.path.exists(m100_npz):
            print('--- M=100 controls SKIPPED: joint baseline npz missing')
        else:
            m100_controls(eng, h_toe, rc_land, dem_flag,
                          do_tp=args.shuffle_tp, do_elev=args.shuffle_elev)

    # ---------------- summary CSV (from m10) ----------------
    if 'm10' not in results:
        return
    q_lpsm, p_gt1, q_mean, ts_str, waves, n_weir = results['m10']
    S, T, M = q_lpsm.shape
    max_p = np.nanmax(p_gt1, axis=1)
    ts_of_max_p = ts_str[np.argmax(np.nan_to_num(p_gt1, nan=-1.0), axis=1)]
    max_mean_q = np.nanmax(q_mean, axis=1)
    ts_of_max_q = ts_str[np.argmax(np.nan_to_num(q_mean, nan=-1.0), axis=1)]
    summary = pd.DataFrame({
        'seg_id': eng.seg_id.values,
        'mid_lat': eng.mid_lat.values,
        'mid_lon': eng.mid_lon.values,
        'coast_type': eng.coast_type.values,
        'h_toe': h_toe.round(3),
        'crest_proxy': rc_land.round(3),
        'crest_proxy_raw': np.round(rc_raw, 3),
        'crest_flag': dem_flag,
        'max_p_gt1': max_p.round(4),
        'ts_of_max_p': ts_of_max_p,
        'max_mean_q_lpsm': max_mean_q.round(2),
        'ts_of_max_q': ts_of_max_q,
        'n_weir_fallback': n_weir,
    })
    if crest_src is not None:
        summary['crest_src'] = crest_src
        summary['crest_flag_dtm'] = dem_flag_dtm
    out_csv = os.path.join(OUT_DIR, f'{args.case}_segment_summary{sfx}.csv')
    summary.to_csv(out_csv, index=False)
    print(f'wrote {out_csv}')

    # ---------------- verification ----------------
    print('\n================ VERIFICATION ================')
    print(f'engineered segments processed: {S} '
          f'(of {n_eng_all} eurotop_applicable; '
          f'{n_eng_all - S} lack node_idx -> q NaN, excluded)')
    top = summary.nlargest(15, 'max_p_gt1')
    print('\ntop 15 segments by event-max P(q>1 L/s/m) '
          '(cf = crest_flag: 1 => DEM saw sea/void landward, crest floored at '
          '1.5 m — typical on post-SRTM reclamation, treat q as unreliable):')
    for _, r in top.iterrows():
        area = nearest_area(r.mid_lat, r.mid_lon)
        nrm = eng.loc[eng.seg_id == r.seg_id, 'normal_deg'].iloc[0]
        print(f'  seg {int(r.seg_id):5d}  P={r.max_p_gt1:.2f}  '
              f'{r.coast_type:16s} {area:38s} '
              f'({r.mid_lat:.3f},{r.mid_lon:.3f}) normal={nrm:5.1f}  '
              f'ts_max={r.ts_of_max_p}  qmean_max={r.max_mean_q_lpsm:8.1f} '
              f'cf={r.crest_flag}')
    n_cf1_top = int((top.crest_flag == 1).sum())
    print(f'  NOTE: {n_cf1_top}/15 top segments have crest_flag=1 '
          f'(unreliable crest proxy); overall flag counts: '
          f'{summary.crest_flag.value_counts().to_dict()}')
    top_rel = summary[summary.crest_flag != 1].nlargest(15, 'max_p_gt1')
    print('\ntop 15 EXCLUDING crest_flag=1 (physically more reliable):')
    for _, r in top_rel.iterrows():
        area = nearest_area(r.mid_lat, r.mid_lon)
        nrm = eng.loc[eng.seg_id == r.seg_id, 'normal_deg'].iloc[0]
        print(f'  seg {int(r.seg_id):5d}  P={r.max_p_gt1:.2f}  '
              f'{r.coast_type:16s} {area:38s} '
              f'({r.mid_lat:.3f},{r.mid_lon:.3f}) normal={nrm:5.1f}  '
              f'ts_max={r.ts_of_max_p}  qmean_max={r.max_mean_q_lpsm:8.1f} '
              f'cf={r.crest_flag}')

    frac = np.nanmean(p_gt1 > 0.5, axis=0)  # fraction of segments with majority P
    mean_p = np.nanmean(p_gt1, axis=0)
    print('\nterritory-wide overtopping activity (m10): fraction of segments '
          'with P(q>1)>0.5 / mean P, every 3 h:')
    vday, (vh0, vh1) = case['verif_day'], case['verif_hours']
    for i in range(0, T, 3):
        mark = ' <==' if ts_str[i].startswith(vday) and \
            vh0 <= int(ts_str[i][-2:]) <= vh1 else ''
        print(f'  {ts_str[i]}  frac={frac[i]:.3f}  meanP={mean_p[i]:.3f}{mark}')
    i_pk = int(mean_p.argmax())
    print(f'  peak of territory-wide mean P at {ts_str[i_pk]} '
          f'(closest approach ~{case["closest_approach"]})')

    # dm convention plausibility: dm at top-segment nodes during peak window
    peak_rows = [i for i, t in enumerate(ts_str)
                 if t.startswith(vday) and vh0 <= int(t[-2:]) <= vh1]
    top_nodes = eng.node_idx.values[summary.max_p_gt1.values >=
                                    np.quantile(summary.max_p_gt1, 0.99)]
    dm_pk = waves[peak_rows][:, :, :, 2][:, :, np.isin(eng.node_idx.values, top_nodes)]
    print(f'\ndm at top-1% segments during {vday} {vh0:02d}..{vh1:02d} UTC: '
          f'mean={np.nanmean(dm_pk):.0f} deg, '
          f'P10..P90={np.nanpercentile(dm_pk, [10, 90]).round(0)}; '
          f'segment seaward normals mean='
          f'{eng.normal_deg.values[summary.max_p_gt1.values >= np.quantile(summary.max_p_gt1, 0.99)].mean():.0f} deg '
          '(aligned -> waves arrive from the direction the coast faces: '
          'dm is a coming-from convention, matching beta = |dm - normal|)')
    n_weir_tot = int(n_weir.sum())
    print(f'weir-fallback (Rc<0) occurrences: {n_weir_tot} '
          f'({n_weir_tot / (S * T * M) * 100:.3f}% of seg x ts x member)')
    q_nan = int(np.isnan(q_lpsm).sum())
    print(f'NaN in q_lpsm: {q_nan}')
    if 'm100' in results:
        q100 = results['m100'][0]
        print(f'm100: q_lpsm {q100.shape}, max mean q = '
              f'{np.nanmax(results["m100"][2]):.1f} L/s/m, '
              f'max P(q>1) = {np.nanmax(results["m100"][1]):.2f}, '
              f'NaN cells: {int(np.isnan(q100).sum())}')


if __name__ == '__main__':
    main()
