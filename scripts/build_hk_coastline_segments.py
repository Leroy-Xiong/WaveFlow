#!/usr/bin/env python
"""
build_hk_coastline_segments.py — Build Figure 6 (panel a) HK coastline segment dataset.

Pipeline
--------
1. Fetch OSM data via Overpass API (cached to data/hk_coastline/osm_raw/ so
   re-runs do not re-hit the API):
     - natural=coastline ways (out geom)          -> coastline geometry
     - man_made=seawall|embankment|groyne, barrier=retaining_wall -> defense labeling
     - natural=beach / surface=sand ways+relations -> natural_sandy labeling
     - natural=wetland / natural=mudflat ways      -> lowlying_open labeling
2. Split coastline ways into ~150 m segments (per-way: n_seg = round(L/150),
   even resampling; local equirectangular meter frame around 22.4N).
3. Seaward normal per segment: unit normal candidates (left/right of the
   along-shore tangent); test which side has model ocean nodes closer
   (offset probe points 500 m each side, nearest-node distance via cKDTree).
   If both probes are > 1 km from any node, fall back to projecting the
   nearest-node direction onto the normals. Angle = degrees clockwise from
   north.
4. Snap each segment to the nearest model ocean node
   (data/hindcast/hk_coordinates.npy, 7788 unstructured nodes); max distance
   1.5 km, else node_idx = -1 (no wave forcing).
5. Coast-type classification heuristic (documented, to be manually
   spot-checked against satellite imagery later):
     - tags on the coastline way itself (OSM HK seawalls are mostly NOT
       tagged man_made=seawall — zero such ways in bbox — but carry
       seamark/wall/reclaimed tags instead):
         * seamark:shoreline_construction:category=sea_wall, wall=seawall,
           barrier=wall, reinforced=yes                 -> seawall_vertical
         * seamark category rip_rap / man_made=breakwater
                                                         -> revetment_slope
         * reclaimed=yes / anthropogenic=yes (HK reclamation edges are
           engineered walls/revetments)                  -> seawall_vertical
     - else defense way within 60 m of midpoint          -> engineered:
         * tags hint slope (key contains 'slope', or value contains
           'slop'/'diagonal'/'riprap'/'revetment')       -> revetment_slope
         * otherwise                                     -> seawall_vertical
     - else beach/sand way within 150 m                  -> natural_sandy
     - else wetland/mudflat within 150 m, or within 800 m of a known
       low-lying spot (Tai O / Lau Fau Shan / Ma Wan Chung)
                                                         -> lowlying_open
     - else (HK natural coast is predominantly rocky headland)
                                                         -> natural_rocky
     ('unknown' reserved; currently unused.)
6. EurOtop defaults per class:
     seawall_vertical  gamma_f=1.0, slope=vertical,    applicable=True
     revetment_slope   gamma_f=0.8 (rock armor), 1:2,  applicable=True
     natural_*/lowlying_open  gamma_f=1.0,             applicable=False
     (not meaningful for EurOtop; kept for surge/elev display).

Outputs
-------
  data/hk_coastline/hk_coastline_segments.csv
  data/hk_coastline/hk_coastline_segments.geojson
  output/plot/figure6/coastline_segments_check.png  (sanity plot)

Usage: python scripts/build_hk_coastline_segments.py
"""

import json
import logging
import math
import os
import time

import numpy as np
import requests
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO, "data/hk_coastline/osm_raw")
OUT_DIR = os.path.join(REPO, "data/hk_coastline")
PLOT_DIR = os.path.join(REPO, "output/plot/figure6")
NODES_NPY = os.path.join(REPO, "data/hindcast/hk_coordinates.npy")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# overpass-api.de rejects the default python-requests UA with 406
HEADERS = {"User-Agent": "hk-coastline-segments/1.0 (wave-model Fig 6 research)"}
BBOX = (22.15, 113.82, 22.58, 114.45)  # south, west, north, east

SEG_LEN_M = 150.0
NODE_MAX_DIST_M = 1500.0
DEFENSE_NEAR_M = 60.0
BEACH_NEAR_M = 150.0
WETLAND_NEAR_M = 150.0
NORMAL_PROBE_M = 500.0

# Local equirectangular meter frame around 22.4N (good to <0.5% over the bbox)
LAT0 = 22.4
M_PER_DEG_LAT = 111132.954
M_PER_DEG_LON = 111132.954 * math.cos(math.radians(LAT0))

# Known low-lying spots (lat, lon), radius 800 m
LOWLYING_SPOTS = [
    ("Tai O", 22.2539, 113.8611),
    ("Lau Fau Shan", 22.4689, 113.9833),
    ("Ma Wan Chung", 22.2778, 113.9389),
]
LOWLYING_RADIUS_M = 800.0

CLASS_STYLE = {  # (gamma_f, eurotop_applicable, color)
    "seawall_vertical": (1.0, True, "#d62728"),
    "revetment_slope": (0.8, True, "#ff7f0e"),
    "natural_rocky": (1.0, False, "#8c564b"),
    "natural_sandy": (1.0, False, "#e3c46b"),
    "lowlying_open": (1.0, False, "#2ca02c"),
    "unknown": (1.0, False, "#7f7f7f"),
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("hk_coastline")


# --------------------------------------------------------------------------
# Overpass fetch with cache-if-exists and one retry after 60 s
# --------------------------------------------------------------------------
def overpass_query(query: str, cache_name: str) -> dict:
    os.makedirs(RAW_DIR, exist_ok=True)
    cache_path = os.path.join(RAW_DIR, cache_name)
    if os.path.exists(cache_path):
        log.info("cache hit: %s", cache_path)
        with open(cache_path) as f:
            return json.load(f)

    for attempt in (1, 2):
        try:
            log.info("Overpass POST (%s), attempt %d", cache_name, attempt)
            r = requests.post(OVERPASS_URL, data={"data": query},
                              headers=HEADERS, timeout=300)
            r.raise_for_status()
            data = r.json()
            with open(cache_path, "w") as f:
                json.dump(data, f)
            log.info("cached %d elements -> %s", len(data.get("elements", [])),
                     cache_path)
            return data
        except Exception as e:  # noqa: BLE001 - log and retry once
            log.warning("Overpass attempt %d failed: %s", attempt, e)
            if attempt == 1:
                log.info("sleeping 60 s before retry")
                time.sleep(60)
    # Fallback: overpass-api.de intermittently rejects the python-requests
    # TLS fingerprint (406/504) while curl to the same endpoint succeeds.
    log.warning("requests failed twice; falling back to curl for %s", cache_name)
    import subprocess
    out = subprocess.run(
        ["curl", "-sf", "-X", "POST", "--data-urlencode", f"data={query}",
         OVERPASS_URL],
        check=True, capture_output=True, timeout=600,
    )
    data = json.loads(out.stdout)
    with open(cache_path, "w") as f:
        json.dump(data, f)
    log.info("cached %d elements -> %s (via curl)",
             len(data.get("elements", [])), cache_path)
    return data


def build_queries():
    s, w, n, e = BBOX
    bb = f"({s},{w},{n},{e})"
    q_coast = f'[out:json][timeout:180];way["natural"="coastline"]{bb};out geom;'
    q_defense = (
        f'[out:json][timeout:180];('
        f'way["man_made"="seawall"]{bb};'
        f'way["man_made"="embankment"]{bb};'
        f'way["man_made"="groyne"]{bb};'
        f'way["barrier"="retaining_wall"]{bb};'
        f');out geom;'
    )
    q_beach = (
        f'[out:json][timeout:180];('
        f'way["natural"="beach"]{bb};'
        f'way["surface"="sand"]{bb};'
        f'relation["natural"="beach"]{bb};'
        f');out geom;'
    )
    q_wet = (
        f'[out:json][timeout:180];('
        f'way["natural"="wetland"]{bb};'
        f'way["natural"="mudflat"]{bb};'
        f');out geom;'
    )
    return {
        "coastline.json": q_coast,
        "defense.json": q_defense,
        "beach.json": q_beach,
        "wetland.json": q_wet,
    }


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def to_xy(lat, lon):
    return lon * M_PER_DEG_LON, lat * M_PER_DEG_LAT


def way_points_xy(el):
    """List of (x, y) meter tuples from an OSM way element with geometry."""
    geom = el.get("geometry") or []
    return [to_xy(p["lat"], p["lon"]) for p in geom]


def collect_geoms_xy(data, include_relation_members=True):
    """Return list of (element, [xy points]) for ways; relation way-members too."""
    out = []
    for el in data.get("elements", []):
        if el.get("type") == "way":
            pts = way_points_xy(el)
            if len(pts) >= 2:
                out.append((el, pts))
        elif include_relation_members and el.get("type") == "relation":
            for m in el.get("members", []):
                if m.get("type") == "way" and m.get("geometry"):
                    pts = [to_xy(p["lat"], p["lon"]) for p in m["geometry"]]
                    if len(pts) >= 2:
                        out.append((el, pts))
    return out


def sample_geoms_xy(geoms, step_m=25.0):
    """Densely sample polyline geometries -> (N,2) meter array for KDTree."""
    pts = []
    for _, xy in geoms:
        arr = np.asarray(xy)
        d = np.r_[0.0, np.cumsum(np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1])))]
        L = d[-1]
        if L <= 0:
            continue
        t = np.arange(0.0, L + 1e-6, step_m)
        pts.append(np.c_[np.interp(t, d, arr[:, 0]), np.interp(t, d, arr[:, 1])])
    if not pts:
        return np.empty((0, 2))
    return np.vstack(pts)


def split_way(xy, seg_len=SEG_LEN_M):
    """Split polyline into segments of ~seg_len. Yields (p0, p1, mid, length)."""
    arr = np.asarray(xy, dtype=float)
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1])))]
    L = d[-1]
    if L < 1.0:
        return
    n_seg = max(1, int(round(L / seg_len)))
    edges = np.linspace(0.0, L, n_seg + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])

    def at(t):
        return np.interp(t, d, arr[:, 0]), np.interp(t, d, arr[:, 1])

    for i in range(n_seg):
        x0, y0 = at(edges[i])
        x1, y1 = at(edges[i + 1])
        xm, ym = at(mids[i])
        yield (x0, y0), (x1, y1), (xm, ym), edges[i + 1] - edges[i]


def normal_deg(p0, p1, mid, node_tree):
    """Seaward unit normal, degrees clockwise from north."""
    tx, ty = p1[0] - p0[0], p1[1] - p0[1]
    n = math.hypot(tx, ty)
    if n == 0.0:
        # degenerate chord (duplicated OSM vertices): point toward nearest node
        _, idx = node_tree.query([mid[0], mid[1]])
        vx, vy = node_tree.data[idx] - np.array(mid)
        return math.degrees(math.atan2(vx, vy)) % 360.0
    tx, ty = tx / n, ty / n
    n1 = (-ty, tx)  # left normal
    n2 = (ty, -tx)  # right normal
    d1, _ = node_tree.query([mid[0] + NORMAL_PROBE_M * n1[0],
                             mid[1] + NORMAL_PROBE_M * n1[1]])
    d2, _ = node_tree.query([mid[0] + NORMAL_PROBE_M * n2[0],
                             mid[1] + NORMAL_PROBE_M * n2[1]])
    if min(d1, d2) <= 1000.0:
        nx, ny = n1 if d1 <= d2 else n2
    else:
        # fallback: side toward the nearest node projected on the normals
        _, idx = node_tree.query([mid[0], mid[1]])
        vx, vy = node_tree.data[idx] - np.array(mid)
        nx, ny = n1 if (vx * n1[0] + vy * n1[1]) >= 0 else n2
    return math.degrees(math.atan2(nx, ny)) % 360.0


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def tags_indicate_slope(tags):
    for k, v in (tags or {}).items():
        kl, vl = k.lower(), str(v).lower()
        if "slope" in kl:
            return True
        if any(s in vl for s in ("slop", "diagonal", "riprap", "revetment")):
            return True
    return False


def classify(mid, way_tags, defense_geoms, defense_tree, beach_tree, wet_tree):
    """Heuristic coast-type label for a segment midpoint (x, y meters)."""
    t = way_tags or {}
    seamark_cat = t.get("seamark:shoreline_construction:category", "")
    if (seamark_cat == "sea_wall" or t.get("wall") == "seawall"
            or t.get("barrier") == "wall" or t.get("reinforced") == "yes"):
        return "seawall_vertical"
    if seamark_cat in ("rip_rap", "groyne", "breakwater") \
            or t.get("man_made") == "breakwater":
        return "revetment_slope"
    if t.get("reclaimed") == "yes" or t.get("anthropogenic") == "yes":
        return "seawall_vertical"
    if defense_tree is not None:
        dist, idx = defense_tree.query([mid[0], mid[1]])
        if dist <= DEFENSE_NEAR_M:
            tags = defense_geoms[idx][0].get("tags", {})
            return "revetment_slope" if tags_indicate_slope(tags) \
                else "seawall_vertical"
    if beach_tree is not None:
        dist, _ = beach_tree.query([mid[0], mid[1]])
        if dist <= BEACH_NEAR_M:
            return "natural_sandy"
    if wet_tree is not None:
        dist, _ = wet_tree.query([mid[0], mid[1]])
        if dist <= WETLAND_NEAR_M:
            return "lowlying_open"
    for _, lat, lon in LOWLYING_SPOTS:
        sx, sy = to_xy(lat, lon)
        if math.hypot(mid[0] - sx, mid[1] - sy) <= LOWLYING_RADIUS_M:
            return "lowlying_open"
    return "natural_rocky"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1) fetch / load OSM
    raw = {name: overpass_query(q, name) for name, q in build_queries().items()}

    # 2) model ocean nodes
    nodes_ll = np.load(NODES_NPY)  # (7788, 2) = (lat, lon)
    nodes_xy = np.c_[nodes_ll[:, 1] * M_PER_DEG_LON, nodes_ll[:, 0] * M_PER_DEG_LAT]
    node_tree = cKDTree(nodes_xy)
    log.info("loaded %d model nodes", len(nodes_ll))

    # 3) reference geometries for labeling
    defense_geoms = collect_geoms_xy(raw["defense.json"], include_relation_members=False)
    defense_tree = None
    if defense_geoms:
        # KDTree over per-way representative points; we keep per-way tags, so
        # use densely sampled points but remember the way index per sample.
        pts, owner = [], []
        for i, (_, xy) in enumerate(defense_geoms):
            arr = np.asarray(xy)
            d = np.r_[0.0, np.cumsum(np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1])))]
            L = d[-1]
            if L <= 0:
                continue
            t = np.arange(0.0, L + 1e-6, 25.0)
            pts.append(np.c_[np.interp(t, d, arr[:, 0]), np.interp(t, d, arr[:, 1])])
            owner.extend([i] * len(t))
        if pts:
            raw_tree = cKDTree(np.vstack(pts))
            defense_owner = np.asarray(owner)

            # query returns sample idx -> map back to the owning way element
            class _DefenseTree:
                def query(self, p):
                    dist, si = raw_tree.query(p)
                    return dist, int(defense_owner[si])
            defense_tree = _DefenseTree()
    beach_tree = None
    bp = sample_geoms_xy(collect_geoms_xy(raw["beach.json"]))
    if len(bp):
        beach_tree = cKDTree(bp)
    wet_tree = None
    wp = sample_geoms_xy(collect_geoms_xy(raw["wetland.json"]))
    if len(wp):
        wet_tree = cKDTree(wp)
    log.info("label geoms: defense=%d ways, beach pts=%d, wetland pts=%d",
             len(defense_geoms), len(bp), len(wp))

    # 4) build segments from coastline ways
    segments = []
    coast_geoms = collect_geoms_xy(raw["coastline.json"], include_relation_members=False)
    log.info("coastline ways: %d", len(coast_geoms))
    for el, xy in coast_geoms:
        for p0, p1, mid, length in split_way(xy):
            seg = {
                "p0": p0, "p1": p1, "mid": mid,
                "length_m": length,
                "normal_deg": normal_deg(p0, p1, mid, node_tree),
            }
            # snap to model node
            dist, idx = node_tree.query([mid[0], mid[1]])
            seg["node_idx"] = int(idx) if dist <= NODE_MAX_DIST_M else -1
            seg["node_dist_m"] = float(dist)
            seg["coast_type"] = classify(mid, el.get("tags", {}),
                                         defense_geoms, defense_tree,
                                         beach_tree, wet_tree)
            segments.append(seg)
    log.info("built %d segments", len(segments))

    # 5) EurOtop defaults + lat/lon columns
    for i, seg in enumerate(segments):
        seg["seg_id"] = i
        gamma, applic, _ = CLASS_STYLE[seg["coast_type"]]
        seg["gamma_f"] = gamma
        seg["eurotop_applicable"] = applic
        seg["mid_lat"] = seg["mid"][1] / M_PER_DEG_LAT
        seg["mid_lon"] = seg["mid"][0] / M_PER_DEG_LON
        seg["start_lat"] = seg["p0"][1] / M_PER_DEG_LAT
        seg["start_lon"] = seg["p0"][0] / M_PER_DEG_LON
        seg["end_lat"] = seg["p1"][1] / M_PER_DEG_LAT
        seg["end_lon"] = seg["p1"][0] / M_PER_DEG_LON

    # 6) write CSV
    csv_path = os.path.join(OUT_DIR, "hk_coastline_segments.csv")
    cols = ["seg_id", "mid_lat", "mid_lon", "start_lat", "start_lon",
            "end_lat", "end_lon", "length_m", "normal_deg", "node_idx",
            "node_dist_m", "coast_type", "gamma_f", "eurotop_applicable"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for s in segments:
            f.write(",".join([
                str(s["seg_id"]),
                f"{s['mid_lat']:.6f}", f"{s['mid_lon']:.6f}",
                f"{s['start_lat']:.6f}", f"{s['start_lon']:.6f}",
                f"{s['end_lat']:.6f}", f"{s['end_lon']:.6f}",
                f"{s['length_m']:.1f}", f"{s['normal_deg']:.1f}",
                str(s["node_idx"]), f"{s['node_dist_m']:.1f}",
                s["coast_type"], f"{s['gamma_f']:.2f}",
                "true" if s["eurotop_applicable"] else "false",
            ]) + "\n")
    log.info("wrote %s", csv_path)

    # 7) write GeoJSON (manual, no geopandas)
    feats = []
    for s in segments:
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [
                [round(s["start_lon"], 6), round(s["start_lat"], 6)],
                [round(s["end_lon"], 6), round(s["end_lat"], 6)],
            ]},
            "properties": {c: s[c] for c in cols},
        })
    gj_path = os.path.join(OUT_DIR, "hk_coastline_segments.geojson")
    with open(gj_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    log.info("wrote %s", gj_path)

    # 8) summary stats
    from collections import Counter
    cnt = Counter(s["coast_type"] for s in segments)
    n_matched = sum(1 for s in segments if s["node_idx"] >= 0)
    log.info("=== SUMMARY ===")
    log.info("total segments: %d (total length %.1f km)",
             len(segments), sum(s["length_m"] for s in segments) / 1000.0)
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        log.info("  %-18s %5d  (%4.1f%%)", k, v, 100.0 * v / len(segments))
    log.info("node-matched (<=1.5 km): %d / %d (%.1f%%)",
             n_matched, len(segments), 100.0 * n_matched / len(segments))

    # 9) sanity plot
    os.environ.setdefault("CARTOPY_DATA_DIR",
                          os.path.join(REPO, "data/cartopy_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_png = os.path.join(PLOT_DIR, "coastline_segments_check.png")

    def _scatter(ax, **kw):
        ax.scatter(nodes_ll[:, 1], nodes_ll[:, 0], s=1, c="0.75",
                   label="model nodes", **kw)
        for ctype, (_, _, color) in CLASS_STYLE.items():
            sel = [s for s in segments if s["coast_type"] == ctype]
            if not sel:
                continue
            ax.scatter([s["mid_lon"] for s in sel], [s["mid_lat"] for s in sel],
                       s=6, c=color, label=f"{ctype} ({len(sel)})", zorder=5,
                       **kw)
        ax.legend(loc="lower right", fontsize=7, markerscale=3)
        ax.set_title("HK coastline segments by coast type (Fig 6 panel a check)")

    def _plain_plot():
        fig, ax = plt.subplots(figsize=(9, 8))
        _scatter(ax)
        ax.set_aspect(1 / math.cos(math.radians(LAT0)))
        ax.set_xlim(BBOX[1], BBOX[3])
        ax.set_ylim(BBOX[0], BBOX[2])
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        ax.set_extent([BBOX[1], BBOX[3], BBOX[0], BBOX[2]])
        try:
            ax.add_feature(cfeature.GSHHSFeature(scale="full", levels=[1]),
                           facecolor="#f2f2f2", edgecolor="#bbbbbb", lw=0.4)
            # GSHHS download is lazy; force it now so failure falls back
            fig.canvas.draw()
        except Exception as e:  # noqa: BLE001
            log.warning("GSHHS basemap failed (%s); plotting without it", e)
            plt.close(fig)
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
            ax.set_extent([BBOX[1], BBOX[3], BBOX[0], BBOX[2]])
        _scatter(ax, transform=ccrs.PlateCarree())
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log.warning("cartopy plot failed (%s); plain scatter fallback", e)
        _plain_plot()
    log.info("wrote %s", out_png)


if __name__ == "__main__":
    main()
