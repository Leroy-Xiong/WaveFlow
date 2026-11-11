#!/usr/bin/env python
"""Fetch the official Hong Kong coastline (shoreline) from LandsD digital
topographic map open data (CSDI / hkmapservice direct downloads).

Products
--------
- iB1000  (1:1,000 Digital Topographic Map)  -- dataset landsd_rcd_1637223748322_25497
    Covers urban/developed areas only. Coastline layer:
    Layers/Hydrography/Shoreline.gml (SHORELINETYPE = HWM / SWA).
- iB5000  (1:5,000 Digital Topographic Map)  -- dataset landsd_rcd_1637224243141_96556
    Covers the whole HKSAR (193 sheets). Coastline carried in
    Layers/Hydrography/HydrographyLine.gml with FEATURECODE = HWM / SEW.

Route
-----
1. CSDI portal ArcGIS FeatureServer TileIndex layer: per-sheet metadata incl.
   the direct GML download URL of every map sheet.
2. Select sheets whose tile polygon intersects a buffer around our OSM
   coastline (data/hk_coastline/hk_coastline_segments.csv).
3. Download each sheet's GML zip from open.hkmapservice.gov.hk (directDownload).
4. Parse the shoreline layer out of every zip and merge into one dataset
   per product (EPSG:2326 native + WGS84 copy).

Outputs (under data/hk_coastline/landsd_official/)
-------------------------------------------------
- <prod>_tileindex.geojson              : tile index (EPSG:2326, axis-fixed)
- <prod>_sheets/<SHEETNO>.zip           : raw per-sheet GML zips (cached)
- landsd_shoreline_<prod>_epsg2326.geojson / _wgs84.geojson
- <prod>_sheet_selection.csv

Usage:
    python scripts/fetch_landsd_coastline.py --product iB1000
    python scripts/fetch_landsd_coastline.py --product iB5000
Rerunnable: cached tile index and already-downloaded zips are reused.
"""
import argparse
import io
import json
import ssl
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hk_coastline" / "landsd_official"
SEG_CSV = ROOT / "data" / "hk_coastline" / "hk_coastline_segments.csv"

PRODUCTS = {
    "ib1000": {
        "dataset_id": "landsd_rcd_1637223748322_25497",
        # (gml path suffix, feature tag, property tags, filter tag, keep values)
        "gml_suffix": "Layers/Hydrography/Shoreline.gml",
        "feature": "Shoreline",
        "prop_tags": ("SHORELINEID", "SHORELINETYPE", "STATUS", "LASTUPDATEDATE"),
        "filter_tag": None,  # keep everything in Shoreline.gml
        "keep_values": None,
        # portal FeatureServer geojson arrives as (N, E) -> swap to (E, N);
        # FME-generated GML posList is already (E, N) -> no swap.
        # (verified: sheet envelope matches swapped tile bounds)
        # WARNING: this (E, N) assumption holds for the Hydrography GMLs but
        # NOT for Layers/Relief/SpotHeight.gml, whose gml:pos follows strict
        # EPSG:2326 axis order (N, E) in the same zips — see
        # scripts/extract_ib1000_spotheights.py (validated against the
        # official 5 m DTM: median |spot-DTM| = 1.1 m after swapping).
        "swap_tileindex": True,
        "swap_gml": False,
    },
    "ib5000": {
        "dataset_id": "landsd_rcd_1637224243141_96556",
        "gml_suffix": "Layers/Hydrography/HydrographyLine.gml",
        "feature": "HydrographyLine",
        "prop_tags": ("FEATURE_ID", "FEATURECODE", "DATASTATUS",
                      "LASTUPDATEDATE"),
        "filter_tag": "FEATURECODE",
        # HWM = high water mark, SEW = seawall edge -> the coastline
        "keep_values": {"HWM", "SEW"},
        # same as iB1000 (verified against sheet 11-NE-B envelope)
        "swap_tileindex": True,
        "swap_gml": False,
    },
    "ib10000": {
        "dataset_id": "landsd_rcd_1637223983699_57817",
        "gml_suffix": "Layers/Hydrography/HYDRLINE.gml",
        "feature": "HYDRLINE",
        "prop_tags": ("GRAPHICID", "CLASS", "TYPE", "LASTUPDATEDATE"),
        "filter_tag": "CLASS",
        # SLN/COA = coastline
        "keep_values": {"SLN"},
        # this series' portal geojson is already (E, N), while its GML
        # follows the strict EPSG:2326 axis order (N, E) -> swap posList.
        # (verified against sheet 8-SW envelope)
        "swap_tileindex": False,
        "swap_gml": True,
    },
}

GML_NS = "http://www.opengis.net/gml"
FME_NS = "http://www.safe.com/gml/fme"

# open.hkmapservice.gov.hk has an incomplete certificate chain; the host is a
# LandsD government server reachable only with verification relaxed.
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-script"}


def http_get(url, timeout=180):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as r:
        return r.read()


def fetch_tileindex(prod, tileindex_path):
    """Download the full tile index (paged) as GeoJSON in EPSG:2326.

    Axis order differs by product (see PRODUCTS[*]['swap_tileindex']):
    some CSDI FeatureServers emit GeoJSON coordinates in EPSG axis order
    (northing, easting) for outSR=2326 instead of the GeoJSON (x, y)
    convention; those are swapped back to (easting, northing) here.
    """
    fs_url = (f"https://portal.csdi.gov.hk/server/rest/services/common/"
              f"{prod['dataset_id']}/FeatureServer/0/query")
    feats = []
    offset = 0
    while True:
        url = (f"{fs_url}?where=1%3D1&outFields=SHEETNO,GML,REVISIONDATE"
               f"&returnGeometry=true&f=geojson&outSR=2326"
               f"&resultRecordCount=2000&resultOffset={offset}")
        d = json.loads(http_get(url))
        feats.extend(d["features"])
        print(f"  tileindex: got {len(d['features'])} (total {len(feats)})")
        if len(d["features"]) < 2000:
            break
        offset += 2000
    if prod["swap_tileindex"]:
        for f in feats:
            ring = f["geometry"]["coordinates"][0]
            f["geometry"]["coordinates"][0] = [[y, x] for x, y in ring]
    tileindex_path.write_text(json.dumps(
        {"type": "FeatureCollection", "features": feats}))
    return feats


def select_sheets(feats, buffer_m, selection_csv):
    """Pick sheets intersecting a buffer around the OSM coastline."""
    tr = Transformer.from_crs(4326, 2326, always_xy=True)
    seg = pd.read_csv(SEG_CSV)
    lines = []
    for r in seg.itertuples():
        x1, y1 = tr.transform(r.start_lon, r.start_lat)
        x2, y2 = tr.transform(r.end_lon, r.end_lat)
        lines.append(LineString([(x1, y1), (x2, y2)]))
    coast_buf = unary_union(lines).buffer(buffer_m)

    rows = []
    for f in feats:
        g = shape(f["geometry"])
        rows.append({
            "sheet": f["properties"]["SHEETNO"],
            "url": f["properties"]["GML"],
            "revisiondate": f["properties"].get("REVISIONDATE"),
            "selected": bool(g.intersects(coast_buf)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(selection_csv, index=False)
    return df[df["selected"]].reset_index(drop=True)


def download_sheet(sheets_dir, sheet, url, retries=6):
    """Download one sheet zip; returns (sheet, ok, msg)."""
    dest = sheets_dir / f"{sheet}.zip"
    if dest.exists() and zipfile.is_zipfile(dest):
        return sheet, True, "cached"
    for attempt in range(retries):
        try:
            data = http_get(url, timeout=300)
            if not zipfile.is_zipfile(io.BytesIO(data)):
                raise ValueError(f"not a zip ({len(data)} bytes)")
            dest.write_bytes(data)
            return sheet, True, f"{len(data)} bytes"
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            time.sleep(5 * (attempt + 1))
    return sheet, False, msg


def parse_gml(raw, prod):
    """Parse one shoreline GML file -> list of (props dict, [(x,y),...])
    in EPSG:2326."""
    root = ET.fromstring(raw)
    out = []
    for feat in root.iter(f"{{{FME_NS}}}{prod['feature']}"):
        props = {}
        for tag in prod["prop_tags"]:
            el = feat.find(f"{{{FME_NS}}}{tag}")
            props[tag.lower()] = el.text if el is not None else None
        if prod["filter_tag"] is not None:
            if props.get(prod["filter_tag"].lower()) not in prod["keep_values"]:
                continue
        for ls in feat.iter(f"{{{GML_NS}}}LineString"):
            pl = ls.find(f"{{{GML_NS}}}posList")
            if pl is None or not pl.text:
                continue
            v = [float(x) for x in pl.text.split()]
            dim = int(ls.get("srsDimension", "0")) or 0
            if dim not in (2, 3) or len(v) % dim:
                dim = 3 if len(v) % 3 == 0 else 2
            pts = [(v[i], v[i + 1]) for i in range(0, len(v) - dim + 1, dim)]
            if prod["swap_gml"]:
                pts = [(y, x) for x, y in pts]  # (N, E) -> (E, N)
            if len(pts) >= 2:
                out.append((props, pts))
    return out


def build_merged_shoreline(prod, product, sheets_dir, sheets):
    """Parse all cached zips, merge shoreline lines, write GeoJSON outputs."""
    tr = Transformer.from_crs(2326, 4326, always_xy=True)
    feats2326, feats84 = [], []
    seen = set()
    n_files = 0
    for sheet in sheets:
        zpath = sheets_dir / f"{sheet}.zip"
        if not zpath.exists():
            continue
        with zipfile.ZipFile(zpath) as z:
            names = [n for n in z.namelist()
                     if n.endswith(prod["gml_suffix"])]
            if not names:
                continue
            raw = z.read(names[0])
        parsed = parse_gml(raw, prod)
        if parsed:
            n_files += 1
        for props, pts in parsed:
            # dedupe segments shared across sheet boundaries
            key = (next(iter(props.values())), round(pts[0][0], 2),
                   round(pts[0][1], 2), round(pts[-1][0], 2),
                   round(pts[-1][1], 2), len(pts))
            if key in seen:
                continue
            seen.add(key)
            p = dict(props)
            p["sheet"] = sheet
            feats2326.append({"type": "Feature", "properties": p,
                              "geometry": mapping(LineString(pts))})
            arr = np.asarray(pts)
            lon, lat = tr.transform(arr[:, 0], arr[:, 1])
            feats84.append({"type": "Feature", "properties": p,
                            "geometry": {"type": "LineString",
                                         "coordinates": np.column_stack(
                                             [lon, lat]).tolist()}})
    for path, feats in (
            (OUT / f"landsd_shoreline_{product}_epsg2326.geojson", feats2326),
            (OUT / f"landsd_shoreline_{product}_wgs84.geojson", feats84)):
        path.write_text(json.dumps({"type": "FeatureCollection",
                                    "features": feats}))
    print(f"  parsed {n_files} shoreline-bearing files, "
          f"{len(feats2326)} unique linestrings")
    return feats2326


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", choices=sorted(PRODUCTS), required=True)
    ap.add_argument("--buffer-m", type=float, default=1000.0,
                    help="buffer around OSM coastline for sheet selection")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    product = args.product
    prod = PRODUCTS[product]
    sheets_dir = OUT / f"{product}_sheets"
    tileindex = OUT / f"{product}_tileindex.geojson"
    selection_csv = OUT / f"{product}_sheet_selection.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    if tileindex.exists():
        print(f"using cached tile index: {tileindex}")
        feats = json.loads(tileindex.read_text())["features"]
    else:
        print(f"fetching {product} tile index from CSDI FeatureServer ...")
        feats = fetch_tileindex(prod, tileindex)
    print(f"tile index: {len(feats)} sheets")

    sel = select_sheets(feats, args.buffer_m, selection_csv)
    print(f"selected {len(sel)} coastal sheets (buffer {args.buffer_m} m)")

    def cached(s):
        p = sheets_dir / f"{s}.zip"
        return p.exists() and zipfile.is_zipfile(p)

    todo = sel[~sel["sheet"].apply(cached)]
    print(f"to download: {len(todo)} (rest cached)")

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_sheet, sheets_dir, r.sheet, r.url): r.sheet
                for r in todo.itertuples()}
        done = 0
        for fut in as_completed(futs):
            sheet, ok, msg = fut.result()
            done += 1
            if not ok:
                failures.append((sheet, msg))
                print(f"  FAIL {sheet}: {msg}")
            if done % 50 == 0:
                print(f"  downloaded {done}/{len(todo)}")
    print(f"download complete, {len(failures)} failures")
    if failures:
        print("failures:", failures[:20])

    ok_sheets = [s for s in sel["sheet"] if cached(s)]
    build_merged_shoreline(prod, product, sheets_dir, ok_sheets)
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
