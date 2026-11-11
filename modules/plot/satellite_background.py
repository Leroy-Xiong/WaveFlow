"""Satellite background for wave maps.

Adds high-resolution satellite imagery as the map background with the ocean
masked out (painted white) so only land shows satellite detail. The ocean
mask and coastline come from the SCHISM model's own boundary polylines
(mainland + islands, extracted from hgrid.gr3 — the same mesh that defines
the wave nodes), so the coastline is an exact vector polyline in the model's
coordinate system rather than an external shoreline dataset or a heuristic
buffer around the nodes.
"""
import os

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

import io
import time
import urllib.error
import urllib.request


class _RetryGoogleTiles(cimgt.GoogleTiles):
    """GoogleTiles with retry-on-failure for each tile download.

    Cartopy's built-in ``get_image`` catches all download errors and
    returns a blank gray tile — that produces the white squares.  This
    subclass retries the HTTP fetch up to 4 times (with back-off) before
    falling back to the gray placeholder.
    """

    def get_image(self, tile):
        from urllib.request import HTTPError, Request, URLError, urlopen
        from PIL import Image
        import numpy as np

        if self.cache_path is not None:
            filename = "_".join([str(i) for i in tile]) + ".npy"
            cached_file = self._cache_dir / filename
        else:
            cached_file = None

        if cached_file in self.cache:
            img = np.load(cached_file, allow_pickle=False)
        else:
            url = self._image_url(tile)
            im_data = None
            for attempt in range(4):
                try:
                    request = Request(url, headers={"User-Agent": self.user_agent})
                    fh = urlopen(request, timeout=30)
                    im_data = io.BytesIO(fh.read())
                    fh.close()
                    break
                except (HTTPError, URLError, OSError, ConnectionResetError,
                        TimeoutError) as err:
                    if attempt == 3:
                        print(f'Tile download failed after 4 retries: {err}')
                    else:
                        time.sleep(1.0 * (attempt + 1))
            if im_data is not None:
                img = Image.open(im_data)
            else:
                img = Image.fromarray(np.full((256, 256, 3), (250, 250, 250),
                                              dtype=np.uint8))

            img = img.convert(self.desired_tile_form)
            if self.cache_path is not None:
                np.save(cached_file, img, allow_pickle=False)
                self.cache.add(cached_file)

        return img, self.tileextent(tile), 'lower'


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Persist downloaded map tiles inside the (git-ignored) project data dir —
# the cartopy default is /tmp, which is wiped on reboot.
import cartopy
cartopy.config["cache_dir"] = os.path.join(_ROOT, "data", "cartopy_cache")

# Esri World Imagery (Maxar) — highest-resolution freely available global
# satellite tiles. The ArcGIS tile service uses z/y/x ordering.
# cache=True: persist downloaded tiles on disk (cartopy cache dir), so
# repeat figure generations load tiles locally instead of re-downloading
# thousands of tiles over HTTP on every run.
_TILES = _RetryGoogleTiles(
    url='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/'
        'MapServer/tile/{z}/{y}/{x}',
    cache=True,
)

# Tile zoom level (higher = finer detail, more tiles). Cartopy caches tiles
# on disk, so the cost is paid once per region/zoom.
ZOOM = 13

_BOUNDARIES_PATH = os.path.join(
    _ROOT, 'data', 'hindcast', 'land_boundaries', 'boundaries.npz'
)

# Ocean geometry cache — identical for every figure of the same region.
_ocean_cache = {}


# ---------------------------------------------------------------------------
# Warped-tile caches.  Fetching + warping the tile mosaic is the dominant
# cost of these figures (tens of seconds per axes), and every map of the
# same region/projection needs the identical warped image — so warp ONCE
# per (region | projection, zoom, resolution), cache the result on disk
# (survives across runs), and let each axes do a cheap same-projection
# imshow (cartopy skips re-warping when image transform == axes proj).
# ---------------------------------------------------------------------------
_WARPED_DIR = os.path.join(cartopy.config["cache_dir"], "warped")
_warped_cache = {}


def _load_warped(key):
    if key in _warped_cache:
        return _warped_cache[key]
    path = os.path.join(_WARPED_DIR, key + ".npy")
    if os.path.exists(path):
        img = np.load(path)
        _warped_cache[key] = img
        return img
    return None


def _store_warped(key, img):
    os.makedirs(_WARPED_DIR, exist_ok=True)
    np.save(os.path.join(_WARPED_DIR, key + ".npy"), img)
    _warped_cache[key] = img
    return img


def warped_tiles_region(region, zoom=ZOOM, nx=1200):
    """Satellite tile mosaic of ``region`` (min_lon, max_lon, min_lat,
    max_lat), warped once to PlateCarree.  Returns (img, extent) with
    img row 0 at the southern edge (imshow origin='lower')."""
    from cartopy.img_transform import warp_array

    min_lon, max_lon, min_lat, max_lat = [round(float(v), 6) for v in region]
    ny = max(2, int(round(nx * (max_lat - min_lat) / (max_lon - min_lon))))
    key = f"region_z{zoom}_{min_lon}_{max_lon}_{min_lat}_{max_lat}_{nx}x{ny}"
    img = _load_warped(key)
    if img is not None:
        return img, (min_lon, max_lon, min_lat, max_lat)

    merc = _TILES.crs
    domain = merc.project_geometry(
        box(min_lon, min_lat, max_lon, max_lat), ccrs.PlateCarree())
    mosaic, src_ext, _ = _TILES.image_for_domain(domain, zoom)
    img, _ = warp_array(np.asarray(mosaic), target_proj=ccrs.PlateCarree(),
                        source_proj=merc, target_res=(nx, ny),
                        source_extent=src_ext,
                        target_extent=(min_lon, max_lon, min_lat, max_lat))
    img = np.ma.filled(img, 255)
    return _store_warped(key, img), (min_lon, max_lon, min_lat, max_lat)


def warped_tiles_global(proj, zoom, nx=1500):
    """Global satellite tile mosaic warped once to ``proj`` (whole-globe
    extent).  Returns (img, extent), img row 0 at the bottom
    (imshow origin='lower')."""
    from cartopy.img_transform import warp_array

    ny = nx // 2
    # Projection parameters (e.g. the orthographic centre) are part of the
    # warped image's identity — two globes with different centres must not
    # share a cache entry.
    _params = "_".join(f"{k}{v}" for k, v in sorted(proj.proj4_params.items())
                       if k not in ("globe", "datum", "ellps", "a", "b",
                                    "units", "no_defs"))
    key = f"global_z{zoom}_{type(proj).__name__}_{nx}x{ny}_{_params}"
    # mesh_projection's default target domain == the projection limits
    ext = (proj.x_limits[0], proj.x_limits[1],
           proj.y_limits[0], proj.y_limits[1])
    img = _load_warped(key)
    if img is not None:
        return img, ext

    merc = _TILES.crs
    # tiles stop at ~85.05 deg N/S; stay just inside
    y_max = merc.transform_point(0.0, 85.0, ccrs.PlateCarree())[1]
    domain = box(merc.x_limits[0], -y_max, merc.x_limits[1], y_max)
    mosaic, src_ext, _ = _TILES.image_for_domain(domain, zoom)
    img, _ = warp_array(np.asarray(mosaic), target_proj=proj,
                        source_proj=merc, target_res=(nx, ny),
                        source_extent=src_ext,
                        mask_extrapolated=True)
    img = np.ma.filled(img, 255)
    return _store_warped(key, img), ext


def _segments(coords, offsets):
    return [coords[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


_HK_COORD_PATH = os.path.join(_ROOT, 'data', 'hindcast', 'hk_coordinates.npy')
_HK_OCEAN_WKB = os.path.join(_ROOT, 'data', 'hindcast',
                             'hk_ocean_boundary_wkb.npz')

# GSHHS full-resolution shoreline (same data source the UQ figures use via
# cfeature.GSHHSFeature; cartopy's own download URL 404s, so the shapefile is
# fetched once from the SOEST mirror
# https://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip and kept here).
_GSHHS_SHP = os.path.join(_ROOT, 'data', 'hindcast', 'land_boundaries',
                          'GSHHS_f_L1.shp')
# Land polygons clipped to a generous HK domain, cached as WKB — reading the
# 160 MB shapefile takes ~1 min, the cache makes it instant.
_GSHHS_LAND_WKB = os.path.join(_ROOT, 'data', 'hindcast', 'land_boundaries',
                               'gshhs_land_wkb.npz')
_GSHHS_DOM = (112.0, 20.5, 116.5, 24.2)      # min_lon, min_lat, max_lon, max_lat


def _gshhs_land():
    """GSHHS full-res land polygons near HK (list of shapely polygons).

    Returns None when neither the cache nor the raw shapefile is available.
    """
    from shapely import from_wkb
    if os.path.exists(_GSHHS_LAND_WKB):
        d = np.load(_GSHHS_LAND_WKB, allow_pickle=True)
        return [from_wkb(w) for w in d['land_wkb']]
    if not os.path.exists(_GSHHS_SHP):
        return None
    from cartopy.io.shapereader import BasicReader
    from shapely import to_wkb
    dom = box(_GSHHS_DOM[0], _GSHHS_DOM[1], _GSHHS_DOM[2], _GSHHS_DOM[3])
    polys = []
    for g in BasicReader(_GSHHS_SHP).geometries():
        if g.intersects(dom):
            polys.append(g.intersection(dom))
    try:
        np.savez(_GSHHS_LAND_WKB,
                 land_wkb=np.array([to_wkb(p) for p in polys], dtype=object))
    except Exception as e:
        print(f"Warning: failed to cache GSHHS land polygons: {e}")
    return polys


def _node_derived_ocean():
    """Rebuild the ocean polygon from the wave-node point cloud itself.

    Fallback for machines without ``boundaries.npz`` (the hgrid.gr3
    boundary polylines).  Same recipe as the global ocean boundary
    (modules/plot/global_ocean_boundary.py): Delaunay-triangulate the
    nodes, drop "bridge" triangles whose longest edge far exceeds the
    local node spacing (these span land, channels and islands where the
    mesh has no nodes), and union the kept triangles.  The result hugs
    the model's own mesh boundary, so it is far more faithful than any
    external shoreline dataset.  Cached as WKB next to the coordinates.
    """
    from shapely import from_wkb
    if os.path.exists(_HK_OCEAN_WKB):
        try:
            d = np.load(_HK_OCEAN_WKB, allow_pickle=True)
            return from_wkb(d['ocean_wkb'][0])
        except Exception as e:
            print(f"Warning: failed to load HK ocean cache, recomputing: {e}")
    from scipy.spatial import Delaunay
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from modules.plot.global_ocean_boundary import (
        F_BRIDGE, _great_circle_km, _nearest_neighbor_km)

    coords = np.load(_HK_COORD_PATH)                 # (N, 2) lat/lon
    lats = coords[:, 0].astype(np.float64)
    lons = coords[:, 1].astype(np.float64)
    pts = np.column_stack([lons, lats])
    nn_km = _nearest_neighbor_km(lats, lons)
    sim = Delaunay(pts).simplices
    v = pts[sim]                                     # (T, 3, 2) lon/lat
    e0 = _great_circle_km(v[:, 0, 0], v[:, 0, 1], v[:, 1, 0], v[:, 1, 1])
    e1 = _great_circle_km(v[:, 1, 0], v[:, 1, 1], v[:, 2, 0], v[:, 2, 1])
    e2 = _great_circle_km(v[:, 2, 0], v[:, 2, 1], v[:, 0, 0], v[:, 0, 1])
    max_edge = np.maximum(np.maximum(e0, e1), e2)
    local_spacing = np.max(nn_km[sim], axis=1)
    keep = max_edge < F_BRIDGE * local_spacing
    ocean = unary_union([Polygon(pts[s]) for s in sim[keep]])
    print(f"HK ocean boundary rebuilt from {len(lats)} nodes "
          f"({keep.sum()}/{len(keep)} triangles kept)")
    try:
        from shapely import to_wkb
        np.savez(_HK_OCEAN_WKB,
                 ocean_wkb=np.array([to_wkb(ocean)], dtype=object))
    except Exception as e:
        print(f"Warning: failed to cache HK ocean boundary: {e}")
    return ocean


def _load_ocean_geometry(region):
    """Build the ocean polygon and coastline polyline from the model's
    boundary segments, clipped to ``region`` (min_lon, max_lon, min_lat,
    max_lat). The ocean is the model domain (mainland + open boundary ring)
    with the island polygons removed.  When the hgrid boundary file is not
    available, the GSHHS full-resolution shoreline is used instead (the same
    data source as the UQ figures); the wave-node Delaunay reconstruction
    is only a last resort when GSHHS is missing too.
    """
    key = tuple(region)
    if key not in _ocean_cache:
        min_lon, max_lon, min_lat, max_lat = region
        clip = box(min_lon, min_lat, max_lon, max_lat)
        if os.path.exists(_BOUNDARIES_PATH):
            d = np.load(_BOUNDARIES_PATH)
            mainland = _segments(d['mainland_coords'], d['mainland_offsets'])
            opens = _segments(d['open_coords'], d['open_offsets'])
            islands = [
                Polygon(ring) for ring in _segments(d['island_coords'], d['island_offsets'])
            ]

            # The mainland and open segments connect end-to-end into the closed
            # outer ring of the model domain (interior = water).
            ring = np.vstack([mainland[0], opens[1], mainland[1], opens[2],
                              mainland[2], opens[0]])
            domain = Polygon(ring)
            if not domain.is_valid:
                domain = domain.buffer(0)
            islands = [p if p.is_valid else p.buffer(0) for p in islands]
            ocean = domain.difference(unary_union(islands))
            _ocean_cache[key] = (ocean.intersection(clip),
                                 ocean.boundary.intersection(clip))
        else:
            land = _gshhs_land()
            if land is not None:
                # Real high-res shoreline (same source as the UQ figures):
                # ocean = map extent minus GSHHS land.  The node-derived
                # Delaunay boundary is NOT trustworthy near the region edge
                # and in sparsely-meshed bays — it keeps bridge triangles
                # that cut straight across peninsulas.
                land_u = unary_union(land)
                _ocean_cache[key] = (clip.difference(land_u),
                                     land_u.boundary.intersection(clip))
            else:
                ocean = _node_derived_ocean()
                _ocean_cache[key] = (ocean.intersection(clip),
                                     ocean.boundary.intersection(clip))
    return _ocean_cache[key]


def add_satellite_background(ax, region, zoom=ZOOM, ocean_color='white'):
    """Draw satellite imagery with the model ocean cut out.

    The satellite image (warped to PlateCarree once per region/zoom and
    cached on disk) is drawn first, the model ocean (from the SCHISM
    boundary polylines) is painted over it in ``ocean_color``, and the
    model coastline is drawn on top as a fine vector polyline.
    """
    img, ext = warped_tiles_region(region, zoom)
    ax.imshow(img, extent=ext, origin='lower',
              transform=ccrs.PlateCarree(), interpolation='bilinear',
              zorder=1)

    ocean, coastline = _load_ocean_geometry(region)
    ax.add_geometries(
        [ocean], crs=ccrs.PlateCarree(),
        facecolor=ocean_color, edgecolor='none', zorder=2,
    )
    ax.add_geometries(
        [coastline], crs=ccrs.PlateCarree(),
        facecolor='none', edgecolor='black', linewidth=0.6, zorder=4,
    )


# Zoom level for full-globe views (Robinson / Mollweide).
GLOBAL_ZOOM = 6

# Fill color for the polar gaps: web-mercator satellite tiles only reach
# about +/-85.05 deg, anything poleward of that has no imagery.  Sampled
# from the Antarctic ice in the Esri imagery (median of an ice patch);
# used as fallback when the mirrored tile strip cannot be fetched.
POLAR_GAP_FILL = '#ECF0FA'

# Cache for the mirrored polar strips, keyed by (zoom, south).
_polar_strip_cache = {}


def _polar_mirror_strip(tiles, zoom, south):
    """Imagery strip used to fill the polar tile gap seamlessly.

    Takes the southernmost (or northernmost) tile row at ``zoom``, keeps
    the half nearest the web-mercator limit (~2.2 deg of imagery, deep
    ice sheet / sea ice, no coastline) and flips it vertically.  Stretched
    over the 85.05..90 deg gap it mirrors the real imagery across the
    tile boundary, so the transition at 85.05 deg is seamless.
    """
    key = (zoom, south)
    if key not in _polar_strip_cache:
        n = 2 ** zoom
        row = n - 1 if south else 0
        imgs = []
        for x in range(n):
            img, _, _ = tiles.get_image((x, row, zoom))
            # 缓存命中时 get_image 返回 ndarray (np.load), 只有 PIL Image
            # 才有 .convert
            if isinstance(img, np.ndarray):
                imgs.append(img)
            else:
                imgs.append(np.asarray(img.convert('RGB')))
        strip = np.hstack(imgs)
        half = strip.shape[0] // 2
        strip = strip[half:] if south else strip[:half]
        _polar_strip_cache[key] = strip[::-1]
    return _polar_strip_cache[key]


def add_satellite_background_global(ax, zoom=GLOBAL_ZOOM):
    """Draw satellite imagery as the background for a global map.

    No ocean masking is applied — the data overlay (scatter / pcolormesh /
    contourf) covers the ocean area. Coastlines should be added separately
    by the caller for geographic context.  Poleward of the web-mercator
    limit (~85.05 deg) where no tiles exist, the nearest tile row is
    mirrored across the boundary and stretched over the gap (seamless
    ice fill), with a flat ice color underneath as fallback.
    """
    polar_gaps = [box(-180.0, -90.0, 180.0, -85.05),
                  box(-180.0, 85.05, 180.0, 90.0)]
    ax.add_geometries(polar_gaps, crs=ccrs.PlateCarree(),
                      facecolor=POLAR_GAP_FILL, edgecolor='none', zorder=0.5)
    try:
        for ymin, ymax, south in ((-90.0, -85.05, True), (85.05, 90.0, False)):
            strip = _polar_mirror_strip(_TILES, zoom, south)
            ax.imshow(strip, extent=[-180.0, 180.0, ymin, ymax],
                      origin='upper', transform=ccrs.PlateCarree(),
                      interpolation='bilinear', zorder=0.6)
    except Exception as e:
        print(f"Warning: polar mirror fill unavailable, using flat color: {e}")
    # warp the global tile mosaic to this axes' projection once (cached on
    # disk), then a cheap same-projection imshow
    img, ext = warped_tiles_global(ax.projection, zoom)
    ax.imshow(img, extent=ext, origin='lower', transform=ax.projection,
              interpolation='bilinear', zorder=1)
