"""Extract a precise global ocean polygon / coastline from the unstructured
wave-model node coordinates (SCHISM/WWM grid, ocean-only nodes).

The source netCDF has no connectivity table, so the boundary is recovered
geometrically from the point cloud:

1. Longitudes are mapped to [0, 360).  To avoid an artificial cut at the
   0/360 seam, nodes with lon < 30 are duplicated at +360 and nodes with
   lon > 330 are duplicated at -360 ("ghost" nodes), and a 2D Delaunay
   triangulation of the extended point set is built.
2. Each physical triangle near the seam exists twice in the plane (once per
   ghost-band interface), so for all counting purposes only triangles whose
   centroid longitude falls in [0, 360) are used - this counts every
   physical triangle exactly once and automatically discards the pure-ghost
   copies and the artificial hull edges at the ghost-band cuts.
3. Triangles whose longest great-circle edge exceeds ``F_BRIDGE`` times the
   local node spacing (maximum nearest-neighbour distance of the triangle's
   vertices; max is used so that legitimate triangles at resolution
   transitions survive) are dropped - these are "bridge" triangles spanning
   land, bays or islands where no nodes exist.
4. Boundary edges are edges referenced by exactly one kept triangle
   (ghost vertices are mapped back to their real node id).
5. Edges are chained into polylines and longitudes are unwrapped along each
   polyline.  Closed loops that encircle a pole (longitude winding +-360)
   cannot be represented as a plane ring; they are closed explicitly via
   the nearest pole.  All other lines are fed to ``shapely.polygonize``.
6. The ocean is the unbounded face of this arrangement (the node set spans
   all longitudes, so its outer "boundary" is just the domain edge), hence
   ``ocean = box(0,-90,360,90) - land`` where land is the union of all
   polygonized faces whose node density is implausibly low for the mesh
   (continents, islands, and bays closed off by dropped bridge triangles).
   Faces with enough nodes (e.g. inland seas covered by the grid) stay
   ocean.  Beyond |lat| 60 the sparse mesh makes density unreliable (real
   islands merge with channels and contain a few nodes; mesh gaps are
   node-free but have the same size/compactness as islands), so those
   faces are instead classified by their area overlap with Natural Earth
   10m land (land iff overlap fraction >= 0.5); the displayed coastline
   always remains the node-derived boundary.  Pole-cap polygons (beyond
   the outermost node ring) are always land, since NE does not model the
   Antarctic ice shelves consistently.
7. The result is split at the dateline and longitudes > 180 are shifted by
   -360, yielding a MultiPolygon in [-180, 180] with no ring crossing the
   dateline.  ``coastline = ocean.boundary``.

Results are cached as WKB in ``data/global_hindcast/ocean_boundary_wkb.npz``.
"""

import os

import numpy as np

# Bridge-triangle rejection factor: a triangle is kept only if its longest
# edge (great-circle km) is shorter than F_BRIDGE times the local spacing.
F_BRIDGE = 3.0

# Half-width (degrees) of the ghost bands used to close the 0/360 seam.
GHOST_BAND = 30.0

_EARTH_RADIUS_KM = 6371.0

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(_MODULE_DIR)),
                         'data', 'global_hindcast')
_COORD_PATH = os.path.join(_DATA_DIR, 'global_coordinates.npy')
_CACHE_PATH = os.path.join(_DATA_DIR, 'ocean_boundary_wkb.npz')

# In-process cache: (ocean_geometry, coastline_geometry)
_geometry_cache = None


def get_global_ocean_geometry():
    """Return ``(ocean_geometry, coastline_geometry)`` as shapely
    MultiPolygon / MultiLineString in lon/lat within [-180, 180].

    The result is computed once and cached on disk
    (``data/global_hindcast/ocean_boundary_wkb.npz``) and in memory.
    """
    global _geometry_cache
    if _geometry_cache is not None:
        return _geometry_cache
    from shapely import from_wkb
    if os.path.exists(_CACHE_PATH):
        try:
            data = np.load(_CACHE_PATH, allow_pickle=True)
            ocean = from_wkb(data['ocean_wkb'][0])
            coast = from_wkb(data['coast_wkb'][0])
            _geometry_cache = (ocean, coast)
            return _geometry_cache
        except Exception as e:
            print(f"Warning: failed to load ocean boundary cache, recomputing: {e}")
    ocean, coast = _compute_global_ocean_geometry()
    try:
        from shapely import to_wkb
        np.savez(_CACHE_PATH,
                 ocean_wkb=np.array([to_wkb(ocean)], dtype=object),
                 coast_wkb=np.array([to_wkb(coast)], dtype=object))
        print(f"Ocean boundary geometry cached to: {_CACHE_PATH}")
    except Exception as e:
        print(f"Warning: failed to cache ocean boundary geometry: {e}")
    _geometry_cache = (ocean, coast)
    return _geometry_cache


def _great_circle_km(lon1, lat1, lon2, lat2):
    """Vectorised haversine distance (km); inputs in degrees."""
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat / 2) ** 2 + \
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _nearest_neighbor_km(lats, lons):
    """Nearest-neighbour distance (km) of each node on the unit sphere."""
    from scipy.spatial import cKDTree
    phi = np.radians(lats)
    lam = np.radians(lons)
    pts3d = np.column_stack([np.cos(phi) * np.cos(lam),
                             np.cos(phi) * np.sin(lam),
                             np.sin(phi)])
    tree = cKDTree(pts3d)
    chord, _ = tree.query(pts3d, k=2)
    chord = chord[:, 1]
    # chord length -> great-circle distance
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2, 0, 1))


def _chain_edges_into_lines(edge_list):
    """Assemble boundary edges (list of (i, j) node-id pairs) into polylines.

    Returns a list of polylines, each a list of node ids in walking order.
    """
    from collections import defaultdict
    adjacency = defaultdict(list)  # node id -> list of edge indices
    for ei, (a, b) in enumerate(edge_list):
        adjacency[a].append(ei)
        adjacency[b].append(ei)
    used = np.zeros(len(edge_list), dtype=bool)
    lines = []
    # Start open chains from degree-1 nodes so rings are not split first.
    starts = [n for n, e in adjacency.items() if len(e) == 1]
    starts += [n for n, e in adjacency.items() if len(e) > 1]
    for start in starts:
        while True:
            free = [ei for ei in adjacency[start] if not used[ei]]
            if not free:
                break
            line = [start]
            ei = free[0]
            used[ei] = True
            a, b = edge_list[ei]
            cur = b if a == start else a
            line.append(cur)
            while True:
                nxt = [e2 for e2 in adjacency[cur] if not used[e2]]
                if not nxt:
                    break
                e2 = nxt[0]
                used[e2] = True
                a, b = edge_list[e2]
                cur = b if a == cur else a
                line.append(cur)
            lines.append(line)
    return lines


def _build_node_cells(lon360, lats):
    """Bin node indices into 1x1 degree cells for fast bbox lookups."""
    from collections import defaultdict
    cells = defaultdict(list)
    ilon = np.floor(lon360).astype(int)
    ilat = np.clip(np.floor(lats).astype(int), -90, 89)
    for i in range(len(lon360)):
        cells[(ilon[i], ilat[i])].append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in cells.items()}


def _polygon_node_count(poly, lon360, lats, cells, max_count):
    """Count nodes inside ``poly`` (unwrapped-longitude frame), stopping
    early once ``max_count`` is reached."""
    from shapely import contains_xy
    minx, miny, maxx, maxy = poly.bounds
    cx = 0.5 * (minx + maxx)
    ilat0 = max(-90, int(np.floor(miny)))
    ilat1 = min(89, int(np.floor(maxy)))
    count = 0
    for ilon_abs in range(int(np.floor(minx)), int(np.floor(maxx)) + 1):
        ilon = ilon_abs % 360
        for ilat in range(ilat0, ilat1 + 1):
            idx = cells.get((ilon, ilat))
            if idx is None:
                continue
            # shift node longitudes into the polygon's frame
            slon = cx + ((lon360[idx] - cx + 180.0) % 360.0 - 180.0)
            sel = (slon >= minx) & (slon <= maxx)
            if not sel.any():
                continue
            idx2 = idx[sel]
            count += int(contains_xy(poly, slon[sel], lats[idx2]).sum())
            if count >= max_count:
                return count
    return count


# A face is ocean only if its node density is plausible for the wave mesh
# (global average ~14 nodes/deg^2; the coarsest open-ocean regions still
# have >1 node/deg^2).  This makes the classification robust to a handful
# of nodes accidentally enclosed by a huge land face through spur edges.
_OCEAN_NODE_DENSITY = 0.1  # nodes per deg^2

# Beyond this |latitude|, faces are classified by their overlap with
# Natural Earth 10m land instead of node density (see step 6 above).
# 60 degrees covers the coarsely-meshed polar oceans (Kara/Barents/
# Beaufort seas, Hudson Bay, the Arctic Archipelago, the Southern Ocean
# and the Antarctic shelf seas).
_POLAR_REF_LAT = 60.0
# Land-fraction threshold: measured 0.57-0.94 for real islands vs
# 0.000-0.054 for mesh gaps / sea faces.
_POLAR_LAND_FRAC = 0.5


def _ne_land_fractions(faces, skip_ids=()):
    """Return ``{id(face): area fraction of the face covered by Natural
    Earth 10m land}`` for faces with |centroid latitude| north/south of
    ``_POLAR_REF_LAT`` (faces in ``skip_ids`` are ignored).

    Face frames may be unwrapped (e.g. lon 340..410), so intersections are
    accumulated over the three +-360 degree shifts of each face.
    """
    import cartopy.io.shapereader as shpreader
    from shapely import STRtree
    from shapely.ops import transform as shp_transform
    path = shpreader.natural_earth(resolution='10m', category='physical',
                                   name='land')
    polys = [rec.geometry for rec in shpreader.Reader(path).records()]
    tree = STRtree(polys)
    fracs = {}
    for face in faces:
        if id(face) in skip_ids or abs(face.centroid.y) <= _POLAR_REF_LAT:
            continue
        total = 0.0
        for s in (0.0, 360.0, -360.0):
            f2 = shp_transform(
                lambda x, y, z=None, s=s: (x + s, y) if z is None
                else (x + s, y, z), face)
            if f2.bounds[2] < -180.0 or f2.bounds[0] > 180.0:
                continue
            for gi in tree.query(f2):
                g = polys[gi]
                if g.intersects(f2):
                    total += g.intersection(f2).area
        fracs[id(face)] = total / face.area if face.area > 0 else 0.0
    return fracs


def _compute_global_ocean_geometry():
    import time
    from scipy.spatial import Delaunay
    from shapely import LineString, Polygon, box, polygonize, unary_union
    from shapely.ops import transform as shp_transform

    t0 = time.time()
    coords = np.load(_COORD_PATH)
    lats = coords[:, 0].astype(np.float64)
    lons = coords[:, 1].astype(np.float64)
    lon360 = np.mod(lons, 360.0)
    n = len(lats)
    print(f"Building global ocean boundary from {n} nodes ...")

    # ---- 1. ghost nodes to close the 0/360 seam + Delaunay ----
    low = lon360 < GHOST_BAND
    high = lon360 > 360.0 - GHOST_BAND
    ext_lon = np.concatenate([lon360, lon360[low] + 360.0, lon360[high] - 360.0])
    ext_lat = np.concatenate([lats, lats[low], lats[high]])
    ext_id = np.concatenate([np.arange(n), np.nonzero(low)[0], np.nonzero(high)[0]])
    ext_pts = np.column_stack([ext_lon, ext_lat])

    nn_km = _nearest_neighbor_km(lats, lons)
    tri = Delaunay(ext_pts)
    simplices = tri.simplices
    print(f"  Delaunay: {len(simplices)} triangles ({time.time() - t0:.1f}s)")

    # ---- 2. keep each physical triangle exactly once ----
    # Ghost bands duplicate lon [0,30) at [360,390) and (330,360] at
    # [-30,0); every physical triangle near the seam therefore appears at
    # two plane locations.  Counting only triangles whose centroid lies in
    # [0,360) keeps exactly one copy and also discards the artificial hull
    # triangles at the ghost-band cuts (lon ~ -30 and ~ 390).
    tri_clon = ext_lon[simplices].mean(axis=1)
    in_range = (tri_clon >= 0.0) & (tri_clon < 360.0)
    v_id = ext_id[simplices]
    dup = (v_id[:, 0] == v_id[:, 1]) | (v_id[:, 0] == v_id[:, 2]) | \
          (v_id[:, 1] == v_id[:, 2])

    # ---- 3. drop bridge triangles (longest edge >> local spacing) ----
    tri_lon = ext_lon[simplices]
    tri_lat = ext_lat[simplices]
    e0 = _great_circle_km(tri_lon[:, 0], tri_lat[:, 0], tri_lon[:, 1], tri_lat[:, 1])
    e1 = _great_circle_km(tri_lon[:, 1], tri_lat[:, 1], tri_lon[:, 2], tri_lat[:, 2])
    e2 = _great_circle_km(tri_lon[:, 2], tri_lat[:, 2], tri_lon[:, 0], tri_lat[:, 0])
    max_edge = np.maximum(np.maximum(e0, e1), e2)
    # Use the *max* vertex spacing as reference: at resolution transitions
    # (fine coastal nodes meeting coarse offshore nodes) legitimate mesh
    # triangles have edges much longer than the finer side's spacing, so a
    # median/min reference would drop them and create spurious boundary
    # spurs.  Bridges across land connect nodes that are *all* locally
    # fine-spaced, so their long edges still exceed F_BRIDGE x max spacing.
    local_spacing = np.max(nn_km[v_id], axis=1)
    keep = in_range & ~dup & (max_edge < F_BRIDGE * local_spacing)
    kept_id = v_id[keep]
    print(f"  kept {keep.sum()} / {len(keep)} triangles "
          f"(f={F_BRIDGE}, {time.time() - t0:.1f}s)")

    # ---- 4. boundary edges = referenced exactly once ----
    e_a = np.concatenate([kept_id[:, 0], kept_id[:, 1], kept_id[:, 2]])
    e_b = np.concatenate([kept_id[:, 1], kept_id[:, 2], kept_id[:, 0]])
    lo_id = np.minimum(e_a, e_b)
    hi_id = np.maximum(e_a, e_b)
    edge_key = lo_id.astype(np.int64) * np.int64(n) + hi_id.astype(np.int64)
    uniq, counts = np.unique(edge_key, return_counts=True)
    if (counts > 2).any():
        print(f"  warning: {(counts > 2).sum()} non-manifold edges (count>2)")
    b_keys = uniq[counts == 1]
    b_lo = (b_keys // np.int64(n)).astype(np.int64)
    b_hi = (b_keys % np.int64(n)).astype(np.int64)
    edge_list = list(zip(b_lo.tolist(), b_hi.tolist()))
    print(f"  boundary edges: {len(edge_list)} ({time.time() - t0:.1f}s)")

    # ---- 5. chain edges into polylines, unwrap longitudes ----
    lines = _chain_edges_into_lines(edge_list)
    geoms = []
    pole_polys = []
    for line in lines:
        if len(line) < 2:
            continue
        xs = lon360[np.asarray(line)]
        ys = lats[np.asarray(line)]
        # unwrap: keep each longitude within +/-180 of the previous one
        xu = np.empty_like(xs)
        xu[0] = xs[0]
        for k in range(1, len(xs)):
            d = (xs[k] - xu[k - 1] + 180.0) % 360.0 - 180.0
            xu[k] = xu[k - 1] + d
        if line[0] == line[-1] and abs(xu[-1] - xu[0]) > 180.0:
            # Closed loop encircling a pole: not representable as a plane
            # ring (longitude winds +-360).  Close it via the nearest pole.
            # Re-start the ring at its southernmost (resp. northernmost)
            # point so the vertical closure segments towards the pole can
            # never cross the ring or cut through ocean on the side.
            pole_lat = 90.0 if ys.mean() > 0 else -90.0
            w = 360.0 if xu[-1] > xu[0] else -360.0
            k0 = int(np.argmax(ys)) if pole_lat > 0 else int(np.argmin(ys))
            xs = np.roll(xs[:-1], -k0)
            ys = np.roll(ys[:-1], -k0)
            xu = np.empty_like(xs)
            xu[0] = xs[0]
            for k in range(1, len(xs)):
                d = (xs[k] - xu[k - 1] + 180.0) % 360.0 - 180.0
                xu[k] = xu[k - 1] + d
            ring = list(zip(xu.tolist(), ys.tolist()))
            # Close the loop exactly on the start meridian.  The ring's two
            # end nodes differ slightly in longitude (~0.5 deg); if the
            # polygon were closed between the *actual* end point and the
            # start point, the cap would leave a thin sliver between their
            # two meridians uncovered (a spurious ocean wedge reaching the
            # pole).  Appending the start point at +w makes both closure
            # segments share one meridian so they coincide after folding.
            ring.append((xu[0] + w, ys[0]))
            ring += [(xu[0] + w, pole_lat), (xu[0], pole_lat)]
            try:
                pole_polys.append(Polygon(ring).buffer(0))
            except Exception as ex:
                print(f"  warning: failed to build pole polygon: {ex}")
            continue
        geoms.append(LineString(np.column_stack([xu, ys])))
    print(f"  polylines: {len(lines)}, pole-encircling loops: {len(pole_polys)} "
          f"({time.time() - t0:.1f}s)")

    # ---- 6. faces -> land = faces without nodes; ocean = box - land ----
    # polygonize needs fully noded input (junction nodes where dropped
    # bridge triangles meet the coast are shared by 3+ lines), otherwise it
    # silently drops most faces with an "invalid value" warning.
    from shapely import get_parts, union_all
    noded = [p for p in get_parts(union_all(geoms))
             if p.geom_type in ('LineString', 'LinearRing')]
    faces = list(polygonize(noded).geoms) + pole_polys
    print(f"  faces: {len(faces)} ({time.time() - t0:.1f}s)")
    cells = _build_node_cells(lon360, lats)
    # Pole-cap polygons (beyond the outermost node ring, node-free by
    # construction) are always land; NE does not model the Antarctic ice
    # shelves consistently (Ross shelf is "water", Filchner is "land"), so
    # they must not go through the NE arbitration below.
    pole_ids = {id(p) for p in pole_polys}
    # In sparsely-meshed polar regions (|lat| > 60) node density alone
    # cannot distinguish real islands (some merge with channels and contain
    # a few nodes) from gaps in the coarse mesh (node-free, but with the
    # same size/compactness as islands - measured: both span 0.2-16 deg^2
    # with perimeter^2/area 15-50).  Natural Earth 10m land is therefore
    # used as an external land/sea reference *only* to classify these
    # faces; the displayed coastline remains the node-derived boundary.
    # Measured separation is clean: real islands have land fraction
    # 0.57-0.94, mesh gaps and sea faces 0.000-0.054.
    try:
        polar_frac = _ne_land_fractions(faces, pole_ids)
    except Exception as e:
        print(f"  warning: Natural Earth reference unavailable ({e}); "
              f"polar faces fall back to the node-density rule")
        polar_frac = None
    land = []
    for face in faces:
        if face.is_empty:
            continue
        if id(face) in pole_ids:
            land.append(face)
            continue
        if polar_frac is not None and id(face) in polar_frac:
            if polar_frac[id(face)] >= _POLAR_LAND_FRAC:
                land.append(face)
            continue
        # Node-free faces north of 84N are gaps in the sparse Arctic mesh
        # (no land exists above 84N), not islands - keep them ocean.
        if face.centroid.y > 84.0:
            continue
        need = max(1, int(np.ceil(face.area * _OCEAN_NODE_DENSITY)))
        if _polygon_node_count(face, lon360, lats, cells, need) < need:
            land.append(face)
    print(f"  land faces: {len(land)} ({time.time() - t0:.1f}s)")

    domain = box(0.0, -90.0, 360.0, 90.0)
    land_u = unary_union(land)
    # land rings may live in an unwrapped frame (e.g. lon 340..410 for
    # Africa); subtract shifted copies so the whole domain is covered.
    to_subtract = [land_u]
    for shift in (-360.0, 360.0):
        to_subtract.append(shp_transform(
            lambda x, y, z=None, s=shift: (x + s, y) if z is None else (x + s, y, z),
            land_u))
    ocean = domain
    for geom in to_subtract:
        ocean = ocean.difference(geom)

    # ---- 7. split at the dateline and fold into [-180, 180] ----
    west = ocean.intersection(box(-180.0, -90.0, 180.0, 90.0))
    east = ocean.intersection(box(180.0, -90.0, 540.0, 90.0))

    def _shift(x, y, z=None):
        return (x - 360.0, y) if z is None else (x - 360.0, y, z)

    east = shp_transform(_shift, east)
    ocean = unary_union([west, east])
    # Drop artificial segments running along the domain-box edges: the
    # dateline cut at lon +-180 and the pole edges at lat +-90 survive the
    # union wherever the ocean touches them (e.g. the ice-free Arctic cap,
    # whose "boundary" is the box top edge), and they draw as spurious
    # meridian lines on polar projections.  Real coastlines never run
    # exactly along the box edges, so this removes only artifacts.
    from shapely import MultiLineString, get_parts
    box_bdry = box(-180.0, -90.0, 180.0, 90.0).boundary
    coastline = ocean.boundary.difference(box_bdry.buffer(1e-9))
    coastline = MultiLineString([
        p for p in get_parts(coastline)
        if p.geom_type in ("LineString", "LinearRing") and p.length > 0])
    print(f"  done in {time.time() - t0:.1f}s: {ocean.geom_type}, "
          f"area={ocean.area:.1f} deg^2")
    return ocean, coastline
