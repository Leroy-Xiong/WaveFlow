"""
Clean WATLEV garbage values in HK hindcast data products.

Background
----------
The WATLEV (water level / elev) channel at a fixed cluster of mesh nodes
around (22.28-22.30N, 113.90-113.94E) contains physically impossible values
(|elev| up to ~100 m, real HK tide is ~±1.5 m), present through the whole
1970-2022 record.  A few other nodes show sporadic spikes.  These values
pollute training targets and the evaluation GT.

This script fills those values with IDW interpolation from the 4 nearest
clean nodes (or clean grid cells for the regular-grid product), then
recomputes elev statistics.  Only the elev channel is touched.

Steps
-----
    weights : precompute IDW fill weights (nodes + grid cells), save to
              data/hindcast/watlev_fix_weights.npz
    nc      : fill processed/wave_*.nc           (node GT + source of truth)
    npz     : fill processed_npz/wave_*.npz      (irregular-model training data)
    grid    : fill preinterpolated_hk_zoom/*/wave_*.npy  (grid-model training data)
    stats   : recompute elev stats from cleaned data -> wave_data_stats.json
    verify  : rescan data for remaining |elev| > THRESH

Usage
-----
    python -m modules.datasets.clean_watlev --step all --workers 24
    python -m modules.datasets.clean_watlev --step nc --workers 24
    python -m modules.datasets.clean_watlev --step grid --workers 24
    python -m modules.datasets.clean_watlev --step stats
    python -m modules.datasets.clean_watlev --step verify
"""

import os
import json
import time
import argparse
import numpy as np
from netCDF4 import Dataset
from multiprocessing import Pool
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)

PROCESSED_DIR = './data/hindcast/processed'
NPZ_DIR = './data/hindcast/processed_npz'
GRID_DIR = './data/hindcast/preinterpolated_hk_zoom'
COORD_FILE = './data/hindcast/hk_coordinates.npy'
LAND_MASK_FILE = './data/hindcast/land_mask_75x107.npy'
STATS_FILE = './data/hindcast/wave_data_stats.json'
WEIGHTS_FILE = './data/hindcast/watlev_fix_weights.npz'

HK_REGION = (113.6, 114.6, 22.0, 22.7)

# Nodes (indices into the 7788 HK node array / hk_coordinates.npy) with
# physically impossible WATLEV values, determined by full-record scans
# (>5 m) plus IDW residual scans (>1 m vs 4 nearest clean neighbours):
#   * the 12-node cluster at (22.28-22.30N, 113.90-113.94E) + east node
#     1067 (persistent ~-2.3 m offset): garbage at (almost) all timestamps
#     -> UNCONDITIONAL fill
#   * sporadic nodes (0, 50, 1066, 2875, 3770): rare spikes, real tide
#     otherwise -> conditional fill when |elev| > COND_THRESH
BAD_HK_INDICES = [0, 50, 1066, 1067, 2875, 3770,
                  5419, 5420, 5495, 5569, 5570,
                  5642, 5643, 5712, 5713, 5781, 5782]
BAD_UNCONDITIONAL = [1067, 5419, 5420, 5495, 5569, 5570, 5642,
                     5643, 5712, 5713, 5781, 5782]
COND_THRESH = 2.5        # m — fill sporadic nodes when |elev| exceeds this
K = 4                    # IDW neighbours
GRID_FILL_RADIUS = 0.05  # degrees around bad nodes (contamination footprint)

_CTX = None  # per-process context (weights, index maps)


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------
def _hk_full_indices():
    """Full-mesh indices of the 7788 HK nodes (from the first NC file)."""
    with Dataset(os.path.join(PROCESSED_DIR, 'wave_19700101T00.nc')) as ds:
        lon = ds['lon'][:]
        lat = ds['lat'][:]
    mask = ((lon >= HK_REGION[0]) & (lon <= HK_REGION[1]) &
            (lat >= HK_REGION[2]) & (lat <= HK_REGION[3]))
    return np.where(mask)[0]


def compute_node_weights():
    """IDW weights: for each bad node, K nearest clean HK nodes."""
    coords = np.load(COORD_FILE)  # (7788, 2) [lat, lon]
    bad = np.array(BAD_HK_INDICES)
    clean = np.setdiff1d(np.arange(len(coords)), bad)
    tree = cKDTree(coords[:, ::-1])  # [lon, lat]
    nbr_idx = np.zeros((len(bad), K), dtype=np.int64)
    nbr_w = np.zeros((len(bad), K), dtype=np.float64)
    for j, node in enumerate(bad):
        d, idx = tree.query(coords[node][::-1], k=K + len(bad))
        sel = [i for i in idx if i not in bad][:K]
        dd, _ = tree.query(coords[node][::-1], k=sel)
        w = 1.0 / (dd + 1e-9)
        nbr_idx[j] = sel
        nbr_w[j] = w / w.sum()
    return bad, nbr_idx, nbr_w


def compute_grid_weights():
    """IDW weights for contaminated grid cells (within GRID_FILL_RADIUS of a
    bad node): each contaminated cell is filled from K nearest clean cells."""
    nc = np.load(COORD_FILE)
    bad_xy = nc[BAD_HK_INDICES][:, ::-1]  # [lon, lat]
    lons = np.linspace(HK_REGION[0], HK_REGION[1], 107)
    lats = np.linspace(HK_REGION[2], HK_REGION[3], 75)
    LON, LAT = np.meshgrid(lons, lats)
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    land = np.load(LAND_MASK_FILE).astype(bool).ravel()

    tree = cKDTree(bad_xy)
    dist, _ = tree.query(pts)
    fill_mask = (~land) & (dist < GRID_FILL_RADIUS)
    fill_idx = np.where(fill_mask)[0]
    clean_cells = np.where((~land) & (~fill_mask))[0]

    tree2 = cKDTree(pts[clean_cells])
    nbr_idx = np.zeros((len(fill_idx), K), dtype=np.int64)
    nbr_w = np.zeros((len(fill_idx), K), dtype=np.float64)
    for j, cell in enumerate(fill_idx):
        d, idx = tree2.query(pts[cell], k=K)
        w = 1.0 / (d + 1e-9)
        nbr_idx[j] = clean_cells[idx]
        nbr_w[j] = w / w.sum()
    return fill_idx, nbr_idx, nbr_w


def step_weights():
    print_header('weights')
    bad, nidx, w = compute_node_weights()
    fidx, gnidx, gw = compute_grid_weights()
    hk_full = _hk_full_indices()
    uncond = np.array([i in BAD_UNCONDITIONAL for i in bad], dtype=bool)
    np.savez_compressed(
        WEIGHTS_FILE,
        hk_full_indices=hk_full,
        bad_hk_indices=bad,
        bad_unconditional=uncond,
        node_nbr_indices=nidx,
        node_weights=w,
        grid_fill_indices=fidx,
        grid_nbr_indices=gnidx,
        grid_weights=gw,
    )
    print(f'  HK nodes: {len(hk_full)}, bad nodes: {len(bad)} '
          f'({uncond.sum()} unconditional, {len(bad) - uncond.sum()} conditional), K={K}')
    print(f'  grid: {len(fidx)} contaminated cells of 8025')
    print(f'  saved {WEIGHTS_FILE}')


# ---------------------------------------------------------------------------
# Per-process context + workers
# ---------------------------------------------------------------------------
def _init_worker(weights_path):
    global _CTX
    d = np.load(weights_path)
    _CTX = dict(
        hk_full=d['hk_full_indices'],
        bad=d['bad_hk_indices'],
        uncond=d['bad_unconditional'],
        node_nbr=d['node_nbr_indices'],   # HK-filtered indices
        node_w=d['node_weights'],
        grid_fill=d['grid_fill_indices'],
        grid_nbr=d['grid_nbr_indices'],
        grid_w=d['grid_weights'],
    )
    d.close()


def _fill_node_values(elev_hk, ctx):
    """Fill bad nodes of a (7788,) elev array in place.

    Unconditional bad nodes are always filled with IDW from clean neighbours;
    conditional (sporadic) ones only when |elev| > COND_THRESH.
    """
    bad = ctx['bad']
    nbr = ctx['node_nbr']
    w = ctx['node_w']
    uncond = ctx['uncond']
    vals = elev_hk[bad]
    nbr_vals = elev_hk[nbr]
    changed = uncond | (np.abs(vals) > COND_THRESH)
    for j in range(len(bad)):
        if not changed[j]:
            continue
        v = nbr_vals[j]
        ok = np.isfinite(v)
        if ok.sum() < 1:
            continue
        wj = w[j][ok]
        wj = wj / wj.sum()
        vals[j] = float((v[ok] * wj).sum())
    elev_hk[bad] = vals


def _nc_worker(args):
    fpath, _ = args
    ctx = _CTX
    with Dataset(fpath, 'r+') as ds:
        elev = ds['elev'][:]
        hk = elev[ctx['hk_full']]  # copy
        _fill_node_values(hk, ctx)
        elev[ctx['hk_full']] = hk
        ds['elev'][:] = elev
    return 1


def _npz_worker(args):
    fpath, _ = args
    ctx = _CTX
    d = np.load(fpath)
    elev = d['elev'].copy()
    hs, tp, dm = d['hs'], d['tp'], d['dm']
    d.close()
    new = elev.copy()
    _fill_node_values(new, ctx)
    if np.array_equal(new, elev, equal_nan=True):
        return 0
    np.savez_compressed(fpath, hs=hs, tp=tp, dm=dm, elev=new)
    return 1


def _grid_worker(args):
    fpath, _ = args
    ctx = _CTX
    g = np.load(fpath)
    elev = g[3].copy()
    flat = elev.ravel()
    fidx = ctx['grid_fill']
    nbr = g[3].ravel()[ctx['grid_nbr']]  # (n_fill, K)
    w = ctx['grid_w']
    for j in range(len(fidx)):
        v = nbr[j]
        ok = np.isfinite(v)
        if ok.sum() < 1:
            continue
        wj = w[j][ok]
        wj = wj / wj.sum()
        flat[fidx[j]] = float((v[ok] * wj).sum())
    if np.array_equal(flat, g[3].ravel(), equal_nan=True):
        return 0
    g[3] = flat.reshape(elev.shape)
    np.save(fpath, g)
    return 1


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def _file_list(root, pattern):
    files = []
    for dirpath, _, fnames in os.walk(root):
        for f in sorted(fnames):
            if f.startswith('wave_') and f.endswith(pattern):
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def _run(files, worker, num_workers):
    t0 = time.time()
    total = len(files)
    print(f'  files: {total}')
    if num_workers <= 1:
        _init_worker(WEIGHTS_FILE)
        done = sum(worker((f, None)) for f in files)
    else:
        with Pool(num_workers, initializer=_init_worker,
                  initargs=(WEIGHTS_FILE,)) as pool:
            results = pool.map(worker, [(f, None) for f in files],
                               chunksize=200)
        done = sum(results)
    print(f'  {done}/{total} files modified in {time.time() - t0:.1f}s')


def step_nc(num_workers):
    print_header('nc: fill processed/wave_*.nc')
    _run(_file_list(PROCESSED_DIR, '.nc'), _nc_worker, num_workers)


def step_npz(num_workers):
    print_header('npz: fill processed_npz/wave_*.npz')
    _run(_file_list(NPZ_DIR, '.npz'), _npz_worker, num_workers)


def step_grid(num_workers):
    print_header('grid: fill preinterpolated_hk_zoom/*/wave_*.npy')
    _run(_file_list(GRID_DIR, '.npy'), _grid_worker, num_workers)


def _stats_worker(fpath):
    with Dataset(fpath) as ds:
        lon = ds['lon'][:]
        lat = ds['lat'][:]
        hkmask = ((lon >= HK_REGION[0]) & (lon <= HK_REGION[1]) &
                  (lat >= HK_REGION[2]) & (lat <= HK_REGION[3]))
        e = ds['elev'][:].astype(np.float64)[hkmask]
    e = e[np.isfinite(e)]
    if e.size == 0:
        return (0, 0.0, 0.0, np.inf, -np.inf)
    return (e.size, float(e.sum()), float((e ** 2).sum()),
            float(e.min()), float(e.max()))


def step_stats(num_workers):
    print_header('stats: recompute elev statistics')
    files = _file_list(PROCESSED_DIR, '.nc')
    with Pool(num_workers) as pool:
        res = pool.map(_stats_worker, files, chunksize=500)
    n = sum(r[0] for r in res)
    s = sum(r[1] for r in res)
    s2 = sum(r[2] for r in res)
    emin = min(r[3] for r in res if r[0])
    emax = max(r[4] for r in res if r[0])
    mean = s / n
    std = float(np.sqrt(s2 / n - mean ** 2))
    print(f'  n={n} elev_mean={mean:.4f} elev_std={std:.4f} '
          f'min={emin:.3f} max={emax:.3f}')

    with open(STATS_FILE) as f:
        stats = json.load(f)
    print(f'  old elev stats: mean={stats.get("elev_mean")}, '
          f'std={stats.get("elev_std")}')
    stats['elev_mean'] = mean
    stats['elev_std'] = std
    stats['elev_min'] = emin
    stats['elev_max'] = emax
    stats['elev_source'] = ('cleaned WATLEV (bad nodes filled by IDW), '
                            'computed over HK-region nodes, full record')
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'  updated {STATS_FILE}')


def _scan_hk_bad(fpath):
    with Dataset(fpath) as ds:
        lon = ds['lon'][:]
        lat = ds['lat'][:]
        hkmask = ((lon >= HK_REGION[0]) & (lon <= HK_REGION[1]) &
                  (lat >= HK_REGION[2]) & (lat <= HK_REGION[3]))
        e = ds['elev'][:].astype(np.float64)[hkmask]
    return np.where(np.abs(e) > COND_THRESH)[0].tolist()


def step_verify(num_workers):
    print_header(f'verify: rescan |elev| > {COND_THRESH} m over HK nodes')
    files = _file_list(PROCESSED_DIR, '.nc')
    from collections import Counter
    with Pool(num_workers) as pool:
        res = pool.map(_scan_hk_bad, files, chunksize=500)
    c = Counter()
    for r in res:
        c.update(r)
    print(f'  nodes still bad (any timestamp): {len(c)}')
    for i, n in c.most_common(15):
        print(f'    hk-idx {i}: {n} timestamps')
    nc = np.load(COORD_FILE)
    for i in c:
        print(f'    node {i} at ({nc[i,0]:.4f}, {nc[i,1]:.4f})')


def print_header(title):
    print('\n' + '=' * 60)
    print(f'  {title}')
    print('=' * 60)


def main():
    ap = argparse.ArgumentParser(description='Clean WATLEV garbage values')
    ap.add_argument('--step', default='all',
                    choices=['all', 'weights', 'nc', 'npz', 'grid',
                             'stats', 'verify'])
    ap.add_argument('--workers', '-w', type=int, default=24)
    args = ap.parse_args()

    if args.step in ('all', 'weights'):
        step_weights()
    if args.step in ('all', 'nc'):
        step_nc(args.workers)
    if args.step in ('all', 'npz'):
        step_npz(args.workers)
    if args.step in ('all', 'grid'):
        step_grid(args.workers)
    if args.step in ('all', 'stats'):
        step_stats(args.workers)
    if args.step in ('all', 'verify'):
        step_verify(args.workers)
    print('\nDone.')


if __name__ == '__main__':
    main()
