"""Quick-look plots for a sampling run.

Reads the ``npz`` files written by ``waveflow sample`` and draws a small map
panel per variable: one member, the ensemble mean and the ensemble spread.

    python -m waveflow plot --run runs/hk_20211013T00
    python -m waveflow plot --run runs/global_20220924T00 --var hs --out fig.png

The figures are diagnostics, not publication artwork: they use cartopy's
Natural Earth coastlines and a linear colour scale.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

STYLE = {
    "hs": ("H$_s$ (m)", "turbo", 0.0),
    "tp": ("T$_{10}$ (s)", "cividis", 0.0),
    "dm": ("D$_m$ (deg)", "twilight", 0.0),
    "elev": ("Elev (m)", "RdBu_r", None),
}

DOMAINS = {
    7788: ("hk", "data/hindcast/hk_coordinates.npy"),
    651652: ("global", "data/global_hindcast/global_coordinates.npy"),
}


def _load_samples(run_dir: Path) -> tuple[dict[str, np.ndarray], str]:
    """Stack every ``sample_*.npz`` of the first timestamp in ``run_dir``."""
    samples = run_dir / "samples"
    if not samples.is_dir():
        samples = run_dir if list(run_dir.glob("sample_*.npz")) else samples
    files = sorted(samples.glob("sample_*.npz"))
    if not files:
        raise SystemExit(f"no sample_*.npz under {run_dir} (expected {run_dir}/samples)")

    first = np.load(files[0])
    timestamp = str(first["timestamp"])
    same = []
    for path in files:
        with np.load(path) as d:
            if str(d["timestamp"]) == timestamp:
                same.append((int(d["ensemble_index"]), d["sample"].astype(np.float32)))
    same.sort(key=lambda item: item[0])
    stack = np.stack([s for _, s in same])
    return {"stack": stack, "timestamp": timestamp,
            "members": [i for i, _ in same]}, timestamp


def _coords(shape_nodes: int, repo: Path) -> tuple[np.ndarray, str]:
    entry = DOMAINS.get(shape_nodes)
    if entry is None:
        raise SystemExit(
            f"unexpected node count {shape_nodes}; pass --coords to plot a custom mesh")
    domain, rel = entry
    path = repo / rel
    if not path.exists():
        raise SystemExit(
            f"missing {path}; run `python -m waveflow setup --weights` first")
    return np.load(path), domain


def _figure(timeseries: dict, var: str, coords: np.ndarray, domain: str,
            repo: Path, dpi: int) -> "object":
    import matplotlib

    matplotlib.use("Agg")
    cache = repo / "downloads" / "cartopy"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CARTOPY_DATA_DIR", str(cache))
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    stack = timeseries["stack"]                       # (M, N, 4)
    index = ["hs", "tp", "dm", "elev"].index(var)
    field = stack[:, :, index]
    mean = np.nanmean(field, axis=0)
    std = np.nanstd(field, axis=0)
    member = field[0]

    lon, lat = coords[:, 1], coords[:, 0]
    label, cmap, floor = STYLE[var]
    if floor is None:                                  # water level: symmetric
        vmax = float(np.nanpercentile(np.abs(mean), 99.5)) or 1.0
        vmin, vmax = -vmax, vmax
    else:
        vmin = 0.0
        vmax = float(np.nanpercentile(mean, 99.5)) or 1.0
    std_vmax = float(np.nanpercentile(std, 99.5)) or 1.0

    if domain == "global":
        proj = ccrs.Robinson()
        figsize = (15.0, 4.4)
        size = 0.8
        land = True
        keep = np.abs(lat) < 87.0          # Robinson clips the poles
        lon, lat = lon[keep], lat[keep]
    else:
        proj = ccrs.PlateCarree()
        figsize = (12.0, 4.6)
        size = 4.0
        land = False
        keep = None

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi,
                             subplot_kw={"projection": proj})
    if keep is not None:
        member, mean, std = member[keep], mean[keep], std[keep]
    panels = [
        (member, f"member {timeseries['members'][0]}", cmap, vmin, vmax),
        (mean, "ensemble mean", cmap, vmin, vmax),
        (std, "ensemble std", "viridis", 0.0, std_vmax),
    ]
    for ax, (values, title, cmap_i, lo, hi) in zip(axes, panels):
        ax.set_global() if domain == "global" else None
        ax.coastlines(resolution="110m" if domain == "global" else "10m",
                      linewidth=0.4, color="0.35")
        if land:
            ax.add_feature(cfeature.LAND, facecolor="0.92", zorder=0)
        m = ax.scatter(lon, lat, c=values, s=size, cmap=cmap_i,
                       vmin=lo, vmax=hi, transform=ccrs.PlateCarree(),
                       linewidths=0, rasterized=True,
                       edgecolors="none")
        if domain != "global":
            pad = 0.25
            ax.set_extent([lon.min() - pad, lon.max() + pad,
                           lat.min() - pad, lat.max() + pad],
                          crs=ccrs.PlateCarree())
        ax.set_title(f"{title}", fontsize=10)
        cbar = fig.colorbar(m, ax=ax, orientation="horizontal",
                            fraction=0.044, pad=0.06, shrink=0.9)
        cbar.set_label(label if title != "ensemble std" else f"std {label}",
                       fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        f"WaveFlow {domain} · {timeseries['timestamp']} · "
        f"{len(timeseries['members'])} members", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.10, wspace=0.04)
    return fig


def plot_run(run_dir: str | Path, var: str = "hs", out: str | Path | None = None,
             dpi: int = 200) -> Path:
    """Draw ``member / mean / std`` maps for one variable of a sampling run."""
    from . import bootstrap

    run = Path(run_dir)
    data, timestamp = _load_samples(run)
    coords, domain = _coords(data["stack"].shape[1], bootstrap.REPO)
    fig = _figure(data, var, coords, domain, bootstrap.REPO, dpi)
    target = Path(out) if out else run / "figures" / f"{var}_{timestamp}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return target
