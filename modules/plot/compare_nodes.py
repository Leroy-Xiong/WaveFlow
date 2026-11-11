"""
Compare spatial node distributions across hindcast and regular-grid data sources
in the Hong Kong region.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point

from modules.utils import HK_REGION


def visualize_hk_node_comparison(
    global_coord_path='./data/global_hindcast/global_coordinates.npy',
    hk_coord_path='./data/hindcast/hk_coordinates.npy',
    save_path='./output/plot/data/hk/node_distribution_comparison.png',
    dpi=200,
):
    """Compare HK-region node distributions across four data sources.

    Plots ocean-only nodes for:
     - 0.25° regular grid (ERA5 / AI forecast)
     - 0.083° regular grid (numerical model)
     - Global hindcast unstructured mesh (subset to HK)
     - HK local hindcast unstructured mesh
    """
    min_lon, max_lon, min_lat, max_lat = HK_REGION

    coords_hk = np.load(hk_coord_path)
    coords_global = np.load(global_coord_path)

    mask_global = (
        (coords_global[:, 1] >= min_lon) & (coords_global[:, 1] <= max_lon) &
        (coords_global[:, 0] >= min_lat) & (coords_global[:, 0] <= max_lat)
    )
    coords_global_hk = coords_global[mask_global]

    def _make_regular_grid(res_deg):
        lons = np.arange(min_lon, max_lon + res_deg * 0.5, res_deg)
        lats = np.arange(min_lat, max_lat + res_deg * 0.5, res_deg)
        lon_g, lat_g = np.meshgrid(lons, lats)
        return np.column_stack([lat_g.ravel(), lon_g.ravel()])

    coords_025deg = _make_regular_grid(0.25)
    coords_0083deg = _make_regular_grid(0.083)

    land_shp = shpreader.natural_earth(resolution='10m', category='physical', name='land')
    land_geoms = list(shpreader.Reader(land_shp).geometries())

    def _filter_ocean(coords):
        mask = np.ones(len(coords), dtype=bool)
        for i, (lat, lon) in enumerate(coords):
            pt = Point(lon, lat)
            for geom in land_geoms:
                if geom.contains(pt):
                    mask[i] = False
                    break
        return coords[mask]

    coords_025deg = _filter_ocean(coords_025deg)
    coords_0083deg = _filter_ocean(coords_0083deg)

    print(f"Global hindcast (HK ocean):  {coords_global_hk.shape[0]} nodes")
    print(f"HK local hindcast (ocean):   {coords_hk.shape[0]} nodes")
    print(f"0.25 deg regular (ocean):    {coords_025deg.shape[0]} nodes")
    print(f"0.083 deg regular (ocean):   {coords_0083deg.shape[0]} nodes")

    fig, axes = plt.subplots(1, 4, figsize=(28, 7),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    panels = [
        (coords_025deg, 'black', 150,
         f"0.25° Regular Grid (AI forecast)\n{coords_025deg.shape[0]} ocean nodes"),
        (coords_0083deg, 'black', 50,
         f"0.083° Regular Grid (numerical model)\n{coords_0083deg.shape[0]} ocean nodes"),
        (coords_global_hk, 'black', 30,
         f"Global Hindcast (unstructured)\n{coords_global_hk.shape[0]} ocean nodes"),
        (coords_hk, 'black', 8,
         f"HK Local Hindcast (unstructured)\n{coords_hk.shape[0]} ocean nodes"),
    ]

    for ax, (coords, color, size, title) in zip(axes, panels):
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, color='whitesmoke', alpha=1.0, zorder=2)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5,
                       alpha=0.7, zorder=3)
        ax.scatter(coords[:, 1], coords[:, 0], c=color, s=size, marker='o',
                   edgecolors='none', alpha=0.7, transform=ccrs.PlateCarree(),
                   zorder=4)
        gl = ax.gridlines(draw_labels=True, alpha=0.3, zorder=5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}
        ax.text(0.02, 0.98, f'Ocean nodes: {coords.shape[0]}',
                transform=ax.transAxes, fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.set_title(title, fontsize=25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Node comparison saved to: {save_path}")
    return fig
