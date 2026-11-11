import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
from numpy import ma
import pandas as pd
import seaborn as sns

from modules.utils import WAVE_VARIABLES, ERA5_VARIABLE_NAMES, MAJOR_CITIES
try:
    from modules.plot.satellite_background import add_satellite_background, add_satellite_background_global
except ImportError:
    add_satellite_background = None
    add_satellite_background_global = None
def plot_cluster_results(cluster_result, save_path=None, dpi=300, figsize=(24, 6), 
                         use_spherical_view=False):
    """
    Visualize clustering results with three subplots:
    1. Scatter plot colored by cluster labels
    2. Scatter plot with cluster centers
    3. Histogram of cluster sizes
    
    For global data, can also generate spherical 3D views from multiple angles.
    
    Parameters:
    -----------
    cluster_result : dict
        Dictionary containing clustering results with keys:
        - 'coordinates': Node coordinates [N, 2] (lat, lon)
        - 'labels': Cluster labels [N,]
        - 'centers': Cluster centers [n_clusters, 2]
        - 'cluster_sizes': Number of nodes in each cluster [n_clusters,]
        - 'n_clusters': Number of clusters
        - 'method': Clustering method used
    save_path : str, optional
        Path to save the figure. If None, displays the plot
    dpi : int
        Resolution of the output figure (default: 150)
    figsize : tuple
        Figure size (width, height) in inches (default: (18, 6))
    use_spherical_view : bool
        If True, also generate 3D spherical view visualizations from multiple angles.
        Only used for global data. Default is False.
    
    Returns:
    --------
    matplotlib.figure.Figure: The created figure object
    """
    from mpl_toolkits.mplot3d import Axes3D
    # Extract data from result dictionary
    coordinates = cluster_result['coordinates']
    labels = cluster_result['labels']
    centers = cluster_result['centers']
    cluster_sizes = cluster_result['cluster_sizes']
    n_clusters = cluster_result['n_clusters']
    method = cluster_result.get('method', 'unknown')
    
    # Create a shuffled color mapping to make adjacent clusters more distinguishable
    # Map original cluster IDs to shuffled IDs for better color contrast
    np.random.seed(42)  # Set seed for reproducibility
    unique_labels = np.unique(labels)
    shuffled_indices = np.random.permutation(len(unique_labels))
    color_mapping = {original: shuffled for original, shuffled in zip(unique_labels, shuffled_indices)}
    labels_shuffled = np.array([color_mapping[label] for label in labels])
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Subplot 1: Scatter plot colored by cluster labels
    ax1 = axes[0]
    scatter1 = ax1.scatter(coordinates[:, 1], coordinates[:, 0], 
                          c=labels_shuffled, cmap='nipy_spectral', s=1, alpha=1)
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.set_title(f'Clustering Results\n{n_clusters} Clusters ({method})', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for clusters
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
    cbar1.set_label('Cluster ID', fontsize=11)
    
    # Subplot 2: Scatter with cluster centers highlighted
    ax2 = axes[1]
    scatter2 = ax2.scatter(coordinates[:, 1], coordinates[:, 0], 
                          c=labels_shuffled, cmap='nipy_spectral', s=1, alpha=0.4)
    # Plot cluster centers with larger markers
    ax2.scatter(centers[:, 1], centers[:, 0], 
               c='red', marker='X', s=100, edgecolors='black', 
               linewidths=1.5, label='Cluster Centers', zorder=5)
    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12)
    ax2.set_title(f'Clusters with Centers\nTotal Nodes: {len(coordinates)}', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    
    # Subplot 3: Histogram of cluster sizes
    ax3 = axes[2]
    ax3.hist(cluster_sizes, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(cluster_sizes.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {cluster_sizes.mean():.1f}')
    ax3.axvline(cluster_sizes.min(), color='green', linestyle=':', 
               linewidth=2, label=f'Min: {cluster_sizes.min()}')
    ax3.axvline(cluster_sizes.max(), color='orange', linestyle=':', 
               linewidth=2, label=f'Max: {cluster_sizes.max()}')
    ax3.set_xlabel('Cluster Size (# of nodes)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title(f'Cluster Size Distribution\nStd: {cluster_sizes.std():.2f}', fontsize=13)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text box
    stats_text = (
        f'Balance Ratio: {cluster_sizes.max()/cluster_sizes.min():.2f}\n'
        f'Target Size: {len(coordinates)/n_clusters:.1f}'
    )
    ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Cluster visualization saved to: {save_path}")
    
    # Generate spherical 3D views for global data
    if use_spherical_view and save_path:
        _plot_cluster_spherical_views(cluster_result, save_path, dpi)
    
    return fig


def _plot_cluster_spherical_views(cluster_result, save_path, dpi=300):
    """
    Generate 3D spherical view visualizations of clustering results from multiple angles.
    
    Parameters:
    -----------
    cluster_result : dict
        Dictionary containing clustering results
    save_path : str
        Base path for saving the spherical view figures
    dpi : int
        Resolution of the output figures
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    coordinates = cluster_result['coordinates']
    labels = cluster_result['labels']
    centers = cluster_result['centers']
    n_clusters = cluster_result['n_clusters']
    method = cluster_result.get('method', 'unknown')
    
    # Create shuffled color mapping for better contrast
    np.random.seed(42)
    unique_labels = np.unique(labels)
    shuffled_indices = np.random.permutation(len(unique_labels))
    color_mapping = {original: shuffled for original, shuffled in zip(unique_labels, shuffled_indices)}
    labels_shuffled = np.array([color_mapping[label] for label in labels])
    
    # Convert lat/lon to 3D Cartesian coordinates on unit sphere
    lat_rad = np.radians(coordinates[:, 0])
    lon_rad = np.radians(coordinates[:, 1])
    
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    # Convert centers to 3D on unit sphere surface
    center_lat_rad = np.radians(centers[:, 0])
    center_lon_rad = np.radians(centers[:, 1])
    cx = np.cos(center_lat_rad) * np.cos(center_lon_rad)
    cy = np.cos(center_lat_rad) * np.sin(center_lon_rad)
    cz = np.sin(center_lat_rad)
    
    # Normalize to ensure centers are exactly on unit sphere surface
    center_norm = np.sqrt(cx**2 + cy**2 + cz**2)
    cx = cx / center_norm
    cy = cy / center_norm
    cz = cz / center_norm
    
    # Define view angles: (elev, azim, name)
    view_angles = [
        (20, -60, 'front_right'),
        (20, 120, 'front_left'),
        (20, -150, 'back_right'),
        (20, 60, 'back_left'),
        (90, 0, 'north_pole'),
        (-90, 0, 'south_pole'),
    ]
    
    base_path = save_path.replace('.png', '')
    
    for elev, azim, view_name in view_angles:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points colored by cluster
        scatter = ax.scatter(x, y, z, c=labels_shuffled, cmap='nipy_spectral', 
                           s=0.5, alpha=0.8)
        
        # Plot cluster centers with larger markers on top of data points
        ax.scatter(cx, cy, cz, c='red', marker='X', s=120, 
                  edgecolors='black', linewidths=2, label='Cluster Centers', zorder=10)
        
        # Draw wireframe sphere for reference
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = 0.99 * np.outer(np.cos(u), np.sin(v))
        y_sphere = 0.99 * np.outer(np.sin(u), np.sin(v))
        z_sphere = 0.99 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', 
                         alpha=0.1, linewidth=0.5)
        
        # Set equal aspect ratio
        max_range = 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Remove axis labels for cleaner look
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Title
        ax.set_title(f'Global Clustering - {n_clusters} Clusters ({method})\nView: {view_name.replace("_", " ").title()}', 
                    fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Cluster ID', fontsize=10)
        
        # Save figure
        spherical_save_path = f"{base_path}_spherical_{view_name}.png"
        plt.savefig(spherical_save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Spherical view ({view_name}) saved to: {spherical_save_path}")
        
        plt.close(fig)