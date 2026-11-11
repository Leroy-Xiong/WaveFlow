"""
Coordinate utilities for exporting and clustering geographic coordinates.

This module provides reusable functions for coordinate export and clustering
that can be used by different data processors (WaveDataProcessor, GlobalHindcastProcessor, etc.)
"""

import os
import numpy as np
from sklearn.cluster import KMeans
from modules.plot.plot_wave import plot_cluster_results


def spherical_mean(coordinates):
    """
    Calculate the spherical mean (centroid) of geographic coordinates.
    
    This method computes the true geographic center of a set of points on the
    Earth's surface, accounting for the Earth's curvature. It converts coordinates
    to 3D Cartesian space, computes the mean, then projects back to lat/lon.
    
    Parameters:
    -----------
    coordinates : np.ndarray
        Array of shape [N, 2] containing [lat, lon] coordinates in degrees
    
    Returns:
    --------
    np.ndarray: Spherical mean [lat, lon] in degrees, shape [2,]
    """
    if len(coordinates) == 0:
        return np.array([0.0, 0.0])
    
    if len(coordinates) == 1:
        return coordinates[0].copy()
    
    # Convert to radians
    lat_rad = np.radians(coordinates[:, 0])
    lon_rad = np.radians(coordinates[:, 1])
    
    # Convert to 3D Cartesian coordinates
    # x = cos(lat) * cos(lon)
    # y = cos(lat) * sin(lon)
    # z = sin(lat)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    # Compute mean in Cartesian space
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)
    
    # Normalize to unit sphere
    norm = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)
    if norm < 1e-10:
        # All points cancel out (e.g., antipodal points), fall back to simple mean
        return np.array([np.mean(coordinates[:, 0]), np.mean(coordinates[:, 1])])
    
    x_mean /= norm
    y_mean /= norm
    z_mean /= norm
    
    # Convert back to lat/lon
    lat_mean = np.degrees(np.arcsin(z_mean))
    lon_mean = np.degrees(np.arctan2(y_mean, x_mean))
    
    return np.array([lat_mean, lon_mean])


def haversine_distance(coord1, coord2):
    """
    Calculate the great circle distance between two points on the earth.
    
    Uses the Haversine formula to compute distance between points specified
    in decimal degrees (latitude, longitude).
    
    Parameters:
    -----------
    coord1 : np.ndarray
        Array of shape [N, 2] containing [lat, lon] coordinates
    coord2 : np.ndarray
        Array of shape [M, 2] containing [lat, lon] coordinates
    
    Returns:
    --------
    np.ndarray: Distance matrix of shape [N, M] in kilometers
    """
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert to radians
    lat1 = np.radians(coord1[:, 0:1])  # [N, 1]
    lon1 = np.radians(coord1[:, 1:2])  # [N, 1]
    lat2 = np.radians(coord2[:, 0])    # [M,]
    lon2 = np.radians(coord2[:, 1])    # [M,]
    
    # Haversine formula
    dlat = lat2 - lat1  # Broadcasting: [N, M]
    dlon = lon2 - lon1  # Broadcasting: [N, M]
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def export_coordinates(lat, lon, save_path=None, description="coordinates"):
    """
    Export coordinates (lat, lon) as numpy array.
    
    The output shape is [N, 2] where N is the number of nodes,
    and the 2 columns are [latitude, longitude].
    
    Parameters:
    -----------
    lat : np.ndarray
        Array of latitudes with shape [N,]
    lon : np.ndarray
        Array of longitudes with shape [N,]
    save_path : str, optional
        Path to save the .npy file. If None, returns coordinates without saving.
    description : str, optional
        Description of the coordinates for logging purposes (e.g., "HK", "Global")
    
    Returns:
    --------
    np.ndarray: Array of shape [N, 2] containing [lat, lon] coordinates
    """
    # Combine into [N, 2] array
    coordinates = np.column_stack([lat, lon])
    
    # Save to .npy file if path is provided
    if save_path is not None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to .npy file
        np.save(save_path, coordinates)
        
        print(f"{description.capitalize()} coordinates exported successfully!")
        print(f"Shape: {coordinates.shape}")
        print(f"Saved to: {save_path}")
        print(f"Latitude range: [{lat.min():.4f}, {lat.max():.4f}]")
        print(f"Longitude range: [{lon.min():.4f}, {lon.max():.4f}]")
    
    return coordinates


def cluster_coordinates(coordinates, n_clusters=256, method='recursive_bisection', 
                       save_path=None, vis_save_path=None, use_geographic_distance=True,
                       use_spherical_view=False):
    """
    Cluster coordinates into n_clusters groups with balanced node counts.
    
    This method performs clustering to ensure each cluster has approximately 
    equal number of nodes. The clustering considers spatial proximity while 
    maintaining balance.
    
    Parameters:
    -----------
    coordinates : np.ndarray
        Array of shape [N, 2] containing [lat, lon] coordinates
    n_clusters : int
        Number of clusters to create. Default is 256.
    method : str
        Clustering method to use. Options:
        - 'recursive_bisection': Recursive bisection for strictly balanced, spatially contiguous clusters
        - 'recursive_diameter': Recursive bisection based on geographic diameter (farthest pair), uses real Earth distance
        - 'balanced_kmeans': K-means with balanced reassignment
        - 'kmeans': Standard K-means (may be unbalanced)
    save_path : str, optional
        Path to save the clustering results. If None, only returns results without saving.
    vis_save_path : str, optional
        Path to save the visualization plot.
        If None, auto-generates based on save_path (replaces .npz with _visualization.png)
        Set to False to skip visualization
    use_geographic_distance : bool
        If True, use Haversine formula for geographic distance calculation.
        If False, use Euclidean distance on raw coordinates.
        Default is True.
    use_spherical_view : bool
        If True, also generate 3D spherical view visualizations from multiple angles.
        Only recommended for global data. Default is False.
    
    Returns:
    --------
    dict: Dictionary containing:
        - 'labels': Cluster labels for each node [N,]
        - 'centers': Cluster center coordinates [n_clusters, 2]
        - 'cluster_sizes': Number of nodes in each cluster [n_clusters,]
        - 'coordinates': Original coordinates [N, 2]
        - 'n_clusters': Number of clusters (int)
        - 'method': Clustering method used (str)
        - 'max_points': Maximum number of points in any patch (int)
        - 'local_coords_template': Pre-calculated local coordinates [n_clusters, max_points, 2]
        - 'mask_template': Pre-calculated validity mask [n_clusters, max_points]
    """
    n_nodes = coordinates.shape[0]
    print(f"\nClustering {n_nodes} nodes into {n_clusters} clusters...")
    
    if method == 'recursive_bisection':
        print("Using Recursive Bisection for strictly balanced, spatially contiguous clusters...")
        labels = _recursive_bisection_clustering(coordinates, n_clusters)
        
        # Calculate centers using spherical mean for geographic accuracy
        centers = np.zeros((n_clusters, 2))
        for i in range(n_clusters):
            mask = (labels == i)
            if np.any(mask):
                centers[i] = spherical_mean(coordinates[mask])

    elif method == 'balanced_kmeans':
        # Step 1: Use K-means to find initial cluster centers
        print("Step 1: Finding initial cluster centers with K-means...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        initial_labels = kmeans.fit_predict(coordinates)
        centers = kmeans.cluster_centers_
        
        # Step 2: Balanced reassignment using distance-based greedy algorithm
        print("Step 2: Performing balanced reassignment...")
        labels = _balanced_assignment(coordinates, centers, n_clusters, 
                                       use_geographic_distance=use_geographic_distance)
        
    elif method == 'recursive_diameter':
        print("Using Recursive Diameter clustering based on geographic distance...")
        labels = _recursive_diameter_clustering(coordinates, n_clusters)
        
        # Calculate centers using spherical mean for geographic accuracy
        centers = np.zeros((n_clusters, 2))
        for i in range(n_clusters):
            mask = (labels == i)
            if np.any(mask):
                centers[i] = spherical_mean(coordinates[mask])
    
    elif method == 'kmeans':
        # Standard K-means
        print("Using standard K-means clustering...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(coordinates)
        centers = kmeans.cluster_centers_
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'recursive_bisection', 'recursive_diameter', 'balanced_kmeans' or 'kmeans'.")
    
    # Calculate cluster sizes
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    
    # Recalculate centers based on final assignment using spherical mean
    centers = np.array([spherical_mean(coordinates[labels == i]) if cluster_sizes[i] > 0 
                       else centers[i] for i in range(n_clusters)])
    
    # Print statistics
    print("\n" + "="*60)
    print("Clustering Results:")
    print("="*60)
    print(f"Total nodes: {n_nodes}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Average cluster size: {n_nodes/n_clusters:.2f}")
    print(f"Min cluster size: {cluster_sizes.min()}")
    print(f"Max cluster size: {cluster_sizes.max()}")
    print(f"Std of cluster sizes: {cluster_sizes.std():.2f}")
    print(f"Balance ratio (max/min): {cluster_sizes.max()/cluster_sizes.min():.2f}")
    
    # Calculate max points per patch (for padding)
    max_points = cluster_sizes.max()
    
    # Pre-calculate local_coords and mask for all patches
    # This avoids recalculating them every time data is loaded
    print("\nPre-calculating local_coords and mask for all patches...")
    local_coords_template = np.zeros((n_clusters, max_points, 2), dtype=np.float32)
    local_coords_3d_template = np.zeros((n_clusters, max_points, 3), dtype=np.float32)
    mask_template = np.zeros((n_clusters, max_points), dtype=bool)
    
    for cluster_id in range(n_clusters):
        node_indices = np.where(labels == cluster_id)[0]
        n_points = len(node_indices)
        
        # Calculate local coordinates (relative to cluster center)
        # coordinates are [lat, lon], centers are [lat, lon]
        node_coords = coordinates[node_indices]  # [n_points, 2]
        center = centers[cluster_id]  # [2,]
        local_coords_template[cluster_id, :n_points, :] = node_coords - center
        
        # Calculate 3D Cartesian local coordinates (for global data)
        # Convert node coordinates to 3D Cartesian on unit sphere
        lat_rad = np.radians(node_coords[:, 0])
        lon_rad = np.radians(node_coords[:, 1])
        node_3d = np.stack([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad)
        ], axis=-1)  # [n_points, 3]
        
        # Convert cluster center to 3D Cartesian
        center_lat_rad = np.radians(center[0])
        center_lon_rad = np.radians(center[1])
        center_3d = np.array([
            np.cos(center_lat_rad) * np.cos(center_lon_rad),
            np.cos(center_lat_rad) * np.sin(center_lon_rad),
            np.sin(center_lat_rad)
        ])  # [3,]
        
        # Calculate 3D offset (node position relative to center on unit sphere)
        local_coords_3d_template[cluster_id, :n_points, :] = node_3d - center_3d
        
        # Set mask to True for valid points
        mask_template[cluster_id, :n_points] = True
    
    print(f"  - local_coords shape: {local_coords_template.shape}")
    print(f"  - local_coords_3d shape: {local_coords_3d_template.shape}")
    print(f"  - mask shape: {mask_template.shape}")
    print(f"  - Max points per patch: {max_points}")
    
    # Create result dictionary
    result = {
        'labels': labels,
        'centers': centers,
        'cluster_sizes': cluster_sizes,
        'n_clusters': n_clusters,
        'method': method,
        'coordinates': coordinates,
        'max_points': max_points,
        'local_coords_template': local_coords_template,
        'local_coords_3d_template': local_coords_3d_template,
        'mask_template': mask_template
    }
    
    # Save results if path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **result)
        print(f"\nClustering results saved to: {save_path}")
    
    # Visualize clustering results
    if vis_save_path is not False:
        # Set visualization save path
        if vis_save_path is None and save_path is not None:
            vis_save_path = save_path.replace('.npz', '_visualization.png')
        
        if vis_save_path is not None:
            plot_cluster_results(result, save_path=vis_save_path, dpi=300, 
                               use_spherical_view=use_spherical_view)
    
    return result


def _balanced_assignment(coordinates, centers, n_clusters, use_geographic_distance=True):
    """
    Perform balanced assignment of points to cluster centers.
    
    Uses a greedy algorithm to ensure each cluster gets approximately
    the same number of points while minimizing total distance.
    
    Parameters:
    -----------
    coordinates : np.ndarray
        Array of shape [N, 2] containing coordinates
    centers : np.ndarray
        Array of shape [n_clusters, 2] containing cluster centers
    n_clusters : int
        Number of clusters
    use_geographic_distance : bool
        If True, use Haversine formula for geographic distance calculation.
        If False, use Euclidean distance on raw coordinates.
    
    Returns:
    --------
    np.ndarray: Cluster labels for each point [N,]
    """
    n_nodes = coordinates.shape[0]
    target_size = n_nodes // n_clusters
    
    # Calculate distance matrix [n_nodes, n_clusters]
    if use_geographic_distance:
        distances = haversine_distance(coordinates, centers)
    else:
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(coordinates, centers)
    
    # Initialize
    labels = np.full(n_nodes, -1, dtype=int)
    cluster_sizes = np.zeros(n_clusters, dtype=int)
    
    # Sort all (point, cluster) pairs by distance
    sorted_indices = np.argsort(distances.flatten())
    
    for flat_idx in sorted_indices:
        point_idx = flat_idx // n_clusters
        cluster_idx = flat_idx % n_clusters
        
        # Skip if point already assigned
        if labels[point_idx] != -1:
            continue
        
        # Assign if cluster not full
        if cluster_sizes[cluster_idx] < target_size:
            labels[point_idx] = cluster_idx
            cluster_sizes[cluster_idx] += 1
        
        # Check if all points are assigned
        if np.all(labels != -1):
            break
    
    # Assign any remaining points to nearest non-full cluster
    for point_idx in np.where(labels == -1)[0]:
        nearest_clusters = np.argsort(distances[point_idx])
        for cluster_idx in nearest_clusters:
            labels[point_idx] = cluster_idx
            cluster_sizes[cluster_idx] += 1
            break
    
    return labels


def _find_farthest_pair(coordinates):
    """
    Find the pair of points with maximum geographic distance.
    
    Uses a heuristic approach to approximate the diameter of the point set
    without computing the full O(N^2) distance matrix.
    
    Parameters:
    -----------
    coordinates : np.ndarray
        Array of shape [N, 2] containing [lat, lon] coordinates
    
    Returns:
    --------
    tuple: (idx_a, idx_b) indices of the farthest pair
    """
    n_points = coordinates.shape[0]
    
    if n_points <= 2:
        return 0, min(1, n_points - 1)
    
    # Step 1: Find a point on the convex hull approximation
    # Use extreme points in lat/lon as seeds
    lat_min_idx = np.argmin(coordinates[:, 0])
    lat_max_idx = np.argmax(coordinates[:, 0])
    lon_min_idx = np.argmin(coordinates[:, 1])
    lon_max_idx = np.argmax(coordinates[:, 1])
    
    # Step 2: From each extreme point, find the farthest point
    extreme_indices = [lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx]
    max_dist = 0
    farthest_pair = (0, 0)
    
    for start_idx in extreme_indices:
        # Calculate distances from this point to all others
        distances = haversine_distance(
            coordinates[start_idx:start_idx+1],
            coordinates
        ).flatten()
        
        far_idx = np.argmax(distances)
        if distances[far_idx] > max_dist:
            max_dist = distances[far_idx]
            farthest_pair = (start_idx, far_idx)
    
    # Step 3: One more iteration from the found farthest point
    # to refine the estimate
    a_idx, b_idx = farthest_pair
    distances_from_b = haversine_distance(
        coordinates[b_idx:b_idx+1],
        coordinates
    ).flatten()
    
    refined_idx = np.argmax(distances_from_b)
    if distances_from_b[refined_idx] > max_dist:
        farthest_pair = (b_idx, refined_idx)
    
    return farthest_pair


def _recursive_diameter_clustering(coordinates, n_clusters):
    """
    Perform recursive clustering based on geographic diameter (farthest pair).
    
    This method recursively splits the data by finding the farthest pair of points
    and partitioning based on proximity to these two points. This ensures that
    the splitting considers real Earth distances (via Haversine formula) rather
    than just coordinate differences.
    
    The algorithm ensures balanced cluster sizes by using the distance ratio
    as a sorting key to achieve approximately equal-sized splits.
    
    Parameters:
    -----------
    coordinates : np.ndarray
        Array of shape [N, 2] containing [lat, lon] coordinates
    n_clusters : int
        Number of clusters to create
    
    Returns:
    --------
    np.ndarray: Cluster labels for each point [N,]
    """
    n_nodes = coordinates.shape[0]
    labels = np.zeros(n_nodes, dtype=int)
    
    # Initialize with all points in cluster 0
    clusters_to_split = [(0, np.arange(n_nodes))]
    next_cluster_id = 1
    
    while clusters_to_split and next_cluster_id < n_clusters:
        # Get the cluster with most points
        cluster_id, indices = clusters_to_split.pop(0)
        
        n_points = len(indices)
        if n_points <= 1:
            continue
        
        # Get coordinates for this cluster
        coords = coordinates[indices]
        
        # Find the farthest pair using geographic distance
        local_idx_a, local_idx_b = _find_farthest_pair(coords)
        
        # Get the actual point coordinates of the farthest pair
        point_a = coords[local_idx_a:local_idx_a+1]  # [1, 2]
        point_b = coords[local_idx_b:local_idx_b+1]  # [1, 2]
        
        # Calculate distance from all points to both farthest points
        dist_to_a = haversine_distance(coords, point_a).flatten()
        dist_to_b = haversine_distance(coords, point_b).flatten()
        
        # Calculate distance ratio: dist_to_a / (dist_to_a + dist_to_b)
        # Points with ratio < 0.5 are closer to A, ratio > 0.5 are closer to B
        total_dist = dist_to_a + dist_to_b
        # Avoid division by zero
        total_dist = np.where(total_dist == 0, 1, total_dist)
        dist_ratio = dist_to_a / total_dist
        
        # Sort points by distance ratio to find the balanced split point
        sorted_local_indices = np.argsort(dist_ratio)
        
        # Split at the middle to ensure balanced cluster sizes
        mid = n_points // 2
        
        # Get the original indices for each group
        left_local_indices = sorted_local_indices[:mid]
        right_local_indices = sorted_local_indices[mid:]
        
        left_indices = indices[left_local_indices]
        right_indices = indices[right_local_indices]
        
        # Assign to two new clusters
        labels[left_indices] = cluster_id
        labels[right_indices] = next_cluster_id
        
        # Add both clusters to the list for further splitting if needed
        if len(left_indices) > 1:
            clusters_to_split.append((cluster_id, left_indices))
        if len(right_indices) > 1:
            clusters_to_split.append((next_cluster_id, right_indices))
        
        # Sort clusters by size (largest first) for next iteration
        clusters_to_split.sort(key=lambda x: len(x[1]), reverse=True)
        
        next_cluster_id += 1
    
    return labels


def _recursive_bisection_clustering(coordinates, n_clusters):
    """
    Perform recursive bisection clustering for balanced, spatially contiguous clusters.
    
    This method recursively splits the data along the longest dimension
    to create balanced clusters.
    
    Parameters:
    -----------
    coordinates : np.ndarray
        Array of shape [N, 2] containing [lat, lon] coordinates
    n_clusters : int
        Number of clusters to create
    
    Returns:
    --------
    np.ndarray: Cluster labels for each point [N,]
    """
    n_nodes = coordinates.shape[0]
    labels = np.zeros(n_nodes, dtype=int)
    
    # Initialize with all points in cluster 0
    clusters_to_split = [(0, np.arange(n_nodes))]
    next_cluster_id = 1
    
    while clusters_to_split and next_cluster_id < n_clusters:
        # Get the cluster with most points
        cluster_id, indices = clusters_to_split.pop(0)
        
        if len(indices) <= 1:
            continue
        
        # Get coordinates for this cluster
        coords = coordinates[indices]
        
        # Determine the longest dimension (lat or lon)
        lat_range = coords[:, 0].max() - coords[:, 0].min()
        lon_range = coords[:, 1].max() - coords[:, 1].min()
        
        # Split along the longest dimension
        if lat_range > lon_range:
            split_dim = 0  # Split by latitude
        else:
            split_dim = 1  # Split by longitude
        
        # Sort by the split dimension and split in half
        sorted_indices = np.argsort(coords[:, split_dim])
        mid = len(indices) // 2
        
        # Assign to two new clusters
        left_indices = indices[sorted_indices[:mid]]
        right_indices = indices[sorted_indices[mid:]]
        
        labels[left_indices] = cluster_id
        labels[right_indices] = next_cluster_id
        
        # Add both clusters to the list for further splitting if needed
        if len(left_indices) > 1:
            clusters_to_split.append((cluster_id, left_indices))
        if len(right_indices) > 1:
            clusters_to_split.append((next_cluster_id, right_indices))
        
        # Sort clusters by size (largest first) for next iteration
        clusters_to_split.sort(key=lambda x: len(x[1]), reverse=True)
        
        next_cluster_id += 1
    
    return labels
