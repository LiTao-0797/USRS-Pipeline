# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np
import os
import time
from plyfile import PlyData, PlyElement
from jakteristics import compute_features as compute_features_jakteristics

def compute_verticality_feature(points, search_radius=0.5, num_threads=4):
    """
    Computes a verticality feature, incorporating NaN handling logic.
    Based on TreeLearn logic, specifically designed to distinguish vertical stems from horizontal ground/branches.
    """
    if not isinstance(points, np.ndarray):
        points = np.asarray(points)
        
    print(f"[Feature Computation] Computing verticality (radius: {search_radius}m, threads: {num_threads})...")
    
    # Compute the raw feature using jakteristics (ensure float64 for precision)
    features = compute_features_jakteristics(
        points.astype(np.float64), 
        search_radius=search_radius, 
        num_threads=num_threads, 
        feature_names=['verticality']
    )
    
    #  NaN handling logic
    ind_nan = np.isnan(features)
    nan_count = np.sum(ind_nan)
    
    if nan_count > 0:
        mean_val = np.nanmean(features, axis=0)
        print(f"[Feature Repair] Found {nan_count} NaN values，filled with mean value of {mean_val[0]:.4f} .")
        for i in range(features.shape[1]):
            features[ind_nan[:, i], i] = mean_val[i]
            
    return features.astype(np.float32)


def load_and_standardize_ply(file_path, device, threshold):
    """
    Loads a PLY file and automatically applies an offset if necessary.
    """
    print(f"Parsing: {os.path.basename(file_path)}")
    plydata = PlyData.read(file_path)
    el = plydata.elements[0]
    
    # Read raw data as float64 to ensure precision with large coordinates
    x = np.array(el.data['x'], dtype=np.float64)
    y = np.array(el.data['y'], dtype=np.float64)
    z = np.array(el.data['z'], dtype=np.float64)
    points = np.stack([x, y, z], axis=1)

    # Automatically detect coordinate scale
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    applied_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    
    # Check if X or Y coordinates exceed the threshold 
    # Z is usually smaller and not used as the primary indicator
    if np.any(np.abs(min_bound[:2]) > threshold):
        # Calculate offset: align minimum values to 0 (using floor for easier memorization)
        applied_offset = -np.floor(min_bound)
        points += applied_offset
        print("[Warning]: Global coordinate system/large coordinates detected!")
        print(f"[Action]: Applied offset: {applied_offset}")
    else:
        print(f"[Information]: Coordinate scale normal (Max X: {max_bound[0]:.2f})，keeping original coordinates.")

    # Convert to Open3D Tensor format
    cloud = o3d.t.geometry.PointCloud(device)
    cloud.point["positions"] = o3d.core.Tensor(points.astype(np.float32), device=device)
    
    return cloud, applied_offset

def get_device():
    """Gets the optimal device for Open3D: GPU if CUDA is available, otherwise CPU."""
    if o3d.core.cuda.is_available():
        return o3d.core.Device("CUDA:0")
    return o3d.core.Device("CPU:0")

def estimate_normals_custom(cloud, radius_multiplier=8, max_nn=30):
    """
    Estimate normals adaptively based on point cloud average spacing
    
    Parameters:
        cloud: o3d.t.geometry.PointCloud (Tensor interface) or o3d.geometry.PointCloud
        radius_multiplier: Radius as a multiple of the average spacing (recommended 5-10)
        max_nn: Maximum number of neighbors to search
    """
    
    device = get_device()
    print(f"Current computation device: {device}")
    
    # Ensure input is Legacy format for spacing calculation 
    # Tensor interface has complex nn distance calculation
    is_tensor = isinstance(cloud, o3d.t.geometry.PointCloud)
    legacy_cloud = cloud.to_legacy() if is_tensor else cloud

    if len(legacy_cloud.points) == 0:
        print("Warning: Point cloud is empty, cannot compute normals.")
        return cloud

    # Compute average spacing (Average Spacing)
    distances = legacy_cloud.compute_nearest_neighbor_distance()
    avg_spacing = np.mean(distances)
    
    # Calculate adaptive radius
    adaptive_radius = avg_spacing * radius_multiplier
    print(f"Detected point cloud average spacing: {avg_spacing:.4f}m")
    print(f"Set adaptive normal search radius: {adaptive_radius:.4f}m (multiplier: {radius_multiplier})")

    # Execute normal estimation
    if is_tensor:
        gpu_cloud = cloud.to(device)
        gpu_cloud.estimate_normals(max_nn=max_nn, radius=adaptive_radius)
        gpu_cloud.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
        return gpu_cloud
    else:
        legacy_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=adaptive_radius, max_nn=max_nn)
        )
        legacy_cloud.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
        return legacy_cloud

def process_and_save_cloud(input_path, output_path, offset, radius):
    """
    Read the global coordinate point cloud, compute verticality features using local coordinates, 
    and save in global coordinates.
    """
    if not os.path.exists(input_path):
        print(f"[Warning] File does not exist, skipping: {input_path}")
        return False
        
    print(f"Reading point cloud: {os.path.basename(input_path)}")
    # Use Tensor interface for reading to allow direct injection of attribute channels later
    pcd = o3d.t.io.read_point_cloud(input_path)
    
    # Convert to float64 NumPy array for local transformation, 
    # avoiding precision loss from float32 large coordinate subtraction
    orig_points = pcd.point.positions.numpy().astype(np.float64)
    
    # Temporarily translate to near the origin (local coordinate system)
    shifted_points = orig_points + offset
    
    # Perform multi-threaded verticality calculation in local coordinate system
    print(f"Calculating verticality (search radius: {radius}m)...")
    start_time = time.time()
    vert_feats = compute_verticality_feature(
        shifted_points, 
        search_radius=radius, 
        num_threads=os.cpu_count() // 2
    )
    print(f"Calculation complete, time taken: {time.time() - start_time:.2f} sec")
    
    # Inject the calculated verticality directly into the original global point cloud 
    # original point cloud coordinates remain unchanged
    pcd.point["verticality"] = o3d.core.Tensor(vert_feats.reshape(-1, 1), o3d.core.float32)
    
    # Save the updated global coordinate point cloud
    print(f"Exporting global PLY file: {output_path}")
    o3d.t.io.write_point_cloud(output_path, pcd)
    return True

if __name__ == "__main__":
    # --- Test module: Generate a virtual point cloud plane with noise ---
    print("Creating test point cloud data...")
    
    # Create a 10x10 grid of points (100 points)
    x = np.linspace(0, 5000, 1000) 
    y = np.linspace(0, 5000, 1000)
    xv, yv = np.meshgrid(x, y)
    zv = np.zeros_like(xv) + np.random.normal(0, 0.02, xv.shape)    # Add slight height noise
    
    points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)
    
    # Convert to Open3D Tensor point cloud
    test_cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(points.astype(np.float32)))
    # Introduced from version 0.13.0, a new geometry processing module based on tensors was added,
    # which is under the open3d.t namespace.
    # Tensor point clouds can be stored on GPU (by specifying device parameter such as cuda:0).
    # This means that computation-intensive tasks like normal estimation, voxel filtering downsampling,
    # and transformation matrix operations can leverage GPU parallel computing power to achieve performance improvements of dozens of times
    
    print("Starting test estimate_normals_custom...")
    test_cloud = estimate_normals_custom(test_cloud, radius_multiplier=6)
    
    # Check results
    if "normals" in test_cloud.point:
        normals = test_cloud.point["normals"].numpy()
        avg_normal = np.mean(normals, axis=0)
        print(f"Test successful! Average normal vector: {avg_normal}")
        print("Tip: For a horizontal plane, the normal should be close to [0, 0, 1]")
        
        # Visualization
        o3d.visualization.draw_geometries([test_cloud.to_legacy()],    # Must convert back to legacy format before visualization
                                        window_name="Normal Estimation Test",
                                        point_show_normal=True)
    else:
        print("Test failed: Normals could not be generated.")
    
