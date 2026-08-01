# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import laspy
import open3d as o3d
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.data_prepare_utils import compute_verticality_feature

class Config:
    # Input paths
    raw_laz_path = os.path.join(ROOT_DIR, "datasets\\wytham_vox0.1.laz")
    
    # Output paths
    output_base = os.path.join(BASE_DIR, "processed_data")
    
    coord_threshold = 1000000.0    # Large coordinate threshold
    verticality_radius = 0.5    # Verticality search radius optimized for L1W
    
    use_outlier_removal = True    # Whether to enable outlier removal
    nb_neighbors_stat = 30    # Statistical filtering: number of neighbors
    std_ratio_stat = 2.0    # Statistical filtering: standard deviation multiplier

# Create output directory
os.makedirs(Config.output_base, exist_ok=True)

print(f"Reading full-scene LAZ file: {os.path.basename(Config.raw_laz_path)}")
las = laspy.read(Config.raw_laz_path)

# Extract raw data
points = np.vstack((las.x, las.y, las.z)).T
classification = np.array(las.classification).astype(np.int32)
tree_ids = np.array(las.treeID).astype(np.int32)

print(f"[Info] Total original points: {len(points):,}")

# Coordinate offset processing (keep consistent with subsequent processing)
min_bound = np.min(points, axis=0)
applied_offset = np.array([0.0, 0.0, 0.0])

if np.any(np.abs(min_bound[:2]) > Config.coord_threshold):
    applied_offset = -np.floor(min_bound)
    print(f"[Action] Applied Offset: {applied_offset}")
else:
    print("[Info] Coordinate system is normal, no offset needed.")

shifted_points = points + applied_offset

# Outlier/Flying Point Removal
if Config.use_outlier_removal:
    print("\nStarting outlier removal...")
    print(f"\nExecuting Statistical Outlier Removal (Neighbors: {Config.nb_neighbors_stat}, Std Ratio Threshold: {Config.std_ratio_stat})...")
    
    # Create temporary Open3D point cloud for filtering
    temp_cloud = o3d.geometry.PointCloud()
    temp_cloud.points = o3d.utility.Vector3dVector(shifted_points.astype(np.float64))
    
    # Statistical filtering
    cl, ind = temp_cloud.remove_statistical_outlier(
        nb_neighbors=Config.nb_neighbors_stat, 
        std_ratio=Config.std_ratio_stat
    )
    
    # Apply filtering results
    inlier_mask_stat = np.zeros(len(shifted_points), dtype=bool)
    inlier_mask_stat[ind] = True
    
    shifted_points = shifted_points[inlier_mask_stat]
    classification = classification[inlier_mask_stat]
    tree_ids = tree_ids[inlier_mask_stat]
    print(f"[Statistical Filtering] Points retained: {len(shifted_points):,} ({len(shifted_points)/len(inlier_mask_stat)*100:.2f}%)")

# Calculate Verticality 
start_time = time.time()
vert_feats = compute_verticality_feature(
    shifted_points, 
    search_radius=Config.verticality_radius, 
    num_threads=os.cpu_count() // 2    # Automatically allocate threads
)
print(f"[Time] Verticality calculation duration: {time.time() - start_time:.2f} seconds.")

# Encapsulate as Open3D Tensor PointCloud
full_cloud = o3d.t.geometry.PointCloud()
full_cloud.point.positions = o3d.core.Tensor(shifted_points.astype(np.float32))

# Inject all core attributes
full_cloud.point["gt_tree_id"] = o3d.core.Tensor(tree_ids.reshape(-1, 1), o3d.core.int32)
full_cloud.point["classification"] = o3d.core.Tensor(classification.reshape(-1, 1), o3d.core.int32)
full_cloud.point["verticality"] = o3d.core.Tensor(vert_feats.reshape(-1, 1), o3d.core.float32)

# Save baseline PLY
base_ply_path = os.path.join(Config.output_base, "Wytham_full_area_base.ply")
print("Exporting standard baseline PLY (with Verticality)...")
o3d.t.io.write_point_cloud(base_ply_path, full_cloud)

print("\n[Task Completed]:")
print(f"File path: {base_ply_path}")
print("Attributes included: positions, gt_tree_id, classification, verticality")
print(f"Full scene range: {np.ptp(shifted_points, axis=0)}")
