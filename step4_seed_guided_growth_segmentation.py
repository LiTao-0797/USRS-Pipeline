# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import open3d as o3d
import time
from scipy.spatial import cKDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.ground_segmentation import GroundSegmentation, create_dtm_mesh
from utils.instance_segmentation import TreeInstanceSegmentation, fuse_seeds


class Config:
    # Input paths
    step2_results = os.path.join(BASE_DIR, "step2_results")
    step3_final_results = os.path.join(BASE_DIR, "step3_final_results")
    
    # Output paths
    output_dir = os.path.join(BASE_DIR, "step4_segmentation_results")
    
    # EVO dataset specific offset
    offset = np.array([-399000.0, -6786000.0, 0.0])
    
    # Ground DTM and isolation parameters (using CSF, points with height below 0.5m are forcibly classified as ground)
    ground_buffer = 0.5
    ground_params = {
        "csf_threshold": 0.5,    # Ground classification threshold (meters)
        "csf_resolution": 1.0,    # Global cloth mesh resolution
        "csf_rigidness": 2,              # Cloth rigidity (1-Low, 2-Medium, 3-High)
        "csf_correct_steep_slope": False,
        "csf_iterations": 500,    # Maximum simulation iterations
        "dtm_resolution": 0.5,    # DTM mesh resolution (meters)
        "dtm_k": 20,    # Number of nearest neighbors for interpolation
        "dtm_p": 1.0,    # IDW distance weight exponent
        "dtm_voxel_size": 0.3    # Voxel size for sparse ground points (meters)
    }
    
    # 3D Region Growing control parameters (for 10cm voxelized vegetation, 
    # enabling high-fullness growth without secondary downsampling)
    grower_params = {
        "voxel_size": 0.18,    # Growth step (diagonal protection line for 10cm grid set to 0.18m)
        "z_scale": 2.0,    # Vertical growth upward traction compression factor
        "min_total_assignment_ratio": 0.00001,
        "min_tree_assignment_ratio": 0.3,
        "max_search_radius": 1.5,    # Maximum allowed obstacle-crossing search radius (meters)
        "decrease_search_radius_after_num_iter": 10,
        "max_iterations": 500,
        "early_stop_ratio": 0.99
    }
    
    # Adaptive trunk configuration parameters
    trunk_settings = {
        "seed_diameter_factor": 1.10,    # 10% tolerance factor for adaptive outward expansion of seed cylinder diameter
        "seed_height_range": (1.0, 1.6)    # Seed sampling interval (around breast height 1.3m)
    }
    
    # Global fusion deduplication distance threshold for multi-tile seeds (meters)
    seed_fuse_dist = 0.6
    
    # Whether to enable step-by-step and final result visualization checks
    is_vis = True  


np.random.seed(42)    # Fix random seed to ensure consistent tree coloring in each segmentation
os.makedirs(Config.output_dir, exist_ok=True)

# Automatically search and batch merge globally aligned high-density tile point clouds exported from Step 3
ply_files = glob.glob(os.path.join(Config.step3_final_results, "tile_*.ply"))
if len(ply_files) == 0:
    print("[Error] No optimized aligned PLY tiles found. Please run step3_global_optimization.py first.")
    sys.exit(1)

print(f"Loading and merging {len(ply_files)} globally consistent PLY tiles...")
all_positions = []
all_verticality = []
all_colors = []

offset_tensor = o3d.core.Tensor(Config.offset, o3d.core.float32)

for f in sorted(ply_files):
    pcd = o3d.t.io.read_point_cloud(f)
    pcd.translate(offset_tensor)    # Translate coordinates to near origin to prevent float32 precision loss
    
    all_positions.append(pcd.point.positions.numpy())
    if "verticality" in pcd.point:
        all_verticality.append(pcd.point["verticality"].numpy().flatten())
    else:
        all_verticality.append(np.zeros(len(pcd.point.positions), dtype=np.float32))
        
    if "colors" in pcd.point:
        all_colors.append(pcd.point.colors.numpy())
    else:
        all_colors.append(np.full((len(pcd.point.positions), 3), 0.7, dtype=np.float32))

# Vertically stack and merge into unified full-scene high-precision data
positions = np.vstack(all_positions)
verticality = np.concatenate(all_verticality)
colors = np.vstack(all_colors)
print(f"[Merge Complete] Total points in original full-resolution point cloud: {len(positions):,}")

# Spatial high-precision downsampling 
# 0.10m spatial downsampling to reduce 3D growth computation load while preserving precise skeleton contours
full_cloud_temp = o3d.t.geometry.PointCloud()
full_cloud_temp.point.positions = o3d.core.Tensor(positions.astype(np.float32))
full_cloud_temp.point.colors = o3d.core.Tensor(colors.astype(np.float32))
full_cloud_temp.point["verticality"] = o3d.core.Tensor(verticality.reshape(-1, 1).astype(np.float32))

voxel_size = 0.10
print(f"Performing {voxel_size}m adaptive downsampling on the merged global point cloud...")
down_cloud = full_cloud_temp.voxel_down_sample(voxel_size=voxel_size)

# Extract attributes of the downsampled point cloud
down_positions = down_cloud.point.positions.numpy().astype(np.float64)
down_verticality = down_cloud.point["verticality"].numpy().flatten()
print(f"[Downsampling Complete] Point count for 3D growth stage: {len(down_positions):,}")

# Multi-tile trunk center seed point fusion and deduplication
print("Fusing global trunk core seed points and their feature diameters...")
npz_files = glob.glob(os.path.join(Config.step2_results, "*_trunk_info.npz"))
if len(npz_files) == 0:
    print("[Error] No trunk seed NPZ data found. Please run step2 first.")
    sys.exit(1)

seg_config = {"seed_fuse_dist": Config.seed_fuse_dist}
# Extract global deduplicated centers, confidence scores, and fitted diameters
stem_seeds, stem_scores, stem_diameters = fuse_seeds(npz_files, seg_config)
print(f"[Seed Fusion] After global fusion and clarification, {len(stem_seeds)} potential single-tree trunks found.")

# Global undulating surface fitting and elevation interpolation normalization
print("Initializing GroundSegmentation to fit global undulating surface (DTM) for downsampled point cloud...")
segmenter = GroundSegmentation(
    csf_threshold=Config.ground_params["csf_threshold"],
    csf_resolution=Config.ground_params["csf_resolution"],
    csf_rigidness=Config.ground_params["csf_rigidness"],
    csf_correct_steep_slope=Config.ground_params["csf_correct_steep_slope"],
    csf_iterations=Config.ground_params["csf_iterations"],
    dtm_resolution=Config.ground_params["dtm_resolution"],
    dtm_k=Config.ground_params["dtm_k"],
    dtm_p=Config.ground_params["dtm_p"],
    dtm_voxel_size=Config.ground_params["dtm_voxel_size"]
)

# Use downsampled point cloud for terrain and continuous elevation map interpolation calculation
down_ground_mask, dtm, dtm_offset = segmenter.process(down_positions)

# Use the fitted DTM bilinear elevation mapper 
# to calculate normalized relative heights for both downsampled and full-resolution original large point clouds
print("Calculating point cloud local normalized relative heights using high-precision surface model...")
down_rel_heights = segmenter.get_normalized_heights(down_positions)
full_rel_heights = segmenter.get_normalized_heights(positions)


# [Visualization] Global Terrain Fit Check
if Config.is_vis:
    print("\n[Check]: Loading undulating terrain fit check interface...")
    vis_ground_pcd = o3d.geometry.PointCloud()
    vis_ground_pcd.points = o3d.utility.Vector3dVector(down_positions)
    
    # Render ground points in black, vegetation points in gray
    vis_colors = np.full((len(down_positions), 3), [0.8, 0.8, 0.8])
    vis_colors[down_rel_heights <= Config.ground_buffer] = [0.0, 0.0, 0.0]
    vis_ground_pcd.colors = o3d.utility.Vector3dVector(vis_colors)
    
    dtm_mesh = create_dtm_mesh(segmenter.dtm_data)
    
    print("Opening 3D visualization window (Color: DTM terrain mesh, Black: Extracted ground points)...")
    o3d.visualization.draw_geometries(
        [vis_ground_pcd, dtm_mesh], 
        window_name="Global CSF DTM Verification", 
        mesh_show_back_face=True
    )
    del vis_ground_pcd, dtm_mesh, vis_colors

# High-precision adaptive outward selection for trunk seeds
print("Extracting adaptive outward expansion seed sources at 1.3m cylinder...")
down_cylinder_seed_mask = np.zeros(len(down_positions), dtype=bool)

# Perform spatial coordinate retrieval only within the breast height Z range
z_seed_mask = (down_rel_heights >= Config.trunk_settings["seed_height_range"][0]) & \
              (down_rel_heights <= Config.trunk_settings["seed_height_range"][1])
z_seed_idx = np.where(z_seed_mask)[0]
z_seed_pts = down_positions[z_seed_mask]

if len(z_seed_pts) > 0 and len(stem_seeds) > 0:
    tree_2d = cKDTree(z_seed_pts[:, :2])
    for i in range(len(stem_seeds)):
        center = stem_seeds[i]
        diameter = stem_diameters[i]
        
        # Use diameter physical parameters for 10% expanded radius spatial retrieval
        radius = (diameter / 2.0) * Config.trunk_settings["seed_diameter_factor"]
        
        near_idx = tree_2d.query_ball_point(center[:2], r=radius)
        if len(near_idx) > 0:
            down_cylinder_seed_mask[z_seed_idx[near_idx]] = True


# [Visualization] Debug-level Trunk Alignment Check (Red=Cylinder Seeds, Light-Gray=All 1-4m Candidate Trunks)
if Config.is_vis:
    print("\n[Check]: Loading debug-level trunk alignment check interface...")
    
    # Create spatial merge mask: includes original slice point cloud in 1~4m range, and adaptive cylinder seed points at 1.3m
    raw_slice_mask = (down_rel_heights >= 1.0) & (down_rel_heights <= 4.0)
    render_mask = raw_slice_mask | down_cylinder_seed_mask
    
    render_points = down_positions[render_mask]
    render_seed_mask = down_cylinder_seed_mask[render_mask]
    
    print(f"[Info] Points used for debug rendering after filtering: {len(render_points):,}")
    
    # Default: All candidate trunks in 1~4m range colored light gray
    colors_render = np.full((len(render_points), 3), [0.85, 0.85, 0.85], dtype=np.float32)
    # Seed points: Cylinder seeds colored red
    colors_render[render_seed_mask] = [1.0, 0.0, 0.0]
    
    vis_stem_cloud = o3d.geometry.PointCloud()
    vis_stem_cloud.points = o3d.utility.Vector3dVector(render_points)
    vis_stem_cloud.colors = o3d.utility.Vector3dVector(colors_render)
    
    # Slight spatial downsampling to ensure smooth interaction in the rendering window
    vis_stem_pcd = vis_stem_cloud.voxel_down_sample(0.05)
    
    print("Opening 3D visualization window...")
    print("[Debug Tips]:")
    print("1. Red rings: Cylinder seed points extracted by adaptively expanding 10% based on each single tree's unique DBH.")
    print("2. Light gray columns: Original upright vegetation slice points in the 1~4m range across the whole scene.")
    print("3. Rotate and zoom carefully: Inspect if the red seed points around each tree's bark are tightly and completely selected.")
    
    o3d.visualization.draw_geometries(
        [vis_stem_pcd], 
        window_name="Debug-level Stem Verification (Red=Seeds, Light-Gray=All 1-4m Stems)", 
        mesh_show_back_face=True
    )
    del vis_stem_pcd, vis_stem_cloud, colors_render, render_points

# 3D Region Growing Competition (No secondary sparsification, 10cm full-resolution high-fullness growth)
print("-> Starting pure NumPy-level high-concurrency 3D region growing competition algorithm...")
grower = TreeInstanceSegmentation(
    voxel_size=Config.grower_params["voxel_size"],
    z_scale=Config.grower_params["z_scale"],
    min_total_assignment_ratio=Config.grower_params["min_total_assignment_ratio"],
    min_tree_assignment_ratio=Config.grower_params["min_tree_assignment_ratio"],
    max_search_radius=Config.grower_params["max_search_radius"],
    decrease_search_radius_after_num_iter=Config.grower_params["decrease_search_radius_after_num_iter"],
    max_iterations=Config.grower_params["max_iterations"],
    early_stop_ratio=Config.grower_params["early_stop_ratio"]
)

start_grow_time = time.time()
down_labels = grower.segment_cloud(
        points=down_positions,
        rel_heights=down_rel_heights,
        trunk_centers=stem_seeds,
        cylinder_seed_mask=down_cylinder_seed_mask,
        verticality=down_verticality
)
print(f"[Time Stats] 3D region growing duration: {time.time() - start_grow_time:.2f} seconds.")

# Regrowth post-processing healing and "skirt" trimming at the lower side of the tree crown based on D_alt
down_labels_cleaned = grower.clean_labels(
    points=down_positions,
    rel_heights=down_rel_heights,
    segmented_labels=down_labels,
    cylinder_seed_mask=down_cylinder_seed_mask,
    trunk_centers=stem_seeds
)

# Dual-channel 1-NN Spatial Label High-Speed Back-Projection (Label Propagation)
# project the labels from the memory-downsampled segmentation back to the 14 million+ original full-resolution point cloud
print("Starting dual-channel 1-NN spatial label high-speed back-projection...")
start_prop_time = time.time()

# Full-resolution original large point cloud classification: 
# All points below ground threshold assigned -2 (absolute ground isolation to prevent label leakage)
ground_mask = full_rel_heights < Config.ground_buffer
veg_mask = ~ground_mask

veg_points_full = positions[veg_mask]

# Extract coordinates and corresponding labels of the non-ground vegetation subset from the downsampled point cloud
down_veg_mask = down_rel_heights >= Config.ground_buffer
down_veg_points = down_positions[down_veg_mask]
down_veg_labels = down_labels_cleaned[down_veg_mask]

# Initialize final instance label array for full resolution
final_labels = np.full(len(positions), -1, dtype=np.int32)
final_labels[ground_mask] = -2    # Ground points fixed to -2

if len(down_veg_points) > 0 and len(veg_points_full) > 0:
    # Build spatial back-projection search tree
    prop_tree = cKDTree(down_veg_points)
    _, prop_idx = prop_tree.query(veg_points_full, k=1, workers=-1)
    
    # Original point cloud inherits corresponding single-tree IDs
    final_labels[veg_mask] = down_veg_labels[prop_idx]

print(f"[Time Stats] 1-NN spatial back-projection cumulative duration: {time.time() - start_prop_time:.2f} seconds.")


# [Visualization] Ground Hard Isolation Verification
if Config.is_vis:
    print("\n[Check]: Displaying ground hard isolation effect (Orange: Ground, Gray: Vegetation Skeleton)...")
    indices = np.arange(len(positions))
    # Sample 1 million points for smooth rendering
    vis_idx = np.random.choice(indices, size=min(len(indices), 1000000), replace=False)
    
    vis_pcd_legacy = o3d.geometry.PointCloud()
    vis_pcd_legacy.points = o3d.utility.Vector3dVector(positions[vis_idx])
    
    vis_colors = np.full((len(vis_idx), 3), [0.75, 0.75, 0.75])
    # Points belonging to the absolute ground surface in the sample are colored orange
    local_ground_mask = final_labels[vis_idx] == -2
    vis_colors[local_ground_mask] = [1.0, 0.5, 0.0]
    vis_pcd_legacy.colors = o3d.utility.Vector3dVector(vis_colors)
    
    o3d.visualization.draw_geometries(
        [vis_pcd_legacy], 
        window_name="Ground Isolation Verification (Orange: Ground, Gray: Veg)",
        mesh_show_back_face=True
    )
    del vis_pcd_legacy, vis_colors, vis_idx, indices

# Assemble high-fidelity global single-tree segmentation PLY results and save (automatically applying coordinate restoration to retain global UTM large coordinates)
result_pcd = o3d.t.geometry.PointCloud()
result_pcd.point.positions = o3d.core.Tensor(positions.astype(np.float32))
result_pcd.point["tree_id"] = o3d.core.Tensor(final_labels.reshape(-1, 1), o3d.core.int32)

# Render random bright colors
max_id = final_labels.max()
unique_colors = np.random.uniform(0.15, 1.0, size=(max_id + 2, 3)).astype(np.float32)

color_array = np.zeros((len(final_labels), 3), dtype=np.float32)

# Assign high-contrast random colors to points belonging to each single tree (ID >= 0)
tree_points_mask = final_labels >= 0
color_array[tree_points_mask] = unique_colors[final_labels[tree_points_mask]]
color_array[final_labels == -2] = [0.4, 0.3, 0.2]    # Ground: Brown
color_array[final_labels == -1] = [0.85, 0.85, 0.85]    # Unclassified/Noise: Light Gray

result_pcd.point.colors = o3d.core.Tensor(color_array)

# Before saving to disk, perform reverse translation to return coordinates to the original high-precision global large coordinates
print("Restoring absolute coordinate system translation...")
offset_back_tensor = o3d.core.Tensor(-Config.offset, o3d.core.float32)
result_pcd.translate(offset_back_tensor)

save_path = os.path.join(Config.output_dir, "seed_guided_growth_global_seg.ply")
print(f"Saving high-fidelity global single-tree segmentation PLY results: {save_path}")
o3d.t.io.write_point_cloud(save_path, result_pcd)
print("[File Save Complete]")


# [Visualization] Full-scene single-tree segmentation final effect rendering (Automatically hiding ground and gray noise, displaying only pure single trees)
if Config.is_vis:
    print("\n[Check]: Loading panoramic single-tree segmentation results (Ground background and gray noise points automatically hidden)...")
    
    # Restore local translation for display centered in the rendering window to prevent drift during rotation under large coordinates
    result_pcd.translate(o3d.core.Tensor(Config.offset, o3d.core.float32))
    
    # Strictly filter and extract only points with single-tree labels (ID >= 0), completely removing ground (-2) and gray noise (-1)
    tree_mask = final_labels >= 0
    vis_pts = positions[tree_mask]
    final_colors = color_array[tree_mask]
    
    # Build traditional Open3D point cloud for pure single-tree visualization
    vis_pcd_legacy = o3d.geometry.PointCloud()
    vis_pcd_legacy.points = o3d.utility.Vector3dVector(vis_pts)
    vis_pcd_legacy.colors = o3d.utility.Vector3dVector(final_colors)
    
    print("Performing 0.20m voxel downsampling on 3D colored single-tree point cloud to ensure smooth rendering...")
    vis_pcd = vis_pcd_legacy.voxel_down_sample(0.20)
    
    # Create visualizer to optimize rendering appearance
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Full Forest Instance Segmentation (Colors=Trees, Background/Noise=Removed)")
    vis.add_geometry(vis_pcd)
    
    opt = vis.get_render_option()
    opt.point_size = 2.0
    
    print("[Operation Tips]:")
    print("1. Brown ground (-2) and gray noise points (-1) are fully filtered and hidden.")
    print("2. The screen only displays: 100% pure single tree bodies grown guided by the 'Unified Seed Framework'.")
    print("3. Rotate and zoom to carefully check the outer contours and spatial boundaries of single tree crowns.")
    
    vis.run()
    vis.destroy_window()

print("\n" + "=" * 50)
print("EVO dataset single-tree segmentation execution completed.")
print(f"Successfully identified and segmented single trees: {len(stem_seeds)} trees.")
print(f"Final results saved to disk at: {save_path}")
print("=" * 50)