# -*- coding: utf-8 -*-
"""
Step 2 (Section A): Global DTM Generation and Height Normalization
Using the newly written GroundSegmentation module.
"""

import os
import sys
import numpy as np
import open3d as o3d
import time
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

from utils.ground_segmentation import GroundSegmentation, create_dtm_mesh
from utils.tree_trunk_segmentation_plus import TreeTrunkSegmentationPlus
from utils.instance_segmentation import TreeInstanceSegmentation
from utils.accuracy_test import perform_tree_matching, evaluate_partitioned_accuracy


class Config:
    # Path configuration
    input_ply = os.path.join(BASE_DIR, "processed_data", "Wytham_full_area_base.ply")
    output_dir = os.path.join(BASE_DIR, "step2_results")
    
    # Export baseline PLY containing normalized elevation
    output_normalized_ply = os.path.join(output_dir, "Wytham_full_area_normalized.ply")
    
    is_vis = True    # Enable step-by-step visualization checks
    
    # DTM parameters (optimized for Wytham 54.6 million large-scale point cloud)
    csf_params = {
        "csf_threshold": 0.5,    # Ground classification threshold (meters)
        "csf_resolution": 1.0,    # Global cloth mesh resolution (1.0m for better global smoothness)
        "csf_rigidness": 2,    # Cloth rigidity
        "csf_correct_steep_slope": False,
        "csf_iterations": 500    # Maximum simulation iterations
    }
    dtm_params = {
        "dtm_resolution": 0.5,    # DTM mesh resolution (0.5m to ensure elevation accuracy)
        "dtm_k": 20,    # IDW interpolation nearest neighbor count
        "dtm_p": 1.0,    # IDW distance weight exponent
        "dtm_voxel_size": 0.5    # Voxel size for sparse ground points (set to 0.5m to significantly improve large-scale interpolation speed)
    }


# Create result save directory
os.makedirs(Config.output_dir, exist_ok=True)


# Global blind DTM processing and height normalization
print(f"Loading Step 1 preprocessed baseline point cloud: {os.path.basename(Config.input_ply)} ...")
if not os.path.exists(Config.input_ply):
    raise FileNotFoundError("Baseline PLY file not found. Please run Step 1 preprocessing script first.")

full_cloud = o3d.t.io.read_point_cloud(Config.input_ply)
print(f"[Point Cloud Loaded Successfully]: {full_cloud}")

# Extract pure 3D coordinate array
points = full_cloud.point.positions.numpy()
print(f"[Info] Total points to process: {len(points):,}")

# Instantiate the new global ground segmenter
segmenter = GroundSegmentation(
    csf_threshold=Config.csf_params["csf_threshold"],
    csf_resolution=Config.csf_params["csf_resolution"],
    csf_rigidness=Config.csf_params["csf_rigidness"],
    csf_correct_steep_slope=Config.csf_params["csf_correct_steep_slope"],
    csf_iterations=Config.csf_params["csf_iterations"],
    dtm_resolution=Config.dtm_params["dtm_resolution"],
    dtm_k=Config.dtm_params["dtm_k"],
    dtm_p=Config.dtm_params["dtm_p"],
    dtm_voxel_size=Config.dtm_params["dtm_voxel_size"]
)

# Start filtering and interpolation calculation
start_time = time.time()
ground_mask, dtm, dtm_offset = segmenter.process(points)
print(f"[Time Stats] Global DTM fitting duration: {time.time() - start_time:.2f} 秒")

# Based on the fitted DTM, calculate the exact relative height (net height) of all points to the ground
print("Calculating local net heights (Normalized Heights) for the full-scene point cloud using bilinear interpolation......")
start_time = time.time()
rel_heights = segmenter.get_normalized_heights(points)
print(f"[Time Stats] Normalization calculation duration: {time.time() - start_time:.2f} seconds")
print(f"[Normalization Result] Local net height range: {rel_heights.min():.2f}m ~ {rel_heights.max():.2f}m")

# Write the calculated rel_heights into Open3D Tensor PointCloud
full_cloud.point["rel_heights"] = o3d.core.Tensor(rel_heights.reshape(-1, 1), o3d.core.float32)

# Save the normalized point cloud as a new temporary baseline PLY file, serving as the unified data source for subsequent modules
print(f"Exporting baseline PLY file containing normalized elevation attributes: {Config.output_normalized_ply} ...")
o3d.t.io.write_point_cloud(Config.output_normalized_ply, full_cloud)
print("[File Save Complete]")


# [Visualization] Full-scene undulating DTM fit verification
if Config.is_vis:
    print("\n[Check]: Loading visualization interface (checking fit for large-scale terrain)...")
    
    # Voxel downsampling of high-density point cloud (0.3m spacing, only for smooth rendering)
    print("Performing spatial sparsification on high-density point cloud for rendering...")
    vis_pcd_t = full_cloud.voxel_down_sample(0.3)
    vis_points = vis_pcd_t.point.positions.numpy()
    rel_heights_down = vis_pcd_t.point["rel_heights"].numpy().flatten()
    
    # Convert to traditional Open3D visualization format
    vis_pcd = vis_pcd_t.to_legacy()
    
    # Initialize colors: Non-ground points (net height > 0.3m) rendered in light gray
    vis_colors = np.full((len(vis_points), 3), [0.8, 0.8, 0.8])
    
    # Ground points (net height <= 0.3m) rendered in black for contrast with the colored terrain mesh
    ground_mask_down = rel_heights_down <= 0.3
    vis_colors[ground_mask_down] = [0.0, 0.0, 0.0]
    vis_pcd.colors = o3d.utility.Vector3dVector(vis_colors)
    
    # Generate 3D undulating terrain triangular mesh
    print("Generating full-scene DTM triangular mesh...")
    dtm_mesh = create_dtm_mesh(segmenter.dtm_data)
    
    print("Opening 3D visualization window...")
    print("[Operation Tips]：")
    print("1. The colored undulating surface is the generated global DTM, black points are extracted ground points, and gray points are vegetation above the ground.")
    print("2. Check if the black point cloud fits tightly against the colored DTM surface.")
    print("3. Check if the transitions on slopes are smooth without sudden seams.")
    
    o3d.visualization.draw_geometries(
        [vis_pcd, dtm_mesh], 
        window_name="Global DTM Verification (CSF + IDW)", 
        mesh_show_back_face=True
    )
    
    del vis_pcd, dtm_mesh, vis_pcd_t


# Global trunk localization, DBH fitting, and "Cylinder Seed Source" extraction
print("\n" + "="*50)
print("Starting trunk feature extraction logic...")

# Instantiate trunk extractor
trunk_extractor = TreeTrunkSegmentationPlus(
    eps_2d=0.20,    # 2D DBSCAN radius (2 voxel widths)
    min_samples_2d=20,    # 2D minimum point count
    eps_3d=0.25,    # 3D DBSCAN radius
    min_samples_3d=5,
    min_cluster_height=1.5,    # Trunk slice extension height check threshold
    layer_height=0.20,    # high-fidelity 20cm fitting layer thickness
    layer_overlap=0.10,    # 10cm fitting layer overlap (sliding step 10cm, generating 19 slice layers in the 1.0m~3.0m range)
    max_std_diameter=0.10,    # 10cm diameter variation standard deviation, highly compatible with sparse voxel jitter
    max_std_position=None,    # Directly disable center standard deviation, rely on regression equation to correct slanted trees
    std_num_layers=6,    # Fitting consistency in any 6 layers is considered, automatically stripping bad layers and tolerating forks
    seed_diameter_factor=1.05,    # Cylinder seed source horizontal expansion factor
    seed_height_range=(1.0, 1.6)    # Cylinder seed source height sampling range (around breast height 1.3m)
)

# Extract full-scene verticality features from point cloud
np_vert = full_cloud.point.verticality.numpy().flatten()

# Execute extraction algorithm
start_time = time.time()
# Pass verticality features as the third parameter for internal denoising cleaning of slice layers
trunk_centers, trunk_diameters, cylinder_seed_mask, trunk_points_mask = \
    trunk_extractor.find_tree_trunks(points, rel_heights, verticality=np_vert)
print(f"[Time Stats] Global trunk localization and seed extraction duration: {time.time() - start_time:.2f} seconds.")

# Save extracted single-tree feature parameters
trunk_features_path = os.path.join(Config.output_dir, "Wytham_stem_features.npz")
np.savez(
    trunk_features_path, 
    centers=trunk_centers, 
    diameters=trunk_diameters
)
print(f"Trunk DBH feature data successfully exported to: {trunk_features_path}")


# [Visualization] Debug-level trunk fitting check (Red=Cylinder Seeds, Light-Gray=All 1-4m Candidate Trunks)
if Config.is_vis:
    print("\n[Check]: Loading debug-level visualization interface...")
    
    # Create spatial merge mask: includes original slice full-point cloud in 1~4m range, and cylinder seed points at 1.3m
    raw_slice_mask = (rel_heights >= 1.0) & (rel_heights <= 4.0)
    render_mask = raw_slice_mask | cylinder_seed_mask
    
    # Perform spatial filtering, discarding high-altitude leaves and ground noise
    render_points = points[render_mask]
    render_seed_mask = cylinder_seed_mask[render_mask]
    
    print(f"[Info] Points used for debug rendering after filtering: {len(render_points):,}")
    
    # Default: 1~4m original slice full-point cloud colored light gray [0.85, 0.85, 0.85], representing the true spatial distribution of all physical trunks
    colors_render = np.full((len(render_points), 3), [0.85, 0.85, 0.85], dtype=np.float32)
    
    # Seed points: 1.3m cylinder seed points that successfully passed multi-layer standard deviation checks and fitting are colored bright red [1.0, 0.0, 0.0]
    colors_render[render_seed_mask] = [1.0, 0.0, 0.0]
    
    # Create temporary Tensor PointCloud for visualization
    vis_cloud = o3d.t.geometry.PointCloud()
    vis_cloud.point.positions = o3d.core.Tensor(render_points, o3d.core.float32)
    vis_cloud.point.colors = o3d.core.Tensor(colors_render, o3d.core.float32)
    
    # Perform slightly mild spatial downsampling 
    # 0.1m spacing, maintaining the physical contour of the original 10cm point cloud, while facilitating rotation and zooming
    print("Performing spatial downsampling on debug point cloud...")
    vis_pcd_t = vis_cloud.voxel_down_sample(0.1)
    vis_pcd = vis_pcd_t.to_legacy()
    
    print("Opening 3D visualization window...")
    print("[Visulization Tips]：")
    print("1. Light gray point cloud: [All physical trunk point clouds] in the 1~4m height range across the full scene (including trees filtered out by fitting).")
    print("2. Red ring point cloud: Cylinder seeds successfully marked at $1.3m$ after passing the health check.")
    
    o3d.visualization.draw_geometries(
        [vis_pcd], 
        window_name="Debug-level Stem Verification (Red=Seeds, Light-Gray=All 1-4m Stems)", 
        mesh_show_back_face=True
    )
    
    del vis_pcd, vis_pcd_t, vis_cloud, colors_render, render_points


# Global ground isolation and background labeling
print("\n" + "="*50)
print("Executing global ground isolation and background label assignment...")

# Set ground normalized height threshold (Points with net height below 0.5m are forcibly classified as ground/background)
ground_buffer = 0.5  

# Create global single-tree instance prediction label array, initialized to -1 (default all are ground background)
final_labels = np.full(len(points), -1, dtype=np.int32)

# Extract masks for ground points and vegetation points
ground_mask = rel_heights < ground_buffer
veg_mask = ~ground_mask

# In our final_labels array, keep ground points as -1 
# Subsequent region growing algorithms will only perform label competition and spreading within the veg_mask (vegetation point cloud subset)
print(f"[Isolation Result] Ground/Background Points (Label=-1): {np.sum(ground_mask):,} ({np.sum(ground_mask)/len(points)*100:.2f}%)")
print(f"[Isolation Result] Vegetation Points to Grow: {np.sum(veg_mask):,} ({np.sum(veg_mask)/len(points)*100:.2f}%)")


#  [Visualization] Ground Isolation Verification (Ground Isolation)
if Config.is_vis:
    print("\n[Check]: Displaying ground isolation effect (Orange: Ground, Gray: Vegetation)...")
    
    # Rendering 52 million points on Windows has high overhead, randomly sample 2,000,000 points for smooth display
    indices = np.arange(len(points))
    vis_idx = np.random.choice(indices, size=min(len(indices), 2000000), replace=False)
    
    vis_pcd_legacy = o3d.geometry.PointCloud()
    vis_pcd_legacy.points = o3d.utility.Vector3dVector(points[vis_idx])
    
    # Initialize color array to all gray [0.7, 0.7, 0.7] (representing vegetation)
    vis_colors = np.full((len(vis_idx), 3), [0.7, 0.7, 0.7])
    
    # Filter out ground points in the sampled points and color them orange [1.0, 0.5, 0.0]
    local_ground_mask = ground_mask[vis_idx]
    vis_colors[local_ground_mask] = [1.0, 0.5, 0.0]
    
    vis_pcd_legacy.colors = o3d.utility.Vector3dVector(vis_colors)
    
    print("[Operation Tips]：")
    print("1. Orange areas: Points forcibly isolated and classified as ground background (Label = -1).")
    print("2. Gray areas: Vegetation skeleton and tree crown points retained for subsequent 3D region growing.")
    
    o3d.visualization.draw_geometries(
        [vis_pcd_legacy], 
        window_name="Ground Isolation Verification (Orange: Ground, Gray: Veg)",
        mesh_show_back_face=True
    )
    
    del vis_pcd_legacy, vis_colors, vis_idx, indices


# Pure NumPy ultra-fast 3D region growing competition phase
print("\n" + "="*50)
print("Starting pure NumPy-level 3D region growing competition algorithm...")

# [Dual-Channel Switch]: Strongly recommended to set to True for 52 million point clouds
down_sample = True  

if not down_sample:
    # Channel 1: Direct growth at original full resolution (Suitable for small areas, lightweight point clouds)
    print("[Channel Selection] Running direct 3D region growing at [Original Full Resolution]...")
    grower = TreeInstanceSegmentation(
        voxel_size=0.18,    # riginal 18cm growth step
        z_scale=2.0,
        min_total_assignment_ratio=0.00001,
        min_tree_assignment_ratio=0.3,
        max_search_radius=1.5,    # Maximum allowed obstacle-crossing radius
        decrease_search_radius_after_num_iter=10,
        max_iterations=500,
        early_stop_ratio=0.9900    # High-precision adaptive early stop set to 99.0%
    )
    
    # Extract full high-precision verticality
    veg_verticality_full = full_cloud.point.verticality.numpy().flatten()
    
    start_grow_time = time.time()
    segmented_labels = grower.segment_cloud(
        points=points,
        rel_heights=rel_heights,
        trunk_centers=trunk_centers,
        cylinder_seed_mask=cylinder_seed_mask,
        verticality=veg_verticality_full        # Pass verticality to eliminate low-height shrubs
    )
    print(f"[Time Stats] Direct region growing duration: {time.time() - start_grow_time:.2f} seconds.")
    
else:
    # Channel 2: High-precision seed protection + 0.2m canopy dimensionality reduction + 1-NN high-speed back-projection 
    # Channel 2 is golden solution for large scenes
    print("[Channel Selection] Running in [High-precision seed protection + 0.2m canopy dimensionality reduction + 1-NN high-speed back-projection] dual-channel mode...")
    
    # Extract vegetation point cloud subset (Relative elevation >= 0.5m)
    veg_mask = rel_heights >= 0.5
    veg_points = points[veg_mask]
    veg_rel_heights = rel_heights[veg_mask]
    veg_cylinder_seed_mask = cylinder_seed_mask[veg_mask]
    
    # Align and extract original high-fidelity verticality features
    veg_verticality = full_cloud.point.verticality.numpy().flatten()[veg_mask]
    
    # Protect high-precision seeds: Completely separate 10cm original seed points from ordinary canopy points
    seed_pts = veg_points[veg_cylinder_seed_mask]
    seed_rel_heights = veg_rel_heights[veg_cylinder_seed_mask]
    seed_verticality = veg_verticality[veg_cylinder_seed_mask]    # Protect seed point verticality
    
    # Ordinary canopy points to be downsampled
    canopy_pts = veg_points[~veg_cylinder_seed_mask]
    canopy_rel_heights = veg_rel_heights[~veg_cylinder_seed_mask]
    canopy_verticality = veg_verticality[~veg_cylinder_seed_mask]
    
    # Perform 0.20m downsampling on ordinary canopy
    print("Performing 0.20m voxel adaptive dimensionality reduction downsampling on ordinary canopy points...")
    canopy_pcd = o3d.geometry.PointCloud()
    canopy_pcd.points = o3d.utility.Vector3dVector(canopy_pts.astype(np.float64))
    down_canopy_pcd = canopy_pcd.voxel_down_sample(0.20)
    down_canopy_pts = np.asarray(down_canopy_pcd.points)
    
    # Align elevation and verticality back to 0.2m space
    print("Aligning attributes of downsampled ordinary canopy...")
    kd_tree_canopy = cKDTree(canopy_pts)
    _, nearest_idx = kd_tree_canopy.query(down_canopy_pts, k=1, workers=-1)
    down_canopy_rel_heights = canopy_rel_heights[nearest_idx]
    down_canopy_verticality = canopy_verticality[nearest_idx]    # Align verticality
    
    # Spatial confluence (Concatenate coordinates, net heights, and verticality)
    growing_points = np.concatenate([seed_pts, down_canopy_pts], axis=0)
    growing_rel_heights = np.concatenate([seed_rel_heights, down_canopy_rel_heights], axis=0)
    growing_verticality = np.concatenate([seed_verticality, down_canopy_verticality], axis=0)
    
    # Seed mask marking
    growing_cylinder_seed_mask = np.zeros(len(growing_points), dtype=bool)
    growing_cylinder_seed_mask[:len(seed_pts)] = True
    
    print(f"[Confluence Complete] Growing point cloud scale: {len(growing_points):,} points")
    
    # Start 20cm step high-energy growth (Parameters aligned and amplified: 0.2m voxel paired with 0.36m growth step)
    grower = TreeInstanceSegmentation(
        voxel_size=0.36,    # Diagonal protection: 0.2m voxel aligned with 0.36m step
        z_scale=2.0,
        min_total_assignment_ratio=0.00001,
        min_tree_assignment_ratio=0.3,
        max_search_radius=1.5,    # Body diagonal protection: 0.2m voxel aligned with 0.36m step
        decrease_search_radius_after_num_iter=10,
        max_iterations=500,
        early_stop_ratio=0.990    # High-precision adaptive early stop adjusted down to 99.0%
    )
    
    start_grow_time = time.time()
    growing_labels = grower.segment_cloud(
        points=growing_points,
        rel_heights=growing_rel_heights,
        trunk_centers=trunk_centers,
        cylinder_seed_mask=growing_cylinder_seed_mask,
        verticality=growing_verticality    # Pass aligned verticality parameters
    )
    print(f"[Time Stats] Dimensionality-reduced region growing duration: {time.time() - start_grow_time:.2f} seconds.")
    
    # 1-NN Spatial Label High-Speed Back-Projection (Label Propagation): Project grown labels back to original points
    print("Starting dual-channel 1-NN spatial label high-speed back-projection...")
    start_prop_time = time.time()
    # Build spatial search tree for growing points
    prop_tree = cKDTree(growing_points)
    # Query 1 nearest neighbor for original vegetation points
    _, prop_idx = prop_tree.query(veg_points, k=1, workers=-1)
    
    # Original vegetation label inheritance
    veg_labels = growing_labels[prop_idx]
    
    # Summarize and align final single-tree labels for full-scene points (Ground and unclaimed points are -1)
    segmented_labels = np.full(len(points), -1, dtype=np.int32)
    segmented_labels[veg_mask] = veg_labels
    print(f"[Time Stats] 1-NN spatial label back-projection duration: {time.time() - start_prop_time:.2f} seconds.")



# Call encapsulated single-tree 3D connected component adaptive variable radius (D_alt) secondary expansion purification 
# Rapidly call grower.clean_labels instance method, passing DTM net height, absolute coordinates, original seeds, and trunk centers
segmented_labels_cleaned = grower.clean_labels(
    points=points,
    rel_heights=rel_heights,
    segmented_labels=segmented_labels,
    cylinder_seed_mask=cylinder_seed_mask,
    trunk_centers=trunk_centers
)

#  Save the final calculated, connected-component-purified single-tree labels (-1=background ground, 0~M-1=single trees) back to Open3D and save the result PLY
full_cloud.point["tree_id"] = o3d.core.Tensor(segmented_labels_cleaned.reshape(-1, 1), o3d.core.int32)
final_output_ply = os.path.join(Config.output_dir, "Wytham_full_area_segmented_final.ply")
print(f"Exporting final single-tree segmentation result PLY: {final_output_ply} ...")
o3d.t.io.write_point_cloud(final_output_ply, full_cloud)
print("[File Save Complete]")


# [Visualization] Panoramic single-tree segmentation effect display (Color: Single trees, Background: Ground removed)
if Config.is_vis:
    print("\n[Check]: Loading panoramic single-tree segmentation visualization interface (automatically removing ground background)...")
    
    # Get mask for non-ground points (Relative net height >= 0.5m, to display the cleanest aerial tree crown panorama)
    non_ground_render_mask = rel_heights >= 0.5
    render_points = points[non_ground_render_mask]
    render_labels = segmented_labels_cleaned[non_ground_render_mask] # 渲染经连通域净化后的干净单木！
    
    print(f" [Info] After removing ground, number of points participating in full-scene colored single-tree rendering: {len(render_points):,}")
    
    # Minimalist random coloring logic: Generate high-saturation random colors for trees
    n_trees = len(trunk_centers)
    np.random.seed(42)    # Fix random seed to ensure reproducible results
    random_colors = np.random.uniform(0.15, 1.0, size=(n_trees, 3)).astype(np.float32)
    
    # Initialize colors: Unsuccessfully segmented stray leaves/noise rendered in very light gray background [0.85, 0.85, 0.85]
    colors_render = np.full((len(render_points), 3), [0.85, 0.85, 0.85], dtype=np.float32)
    
    # Assign corresponding random colors to points successfully segmented as single trees (label >= 0)
    segmented_mask = render_labels >= 0
    colors_render[segmented_mask] = random_colors[render_labels[segmented_mask]]
    
    # Inject colors and coordinates into temporary Open3D Tensor PointCloud
    vis_cloud = o3d.t.geometry.PointCloud()
    vis_cloud.point.positions = o3d.core.Tensor(render_points, o3d.core.float32)
    vis_cloud.point.colors = o3d.core.Tensor(colors_render, o3d.core.float32)
    
    # Voxel downsampling (0.20m, providing smooth rendering on Windows 10)
    print("Performing spatial downsampling on 3D colored single-tree point cloud...")
    vis_pcd_t = vis_cloud.voxel_down_sample(0.20)
    vis_pcd = vis_pcd_t.to_legacy()
    
    print("Opening 3D panorama rendering window...")
    print("[Operation Tips]:")
    print("1. Orange ground is completely hidden, the screen only shows colored single trees.")
    print("2. Colored point cloud: Single-tree bodies with shrubs stripped.")
    print("3. Light gray point cloud: Isolated leaves in the forest that failed trunk consistency checks, did not grow, or were stripped leaking shrubs.")
    
    o3d.visualization.draw_geometries(
        [vis_pcd], 
        window_name="Full Forest Instance Segmentation (Colors=Trees, Ground=Removed)", 
        mesh_show_back_face=True
    )
    
    del vis_pcd, vis_pcd_t, vis_cloud, colors_render, render_points, render_labels


# Accuracy assessment and Table 3 style academic report output
print("\n" + "="*50)
print("Starting Wytham evaluation subset blind box accuracy assessment (based on Classification == 4)...")

# Strictly refresh latest data attributes
np_class = full_cloud.point.classification.numpy().flatten()
np_gt_tree_id = full_cloud.point.gt_tree_id.numpy().flatten()
np_tree_id = full_cloud.point.tree_id.numpy().flatten()
np_pos = full_cloud.point.positions.numpy()

# 2. Spatial cropping filter: Evaluate collision only in Classification == 4 areas (Strictly refine Set A and Set B)
eval_mask = np_class == 4
eval_gt = np_gt_tree_id[eval_mask]
eval_pred = np_tree_id[eval_mask]   # Automatically obtain single-tree ID array after connected component purification (already saved back to full_cloud)
eval_coords = np_pos[eval_mask]

# Execute Hungarian maximum uniqueness collision matching (Using academic standard min_iou = 0.5)
matched_pairs, _, unique_preds, unique_gts = perform_tree_matching(
    eval_gt, 
    eval_pred, 
    min_iou=0.5
)

# Calculate Table 3 single-tree detection and segmentation metrics
n_matched = len(matched_pairs)
n_gt = len(unique_gts)    # Strictly remove gt_id=0, remaining true single-tree count
n_pred = len(unique_preds)  # Strictly remove pred_id=-1, remaining predicted single-tree count

completeness = n_matched / n_gt if n_gt > 0 else 0.0
omission_rate = 1.0 - completeness
commission_rate = (n_pred - n_matched) / n_pred if n_pred > 0 else 0.0
f1_det = 2 * (1.0 - commission_rate) * completeness / (2.0 - commission_rate - omission_rate + 1e-8)

all_prec = [m['prec'] for m in matched_pairs]
all_rec = [m['rec'] for m in matched_pairs]
all_cov = [m['iou'] for m in matched_pairs]

print("\n" + "-"*50)
print("[Table 3 Style Academic Report - Wytham Assessment Accuracy]")
print("-"*50)
print(f"Instance Detection (GT Trees: {n_gt}, Pred Trees: {n_pred}):")
print(f"  Completeness (C):      {completeness*100:.2f}%")
print(f"  Omission (E_om):       {omission_rate*100:.2f}%")
print(f"  Commission (E_com):    {commission_rate*100:.2f}%")
print(f"  F1 Score (Det):        {f1_det*100:.2f}%")
print(f"Instance Segmentation (N_matches: {n_matched} at IoU >= 0.5):")
print(f"  Precision (Prec):      {np.mean(all_prec)*100:.2f}%" if n_matched > 0 else "  Precision (Prec):      N/A")
print(f"  Recall (Rec):          {np.mean(all_rec)*100:.2f}%" if n_matched > 0 else "  Recall (Rec):          N/A")
print(f"  Coverage/mIoU (Cov):   {np.mean(all_cov)*100:.2f}%" if n_matched > 0 else "  Coverage/mIoU (Cov):   N/A")
print("-"*50)


# Plot stratified spatial accuracy trend chart
if n_matched > 0:
    print("Analyzing stratified spatial trend chart data...")
    
    # Establish 1-to-1 trunk center dictionary mapping
    stem_map = {i: trunk_centers[i] for i in range(len(trunk_centers))}
    
    # Align translated function parameters
    xy_res, z_res = evaluate_partitioned_accuracy(
        eval_coords, 
        eval_gt, 
        eval_pred, 
        matched_pairs, 
        stem_map
    )
    
    # Start Matplotlib trend chart plotting
    print("Plotting and saving stratified spatial accuracy trend chart...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Precision", "Recall", "Coverage (IoU)"]
    x_bins = np.arange(1, 11)

    for i in range(3):
        # XY direction: Radial distance trend (expanding outward)
        axes[i].plot(x_bins, xy_res[:, i]*100, 'o-', label="XY (Radial: Inner -> Outer)", color='crimson', lw=2)
        # Z direction: Vertical direction trend (climbing upward)
        axes[i].plot(x_bins, z_res[:, i]*100, 's--', label="Z (Vertical: Bottom -> Top)", color='royalblue', lw=2)
        
        axes[i].set_title(titles[i], fontsize=14, fontweight='bold')
        axes[i].set_ylim(20, 105)
        axes[i].set_xticks(x_bins)
        axes[i].set_xlabel("Partition Bin", fontsize=11)
        axes[i].set_ylabel("Accuracy (%)", fontsize=11)
        axes[i].grid(True, ls='--', alpha=0.5)
        axes[i].legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    #  Automatically save high-fidelity academic trend chart to step2_results directory
    trend_plot_path = os.path.join(Config.output_dir, "Wytham_segmentation_trends.png")
    plt.savefig(trend_plot_path, dpi=300)
    print(f"Trend chart successfully saved to: {trend_plot_path}")
    plt.show()
else:
    print("[Tip] Trend chart plotting skipped due to no successfully matched single trees.")