# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import open3d as o3d
import pickle
import gtsam
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.cloud_io import CloudIO, TileConfigReader
from utils.pose_graph import PoseGraph
from utils.graph_optimization import PoseGraphOptimization
from utils.graph_io import write_pose_graph, save_optimized_pointclouds


class Config:
    # Raw input paths (pointing to the preprocessed Processed folder)
    dataset_root = os.path.join(BASE_DIR, "evo_example_dataset_processed")
    uav_cloud_path = os.path.join(dataset_root, "uav_cloud.ply")
    mls_tiles_folder = os.path.join(dataset_root, "tiles")
    tiles_csv = os.path.join(mls_tiles_folder, "tiles.csv")
    
    # Output paths for Step 2
    step2_output_dir = os.path.join(BASE_DIR, "step2_results")
    results_pkl = os.path.join(step2_output_dir, "registration_results.pkl")
    
    # Final output paths after optimization
    output_folder = os.path.join(BASE_DIR, "step3_final_results")
    
    # EVO dataset specific offset
    offset = np.array([-399000.0, -6786000.0, 0.0])
    
    debug = True    # If enabled, detailed injection processes will be printed to the console
    show_clouds = True    # If enabled, the optimizer will call Open3D to display a comparison of point cloud poses before and after optimization
    
    # Noise settings (to control the relative weight ratio between prior factors and between factors in the factor graph)
    prior_noise = [1e-2, 1e-2, 1e-2, 0.01, 0.01, 0.01] 
    between_noise = [1e-4, 1e-4, 1e-4, 0.001, 0.001, 0.001]


os.makedirs(Config.output_folder, exist_ok=True)
cio = CloudIO(offset=Config.offset)

# Load registration results from Step 2
print("Loading Step 2 registration results summary...")
with open(Config.results_pkl, "rb") as f:
    registration_results = pickle.load(f)

# Read Tile grid structure configuration file (tiles.csv was synced to the processed directory in step1)
reader = TileConfigReader(Path(Config.tiles_csv), offset=None)

# Construct initial pose graph
print("Constructing initial pose graph...")
graph = PoseGraph()

# Iterate through grid configuration to add pose nodes (Nodes)
for tile_id, center in reader.coordinates:
    # Initial pose: Rotation set to identity rotation, 
    # position translation set to the tile's initial grid geometric center
    initial_pose = gtsam.Pose3(gtsam.Rot3(), center)
    graph.add_node(tile_id, (0, 0), initial_pose)
    
    # Load the corresponding tile point cloud with processed verticality
    tile_file = f"tile_{tile_id}.ply"
    tile_path = os.path.join(Config.mls_tiles_folder, tile_file)
    if os.path.exists(tile_path):
        cloud = cio.load_cloud(tile_path)
        graph.add_clouds(tile_id, cloud, tile_file)

# Add physical grid adjacency constraints (Between Factors)
print("Establishing 4-neighborhood grid constraints (high-rigidity Between Factors)...")
coords = reader.get_tiles_coordinates(Path(Config.mls_tiles_folder))
id_to_center = {c[0]: c[1] for c in coords}
existing_ids = set(id_to_center.keys())

# Define high-rigidity (1e6 weight) relative displacement information matrix 
# to pull adjacent tiles together and prevent tearing
info_matrix_between = np.eye(6) * 1e6
num_rows = reader.num_grid_rows
num_cols = reader.num_grid_cols

for tile_id, center in coords:
    row = tile_id // num_cols
    col = tile_id % num_cols

    # Connect "right" neighbor (col + 1)
    right_neighbor_id = row * num_cols + (col + 1)
    if (col + 1) < num_cols and right_neighbor_id in existing_ids:
        relative_pos = id_to_center[right_neighbor_id] - center
        relative_pose = gtsam.Pose3(gtsam.Rot3(), relative_pos)
        graph.add_edge(tile_id, right_neighbor_id, "in-between", relative_pose, info_matrix_between)

    # Connect "down" neighbor (row + 1)
    down_neighbor_id = (row + 1) * num_cols + col
    if (row + 1) < num_rows and down_neighbor_id in existing_ids:
        relative_pos = id_to_center[down_neighbor_id] - center
        relative_pose = gtsam.Pose3(gtsam.Rot3(), relative_pos)
        graph.add_edge(tile_id, down_neighbor_id, "in-between", relative_pose, info_matrix_between)

# Add local registration constraints (Prior Factors / Aerial Factors)
print("Injecting Step 2 registration priors (adaptive weighting mechanism based on confidence)...")
for tile_name, res in registration_results.items():
    # Only inject strong constraint edges for tiles where local registration is marked as successful
    if res['success']:
        try:
            tile_id = int(tile_name.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            print(f"[Skipped] Unable to parse filename ID: {tile_name}")
            continue

        transform = res['transform']
        fitness = res['icp_fitness']
        
        # Target pose = transform bias @ node's initial position
        initial_pose = graph.get_node_pose(tile_id)
        target_pose_mat = transform @ initial_pose.matrix()
        target_pose = gtsam.Pose3(target_pose_mat)
        
        # Weight mapping: exponentially widen the gap using registration score (Fitness^10 * 1e6)
        confidence_weight = np.power(fitness, 10) * 1e6
        info_matrix = np.eye(6) * confidence_weight
        
        graph.add_edge(tile_id, tile_id, "aerial", target_pose, info_matrix)
        
        if Config.debug:
            print(f"[Injected] Tile {tile_id}: Fitness={fitness:.4f}, Weight={confidence_weight:.1e}")

# Check if global constraints are completely lost
if len([r for r in registration_results.values() if r['success']]) == 0:
    print("\nWARNING: No local tile registration was successful. The pose graph will lack global reference!")

# Start GTSAM factor graph global optimization
print("\n" + "=" * 30)
print("Starting global consistency optimization (Pose Graph Optimization)...")
optimizer = PoseGraphOptimization(graph, debug=Config.debug, show_clouds=Config.show_clouds)
optimizer.optimize()
print("Global pose optimization completed!")

# Regenerate and heal seed points (.npz) for all tiles and achieve lossless saving of diameters (DBH)
print("Healing and updating tree trunk seed point coordinates based on global optimization results...")
for node_id, node in graph.nodes.items():
    tile_name = graph.get_node_cloud_name(node_id)
    if not tile_name or tile_name not in registration_results:
        continue
    
    res = registration_results[tile_name]
    raw_pts = res.get("raw_trunk_centers")
    scores = res.get("trunk_scores")
    
    # Extract diameters data archived in step2 directly from the pkl dictionary
    diameters = res.get("trunk_diameters")

    if raw_pts is not None and len(raw_pts) > 0:
        # Defense mechanism: If diameters are accidentally empty or length mismatch, 
        # automatically assign 0.25m default DBH to prevent downstream crashes
        if diameters is None or len(diameters) != len(raw_pts):
            diameters = np.full(len(raw_pts), 0.25)
            
        # Solve for the tile's "final alignment transformation matrix"
        init_pose = graph.get_initial_node_pose(node_id)
        opt_pose = graph.get_node_pose(node_id)
        
        # Incremental transform = optimized pose @ inverse of initial pose
        final_transform = opt_pose.matrix() @ np.linalg.inv(init_pose.matrix())
        
        # Homogeneous coordinate system transformation
        ones = np.ones((raw_pts.shape[0], 1))
        pts_h = np.hstack([raw_pts, ones])
        transformed_centers = (pts_h @ final_transform.T)[:, :3]

        # Overwrite and update NPZ seed feature data 
        # overwrite write to step2 directory for direct extraction in subsequent segmentation
        npz_name = Path(tile_name).stem + "_trunk_info.npz"
        npz_path = os.path.join(Config.step2_output_dir, npz_name)
        
        np.savez_compressed(
            npz_path, 
            centroids=transformed_centers,    # Global tree trunk center points after optimized alignment
            diameters=diameters,    # saved DBH data
            scores=scores,
            raw_local_centers=raw_pts,
            transform=final_transform
        )
        status = "Repaired" if not res['success'] else "Optimized"
        print(f"[{status}] {tile_name}: Generated and saved globally consistent seed points and diameter features.")

# Export globally final optimized aligned point clouds, 
# all feature attributes (verticality, etc.) will be automatically carried over
print("Generating and saving final optimized point clouds (automatically retaining verticality features)......")
save_optimized_pointclouds(Config.output_folder, True, graph, cio)

# Write optimized_graph.g2o pose graph file
write_pose_graph(graph, os.path.join(Config.output_folder, "optimized_graph.g2o"))

print(f"\n[Task Completed] Final optimized registered tile point clouds exported to: {Config.output_folder}")

# Academic metric evaluation report output
print("\n" + "=" * 50)
print("Starting registration algorithm accuracy metric evaluation (Table I & Table II)...")

uav_eval = cio.load_cloud(Config.uav_cloud_path)

# Generate report in Table I style (RMSE comparison before and after optimization)
print("\n[Generating Report] Table I: Accuracy comparison before and after global optimization")
table1_data = optimizer.calculate_paper_metrics(uav_eval)

# Generate report in Table II style (registration statistical distribution per tile)
print("\n[Generating Report] Table II: Final registration statistics per tile")
table2_data = optimizer.calculate_table2_metrics(uav_eval)

# Summary analysis
post_errors = [v[1] for v in table1_data.values()]
print(f"Finest tile error: {np.min(post_errors):.4f} meters")
print(f"Worst tile error:   {np.max(post_errors):.4f} meters")
print(f"Overall average error:   {np.mean(post_errors):.4f} meters")
