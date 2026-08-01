# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import open3d as o3d
from pathlib import Path
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.cloud_io import CloudIO
from utils.logger import ExperimentLogger
from utils.cloud_processing import crop_cloud
from utils.registration import Registration


class Config:
    # Input paths (pointing to the preprocessed Processed folder)
    dataset_root = os.path.join(BASE_DIR, "evo_example_dataset_processed")
    uav_cloud_path = os.path.join(dataset_root, "uav_cloud.ply")
    mls_tiles_folder = os.path.join(dataset_root, "tiles")
    
    # Output paths after preprocessing
    output_base = os.path.join(BASE_DIR, "step2_results")
    
    # EVO-specific offset
    offset = np.array([-399000.0, -6786000.0, 0.0])
    
    # Registration thresholds
    icp_threshold = 0.85    # ICP score must be higher than this value to be considered successful
    min_dist_between_peaks = 2.5    # Minimum distance between peaks for tree top extraction (meters)
    max_cliques = 5    # Maximum number of matches attempted by the Maximum Clique Algorithm
    
    # Debug switch
    debug = True                  # If enabled, 3D windows will pop up to show matching collisions and DTM alignment
    log_level = "DEBUG"
    
    # DTM Geometric Control (CSF + IDW)
    # he parameter package is passed losslessly to the GroundSegmentation module, 
    # providing continuous 3D terrain elevation modeling
    ground_settings = {
        # Core parameters for CSF filtering
        "csf_threshold": 0.5,    # Ground classification threshold (meters)
        "csf_resolution": 1.0,    # Global cloth mesh resolution (1.0m to provide smooth terrain transition)
        "csf_rigidness": 2,    # Cloth rigidity (1-Low, 2-Medium, 3-High)
        "csf_correct_steep_slope": False,    # Whether to perform post-processing correction for steep slopes
        "csf_iterations": 500,    # Maximum simulation iterations
        
        # IDW large-scale terrain interpolation parameters
        "dtm_resolution": 0.5,    # DTM grid resolution (0.5m to ensure elevation accuracy)
        "dtm_k": 20,    # Number of nearest neighbor ground points searched when interpolating each grid point
        "dtm_p": 1.0,    # Power exponent for IDW distance weights
        "dtm_voxel_size": 0.3    # Voxel size for sparsifying ground points before constructing the DTM (meters)
    }

    # Tree Trunk Feature Circle Fitting Extraction Control
    # The parameter package is passed losslessly to the TreeTrunkSegmentationPlus module 
    # to estimate unbiased absolute circle centers in registration
    trunk_settings = {
        # DBSCAN clustering sensitivity
        "eps_2d": 0.20,    # 2D neighborhood radius (meters)
        "min_samples_2d": 20,    # Minimum number of points in 2D
        "eps_3d": 0.25,    # 3D neighborhood radius (meters)
        "min_samples_3d": 5,    # Minimum number of points in 3D
        
        # Multi-layer algebraic circle fitting and geometric inspection parameters
        "min_cluster_height": 1.5,    # Minimum vertical extension height of trunk clusters (meters)
        "layer_height": 0.20,    # Fitting layer thickness (meters)
        "layer_overlap": 0.10,    # Layer overlap sliding step size (meters)
        "max_std_diameter": 0.10,    # Standard deviation of 10cm diameter variation
        "max_std_position": None,    # Disable circle center standard deviation filtering, rely on univariate linear regression to solve slanted tree issues
        "std_num_layers": 6,    # Arbitrarily select 6 layer subsets for inspection; if passed, keep them
        
        # EVO trunk slice height and verticality filtering range
        "h_min": 1.0,    # Lower limit of trunk slice height (meters)
        "h_max": 4.0,    # Upper limit of trunk slice height (meters)
        "verticality_thr": 0.65,    # Verticality filtering threshold for upright trunks
        "fit_min_z": 1.0,    # Lower limit for multi-layer circle fitting regression height (meters)
        "fit_max_z": 3.0,    # Upper limit for multi-layer circle fitting regression height (meters)
        
        # Cylinder seed sampling interval (near breast height 1.3m)
        "seed_diameter_factor": 1.10,    # Adaptive cylinder seed horizontal expansion by 10% tolerance
        "seed_height_range": (1.0, 1.6)    # Seed sampling interval
    }


o3d.utility.random.seed(42)
os.makedirs(Config.output_base, exist_ok=True)

logger = ExperimentLogger(base_dir=os.path.join(BASE_DIR, "logs"), log_pointclouds=True)

cio = CloudIO(offset=Config.offset)

#  Load UAV reference point cloud (global coordinate reference system, automatically applying offset)
print(f"Loading UAV reference cloud: {os.path.basename(Config.uav_cloud_path)}")
uav_cloud = cio.load_cloud(Config.uav_cloud_path)

#  Get tiles to be processed (only process .ply files that actually exist in the folder)
tile_files = sorted([f for f in Path(Config.mls_tiles_folder).iterdir() 
                     if f.suffix == ".ply" and f.name.startswith("tile")])

print(f"Found {len(tile_files)} tiles to register: {[f.name for f in tile_files]}")

registration_results = {}    # Store transformation matrices and scores for each tile

# Loop through each tile and execute the registration flow
for tile_path in tile_files:
    tile_name = tile_path.name
    print("\n" + "=" * 50)
    print(f"Registering: {tile_name}")
    
    # Set an independent log subfolder for the current tile
    logger.set_leaf_logging_folder(tile_path.stem)
    
    # Load Tile (automatically applies offset, coordinates return near origin for precise calculation)
    original_mls_tile = cio.load_cloud(str(tile_path))
    
    # Crop local UAV reference point cloud, expanding crop by 20m to ensure sufficient landmark trees are included
    cropped_uav = crop_cloud(uav_cloud, original_mls_tile, padding=20)

    # Instantiate the registration scheduling module
    reg_module = Registration(
        uav_cloud=cropped_uav,
        mls_cloud=original_mls_tile,
        cloud_io=cio,
        ground_segmentation_method="default",
        correspondence_matching_method="graph",
        mls_feature_extraction_method="tree_segmentation",
        icp_fitness_threshold=Config.icp_threshold,
        min_distance_between_peaks=Config.min_dist_between_peaks,
        max_number_of_clique=Config.max_cliques,
        logger=logger,
        correspondence_graph_distance_threshold=0.2,
        maximum_rotation_offset=1.6,
        ground_params=Config.ground_settings,
        trunk_params=Config.trunk_settings,
        debug=Config.debug
    )
    
    # Execute core registration flow
    success = reg_module.registration()
    
    if success:
        print(f"[Success] ICP Fitness: {reg_module.best_icp_fitness_score:.4f}")
        
        # Save registered point cloud (local_coordinates=False means restoring to UTM global large coordinates)
        transformed_tile = reg_module.transform_cloud(original_mls_tile)
        save_path = os.path.join(Config.output_base, tile_name)
        cio.save_cloud(transformed_tile, save_path, local_coordinates=False)
        
        # Save estimated absolute 1.3m breast-height center points and diameters (.npz)
        if reg_module.trunk_centers is not None and len(reg_module.trunk_centers) > 0:
            # Coordinate transformation: Project unbiased fitted circle centers from local coordinates to the registered reference system using the transformation matrix
            pts = reg_module.trunk_centers
            ones = np.ones((pts.shape[0], 1))
            pts_h = np.hstack([pts, ones])
            transformed_centers = (pts_h @ reg_module.transform.T)[:, :3]

            # Save NPZ data (add diameters attribute, inheriting fitted breast-height diameters for subsequent adaptive cylinder seed generation)
            npz_name = tile_path.stem + "_trunk_info.npz"
            npz_path = os.path.join(Config.output_base, npz_name)
            
            np.savez_compressed(
                npz_path, 
                centroids=transformed_centers,    # Real physical breast-height circle centers after registration (3D coordinates)
                diameters=reg_module.trunk_diameters,    # Inherited high-fidelity single-tree breast-height diameters (1D scalar array)
                scores=reg_module.trunk_scores,    # Registration feature confidence (ones placeholder)
                raw_local_centers=pts,    # Original local circle centers (backup)
                transform=reg_module.transform    # Local to global registration transformation matrix (4x4)
            )
            print(f"[Info] Trunk centers and diameter features saved to: {npz_name} ({len(transformed_centers)} trees.)")
        
    else:
        print("[Failed] ICP registration score threshold not reached or insufficient feature edge matches.")

    # Record registration results and backup to the results dictionary
    registration_results[tile_name] = {
        "success": success,
        "transform": reg_module.transform,
        "icp_fitness": reg_module.best_icp_fitness_score,
        "raw_trunk_centers": reg_module.trunk_centers,
        "trunk_scores": reg_module.trunk_scores,
        "trunk_diameters": reg_module.trunk_diameters
    }

# Serialize and export the registration dictionary (.pkl file, providing absolute constraint input for step3 pose graph optimization)
results_pkl_path = os.path.join(Config.output_base, "registration_results.pkl")
with open(results_pkl_path, "wb") as f:
    pickle.dump(registration_results, f)

print("\n" + "=" * 50)
print("All tile local registration processing completed!")
print(f"Registration result summary saved to: {results_pkl_path}")
