# -*- coding: utf-8 -*-
"""
Created on May 2026
@author: User
"""

import numpy as np
import open3d as o3d
import os
import sys
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(UTILS_DIR)
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.ground_segmentation import GroundSegmentation, create_dtm_mesh



class VerticalRegistration:
    """
    Enhanced Vertical Alignment Class:
    Based on Cloth Simulation Filtering (CSF) and IDW continuous interpolation of undulating terrain,
    calculates the vertical displacement (tz) between UAV point clouds and MLS point clouds.
    """

    def __init__(
        self,
        uav_cloud,
        mls_cloud,
        cloud_io,
        ground_segmentation_method,
        logger,
        debug=False,
        ground_params=None,    # Receives ground parameter dictionary from Config
    ):
        self.uav_cloud = uav_cloud
        self.mls_cloud = mls_cloud
        self.cloud_io = cloud_io
        self.logger = logger
        self.debug = debug
        
        # Parameter alignment protection: If no parameter dictionary is passed, use default stable DTM settings
        self.g_params = ground_params if ground_params is not None else {
            "csf_threshold": 0.5,
            "csf_resolution": 1.0,
            "csf_rigidness": 2,
            "csf_correct_steep_slope": False,
            "csf_iterations": 500,
            "dtm_resolution": 0.5,
            "dtm_k": 20,
            "dtm_p": 1.0,
            "dtm_voxel_size": 0.3
        }

    def process(self):
        """
        Median Undulating Terrain Projection Method:
        1. Instantiate two independent GroundSegmentation engines 
           to preserve the DTM undulating structure of each surface.
        2. Treat MLS ground points as the group to be moved, 
           querying their vertical deviation to the continuous UAV terrain via bilinear interpolation.
        3. Apply height median correction to resist surface noise and 
           output a high-noise-resistant Z-axis offset.
        """
        if self.logger:
            self.logger.info("Starting vertical alignment based on CSF undulating terrain model...")

        #  Convert to float64 NumPy arrays to avoid floating-point precision truncation
        uav_points = self.uav_cloud.point.positions.numpy().astype(np.float64)
        mls_points = self.mls_cloud.point.positions.numpy().astype(np.float64)

        # Initialize elevation processing engines
        uav_engine = GroundSegmentation(
            csf_threshold=self.g_params.get("csf_threshold", 0.5),
            csf_resolution=self.g_params.get("csf_resolution", 1.0),
            csf_rigidness=self.g_params.get("csf_rigidness", 2),
            csf_correct_steep_slope=self.g_params.get("csf_correct_steep_slope", False),
            csf_iterations=self.g_params.get("csf_iterations", 500),
            dtm_resolution=self.g_params.get("dtm_resolution", 0.5),
            dtm_k=self.g_params.get("dtm_k", 20),
            dtm_p=self.g_params.get("dtm_p", 1.0),
            dtm_voxel_size=self.g_params.get("dtm_voxel_size", 0.3)
        )
        
        mls_engine = GroundSegmentation(
            csf_threshold=self.g_params.get("csf_threshold", 0.5),
            csf_resolution=self.g_params.get("csf_resolution", 1.0),
            csf_rigidness=self.g_params.get("csf_rigidness", 2),
            csf_correct_steep_slope=self.g_params.get("csf_correct_steep_slope", False),
            csf_iterations=self.g_params.get("csf_iterations", 500),
            dtm_resolution=self.g_params.get("dtm_resolution", 0.5),
            dtm_k=self.g_params.get("dtm_k", 20),
            dtm_p=self.g_params.get("dtm_p", 1.0),
            dtm_voxel_size=self.g_params.get("dtm_voxel_size", 0.3)
        )

        # Run DTM modeling
        uav_ground_mask, _, _ = uav_engine.process(uav_points)
        mls_ground_mask, _, _ = mls_engine.process(mls_points)

        # Extract MLS ground point coordinates
        mls_ground_points = mls_points[mls_ground_mask]
        
        if len(mls_ground_points) == 0:
            raise ValueError("[Error] Failed to segment any ground points from the current MLS tile, vertical alignment failed.")

        # Continuous terrain bilinear elevation regression query
        # This calculation is essentially: mls_ground_points[:, 2] - terrain_z_uav
        discrepancies = uav_engine.get_normalized_heights(mls_ground_points)

        # Use median to find the optimal geometric fit position
        z_offset = -np.median(discrepancies)

        if self.logger:
            self.logger.debug(f"Z-axis offset derived from undulating DTM matching: {z_offset:.4f}m")

        # Debug-level undulating surface fit visualization verification
        if self.debug:
            print(f"[Debug] Vertical alignment completed. Offset: {z_offset:.3f}")
            
            # Convert to traditional Open3D point cloud format for rendering in interactive window
            uav_ground_vis = o3d.geometry.PointCloud()
            uav_ground_vis.points = o3d.utility.Vector3dVector(uav_points[uav_ground_mask])
            uav_ground_vis.paint_uniform_color([0.0, 0.0, 1.0])    # Blue: UAV reference ground points
            
            mls_ground_vis = o3d.geometry.PointCloud()
            mls_ground_vis.points = o3d.utility.Vector3dVector(mls_ground_points)
            mls_ground_vis.translate([0, 0, z_offset])    # Apply vertical deviation
            mls_ground_vis.paint_uniform_color([1.0, 0.5, 0.0])    # Orange: MLS ground points after offset
            
            try:
                # Generate colored gradient undulating triangle mesh based on DTM grid elevation
                dtm_mesh = create_dtm_mesh(uav_engine.dtm_data)
                
                print("[Tip] Opening undulating terrain matching alignment verification window:")
                print("1. Colored 3D plane: UAV reference undulating DTM terrain.")
                print("2. Blue points: UAV original extracted surface points.")
                print("3. Orange points: MLS local surface points after Z-axis offset correction.")
                print("4. Please rotate and zoom in to check if orange and blue points almost completely overlap with the colored undulating surface.")
                
                o3d.visualization.draw_geometries(
                    [dtm_mesh, uav_ground_vis, mls_ground_vis], 
                    window_name="Robust Z Alignment with Rolling Terrain DTM",
                    mesh_show_back_face=True
                )
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Visualization failed, possibly because necessary graphics rendering libraries are not installed: {e}")
                # Fallback scheme, display only point cloud
                o3d.visualization.draw_geometries(
                    [uav_ground_vis, mls_ground_vis], 
                    window_name="Z Alignment Verification (Points Only)"
                )

        return uav_engine, mls_engine, z_offset

# ==========================================
# Test Run Area (Independent Unit Test Execution)
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("Starting independent unit test for vertical_alignment.py...")
    print("=" * 60)

    # Define test input file paths (pointing to the original large coordinate data folder)
    uav_path = os.path.join(ROOT_DIR, "datasets", "evo_example_dataset", "uav_cloud.ply")
    tile_path = os.path.join(ROOT_DIR, "datasets", "evo_example_dataset", "tiles", "tile_1.ply")
    
    # Dedicated large coordinate offset
    evo_offset = np.array([-399000.0, -6786000.0, 0.0])
    
    if not os.path.exists(uav_path) or not os.path.exists(tile_path):
        print(f"[Error] Test files not found. Please ensure original data is saved in: {os.path.join(ROOT_DIR, 'datasets', 'evo_example_dataset')}")
        sys.exit(1)

    # Load original large coordinate point clouds and manually perform offset alignment
    print(f"Loading and correcting UAV coordinates: {os.path.basename(uav_path)}")
    uav_pcd = o3d.t.io.read_point_cloud(uav_path)
    uav_pcd.translate(o3d.core.Tensor(evo_offset, o3d.core.float32))

    print(f"Loading and correcting MLS coordinates: {os.path.basename(tile_path)}")
    mls_pcd = o3d.t.io.read_point_cloud(tile_path)
    mls_pcd.translate(o3d.core.Tensor(evo_offset, o3d.core.float32))

    # Configure test ground parameters
    test_ground_settings = {
        "csf_threshold": 0.5,
        "csf_resolution": 1.0,
        "csf_rigidness": 2,
        "csf_correct_steep_slope": False,
        "csf_iterations": 500,
        "dtm_resolution": 0.5,
        "dtm_k": 20,
        "dtm_p": 1.0,
        "dtm_voxel_size": 0.3
    }

    # Instantiate and start alignment calculation
    reg = VerticalRegistration(
        uav_cloud=uav_pcd,
        mls_cloud=mls_pcd,
        cloud_io=None,
        ground_segmentation_method="default",
        logger=None,
        debug=True,    # Enable Debug to pop up 3D rendering interface
        ground_params=test_ground_settings
    )

    try:
        uav_eng, mls_eng, tz = reg.process()
        print("\n" + "=" * 60)
        print("Unit test successful!")
        print(f"Estimated global vertical alignment bias tz: {tz:.4f} meters.")
        print("=" * 60)
    except Exception as e:
        print(f"\n[Test Failed] Vertical alignment exception occurred: {e}")