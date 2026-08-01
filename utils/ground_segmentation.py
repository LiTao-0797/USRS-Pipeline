# -*- coding: utf-8 -*-
"""
Ground Segmentation & DTM Generation Module
Integrated Cloth Simulation Filtering (CSF) and IDW Interpolation.
"""
import os
import numpy as np
import open3d as o3d
import CSF
from scipy.spatial import cKDTree
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    """
    System-level deep redirection:
    Redirect standard output to the system's null device (devnull),
    which can completely mute information directly output to the terminal by C/C++ compiled libraries (such as CSF).
    """
    # Backup the system's standard output file descriptor (stdout's fd defaults to 1)
    stdout_fd = 1
    saved_stdout_fd = os.dup(stdout_fd)
    try:
        # Open the system null device (os.devnull)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        # Redirect standard output to the null device
        os.dup2(devnull_fd, stdout_fd)
        os.close(devnull_fd)
        yield
    finally:
        # Restore standard output, making subsequent ordinary prints display normally in the console
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stdout_fd)


class GroundSegmentation:
    """
    Point cloud ground filtering and DTM construction module:
    1. Use CSF (cloth simulation filtering) algorithm to 
       blindly classify ground points without any classification label assistance.
    2. Use multi-threaded accelerated cKDTree + inverse distance weighting (IDW) algorithm to 
       construct large-scale continuous undulating terrain.
    3. Use bilinear interpolation to achieve smooth local elevation normalization.
    """
    def __init__(
        self,
        # CSF filtering parameters
        csf_threshold=0.5,    # Cloth-to-point cloud classification threshold (meters)
        csf_resolution=0.5,    # Cloth grid resolution (meters)
        csf_rigidness=2,    # Cloth rigidity (1-low, 2-medium, 3-high)
        csf_correct_steep_slope=False,    # Whether to perform post-processing correction for steep slopes
        csf_iterations=500,    # Maximum number of iterations
        
        # DTM interpolation parameters
        dtm_resolution=0.5,    # Grid resolution of DTM elevation map (meters)
        dtm_k=20,    # Number of nearest ground points to search for each grid point during interpolation
        dtm_p=1.0,    # Power exponent of IDW distance weights
        dtm_voxel_size=0.3    # Voxel size for sparse voxelization of ground points before DTM construction (meters)
    ):
        self.csf_threshold = csf_threshold
        self.csf_resolution = csf_resolution
        self.csf_rigidness = csf_rigidness
        self.csf_correct_steep_slope = csf_correct_steep_slope
        self.csf_iterations = csf_iterations
        
        self.dtm_resolution = dtm_resolution
        self.dtm_k = dtm_k
        self.dtm_p = dtm_p
        self.dtm_voxel_size = dtm_voxel_size
        
        # Store computed DTM dictionary data
        self.dtm_data = None

    def process(self, points):
        """
        Core processing function:
        Input: points: shape (N, 3) original point cloud 3D coordinates
        Returns: 
            ground_mask: shape (N,) boolean mask identifying ground points
            dtm: shape (H, W) generated 2D digital terrain grid
            dtm_offset: shape (2,) XY coordinate offset of the lower-left corner of the DTM grid
        """
        # Execute CSF blind filtering (without any external labels)
        print("[Ground Filtering] Initializing CSF blind classification...")
        csf = CSF.CSF()
        csf.params.class_threshold = self.csf_threshold
        csf.params.bSloopSmooth = self.csf_correct_steep_slope
        csf.params.interations = self.csf_iterations
        csf.params.cloth_resolution = self.csf_resolution
        csf.params.rigidness = self.csf_rigidness
        csf.params.time_step = 1.0

        # Set coordinate data
        csf.setPointCloud(points.astype(np.float64))
        ground_indices_vec = CSF.VecInt()
        non_ground_indices_vec = CSF.VecInt()
        
        with suppress_stdout():
            csf.do_filtering(ground_indices_vec, non_ground_indices_vec, exportCloth=False)
        
        # Convert to numpy format
        ground_indices = np.array(ground_indices_vec, dtype=np.int64)
        ground_mask = np.zeros(len(points), dtype=bool)
        if len(ground_indices) > 0:
            ground_mask[ground_indices] = True
            
        print(f"[Ground Filtering] Separation completed. Ground points: {np.sum(ground_mask):,}, Vegetation points: {len(points) - np.sum(ground_mask):,}")

        #  Extract ground point coordinates for interpolation
        terrain_xyz = points[ground_mask]

        # Appropriately downsample ground points using voxelization 
        # (to prevent excessive point density from slowing down IDW interpolation speed)
        # Note: Downsampling a 10cm spacing point cloud to 30cm is physically feasible, 
        # which can remove redundant ground points and smooth micro-terrain
        if self.dtm_voxel_size is not None and self.dtm_voxel_size > 0.1:
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(terrain_xyz.astype(np.float64))
            temp_pcd = temp_pcd.voxel_down_sample(self.dtm_voxel_size)
            terrain_xyz = np.asarray(temp_pcd.points)

        # Generate rasterized continuous DTM grid (using cKDTree + IDW acceleration)
        print("[DTM Interpolation] Building global undulating elevation map...")
        dtm, dtm_offset = self._create_dtm(terrain_xyz)
        
        return ground_mask, dtm, dtm_offset

    def _create_dtm(self, terrain_xyz):
        """
        Calculate absolute terrain height at each grid center using inverse distance weighting (IDW)
        """
        min_coords = terrain_xyz[:, :2].min(axis=0)
        max_coords = terrain_xyz[:, :2].max(axis=0)

        # Calculate grid size and starting offset
        dtm_grid_offset = np.floor(min_coords / self.dtm_resolution)
        dtm_size = (np.floor(max_coords / self.dtm_resolution) - dtm_grid_offset + 1).astype(np.int64)

        # Build regular grid points
        grid_x = np.arange(dtm_size[0])
        grid_y = np.arange(dtm_size[1])
        x_mesh, y_mesh = np.meshgrid(grid_x, grid_y)

        # Calculate XY physical coordinates of each grid center in world coordinate system
        dtm_points_x = (dtm_grid_offset[0] + x_mesh.flatten()) * self.dtm_resolution
        dtm_points_y = (dtm_grid_offset[1] + y_mesh.flatten()) * self.dtm_resolution
        dtm_points_xy = np.column_stack((dtm_points_x, dtm_points_y))

        # Build spatial search tree and query neighbors
        kd_tree = cKDTree(terrain_xyz[:, :2])
        # Use workers=-1 to fully utilize all CPU physical threads on Windows 
        # for rapid query completion
        distances, indices = kd_tree.query(
            dtm_points_xy, 
            k=min(self.dtm_k, len(terrain_xyz)), 
            workers=-1
        )

        # Calculate IDW weights, note to avoid division by zero for distance of 0
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = 1.0 / (distances ** self.dtm_p)

        exact_matches = (distances == 0)
        any_exact_match = exact_matches.any(axis=1)

        weights[any_exact_match] = 0.0
        weights[exact_matches] = 1.0

        sum_weights = weights.sum(axis=1, keepdims=True)
        sum_weights[sum_weights == 0] = 1.0    # Avoid division by zero for isolated points
        weights /= sum_weights

        # Fast matrix multiplication to complete elevation interpolation
        dtm_heights = np.sum(terrain_xyz[indices, 2] * weights, axis=1)

        # Organize grid structure
        dtm = dtm_heights.reshape((dtm_size[1], dtm_size[0]))
        dtm_offset = dtm_grid_offset * self.dtm_resolution

        self.dtm_data = {
            "dtm": dtm,
            "dtm_offset": dtm_offset,
            "x_mesh": (dtm_grid_offset[0] + x_mesh) * self.dtm_resolution,
            "y_mesh": (dtm_grid_offset[1] + y_mesh) * self.dtm_resolution
        }

        return dtm, dtm_offset

    def get_normalized_heights(self, points):
        """
        Use bilinear interpolation to calculate local relative elevation (net height) of any point in the point cloud.
        """
        if self.dtm_data is None:
            raise ValueError("Please run process() first to fit terrain and generate DTM grid.")

        dtm = self.dtm_data["dtm"]
        dtm_offset = self.dtm_data["dtm_offset"]
        dtm_height, dtm_width = dtm.shape

        # Calculate continuous floating-point positions of points on the DTM grid
        grid_positions = (points[:, :2] - dtm_offset) / self.dtm_resolution

        # Boundary constraints to avoid exceeding elevation map indices
        grid_positions_x = np.clip(grid_positions[:, 0], 0, dtm_width - 1.0001)
        grid_positions_y = np.clip(grid_positions[:, 1], 0, dtm_height - 1.0001)

        # Extract bottom-left grid indices (x0, y0)
        x0 = np.floor(grid_positions_x).astype(np.int64)
        y0 = np.floor(grid_positions_y).astype(np.int64)

        # Extract top-right grid indices (x1, y1)
        x1 = np.clip(x0 + 1, 0, dtm_width - 1)
        y1 = np.clip(y0 + 1, 0, dtm_height - 1)

        # Calculate interpolation weight ratios
        fx = grid_positions_x - x0
        fy = grid_positions_y - y0

        # Get absolute ground elevation of the four nearest grid points under the feet
        h00 = dtm[y0, x0]
        h10 = dtm[y0, x1]
        h01 = dtm[y1, x0]
        h11 = dtm[y1, x1]

        # Perform bilinear fusion to smooth mountain slope folds
        interp_x0 = h00 * (1.0 - fx) + h10 * fx
        interp_x1 = h01 * (1.0 - fx) + h11 * fx
        terrain_z = interp_x0 * (1.0 - fy) + interp_x1 * fy

        # Absolute Z coordinate minus the interpolated local terrain height Z coordinate
        return points[:, 2] - terrain_z


def create_dtm_mesh(dtm_data):
    """
    Convert generated DTM mesh to Open3D triangle mesh for subsequent high-definition 3D visualization.
    """
    x = dtm_data["x_mesh"]
    y = dtm_data["y_mesh"]
    z = dtm_data["dtm"]
    rows, cols = x.shape

    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    
    triangles = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v00 = r * cols + c
            v01 = r * cols + (c + 1)
            v10 = (r + 1) * cols + c
            v11 = (r + 1) * cols + (c + 1)
            triangles.append([v00, v10, v01])
            triangles.append([v01, v10, v11])
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.compute_vertex_normals()
    
    # Use classic terrain gradient coloring (matplotlib.colormaps.get_cmap("terrain"))
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("terrain")
    z_vals = vertices[:, 2]
    z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-6)
    colors = cmap(z_norm)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    return mesh

# ==========================================
# Independent unit test logic
# ==========================================
if __name__ == "__main__":

    # Configure file paths
    test_file = r"E:\test_reg_add_seg_2\datasets\evo_example_dataset\tiles\tile_1.ply"
    evo_offset = np.array([-399000.0, -6786000.0, 0.0])
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}, please check if the path is correct.")
        
    print(f"Loading test file: {os.path.basename(test_file)}")
    cloud = o3d.t.io.read_point_cloud(test_file)
    points = cloud.point.positions.numpy()
    
    # Apply large coordinate translation to ensure calculation accuracy
    shifted_points = points + evo_offset
    
    # Instantiate and run ground filtering and DTM construction
    # Slightly tighten csf_threshold to 0.3m for tighter local micro-terrain fitting
    seg = GroundSegmentation(
        csf_threshold=0.3,
        csf_resolution=0.5,
        csf_rigidness=2,
        dtm_resolution=0.5,
        dtm_k=20,
        dtm_p=1.0,
        dtm_voxel_size=0.3
    )
    
    ground_mask, dtm, dtm_offset = seg.process(shifted_points)
    
    # Verify elevation normalization function (net height calculation)
    print("Calculating local normalized height (net height) for each point...")
    rel_heights = seg.get_normalized_heights(shifted_points)
    print(f"[Normalization Results] Local net height range: {rel_heights.min():.2f}m ~ {rel_heights.max():.2f}m")
    
    # Generate 3D terrain triangle mesh
    print("Creating 3D digital terrain model grid...")
    dtm_mesh = create_dtm_mesh(seg.dtm_data)
    
    # Prepare Open3D visualization geometries
    legacy_cloud = cloud.to_legacy()
    legacy_cloud.translate(evo_offset)    # Maintain coordinate translation consistency
    
    # Extract ground point cloud (colored: black, to view the fine details of mesh fitting)
    ground_vis = legacy_cloud.select_by_index(np.where(ground_mask)[0])
    ground_vis.paint_uniform_color([0, 0, 0])
    
    # Extract non-ground point cloud (colored: light gray background)
    rest_vis = legacy_cloud.select_by_index(np.where(~ground_mask)[0])
    rest_vis.paint_uniform_color([0.8, 0.8, 0.8])
    
    # 6. Open 3D rendering window
    print("Opening 3D visualization window...")
    print("[Hint] Colored undulating surface is the generated global DTM, black points are extracted ground points, gray points are vegetation above the surface.")
    o3d.visualization.draw_geometries(
        [dtm_mesh, ground_vis, rest_vis],
        window_name="3D DTM & Ground Filtering Verification (CSF + IDW)",
        mesh_show_back_face=True
    )