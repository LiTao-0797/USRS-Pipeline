# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import open3d as o3d
import logging

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(UTILS_DIR)
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from utils.height_image import HeightImage, display_correspondences, draw_correspondences
from utils.graph import Graph, CorrespondenceGraph
from utils.tree_trunk_segmentation_plus import TreeTrunkSegmentationPlus


class NullLogger:
    """
    High robustness null logger:
   Implements standard Logger methods and ExperimentLogger specific methods,
   preventing AttributeError from other underlying modules (such as height_image / graph) during testing.
    """
    def __init__(self):
        pass
    def debug(self, msg, *args, **kwargs): pass
    def info(self, msg, *args, **kwargs): pass
    def warning(self, msg, *args, **kwargs): pass
    def error(self, msg, *args, **kwargs): pass
    def critical(self, msg, *args, **kwargs): pass
    def log_image(self, img, name, *args, **kwargs): pass
    def log_pointcloud(self, cloud, cloud_io, name, *args, **kwargs): pass


class HorizontalRegistration:
    def __init__(
        self,
        uav_cloud,
        uav_ground_plane,    # UAV GroundSegmentation engine instance passed in
        mls_cloud,
        mls_cloud_ground_plane,    # MLS GroundSegmentation engine instance passed in
        min_distance_between_peaks: float,
        max_number_of_clique: int,
        logger,
        distance_threshold: float,
        angle_threshold: float,
        debug: bool = False,
        correspondence_matching_method="graph",
        mls_feature_extraction_method="canopy_map",
        trunk_params=None,
    ):
        self.uav_cloud = uav_cloud
        self.uav_ground_plane = uav_ground_plane
        self.mls_cloud = mls_cloud
        self.mls_cloud_ground_plane = mls_cloud_ground_plane
        self.debug = debug
        
        # If no ExperimentLogger is passed externally, instantiate our own NullLogger
        # Ensure not only standard debug interface but also log_image interface to completely block AttributeError
        if logger is None:
            self.logger = NullLogger()
        else:
            self.logger = logger
            
        self.transforms = []
        self.max_number_of_clique = max_number_of_clique
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.feature_association_method = correspondence_matching_method  # graph
        self.mls_feature_extraction_method = mls_feature_extraction_method  # canopy_map or tree_segmentation
        self.min_distance_between_peaks = min_distance_between_peaks
        self.trunk_params = trunk_params if trunk_params is not None else {}
        self.extracted_trunk_pos = None
        self.extracted_trunk_scores = None
        self.extracted_trunk_diameters= None

    def find_transform(self, src, dst, estimate_scale=False):
        """
        Umeyama algorithm estimates N-dimensional similarity transformation matrix
        """
        src = np.asarray(src)
        dst = np.asarray(dst)

        num = src.shape[0]
        dim = src.shape[1]

        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        src_demean = src - src_mean
        dst_demean = dst - dst_mean

        A = dst_demean.T @ src_demean / num

        d = np.ones((dim,), dtype=np.float64)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1

        T = np.eye(dim + 1, dtype=np.float64)

        U, S, V = np.linalg.svd(A)

        rank = np.linalg.matrix_rank(A)
        if rank == 0:
            return np.nan * T
        elif rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(V) > 0:
                T[:dim, :dim] = U @ V
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                T[:dim, :dim] = U @ np.diag(d) @ V
                d[dim - 1] = s
        else:
            T[:dim, :dim] = U @ np.diag(d) @ V

        if estimate_scale:
            scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
        else:
            scale = 1.0

        T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
        T[:dim, :dim] *= scale

        return T

    def process(self) -> bool:
        uav_proc = HeightImage(
            min_distance_between_peaks=self.min_distance_between_peaks,
            logger=self.logger,
            debug=self.debug,
        )
        mls_proc = HeightImage(
            min_distance_between_peaks=self.min_distance_between_peaks,
            logger=self.logger,
            debug=self.debug,
        )

        uav_canopy = uav_proc.compute_canopy_image(
            self.uav_cloud, self.uav_ground_plane
        )
        # High-resolution canopy map generation still based on original full MLS tiles, 
        # ensuring feature boundary resolution
        mls_canopy = mls_proc.compute_canopy_image(
            self.mls_cloud, self.mls_cloud_ground_plane
        )

        # Extract high-altitude canopy feature points and corresponding plane images
        mls_height_pts, mls_height_img = mls_proc.find_local_maxima(mls_canopy)
        uav_height_pts, uav_height_img = uav_proc.find_local_maxima(uav_canopy)

        # Use Plus version's slice circle fitting algorithm to extract trunk center features
        if self.mls_feature_extraction_method == "tree_segmentation":
            # Memory local downsampling: Only clone a temporary lightweight point cloud for DBSCAN slice calculations, 
            # keeping the original self.mls_cloud 100% complete and with unchanged point count
            voxel_size_for_fit = 0.10    # 10cm voxel spacing
            print(f"[Downsampling] Temporarily downsampling trunk slice point cloud to {voxel_size_for_fit}m in memory to protect DBSCAN clustering efficiency...")
            mls_cloud_down = self.mls_cloud.voxel_down_sample(voxel_size=voxel_size_for_fit)
            
            # Instantiate Plus version extractor and map adaptive construction parameters
            tree_trunk_segmentation = TreeTrunkSegmentationPlus(
                eps_2d=self.trunk_params.get("eps_2d", 0.20),
                min_samples_2d=self.trunk_params.get("min_samples_2d", 20),
                eps_3d=self.trunk_params.get("eps_3d", 0.25),
                min_samples_3d=self.trunk_params.get("min_samples_3d", 5),
                min_cluster_height=self.trunk_params.get("min_cluster_height", 1.5),
                layer_height=self.trunk_params.get("layer_height", 0.20),
                layer_overlap=self.trunk_params.get("layer_overlap", 0.10),
                max_std_diameter=self.trunk_params.get("max_std_diameter", 0.10),
                max_std_position=self.trunk_params.get("max_std_position", None),
                std_num_layers=self.trunk_params.get("std_num_layers", 6),
                
                # Bind new adaptive parameter passing
                h_min=self.trunk_params.get("h_min", 1.0),
                h_max=self.trunk_params.get("h_max", 4.0),
                verticality_thr=self.trunk_params.get("verticality_thr", 0.65),
                fit_min_z=self.trunk_params.get("fit_min_z", 1.0),
                fit_max_z=self.trunk_params.get("fit_max_z", 3.0),
                
                seed_diameter_factor=self.trunk_params.get("seed_diameter_factor", 1.05),
                seed_height_range=self.trunk_params.get("seed_height_range", (1.0, 1.6))
            )
            
            # Extract required feature parameters from temporary downsampled point cloud
            mls_pts_all = mls_cloud_down.point.positions.numpy().astype(np.float64)
            # Rely on DTM engine object to get precise relative elevation (net height)
            mls_rel_heights = self.mls_cloud_ground_plane.get_normalized_heights(mls_pts_all)
            
            # Extract pre-calculated point-level verticality features
            if "verticality" in mls_cloud_down.point:
                mls_verticality = mls_cloud_down.point["verticality"].numpy().flatten()
            else:
                mls_verticality = None
                if self.logger:
                    self.logger.warning("No 'verticality' channel found in downsampled point cloud attributes.")
            
            # Execute Plus version multi-layer algebraic circle fitting extraction algorithm
            trunk_centers, trunk_diameters, cylinder_seed_mask, trunk_points_mask = \
                tree_trunk_segmentation.find_tree_trunks(mls_pts_all, mls_rel_heights, verticality=mls_verticality)
            
            # Directly use geometric unbiased absolute regression circle centers as alignment feature points, score default assigned as 1.0
            self.extracted_trunk_pos = trunk_centers
            self.extracted_trunk_scores = np.ones(len(trunk_centers))
            self.extracted_trunk_diameters = trunk_diameters
            
            # Use original high-density MLS bounding box to define pixel grid canvas boundaries, 
            # project 3D fitted trunk circle centers to pixel coordinates
            bounding_box = self.mls_cloud.get_axis_aligned_bounding_box()
            mls_height_pts = np.zeros((trunk_centers.shape[0], 2), dtype=np.int32)
            for i in range(trunk_centers.shape[0]):
                mls_height_pts[i] = mls_proc.cloud_point_to_pixel(
                    trunk_centers[i], bounding_box, image_resolution=0.1
                )

        elif self.mls_feature_extraction_method != "canopy_map":
            raise ValueError("Unknown method: " + self.mls_feature_extraction_method)

        # Feature map maximum clique relationship search and registration
        if self.feature_association_method == "graph":
            if self.logger:
                self.logger.debug("Creating the feature graphs")
            G = Graph(mls_height_pts, node_prefix="f")
            H = Graph(uav_height_pts, node_prefix="uav")

            if self.logger:
                self.logger.debug(
                    f"Number of nodes and edges of the mls graph {G.graph.number_of_nodes()} {G.graph.number_of_edges()}"
                )
                self.logger.debug(
                    f"Number of nodes and edges of the uav graph {H.graph.number_of_nodes()} {H.graph.number_of_edges()}"
                )

            correspondence_graph = CorrespondenceGraph(
                G, H, self.logger, self.distance_threshold, self.angle_threshold
            )

            if self.logger:
                self.logger.debug("Computing the maximum clique")
            if correspondence_graph.graph.number_of_edges() > 1800000:
                if self.logger:
                    self.logger.debug("Too many edges in the correspondence graph")
                return False
            
            correspondences_list = correspondence_graph.maximum_clique()

            if len(correspondences_list) > self.max_number_of_clique:
                if self.logger:
                    self.logger.debug("Too many cliques, downsampling them")
                correspondences_list = correspondences_list[
                    0 : self.max_number_of_clique
                ]
            elif len(correspondences_list) == 0:
                return False
        else:
            raise ValueError("Unknown method: " + self.feature_association_method)

        for i in range(len(correspondences_list)):
            correspondences = correspondences_list[i]
            if self.debug:
                display_correspondences(
                    mls_height_img,
                    mls_height_pts,
                    uav_height_img,
                    uav_height_pts,
                    correspondences,
                    False,
                    G,
                    H,
                )
            correspondences_img = draw_correspondences(
                mls_height_img,
                mls_height_pts,
                uav_height_img,
                uav_height_pts,
                correspondences,
                False,
                G,
                H,
            )
            
            if self.logger:
                self.logger.log_image(correspondences_img, "correspondences")

            mls_pts = np.zeros((len(correspondences), 2))
            uav_pts = np.zeros((len(correspondences), 2))
            for j in range(len(correspondences)):
                mls_pts[j] = mls_proc.pixel_to_cloud(
                    correspondences[j][0][0], correspondences[j][0][1]
                )
                uav_pts[j] = uav_proc.pixel_to_cloud(
                    correspondences[j][1][0], correspondences[j][1][1]
                )

            if mls_pts.shape[0] < 3:
                return False

            M = self.find_transform(mls_pts, uav_pts)
            self.transforms.append(M)

        return True


# ==========================================
# Test execution area (independent horizontal registration test)
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("Starting horizontal_alignment.py independent unit test...")
    print("=" * 60)

    from utils.ground_segmentation import GroundSegmentation
    from utils.logger import ExperimentLogger    # Import real ExperimentLogger
    
    # Define test input file paths (must point to processed folder with pre-computed verticality preprocessing data)
    processed_root = os.path.join(ROOT_DIR, "evo_example_dataset_processed")
    uav_path = os.path.join(processed_root, "uav_cloud.ply")
    tile_path = os.path.join(processed_root, "tiles", "tile_1.ply")
    
    # Dedicated large coordinate offset
    evo_offset = np.array([-399000.0, -6786000.0, 0.0])
    
    if not os.path.exists(uav_path) or not os.path.exists(tile_path):
        print("[Error] Cannot find processed PLY test files, please ensure step1_pre_data.py has been run!")
        print(f"Missing path one: {uav_path}")
        print(f"Missing path two: {tile_path}")
        sys.exit(1)

    # Load point cloud data that has already been processed for verticality and offset alignment (full resolution)
    print(f"Loading UAV original reference cloud: {os.path.basename(uav_path)}")
    uav_pcd = o3d.t.io.read_point_cloud(uav_path)
    uav_pcd.translate(o3d.core.Tensor(evo_offset, o3d.core.float32))

    print(f"Loading MLS original high-density tile (original full point count protection validation): {os.path.basename(tile_path)}")
    mls_pcd = o3d.t.io.read_point_cloud(tile_path)
    mls_pcd.translate(o3d.core.Tensor(evo_offset, o3d.core.float32))
    
    # Print loading status
    print(f"UAV original point count: {len(uav_pcd.point.positions):,}")
    print(f"MLS original high-density tile point count: {len(mls_pcd.point.positions):,}")

    # Instantiate real experiment manager, 
    # allowing test process to save high-definition registration feature collision images to disk
    log_dir = os.path.join(ROOT_DIR, "logs")
    print(f"Initializing real logging system (output path: {log_dir}) ...")
    logger = ExperimentLogger(base_dir=log_dir)
    logger.set_leaf_logging_folder("test_tile_1")

    # Ground modeling, simulating vertical alignment output
    print("Generating DTM undulating elevation model as feature extraction reference...")
    uav_engine = GroundSegmentation(csf_resolution=1.0, dtm_resolution=0.5, dtm_voxel_size=0.3)
    mls_engine = GroundSegmentation(csf_resolution=1.0, dtm_resolution=0.5, dtm_voxel_size=0.3)
    
    uav_points = uav_pcd.point.positions.numpy().astype(np.float64)
    mls_points = mls_pcd.point.positions.numpy().astype(np.float64)
    
    uav_engine.process(uav_points)
    mls_engine.process(mls_points)

    # Configure trunk fitting parameters 
    # (DBSCAN settings and 1.3m seed sampling range, fully adapted to new interface)
    test_trunk_settings = {
        "eps_2d": 0.20,
        "min_samples_2d": 20,
        "eps_3d": 0.25,
        "min_samples_3d": 5,
        "min_cluster_height": 1.5,
        "layer_height": 0.20,
        "layer_overlap": 0.10,
        "max_std_diameter": 0.10,
        "max_std_position": None,
        "std_num_layers": 6,
        
        "h_min": 1.0,
        "h_max": 4.0,
        "verticality_thr": 0.65,
        "fit_min_z": 1.0,
        "fit_max_z": 3.0,
        
        "seed_diameter_factor": 1.05,
        "seed_height_range": (1.0, 1.6)
    }

    # Instantiate horizontal registration class and pass real logger
    reg = HorizontalRegistration(
        uav_cloud=uav_pcd,
        uav_ground_plane=uav_engine,
        mls_cloud=mls_pcd,
        mls_cloud_ground_plane=mls_engine,
        min_distance_between_peaks=2.5,
        max_number_of_clique=5,
        logger=logger,    # Pass real ExperimentLogger, making registration images save to disk
        distance_threshold=0.2,    # Feature edge relative length tolerance
        angle_threshold=1.6,    # Feature edge relative angle tolerance
        debug=True,    # Enable Debug to pop up matching diagram connection windows
        correspondence_matching_method="graph",
        mls_feature_extraction_method="tree_segmentation",
        trunk_params=test_trunk_settings
    )

    try:
        success = reg.process()
        print("\n" + "=" * 60)
        if success:
            print("Horizontal alignment feature collision and similarity transformation solving successful!")
            print(f"Extracted MLS trunk absolute circle center feature points count: {len(reg.extracted_trunk_pos)} trees")
            print(f"MLS original high-density tile point count (verification original tile not damaged or over-cut): {len(mls_pcd.point.positions):,} (original without any reduction)")
            print(f"Registration feature connection diagram successfully saved to: {logger.current_logging_directory()}/test_tile_1/correspondences.png")
            print(f"Number of candidate 2D transformation matrices (each corresponds to one maximum clique): {len(reg.transforms)}")
            if len(reg.transforms) > 0:
                print("First candidate 3D transformation matrix (2D similarity extension) details:")
                print(reg.transforms[0])
        else:
            print("Alignment failed: No consistent maximum clique detected in correspondence graph.")
        print("=" * 60)
    except Exception as e:
        print(f"\n[Test failed] Horizontal registration encountered exception: {e}")