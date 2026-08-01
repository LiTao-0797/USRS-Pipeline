# -*- coding: utf-8 -*-
"""
Robust Tree Trunk Segmentation and DBH Estimation Module
Using Kasa Algebraic Circle Fitting and Multi-layer Linear Regression.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import itertools


def kasa_circle_fit(pts):
    """
    Kasa algebraic circle fitting algorithm:
    Input: pts: shape (N, 2) horizontal 2D coordinate points
    Returns: center a, b and radius R. Returns -1.0, -1.0, -1.0 if fitting fails
    """
    if len(pts) < 3:  # Follow TreeX official parameter: relax to 3 points, allow sparse small trees with 10cm spacing to 
        return -1.0, -1.0, -1.0
        
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Construct Kasa overdetermined equation A * X = B
    A = np.column_stack((x, y, np.ones_like(x)))
    B = x**2 + y**2
    
    try:
        # Solve equation using least squares
        u, v, w = np.linalg.lstsq(A, B, rcond=None)[0]
        a = u / 2.0
        b = v / 2.0
        R_sq = a**2 + b**2 + w
        if R_sq <= 0:
            return -1.0, -1.0, -1.0
        return a, b, np.sqrt(R_sq)
    except Exception:
        return -1.0, -1.0, -1.0


class TreeTrunkSegmentationPlus:
    """
    Plus version tree trunk segmentation and feature extractor:
    Specifically handles 10cm voxelized point cloud, filters "pseudo trunk" based on hierarchical geometric continuity, and extrapolates DBH at 1.3m height.
    """
    def __init__(
        self,
        # Clustering related parameters
        eps_2d=0.20,    # 2D clustering neighborhood radius (meters)
        min_samples_2d=20,    # Minimum points for 2D clustering (remove sparse pseudo trunks)
        eps_3d=0.25,    # 3D clustering neighborhood radius (meters)
        min_samples_3d=5,    # Minimum points for 3D clustering
        
        # Tree trunk inspection parameters
        min_cluster_height=1.5,    # Minimum vertical extension height of trunk cluster (meters)
        
        # Multi-layer fitting parameters
        layer_height=0.20,    # 20cm layer height
        layer_overlap=0.10,    # 10cm overlap (sliding step size is 10cm)
        
        # inspection standard deviation parameters (adaptive tolerance core)
        max_std_diameter=0.10,    # Allow 10cm
        max_std_position=None,    # Default set to None, directly disable center standard deviation filtering, rely on univariate linear regression to solve slanted tree problem
        std_num_layers=6,    # Core parameter: arbitrarily select 6 layers for sub-set inspection, pass if all pass
        
        # adaptive parameters
        h_min=1.0,    # Minimum trunk slice height (meters)
        h_max=4.0,    # Maximum trunk slice height (meters)
        verticality_thr=0.65,    # Verticality filtering threshold for straight trunks
        fit_min_z=1.0,    # Minimum fitting height for multi-layer circle fitting (meters)
        fit_max_z=3.0,    # Maximum fitting height for multi-layer circle fitting (meters)
        
        # Cylinder seed source extraction parameters
        seed_diameter_factor=1.05,    # Seed cylinder diameter amplification factor for DBH
        seed_height_range=(1.0, 1.6)    # Seed sampling interval around 1.3m height (meters)
    ):
        self.eps_2d = eps_2d
        self.min_samples_2d = min_samples_2d
        self.eps_3d = eps_3d
        self.min_samples_3d = min_samples_3d
        self.min_cluster_height = min_cluster_height
        
        self.layer_height = layer_height
        self.layer_overlap = layer_overlap
        
        self.max_std_diameter = max_std_diameter
        self.max_std_position = max_std_position
        self.std_num_layers = std_num_layers
        
        self.h_min = h_min
        self.h_max = h_max
        self.verticality_thr = verticality_thr
        self.fit_min_z = fit_min_z
        self.fit_max_z = fit_max_z
        
        self.seed_diameter_factor = seed_diameter_factor
        self.seed_height_range = seed_height_range

    def find_tree_trunks(self, points, rel_heights, verticality=None):
        """
        Core tree trunk extraction function:
        Input:
            points: shape (N, 3) original absolute 3D coordinates
            rel_heights: shape (N,) relative ground height
            verticality: shape (N,) global point-level verticality features.
                         Used to filter non-straight lateral branches, leaf fall and closely attached shrubs before clustering.
        Returns:
            trunk_centers: shape (M, 3) absolute 3D center coordinates at 1.3m height
            trunk_diameters: shape (M,) estimated single-tree DBH
            cylinder_seed_mask: shape (N,) boolean mask identifying cylinder seed sources at 1.3m
            trunk_points_mask: shape (N,) boolean mask identifying points belonging to validated 1~4m trunk slice point cloud
        """
        n_points = len(points)
        
        # First stage: DBSCAN continuity preliminary screening (slice according to configured height range)
        print(f"[Clustering stage] Extracting trunk slices {self.h_min:.2f}m ~ {self.h_max:.2f}m...")
        slice_mask = (rel_heights >= self.h_min) & (rel_heights <= self.h_max)
        
        # Perform verticality cleaning before clustering, strongly block interference from lateral branches and closely attached shrubs
        if verticality is not None:
            # Only keep vertically high points, filter out grasses, shrubs and lateral branches
            slice_mask = slice_mask & (verticality >= self.verticality_thr)
            
        slice_pts = points[slice_mask]
        
        if len(slice_pts) == 0:
            return np.empty((0, 3)), np.empty((0,)), np.zeros(n_points, dtype=bool), np.zeros(n_points, dtype=bool)

        # 2D horizontal DBSCAN coarse clustering
        dbscan_2d = DBSCAN(eps=self.eps_2d, min_samples=self.min_samples_2d, n_jobs=-1)
        labels_2d = dbscan_2d.fit_predict(slice_pts[:, :2])
        
        unique_labels_2d = np.unique(labels_2d)
        unique_labels_2d = unique_labels_2d[unique_labels_2d != -1]
        print(f"[Clustering stage] 2D clustering preliminary screening complete, {len(unique_labels_2d)} high straight trunk candidates found after filtering.")

        # 3D vertical depth splitting and extension inspection
        validated_cluster_pts = []
        validated_cluster_indices = []
        
        slice_indices = np.where(slice_mask)[0]
        dbscan_3d = DBSCAN(eps=self.eps_3d, min_samples=self.min_samples_3d, n_jobs=-1)
        
        for label in unique_labels_2d:
            cluster_idx = (labels_2d == label)
            c_pts = slice_pts[cluster_idx]
            c_indices = slice_indices[cluster_idx]
            
            # Perform 3D subdivision
            labels_3d = dbscan_3d.fit_predict(c_pts)
            for l_3d in np.unique(labels_3d):
                if l_3d == -1: continue
                sub_pts = c_pts[labels_3d == l_3d]
                sub_indices = c_indices[labels_3d == l_3d]
                
                # Inspection of vertical extension height (Z direction range)
                z_extent = sub_pts[:, 2].max() - sub_pts[:, 2].min()
                if z_extent >= self.min_cluster_height:
                    validated_cluster_pts.append(sub_pts)
                    validated_cluster_indices.append(sub_indices)

        print(f"[Clustering stage] 3D subdivision and height inspection complete, {len(validated_cluster_pts)} trunk candidates passed inspection.")

        # Second stage: Circle fitting and 1.3m regression estimation (avoid bifurcation by configuring upper and lower limits)
        # Dynamically build layer height boundaries based on passed layer_height and overlap
        step = self.layer_height - self.layer_overlap
        
        layer_bounds = []
        curr_low = self.fit_min_z
        while curr_low + self.layer_height <= self.fit_max_z + 1e-5:
            layer_bounds.append((curr_low, curr_low + self.layer_height))
            curr_low += step
            
        layer_heights = np.array([np.mean(b) for b in layer_bounds])
        num_layers = len(layer_bounds)
        
        # Adaptively set qualified layer threshold: need at least 45% of slice layers to successfully fit and pass standard deviation check
        min_valid_layers = max(3, int(num_layers * 0.45))

        final_trees = []
        trunk_points_mask = np.zeros(n_points, dtype=bool)

        for tree_idx, c_pts in enumerate(validated_cluster_pts):
            c_rel_heights = rel_heights[validated_cluster_indices[tree_idx]]
            
            fitted_centers = []
            fitted_radii = []
            fitted_layer_z = []
            
            # Fit each layer of the candidate tree
            for layer_idx, (low, high) in enumerate(layer_bounds):
                in_layer = (c_rel_heights >= low) & (c_rel_heights <= high)
                layer_pts = c_pts[in_layer]
                
                xc, yc, R = kasa_circle_fit(layer_pts[:, :2])
                if xc != -1.0:
                    fitted_centers.append([xc, yc])
                    fitted_radii.append(R)
                    fitted_layer_z.append(layer_heights[layer_idx])
            
            # Geometric inspection (multi-layer subset combination consistency inspection)
            n_fitted = len(fitted_layer_z)
            if n_fitted < self.std_num_layers:
                continue
                
            fitted_centers = np.array(fitted_centers)
            fitted_radii = np.array(fitted_radii)
            fitted_layer_z = np.array(fitted_layer_z)
            
            # Use numpy vectorized combination mechanism 
            # to instantly calculate standard deviation for all combinations
            valid_indices = np.arange(n_fitted)
            comb_array = np.array(list(itertools.combinations(valid_indices, self.std_num_layers)))
            
            # Calculate standard deviation of all combination diameters
            comb_radii = fitted_radii[comb_array]
            comb_diameters = comb_radii * 2.0
            std_dias = np.std(comb_diameters, axis=1)
            
            # Build filtering mask
            valid_comb_mask = std_dias <= self.max_std_diameter
            
            # If center standard deviation filtering is enabled, perform vectorized judgment
            if self.max_std_position is not None:
                comb_centers = fitted_centers[comb_array]
                std_poss = np.max(np.std(comb_centers, axis=1), axis=1)
                valid_comb_mask &= std_poss <= self.max_std_position
                
            if not np.any(valid_comb_mask):
                continue
                
            # From all qualified subsets, pick the best combination with minimum diameter variation
            valid_indices_in_mask = np.where(valid_comb_mask)[0]
            best_idx_in_mask = np.argmin(std_dias[valid_comb_mask])
            best_comb_idx = valid_indices_in_mask[best_idx_in_mask]
            best_combination = comb_array[best_comb_idx]

            # Pass inspection: use the best layer subset for univariate linear regression extrapolation to 1.3m absolute center and DBH
            try:
                poly_x = np.polyfit(fitted_layer_z[best_combination], fitted_centers[best_combination, 0], 1)
                poly_y = np.polyfit(fitted_layer_z[best_combination], fitted_centers[best_combination, 1], 1)
                poly_r = np.polyfit(fitted_layer_z[best_combination], fitted_radii[best_combination], 1)
                
                # Predict absolute coordinates and diameter at Z = 1.3m
                pred_x = np.polyval(poly_x, 1.3)
                pred_y = np.polyval(poly_y, 1.3)
                pred_dbh = np.polyval(poly_r, 1.3) * 2.0
                
                # Allow minimum 2cm extremely thin young trees to pass
                if 0.02 <= pred_dbh <= 2.0:
                    ground_z = np.mean(c_pts[:, 2] - c_rel_heights)
                    center_1_3m = np.array([pred_x, pred_y, ground_z + 1.3])
                    
                    final_trees.append({
                        "center": center_1_3m,
                        "dbh": pred_dbh
                    })
                    # Mark valid tree
                    trunk_points_mask[validated_cluster_indices[tree_idx]] = True
                    
            except Exception:
                continue

        # Organize output
        n_trees = len(final_trees)
        print(f"[Inspection and fitting] Successfully located robust trees: {n_trees} trees.")
        
        if n_trees == 0:
            return np.empty((0, 3)), np.empty((0,)), np.zeros(n_points, dtype=bool), np.zeros(n_points, dtype=bool)

        trunk_centers = np.array([t["center"] for t in final_trees])
        trunk_diameters = np.array([t["dbh"] for t in final_trees])

        # Third stage: Random box extraction of 1.3m cylinder seed source point cloud
        cylinder_seed_mask = np.zeros(n_points, dtype=bool)
        
        # Only perform fast point cloud distance retrieval in seed height range (based on passed seed_height_range)
        z_seed_mask = (rel_heights >= self.seed_height_range[0]) & (rel_heights <= self.seed_height_range[1])
        z_seed_idx = np.where(z_seed_mask)[0]
        z_seed_pts = points[z_seed_mask]
        
        if len(z_seed_pts) > 0:
            tree_2d = cKDTree(z_seed_pts[:, :2])
            for i in range(n_trees):
                center = trunk_centers[i]
                radius = (trunk_diameters[i] / 2.0) * self.seed_diameter_factor
                
                near_idx = tree_2d.query_ball_point(center[:2], r=radius)
                if len(near_idx) > 0:
                    cylinder_seed_mask[z_seed_idx[near_idx]] = True

        return trunk_centers, trunk_diameters, cylinder_seed_mask, trunk_points_mask