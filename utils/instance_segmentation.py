# -*- coding: utf-8 -*-
"""
High-Performance NumPy-Vectorized 3D Region Growing Segmenter
No Numba JIT compiling required. Optimized for Windows 10 multi-threading.
"""

import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree, NearestNeighbors
import gc


class TreeInstanceSegmentation:
    """
    High-concurrency 3D region growing segmenter (based on pure NumPy matrix operations):
    1. Uses segmented continuous Z-axis spatial compression 
       to promote tree upward growth (4.0x compression for low heights, 2.0x compression for high heights).
    2. Multi-source synchronized wavefront diffusion (BFS), 
       supports multi-core parallel neighbor retrieval and batch optimization.
    3. Introduces dual-key sorting (np.lexsort) 
       to solve nearest neighbor competition at tree junctions at the NumPy level.
    4. Uses 1.3m tree height adaptive expansion lock to prevent lower layers from leaking to shrubs.
    5. Integrated a 99.0% high-precision early stopping mechanism 
       to prevent boundary breaches caused by large step grabs later on.
    6. Encapsulates "tree seed-guided variable-radius re-growth" self-healing purification algorithm,
       using 30th percentile (D_alt) for continuous adaptive pruning of skirt points.
    """
    def __init__(
        self,
        voxel_size=0.18,    # Physical point spacing step size (meters)
        z_scale=2.0,    # Default Z-axis spatial compression factor (to force individual trees to grow upward and block lateral competition)
        min_total_assignment_ratio=0.00001,     # Extremely small assignment rate threshold
        min_tree_assignment_ratio=0.3,    # Single-tree growth rate threshold for automatic radius adjustment
        max_search_radius=1.5,    # Maximum search radius limit (meters)
        decrease_search_radius_after_num_iter=10, # Continuous iteration count to trigger radius convergence
        max_iterations=500,    # Maximum iteration count for region growing
        early_stop_ratio=0.9900    # Stop early and lower it to 99.00% to avoid aggressive point grabbing later with a large radius.
    ):
        self.voxel_size = voxel_size
        self.z_scale = z_scale
        self.min_total_assignment_ratio = min_total_assignment_ratio
        self.min_tree_assignment_ratio = min_tree_assignment_ratio
        self.max_search_radius = max_search_radius
        self.decrease_search_radius_after_num_iter = decrease_search_radius_after_num_iter
        self.max_iterations = max_iterations
        self.early_stop_ratio = early_stop_ratio

    def segment_cloud(self, points, rel_heights, trunk_centers, cylinder_seed_mask, verticality=None):
        """
        Core segmentation function:
        Input:
            points: shape (N, 3) original absolute 3D coordinates
            rel_heights: shape (N,) relative ground height
            trunk_centers: shape (M, 3) absolute 3D center coordinates of tree trunks at 1.3m height (M trees)
            cylinder_seed_mask: shape (N,) boolean mask identifying cylindrical seed sources in original point cloud
            verticality: shape (N,) pre-calculated point-level verticality features. If None, no low-altitude pre-cleaning is performed.
        Returns:
            final_labels: shape (N,) single-tree segmentation ID array for entire point cloud.
                          Ground and unassigned noise points marked as -1, single-trees marked as 0 ~ M-1.
        """
        n_points = len(points)
        n_trees = len(trunk_centers)
        
        # Ground isolation cutting
        print("[Region Growing] Executing ground hard cutoff isolation...")
        veg_mask = rel_heights >= 0.5
        
        # Low-altitude verticality cleaning (protect high-precision seed points from interference)
        if verticality is not None:
            # Vegetation points below 1.3m that are not seed points and have verticality < 0.8 are classified as flat ground shrubs/grass
            # This is modified to 1.3m to fully protect small trees between 1.3-2.0m height from over-cutting effects on low branches
            low_alt_shrub_mask = (rel_heights < 1.3) & (~cylinder_seed_mask) & (verticality < 0.80)
            # Force remove these points from growable vegetation points (set as ground background)
            veg_mask = veg_mask & (~low_alt_shrub_mask)
            print(f"Successfully identified and filtered 1.3m below non-upright ground shrubs/grass: {np.sum(low_alt_shrub_mask):,} points.")
            
        veg_indices = np.where(veg_mask)[0]
        veg_points = points[veg_mask].copy()
        veg_rel_heights = rel_heights[veg_mask]
        
        # Layered continuous anisotropic Z-Scale (based on fixed height 3.0m)
        z_scale_low = 4.0    # Low height (below 3.0m) uses 4.0x Z-axis compression, strongly enhances tree trunk vertical climbing desire
        z_scale_high = self.z_scale    # High height uses default 2.0x Z-axis compression, allowing canopy to naturally spread horizontally
        
        # Use relative height difference to calculate each point's true local ground height
        ground_z = veg_points[:, 2] - veg_rel_heights
        
        # Perform continuous segmented Z-axis scaling (Z' = Z_ground + scaled_relative_height)
        scaled_rel_heights = np.zeros_like(veg_rel_heights)
        low_mask = veg_rel_heights < 3.0
        # Relative height < 3m: divide proportionally by 4.0 (height at 3.0m is 0.75m)
        scaled_rel_heights[low_mask] = veg_rel_heights[low_mask] / z_scale_low
        # Relative height >= 3m: start from 3m/4.0 point and smoothly transition with slope of 1/2.0
        scaled_rel_heights[~low_mask] = 0.75 + (veg_rel_heights[~low_mask] - 3.0) / z_scale_high
        
        veg_points_z_scaled = veg_points.copy()
        veg_points_z_scaled[:, 2] = ground_z + scaled_rel_heights
        
        print("[Region Growing] Building global vegetation 3D spatial index tree...")
        # Build tree only once, avoid rebuilding in loop
        tree = cKDTree(veg_points_z_scaled)
        
        # Initialize local label array for vegetation
        labels = np.full(len(veg_points), -1, dtype=np.int32)
        
        # Seed point initialization: map global cylinder_seed_mask 
        # to local vegetation points and perform 2D nearest neighbor matching assignment ID
        global_seeds_mask = cylinder_seed_mask & veg_mask
        global_seeds_indices = np.where(global_seeds_mask)[0]
        
        if len(global_seeds_indices) == 0:
            print("[Warning] No valid cylindrical seed points detected, growth ends prematurely.")
            return np.full(n_points, -1, dtype=np.int32)
            
        seed_points_for_id = points[global_seeds_indices]
        # Use 2D tree to find nearest trunk_center index for these seed points
        kd_tree_centers = cKDTree(trunk_centers[:, :2])
        _, seed_tree_ids = kd_tree_centers.query(seed_points_for_id[:, :2], k=1)
        
        # Establish bidirectional mapping from global indices to local veg indices
        global_to_local_map = np.full(n_points, -1, dtype=np.int32)
        global_to_local_map[veg_mask] = np.arange(len(veg_points), dtype=np.int32)
        
        local_seed_indices = global_to_local_map[global_seeds_indices]
        labels[local_seed_indices] = seed_tree_ids.astype(np.int32)
        
        # Active growing wavefront queue (first wave seeds)
        active_seeds = local_seed_indices.copy()
        
        search_radius = self.voxel_size
        search_radius_squared = search_radius * search_radius
        iterations_without_radius_increase = 0
        
        # Global multi-source synchronized wavefront diffusion loop
        print(f"[Region Growing] Initiating high-concurrency synchronized growth (max iterations: {self.max_iterations}）...")
        for iteration in range(self.max_iterations):
            # Completely remove early exit restriction on search_radius > self.max_search_radius, change to only exit when no seeds left
            if len(active_seeds) == 0:
                break
                
            seed_coords_z_scaled = veg_points_z_scaled[active_seeds]
            
            # Batch query and instant flattening optimization, Completely eliminate memory surge
            chunk_size = 50000  
            flat_neighbors_list = []
            neighbor_counts_list = []
            
            for start_idx in range(0, len(seed_coords_z_scaled), chunk_size):
                end_idx = min(start_idx + chunk_size, len(seed_coords_z_scaled))
                chunk_coords = seed_coords_z_scaled[start_idx:end_idx]
                
                # Batch multi-threaded query current batch of seeds' spatial neighbors
                chunk_neighbors = tree.query_ball_point(chunk_coords, r=search_radius, workers=-1)
                
                # Instantly record neighbor count for each seed in this batch
                chunk_counts = np.array([len(n) for n in chunk_neighbors], dtype=np.int32)
                neighbor_counts_list.append(chunk_counts)
                
                # Instantly flatten this batch into a continuous NumPy 1D array
                if len(chunk_neighbors) > 0:
                    flat_chunk = np.concatenate(chunk_neighbors)
                    flat_neighbors_list.append(flat_chunk)
                    
            # merge all batch results
            neighbor_counts = np.concatenate(neighbor_counts_list)
            total_neighbors = np.sum(neighbor_counts)
            
            # If no new points detected within current radius, try expanding search radius
            if total_neighbors == 0:
                # Lower layer anti-leak expansion lock. If average height of active seeds is below 2.5m, force not to increase radius!
                avg_seed_height = np.mean(veg_rel_heights[active_seeds]) if len(active_seeds) > 0 else 0.0
                if avg_seed_height < 2.5:
                    search_radius = self.voxel_size    # Lock at base step size, never spread to shrubs
                else:
                    search_radius += self.voxel_size
                    search_radius = min(search_radius, self.max_search_radius)
                
                search_radius_squared = search_radius * search_radius
                active_seeds = np.where(labels != -1)[0]    # Reset active seeds to all assigned points
                gc.collect()    # Force garbage collection
                continue
                
            # Concatenate all batch neighbor index arrays
            flat_neighbors = np.concatenate(flat_neighbors_list)
            
            # Get Tree ID labels of each neighbor's discoverer (seed)
            seed_labels = labels[active_seeds]
            repeated_labels = np.repeat(seed_labels, neighbor_counts)
            
            # Fast filtering: only process neighbors that haven't been claimed yet
            unassigned_mask = labels[flat_neighbors] == -1
            
            if not np.any(unassigned_mask):
                avg_seed_height = np.mean(veg_rel_heights[active_seeds]) if len(active_seeds) > 0 else 0.0
                if avg_seed_height < 2.5:
                    search_radius = self.voxel_size
                else:
                    search_radius += self.voxel_size
                    search_radius = min(search_radius, self.max_search_radius)
                search_radius_squared = search_radius * search_radius
                active_seeds = np.where(labels != -1)[0]
                gc.collect()    # Force garbage collection
                continue
                
            flat_neighbors = flat_neighbors[unassigned_mask]
            repeated_labels = repeated_labels[unassigned_mask]
            
            # Core NumPy matrix design: dual-key sorting (np.lexsort) for rapid solving of nearest neighbor point competition
            repeated_seeds_idx = np.repeat(active_seeds, neighbor_counts)[unassigned_mask]
            distances = np.linalg.norm(
                veg_points_z_scaled[flat_neighbors] - veg_points_z_scaled[repeated_seeds_idx], 
                axis=1
            )
            
            # Sort candidate neighbor indices once, then sort by 3D distance to seed twice
            sort_idx = np.lexsort((distances, flat_neighbors))
            flat_neighbors_sorted = flat_neighbors[sort_idx]
            repeated_labels_sorted = repeated_labels[sort_idx]
            
            # Extract unique candidates and lock their first occurrence position in sorted sequence (i.e., shortest distance position)
            unique_neighbors, first_idx = np.unique(flat_neighbors_sorted, return_index=True)
            
            # Vectorized rapid assignment
            labels[unique_neighbors] = repeated_labels_sorted[first_idx]
            
            # Radius expansion adaptive control and active queue update
            n_unassigned_total = (labels == -1).sum()
            
            # High-precision adaptive early stopping: lowered to 99.0%
            assigned_ratio = 1.0 - (n_unassigned_total / len(labels))
            if assigned_ratio >= self.early_stop_ratio:
                print(f"[Early convergence] Global vegetation assignment rate reached {assigned_ratio*100:.4f}% >= {self.early_stop_ratio*100:.4f}%, triggering high-precision early stopping mechanism!")
                break
                
            newly_assigned_points_ratio = len(unique_neighbors) / max(1, n_unassigned_total)
            
            growing_trees_mask = np.zeros(n_trees, dtype=bool)
            growing_trees_mask[repeated_labels_sorted[first_idx]] = True
            tree_assignment_ratio = np.sum(growing_trees_mask) / n_trees
            
            # If assignment rate is too low, expand radius
            if newly_assigned_points_ratio < self.min_total_assignment_ratio or \
               tree_assignment_ratio < self.min_tree_assignment_ratio:
                
                # Clamped expansion
                avg_seed_height = np.mean(veg_rel_heights[active_seeds]) if len(active_seeds) > 0 else 0.0
                if avg_seed_height < 2.5:
                    search_radius = self.voxel_size
                else:
                    search_radius += self.voxel_size
                    search_radius = min(search_radius, self.max_search_radius)
                search_radius_squared = search_radius * search_radius
                
                # Reuse all assigned points as probe wavefronts
                active_seeds = np.where(labels != -1)[0]
                iterations_without_radius_increase = 0
            else:
                active_seeds = unique_neighbors
                iterations_without_radius_increase += 1
                
            # Automatic convergence mechanism (introduce 1.5x voxel protection line to prevent floating point precision error from contracting to 0.0m)
            if search_radius > self.voxel_size * 1.5 and \
               iterations_without_radius_increase == self.decrease_search_radius_after_num_iter:
                search_radius -= self.voxel_size
                iterations_without_radius_increase = 0

            print(f"[Growth iteration] Iteration {iteration:3d} | Current search radius: {search_radius:.2f}m | Active seeds: {len(active_seeds):7,}")

            gc.collect()

        # Data restoration and global alignment
        final_labels = np.full(n_points, -1, dtype=np.int32)
        final_labels[veg_mask] = labels
        
        return final_labels

    def clean_labels(self, points, rel_heights, segmented_labels, cylinder_seed_mask, trunk_centers):
        """
        Self-healing purification algorithm based on "tree seed-guided variable-radius re-growth"
        Highly intelligent: uses 30th percentile of Z-axis point cloud density distribution as D_alt (canopy base height),
        implements the most rigorous segmented anisotropic Z-Scale physical constraints and height-sensitive variable-radius purification.
        """
        import time
        from scipy.spatial import cKDTree
        from tqdm import tqdm
        import gc

        print("\n Initiating 'tree seed-guided variable-radius (D_alt) re-growth' purification mechanism (cutting grass/shrub skirt)...")
        start_clean_time = time.time()
        cleaned_labels = segmented_labels.copy()
        
        # Count current independent tree IDs participating in high-resolution purification (excluding -1)
        unique_labels = np.unique(segmented_labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        # Record cumulative number of stripped and cut noise points
        stripped_count = 0
        
        # For 10cm resolution purification, set step size to tight 12cm and 20cm
        # Perform simple BFS within each tree without competition
        for i in tqdm(unique_labels, desc="Tree physical connectivity check"):
            # Extract local indices and properties of current tree in 10cm high-resolution point cloud
            tree_idx = np.where(segmented_labels == i)[0]
            if len(tree_idx) == 0:
                continue
                
            tree_pts = points[tree_idx]
            tree_rel_heights = rel_heights[tree_idx]
            tree_seed_mask = cylinder_seed_mask[tree_idx]
            
            # Initial active seeds (local indices)
            active_seeds = np.where(tree_seed_mask)[0]
            
            # Extreme tolerance: if current tree contains no cylindrical seed points, directly treat it as "seedless orphan" and remove the entire tree
            if len(active_seeds) == 0:
                cleaned_labels[tree_idx] = -1
                stripped_count += len(tree_idx)
                continue
                
            # 30th percentile estimation of Canopy Base Height (D_alt) driven by point density
            # Estimate tree-specific canopy base height D_alt
            d_alt = np.percentile(tree_rel_heights, 30.0)
            # Apply safe clamping to prevent boundary line from being distorted due to extreme noise like over-saturation or occlusion
            d_alt = np.clip(d_alt, 1.3, 8.0)
            
            # Perform "segmented continuous anisotropic Z-Scale" transformation for current tree point cloud (ensure mathematical absolute continuity at boundary)
            ground_z = tree_pts[:, 2] - tree_rel_heights
            scaled_rel = np.zeros_like(tree_rel_heights)
            low_mask = tree_rel_heights < d_alt
            
            z_scale_low = 4.0    # Low height (below D_alt) uses 4.0x Z-axis compression, strongly enhance vertical climbing of trunk
            z_scale_high = self.z_scale    # High height uses default 2.0x Z-axis compression
            
            # Continuity transition constant at boundary point
            c_split = d_alt / z_scale_low
            
            # Segmented continuous proportional compression
            scaled_rel[low_mask] = tree_rel_heights[low_mask] / z_scale_low
            scaled_rel[~low_mask] = c_split + (tree_rel_heights[~low_mask] - d_alt) / z_scale_high
            
            tree_pts_z_scaled = tree_pts.copy()
            tree_pts_z_scaled[:, 2] = ground_z + scaled_rel
            
            # Build local spatial search tree for current mini-tree point set
            local_tree = cKDTree(tree_pts_z_scaled)
            
            # Local label status: -1 means unvisited, 0 means covered by current tree's seed wavefront
            local_labels = np.full(len(tree_pts), -1, dtype=np.int32)
            local_labels[active_seeds] = 0
            
            # Local height-related variable-radius BFS growth loop
            while len(active_seeds) > 0:
                # Calculate average relative elevation of current active seeds
                avg_h = np.mean(tree_rel_heights[active_seeds])
                
                # Variable radius strategy combined with D_alt：
                # Low height (< D_alt) extremely restricted: step size set to tight 0.12m, block any lateral grass/shrub
                # High height (>= D_alt) release: restore standard 0.20m, let canopy grow freely
                r = 0.12 if avg_h < d_alt else 0.20
                
                # Query local neighborhood (since dataset is very small, no multi-core enabled)
                neighbors_list = local_tree.query_ball_point(tree_pts_z_scaled[active_seeds], r=r, workers=1)
                
                # Flatten neighbor indices
                flat_neighbors = np.concatenate(neighbors_list) if len(neighbors_list) > 0 else np.empty(0, dtype=np.int64)
                if len(flat_neighbors) == 0:
                    break
                    
                unique_neighbors = np.unique(flat_neighbors)
                
                # Only keep unvisited points
                unassigned_neighbors = unique_neighbors[local_labels[unique_neighbors] == -1]
                if len(unassigned_neighbors) == 0:
                    break
                    
                # Status update
                local_labels[unassigned_neighbors] = 0
                active_seeds = unassigned_neighbors
                
            # Point-specific removal: Remove isolated grass/shrub/noise points not reached by adaptive variable-radius growth wavefront in current tree
            unreached_mask = local_labels == -1
            if np.any(unreached_mask):
                cleaned_labels[tree_idx[unreached_mask]] = -1
                stripped_count += np.sum(unreached_mask)
                
        print(f"[Re-growth purification complete] Total time: {time.time() - start_clean_time:.2f} sec.")
        print(f"[Re-growth purification complete] Successfully stripped and cut ground-level grass/shrub skirt/noise points: {stripped_count:,} points.")
        gc.collect()
        
        return cleaned_labels


def fuse_seeds(npz_files, config):
    """
    Globally deduplicate seeds from multiple Tiles and simultaneously deduplicate, align and return trunk diameter data (diameters).
    Perfectly suitable for EVO's cross-tile seed self-healing re-growth workflow.
    """
    # Read deduplication overlap distance threshold (default 0.6m)
    dist_threshold = config.get("seed_fuse_dist", 0.6)
    
    all_pos = []
    all_scores = []
    all_diameters = []
    
    # Batch load each Tile results
    for f in npz_files:
        data = np.load(f)
        all_pos.append(data['centroids'])
        all_scores.append(data['scores'])
        
        # Load diameter: if some individual tiles are missing, use default 0.25m diameter as fallback to prevent crash
        if 'diameters' in data:
            all_diameters.append(data['diameters'])
        else:
            all_diameters.append(np.full(len(data['centroids']), 0.25))
            
    # Vertically merge into global arrays
    all_pos = np.vstack(all_pos)
    all_scores = np.concatenate(all_scores)
    all_diameters = np.concatenate(all_diameters)
    
    # Spatial search tree deduplication (use BallTree to quickly find overlapping seed clusters)
    tree = BallTree(all_pos)
    indices = tree.query_radius(all_pos, r=dist_threshold)
    
    unique_mask = np.ones(len(all_pos), dtype=bool)
    final_indices = []
    
    # Vectorized deduplication selection: for seeds that overlap in space across multiple tiles, keep the one with highest confidence score, simultaneously remove weak diameter
    for i, cluster in enumerate(indices):
        if not unique_mask[i]: 
            continue
        if len(cluster) > 1:
            # Multiple seeds spatially overlapping, keep the one with highest Score confidence
            best_idx = cluster[np.argmax(all_scores[cluster])]
            final_indices.append(best_idx)
            unique_mask[cluster] = False
        else:
            final_indices.append(i)
            unique_mask[i] = False
            
    # Simultaneously return deduplicated, aligned centroids, confidence scores and fitted diameters
    return all_pos[final_indices], all_scores[final_indices], all_diameters[final_indices]