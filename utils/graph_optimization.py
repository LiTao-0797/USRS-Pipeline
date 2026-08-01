# -*- coding: utf-8 -*-
from utils.pose_graph import PoseGraph
import utils.graph_visualization as vis
import gtsam
import open3d as o3d
import logging
import numpy as np


class PoseGraphOptimization:
    def __init__(self, pose_graph: PoseGraph, debug, show_clouds=False, logger=None):
        self.pose_graph = pose_graph
        self.show_clouds = show_clouds
        self.factor_graph = None
        self.debug = debug
        self.logger = logger or logging.getLogger(
            __name__ + ".null"
        )  # no-op logger if logger is not provided
        self.initialize_factor_graph()

    def initialize_factor_graph(self):
        # Create a factor graph container and add factors to it
        factor_graph = gtsam.NonlinearFactorGraph()

        # add the other edges
        for e in self.pose_graph.edges:
            if e["type"] == "in-between":
                noise = gtsam.noiseModel.Gaussian.Information(e["info"])
                factor_graph.add(
                    gtsam.BetweenFactorPose3(
                        e["parent_id"], e["child_id"], e["pose"], noise
                    )
                )
            elif e["type"] == "aerial":
                noise = gtsam.noiseModel.Gaussian.Information(e["info"])
                factor_graph.add(
                    gtsam.PriorFactorPose3(e["parent_id"], e["pose"], noise)
                )
            elif e["type"] == "loop-closure":
                noise = gtsam.noiseModel.Gaussian.Information(e["info"])
                robust_model = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.DCS.Create(1.0), noise
                )
                factor_graph.add(
                    gtsam.BetweenFactorPose3(
                        e["child_id"], e["parent_id"], e["pose"], robust_model
                    )
                )

        self.factor_graph = factor_graph

    def visualize(self, estimate):
        # Use the filename in the current directory instead of Linux's /tmp
        dot_file = "factor_graph_debug.dot"
        
        try:
            self.factor_graph.saveGraph(dot_file, estimate)
            print(f"Factor graph structure saved to: {dot_file}")

            # Add exception protection to prevent crash if Graphviz is not installed
            try:
                from graphviz import Source
                s = Source.from_file(dot_file)
                # On Windows, view() will attempt to open the default DOT viewer
                # Uncomment the line below to skip automatically opening a window
                #s.view() 
            except ImportError:
                print("Tip: Python-graphviz library not installed, skipping graphical display.")
            except Exception as e:
                print(f"Tip: Unable to preview factor graph (may be missing Graphviz software): {e}")
                
        except Exception as e:
            print(f"Saving factor graph file failed: {e}")

    def optimize(self):
        # Display initial node poses
        for id, node in self.pose_graph.nodes.items():
            self.logger.info(f"Node {id}:")
            self.logger.info(f"  pose: {node['pose']}")

        if self.debug:
            geometries = vis.graph_to_geometries(
                self.pose_graph,
                show_frames=True,
                show_edges=True,
                show_nodes=True,
                show_clouds=False,
                show_coordinate_frame=True,
                odometry_color=vis.GRAY,
                loop_color=vis.RED,
            )
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Initial Pose Graph",
            )

        if self.show_clouds and self.debug:
            geometries = vis.graph_to_geometries(
                self.pose_graph,
                show_frames=True,
                show_edges=True,
                show_nodes=True,
                show_clouds=True,
                show_coordinate_frame=True,
                odometry_color=vis.GRAY,
                loop_color=vis.RED,
            )
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Initial Pose Graph with Clouds",
            )

        initial_estimate = gtsam.Values()
        for id, node in self.pose_graph.nodes.items():
            initial_estimate.insert(id, node["pose"])

        if self.debug:
            self.visualize(initial_estimate)

        parameters = gtsam.GaussNewtonParams()
        optimizer = gtsam.GaussNewtonOptimizer(
            self.factor_graph, initial_estimate, parameters
        )

        # Optimize
        self.logger.info("Optimizing factor graph...")
        result = optimizer.optimize()

        for id, _ in self.pose_graph.nodes.items():
            self.pose_graph.set_node_pose(id, result.atPose3(id))

        # Display results
        for id, node in self.pose_graph.nodes.items():
            self.logger.info(f"Node {id}: pose: {node['pose']}")

        if self.debug:
            geometries = vis.graph_to_geometries(
                self.pose_graph,
                show_frames=True,
                show_edges=True,
                show_nodes=True,
                show_clouds=False,
                show_coordinate_frame=True,
                odometry_color=vis.GRAY,
                loop_color=vis.RED,
            )
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Optimized Pose Graph",
            )

        if self.show_clouds and self.debug:
            geometries = vis.graph_to_geometries(
                self.pose_graph,
                show_frames=True,
                show_edges=True,
                show_nodes=True,
                show_clouds=True,
                show_coordinate_frame=True,
                odometry_color=vis.GRAY,
                loop_color=vis.RED,
            )
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Optimized Pose Graph with clouds",
            )
    
    def calculate_paper_metrics(self, uav_cloud):
        """
        Calculate paper metrics: including Pre-Err, Post-Err, CloudShift, GraphShift
        """
        # Define header and alignment format
        # header = f"{'Tile ID':<15} | {'Pre-Err[m]':<12} | {'Post-Err[m]':<12} | {'CloudShift[m]':<13} | {'GraphShift[m]':<13}"
        header = f"{'Tile ID':<15} | {'Pre-Err[m]':<12} | {'Post-Err[m]':<12}"
        print("\n" + "=" * len(header))
        print(header)
        print("-" * len(header))

        uav_legacy = uav_cloud.to_legacy().voxel_down_sample(0.1)
        results = {}

        for node_id, node in self.pose_graph.nodes.items():
            try:
                tile_name = self.pose_graph.get_node_cloud_name(node_id)
                if not tile_name: continue
                
                cloud = self.pose_graph.get_node_cloud(node_id)
                init_pose = self.pose_graph.get_initial_node_pose(node_id)
                opt_pose = self.pose_graph.get_node_pose(node_id)

                # Calculate error before optimization
                pre_cloud = cloud.clone()
                pre_dists = pre_cloud.to_legacy().compute_point_cloud_distance(uav_legacy)
                pre_err = np.mean(pre_dists)

                # Calculate error after optimization
                post_cloud = cloud.clone()
                rel_transform = opt_pose.matrix() @ np.linalg.inv(init_pose.matrix())
                post_cloud.transform(rel_transform)
                post_dists = post_cloud.to_legacy().compute_point_cloud_distance(uav_legacy)
                post_err = np.mean(post_dists)

                # # Calculate displacement (Shifts)
                # # Graph Shift: movement of the node center
                # g_shift = np.linalg.norm(opt_pose.translation() - init_pose.translation())
                # # Cloud Shift: theoretically equal to g_shift, for strict alignment with the paper, we list it separately
                # c_shift = g_shift 

                # Print each line corresponding to Table I format
                # print(f"{tile_name:<15} | {pre_err:<12.4f} | {post_err:<12.4f} | {c_shift:<13.4f} | {g_shift:<13.4f}")
                print(f"{tile_name:<15} | {pre_err:<12.4f} | {post_err:<12.4f} ")
                
                results[node_id] = [pre_err, post_err]

            except Exception as e:
                # Only print skip message for empty nodes in debug mode
                pass

        print("=" * len(header))
        return results

    def calculate_table2_metrics(self, uav_cloud):
        """
        Calculate and display statistical distribution of each tile after optimization (Mean & Std)
        """
        print("\n" + "="*65)
        print(f"{'Tile ID':<20} | {'Mean Error [m]':<20} | {'Std Dev [m]':<15}")
        print("-" * 65)

        uav_legacy = uav_cloud.to_legacy().voxel_down_sample(0.1)
        table2_results = {}

        for node_id, node in self.pose_graph.nodes.items():
            try:
                tile_name = self.pose_graph.get_node_cloud_name(node_id)
                if not tile_name: continue

                cloud = self.pose_graph.get_node_cloud(node_id).clone()
                init_pose = self.pose_graph.get_initial_node_pose(node_id)
                opt_pose = self.pose_graph.get_node_pose(node_id)
                
                rel_transform = opt_pose.matrix() @ np.linalg.inv(init_pose.matrix())
                cloud.transform(rel_transform)
                
                dists = np.asarray(cloud.to_legacy().compute_point_cloud_distance(uav_legacy))
                
                mean_val = np.mean(dists)
                std_val = np.std(dists)
                
                # Use print to directly output each line of data
                print(f"{tile_name:<20} | {mean_val:<20.4f} | {std_val:<15.4f}")
                
                table2_results[node_id] = (mean_val, std_val)
            except Exception:
                pass

        print("="*65)
        return table2_results
