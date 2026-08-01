# -*- coding: utf-8 -*-
"""
Accuracy Evaluation and Partitioned Trend Analysis Module
Directly transplanted from the verified legacy seg_utils code.
"""

import numpy as np
import scipy.optimize
from tqdm import tqdm


def get_eval_components(preds_mask, labels_mask):
    tp = (preds_mask & labels_mask).sum()
    fp = (preds_mask & ~labels_mask).sum()
    fn = (~preds_mask & labels_mask).sum()
    tn = (~preds_mask & ~labels_mask).sum()
    return tp, fp, tn, fn


def calculate_metrics(tp, fp, fn):
    """caculate Prec, Rec, IoU"""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return prec, rec, iou


def perform_tree_matching(instance_labels, instance_preds, min_iou=0.5):
    """
    Use the Hungarian algorithm for single tree matching and calculate Prec, Rec, Cov
    """
    unique_preds = np.unique(instance_preds[instance_preds >= 0])
    unique_gts = np.unique(instance_labels[instance_labels > 0])
    
    pred_map = {val: i for i, val in enumerate(unique_preds)}
    gt_map = {val: i for i, val in enumerate(unique_gts)}
    
    # Create a matrix to store the results
    iou_matrix = np.zeros((len(unique_preds), len(unique_gts)))
    prec_matrix = np.zeros_like(iou_matrix)
    rec_matrix = np.zeros_like(iou_matrix)
    
    print(f"[Matching] Calculating confusion matrix for {len(unique_preds)}x{len(unique_gts)}...")
    for p_val in unique_preds:
        p_mask = (instance_preds == p_val)
        relevant_gts = np.unique(instance_labels[p_mask])
        for g_val in relevant_gts:
            if g_val <= 0: continue
            g_mask = (instance_labels == g_val)
            
            # Calculate point set components
            tp = (p_mask & g_mask).sum()
            fp = (p_mask & ~g_mask).sum()
            fn = (~p_mask & g_mask).sum()
            
            iou_matrix[pred_map[p_val], gt_map[g_val]] = tp / (tp + fp + fn)
            prec_matrix[pred_map[p_val], gt_map[g_val]] = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec_matrix[pred_map[p_val], gt_map[g_val]] = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Hungarian algorithm
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(iou_matrix, maximize=True)
    
    matched_results = []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= min_iou:
            matched_results.append({
                "pred_id": unique_preds[r],
                "gt_id": unique_gts[c],
                "iou": iou_matrix[r, c],
                "prec": prec_matrix[r, c],
                "rec": rec_matrix[r, c]
            })
            
    return matched_results, iou_matrix, unique_preds, unique_gts


def evaluate_partitioned_accuracy(coords, instance_labels, instance_preds, matched_pairs, stem_centers_dict):
    """
    Performs a partitioned accuracy evaluation
    """
    bins = np.linspace(0, 1.0, 11)
    xy_stats = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(10)}
    z_stats = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(10)}

    for pair in tqdm(matched_pairs, desc="Partitioned accuracy statistics"):
        p_id = pair["pred_id"]
        g_id = pair["gt_id"]
        
        p_mask = (instance_preds == p_id)
        g_mask = (instance_labels == g_id)
        
        # --- XY direction (radial distance) ---
        center_xy = stem_centers_dict.get(p_id, np.mean(coords[p_mask, :2], axis=0))[:2]
        dist_xy = np.linalg.norm(coords[:, :2] - center_xy, axis=1)
        max_dist = np.max(dist_xy[g_mask]) if np.any(g_mask) else 1.0
        norm_dist_xy = np.clip(dist_xy / (max_dist + 1e-6), 0, 0.99)
        
        # --- Z direction (vertical height) ---
        min_z = np.min(coords[g_mask, 2]) if np.any(g_mask) else 0
        tree_heights = coords[:, 2] - min_z
        max_h = np.max(tree_heights[g_mask]) if np.any(g_mask) else 1.0
        norm_h = np.clip(tree_heights / (max_h + 1e-6), 0, 0.99)

        for b in range(10):
            m_xy = (norm_dist_xy >= bins[b]) & (norm_dist_xy < bins[b+1])
            tp_xy, fp_xy, _, fn_xy = get_eval_components(p_mask & m_xy, g_mask & m_xy)
            xy_stats[b]["tp"] += tp_xy
            xy_stats[b]["fp"] += fp_xy
            xy_stats[b]["fn"] += fn_xy
            
            m_z = (norm_h >= bins[b]) & (norm_h < bins[b+1])
            tp_z, fp_z, _, fn_z = get_eval_components(p_mask & m_z, g_mask & m_z)
            z_stats[b]["tp"] += tp_z
            z_stats[b]["fp"] += fp_z
            z_stats[b]["fn"] += fn_z

    xy_results = [calculate_metrics(xy_stats[i]["tp"], xy_stats[i]["fp"], xy_stats[i]["fn"]) for i in range(10)]
    z_results = [calculate_metrics(z_stats[i]["tp"], z_stats[i]["fp"], z_stats[i]["fn"]) for i in range(10)]
    
    return np.array(xy_results), np.array(z_results)