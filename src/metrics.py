import numpy as np
from scipy.ndimage import label
from typing import Tuple
from typing import Tuple, Dict

def count_matching_particles(pred_mask: np.ndarray, 
                           gt_mask: np.ndarray) -> Tuple[int, int, int]:
    """
    Count matching particles between prediction and ground truth masks.
    
    Args:
        pred_mask: Binary prediction mask (0 and 1)
        gt_mask: Binary ground truth mask (0 and 1)
        
    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    # Label connected components in both masks
    pred_labeled, num_pred = label(pred_mask)
    gt_labeled, num_gt = label(gt_mask)
    
    # Initialize counters
    tp = 0
    matched_pred_labels = set()
    matched_gt_labels = set()
    
    # For each predicted particle
    for pred_label in range(1, num_pred + 1):
        pred_particle = pred_labeled == pred_label
        
        # Find any overlap with GT particles
        overlapping_gt_labels = set(gt_labeled[pred_particle]) - {0}
        
        if overlapping_gt_labels:
            # If there's any overlap, count as TP
            tp += 1
            matched_pred_labels.add(pred_label)
            matched_gt_labels.update(overlapping_gt_labels)
    
    # Count unmatched predictions as FP and unmatched GT as FN
    fp = num_pred - len(matched_pred_labels)
    fn = num_gt - len(matched_gt_labels)
    
    return tp, fp, fn

def compute_batch_metrics(total_tp: int, total_fp: int, total_fn: int) -> Dict[str, float]:
    """
    Compute precision, recall and F1 score at batch level.
    
    Args:
        total_tp: Total true positives across batch
        total_fp: Total false positives across batch
        total_fn: Total false negatives across batch
        
    Returns:
        Dictionary containing precision, recall and F1 score
    """
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def compute_per_sample_metrics(pred_masks: np.ndarray, 
                             gt_masks: np.ndarray) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Compute both batch-level metrics and per-sample metrics.
    
    Args:
        pred_masks: Predicted masks (batch_size, height, width)
        gt_masks: Ground truth masks (batch_size, height, width)
        
    Returns:
        Tuple of (batch_metrics, per_sample_metrics)
    """
    # Initialize arrays for per-sample metrics
    batch_size = len(pred_masks)
    per_sample_precision = np.zeros(batch_size)
    per_sample_recall = np.zeros(batch_size)
    per_sample_f1 = np.zeros(batch_size)
    
    # Track batch totals
    total_tp = total_fp = total_fn = 0
    
    # Compute metrics for each sample
    for i, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
        # Get counts for this sample
        tp, fp, fn = count_matching_particles(pred, gt)
        
        # Update batch totals
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Compute per-sample metrics
        if tp + fp > 0:
            per_sample_precision[i] = tp / (tp + fp)
        if tp + fn > 0:
            per_sample_recall[i] = tp / (tp + fn)
        if per_sample_precision[i] + per_sample_recall[i] > 0:
            per_sample_f1[i] = 2 * (per_sample_precision[i] * per_sample_recall[i]) / \
                              (per_sample_precision[i] + per_sample_recall[i])
    
    # Compute batch-level metrics
    batch_metrics = compute_batch_metrics(total_tp, total_fp, total_fn)
    
    # Compute means of per-sample metrics
    per_sample_metrics = {
        'precision': per_sample_precision,
        'recall': per_sample_recall,
        'f1_score': per_sample_f1,
        'mean_precision': np.mean(per_sample_precision),
        'mean_recall': np.mean(per_sample_recall),
        'mean_f1': np.mean(per_sample_f1)
    }
    
    return batch_metrics, per_sample_metrics