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

def count_matching_particles_multiclass(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> Dict[int, Tuple[int, int, int]]:
    """
    Count matching particles between prediction and ground truth masks for multiple classes.
    
    Args:
        pred_mask: Multi-class prediction mask (0 for background, 1+ for different particle classes)
        gt_mask: Multi-class ground truth mask (0 for background, 1+ for different particle classes)
        
    Returns:
        Dictionary mapping class_id to tuple of (true_positives, false_positives, false_negatives)
    """
    # Get unique classes (excluding background class 0)
    classes = sorted(set(np.unique(pred_mask)) | set(np.unique(gt_mask)))
    classes = [c for c in classes if c != 0]
    
    results = {}
    
    # Process each class separately
    for class_id in classes:
        # Create binary masks for current class
        pred_binary = (pred_mask == class_id).astype(np.int32)
        gt_binary = (gt_mask == class_id).astype(np.int32)
        
        # Label connected components
        pred_labeled, num_pred = label(pred_binary)
        gt_labeled, num_gt = label(gt_binary)
        
        # Initialize counters
        tp = 0
        matched_pred_labels = set()
        matched_gt_labels = set()
        
        # For each predicted particle of current class
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
        
        results[class_id] = (tp, fp, fn)
    
    return results

def batch_multiclass_metrics(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray
) -> Dict[int, Tuple[int, int, int]]:
    """
    Process a batch of masks and aggregate the results.
    
    Args:
        pred_masks: Batch of prediction masks [batch_size, height, width]
        gt_masks: Batch of ground truth masks [batch_size, height, width]
        
    Returns:
        Dictionary mapping class_id to aggregated (tp, fp, fn) across the batch
    """
    # Initialize results dictionary
    batch_results = {}
    
    # Process each image in the batch
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        image_results = count_matching_particles_multiclass(pred_mask, gt_mask)
        
        # Aggregate results for each class
        for class_id, (tp, fp, fn) in image_results.items():
            if class_id not in batch_results:
                batch_results[class_id] = [0, 0, 0]
            batch_results[class_id][0] += tp
            batch_results[class_id][1] += fp
            batch_results[class_id][2] += fn
    
    # Convert lists to tuples in final results
    return {k: tuple(v) for k, v in batch_results.items()}