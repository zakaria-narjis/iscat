"""
write  a test script that uses argparse and src.infrence.inference.SegInference the script should take as arg an  nd2  path  and experiment path, the nd2 image has (200,H,W) shape so you init SegInference using the config file in the experiment you can get the image size that the model was trained on. You must create first a function that cut the image into image_size by image_size regions for example (200,256,256)
"""
import torch
import numpy as np
import monai
from monai.metrics import DiceMetric, compute_iou
from monai.networks.utils import one_hot
from src.metrics import batch_multiclass_metrics

def test_model(model, test_loader, device, num_classes):
    """
    Comprehensive test function for segmentation metrics
    
    Args:
        model (nn.Module): Trained segmentation model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to run inference on
        num_classes (int): Number of classes in segmentation task
    
    Returns:
        dict: Comprehensive metrics dictionary
    """
    model.eval()
    
    # Metrics containers
    total_loss = 0
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    miou_metric = monai.metrics.MeanIoU(include_background=False,reduction="mean")
    
    # Class-specific metrics 
    class_metrics = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(num_classes)}
    total_tp = total_fp = total_fn = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            # Predictions
            predictions = model(images)
            if predictions.shape[1] == 1:
                pred_masks = torch.sigmoid(predictions) > 0.5 # Shape: [batch_size, 1, height, width]
                pred_masks = pred_masks.squeeze(1) # Shape: [batch_size, height, width]
                pred_one_hot = one_hot(pred_masks.unsqueeze(1), num_classes=2)
                target_one_hot = one_hot(masks.unsqueeze(1), num_classes=2)    
            else:
                pred_masks = torch.argmax(predictions, dim=1) # Shape: [batch_size, height, width]
                pred_one_hot = one_hot(pred_masks.unsqueeze(1), num_classes=num_classes)
                target_one_hot = one_hot(masks.unsqueeze(1), num_classes=num_classes)           
                    
            # Compute metrics
            dice_metric(y_pred=pred_one_hot, y=target_one_hot)
            miou_metric(y_pred=pred_one_hot, y=target_one_hot)
            
            pred_np = pred_masks.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            # Detailed class metrics
            batch_class_metrics = batch_multiclass_metrics(pred_np, masks_np)
            
            # Aggregate metrics
            for class_id, (tp, fp, fn) in batch_class_metrics.items():
                class_metrics[class_id]['tp'] += tp
                class_metrics[class_id]['fp'] += fp
                class_metrics[class_id]['fn'] += fn
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
    
    # Aggregate metrics correctly
    dice_scores = dice_metric.aggregate().cpu().numpy()
    miou_scores = miou_metric.aggregate().cpu().numpy()

    # Reset metrics to clear accumulated values
    dice_metric.reset()
    miou_metric.reset()
    
    # Precision, Recall, F1 computations
    class_precision_recall = {}
    for class_id, metrics in class_metrics.items():
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_precision_recall[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # Total metrics
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
    
    return {
    'segmentation_scores': {
        'mIoU': float(np.nanmean(miou_scores)),  
        'dice_mean': float(np.nanmean(dice_scores)),
        'dice_per_class': dice_scores.tolist(),
    },
    'detection_scores': {
        'total_precision': total_precision,
        'total_recall': total_recall,
        'total_f1': total_f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'class_metrics': class_precision_recall
    }
}
