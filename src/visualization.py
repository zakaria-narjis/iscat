import numpy as np
import matplotlib.pyplot as plt
import torch
from src.data_processing.utils import Utils
import logging

pil_logger = logging.getLogger('PIL') 
pil_logger.setLevel(logging.INFO)
plt.set_loglevel (level = 'warning')
def normalize_image(image):
    """
    Normalize a 16-bit grayscale image to 8-bit for visualization.

    Parameters:
        image (ndarray): 16-bit grayscale image.

    Returns:
        ndarray: 8-bit grayscale image.
    """
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    return (image * 255).astype(np.uint8)  # Scale to [0, 255]

def overlay_mask(image, mask, color, alpha=0.5):
    """Overlays a mask on an image with a specified color and transparency."""
    if len(image.shape) == 2:
        overlay = np.stack([image] * 3, axis=-1)
    else:
        overlay = image.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(mask, overlay[:, :, c] * (1 - alpha) + color[c] * alpha, overlay[:, :, c])
    return overlay

def plot_image_with_masks(image, predicted_mask, ground_truth_mask, output_path="Cy5.png"):
    image_normalized = normalize_image(image)
    predicted_mask = predicted_mask.astype(bool)
    ground_truth_mask = ground_truth_mask.astype(bool)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original visualization
    ax1.imshow(image_normalized, cmap='gray')
    ax1.set_title("Original Image")
    ax1.axis("off")
    
    # Combined overlay visualization
    combined_image = image_normalized.copy()
    combined_image = overlay_mask(combined_image, predicted_mask, color=(0, 255, 0), alpha=0.5)  # Green for predicted
    combined_image = overlay_mask(combined_image, ground_truth_mask, color=(255, 0, 0), alpha=0.5)  # Red for ground truth
    
    ax2.imshow(combined_image)
    ax2.set_title("Combined Overlay (Red: Ground Truth, Green: Predicted)")
    ax2.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def batch_plot_images_with_masks(images, predicted_masks, ground_truth_masks, output_dir="output"):
    for idx, (image, predicted_mask, ground_truth_mask) in enumerate(zip(images, predicted_masks, ground_truth_masks)):
        output_path = f"{output_dir}/image_{idx}.png"
        plot_image_with_masks(image[0], predicted_mask, ground_truth_mask, output_path)

def predict(model, dataset, mean, std, device, images_idicies=[0,1,2,4]):
    model.eval()
    all_pred_masks = []
    all_gt_masks = []
    all_images = []
    for idx in images_idicies:
        while True:
            image, mask = dataset[idx]  # (image: torch.Size([3, 224, 224]), mask: torch.Size([3, 224, 224]))
            if 1 in mask:
                break
        input_image = image.to(device).unsqueeze(0) # torch.Size([1, 3, 224, 224])
        input_image = Utils.z_score_normalize(input_image, mean, std)
        ground_truth_mask = mask.cpu().numpy()  # Shape: (224, 224)

        with torch.no_grad():
            output = model(input_image)  # Shape: [1, num_classes, 224, 224]
            predicted_mask = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()  # Shape: (224, 224)
        all_pred_masks.append(predicted_mask)
        all_gt_masks.append(ground_truth_mask)
        all_images.append(image.cpu().numpy())
    return all_images, all_pred_masks,all_gt_masks