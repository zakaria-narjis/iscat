import numpy as np
import matplotlib.pyplot as plt
import torch
from src.data_processing.utils import Utils
import logging

pil_logger = logging.getLogger('PIL') 
pil_logger.setLevel(logging.INFO)
plt.set_loglevel (level = 'warning')

def convert_image_8bit(image):
    """
    Convert a 16-bit grayscale image to 8-bit for visualization.

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
    """
    Plots an image with overlaid predicted and ground truth masks for multiple classes.
    
    Args:
        image: The original image (2D or 3D).
        predicted_mask: Predicted segmentation mask (2D, with integer class labels).
        ground_truth_mask: Ground truth segmentation mask (2D, with integer class labels).
        output_path: File path to save the visualization.
    """
    # Define colors for predicted and ground truth masks
    if len(np.unique(predicted_mask)) > 2:
        predicted_colors = {
            1: (0, 255, 0),      # Green for class 1
            2: (0, 0, 255),    # Blue for class 2
        }
        gt_colors = {
            1: (255, 0, 0),      # Red for class 1
            2: (255,255, 0),    # Yellow for class 2
        }
    else:
        predicted_colors = {
            1: (0, 0, 255),    # Blue for class 1
        }
        gt_colors = {
            1: (255,255, 0),    # Yellow for class
        }
    # Normalize image for visualization
    image_normalized = convert_image_8bit(image)
    combined_image = image_normalized.copy()

    # Overlay masks for each class
    for class_label, color in predicted_colors.items():
        class_mask = (predicted_mask == class_label)
        combined_image = overlay_mask(combined_image, class_mask, color=color, alpha=0.5)

    for class_label, color in gt_colors.items():
        class_mask = (ground_truth_mask == class_label)
        combined_image = overlay_mask(combined_image, class_mask, color=color, alpha=0.5)

    # Plot the image and the overlays
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original image visualization
    ax1.imshow(image_normalized, cmap='gray')
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Combined overlay visualization
    ax2.imshow(combined_image)
    ax2.set_title("Combined Overlay")
    ax2.axis("off")

    # Add legend
    legend_elements = []
    for class_label, color in predicted_colors.items():
        legend_elements.append(plt.Line2D([0], [0], color=np.array(color) / 255, lw=2, label=f'Pred Class {class_label}'))
    for class_label, color in gt_colors.items():
        legend_elements.append(plt.Line2D([0], [0], color=np.array(color) / 255, lw=2, linestyle='dashed', label=f'GT Class {class_label}'))
    ax2.legend(handles=legend_elements, loc='lower right', fontsize='x-small')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def batch_plot_images_with_masks(images, predicted_masks, ground_truth_masks, output_dir="output"):
    for idx, (image, predicted_mask, ground_truth_mask) in enumerate(zip(images, predicted_masks, ground_truth_masks)):
        output_path = f"{output_dir}/image_{idx}.png"
        plot_image_with_masks(image[0], predicted_mask, ground_truth_mask, output_path)

def predict(model, dataset, device='cpu', images_indicies=[0,1,2,4]):
    """
    Generate predictions for a set of images in a dataset.
    Args:
        model: Trained model.
        dataset: Dataset object.
        device: Device to use for prediction.
        images_indicies: List of image indices to predict.
    Returns:
        Tuple of lists containing the original images, predicted masks, and ground truth
    """
    model.eval()
    all_pred_masks = []
    all_gt_masks = []
    all_images = []
    for idx in images_indicies:
        while True:
            image, mask = dataset[idx]  # (image: torch.Size([3, 256, 256]), mask: torch.Size([3, 256, 256]))
            if 1 in mask:
                break
        input_image = image.to(device).unsqueeze(0) # torch.Size([1, 3, 256, 256])
        ground_truth_mask = mask.cpu().numpy()  # Shape: (256, 256)

        with torch.no_grad():
            output = model(input_image)  # Shape: [1, num_classes, 256, 256]
            predicted_mask = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()  # Shape: (256, 256)
        all_pred_masks.append(predicted_mask)
        all_gt_masks.append(ground_truth_mask)
        all_images.append(image.cpu().numpy())
    return all_images, all_pred_masks,all_gt_masks