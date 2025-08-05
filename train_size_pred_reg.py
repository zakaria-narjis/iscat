import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.trainers.trainersizereg import ParticleSizeTrainer
# from src.data_processing.size_dataset import ParticleDatasetReg  
from src.data_processing.size_dataset import ParticleDatasetReg, generate_global_size_labels  
from src.models.resnet import ResNet18
import re
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import json
from torchvision.transforms import v2
from matplotlib import pyplot as plt
import logging
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def contrast_from_middle_rows(images: torch.Tensor) -> torch.Tensor:
    """
    Compute standard deviation (RMS contrast) from the 4 middle rows of each image in a batch.
    
    Args:
        images: Tensor of shape (B, C, H, W)
    
    Returns:
        Tensor of shape (B,) with contrast values.
    """
    B, C, H, W = images.shape
    mid = H // 2
    rows = images[:, :, mid - 2:mid + 2, :]  # (B, C, 4, W)

    gray = rows.mean(dim=1)  # Convert to grayscale: (B, 4, W)
    contrast = gray.std(dim=(1, 2))  # Compute std over height and width

    return contrast  # shape: (B,)

def save_metrics_to_json(metrics_dict, output_folder):
    """
    Save training metrics dictionary to a JSON file.

    Args:
        metrics_dict (dict): Metrics dictionary from training
        output_folder (str): Folder path to save the JSON file

    Returns:
        str: Full path to the saved JSON file
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"size_pred_metrics_{timestamp}.json"

    # Full path for JSON file
    full_path = os.path.join(output_folder, filename)

    # Save metrics to JSON
    with open(full_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    return full_path


def load_config(config_path):
    """
    Load configuration file with variable interpolation support using OmegaConf.
    """
    # Load the YAML file using OmegaConf
    config = OmegaConf.load(config_path)

    # Resolve all variable interpolations
    resolved_config = OmegaConf.to_container(config, resolve=True)

    return resolved_config


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="iScat Size Prediction Regression")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/size_reg_config.yaml",
        help="Path to the configuration file",
    )
    return parser


def create_dataloaders(dataset, batch_size, num_workers=0):
    data_loader = DataLoader(
        dataset,
        batch_size=min(len(dataset), batch_size),  # Ensure batch size is at least 1
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    return data_loader


def sanitize_filename(name):
    return re.sub(r"[^\w\-_\. ]", "_", name)


def set_random_seed(seed):
    """
    Set the random seed for reproducibility in Python, NumPy, PyTorch, and CUDA.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)  # Python random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch all GPUs

    # Ensure deterministic behavior (may slow down performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def getdatetime():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def write_config_to_tensorboard(writer, config):
    """
    Write selected configuration parameters to TensorBoard.
    Handles both scalar and list values in the config.

    Args:
        writer: TensorBoard SummaryWriter instance
        config: Nested configuration dictionary
    """
    # Define the important parameters to extract
    important_params = {
        "General": ["seed"],
        "Data": [
            ("data.classes", "Classes"),
            ("data.mean", "Normalization Mean"),
            ("data.std", "Normalization Std"),
            ("data.padding", "Padding"),
            ("data.dataset_path", "Dataset Path"),
        ],
        "Training": [
            ("training.batch_size", "Batch Size"),
            ("training.num_epochs", "Epochs"),
            ("training.device", "Training Device"),
            ("training.optimizer.type", "Optimizer"),
            ("training.optimizer.parameters.lr", "Learning Rate"),
        ],
        "Scheduler": [
            ("training.scheduler.parameters.mode", "Mode"),
            ("training.scheduler.parameters.factor", "Factor"),
            ("training.scheduler.parameters.patience", "Patience"),
        ],
        "Early Stopping": [
            ("training.early_stopping.enabled", "Enabled"),
            ("training.early_stopping.patience", "Patience"),
        ],
    }

    def get_nested_value(config, key_path):
        """Extract value from nested config using dot notation."""
        keys = key_path.split(".")
        value = config
        for k in keys:
            value = value[k]
        return value

    def format_value(value):
        """Format value for display, handling lists and other types."""
        if isinstance(value, list):
            return str(value).replace("[", "").replace("]", "")
        return str(value)

    # Create markdown table for each section
    for section, params in important_params.items():
        table_rows = ["|Parameter|Value|", "|-|-|"]

        for param in params:
            if isinstance(param, tuple):
                key_path, display_name = param
                try:
                    value = get_nested_value(config, key_path)
                    table_rows.append(f"|{display_name}|{format_value(value)}|")
                except (KeyError, TypeError):
                    continue
            else:
                try:
                    value = config[param]
                    table_rows.append(f"|{param}|{format_value(value)}|")
                except (KeyError, TypeError):
                    continue

        writer.add_text(f"Configuration/{section}", "\n".join(table_rows))

def plot_prediction_monotonicity(
    model, data, labels, size_labels, mean, std, device, experiment_dir, class_to_idx, num_samples=1000,
):
    """
    Plot prediction vs contrast to analyze monotonicity.
    
    Args:
        model: Trained model
        data: Pre-computed data array
        labels: Pre-computed labels array
        size_labels: Pre-computed size labels array
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        experiment_dir: Directory to save the plot
        num_samples: Number of samples to use for the plot (default: 1000)
    """
    # Create a dataset for plotting using pre-computed data
    plot_dataset = ParticleDatasetReg(
        data=data,
        labels=labels,
        size_labels=size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        
        transform=None,
        indices=None,
    )
    
    # Limit to num_samples if dataset is larger
    total_samples = len(plot_dataset)
    if total_samples > num_samples:
        # Randomly sample indices
        sample_indices = torch.randperm(total_samples)[:num_samples]
        plot_dataset = ParticleDatasetReg(
            data=data,
            labels=labels,
            size_labels=size_labels,
            mean=mean,
            std=std,
            class_to_idx=class_to_idx,
            
            transform=None,
            indices=sample_indices.tolist(),
        )
    
    # Create dataloader
    plot_dataloader = DataLoader(
        plot_dataset, batch_size=len(plot_dataset), shuffle=False
    )
    
    # Get predictions and calculate contrast
    with torch.no_grad():
        batch_data = next(iter(plot_dataloader))
        images = batch_data[0]  # Shape: (batch_size, channels, height, width)
        predictions = model(images.to(device)).cpu().numpy().flatten()
        
        # Calculate contrast for each image
        contrasts = contrast_from_middle_rows(images)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(contrasts, predictions, alpha=0.6, s=10, color='blue')
    
    # Add trend line (polynomial fit)
    z = np.polyfit(contrasts, predictions, 1)  # Linear fit
    p = np.poly1d(z)
    x_trend = np.linspace(contrasts.min(), contrasts.max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Linear fit (slope: {z[0]:.3f})')
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(contrasts, predictions)[0, 1]
   
    ax.set_xlabel('Contrast')
    ax.set_ylabel('Predicted Size [nm]')
    ax.set_title(f'Prediction vs Contrast Monotonicity\n(Correlation: {correlation:.3f}, N={len(contrasts)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient as text
    ax.text(0.05, 0.95, f'Pearson r = {correlation:.3f}', 
            transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(
        os.path.join(experiment_dir, "prediction_monotonicity.png"),
        format="png",
        dpi=300,
    )
    plt.close(fig)
    
    print(f"Monotonicity plot saved. Correlation: {correlation:.3f}")

def plot_loss_and_distribution(
    model,
    data,
    labels,
    size_labels,
    mean,
    std,
    device,
    loss_log,
    experiment_dir,
    class_to_idx,
    config,
):
    """
    Plot training loss history and compare ground truth vs predicted size distribution.

    Args:
        model: Trained model
        data: Pre-computed data array
        labels: Pre-computed labels array
        size_labels: Pre-computed size labels array
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        loss_log: List of loss values recorded during training
        experiment_dir: Directory to save the plot
    """
    # Create a dataset for plotting using pre-computed data
    plot_dataset = ParticleDatasetReg(
        data=data,
        labels=labels,
        size_labels=size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        transform=None,
        indices=None,
    )

    # Create dataloader
    plot_dataloader = DataLoader(
        plot_dataset, batch_size=len(plot_dataset), shuffle=False
    )
    
    # Get predictions and ground truth
    with torch.no_grad():
        batch_data = next(iter(plot_dataloader))
        images = batch_data[0]
        class_labels = batch_data[1]
        gt_size_labels = batch_data[2]  # Ground truth sizes
        predictions = model(images.to(device)).cpu().detach().numpy().flatten()

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Two subplots side by side

    # Left subplot: Loss logs with log scale
    min_loss = min(loss_log)
    loss_type = config["training"]["loss"]["type"]
    axes[0].plot(loss_log, color="blue", label=f"{loss_type} Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].axhline(y=min_loss, color="blue", linestyle="dashed")
    axes[0].text(
        0, min_loss, f"{min_loss:.4f}", color="blue", verticalalignment="bottom"
    )
    axes[0].set_ylabel(f"{loss_type} (log-scale)")
    axes[0].set_yscale("log")  # Apply log scale to y-axis
    axes[0].legend()
    axes[0].grid(True, which="both", linestyle="--", linewidth=0.5)

    # Right subplot: Histogram of predictions vs ground truth
    axes[1].hist(
        gt_size_labels.numpy(),
        bins=50,
        alpha=0.6,
        label="Pseudo Size labels",
        color="blue",
        density=True,
    )
    axes[1].hist(
        predictions,
        bins=50,
        color="red",
        label="Prediction",
        alpha=0.7,
        density=True,
    )
    axes[1].set_xlabel("Size [nm]")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()

    # Save the figure
    plt.savefig(
        os.path.join(experiment_dir, "distribution_with_loss.png"),
        format="png",
        dpi=300,
    )
    plt.close(fig)
    np.savez(
        os.path.join(experiment_dir, "results.npz"),
        loss_log=loss_log,
        predictions=predictions,
        ground_truth_sizes=gt_size_labels.numpy(),
        size_labels=size_labels
    )

def plot_validation_sample_images(
    model, data, labels, size_labels, mean, std,class_to_idx, device, experiment_dir, val_indices
):
    """
    Plot a grid of sample images from validation set with their predicted sizes.

    Args:
        model: Trained model
        data: Pre-computed data array
        labels: Pre-computed labels array
        size_labels: Pre-computed size labels array
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        experiment_dir: Directory to save the plot
        val_indices: List of validation indices to use for plotting
    """
    # Create a dataset for plotting using only validation indices
    plot_dataset = ParticleDatasetReg(
        data=data,
        labels=labels,
        size_labels=size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        
        transform=None,
        indices=val_indices,
    )

    # Create dataloader with a large batch size
    plot_dataloader = DataLoader(
        plot_dataset, batch_size=len(plot_dataset), shuffle=False
    )

    # Get predictions
    with torch.no_grad():
        batch_data = next(iter(plot_dataloader))
        imgs = batch_data[0]  # (batch_size, channels, height, width)
        ground_truth_sizes = batch_data[2]  # Ground truth sizes
        sizes = model(imgs.to(device)).cpu().squeeze()

        # Find min, max indices
        max_size, max_idx = sizes.max(dim=0)
        min_size, min_idx = sizes.min(dim=0)

        # Compute middle size (median)
        mid_size = sizes.median()
        mid_idx = (sizes - mid_size).abs().argmin()
        mid_idx = torch.tensor([mid_idx], dtype=torch.int64)

        # Create 9 intermediate values between min and max
        intermediate_sizes, intermediate_indices = [], []
        for fraction in torch.linspace(0, 1, steps=9):
            interp_size = min_size + fraction * (max_size - min_size)
            closest_idx = (sizes - interp_size).abs().argmin()
            intermediate_sizes.append(sizes[closest_idx])
            intermediate_indices.append(closest_idx)

        # Convert indices to tensor
        intermediate_indices = torch.tensor(
            intermediate_indices, dtype=torch.int64
        )

        # Get images and resize them
        resized_images = [
            torch.nn.functional.interpolate(
                imgs[idx][0:1].unsqueeze(0),
                size=(16, 32),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)[0]
            for idx in intermediate_indices
        ]

        # Get corresponding ground truth sizes
        gt_sizes = [ground_truth_sizes[idx] for idx in intermediate_indices]

    # Create 3x3 subplot
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))

    # Plot images
    for ax, img, pred_size, gt_size in zip(axes.flat, resized_images, intermediate_sizes, gt_sizes):
        ax.imshow(img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_text(f"prediction: {pred_size.item():.1f} | pseudo label: {gt_size.item():.1f}")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        os.path.join(experiment_dir, "validation_samples_size_predict.png"),
        format="png",
        dpi=300,
    )
    plt.close(fig)

def plot_sample_images(
    model, data, labels, size_labels, mean, std, device, class_to_idx,experiment_dir
):
    """
    Plot a grid of sample images with their predicted sizes.

    Args:
        model: Trained model
        data: Pre-computed data array
        labels: Pre-computed labels array
        size_labels: Pre-computed size labels array
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        experiment_dir: Directory to save the plot
    """
    # Create a dataset for plotting using pre-computed data
    plot_dataset = ParticleDatasetReg(
        data=data,
        labels=labels,
        size_labels=size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        
        transform=None,
        indices=None,
    )

    # Create dataloader with a large batch size
    plot_dataloader = DataLoader(
        plot_dataset, batch_size=len(plot_dataset), shuffle=False
    )

    # Get predictions
    with torch.no_grad():
        batch_data = next(iter(plot_dataloader))
        imgs = batch_data[0]  # (batch_size, channels, height, width)
        ground_truth_sizes = batch_data[2]  # Ground truth sizes
        sizes = model(imgs.to(device)).cpu().squeeze()

        # Find min, max indices
        max_size, max_idx = sizes.max(dim=0)
        min_size, min_idx = sizes.min(dim=0)

        # Compute middle size (median)
        mid_size = sizes.median()
        mid_idx = (sizes - mid_size).abs().argmin()
        mid_idx = torch.tensor([mid_idx], dtype=torch.int64)

        # Create 9 intermediate values between min and max
        intermediate_sizes, intermediate_indices = [], []
        for fraction in torch.linspace(0, 1, steps=9):
            interp_size = min_size + fraction * (max_size - min_size)
            closest_idx = (sizes - interp_size).abs().argmin()
            intermediate_sizes.append(sizes[closest_idx])
            intermediate_indices.append(closest_idx)

        # Convert indices to tensor
        intermediate_indices = torch.tensor(
            intermediate_indices, dtype=torch.int64
        )

        # Get images and resize them
        resized_images = [
            torch.nn.functional.interpolate(
                imgs[idx][0:1].unsqueeze(0),
                size=(16, 32),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)[0]
            for idx in intermediate_indices
        ]

        # Get corresponding ground truth sizes
        gt_sizes = [ground_truth_sizes[idx] for idx in intermediate_indices]

    # Create 3x3 subplot
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))

    # Plot images
    for ax, img, pred_size, gt_size in zip(axes.flat, resized_images, intermediate_sizes, gt_sizes):
        ax.imshow(img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_text(f"prediction: {pred_size.item():.1f} | pseudo label: {gt_size.item():.1f}")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        os.path.join(experiment_dir, "samples_size_predict.png"),
        format="png",
        dpi=300,
    )
    plt.close(fig)


def plot_per_class_performance(
    model, data, labels, size_labels, mean, std, device,class_to_idx, experiment_dir, classes, val_indices=None
):
    """
    Plot scatter plots of predictions vs ground truth for each class on validation dataset.
    
    Args:
        model: Trained model
        data: Pre-computed data array
        labels: Pre-computed labels array
        size_labels: Pre-computed size labels array
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        experiment_dir: Directory to save the plot
        classes: List of classes to include
        val_indices: List of validation indices to filter the dataset
    """
    # Create a dataset for plotting with validation indices
    plot_dataset = ParticleDatasetReg(
        data=data,
        labels=labels,
        size_labels=size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        
        transform=None,
        indices=val_indices,  # Use validation indices
    )

    # Create dataloader
    plot_dataloader = DataLoader(
        plot_dataset, batch_size=len(plot_dataset), shuffle=False
    )

    # Get predictions
    with torch.no_grad():
        batch_data = next(iter(plot_dataloader))
        images = batch_data[0]
        class_labels = batch_data[1]
        gt_size_labels = batch_data[2]  # Ground truth sizes
        predictions = model(images.to(device)).cpu().numpy().flatten()

    # Create subplots for each class
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 5))
    if len(classes) == 1:
        axes = [axes]

    for i, cls in enumerate(classes):
        # Filter data for this class
        class_mask = class_labels == cls
        class_gt = gt_size_labels[class_mask].numpy()
        class_pred = predictions[class_mask]
        
        # Scatter plot
        axes[i].scatter(class_gt, class_pred, alpha=0.6, s=10)
        
        # Perfect prediction line
        min_val = min(class_gt.min(), class_pred.min())
        max_val = max(class_gt.max(), class_pred.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect prediction')
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((class_gt - class_pred) ** 2))
        correlation = np.corrcoef(class_gt, class_pred)[0, 1]
        
        axes[i].set_xlabel(f'Pseudo Size Label [nm]')
        axes[i].set_ylabel(f'Predicted Size [nm]')
        axes[i].set_title(f'Class {cls} (Validation)\nRMSE: {rmse:.2f}, r: {correlation:.3f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(experiment_dir, "per_class_performance_validation.png"),
        format="png",
        dpi=300,
    )
    plt.close(fig)

def calculate_val_metrics(model, data, labels, size_labels, mean, std, device, class_to_idx, classes, val_indices):
    """
    Calculate accuracy metrics for each class based on size prediction ranges.
    
    Args:
        model: Trained model
        data: Pre-computed data array
        labels: Pre-computed labels array
        size_labels: Pre-computed size labels array
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        class_to_idx: Mapping from class names to indices
        classes: List of classes to include in the analysis
        val_indices: List of validation indices to filter the dataset
    
    Returns:
        dict: Dictionary containing class-wise accuracy metrics
    """
    # Create validation dataset
    val_dataset = ParticleDatasetReg(
        data=data,
        labels=labels,
        size_labels=size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        
        transform=None,
        indices=val_indices,  # Use validation indices
    )
    
    # Create dataloader
    val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    # Get predictions and ground truth
    all_class_labels = []
    all_gt_size_labels = []
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch_images, batch_class_labels, batch_gt_size_labels in tqdm(val_dataloader, desc="Calculating class accuracy metrics"):
            predictions = model(batch_images.to(device)).cpu().numpy().flatten()
            all_predictions.extend(predictions)
            all_class_labels.extend(batch_class_labels.numpy())
            all_gt_size_labels.extend(batch_gt_size_labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_class_labels = np.array(all_class_labels)
    all_gt_size_labels = np.array(all_gt_size_labels)
    
    # Calculate size ranges for each class
    class_size_ranges = {}
    mapped_classes_to_analyze = [class_to_idx[cls] for cls in classes]
    
    for mapped_cls_id in mapped_classes_to_analyze:
        # Get original class ID for reporting
        original_class_id = [k for k, v in class_to_idx.items() if v == mapped_cls_id][0]
        
        # Filter ground truth sizes for this class
        class_mask = all_class_labels == mapped_cls_id
        class_gt_sizes = all_gt_size_labels[class_mask]
        
        if len(class_gt_sizes) > 0:
            min_size = np.min(class_gt_sizes)
            max_size = np.max(class_gt_sizes)
            class_size_ranges[original_class_id] = {
                'min_size': float(min_size),
                'max_size': float(max_size),
                'range': float(max_size - min_size)
            }
        else:
            class_size_ranges[original_class_id] = {
                'min_size': None,
                'max_size': None,
                'range': None
            }
    
    # Calculate accuracy metrics for each class
    class_accuracy_metrics = {}
    
    for mapped_cls_id in mapped_classes_to_analyze:
        # Get original class ID
        original_class_id = [k for k, v in class_to_idx.items() if v == mapped_cls_id][0]
        
        # Filter data for this class
        class_mask = all_class_labels == mapped_cls_id
        class_predictions = all_predictions[class_mask]
        class_gt_sizes = all_gt_size_labels[class_mask]
        
        if len(class_predictions) == 0:
            class_accuracy_metrics[original_class_id] = {
                'total_samples': 0,
                'correct_predictions': 0,
                'wrong_predictions': 0,
                'accuracy_percentage': 0.0,
                'size_range': class_size_ranges[original_class_id]
            }
            continue
        
        # Get size range for this class
        if class_size_ranges[original_class_id]['min_size'] is not None:
            min_size = class_size_ranges[original_class_id]['min_size']
            max_size = class_size_ranges[original_class_id]['max_size']
            
            # Check which predictions fall within the correct range
            correct_mask = (class_predictions >= min_size) & (class_predictions <= max_size)
            correct_predictions = np.sum(correct_mask)
            wrong_predictions = len(class_predictions) - correct_predictions
            accuracy_percentage = (correct_predictions / len(class_predictions)) * 100
            
            class_accuracy_metrics[original_class_id] = {
                'total_samples': int(len(class_predictions)),
                'correct_predictions': int(correct_predictions),
                'wrong_predictions': int(wrong_predictions),
                'accuracy_percentage': float(accuracy_percentage),
                'size_range': class_size_ranges[original_class_id],
                'mean_prediction': float(np.mean(class_predictions)),
                'std_prediction': float(np.std(class_predictions)),
                'rmse': float(np.sqrt(np.mean((class_gt_sizes - class_predictions) ** 2)))
            }
        else:
            class_accuracy_metrics[original_class_id] = {
                'total_samples': int(len(class_predictions)),
                'correct_predictions': 0,
                'wrong_predictions': int(len(class_predictions)),
                'accuracy_percentage': 0.0,
                'size_range': class_size_ranges[original_class_id],
                'mean_prediction': float(np.mean(class_predictions)),
                'std_prediction': float(np.std(class_predictions)),
                'rmse': None
            }
    
    # Calculate overall accuracy across all classes
    total_samples = sum([metrics['total_samples'] for metrics in class_accuracy_metrics.values()])
    total_correct = sum([metrics['correct_predictions'] for metrics in class_accuracy_metrics.values()])
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    
    return {
        'class_wise_accuracy': class_accuracy_metrics,
        'overall_accuracy': {
            'total_samples': total_samples,
            'total_correct': total_correct,
            'total_wrong': total_samples - total_correct,
            'overall_accuracy_percentage': overall_accuracy
        }
    }

def calculate_test_metrics(model, data, labels, size_labels, mean, std, device, class_to_idx, classes, test_indices):
    """
    Calculate comprehensive test metrics including class-wise accuracy and overall performance.
    
    Args:
        model: Trained model
        data: Pre-computed data array
        labels: Pre-computed labels array
        size_labels: Pre-computed size labels array
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        class_to_idx: Mapping from class names to indices
        classes: List of classes to include in the analysis
        test_indices: List of test indices to filter the dataset
    
    Returns:
        dict: Dictionary containing test metrics
    """
    # Create test dataset
    test_dataset = ParticleDatasetReg(
        data=data,
        labels=labels,
        size_labels=size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        
        transform=None,
        indices=test_indices,  # Use test indices
    )
    
    # Create dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # Get predictions and ground truth
    all_class_labels = []
    all_gt_size_labels = []
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch_images, batch_class_labels, batch_gt_size_labels in tqdm(test_dataloader, desc="Calculating test metrics"):
            predictions = model(batch_images.to(device)).cpu().numpy().flatten()
            all_predictions.extend(predictions)
            all_class_labels.extend(batch_class_labels.numpy())
            all_gt_size_labels.extend(batch_gt_size_labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_class_labels = np.array(all_class_labels)
    all_gt_size_labels = np.array(all_gt_size_labels)
    
    # Calculate overall test metrics
    overall_rmse = np.sqrt(np.mean((all_gt_size_labels - all_predictions) ** 2))
    overall_mae = np.mean(np.abs(all_gt_size_labels - all_predictions))
    overall_correlation = np.corrcoef(all_gt_size_labels, all_predictions)[0, 1] if len(all_gt_size_labels) > 1 else np.nan
    
    # Calculate size ranges for each class
    class_size_ranges = {}
    mapped_classes_to_analyze = [class_to_idx[cls] for cls in classes]
    
    for mapped_cls_id in mapped_classes_to_analyze:
        # Get original class ID for reporting
        original_class_id = [k for k, v in class_to_idx.items() if v == mapped_cls_id][0]
        
        # Filter ground truth sizes for this class
        class_mask = all_class_labels == mapped_cls_id
        class_gt_sizes = all_gt_size_labels[class_mask]
        
        if len(class_gt_sizes) > 0:
            min_size = np.min(class_gt_sizes)
            max_size = np.max(class_gt_sizes)
            class_size_ranges[original_class_id] = {
                'min_size': float(min_size),
                'max_size': float(max_size),
                'range': float(max_size - min_size)
            }
        else:
            class_size_ranges[original_class_id] = {
                'min_size': None,
                'max_size': None,
                'range': None
            }
    
    # Calculate accuracy metrics for each class
    class_test_metrics = {}
    
    for mapped_cls_id in mapped_classes_to_analyze:
        # Get original class ID
        original_class_id = [k for k, v in class_to_idx.items() if v == mapped_cls_id][0]
        
        # Filter data for this class
        class_mask = all_class_labels == mapped_cls_id
        class_predictions = all_predictions[class_mask]
        class_gt_sizes = all_gt_size_labels[class_mask]
        
        if len(class_predictions) == 0:
            class_test_metrics[original_class_id] = {
                'total_samples': 0,
                'correct_predictions': 0,
                'wrong_predictions': 0,
                'accuracy_percentage': 0.0,
                'size_range': class_size_ranges[original_class_id],
                'rmse': None,
                'mae': None,
                'correlation': None
            }
            continue
        
        # Calculate class-specific metrics
        class_rmse = np.sqrt(np.mean((class_gt_sizes - class_predictions) ** 2))
        class_mae = np.mean(np.abs(class_gt_sizes - class_predictions))
        class_correlation = np.corrcoef(class_gt_sizes, class_predictions)[0, 1] if len(class_gt_sizes) > 1 else np.nan
        
        # Get size range for this class
        if class_size_ranges[original_class_id]['min_size'] is not None:
            min_size = class_size_ranges[original_class_id]['min_size']
            max_size = class_size_ranges[original_class_id]['max_size']
            
            # Check which predictions fall within the correct range
            correct_mask = (class_predictions >= min_size) & (class_predictions <= max_size)
            correct_predictions = np.sum(correct_mask)
            wrong_predictions = len(class_predictions) - correct_predictions
            accuracy_percentage = (correct_predictions / len(class_predictions)) * 100
            
            class_test_metrics[original_class_id] = {
                'total_samples': int(len(class_predictions)),
                'correct_predictions': int(correct_predictions),
                'wrong_predictions': int(wrong_predictions),
                'accuracy_percentage': float(accuracy_percentage),
                'size_range': class_size_ranges[original_class_id],
                'mean_prediction': float(np.mean(class_predictions)),
                'std_prediction': float(np.std(class_predictions)),
                'rmse': float(class_rmse),
                'mae': float(class_mae),
                'correlation': float(class_correlation) if not np.isnan(class_correlation) else None
            }
        else:
            class_test_metrics[original_class_id] = {
                'total_samples': int(len(class_predictions)),
                'correct_predictions': 0,
                'wrong_predictions': int(len(class_predictions)),
                'accuracy_percentage': 0.0,
                'size_range': class_size_ranges[original_class_id],
                'mean_prediction': float(np.mean(class_predictions)),
                'std_prediction': float(np.std(class_predictions)),
                'rmse': float(class_rmse),
                'mae': float(class_mae),
                'correlation': float(class_correlation) if not np.isnan(class_correlation) else None
            }
    
    # Calculate overall accuracy across all classes
    total_samples = sum([metrics['total_samples'] for metrics in class_test_metrics.values()])
    total_correct = sum([metrics['correct_predictions'] for metrics in class_test_metrics.values()])
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    
    return {
        'test_class_wise_metrics': class_test_metrics,
        'test_overall_metrics': {
            'total_samples': total_samples,
            'total_correct': total_correct,
            'total_wrong': total_samples - total_correct,
            'overall_accuracy_percentage': overall_accuracy,
            'overall_rmse': float(overall_rmse),
            'overall_mae': float(overall_mae),
            'overall_correlation': float(overall_correlation) if not np.isnan(overall_correlation) else None
        }
    }
def plot_test_sample_images(model, data, labels, size_labels, mean, std, class_to_idx, device, experiment_dir, test_indices):
    """
    Plot a grid of sample images from test set with their predicted sizes for 3D data.

    Args:
        model: Trained model
        data: Pre-computed data array
        labels: Pre-computed labels array
        size_labels: Pre-computed size labels array
        mean: Normalization mean
        std: Normalization std
        class_to_idx: Mapping from class names to indices
        device: Device to run inference on
        experiment_dir: Directory to save the plot
        test_indices: List of test indices to use for plotting
    """
    # Create a dataset for plotting using only test indices
    plot_dataset = ParticleDatasetReg(
        data=data,
        labels=labels,
        size_labels=size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        
        transform=None,
        indices=test_indices,  # Use test indices
    )

    # If test set is huge, might need to sample. Let's take a fixed number of samples.
    num_samples_to_plot = min(len(plot_dataset), 9 * 10) # Enough for a few rows of 9 images, but only display 9
    if len(plot_dataset) > num_samples_to_plot:
        # Sample indices from the current plot_dataset's scope (which is already test_indices)
        # We need to map these sampled indices back to the original `test_indices` list for the dataset.
        sampled_relative_indices = np.random.choice(len(plot_dataset), num_samples_to_plot, replace=False)
        
        # Create a new dataset instance with the sampled test indices
        plot_dataset = ParticleDatasetReg(
            data=data,
            labels=labels,
            size_labels=size_labels,
            mean=mean,
            std=std,
            class_to_idx=class_to_idx,
            
            transform=None,
            indices=[test_indices[i] for i in sampled_relative_indices],  # Use sampled test indices
        )

    # Create dataloader with a batch size equal to the number of samples to plot
    plot_dataloader = DataLoader(
        plot_dataset, batch_size=len(plot_dataset), shuffle=False
    )

    # Get predictions
    model.eval()
    with torch.no_grad():
        batch_data = next(iter(plot_dataloader))
        imgs = batch_data[0]  # (batch_size, channels, height, width)
        ground_truth_sizes = batch_data[2]  # Ground truth sizes
        
        if imgs.numel() == 0: # Handle empty batch case
            print("No test samples to plot.")
            plt.close()
            return

        sizes = model(imgs.to(device)).cpu().squeeze()

        # Find min, max indices and intermediate values based on predicted sizes for representative samples
        intermediate_sizes_val, intermediate_indices_val = [], []
        
        if sizes.ndim == 0: # Handle case of single prediction
            intermediate_sizes_val.append(sizes)
            intermediate_indices_val.append(torch.tensor(0))
        else:
            # Ensure we pick distinct samples if possible by sorting and then picking evenly spaced
            sorted_sizes, sorted_indices = torch.sort(sizes)
            num_points = 9 # Target 9 images for a 3x3 grid
            
            if len(sorted_sizes) > 0:
                indices_to_pick = np.linspace(0, len(sorted_sizes) - 1, num_points).astype(int)
                
                for idx_in_sorted in indices_to_pick:
                    original_idx_in_batch = sorted_indices[idx_in_sorted]
                    intermediate_sizes_val.append(sizes[original_idx_in_batch])
                    intermediate_indices_val.append(original_idx_in_batch)
            else: # No samples in batch
                print("No samples found in batch for plotting.")
                plt.close()
                return

        intermediate_indices = torch.tensor(
            intermediate_indices_val, dtype=torch.int64
        )
        intermediate_sizes = intermediate_sizes_val

        # Get images and resize them
        resized_images = [
            torch.nn.functional.interpolate(
                imgs[idx][0:1].unsqueeze(0), # Take the first channel (assuming C=1 for grayscale)
                size=(16, 32), # Target H, W for visualization
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)[0] # Remove batch dim and channel dim
            for idx in intermediate_indices
        ]

        # Get corresponding ground truth sizes
        gt_sizes = [ground_truth_sizes[idx] for idx in intermediate_indices]

    # Create 3x3 subplot
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))

    # Plot images (only plot up to 9, even if more intermediates were generated)
    for i, (ax, img, pred_size, gt_size) in enumerate(zip(axes.flat, resized_images, intermediate_sizes, gt_sizes)):
        if i >= 9: # Only plot first 9
            break
        ax.imshow(img.cpu().numpy(), cmap="gray") # Ensure it's numpy for imshow
        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_text(f"prediction: {pred_size.item():.1f} | pseudo label: {gt_size.item():.1f}")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        os.path.join(experiment_dir, "test_samples_size_predict.png"),
        format="png",
        dpi=300,
    )
    plt.close(fig)

def plot_test_per_class_performance(model, data, labels, size_labels, mean, std, device, class_to_idx, experiment_dir, classes, test_indices):
    """
    Plot scatter plots of predictions vs ground truth for each class on test dataset for 3D data.
    
    Args:
        model: Trained model
        data: Pre-computed data array
        labels: Pre-computed labels array
        size_labels: Pre-computed size labels array
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        class_to_idx: Dictionary for class mapping
        experiment_dir: Directory to save the plot
        classes: List of classes to include
        test_indices: List of test indices to filter the dataset
    """
    # Create a dataset for plotting with test indices
    plot_dataset = ParticleDatasetReg(
        data=data,
        labels=labels,
        size_labels=size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        
        transform=None,
        indices=test_indices,  # Use test indices
    )

    # Create dataloader
    plot_dataloader = DataLoader(
        plot_dataset, batch_size=1024, shuffle=False # Use reasonable batch size
    )

    all_class_labels = []
    all_gt_size_labels = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for batch_images, batch_class_labels, batch_gt_size_labels in tqdm(plot_dataloader, desc="Generating per-class test performance data"):
            predictions = model(batch_images.to(device)).cpu().numpy().flatten()
            all_predictions.extend(predictions)
            all_class_labels.extend(batch_class_labels.numpy())
            all_gt_size_labels.extend(batch_gt_size_labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_class_labels = np.array(all_class_labels)
    all_gt_size_labels = np.array(all_gt_size_labels)

    # Convert original class IDs to mapped class IDs for filtering
    mapped_classes_to_plot = [class_to_idx[cls] for cls in classes]

    # Create subplots for each class
    fig, axes = plt.subplots(1, len(mapped_classes_to_plot), figsize=(5 * len(mapped_classes_to_plot), 5))
    if len(mapped_classes_to_plot) == 1:
        axes = [axes] # Ensure axes is iterable for single class

    for i, mapped_cls_id in enumerate(mapped_classes_to_plot):
        # Filter data for this class using the mapped labels
        class_mask = all_class_labels == mapped_cls_id
        class_gt = all_gt_size_labels[class_mask]
        class_pred = all_predictions[class_mask]
        
        # Get original class ID for title
        original_class_id = [k for k, v in class_to_idx.items() if v == mapped_cls_id][0] # Reverse lookup

        if len(class_gt) == 0:
            axes[i].set_title(f'Class {original_class_id} (Test)\nNo samples')
            axes[i].set_xlabel(f'Pseudo Size Label [nm]')
            axes[i].set_ylabel(f'Predicted Size [nm]')
            axes[i].grid(True, alpha=0.3)
            continue

        # Scatter plot
        axes[i].scatter(class_gt, class_pred, alpha=0.6, s=10)
        
        # Perfect prediction line
        min_val = min(class_gt.min(), class_pred.min()) if len(class_gt) > 0 else 0
        max_val = max(class_gt.max(), class_pred.max()) if len(class_gt) > 0 else 1
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect prediction')
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((class_gt - class_pred) ** 2)) if len(class_gt) > 0 else np.nan
        correlation = np.corrcoef(class_gt, class_pred)[0, 1] if len(class_gt) > 1 else np.nan
        
        axes[i].set_xlabel(f'Pseudo Size Label [nm]')
        axes[i].set_ylabel(f'Predicted Size [nm]')
        axes[i].set_title(f'Class {original_class_id} (Test)\nRMSE: {rmse:.2f}, r: {correlation:.3f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(experiment_dir, "per_class_performance_test.png"),
        format="png",
        dpi=300,
    )
    plt.close(fig)
def print_test_metrics(test_metrics):
    """
    Print the test metrics in a readable format.
    Args:
        test_metrics (dict): Dictionary containing test metrics.
    """
    print("\nTest Results Summary:")
    print("-" * 50)
    print(f"Overall Test RMSE: {test_metrics['test_overall_metrics']['overall_rmse']:.2f}")
    print(f"Overall Test MAE: {test_metrics['test_overall_metrics']['overall_mae']:.2f}")
    print(f"Overall Test Correlation: {test_metrics['test_overall_metrics']['overall_correlation']:.3f}")
    print(f"Overall Test Accuracy: {test_metrics['test_overall_metrics']['overall_accuracy_percentage']:.2f}%")
    print(f"Total Test Samples: {test_metrics['test_overall_metrics']['total_samples']}")
    print()
    
    for class_id, metrics_data in test_metrics['test_class_wise_metrics'].items():
        print(f"Class {class_id} Test Results:")
        print(f"  Total samples: {metrics_data['total_samples']}")
        print(f"  RMSE: {metrics_data['rmse']:.2f}" if metrics_data['rmse'] is not None else "  RMSE: N/A")
        print(f"  MAE: {metrics_data['mae']:.2f}" if metrics_data['mae'] is not None else "  MAE: N/A")
        print(f"  Correlation: {metrics_data['correlation']:.3f}" if metrics_data['correlation'] is not None else "  Correlation: N/A")
        print(f"  Accuracy: {metrics_data['accuracy_percentage']:.2f}%")
        print()

def main(args):
    # Load configuration
    config = load_config(args.config)
    set_random_seed(config["seed"])

    # Set up experiment directory and tensorboard writer
    experiment_folder_name = f"ResNet18_Regression_zx_{getdatetime()}"
    experiment_folder_name = experiment_folder_name[
        :100
    ]  # Limit folder name length
    experiment_dir = os.path.join(
        config["logging"]["tensorboard"]["log_dir"], experiment_folder_name
    )
    writer = SummaryWriter(log_dir=experiment_dir)
    write_config_to_tensorboard(writer, config)

    # Set device
    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Generate global size labels once
    print("Generating global size labels...")
    global_data, global_labels, global_size_labels, class_to_idx, mean, std = generate_global_size_labels(
        h5_path=config["data"]["dataset_path"],
        classes=config["data"]["classes"],
    )
    print(f"Generated size labels for {len(global_data)} samples")

    # Extract class labels for stratified split
    class_labels = global_labels.tolist()

    # Stratified split
    # Replace the existing splitter code with:
    # First split: Train + Val (90%) vs Test (10%)
    splitter_train_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=config["seed"])
    train_val_idx, test_idx = next(splitter_train_val_test.split(np.zeros(len(class_labels)), class_labels))

    # Second split: Train (72% overall) vs Val (18% overall) 
    train_val_labels = [class_labels[i] for i in train_val_idx]
    splitter_train_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config["seed"])
    train_relative_idx, val_relative_idx = next(splitter_train_val.split(np.zeros(len(train_val_labels)), train_val_labels))

    # Convert relative indices back to absolute indices
    train_idx = train_val_idx[train_relative_idx]
    val_idx = train_val_idx[val_relative_idx]
    
    print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

    # Define augmentation
    augmentation = v2.Compose(
        [
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomHorizontalFlip(p=0.5),
        ]
    )

    # Create datasets using pre-computed global data
    train_dataset = ParticleDatasetReg(
        data=global_data,
        labels=global_labels,
        size_labels=global_size_labels,
        class_to_idx=class_to_idx,
        mean=mean,
        std=std,
        transform=augmentation,
        indices=train_idx.tolist(),
    )

    val_dataset = ParticleDatasetReg(
        data=global_data,
        labels=global_labels,
        size_labels=global_size_labels,
        class_to_idx=class_to_idx,
        mean=mean,
        std=std,
        transform=None,
        indices=val_idx.tolist(),
    )

    train_loader = create_dataloaders(train_dataset, config["training"]["batch_size"], config["data"]["num_workers"])
    val_loader = create_dataloaders(val_dataset, config["training"]["batch_size"], config["data"]["num_workers"])

    # Initialize model
    model = ResNet18(num_classes=1, in_channels=1)  # For regression, output is a single value
    model = model.to(device)

    # Initialize trainer
    trainer = ParticleSizeTrainer(
        model=model,
        device=device,
        config=config["training"],
        experiment_dir=experiment_dir,
        writer=writer,
        verbose=True,
    )

    # Train the model
    loss_log = trainer.train(train_loader, val_loader, num_epochs=config["training"]["num_epochs"])

    # Save metrics
    metrics = {
        "mse_history": loss_log,
        "final_mse": loss_log[-1] if loss_log else None,
        "num_epochs_trained": len(loss_log),
    }
    # Calculate validation and test metrics
    val_metrics = calculate_val_metrics(
        model=model,
        data=global_data,
        labels=global_labels,
        size_labels=global_size_labels,
        mean=mean,
        std=std,
        device=device,
        class_to_idx=class_to_idx,
        classes=config["data"]["classes"],
        val_indices=val_idx.tolist()
    )

    test_metrics = calculate_test_metrics(
        model=model,
        data=global_data,
        labels=global_labels,
        size_labels=global_size_labels,
        mean=mean,
        std=std,
        device=device,
        class_to_idx=class_to_idx,
        classes=config["data"]["classes"],
        test_indices=test_idx.tolist()
    )

    # Update metrics dictionary
    metrics["val_metrics"] = val_metrics
    metrics["test_metrics"] = test_metrics

    # Print test results
    print_test_metrics(test_metrics)
    # Save metrics to JSON
    save_metrics_to_json(metrics, experiment_dir)
    # Add test plotting calls
    plot_test_per_class_performance(
        model=model,
        data=global_data,
        labels=global_labels,
        size_labels=global_size_labels,
        mean=mean,
        std=std,
        device=device,
        class_to_idx=class_to_idx,
        experiment_dir=experiment_dir,
        classes=config["data"]["classes"],
        test_indices=test_idx.tolist()
    )

    plot_test_sample_images(
        model=model,
        data=global_data,
        labels=global_labels,
        size_labels=global_size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        device=device,
        experiment_dir=experiment_dir,
        test_indices=test_idx.tolist()
    )

    # Plot loss and distribution comparison
    plot_loss_and_distribution(
        model=model,
        data=global_data,
        labels=global_labels,
        size_labels=global_size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        device=device,
        loss_log=loss_log,
        experiment_dir=experiment_dir,
        config=config,
    )

    # Plot sample images with predicted sizes
    plot_sample_images(
        model=model,
        data=global_data,
        labels=global_labels,
        size_labels=global_size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        device=device,
        experiment_dir=experiment_dir,
    )
    
    # Plot prediction monotonicity
    plot_prediction_monotonicity(
        model=model,
        data=global_data,
        labels=global_labels,
        size_labels=global_size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        device=device,
        experiment_dir=experiment_dir,
        num_samples=1000  
    )
    
    # Plot per-class performance
    plot_per_class_performance(
        model=model,
        data=global_data,
        labels=global_labels,
        size_labels=global_size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        device=device,
        experiment_dir=experiment_dir,
        classes=config["data"]["classes"],
        val_indices=val_idx.tolist(),  # Pass validation indices
    )
    
    # Plot validation sample images
    plot_validation_sample_images(
        model=model,
        data=global_data,
        labels=global_labels,
        size_labels=global_size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        device=device,
        experiment_dir=experiment_dir,
        val_indices=val_idx.tolist(),
    )
    print(f"Training completed. Model and plots saved to {experiment_dir}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)