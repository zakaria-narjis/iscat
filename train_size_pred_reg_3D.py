# train_size_pred_reg_3D.py
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.trainers.trainersizereg import ParticleSizeTrainer
from src.data_processing.size_dataset_3D import ParticleDatasetReg, generate_global_size_labels
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
    parser = argparse.ArgumentParser(description="iScat Size Prediction Regression 3D")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/size_pred_config.yaml",
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
    model, h5_path, labels, size_labels, contrasts, mean, std, device, experiment_dir, class_to_idx, original_h5_indices, num_samples=1000, plot_indices=None
):
    """
    Plot prediction vs contrast to analyze monotonicity for 3D data.
    
    Args:
        model: Trained model
        h5_path: Path to HDF5 file
        labels: Pre-computed labels array (filtered and mapped)
        size_labels: Pre-computed size labels array
        contrasts: Pre-computed contrast scores array (for filtered data)
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        experiment_dir: Directory to save the plot
        class_to_idx: Dictionary for class mapping
        original_h5_indices: Array of original HDF5 indices corresponding to labels/size_labels/contrasts
        num_samples: Number of samples to use for the plot (default: 1000)
        plot_indices: Optional list of specific indices (relative to labels/size_labels) to use for plotting.
    """
    
    # Determine indices for plotting
    if plot_indices is None:
        total_samples = len(labels)
        if total_samples > num_samples:
            # Randomly sample indices from the entire filtered dataset
            sample_indices = np.random.choice(total_samples, num_samples, replace=False)
        else:
            sample_indices = np.arange(total_samples)
    else:
        sample_indices = np.array(plot_indices)
        if len(sample_indices) > num_samples: # Still respect num_samples if plot_indices are too many
             sample_indices = np.random.choice(sample_indices, num_samples, replace=False)


    # Create a dataset for plotting using pre-computed data and HDF5 path
    plot_dataset = ParticleDatasetReg(
        h5_path=h5_path,
        labels=labels, # Pass full precomputed labels
        size_labels=size_labels, # Pass full precomputed size labels
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        padding=False,
        transform=None,
        indices=sample_indices.tolist(), # Indices to subset the labels/size_labels
        original_h5_indices=original_h5_indices # Actual HDF5 indices
    )
    
    # Create dataloader - batch_size should be reasonable for memory, or simply the len of dataset if small
    plot_dataloader = DataLoader(
        plot_dataset, batch_size=min(len(plot_dataset), 1024), shuffle=False # Use a reasonable batch size
    )
    
    all_predictions = []
    
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        for batch_images, _, _ in tqdm(plot_dataloader, desc="Generating monotonicity plot data"):
            predictions = model(batch_images.to(device)).cpu().numpy().flatten()
            all_predictions.extend(predictions)
    
    all_predictions = np.array(all_predictions)
    
    # Get corresponding contrasts for the sampled images
    sampled_contrasts = contrasts[sample_indices]

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(sampled_contrasts, all_predictions, alpha=0.6, s=10, color='blue')
    
    # Add trend line (polynomial fit)
    # Ensure there are enough data points for polyfit
    if len(sampled_contrasts) > 1:
        z = np.polyfit(sampled_contrasts, all_predictions, 1)  # Linear fit
        p = np.poly1d(z)
        x_trend = np.linspace(sampled_contrasts.min(), sampled_contrasts.max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Linear fit (slope: {z[0]:.3f})')
        correlation = np.corrcoef(sampled_contrasts, all_predictions)[0, 1]
    else:
        correlation = np.nan # Cannot compute correlation with less than 2 points
        print("Not enough samples to compute correlation for monotonicity plot.")
    
    ax.set_xlabel('Contrast')
    ax.set_ylabel('Predicted Size [nm]')
    ax.set_title(f'Prediction vs Contrast Monotonicity\n(Correlation: {correlation:.3f}, N={len(sampled_contrasts)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient as text
    if not np.isnan(correlation):
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
    model, h5_path, labels, size_labels, mean, std, device, loss_log, experiment_dir, class_to_idx, original_h5_indices
):
    """
    Plot training loss history and compare ground truth vs predicted size distribution for 3D data.

    Args:
        model: Trained model
        h5_path: Path to HDF5 file
        labels: Pre-computed labels array (filtered and mapped)
        size_labels: Pre-computed size labels array
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        loss_log: List of loss values recorded during training
        experiment_dir: Directory to save the plot
        class_to_idx: Dictionary for class mapping
        original_h5_indices: Array of original HDF5 indices corresponding to labels/size_labels
    """
    # Create a dataset for plotting using pre-computed data and HDF5 path
    plot_dataset = ParticleDatasetReg(
        h5_path=h5_path,
        labels=labels,
        size_labels=size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        padding=True, # Original had padding=True, keep it.
        transform=None,
        indices=None, # Use all filtered data for distribution plot
        original_h5_indices=original_h5_indices
    )

    # Create dataloader
    plot_dataloader = DataLoader(
        plot_dataset, batch_size=1024, shuffle=False # Use a reasonable batch size
    )
    
    all_gt_size_labels = []
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch_images, _, batch_gt_size_labels in tqdm(plot_dataloader, desc="Generating distribution plot data"):
            predictions = model(batch_images.to(device)).cpu().numpy().flatten()
            all_predictions.extend(predictions)
            all_gt_size_labels.extend(batch_gt_size_labels.numpy())

    all_predictions = np.array(all_predictions)
    all_gt_size_labels = np.array(all_gt_size_labels)

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left subplot: Loss logs with log scale
    min_loss = min(loss_log)
    axes[0].plot(loss_log, color="blue", label="MSE")
    axes[0].set_xlabel("Epochs")
    axes[0].axhline(y=min_loss, color="blue", linestyle="dashed")
    axes[0].text(
        0, min_loss, f"{min_loss:.4f}", color="blue", verticalalignment="bottom"
    )
    axes[0].set_ylabel("MSE (log-scale)")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, which="both", linestyle="--", linewidth=0.5)

    # Right subplot: Histogram of predictions vs ground truth
    axes[1].hist(
        all_gt_size_labels,
        bins=50,
        alpha=0.6,
        label="Ground Truth",
        color="blue",
        density=True,
    )
    axes[1].hist(
        all_predictions,
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

def plot_validation_sample_images(
    model, h5_path, labels, size_labels, mean, std, class_to_idx, device, experiment_dir, val_indices, original_h5_indices
):
    """
    Plot a grid of sample images from validation set with their predicted sizes for 3D data.

    Args:
        model: Trained model
        h5_path: Path to HDF5 file
        labels: Pre-computed labels array (filtered and mapped)
        size_labels: Pre-computed size labels array
        mean: Normalization mean
        std: Normalization std
        class_to_idx: Dictionary for class mapping
        device: Device to run inference on
        experiment_dir: Directory to save the plot
        val_indices: List of validation indices (relative to labels/size_labels) to use for plotting
        original_h5_indices: Array of original HDF5 indices corresponding to labels/size_labels
    """
    # Create a dataset for plotting using only validation indices
    plot_dataset = ParticleDatasetReg(
        h5_path=h5_path,
        labels=labels,
        size_labels=size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        padding=False,
        transform=None,
        indices=val_indices,  # Use validation indices
        original_h5_indices=original_h5_indices
    )

    # If validation set is huge, might need to sample. Let's take a fixed number of samples.
    num_samples_to_plot = min(len(plot_dataset), 9 * 10) # Enough for a few rows of 9 images, but only display 9
    if len(plot_dataset) > num_samples_to_plot:
        # Sample indices from the current plot_dataset's scope (which is already val_indices)
        # We need to map these sampled indices back to the original `val_indices` list for the dataset.
        sampled_relative_indices = np.random.choice(len(plot_dataset), num_samples_to_plot, replace=False)
        
        # Create a new dataset instance with the sampled validation indices
        plot_dataset = ParticleDatasetReg(
            h5_path=h5_path,
            labels=labels,
            size_labels=size_labels,
            mean=mean,
            std=std,
            class_to_idx=class_to_idx,
            padding=False,
            transform=None,
            indices=np.array(val_indices)[sampled_relative_indices].tolist(), # Correctly map sampled indices back to actual val_indices list
            original_h5_indices=original_h5_indices
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
            print("No validation samples to plot.")
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
        ax.title.set_text(f"pred: {pred_size.item():.1f} | gt: {gt_size.item():.1f}")

    # Add a main title to distinguish from training samples
    fig.suptitle("Validation Sample Images with Size Predictions", fontsize=16, y=0.98)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        os.path.join(experiment_dir, "validation_samples_size_predict.png"),
        format="png",
        dpi=300,
    )
    plt.close(fig)


def plot_sample_images(
    model, h5_path, labels, size_labels, mean, std, device, class_to_idx, experiment_dir, original_h5_indices
):
    """
    Plot a grid of sample images with their predicted sizes for 3D data.

    Args:
        model: Trained model
        h5_path: Path to HDF5 file
        labels: Pre-computed labels array (filtered and mapped)
        size_labels: Pre-computed size labels array
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        class_to_idx: Dictionary for class mapping
        experiment_dir: Directory to save the plot
        original_h5_indices: Array of original HDF5 indices corresponding to labels/size_labels
    """
    # Create a dataset for plotting using pre-computed data and HDF5 path
    plot_dataset = ParticleDatasetReg(
        h5_path=h5_path,
        labels=labels,
        size_labels=size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        padding=False,
        transform=None,
        indices=None, # Use all filtered data, then sample internally
        original_h5_indices=original_h5_indices
    )
    
    num_samples_to_plot = min(len(plot_dataset), 9 * 10) # A reasonable number to sample from total
    if len(plot_dataset) > num_samples_to_plot:
        sample_indices_for_plot = np.random.choice(len(plot_dataset), num_samples_to_plot, replace=False)
        plot_dataset = ParticleDatasetReg(
            h5_path=h5_path,
            labels=labels,
            size_labels=size_labels,
            mean=mean,
            std=std,
            class_to_idx=class_to_idx,
            padding=False,
            transform=None,
            indices=sample_indices_for_plot.tolist(), # Sample from the overall dataset
            original_h5_indices=original_h5_indices
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
            print("No samples to plot.")
            plt.close()
            return
            
        sizes = model(imgs.to(device)).cpu().squeeze()

        # Find min, max indices and intermediate values based on predicted sizes for representative samples
        intermediate_sizes_val, intermediate_indices_val = [], []
        
        if sizes.ndim == 0: # Handle case of single prediction
            intermediate_sizes_val.append(sizes)
            intermediate_indices_val.append(torch.tensor(0))
        else:
            sorted_sizes, sorted_indices = torch.sort(sizes)
            num_points = 9
            
            if len(sorted_sizes) > 0:
                indices_to_pick = np.linspace(0, len(sorted_sizes) - 1, num_points).astype(int)
                
                for idx_in_sorted in indices_to_pick:
                    original_idx_in_batch = sorted_indices[idx_in_sorted]
                    intermediate_sizes_val.append(sizes[original_idx_in_batch])
                    intermediate_indices_val.append(original_idx_in_batch)
            else:
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
    for i, (ax, img, pred_size, gt_size) in enumerate(zip(axes.flat, resized_images, intermediate_sizes, gt_sizes)):
        if i >= 9:
            break
        ax.imshow(img.cpu().numpy(), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_text(f"pred: {pred_size.item():.1f} | gt: {gt_size.item():.1f}")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        os.path.join(experiment_dir, "samples_size_predict.png"),
        format="png",
        dpi=300,
    )
    plt.close(fig)


def plot_per_class_performance(
    model, h5_path, labels, size_labels, mean, std, device, class_to_idx, experiment_dir, classes, val_indices, original_h5_indices
):
    """
    Plot scatter plots of predictions vs ground truth for each class on validation dataset for 3D data.
    
    Args:
        model: Trained model
        h5_path: Path to HDF5 file
        labels: Pre-computed labels array (filtered and mapped)
        size_labels: Pre-computed size labels array
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        class_to_idx: Dictionary for class mapping
        experiment_dir: Directory to save the plot
        classes: List of classes (original IDs) to include
        val_indices: List of validation indices (relative to labels/size_labels) to filter the dataset
        original_h5_indices: Array of original HDF5 indices corresponding to labels/size_labels
    """
    # Create a dataset for plotting with validation indices
    plot_dataset = ParticleDatasetReg(
        h5_path=h5_path,
        labels=labels,
        size_labels=size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        padding=False,
        transform=None,
        indices=val_indices,  # Use validation indices
        original_h5_indices=original_h5_indices
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
        for batch_images, batch_class_labels, batch_gt_size_labels in tqdm(plot_dataloader, desc="Generating per-class performance data"):
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
            axes[i].set_title(f'Class {original_class_id} (Validation)\nNo samples')
            axes[i].set_xlabel(f'Ground Truth Size [nm]')
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
        
        axes[i].set_xlabel(f'Ground Truth Size [nm]')
        axes[i].set_ylabel(f'Predicted Size [nm]')
        axes[i].set_title(f'Class {original_class_id} (Validation)\nRMSE: {rmse:.2f}, r: {correlation:.3f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(experiment_dir, "per_class_performance_validation.png"),
        format="png",
        dpi=300,
    )
    plt.close(fig)


def main(args):
    # Load configuration
    config = load_config(args.config)
    set_random_seed(config["seed"])

    # Set up experiment directory and tensorboard writer
    experiment_folder_name = f"ResNet18_Regression_3D_{getdatetime()}" # Added _3D
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

    # Generate global size labels, original HDF5 indices, and contrasts once
    print("Generating global size labels, original HDF5 indices, and contrasts...")
    all_mapped_labels, all_size_labels, class_to_idx, mean, std, all_original_h5_indices, all_filtered_contrast_scores = generate_global_size_labels(
        h5_path=config["data"]["dataset_path"],
        classes=config["data"]["classes"],
        mean=config["data"]["mean"],
        std=config["data"]["std"]
    )
    print(f"Generated data for {len(all_mapped_labels)} samples")

    # Extract class labels for stratified split (use the mapped labels)
    class_labels_for_split = all_mapped_labels.tolist()

    # Stratified split - indices are relative to the filtered dataset (all_mapped_labels, etc.)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config["seed"])
    train_idx, val_idx = next(splitter.split(np.zeros(len(class_labels_for_split)), class_labels_for_split))
    
    print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

    # Define augmentation
    augmentation = v2.Compose(
        [
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomHorizontalFlip(p=0.5),
        ]
    )

    # Create datasets using pre-computed global labels/size_labels and HDF5 path for data access
    train_dataset = ParticleDatasetReg(
        h5_path=config["data"]["dataset_path"],
        labels=all_mapped_labels,
        size_labels=all_size_labels,
        class_to_idx=class_to_idx,
        mean=mean,
        std=std,
        padding=config["data"]["padding"],
        transform=augmentation,
        indices=train_idx.tolist(), # Indices into the all_mapped_labels/all_size_labels
        original_h5_indices=all_original_h5_indices # The actual HDF5 indices
    )

    val_dataset = ParticleDatasetReg(
        h5_path=config["data"]["dataset_path"],
        labels=all_mapped_labels,
        size_labels=all_size_labels,
        class_to_idx=class_to_idx,
        mean=mean,
        std=std,
        padding=config["data"]["padding"],
        transform=None,
        indices=val_idx.tolist(),
        original_h5_indices=all_original_h5_indices
    )

    train_loader = create_dataloaders(train_dataset, config["training"]["batch_size"], config["data"]["num_workers"])
    val_loader = create_dataloaders(val_dataset, config["training"]["batch_size"], config["data"]["num_workers"])

    # Initialize model
    model = ResNet18(num_classes=1, in_channels=201)  # For regression, output is a single value
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
    save_metrics_to_json(metrics, experiment_dir)

    # Plot loss and distribution comparison
    plot_loss_and_distribution( 
        model=model,
        h5_path=config["data"]["dataset_path"],
        labels=all_mapped_labels,
        size_labels=all_size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        device=device,
        loss_log=loss_log,
        experiment_dir=experiment_dir,
        original_h5_indices=all_original_h5_indices 
    )

    # Plot sample images with predicted sizes (from entire dataset sample)
    plot_sample_images(
        model=model,
        h5_path=config["data"]["dataset_path"],
        labels=all_mapped_labels,
        size_labels=all_size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        device=device,
        experiment_dir=experiment_dir,
        original_h5_indices=all_original_h5_indices
    )
    
    # Plot prediction monotonicity
    plot_prediction_monotonicity(
        model=model,
        h5_path=config["data"]["dataset_path"],
        labels=all_mapped_labels,
        size_labels=all_size_labels,
        contrasts=all_filtered_contrast_scores, 
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        device=device,
        experiment_dir=experiment_dir,
        num_samples=1000,
        original_h5_indices=all_original_h5_indices
    )
    
    # Plot per-class performance (using validation set)
    plot_per_class_performance(
        model=model,
        h5_path=config["data"]["dataset_path"],
        labels=all_mapped_labels,
        size_labels=all_size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        device=device,
        experiment_dir=experiment_dir,
        classes=config["data"]["classes"],
        val_indices=val_idx.tolist(), 
        original_h5_indices=all_original_h5_indices
    )
    
    # Plot validation sample images (explicitly from validation set)
    plot_validation_sample_images(
        model=model,
        h5_path=config["data"]["dataset_path"],
        labels=all_mapped_labels,
        size_labels=all_size_labels,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
        device=device,
        experiment_dir=experiment_dir,
        val_indices=val_idx.tolist(),
        original_h5_indices=all_original_h5_indices
    )
    print(f"Training completed. Model and plots saved to {experiment_dir}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)