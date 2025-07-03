import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.trainers.trainersizereg import ParticleSizeTrainer
from src.data_processing.size_dataset import ParticleDatasetReg  
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
import matplotlib
import logging

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
        default="configs/size_pred_config.yaml",
        help="Path to the configuration file",
    )
    return parser


def create_dataloaders(train_dataset, batch_size, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    return train_loader


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
    model, dataset_path, classes, mean, std, device, experiment_dir, num_samples=1000
):
    """
    Plot prediction vs contrast to analyze monotonicity.
    
    Args:
        model: Trained model
        dataset_path: Path to the HDF5 dataset
        classes: List of classes to include
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        experiment_dir: Directory to save the plot
        num_samples: Number of samples to use for the plot (default: 1000)
    """
    # Create a dataset for plotting
    plot_dataset = ParticleDatasetReg(
        h5_path=dataset_path,
        classes=classes,
        mean=mean,
        std=std,
        padding=False,
        transform=None,
        indices=None,
    )
    
    # Limit to num_samples if dataset is larger
    total_samples = len(plot_dataset)
    if total_samples > num_samples:
        # Randomly sample indices
        sample_indices = torch.randperm(total_samples)[:num_samples]
        plot_dataset = ParticleDatasetReg(
            h5_path=dataset_path,
            classes=classes,
            mean=mean,
            std=std,
            padding=False,
            transform=None,
            indices=sample_indices.tolist(),
        )
    
    # Create dataloader
    plot_dataloader = DataLoader(
        plot_dataset, batch_size=len(plot_dataset), shuffle=False
    )
    
    # Get predictions and calculate contrast
    with torch.no_grad():
        data = next(iter(plot_dataloader))
        images = data[0]  # Shape: (batch_size, channels, height, width)
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
    dataset_path,
    classes,
    mean,
    std,
    device,
    loss_log,
    experiment_dir,
):
    """
    Plot training loss history and compare ground truth vs predicted size distribution.

    Args:
        model: Trained model
        dataset_path: Path to the HDF5 dataset
        classes: List of classes to include
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        loss_log: List of loss values recorded during training
        experiment_dir: Directory to save the plot
    """
    # Create a dataset for plotting
    plot_dataset = ParticleDatasetReg(
        h5_path=dataset_path,
        classes=classes,
        mean=mean,
        std=std,
        padding=True,
        transform=None,
        indices=None,
    )

    # Create dataloader
    plot_dataloader = DataLoader(
        plot_dataset, batch_size=len(plot_dataset), shuffle=False
    )
    
    # Get predictions and ground truth
    with torch.no_grad():
        data = next(iter(plot_dataloader))
        images = data[0]
        class_labels = data[1]
        size_labels = data[2]  # Ground truth sizes
        predictions = model(images.to(device)).cpu().detach().numpy().flatten()

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Two subplots side by side

    # Left subplot: Loss logs with log scale
    min_loss = min(loss_log)
    axes[0].plot(loss_log, color="blue", label="RMSE")
    axes[0].set_xlabel("Epochs")
    axes[0].axhline(y=min_loss, color="blue", linestyle="dashed")
    axes[0].text(
        0, min_loss, f"{min_loss:.4f}", color="blue", verticalalignment="bottom"
    )
    axes[0].set_ylabel("RMSE (log-scale)")
    axes[0].set_yscale("log")  # Apply log scale to y-axis
    axes[0].legend()
    axes[0].grid(True, which="both", linestyle="--", linewidth=0.5)

    # Right subplot: Histogram of predictions vs ground truth
    axes[1].hist(
        size_labels.numpy(),
        bins=50,
        alpha=0.6,
        label="Ground Truth",
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


def plot_sample_images(
    model, dataset_path, classes, mean, std, device, experiment_dir
):
    """
    Plot a grid of sample images with their predicted sizes.

    Args:
        model: Trained model
        dataset_path: Path to the HDF5 dataset
        classes: List of classes to include
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        experiment_dir: Directory to save the plot
    """
    # Create a dataset for plotting
    plot_dataset = ParticleDatasetReg(
        h5_path=dataset_path,
        classes=classes,
        mean=mean,
        std=std,
        padding=False,
        transform=None,
        indices=None,
    )

    # Create dataloader with a large batch size
    plot_dataloader = DataLoader(
        plot_dataset, batch_size=len(plot_dataset), shuffle=False
    )

    # Get predictions
    with torch.no_grad():
        data = next(iter(plot_dataloader))
        imgs = data[0]  # (batch_size, channels, height, width)
        ground_truth_sizes = data[2]  # Ground truth sizes
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
    model, dataset_path, classes, mean, std, device, experiment_dir
):
    """
    Plot scatter plots of predictions vs ground truth for each class.
    
    Args:
        model: Trained model
        dataset_path: Path to the HDF5 dataset
        classes: List of classes to include
        mean: Normalization mean
        std: Normalization std
        device: Device to run inference on
        experiment_dir: Directory to save the plot
    """
    # Create a dataset for plotting
    plot_dataset = ParticleDatasetReg(
        h5_path=dataset_path,
        classes=classes,
        mean=mean,
        std=std,
        padding=False,
        transform=None,
        indices=None,
    )

    # Create dataloader
    plot_dataloader = DataLoader(
        plot_dataset, batch_size=len(plot_dataset), shuffle=False
    )

    # Get predictions
    with torch.no_grad():
        data = next(iter(plot_dataloader))
        images = data[0]
        class_labels = data[1]
        size_labels = data[2]  # Ground truth sizes
        predictions = model(images.to(device)).cpu().numpy().flatten()

    # Create subplots for each class
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 5))
    if len(classes) == 1:
        axes = [axes]

    for i, cls in enumerate(classes):
        # Filter data for this class
        class_mask = class_labels == cls
        class_gt = size_labels[class_mask].numpy()
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
        
        axes[i].set_xlabel(f'Ground Truth Size [nm]')
        axes[i].set_ylabel(f'Predicted Size [nm]')
        axes[i].set_title(f'Class {cls}\nRMSE: {rmse:.2f}, r: {correlation:.3f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(experiment_dir, "per_class_performance.png"),
        format="png",
        dpi=300,
    )
    plt.close(fig)


def main(args):
    # Load configuration
    config = load_config(args.config)
    set_random_seed(config["seed"])

    # Set up experiment directory and tensorboard writer
    experiment_folder_name = f"ResNet18_Regression_{getdatetime()}"
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

    augmentation = v2.Compose(
        [
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomHorizontalFlip(p=0.5),
        ]
    )
    
    # Create dataset
    train_dataset = ParticleDatasetReg(
        h5_path=config["data"]["dataset_path"],
        classes=config["data"]["classes"],
        transform=augmentation,
        mean=config["data"]["mean"],
        std=config["data"]["std"],
        padding=config["data"]["padding"],
        indices=None,
    )

    # Create dataloader
    train_loader = create_dataloaders(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # Initialize model
    model = ResNet18(num_classes=1)  # For regression, output is a single value
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
    loss_log = trainer.train(
        train_loader, num_epochs=config["training"]["num_epochs"]
    )

    # Save metrics
    metrics = {
        "rmse_history": loss_log,
        "final_rmse": loss_log[-1] if loss_log else None,
        "num_epochs_trained": len(loss_log),
    }
    save_metrics_to_json(metrics, experiment_dir)

    # Plot loss and distribution comparison
    plot_loss_and_distribution(
        model=model,
        dataset_path=config["data"]["dataset_path"],
        classes=config["data"]["classes"],
        mean=config["data"]["mean"],
        std=config["data"]["std"],
        device=device,
        loss_log=loss_log,
        experiment_dir=experiment_dir,
    )

    # Plot sample images with predicted sizes
    plot_sample_images(
        model=model,
        dataset_path=config["data"]["dataset_path"],
        classes=config["data"]["classes"],
        mean=config["data"]["mean"],
        std=config["data"]["std"],
        device=device,
        experiment_dir=experiment_dir,
    )
    
    # Plot prediction monotonicity
    plot_prediction_monotonicity(
        model=model,
        dataset_path=config["data"]["dataset_path"],
        classes=config["data"]["classes"],
        mean=config["data"]["mean"],
        std=config["data"]["std"],
        device=device,
        experiment_dir=experiment_dir,
        num_samples=1000  
    )
    
    # Plot per-class performance
    plot_per_class_performance(
        model=model,
        dataset_path=config["data"]["dataset_path"],
        classes=config["data"]["classes"],
        mean=config["data"]["mean"],
        std=config["data"]["std"],
        device=device,
        experiment_dir=experiment_dir,
    )

    print(f"Training completed. Model and plots saved to {experiment_dir}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)