import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.trainers.trainersize import TrainerSize
from src.data_processing.size_dataset import ParticleDataset
from src.models.resnet import ResNet18
import re
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import json
from torchvision.transforms import v2  

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
    with open(full_path, 'w') as f:
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
    parser = argparse.ArgumentParser(description='iScat Size Prediction')
    parser.add_argument('--config', type=str, default="configs/size_pred_config.yaml", help='Path to the configuration file')
    return parser


def create_dataloaders(train_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
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
        'General': ['seed'],
        'Data': [
            ('data.classes', 'Classes'),
            ('data.mean', 'Normalization Mean'),
            ('data.std', 'Normalization Std'),
            ('data.padding', 'Padding'),
            ('data.dataset_path', 'Dataset Path')
        ],
        'Training': [
            ('training.batch_size', 'Batch Size'),
            ('training.num_epochs', 'Epochs'),
            ('training.device', 'Training Device'),
            ('training.optimizer.type', 'Optimizer'),
            ('training.optimizer.parameters.lr', 'Learning Rate')
        ],
        'Target Distribution': [
            ('training.target_distribution.num_points', 'Number of Points'),
            ('training.target_distribution.mean', 'Distribution Mean'),
            ('training.target_distribution.std', 'Distribution Std'),
            ('training.target_distribution.min_value', 'Min Value'),
            ('training.target_distribution.max_value', 'Max Value'),
            ('training.target_distribution.k', 'KNN K Value'),
            ('training.target_distribution.cycle', 'Cycle')
        ]
    }

    def get_nested_value(config, key_path):
        """Extract value from nested config using dot notation."""
        keys = key_path.split('.')
        value = config
        for k in keys:
            value = value[k]
        return value

    def format_value(value):
        """Format value for display, handling lists and other types."""
        if isinstance(value, list):
            return str(value).replace('[', '').replace(']', '')
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

        writer.add_text(
            f"Configuration/{section}",
            "\n".join(table_rows)
        )


def main(args):  
    # Load configuration
    config = load_config(args.config)
    set_random_seed(config['seed'])
    
    # Set up experiment directory and tensorboard writer
    experiment_folder_name = f'ResNet18_{getdatetime()}'
    experiment_folder_name = experiment_folder_name[:100]  # Limit folder name length
    experiment_dir = os.path.join(config['logging']['tensorboard']['log_dir'], experiment_folder_name)
    writer = SummaryWriter(log_dir=experiment_dir)
    write_config_to_tensorboard(writer, config)

    # Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    augmentation = v2.Compose([
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
    ])
    # Create dataset
    train_dataset = ParticleDataset(
        h5_path=config['data']['dataset_path'],
        classes=config['data']['classes'],
        transform=augmentation,  
        mean=config['data']['mean'],
        std=config['data']['std'],
        padding=config['data']['padding'],
        indices=None,
    )

    # Create dataloader
    train_loader = create_dataloaders(
        train_dataset, batch_size=config['training']['batch_size']
    )

    # Initialize model
    model = ResNet18(num_classes=1)  # For regression, output is a single value
    model = model.to(device)
    
    # Initialize trainer
    trainer = TrainerSize(
        model=model,
        device=device,
        config=config['training'],
        experiment_dir=experiment_dir,
        writer=writer,
        verbose=True
    )

    # Train the model
    loss_log = trainer.train(
        train_loader,
        num_epochs=config['training']['num_epochs']
    )
    
    # Save metrics
    metrics = {
        "loss_history": loss_log,
        "final_loss": loss_log[-1] if loss_log else None,
        "num_epochs_trained": len(loss_log)
    }
    save_metrics_to_json(metrics, experiment_dir)

    print(f"Training completed. Model saved to {experiment_dir}")
    
    # Clean up
    train_dataset.close()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
