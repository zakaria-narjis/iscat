import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.data_processing.dataset import iScatDataset 
from src.trainers import Trainer
from src.models.Unet import UNet
from src.data_processing.utils import Utils  # Assuming utility functions like get_data_paths are in utils.py
import re
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np

def load_config(config_path):
    """
    Load configuration file with variable interpolation support using OmegaConf.
    """
    # Load the YAML file using OmegaConf
    config = OmegaConf.load(config_path)
    
    # Resolve all variable interpolations
    resolved_config = OmegaConf.to_container(config, resolve=True)
    
    return resolved_config


def get_args_parser(add_help:bool=True):
    parser = argparse.ArgumentParser(description='iScat Segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    return parser

def create_dataloaders(train_dataset, valid_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
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
        'General': ['seed', 'experiment_name'],
        'Data': [
            ('data.image_size', 'Image Size'),
            ('data.image_indices', 'Image Indices'),
            ('data.fluo_masks_indices', 'Fluorescence Mask Indices'),
            ('data.seg_method', 'Segmentation Method'),
            ('data.data_type', 'Data Type')
        ],
        'Training': [
            ('training.batch_size', 'Batch Size'),
            ('training.num_epochs', 'Epochs'),
            ('training.loss_type', 'Loss Type')
        ],
        'Model': [
            ('model.type', 'Model Type'),
            ('model.num_classes', 'Number of Classes'),
            ('model.init_features', 'Initial Features')
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
    experiment_name = sanitize_filename(config['experiment_name'])
    experiment_folder_name = f'{config["model"]["type"]}_{config["data"]["data_type"]}_{getdatetime()}'
    experiment_folder_name = experiment_folder_name[:100]  # Limit folder name length
    experiment_dir = os.path.join(config['logging']['tensorboard']['log_dir'], experiment_folder_name)
    writer = SummaryWriter(log_dir=experiment_dir)
    write_config_to_tensorboard(writer, config)

    # Log config to TensorBoard

    # Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
          
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    # Get data paths
    image_paths, target_paths = [], []
    for data_path in config['data']['data_paths']:
        i, t = Utils.get_data_paths(data_path, mode=config['data']['data_type'],image_indices=config['data']['image_indices'])
        image_paths.extend(i)
        target_paths.extend(t)

    # Create datasets
    train_dataset = iScatDataset(
        image_paths[:-2],
        target_paths[:-2],
        preload_image=config['data']['train_dataset']['preload_image'],
        image_size=tuple(config['data']['train_dataset']['image_size']),
        apply_augmentation=config['data']['train_dataset']['apply_augmentation'],
        normalize=config['data']['train_dataset']['normalize'],
        device=device,
        fluo_masks_indices=config['data']['train_dataset']['fluo_masks_indices'],
        seg_method=config['data']['train_dataset']['seg_method']
    )

    valid_dataset = iScatDataset(
        image_paths[-2:],
        target_paths[-2:],
        preload_image=config['data']['valid_dataset']['preload_image'],
        image_size=tuple(config['data']['valid_dataset']['image_size']),
        apply_augmentation=config['data']['valid_dataset']['apply_augmentation'],
        normalize=config['data']['valid_dataset']['normalize'],
        device=device,
        fluo_masks_indices=config['data']['valid_dataset']['fluo_masks_indices'],
        seg_method=config['data']['valid_dataset']['seg_method']
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, valid_dataset, batch_size=config['training']['batch_size']
    )

    # Initialize model
    if type(config['data']['image_indices'])==int:
        input_channels = config['data']['image_indices']
    elif type(config['data']['image_indices'])==list:
        input_channels = len(config['data']['image_indices'])
    else:
        raise ValueError("Invalid image_indices type")
   
    model = UNet(
        in_channels=input_channels,
        num_classes=config['model']['num_classes'],
        init_features=config['model']['init_features'],
        pretrained=config['model']['pretrained']
    )
    if config['training']['class_weights']['use']:
        class_weights=Utils.calculate_class_weights_from_masks(train_dataset.masks).to(device)
    else:
        class_weights=None
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=device,
        config=config['training'],
        writer=writer,
        experiment_dir=experiment_dir,
        class_weights=class_weights
    )

    # Train the model
    trainer.train(
        train_loader, val_loader,
        num_epochs=config['training']['num_epochs']
    )

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
