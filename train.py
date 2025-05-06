import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.data_processing.dataset import iScatDataset
from src.trainers.trainer2 import Trainer
from src.models.Unet import UNet
from src.data_processing.utils import Utils 
import re
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from src.visualization import predict, batch_plot_images_with_masks
from sklearn.model_selection import train_test_split
import h5py
from src.models.Unet_networks import AttU_Net, R2AttU_Net, R2U_Net, U_Net
from src.metrics import batch_multiclass_metrics
from test import test_model
import json

def save_metrics_to_json(metrics_dict, output_folder):
    """
    Save test metrics dictionary to a JSON file.
    
    Args:
        metrics_dict (dict): Metrics dictionary from test function
        output_folder (str): Folder path to save the JSON file
    
    Returns:
        str: Full path to the saved JSON file
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_metrics_{timestamp}.json"
    
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


def get_args_parser(add_help:bool=True):
    parser = argparse.ArgumentParser(description='iScat Segmentation')
    parser.add_argument('--config', type=str, default="configs/seg_config.yaml", help='Path to the configuration file')
    return parser

def create_dataloaders(train_dataset, valid_dataset,test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader,test_loader

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
            ('data.image_size', 'Image Size'),
            ('data.z_chunk_size', 'Z-Stack Chunk Size'),
            ('data.fluo_masks_indices', 'Fluorescence Mask Indices'),
            ('data.seg_method', 'Segmentation Method'),
            ('data.data_type', 'Data Type'),
            ('data.normalize', 'Normalization Method')
        ],
        'Training': [
            ('training.batch_size', 'Batch Size'),
            ('training.num_epochs', 'Epochs'),
            ('training.device', 'Training Device'),
            ('training.loss_type', 'Loss Type'),
            ('training.optimizer.type', 'Optimizer'),
            ('training.optimizer.parameters.lr', 'Learning Rate')
        ],
        'Model': [
            ('model.type', 'Model Type'),
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
    # experiment_name = sanitize_filename(config['experiment_name'])
    experiment_folder_name = f'{config["model"]["type"]}_{config["data"]["data_type"]}_{getdatetime()}'
    experiment_folder_name = experiment_folder_name[:100]  # Limit folder name length
    experiment_dir = os.path.join(config['logging']['tensorboard']['log_dir'], experiment_folder_name)
    writer = SummaryWriter(log_dir=experiment_dir)
    write_config_to_tensorboard(writer, config)

    # Log config to TensorBoard

    # Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

    if config['data']['multi_class']:
        num_classes = len(config['data']['train_dataset']['classes']) + 1 
    else:        
        num_classes = 1
    in_channels = config['data']['z_chunk_size']
    out_channels = num_classes
    config['training']['num_classes'] = num_classes
    config['model']['in_channels'] = in_channels
    config['model']['out_channels'] = out_channels
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    # Get data paths
    if config['data']['data_type'] == 'Brightfield':
        hdf5_path = os.path.join(config["data"]["dataset_folder_path"], 'brightfield.hdf5')
    elif config['data']['data_type'] == 'Laser':
        hdf5_path = os.path.join(config["data"]["dataset_folder_path"], 'Laser.hdf5')

    with h5py.File(hdf5_path, "r") as f:
        num_samples = f["image_patches"].shape[0]
    
    indices = np.arange(num_samples)
    train_indices, temp_indices = train_test_split(indices, test_size=1-config['training']['train_split_size'], random_state=config['seed'])
    valid_indices, test_indices = train_test_split(temp_indices, test_size=1/3, random_state=config['seed'])   
    # Create datasets
    train_dataset = iScatDataset(
        hdf5_path=hdf5_path,
        indices=train_indices,
        classes=config['data']['train_dataset']['classes'],
        apply_augmentation=config['data']['train_dataset']['apply_augmentation'],
        normalize=config['data']['train_dataset']['normalize'],
        multi_class=config['data']['multi_class'],
    )

    valid_dataset = iScatDataset(
        hdf5_path=hdf5_path,
        indices=valid_indices,
        classes=config['data']['valid_dataset']['classes'],
        apply_augmentation=config['data']['valid_dataset']['apply_augmentation'],  
        normalize=config['data']['valid_dataset']['normalize'],
        multi_class=config['data']['multi_class'],
    )
    test_dataset = iScatDataset(
        hdf5_path=hdf5_path,
        indices=test_indices,
        classes=config['data']['train_dataset']['classes'],
        apply_augmentation=False,
        normalize=config['data']['train_dataset']['normalize'],
        multi_class=config['data']['multi_class'],
    )
    # Create dataloaders
    train_loader, val_loader ,test_loader= create_dataloaders(
        train_dataset, valid_dataset, test_dataset, batch_size=config['training']['batch_size']
    )

    if config['model']['type'] == 'U_Net':
        model = U_Net(
            img_ch=in_channels,
            output_ch=out_channels,
        )
    elif config['model']['type'] == 'AttU_Net':
        model = AttU_Net(
            img_ch=in_channels,
            output_ch=out_channels
        )
    elif config['model']['type'] == 'R2AttU_Net':
        model = R2AttU_Net(
            img_ch=in_channels,
            output_ch=out_channels
        )
    elif config['model']['type'] == 'R2U_Net':
        model = R2U_Net(
            img_ch=in_channels,
            output_ch=out_channels
        )
    else:
        raise ValueError(f"Invalid model type: {config['model']['type']}")
    
    if config['training']['class_weights']['use']:
        class_weights=Utils.calculate_class_weights_from_masks(Utils.load_masks_from_hdf5(hdf5_path,indices=config['data']['train_dataset']['classes'])).to(device)
        if num_classes == 1:
            class_weights = class_weights[1]/class_weights[0]
    else:
        class_weights=None
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=device,
        config=config['training'],
        writer=writer,
        experiment_dir=experiment_dir,
        class_weights=class_weights,
    )

    # Train the model
    trainer.train(
        train_loader, val_loader,
        num_epochs=config['training']['num_epochs']
    )

    test_results = test_model(model, test_loader, device, num_classes)
    save_metrics_to_json(test_results, experiment_dir)
    all_images, all_pred_masks,all_gt_masks = predict(model=model, dataset=test_dataset,device=device, images_indicies=[0,1,2,4]) 
    batch_plot_images_with_masks(all_images, all_pred_masks,all_gt_masks, output_dir=experiment_dir)
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
