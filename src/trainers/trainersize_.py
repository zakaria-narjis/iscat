import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.knn_loss import knn_divergence
import logging
import os
import torch
import torch.nn as nn
from tqdm import tqdm

def generate_label_distribution(num_points=10000, mean=76, std=22.5, min_value=10, max_value=None):
    """
    Generate a tensor of points sampled from a normal distribution with specified mean and standard deviation
    while rejecting points outside the optional min and max value constraints.
    
    Args:
        num_points (int): Number of points to generate
        mean (float): Mean of the distribution
        std (float): Standard deviation of the distribution
        min_value (float, optional): Minimum value of the distribution (inclusive)
        max_value (float, optional): Maximum value of the distribution (inclusive)
    
    Returns:
        torch.Tensor: Tensor of generated points within the specified range
    """
    points = torch.empty(0)  # Initialize an empty tensor to store valid points

    while points.numel() < num_points:
        # Generate points from normal distribution
        generated_points = torch.normal(mean=mean, std=std, size=(num_points,))
        
        # Filter points based on the min and max values
        if min_value is not None:
            generated_points = generated_points[generated_points >= min_value]
        if max_value is not None:
            generated_points = generated_points[generated_points <= max_value]
        
        # Add the valid points to the tensor
        points = torch.cat((points, generated_points))
    # Return only the first `num_points` points
    return points[:num_points]

def setup_device_and_model(device_config, model):
    """
    Setup device configuration and wrap model for multi-GPU training if needed.
    More robust version for DataParallel.
    
    Args:
        device_config: Either int (single GPU), list of ints (multi-GPU), or string ('cpu')
        model: The neural network model
    
    Returns:
        tuple: (device, model, is_parallel)
    """
    is_parallel = False
    
    # Handle different device configurations
    if isinstance(device_config, str):
        if device_config.lower() == 'cpu':
            device = torch.device('cpu')
            model = model.to(device)
            return device, model, is_parallel
        else:
            device = torch.device(device_config)
            model = model.to(device)
            return device, model, is_parallel
    
    elif isinstance(device_config, int):
        # Single GPU
        if torch.cuda.is_available() and device_config < torch.cuda.device_count():
            device = torch.device(f'cuda:{device_config}')
            model = model.to(device)
        else:
            print(f"Warning: GPU {device_config} not available, falling back to CPU")
            device = torch.device('cpu')
            model = model.to(device)
        return device, model, is_parallel
    
    elif isinstance(device_config, list):
        # Multi-GPU setup
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = torch.device('cpu')
            model = model.to(device)
            return device, model, is_parallel
        
        # Filter available GPUs
        available_gpus = [gpu for gpu in device_config if gpu < torch.cuda.device_count()]
        
        if not available_gpus:
            print(f"Warning: No GPUs from {device_config} are available, falling back to CPU")
            device = torch.device('cpu')
            model = model.to(device)
            return device, model, is_parallel
        
        if len(available_gpus) == 1:
            device = torch.device(f'cuda:{available_gpus[0]}')
            model = model.to(device)
            return device, model, is_parallel
        
        # BEST APPROACH: Let DataParallel handle everything
        device = torch.device(f'cuda:{available_gpus[0]}')  # Primary device for loss computation
        
        # Clear any existing GPU memory
        torch.cuda.empty_cache()
        
        # Create DataParallel model without pre-moving to device
        # DataParallel will handle the initial placement
        if torch.cuda.is_available():
            model = nn.DataParallel(model, device_ids=available_gpus)
            # Move to cuda (lets DataParallel decide the distribution)
            model = model.cuda()
            is_parallel = True
            
            print(f"Using DataParallel with GPUs: {available_gpus}")
            print(f"Model placed on: cuda (DataParallel managed)")
            
            # Verify GPU distribution
            print("GPU memory after DataParallel setup:")
            for gpu_id in available_gpus:
                torch.cuda.synchronize(gpu_id)  # Ensure operations are complete
                allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                print(f"  GPU {gpu_id}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        return device, model, is_parallel
    
    else:
        raise ValueError(f"Unsupported device configuration: {device_config}")

class TrainerSize(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        device_config,  # Changed from device to device_config
        config: dict,
        experiment_dir: str,
        writer: SummaryWriter = None,
        verbose: bool = True
    ):
        super(TrainerSize, self).__init__()
     
        # Setup device and model (including DataParallel if needed)
        self.device, self.model, self.is_parallel = setup_device_and_model(device_config, model)
        # Add this to TrainerSize.__init__ after device setup
        if self.is_parallel:
            print(f"GPU memory before training:")
            for i, gpu_id in enumerate(self.model.device_ids):
                print(f"GPU {gpu_id}: {torch.cuda.memory_allocated(gpu_id) / 1024**3:.2f} GB allocated")           
        self.config = config
        self.experiment_dir = experiment_dir
        self.writer = writer
        self.verbose = verbose
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['optimizer']['parameters']['lr'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode=self.config['scheduler']['parameters']["mode"], 
            factor=self.config['scheduler']['parameters']['factor'], 
            patience=self.config['scheduler']['parameters']['patience'], 
        )
        self.checkpoint_path = os.path.join(experiment_dir, 'best_model.pth')

        # Configure logging
        self.logger = logging.getLogger(__name__)
        log_level = logging.DEBUG if verbose else logging.WARNING
        logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
        
        # Log device setup
        if self.is_parallel:
            gpu_ids = self.model.device_ids
            self.logger.info(f"Training with DataParallel on GPUs: {gpu_ids}")
        else:
            self.logger.info(f"Training on device: {self.device}")

    def train_epoch(self, train_dataloader: DataLoader, label_points: torch.Tensor, k_neighbors: torch.Tensor):
        """
        Train the model for one epoch.
        
        Args:
            train_dataloader (DataLoader): Training data loader
            label_points (torch.Tensor): Label points for training
            k_neighbors (torch.Tensor): K neighbors for KNN loss
        
        Returns:
            float: Average loss for the epoch
        """
        total_loss = 0
        for batch_images, _ in train_dataloader:
            # FIX: Ensure target_distribution is on the same device as batch_images
            target_distribution = torch.clone(label_points).to(self.device, non_blocking=True)
            # batch_images = batch_images.to(self.device, non_blocking=True)
            batch_images = batch_images.cuda()
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass: generate predictions
            batch_predictions = self.model(batch_images)
            
            # Compute KNN divergence loss
            loss = knn_divergence(batch_predictions, 
                                target_distribution, 
                                k_neighbors=k_neighbors,
                                method="absolute")

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_dataloader)
        return avg_loss

    def train(self, train_dataloader: DataLoader, num_epochs: int):
        """
        Train the model for a specified number of epochs.
        
        Args:
            train_dataloader (DataLoader): Dataloader for training
            num_epochs (int): Number of epochs to train
        
        Returns:
            list: Loss history
        """
        # Early stopping parameters
        best_loss = float('inf')
        no_improve = 0
        loss_log = []
        
        # FIX: Move k_neighbors to the correct device
        k_neighbors = torch.arange(2, self.config["target_distribution"]["k"], dtype=torch.int).to(self.device)
        
        # Training loop
        for epoch in tqdm(range(num_epochs), disable=not self.logger.isEnabledFor(logging.DEBUG)):
            if epoch % self.config["target_distribution"]["cycle"] == 0:
                # Generate new label points every N epochs
                label_points = generate_label_distribution(
                    num_points=self.config["target_distribution"]["num_points"],
                    mean=self.config["target_distribution"]["mean"],
                    std=self.config["target_distribution"]["std"],
                    min_value=self.config["target_distribution"]["min_value"],
                    max_value=self.config["target_distribution"]["max_value"]
                ).to(self.device, non_blocking=True)
                
            self.model.train()
            avg_loss = self.train_epoch(train_dataloader, label_points, k_neighbors)
            loss_log.append(avg_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Learning rate scheduling
            self.scheduler.step(avg_loss)

            # Log the loss
            self.logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.2e}')
            if self.writer:
                self.writer.add_scalar('Loss/train', avg_loss, epoch)

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                # Save the model state dict properly for DataParallel
                model_state_dict = self.model.module.state_dict() if self.is_parallel else self.model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_loss,
                    'is_parallel': self.is_parallel,
                }, self.checkpoint_path)
                no_improve = 0             
            else:
                no_improve += 1

            if no_improve >= self.config["early_stopping"]["patience"] and self.config['early_stopping']['enabled']:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return loss_log
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint with proper handling of DataParallel models.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state dict
        model_state_dict = checkpoint['model_state_dict']
        
        if self.is_parallel:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
        
        # Load optimizer state dict
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
        
        return checkpoint['epoch'], checkpoint['loss']