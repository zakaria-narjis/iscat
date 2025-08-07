import torch
import torch.nn as nn
from torch.nn import L1Loss, MSELoss, HuberLoss
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import numpy as np
from tqdm import tqdm
from src.losses import RMSELoss

class WeightedLoss(nn.Module):
    """Wrapper for applying class weights to loss functions"""
    def __init__(self, base_loss_fn, class_weights):
        super(WeightedLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.class_weights = class_weights
    
    def forward(self, predictions, targets, class_labels):
        # Calculate base loss for each sample
        if isinstance(self.base_loss_fn, (L1Loss, MSELoss)):
            sample_losses = torch.abs(predictions - targets) if isinstance(self.base_loss_fn, L1Loss) else (predictions - targets) ** 2
        elif isinstance(self.base_loss_fn, HuberLoss):
            sample_losses = torch.where(
                torch.abs(predictions - targets) <= self.base_loss_fn.delta,
                0.5 * (predictions - targets) ** 2,
                self.base_loss_fn.delta * (torch.abs(predictions - targets) - 0.5 * self.base_loss_fn.delta)
            )
        else:  # RMSELoss or custom
            sample_losses = (predictions - targets) ** 2
        
        # Apply class weights
        weights = torch.tensor([self.class_weights[cls.item()] for cls in class_labels], 
                              device=predictions.device, dtype=predictions.dtype)
        weighted_losses = sample_losses * weights
        
        return weighted_losses.mean()

class ParticleSizeTrainer(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict,
        experiment_dir: str,
        writer: SummaryWriter = None,
        verbose: bool = True,
    ):
        super(ParticleSizeTrainer, self).__init__()
        self.model = model
        self.device = device
        self.config = config
        self.experiment_dir = experiment_dir
        self.writer = writer
        self.verbose = verbose
        self.training_noise_std = config["training_noise_std"]
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.config["optimizer"]["parameters"]["lr"]
            , weight_decay=config["optimizer"]["parameters"].get("weight_decay", 5e-2)
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.config["scheduler"]["parameters"]["mode"],
            factor=self.config["scheduler"]["parameters"]["factor"],
            patience=self.config["scheduler"]["parameters"]["patience"],
        )
        
        # Loss function (base, will be wrapped if weighting is enabled)
        if config["loss"]["type"] == "L1":
            self.base_criterion = L1Loss(**config["loss"]["parameters"])
        elif config["loss"]["type"] == "MSE":
            self.base_criterion = MSELoss(**config["loss"]["parameters"])
        elif config["loss"]["type"] == "Huber":
            self.base_criterion = HuberLoss(**config["loss"]["parameters"])
        elif config["loss"]["type"] == "RMSE":
            self.base_criterion = RMSELoss(**config["loss"]["parameters"])
        else:
            raise ValueError(f"Unsupported loss type: {config['loss']['type']}")
        
        # Initialize criterion (will be set in train method if weighting is enabled)
        self.criterion = self.base_criterion
        
        # Checkpoint path
        self.checkpoint_path = os.path.join(experiment_dir, "best_model.pth")
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        log_level = logging.DEBUG if verbose else logging.WARNING
        logging.basicConfig(
            level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        
        # Get unique classes from config for per-class logging
        self.classes = self.config.get("classes")
        self.class_weights = None
        
    def calculate_class_weights(self, train_dataloader):
        """
        Calculate class weights based on frequency and range.
        
        Formula: w_i = (1/range_i) * (K * sqrt(1/n_i) / sum(sqrt(1/n_j)))
        Where:
        - range_i: max - min size for class i
        - n_i: number of samples for class i  
        - K: total number of classes
        
        Args:
            train_dataloader (DataLoader): Training dataloader
            
        Returns:
            dict: Class weights mapping {class: weight}
        """
        self.logger.info("Calculating class weights based on frequency and range...")
        
        # Collect all data to compute statistics
        class_sizes = {cls: [] for cls in self.classes}
        class_counts = {cls: 0 for cls in self.classes}
        
        # Iterate through the dataset to collect statistics
        for batch_data, class_labels, size_labels in train_dataloader:
            for cls_label, size_label in zip(class_labels, size_labels):
                cls = cls_label.item() if torch.is_tensor(cls_label) else cls_label
                size = size_label.item() if torch.is_tensor(size_label) else size_label
                
                if cls in class_sizes:
                    class_sizes[cls].append(size)
                    class_counts[cls] += 1
        
        # Calculate range for each class
        class_ranges = {}
        for cls in self.classes:
            if len(class_sizes[cls]) > 0:
                class_ranges[cls] = max(class_sizes[cls]) - min(class_sizes[cls])
                # Avoid division by zero for classes with no variation
                if class_ranges[cls] == 0:
                    class_ranges[cls] = 1.0
            else:
                class_ranges[cls] = 1.0
                class_counts[cls] = 1  # Avoid division by zero
        
        # Calculate frequency weights: K * sqrt(1/n_i) / sum(sqrt(1/n_j))
        K = len(self.classes)
        sqrt_inv_counts = {cls: np.sqrt(1 / class_counts[cls]) for cls in self.classes}
        sum_sqrt_inv_counts = sum(sqrt_inv_counts.values())
        
        frequency_weights = {cls: K * sqrt_inv_counts[cls] / sum_sqrt_inv_counts 
                           for cls in self.classes}
        
        # Calculate final weights: (1/range) * frequency_weight
        class_weights = {cls: (1 / class_ranges[cls]) * frequency_weights[cls] 
                        for cls in self.classes}
        
        # Log statistics
        total_samples = sum(class_counts.values())
        for cls in self.classes:
            self.logger.info(
                f"Class {cls}: count={class_counts[cls]}, "
                f"range={class_ranges[cls]:.4f}, "
                f"frequency_weight={frequency_weights[cls]:.4f}, "
                f"final_weight={class_weights[cls]:.4f}, "
                f"percentage={100*class_counts[cls]/total_samples:.2f}%"
            )
        
        return class_weights
        
    def train_epoch(self, train_dataloader: DataLoader):
        """
        Train the model for one epoch.
        
        Args:
            train_dataloader (DataLoader): Training dataloader
            
        Returns:
            tuple: (total_error, per_class_error_dict)
        """
        total_loss = 0
        total_samples = 0
        
        # Track per-class metrics
        class_losses = {cls: 0 for cls in self.classes}
        class_counts = {cls: 0 for cls in self.classes}
        
        self.model.train()
        
        for batch_data, class_labels, size_labels in train_dataloader:
            batch_data = batch_data.to(self.device)
            class_labels = class_labels.to(self.device)
            size_labels = size_labels.to(self.device).float()
            
            if self.training_noise_std > 0:
                noise = torch.randn_like(size_labels) * self.training_noise_std
                size_labels = size_labels + noise
                
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_data).squeeze()
            
            # Compute loss
            if self.class_weights is not None:
                # Use weighted loss
                loss = self.criterion(predictions, size_labels, class_labels)
            else:
                # Use standard loss
                loss = self.criterion(predictions, size_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update total metrics
            total_loss += loss.item() * len(size_labels)
            total_samples += len(size_labels)
            
            # Update per-class metrics
            for cls in self.classes:
                class_mask = class_labels == cls
                if class_mask.sum() > 0:
                    class_preds = predictions[class_mask]
                    class_targets = size_labels[class_mask]
                    class_loss = self.base_criterion(class_preds, class_targets)
                    class_losses[cls] += class_loss.item() * class_mask.sum().item()
                    class_counts[cls] += class_mask.sum().item()
        
        # Calculate average losses and convert to error
        avg_loss = total_loss / total_samples
        
        # Calculate per-class error
        per_class_error = {}
        for cls in self.classes:
            if class_counts[cls] > 0:
                class_error = class_losses[cls] / class_counts[cls]
                per_class_error[cls] = class_error
            else:
                per_class_error[cls] = 0.0
        
        return avg_loss, per_class_error
    
    def train(self, train_dataloader, val_dataloader, num_epochs):
        # Check if frequency weighting is enabled
        use_frequency_weighting = self.config["loss"].get("frequency_weighting", False)
        
        if use_frequency_weighting:
            # Calculate class weights
            self.class_weights = self.calculate_class_weights(train_dataloader)
            # Wrap the base criterion with weighted loss
            self.criterion = WeightedLoss(self.base_criterion, self.class_weights)
            self.logger.info(f"Frequency weighting enabled - using weighted loss function, class weights: {self.class_weights}")
        else:
            self.logger.info("Frequency weighting disabled - using standard loss function")
        
        best_val_error = float("inf")
        no_improve = 0
        loss_log = []

        for epoch in tqdm(range(num_epochs), disable=not self.logger.isEnabledFor(logging.DEBUG)):
            train_error, per_class_error = self.train_epoch(train_dataloader)
            val_error = self.validate_epoch(val_dataloader)
            loss_log.append(val_error)
            self.scheduler.step(val_error)

            self.logger.info(
                        f"Epoch [{epoch+1}/{num_epochs}], "
                        f"Train {self.config['loss']['type']} error: {train_error:.4f}, "
                        f"Val {self.config['loss']['type']} error: {val_error:.4f}, "
                        f"Patience: {no_improve}/{self.config['early_stopping']['patience']}"
                    )

            
            if self.writer:
                self.writer.add_scalar("Loss/Train", train_error, epoch)
                self.writer.add_scalar("Loss/Validation", val_error, epoch)
                
                # Log per-class errors if available
                for cls, error in per_class_error.items():
                    self.writer.add_scalar(f"Loss/Train_Class_{cls}", error, epoch)

            if val_error < best_val_error:
                best_val_error = val_error
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    f"val_{self.config['loss']['type']}_error": best_val_error,
                    "class_weights": self.class_weights,
                }, self.checkpoint_path)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.config["early_stopping"]["patience"] and self.config["early_stopping"]["enabled"]:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return loss_log

    
    def validate_epoch(self, val_dataloader):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for batch_data, class_labels, size_labels in val_dataloader:
                batch_data = batch_data.to(self.device)
                class_labels = class_labels.to(self.device)
                size_labels = size_labels.to(self.device).float()
                predictions = self.model(batch_data).squeeze()
                
                # Use base criterion for validation (unweighted)
                loss = self.base_criterion(predictions, size_labels)
                total_loss += loss.item() * len(size_labels)
                total_samples += len(size_labels)
        return total_loss / total_samples

    
    def save_checkpoint(self, epoch: int, loss: float, filepath: str = None):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch
            loss (float): Current loss
            filepath (str): Path to save checkpoint
        """
        if filepath is None:
            filepath = self.checkpoint_path
            
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "error": loss,
                "class_weights": self.class_weights,
            },
            filepath,
        )
        
    def load_checkpoint(self, filepath: str = None):
        """
        Load model checkpoint.
        
        Args:
            filepath (str): Path to checkpoint file
            
        Returns:
            dict: Checkpoint data
        """
        if filepath is None:
            filepath = self.checkpoint_path
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        # Load class weights if available
        if "class_weights" in checkpoint:
            self.class_weights = checkpoint["class_weights"]
            
        return checkpoint