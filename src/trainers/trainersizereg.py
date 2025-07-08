import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import os
from tqdm import tqdm
import numpy as np


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
        self.training_noise_std = config.get("training_noise_std", 2.0)
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
        
        # Loss function (mse)
        self.criterion = nn.MSELoss()
        
        # Checkpoint path
        self.checkpoint_path = os.path.join(experiment_dir, "best_model.pth")
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        log_level = logging.DEBUG if verbose else logging.WARNING
        logging.basicConfig(
            level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        
        # Get unique classes from config for per-class logging
        self.classes = self.config.get("classes", [0, 1])
        
    def train_epoch(self, train_dataloader: DataLoader):
        """
        Train the model for one epoch.
        
        Args:
            train_dataloader (DataLoader): Training dataloader
            
        Returns:
            tuple: (total_mse, per_class_mse_dict)
        """
        total_loss = 0
        total_samples = 0
        
        # Track per-class metrics
        class_losses = {cls: 0 for cls in self.classes}
        class_counts = {cls: 0 for cls in self.classes}
        
        self.model.train()
        
        for batch_data, class_labels, size_labels in train_dataloader:
            batch_data = batch_data.to(self.device)
            size_labels = size_labels.to(self.device).float()
            if self.training_noise_std > 0:
                noise = torch.randn_like(size_labels) * self.training_noise_std
                size_labels = size_labels + noise
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_data).squeeze()
            
            # Compute loss (MSE)
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
                    class_loss = self.criterion(class_preds, class_targets)
                    class_losses[cls] += class_loss.item() * class_mask.sum().item()
                    class_counts[cls] += class_mask.sum().item()
        
        # Calculate average losses and convert to mse
        avg_mse = total_loss / total_samples
        
        # Calculate per-class mse
        per_class_mse = {}
        for cls in self.classes:
            if class_counts[cls] > 0:
                class_mse = class_losses[cls] / class_counts[cls]
                per_class_mse[cls] = class_mse
            else:
                per_class_mse[cls] = 0.0
        
        return avg_mse, per_class_mse
    
    def train(self, train_dataloader, val_dataloader, num_epochs):
        best_val_mse = float("inf")
        no_improve = 0
        loss_log = []

        for epoch in tqdm(range(num_epochs), disable=not self.logger.isEnabledFor(logging.DEBUG)):
            train_mse, per_class_mse = self.train_epoch(train_dataloader)
            val_mse = self.validate_epoch(val_dataloader)
            loss_log.append(val_mse)
            self.scheduler.step(val_mse)

            self.logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")
            
            if self.writer:
                self.writer.add_scalar("MSE/Train", train_mse, epoch)
                self.writer.add_scalar("MSE/Validation", val_mse, epoch)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "mse": best_val_mse,
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
                size_labels = size_labels.to(self.device).float()
                predictions = self.model(batch_data).squeeze()
                loss = self.criterion(predictions, size_labels)
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
                "mse": loss,
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
            
        return checkpoint