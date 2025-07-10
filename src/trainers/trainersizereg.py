import torch
import torch.nn as nn
from torch.nn import L1Loss, MSELoss, HuberLoss
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import os
from tqdm import tqdm
from src.losses import RMSELoss

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
        
        # Loss function
        if config["loss"]["type"] == "L1":
            self.criterion = L1Loss(**config["loss"]["parameters"])
        elif config["loss"]["type"] == "MSE":
            self.criterion = MSELoss(**config["loss"]["parameters"])
        elif config["loss"]["type"] == "Huber":
            self.criterion = HuberLoss(**config["loss"]["parameters"])
        elif config["loss"]["type"] == "RMSE":
            self.criterion = RMSELoss(**config["loss"]["parameters"])
        else:
            raise ValueError(f"Unsupported loss type: {config['loss']['type']}")
        
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
                self.writer.add_scalar("MSE/Train", train_error, epoch)
                self.writer.add_scalar("MSE/Validation", val_error, epoch)

            if val_error < best_val_error:
                best_val_error = val_error
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    f"val {self.config['loss']['type']} error": best_val_error,
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
                "error": loss,
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