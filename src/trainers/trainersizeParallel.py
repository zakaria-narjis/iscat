import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.knn_loss import knn_divergence
import logging
import os
from tqdm import tqdm
import torch.distributed as dist


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def generate_label_distribution(
    num_points=10000, mean=76, std=22.5, min_value=10, max_value=None
):
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


class TrainerSize(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict,
        experiment_dir: str,
        writer: SummaryWriter = None,
        verbose: bool = True,
    ):
        super(TrainerSize, self).__init__()
        self.model = model
        self.device = device
        self.config = config
        self.experiment_dir = experiment_dir
        self.writer = writer
        self.verbose = verbose
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.config["optimizer"]["parameters"]["lr"]
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.config["scheduler"]["parameters"]["mode"],
            factor=self.config["scheduler"]["parameters"]["factor"],
            patience=self.config["scheduler"]["parameters"]["patience"],
        )

        # Configure logging
        self.logger = logging.getLogger(__name__)
        log_level = logging.DEBUG if verbose else logging.WARNING
        logging.basicConfig(
            level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def train_epoch(
        self,
        train_dataloader: DataLoader,
        label_points: torch.Tensor,
        k_neighbors: torch.Tensor,
    ):
        """
        Train the model for one epoch.

        Args:
            epoch (int): Current epoch number
            dataloaders (list): List of dataloaders for training
            label_points (list): List of label points for training

        Returns:
            None
        """
        total_loss = 0
        for batch_images, _ in train_dataloader:
            target_distribution = torch.clone(label_points)
            batch_images = batch_images.to(self.device)
            # Zero gradients
            self.optimizer.zero_grad()
            # Forward pass: generate predictions
            batch_predictions = self.model(batch_images)
            # Compute KNN divergence loss
            loss = knn_divergence(
                batch_predictions,
                target_distribution,
                k_neighbors=k_neighbors,
                method="absolute",
            )

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
            None
        """
        # Early stopping parameters
        best_loss = float("inf")
        no_improve = 0
        loss_log = []
        k_neighbors = torch.arange(
            2, self.config["target_distribution"]["k"], dtype=torch.int
        )
        # Training loop
        for epoch in tqdm(
            range(num_epochs),
            disable=not self.logger.isEnabledFor(logging.DEBUG),
        ):
            train_dataloader.sampler.set_epoch(epoch)
            if epoch % self.config["target_distribution"]["cycle"] == 0:
                # Generate new label points every 5 epochs
                label_points = generate_label_distribution(
                    num_points=self.config["target_distribution"]["num_points"],
                    mean=self.config["target_distribution"]["mean"],
                    std=self.config["target_distribution"]["std"],
                    min_value=self.config["target_distribution"]["min_value"],
                    max_value=self.config["target_distribution"]["max_value"],
                ).to(self.device, non_blocking=True)
            self.model.train()
            avg_loss = self.train_epoch(
                train_dataloader, label_points, k_neighbors
            )
            if is_main_process():
                loss_log.append(avg_loss)
                current_lr = self.optimizer.param_groups[0]["lr"]
                # Learning rate scheduling
                self.scheduler.step(avg_loss)

                # Log the loss
                self.logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.2e}"
                )
                if self.writer:
                    self.writer.add_scalar("Loss/train", avg_loss, epoch)

                # Early stopping check
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "loss": best_loss,
                        },
                        os.path.join(self.experiment_dir, "best_model.pth"),
                    )
                    no_improve = 0
                else:
                    no_improve += 1

                if (
                    no_improve >= self.config["early_stopping"]["patience"]
                    and self.config["early_stopping"]["enabled"]
                ):
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        return loss_log
