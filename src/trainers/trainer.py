import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import MeanIoU
from monai.networks.utils import one_hot
from enum import Enum

class LossType(Enum):
    CROSSENTROPY = "crossentropy"
    DICE = "dice"
    COMBINED = "combined"

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict,
        experiment_dir: str,
        class_weights=None,
        writer: SummaryWriter = None
    ):
        """
        Args:
            model (nn.Module): PyTorch model to train.
            device (torch.device): Device to use for training.
            config (dict): Configuration dictionary.
            experiment_dir (str): Directory to save logs and checkpoints.
            class_weights (list): Class weights for loss computation.
            writer (SummaryWriter): Tensorboard writer.
        """
        self.num_classes = config.get('num_classes', 2)
        self.model = model.to(device)
        self.device = device
        self.loss_type = config.get("loss_type")
        self.class_weights = class_weights
        self.config = config
        self.earlystoping_patience = config.get('earlystoping_patience', 10)

        # Save config to log directory
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

        # Initialize loss function
        self.loss = self._initialize_loss()

        # Initialize metrics
        self.miou_metric = MeanIoU(include_background=True, reduction="mean")
        self.optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=self.config['scheduler_factor'], patience=self.config['scheduler_patience'], verbose=True
        )

        self.writer = writer
        self.checkpoint_path = os.path.join(experiment_dir, 'best_model.pth')

    def _initialize_loss(self):
        if self.num_classes == 1:
            if self.loss_type == LossType.CROSSENTROPY:
                return nn.BCEWithLogitsLoss()
            elif self.loss_type == LossType.DICE:
                return DiceLoss(sigmoid=True, squared_pred=False, batch=True, reduction="mean")
            else:
                return DiceCELoss(sigmoid=True, squared_pred=False, batch=True, reduction="mean")
        else:
            if self.loss_type == LossType.CROSSENTROPY:
                return nn.CrossEntropyLoss(weight=self.class_weights)
            elif self.loss_type == LossType.DICE:
                return DiceLoss(softmax=True, squared_pred=False, batch=True, reduction="mean")
            else:
                return DiceCELoss(softmax=True, squared_pred=False, batch=True, reduction="mean", weight=self.class_weights)

    def compute_loss(self, predictions, targets):
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
            if self.num_classes > 1:
                targets = one_hot(targets, num_classes=self.num_classes)
        targets = targets.float()
        return self.loss(predictions, targets)

    def compute_metrics(self, predictions, targets):
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)

        if self.num_classes == 1:
            pred_masks = (torch.sigmoid(predictions) > 0.5).float()
            pred_one_hot = torch.cat([1 - pred_masks, pred_masks], dim=1)
        else:
            pred_masks = (torch.softmax(predictions, dim=1) > 0.5).float()
            pred_one_hot = pred_masks

        target_one_hot = one_hot(targets, num_classes=self.num_classes)
        return self.miou_metric(pred_one_hot, target_one_hot).nanmean().item()

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_miou = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.compute_loss(predictions, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_miou += self.compute_metrics(predictions, masks)

            step = epoch * len(train_loader) + batch_idx
            if self.writer is not None:
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                self.writer.add_scalar('Train/mIoU', total_miou / (batch_idx + 1), step)

        return total_loss / len(train_loader), total_miou / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_miou = 0.0

        for images, masks in val_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            predictions = self.model(images)
            total_loss += self.compute_loss(predictions, masks).item()
            total_miou += self.compute_metrics(predictions, masks)

        return total_loss / len(val_loader), total_miou / len(val_loader)

    def train(self, train_loader, val_loader, num_epochs):
        best_val_miou = 0.0
        no_improve = 0
        for epoch in range(num_epochs):
            train_loss, train_miou = self.train_epoch(train_loader, epoch)
            val_loss, val_miou = self.validate(val_loader)

            self.scheduler.step(val_loss)
            if self.writer is not None:
                self.writer.add_scalar('Validation/Loss', val_loss, epoch)
                self.writer.add_scalar('Validation/mIoU', val_miou, epoch)

            if val_miou > best_val_miou:
                best_val_miou = val_miou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_miou': val_miou,
                }, self.checkpoint_path)
                no_improve = 0
            else:
                no_improve += 1

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}, Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")
            if no_improve >= self.earlystoping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
