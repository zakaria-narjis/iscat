import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceLoss, DiceCELoss, TverskyLoss
from monai.metrics import MeanIoU
from monai.networks.utils import one_hot
from tqdm import tqdm
import logging
from src.metrics import batch_multiclass_metrics
from src.trainers.utils import is_main_process
import torch.distributed as dist

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict,
        experiment_dir: str,
        class_weights=None,
        writer: SummaryWriter = None,
        verbose: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Args:
            model (nn.Module): PyTorch model to train.
            device (torch.device): Device to use for training.
            config (dict): Configuration dictionary.
            experiment_dir (str): Directory to save logs and checkpoints.
            class_weights (list): Class weights for loss computation.
            writer (SummaryWriter): Tensorboard writer.
            verbose (bool): If False, suppress output logs.
            rank (int): Process rank for distributed training.
            world_size (int): Total number of processes for distributed training.
        """
        self.num_classes = config["num_classes"]
        self.model = model.to(device)
        self.device = device
        self.loss_type = config["loss"]["loss_type"]
        self.class_weights = class_weights
        self.config = config
        self.earlystoping_patience = config["early_stopping"]["patience"]
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0

        # Configure logging
        self.logger = logging.getLogger(__name__)
        log_level = logging.DEBUG if verbose and self.is_main_process else logging.WARNING
        logging.basicConfig(
            level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # Initialize loss function
        self.loss = self._initialize_loss()

        # Initialize metrics
        self.miou_metric = MeanIoU(include_background=True, reduction="mean")
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.config["optimizer"]["parameters"]["lr"]
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config["scheduler"]["parameters"]["factor"],
            patience=self.config["scheduler"]["parameters"]["patience"],
        )

        self.writer = writer
        self.checkpoint_path = os.path.join(experiment_dir, "best_model.pth") if experiment_dir else None

    def _initialize_loss(self):
        if self.num_classes == 1:
            if self.loss_type == "crossentropy":
                return nn.BCEWithLogitsLoss()
            elif self.loss_type == "dice":
                return DiceLoss(
                    sigmoid=True,
                    squared_pred=True,
                    batch=True,
                    reduction="mean",
                )
            elif self.loss_type == "dicece":
                return DiceCELoss(
                    sigmoid=True,
                    squared_pred=True,
                    batch=True,
                    reduction="mean",
                    weight=self.class_weights,
                    lambda_ce=self.config["loss"]["parameters"]["lambda_ce"],
                    lambda_dice=self.config["loss"]["parameters"][
                        "lambda_dice"
                    ],
                )
            elif self.loss_type == "tversky":
                if self.is_main_process:
                    self.logger.info("Using Tversky Loss ")
                return TverskyLoss(
                    sigmoid=True,
                    batch=True,
                    reduction="mean",
                    alpha=self.config["loss"]["parameters"]["alpha"],
                    beta=self.config["loss"]["parameters"]["beta"],
                )
            else:
                raise ValueError(f"Invalid loss type: {self.loss_type}")
        else:
            if self.loss_type == "crossentropy":
                if self.is_main_process:
                    self.logger.info("Using CrossEntropy Loss ")
                return nn.CrossEntropyLoss(weight=self.class_weights)
            elif self.loss_type == "dice":
                if self.is_main_process:
                    self.logger.info("Using Dice Loss ")
                return DiceLoss(
                    softmax=True,
                    squared_pred=True,
                    batch=True,
                    reduction="mean",
                    include_background=False,
                )
            elif self.loss_type == "tversky":
                if self.is_main_process:
                    self.logger.info("Using Tversky Loss ")
                return TverskyLoss(
                    softmax=True,
                    batch=True,
                    reduction="mean",
                    alpha=self.config["loss"]["parameters"]["alpha"],
                    beta=self.config["loss"]["parameters"]["beta"],
                    include_background=False,
                )
            elif self.loss_type == "dicece":
                if self.is_main_process:
                    self.logger.info("Using Dice CrossEntropy Loss ")
                return DiceCELoss(
                    softmax=True,
                    squared_pred=True,
                    batch=True,
                    reduction="mean",
                    weight=self.class_weights,
                )
            else:
                raise ValueError(f"Invalid loss type: {self.loss_type}")

    def compute_loss(self, predictions, targets):
        """
        Compute loss given model predictions and target masks.
        Args:
            predictions (torch.Tensor): Model predictions. Shape: [B, N, H, W].
            targets (torch.Tensor): Target masks. Shape: [B, 1, H, W].
        """
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
        if self.num_classes > 1:
            targets = one_hot(targets, num_classes=self.num_classes, dim=1)
        return self.loss(predictions, targets)
    
    @torch.no_grad()
    def compute_metrics(self, predictions, targets):
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
        if self.num_classes > 1:
            pred_one_hot = one_hot(
                predictions.argmax(dim=1, keepdim=True),
                num_classes=self.num_classes,
            )  # [B, N, H, W]
            target_one_hot = one_hot(
                targets, num_classes=self.num_classes
            )  # [B, N, H, W]
        else:
            pred_one_hot = torch.sigmoid(predictions) > 0.5  # [B, 1, H, W]
            pred_one_hot = one_hot(pred_one_hot, num_classes=2)
            target_one_hot = one_hot(targets, num_classes=2)

        metric = self.miou_metric(pred_one_hot, target_one_hot)
        return metric.nanmean().item()

    def train_epoch(self, train_loader):
        self.model.train()
        # Keep losses on GPU to avoid frequent CPU transfers
        total_loss = torch.tensor(0.0, device=self.device)
        total_miou = torch.tensor(0.0, device=self.device)
        num_batches = torch.tensor(0.0, device=self.device)
        self.logger.info("Training epoch")
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.compute_loss(predictions, masks)
            loss.backward()
            self.optimizer.step()

            # Accumulate on GPU - no .item() calls
            total_loss += loss.detach()
            miou = torch.tensor(self.compute_metrics(predictions, masks), device=self.device)
            total_miou += miou
            num_batches += 1
        self.logger.info("Finished training epoch")
        # Single collective operation at the end
        if self.world_size > 1:
            # Reduce across all processes
            # dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            # dist.all_reduce(total_miou, op=dist.ReduceOp.SUM)
            # dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
            
            # Average across processes
            total_loss /= self.world_size
            total_miou /= self.world_size
            num_batches /= self.world_size

        # Convert to CPU only once at the end
        avg_loss = (total_loss / num_batches).item()
        avg_miou = (total_miou / num_batches).item()

        return avg_loss, avg_miou

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        # Keep metrics on GPU initially
        total_loss = torch.tensor(0.0, device=self.device)
        total_miou = torch.tensor(0.0, device=self.device)
        num_batches = torch.tensor(0.0, device=self.device)
        
        # For detailed metrics, accumulate on CPU (unavoidable for complex operations)
        class_metrics = {}
        total_tp = total_fp = total_fn = 0

        for images, masks in val_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            predictions = self.model(images)

            # GPU operations
            loss = self.compute_loss(predictions, masks)
            miou = torch.tensor(self.compute_metrics(predictions, masks), device=self.device)
            
            total_loss += loss.detach()
            total_miou += miou
            num_batches += 1

            # CPU operations for detailed metrics (keep minimal)
            if self.num_classes == 1:
                pred_masks = (torch.sigmoid(predictions).cpu().numpy() > 0.5).squeeze(1)
            else:
                pred_masks = torch.argmax(predictions, dim=1).cpu().numpy()
            gt_masks = masks.cpu().numpy()

            batch_metrics = batch_multiclass_metrics(pred_masks, gt_masks)
            for class_id, (tp, fp, fn) in batch_metrics.items():
                if class_id not in class_metrics:
                    class_metrics[class_id] = {"tp": 0, "fp": 0, "fn": 0}
                class_metrics[class_id]["tp"] += tp
                class_metrics[class_id]["fp"] += fp
                class_metrics[class_id]["fn"] += fn
                total_tp += tp
                total_fp += fp
                total_fn += fn

        # Reduce basic metrics across processes (single collective op)
        if self.world_size > 1:

            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_miou, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
            
            total_loss /= self.world_size
            total_miou /= self.world_size
            num_batches /= self.world_size

        avg_loss = (total_loss / num_batches).item()
        avg_miou = (total_miou / num_batches).item()

        # Only gather detailed metrics if main process (reduce communication)
        if self.world_size > 1 and self.is_main_process:
            # Gather detailed metrics only from main process of each node
            all_class_metrics = [None] * self.world_size
            all_total_metrics = [None] * self.world_size
            dist.all_gather_object(all_class_metrics, class_metrics)
            dist.all_gather_object(all_total_metrics, (total_tp, total_fp, total_fn))
            
            # Aggregate detailed metrics
            aggregated_class_metrics = {}
            for class_metrics_single in all_class_metrics:
                for class_id, metrics in class_metrics_single.items():
                    if class_id not in aggregated_class_metrics:
                        aggregated_class_metrics[class_id] = {"tp": 0, "fp": 0, "fn": 0}
                    aggregated_class_metrics[class_id]["tp"] += metrics["tp"]
                    aggregated_class_metrics[class_id]["fp"] += metrics["fp"]
                    aggregated_class_metrics[class_id]["fn"] += metrics["fn"]
            
            class_metrics = aggregated_class_metrics
            total_tp = sum(metrics[0] for metrics in all_total_metrics)
            total_fp = sum(metrics[1] for metrics in all_total_metrics)
            total_fn = sum(metrics[2] for metrics in all_total_metrics)

        # Compute precision/recall/f1
        class_precision_recall = {}
        for class_id, metrics in class_metrics.items():
            tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            class_precision_recall[class_id] = {"precision": precision, "recall": recall, "f1": f1}

        total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0

        return avg_loss, avg_miou, total_precision, total_recall, total_f1, class_precision_recall

    def train(self, train_loader, val_loader, num_epochs):
        best_val_loss = float("inf")
        no_improve = 0
        
        for epoch in tqdm(
            range(num_epochs),
            disable=not (self.logger.isEnabledFor(logging.DEBUG) and self.is_main_process),
        ):
            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            train_loss, train_miou = self.train_epoch(train_loader)
            val_loss, val_miou, val_precision, val_recall, val_f1, class_metrics = self.validate(val_loader)
            # TOD DO: REDUCING/aggregation/gathering... should be done here and only the main process should log and only the main process who should get the copy

            # Only main process handles scheduling, logging, and checkpointing
            if self.is_main_process:
                self.scheduler.step(val_loss)
                
                # Logging and checkpointing code...
                if self.writer is not None:
                    self.writer.add_scalar("Train/Loss", train_loss, epoch)
                    self.writer.add_scalar("Train/mIoU", train_miou, epoch)
                    self.writer.add_scalar("Validation/Loss", val_loss, epoch)
                    self.writer.add_scalar("Validation/mIoU", val_miou, epoch)
                    self.writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]["lr"], epoch)
                    self.writer.add_scalar("Validation/Total_F1", val_f1, epoch)
                    self.writer.add_scalar("Validation/Total_Precision", val_precision, epoch)
                    self.writer.add_scalar("Validation/Total_Recall", val_recall, epoch)

                    for class_id, metrics in class_metrics.items():
                        self.writer.add_scalar(f"Validation/Class_{class_id}_Precision", metrics["precision"], epoch)
                        self.writer.add_scalar(f"Validation/Class_{class_id}_Recall", metrics["recall"], epoch)
                        self.writer.add_scalar(f"Validation/Class_{class_id}_F1", metrics["f1"], epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.checkpoint_path:
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_miou": val_miou,
                            "val_loss": val_loss,
                        }, self.checkpoint_path)
                    no_improve = 0
                else:
                    no_improve += 1

                self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}, Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                self.logger.info(f"Total F1: {val_f1:.4f}, Total Precision: {val_precision:.4f}, Total Recall: {val_recall:.4f}")

                for class_id, metrics in class_metrics.items():
                    self.logger.info(f"Class {class_id}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

            # Broadcast early stopping decision (single collective op)
            if self.world_size > 1:
                should_early_stop = torch.tensor([no_improve >= self.earlystoping_patience and self.config["early_stopping"]["enabled"]], dtype=torch.bool, device=self.device)
                dist.broadcast(should_early_stop, src=0)
                if should_early_stop.item():
                    if self.is_main_process:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if no_improve >= self.earlystoping_patience and self.config["early_stopping"]["enabled"]:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            