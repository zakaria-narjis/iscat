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
import torch.distributed as dist

class Trainer:
    def __init__(self, model, device, config, experiment_dir, class_weights=None, writer=None, verbose=True, rank=0, world_size=1):
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

        self.logger = logging.getLogger(__name__)
        log_level = logging.DEBUG if verbose and self.is_main_process else logging.WARNING
        logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

        self.loss = self._initialize_loss()
        self.miou_metric = MeanIoU(include_background=True, reduction="mean")
        self.optimizer = optim.Adam(model.parameters(), lr=config["optimizer"]["parameters"]["lr"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=config["scheduler"]["parameters"]["factor"], patience=config["scheduler"]["parameters"]["patience"])

        self.writer = writer
        self.checkpoint_path = os.path.join(experiment_dir, "best_model.pth") if experiment_dir else None

    def _initialize_loss(self):
        if self.num_classes == 1:
            if self.loss_type == "crossentropy":
                return nn.BCEWithLogitsLoss()
            elif self.loss_type == "dice":
                return DiceLoss(sigmoid=True, squared_pred=True, batch=True, reduction="mean")
            elif self.loss_type == "dicece":
                return DiceCELoss(sigmoid=True, squared_pred=True, batch=True, reduction="mean", weight=self.class_weights, lambda_ce=self.config["loss"]["parameters"]["lambda_ce"], lambda_dice=self.config["loss"]["parameters"]["lambda_dice"])
            elif self.loss_type == "tversky":
                return TverskyLoss(sigmoid=True, batch=True, reduction="mean", alpha=self.config["loss"]["parameters"]["alpha"], beta=self.config["loss"]["parameters"]["beta"])
        else:
            if self.loss_type == "crossentropy":
                return nn.CrossEntropyLoss(weight=self.class_weights)
            elif self.loss_type == "dice":
                return DiceLoss(softmax=True, squared_pred=True, batch=True, reduction="mean", include_background=False)
            elif self.loss_type == "tversky":
                return TverskyLoss(softmax=True, batch=True, reduction="mean", alpha=self.config["loss"]["parameters"]["alpha"], beta=self.config["loss"]["parameters"]["beta"], include_background=False)
            elif self.loss_type == "dicece":
                return DiceCELoss(softmax=True, squared_pred=True, batch=True, reduction="mean", weight=self.class_weights)

    def compute_loss(self, predictions, targets):
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
            pred_one_hot = one_hot(predictions.argmax(dim=1, keepdim=True), num_classes=self.num_classes)
            target_one_hot = one_hot(targets, num_classes=self.num_classes)
        else:
            pred_one_hot = torch.sigmoid(predictions) > 0.5
            pred_one_hot = one_hot(pred_one_hot, num_classes=2)
            target_one_hot = one_hot(targets, num_classes=2)
        metric = self.miou_metric(pred_one_hot, target_one_hot)
        return metric.nanmean().item()

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        total_miou = torch.tensor(0.0, device=self.device)
        num_batches = torch.tensor(0.0, device=self.device)
        for images, masks in train_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.compute_loss(predictions, masks)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach()
            miou = torch.tensor(self.compute_metrics(predictions, masks), device=self.device)
            total_miou += miou
            num_batches += 1
        return total_loss.item(), total_miou.item(), num_batches.item()

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        total_miou = torch.tensor(0.0, device=self.device)
        num_batches = torch.tensor(0.0, device=self.device)
        class_metrics = {}
        total_tp = total_fp = total_fn = 0

        for images, masks in val_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            predictions = self.model(images)
            loss = self.compute_loss(predictions, masks)
            miou = torch.tensor(self.compute_metrics(predictions, masks), device=self.device)
            total_loss += loss.detach()
            total_miou += miou
            num_batches += 1

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

        return total_loss.item(), total_miou.item(), num_batches.item(), total_tp, total_fp, total_fn, class_metrics

    def train(self, train_loader, val_loader, num_epochs):
        best_val_loss = float("inf")
        no_improve = 0

        for epoch in tqdm(range(num_epochs), disable=not (self.logger.isEnabledFor(logging.DEBUG) and self.is_main_process)):
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            train_loss, train_miou, train_batches = self.train_epoch(train_loader)
            val_loss, val_miou, val_batches, total_tp, total_fp, total_fn, class_metrics = self.validate(val_loader)

            # Distributed reduction starts here
            tensors = [
                torch.tensor(train_loss, device=self.device),
                torch.tensor(train_miou, device=self.device),
                torch.tensor(train_batches, device=self.device),
                torch.tensor(val_loss, device=self.device),
                torch.tensor(val_miou, device=self.device),
                torch.tensor(val_batches, device=self.device),
                torch.tensor(total_tp, device=self.device),
                torch.tensor(total_fp, device=self.device),
                torch.tensor(total_fn, device=self.device)
            ]

            gathered = [[torch.zeros_like(t) for _ in range(self.world_size)] for t in tensors]
            for i, t in enumerate(tensors):
                dist.all_gather(gathered[i], t)

            class_metrics_list = [None for _ in range(self.world_size)]
            dist.all_gather_object(class_metrics_list, class_metrics)

            if self.is_main_process:
                gathered_values = [sum([v.item() for v in group]) for group in gathered]
                train_loss = gathered_values[0] / gathered_values[2]
                train_miou = gathered_values[1] / gathered_values[2]
                val_loss = gathered_values[3] / gathered_values[5]
                val_miou = gathered_values[4] / gathered_values[5]
                total_tp, total_fp, total_fn = gathered_values[6], gathered_values[7], gathered_values[8]

                merged_class_metrics = {}
                for metrics in class_metrics_list:
                    for class_id, stats in metrics.items():
                        if class_id not in merged_class_metrics:
                            merged_class_metrics[class_id] = {"tp": 0, "fp": 0, "fn": 0}
                        merged_class_metrics[class_id]["tp"] += stats["tp"]
                        merged_class_metrics[class_id]["fp"] += stats["fp"]
                        merged_class_metrics[class_id]["fn"] += stats["fn"]

                class_metrics = merged_class_metrics
                total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0

                self.scheduler.step(val_loss)
                if self.writer is not None:
                    self.writer.add_scalar("Train/Loss", train_loss, epoch)
                    self.writer.add_scalar("Train/mIoU", train_miou, epoch)
                    self.writer.add_scalar("Validation/Loss", val_loss, epoch)
                    self.writer.add_scalar("Validation/mIoU", val_miou, epoch)
                    self.writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]["lr"], epoch)
                    self.writer.add_scalar("Validation/Total_F1", total_f1, epoch)
                    self.writer.add_scalar("Validation/Total_Precision", total_precision, epoch)
                    self.writer.add_scalar("Validation/Total_Recall", total_recall, epoch)

                    for class_id, metrics in class_metrics.items():
                        self.writer.add_scalar(f"Validation/Class_{class_id}_Precision", metrics["tp"] / (metrics["tp"] + metrics["fp"] + 1e-8), epoch)
                        self.writer.add_scalar(f"Validation/Class_{class_id}_Recall", metrics["tp"] / (metrics["tp"] + metrics["fn"] + 1e-8), epoch)
                        p = metrics["tp"] / (metrics["tp"] + metrics["fp"] + 1e-8)
                        r = metrics["tp"] / (metrics["tp"] + metrics["fn"] + 1e-8)
                        f1 = 2 * (p * r) / (p + r + 1e-8)
                        self.writer.add_scalar(f"Validation/Class_{class_id}_F1", f1, epoch)

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
                self.logger.info(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}, Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")
                self.logger.info(f"Total F1: {total_f1:.4f}, Total Precision: {total_precision:.4f}, Total Recall: {total_recall:.4f}")
                self.logger.info(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, Early Stopping No Improve: {no_improve}/{self.earlystoping_patience}")

            should_stop = torch.tensor([no_improve >= self.earlystoping_patience and self.config["early_stopping"]["enabled"]], dtype=torch.bool, device=self.device)
            dist.broadcast(should_stop, src=0)
            if should_stop.item():
                break
