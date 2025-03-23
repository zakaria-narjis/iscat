import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceLoss, DiceCELoss, TverskyLoss
from monai.metrics import MeanIoU
from monai.networks.utils import one_hot
from enum import Enum
from src.data_processing.utils import Utils
from tqdm import tqdm
import logging
from src.metrics import count_matching_particles,batch_multiclass_metrics

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict,
        experiment_dir: str,
        class_weights=None,
        writer: SummaryWriter = None,
        verbose: bool = True
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
        """
        self.num_classes = config['num_classes']
        self.model = model.to(device)
        self.device = device
        self.loss_type = config['loss']['loss_type']
        self.class_weights = class_weights
        self.config = config
        self.earlystoping_patience = config['early_stopping']['patience']
        self.experiment_dir = experiment_dir

        # Configure logging
        self.logger = logging.getLogger(__name__)
        log_level = logging.DEBUG if verbose else logging.WARNING
        logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

        # Fine-tuning configuration
        self.fine_tuning_enabled = config.get('fine_tuning', {}).get('enabled', False)
        self.freeze_encoder = config.get('fine_tuning', {}).get('freeze_encoder', False)
        self.encoder_lr_factor = config.get('fine_tuning', {}).get('encoder_lr_factor', 0.1)
        self.progressive_unfreeze = config.get('fine_tuning', {}).get('progressive_unfreeze', {}).get('enabled', False)
        self.epochs_per_layer = config.get('fine_tuning', {}).get('progressive_unfreeze', {}).get('epochs_per_layer', 5)
        
        # List of encoder modules (to be identified in the model)
        self.encoder_modules = []
        
        # Apply fine-tuning settings if enabled
        if self.fine_tuning_enabled:
            self._setup_fine_tuning()

        # Initialize loss function
        self.loss = self._initialize_loss()

        # Initialize metrics
        self.miou_metric = MeanIoU(include_background=True, reduction="mean")
        
        # Initialize optimizer with different learning rates for encoder and decoder if needed
        self.optimizer = self._initialize_optimizer()
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=self.config['scheduler']['parameters']['factor'], 
            patience=self.config['scheduler']['parameters']['patience'], 
        )

        self.writer = writer
        self.checkpoint_path = os.path.join(experiment_dir, 'best_model.pth')
        
    def _setup_fine_tuning(self):
        """
        Set up fine-tuning by identifying encoder layers and applying freezing if needed.
        """
        # Identify encoder modules in U-Net
        self.logger.info("Setting up fine-tuning...")
        
        # Get a list of encoder modules based on model architecture
        # For U-Net, the encoder typically includes Conv1 through Conv5
        if hasattr(self.model, 'Conv1'):
            self.encoder_modules = [
                ('Conv1', self.model.Conv1),
                ('Conv2', self.model.Conv2),
                ('Conv3', self.model.Conv3),
                ('Conv4', self.model.Conv4),
                ('Conv5', self.model.Conv5) if hasattr(self.model, 'Conv5') else None
            ]
            self.encoder_modules = [m for m in self.encoder_modules if m is not None]
        
        # Log the identified encoder modules
        self.logger.info(f"Identified encoder modules: {[name for name, _ in self.encoder_modules]}")
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            self._freeze_encoder()
            self.logger.info("Encoder frozen for fine-tuning")
    
    def _freeze_encoder(self, up_to_layer=None):
        """
        Freeze encoder layers up to a specified layer (inclusive).
        If up_to_layer is None, freeze all encoder layers.
        
        Args:
            up_to_layer (int): Index of the last layer to freeze (0-based).
        """
        # Determine how many layers to freeze
        layers_to_freeze = self.encoder_modules if up_to_layer is None else self.encoder_modules[:up_to_layer+1]
        
        # Freeze selected layers
        for name, module in layers_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
            self.logger.info(f"Froze parameters in {name}")
    
    def _unfreeze_layer(self, layer_idx):
        """
        Unfreeze a specific encoder layer.
        
        Args:
            layer_idx (int): Index of the layer to unfreeze (0-based).
        """
        if layer_idx < len(self.encoder_modules):
            name, module = self.encoder_modules[layer_idx]
            for param in module.parameters():
                param.requires_grad = True
            self.logger.info(f"Unfroze parameters in {name}")
    
    def _initialize_optimizer(self):
        """
        Initialize optimizer with separate parameter groups for encoder and decoder
        if fine-tuning is enabled.
        """
        base_lr = self.config['optimizer']['parameters']['lr']
        
        if not self.fine_tuning_enabled or not self.encoder_modules:
            # Simple case: use the same learning rate for all parameters
            return optim.Adam(self.model.parameters(), lr=base_lr)
        
        # Create parameter groups with different learning rates
        encoder_params = []
        for _, module in self.encoder_modules:
            encoder_params.extend(module.parameters())
        
        # Separate encoder and decoder parameters
        encoder_params_set = set(encoder_params)
        decoder_params = [p for p in self.model.parameters() if p not in encoder_params_set]
        
        # Define parameter groups with different learning rates
        param_groups = [
            {'params': encoder_params, 'lr': base_lr * self.encoder_lr_factor},
            {'params': decoder_params, 'lr': base_lr}
        ]
        
        self.logger.info(f"Initialized optimizer with encoder LR={base_lr * self.encoder_lr_factor}, decoder LR={base_lr}")
        return optim.Adam(param_groups)

    def _initialize_loss(self):
        if self.num_classes == 1:
            if self.loss_type == "crossentropy":
                return nn.BCEWithLogitsLoss()
            elif self.loss_type == "dice":
                return DiceLoss(sigmoid=True, squared_pred=True, batch=True, reduction="mean")
            elif self.loss_type == "dicece":
                return DiceCELoss(sigmoid=True, 
                                  squared_pred=True, 
                                  batch=True, 
                                  reduction="mean", 
                                  weight=self.class_weights,
                                  lambda_ce=self.config['loss']['parameters']['lambda_ce'],
                                  lambda_dice=self.config['loss']['parameters']['lambda_dice']
                                 )
            elif self.loss_type == "tversky":
                self.logger.info("Using Tversky Loss ")
                return TverskyLoss(sigmoid=True, 
                                   batch=True, 
                                   reduction="mean", 
                                   alpha=self.config['loss']['parameters']['alpha'], 
                                   beta=self.config['loss']['parameters']['beta'],
                                   )
            else:
                raise ValueError(f"Invalid loss type: {self.loss_type}")
        else:
            if self.loss_type == "crossentropy":
                self.logger.info("Using CrossEntropy Loss ")
                return nn.CrossEntropyLoss(weight=self.class_weights)
            elif self.loss_type == "dice":
                self.logger.info("Using Dice Loss ")
                return DiceLoss(softmax=True, squared_pred=True, batch=True, reduction="mean",include_background=False)
            elif self.loss_type == "tversky":
                self.logger.info("Using Tversky Loss ")
                return TverskyLoss(softmax=True, 
                                   batch=True, 
                                   reduction="mean", 
                                   alpha=self.config['loss']['parameters']['alpha'], 
                                   beta=self.config['loss']['parameters']['beta'],
                                   include_background=False)
            elif self.loss_type == "dicece":
                self.logger.info("Using Dice CrossEntropy Loss ")
                return DiceCELoss(softmax=True, squared_pred=True, batch=True, reduction="mean", weight=self.class_weights)
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
            targets =  targets.unsqueeze(1)     
        if self.num_classes>1:
            targets = one_hot(targets, num_classes=self.num_classes, dim=1)
        return self.loss(predictions, targets)

    def compute_metrics(self, predictions, targets):
        if len(targets.shape) == 3:
            targets =  targets.unsqueeze(1)     
        if self.num_classes>1:
            pred_one_hot = one_hot(predictions.argmax(dim=1, keepdim=True), num_classes=self.num_classes) # [B, N, H, W]
            target_one_hot = one_hot(targets, num_classes=self.num_classes) # [B, N, H, W]
        else:
            pred_one_hot = torch.sigmoid(predictions) > 0.5 # [B, 1, H, W]
            pred_one_hot = one_hot(pred_one_hot, num_classes=2)
            target_one_hot = one_hot(targets, num_classes=2) 

        metric = self.miou_metric(pred_one_hot, target_one_hot) 
        return metric.nanmean().item()

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_miou = 0.0

        # Progressive unfreezing: check if it's time to unfreeze a layer
        if self.fine_tuning_enabled and self.progressive_unfreeze and self.freeze_encoder:
            # Calculate which layer to unfreeze based on current epoch
            layer_to_unfreeze = epoch // self.epochs_per_layer
            if layer_to_unfreeze < len(self.encoder_modules):
                # Re-freeze all encoder layers first
                self._freeze_encoder()
                # Unfreeze layers up to the current one
                for i in range(layer_to_unfreeze + 1):
                    self._unfreeze_layer(i)
                    
                # Log the unfreezing action
                self.logger.info(f"Epoch {epoch}: Progressive unfreezing up to layer {layer_to_unfreeze}")
                
                # Save the current unfreeze state for reference
                unfreeze_state = {
                    'epoch': epoch,
                    'unfrozen_layers': [self.encoder_modules[i][0] for i in range(layer_to_unfreeze + 1)]
                }
                torch.save(unfreeze_state, os.path.join(self.experiment_dir, f'unfreeze_state_epoch_{epoch}.pth'))

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(self.device), masks.to(self.device)    
            if len(masks.shape) == 3:
                 masks =  masks.unsqueeze(1)        
            # Forward pass and loss computation on GPU
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.compute_loss(predictions, masks)
            loss.backward()
            self.optimizer.step()
            
            # Loss and mIoU can stay on GPU if they're simple operations
            total_loss += loss.item()
            total_miou += self.compute_metrics(predictions, masks)
            

        # Compute final metrics once at the end
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_miou = total_miou / n_batches

        return avg_loss, avg_miou

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_miou = 0.0
        
        # Initialize dictionaries for class-specific metrics
        class_metrics = {}
        # Initialize total particle metric counters
        total_tp = total_fp = total_fn = 0
        
        for images, masks in val_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            # Forward pass on GPU
            predictions = self.model(images)
            
            # Compute loss and mIoU on GPU
            total_loss += self.compute_loss(predictions, masks).item()
            total_miou += self.compute_metrics(predictions, masks)
            
            # Convert predictions and masks to CPU numpy arrays
            if self.num_classes == 1:
                pred_masks = torch.sigmoid(predictions).cpu().numpy() > 0.5 # [B, 1, H, W]
                pred_masks = pred_masks.squeeze(1) # [B, H, W]
            else:
                pred_masks = torch.argmax(predictions, dim=1).cpu().numpy() # [B, H, W]
            gt_masks = masks.cpu().numpy()
            
            # Process entire batch at once
            batch_metrics = batch_multiclass_metrics(pred_masks, gt_masks)
            
            # Update metrics for each class
            for class_id, (tp, fp, fn) in batch_metrics.items():
                if class_id not in class_metrics:
                    class_metrics[class_id] = {'tp': 0, 'fp': 0, 'fn': 0}
                class_metrics[class_id]['tp'] += tp
                class_metrics[class_id]['fp'] += fp
                class_metrics[class_id]['fn'] += fn
                
                # Update total counters
                total_tp += tp
                total_fp += fp
                total_fn += fn
        
        # Compute final metrics
        n_batches = len(val_loader)
        avg_loss = total_loss / n_batches
        avg_miou = total_miou / n_batches
        
        # Compute class-specific precision and recall
        class_precision_recall = {}
        for class_id, metrics in class_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            class_precision_recall[class_id] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
        
        # Compute total precision and recall
        total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
        
        return avg_loss, avg_miou, total_precision, total_recall, total_f1, class_precision_recall

    def train(self, train_loader, val_loader, num_epochs):
        best_val_loss = float('inf')
        no_improve = 0
        
        # Log fine-tuning configuration
        if self.fine_tuning_enabled:
            self.logger.info(f"Fine-tuning enabled: Encoder freeze={self.freeze_encoder}, "
                            f"Encoder LR factor={self.encoder_lr_factor}, "
                            f"Progressive unfreeze={self.progressive_unfreeze}")
            if self.progressive_unfreeze:
                self.logger.info(f"Progressive unfreezing: {self.epochs_per_layer} epochs per layer")
        
        for epoch in tqdm(range(num_epochs), disable=not self.logger.isEnabledFor(logging.DEBUG)):
            train_loss, train_miou = self.train_epoch(train_loader, epoch)
            val_loss, val_miou, val_precision, val_recall, val_f1, class_metrics = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            if self.writer is not None:
                # Log general metrics
                self.writer.add_scalar('Train/Loss', train_loss, epoch)
                self.writer.add_scalar('Train/mIoU', train_miou, epoch)
                self.writer.add_scalar('Validation/Loss', val_loss, epoch)
                self.writer.add_scalar('Validation/mIoU', val_miou, epoch)
                
                # If fine-tuning is enabled with different learning rates, log them separately
                if self.fine_tuning_enabled and len(self.optimizer.param_groups) > 1:
                    self.writer.add_scalar('Learning Rate/Encoder', self.optimizer.param_groups[0]['lr'], epoch)
                    self.writer.add_scalar('Learning Rate/Decoder', self.optimizer.param_groups[1]['lr'], epoch)
                else:
                    self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)
                
                # Log total precision and recall
                self.writer.add_scalar('Validation/Total_F1', val_f1, epoch)
                self.writer.add_scalar('Validation/Total_Precision', val_precision, epoch)
                self.writer.add_scalar('Validation/Total_Recall', val_recall, epoch)
                
                # Log class-specific metrics
                for class_id, metrics in class_metrics.items():
                    self.writer.add_scalar(f'Validation/Class_{class_id}_Precision', 
                                        metrics['precision'], epoch)
                    self.writer.add_scalar(f'Validation/Class_{class_id}_Recall', 
                                        metrics['recall'], epoch)
                    self.writer.add_scalar(f'Validation/Class_{class_id}_F1', 
                                        metrics['f1'], epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_miou': val_miou,
                    'val_loss': val_loss,
                    'fine_tuning_config': {
                        'fine_tuning_enabled': self.fine_tuning_enabled,
                        'freeze_encoder': self.freeze_encoder,
                        'progressive_unfreeze': self.progressive_unfreeze,
                        'encoder_lr_factor': self.encoder_lr_factor
                    } if self.fine_tuning_enabled else None
                }, self.checkpoint_path)
                no_improve = 0
                self.logger.info(f"Saved best model at epoch {epoch+1} with val_loss={val_loss:.4f}")
            else:
                no_improve += 1
            
            # Log metrics for each class
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                            f"Train Loss: {train_loss:.4f}, "
                            f"Train mIoU: {train_miou:.4f}, "
                            f"Val Loss: {val_loss:.4f}, "
                            f"Val mIoU: {val_miou:.4f}")
            
            # Log learning rates for encoder and decoder if fine-tuning is enabled
            if self.fine_tuning_enabled and len(self.optimizer.param_groups) > 1:
                self.logger.info(f"LR Encoder: {self.optimizer.param_groups[0]['lr']:.2e}, "
                                f"LR Decoder: {self.optimizer.param_groups[1]['lr']:.2e}")
            else:
                self.logger.info(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            self.logger.info(f"Total F1: {val_f1:.4f}, "
                            f"Total Precision: {val_precision:.4f}, "
                            f"Total Recall: {val_recall:.4f}")
            
            for class_id, metrics in class_metrics.items():
                self.logger.info(f"Class {class_id}: "
                                f"Precision: {metrics['precision']:.4f}, "
                                f"Recall: {metrics['recall']:.4f}, "
                                f"F1: {metrics['f1']:.4f}")
            
            if no_improve >= self.earlystoping_patience and self.config['early_stopping']['enabled']:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Return best validation metrics
        self.logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        return best_val_loss