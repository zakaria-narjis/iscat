from src.models.Unet_networks import AttU_Net, R2AttU_Net, R2U_Net
from src.models.Unet import UNet
import yaml
import os
import torch
from src.data_processing.utils import Utils

class SegInference:
    def __init__(self, experiment_path, device):
        """
        Initialize the inference class.

        Args:
            experiment_path (str): Path to the experiment directory.
            device (torch.device): Device to load the model on.
        """
        model_path = os.path.join(experiment_path, "best_model.pth")
        self.config = self.load_config(experiment_path)
        self.model = self.load_model(model_path, self.config, device)
        self.device = device

    def load_config(self, experiment_path):
        """
        Load the configuration file of an experiment.

        Args:
            experiment_path (str): Path to the experiment directory.

        Returns:
            dict: Configuration dictionary.
        """
        config_path = os.path.join(experiment_path, "config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_model(self,model_path,config, device):
        """
        Load a trained model from a checkpoint file.

        Args:
            experiment_path (str): Path to the experiment directory.
            device (torch.device): Device to load the model on.

        Returns:
            torch.nn.Module: Trained model.
        """
        checkpoint = torch.load(model_path, weights_only=False)
        model_name = config['model']['type']
        if model_name == 'U_Net':
            model = UNet(img_ch=config['model']['in_channels'], output_ch=config['model']['out_channels'])
        elif model_name == 'AttU_Net':
            model = AttU_Net(img_ch=config['model']['in_channels'], output_ch=config['model']['out_channels'])
        elif model_name == 'R2AttU_Net':
            model = R2AttU_Net(img_ch=config['model']['in_channels'], output_ch=config['model']['out_channels'])
        elif model_name == 'R2U_Net':
            model = R2U_Net(img_ch=config['model']['in_channels'], output_ch=config['model']['out_channels'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        model.eval()
        return model

    def predict(self, x):
        """
        Perform inference on an input image.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            np.ndarray: Predicted mask.
        """
        x = Utils.extract_averaged_frames(x, num_frames=self.config['model']['in_channels']) # Extract averaged frames
        x = Utils.z_score_normalize(x, mean=x.mean(), std=x.std()) # Normalize the image
        x = torch.Tensor(x)
        if x.dim() == 3:
            x = x.unsqueeze(0) # Add batch dimension       
        with torch.no_grad():
            x = x.to(self.device)
            pred = self.model(x) # shape: (B, C, H, W)
            pred = torch.argmax(pred, dim=1) # shape: (B, H, W)
            if x.dim() == 3:
                pred = pred.squeeze(0) # Remove batch dimension
            pred = pred.cpu().numpy()

        return pred