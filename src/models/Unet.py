import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, init_features=32):
        super().__init__()  # Properly initialize the nn.Module parent class
        
        # Load the pretrained model and modify the final layer
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', 
                               in_channels=in_channels, 
                               out_channels=1, 
                               init_features=init_features, 
                               pretrained=False)
        
        # Replace the final convolution layer to match number of classes
        model.conv = nn.Conv2d(init_features, num_classes, kernel_size=1)
        
        self.model = model
    
    def forward(self, x):
        return self.model(x)
