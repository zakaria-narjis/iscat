import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, init_features=64,pretrained=False):
        super().__init__()  # Properly initialize the nn.Module parent class
        
        # Load the pretrained model and modify the final layer
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', 
                               in_channels=img_ch, 
                               out_channels=1, 
                               init_features=init_features, 
                               pretrained=pretrained)
        
        # Replace the final convolution layer to match number of classes
        model.conv = nn.Conv2d(init_features, output_ch, kernel_size=1)
        
        self.model = model
    
    def forward(self, x):
        return self.model(x)
