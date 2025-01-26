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

class UNetBoundaryAware(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, init_features=64,pretrained=False):
        super(UNetBoundaryAware, self).__init__()
        # Define your encoder and decoder here
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', 
                               in_channels=img_ch, 
                               out_channels=1, 
                               init_features=init_features, 
                               pretrained=pretrained)
        self.model.conv = nn.Identity()
        self.segmentation_head = nn.Conv2d(init_features, output_ch, kernel_size=1)
        self.boundary_head = nn.Conv2d(init_features, 1, kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        seg_mask = self.segmentation_head(x)
        boundary_map = self.boundary_head(x)
        boundary_map = boundary_map.squeeze(1)
        return seg_mask, boundary_map