import torch
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, num_classes=1,in_channels=1):
        super(ResNet18, self).__init__()
        self.resnet = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", weights=None
        )
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
         # Remove the original fully connected layer
        self.size_head = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, self.resnet.fc.in_features),
            nn.ReLU(),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )
        self.resnet.fc = nn.Identity() 
    def forward(self, x):
        features = self.resnet(x)
        size_output = self.size_head(features)       
        return size_output
