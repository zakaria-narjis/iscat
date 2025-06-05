import torch
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet18, self).__init__()
        self.resnet = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", weights=None
        )
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, self.resnet.fc.in_features),
            nn.ReLU(),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
