import torch.nn as nn
from torchvision import models


class ResnetBackBone(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights='DEFAULT')
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        return x