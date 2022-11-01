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

class EfficientBackBone(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights='DEFAULT')
        self.backbone.fc = nn.Sequential(
            nn.Dropout2d(0.3, inplace=True),
            nn.Linear(1536, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        return x

class ViTBackBone(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.vit_b_16(weights='DEFAULT')
        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x