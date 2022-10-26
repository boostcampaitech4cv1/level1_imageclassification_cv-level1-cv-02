import torch
import torch.nn as nn
from torchvision import models


class ResnetBackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True) # 예전 버전이라 weights="DEFAULT" 사용불가
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 3)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        return x