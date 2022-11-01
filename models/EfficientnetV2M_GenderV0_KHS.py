
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models


class EfficientnetV2M_GenderV0_KHS(nn.Module):
    def __init__(self):
        super(EfficientnetV2M_GenderV0_KHS, self).__init__()
        self.backborn = models.efficientnet_v2_m(weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        for p in self.backborn.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1),)

    def forward(self, x):
        x = self.backborn(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x
