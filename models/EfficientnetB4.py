
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models


class EfficientnetB4(nn.Module):
    def __init__(self):
        super(EfficientnetB4, self).__init__()
        self.number_of_class = 18
        self.backborn = models.efficientnet_v2_m(
            weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, self.number_of_class)
        )

    def forward(self, x):
        x = self.backborn(x)
        x = self.classifier(x)
        return x
