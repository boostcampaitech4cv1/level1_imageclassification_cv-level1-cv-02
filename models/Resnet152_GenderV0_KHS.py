
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models


class Resnet152_GenderV0_KHS(nn.Module):
    def __init__(self: int):
        super(Resnet152_GenderV0_KHS, self).__init__()
        self.backborn = models.resnet152(
            weights=models.ResNet152_Weights.IMAGENET1K_V1)
        # for p in self.backborn.parameters():
        #     p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.backborn(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x
