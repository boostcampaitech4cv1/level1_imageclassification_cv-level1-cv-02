
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models


class ResNext_AgeV0_KHS(nn.Module):
    def __init__(self, number_of_classes: int):
        super(ResNext_AgeV0_KHS, self).__init__()
        self.backborn = torch.hub.load(
            'pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
        # for p in self.backborn.parameters():
        #     p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(512, number_of_classes)
        )

    def forward(self, x):
        x = self.backborn(x)
        x = self.classifier(x)
        return x
