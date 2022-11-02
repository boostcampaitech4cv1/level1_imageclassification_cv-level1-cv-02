
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models


class ResNext_GenderV0_SAWOL_mask(nn.Module):
    def __init__(self: int):
        super(ResNext_GenderV0_SAWOL_mask, self).__init__()
        self.backborn = torch.hub.load(
            'pytorch/vision:v0.10.0', 'resnext50_32x4d', weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(512, 3),
        )

    def forward(self, x):
        x = self.backborn(x)
        x = self.classifier(x)
        return x
    
class ResNext_GenderV0_SAWOL_age(nn.Module):
    def __init__(self: int):
        super(ResNext_GenderV0_SAWOL_age, self).__init__()
        self.backborn = torch.hub.load(
            'pytorch/vision:v0.10.0', 'resnext50_32x4d', weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.backborn(x)
        x = self.classifier(x)
        return x

class ResNext_GenderV0_SAWOL_gender(nn.Module):
    def __init__(self: int):
        super(ResNext_GenderV0_SAWOL_gender, self).__init__()
        self.backborn = torch.hub.load(
            'pytorch/vision:v0.10.0', 'resnext50_32x4d', weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.backborn(x)
        x = self.classifier(x)
        return x