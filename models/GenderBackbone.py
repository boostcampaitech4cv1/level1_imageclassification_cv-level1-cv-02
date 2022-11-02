
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


class Resnet152_GenderV0_KHS(nn.Module):
    def __init__(self):
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

class ResNext_GenderV0_KHS(nn.Module):
    def __init__(self):
        super(ResNext_GenderV0_KHS, self).__init__()
        self.backborn = torch.hub.load(
            'pytorch/vision:v0.10.0', 'resnext50_32x4d', weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        # for p in self.backborn.parameters():
        #     p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.backborn(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x

class VIT_GenderV0_KHS(nn.Module):
    def __init__(self):
        super(VIT_GenderV0_KHS, self).__init__()
        self.backborn = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_V1)
        # for p in self.backborn.parameters():
        #     p.requires_grad = False
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