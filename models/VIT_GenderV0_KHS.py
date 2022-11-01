
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models


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
