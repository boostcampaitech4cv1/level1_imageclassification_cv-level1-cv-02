
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models


class VIT_V0_KHS(nn.Module):
    def __init__(self, is_freeze:bool = True):
        super(VIT_V0_KHS, self).__init__()
        self.backborn = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_V1)
        if(is_freeze == True):
            for p in self.backborn.parameters():
                p.requires_grad = False
        self.backborn.heads = nn.Sequential(nn.Linear(768, 18))

    def forward(self, x):
        x = self.backborn(x)
        return x
