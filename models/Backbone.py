import torch.nn as nn
from torchvision import models


class ResnetBackBone(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights='DEFAULT')
        self.backbone.fc = nn.Linear(2048, num_classes)
        
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


class EfficientNet_V0_KHS(nn.Module):
    def __init__(self, is_freeze:bool = True):
        super(EfficientNet_V0_KHS, self).__init__()
        self.number_of_class = 18
        self.backborn = models.efficientnet_v2_m(
            weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        if(is_freeze == True):
            for p in self.backborn.parameters():
                p.requires_grad = False
        self.backborn.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, self.number_of_class)
        )

    def forward(self, x):
        x = self.backborn(x)
        return x

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

class VIT_V1_KHS(nn.Module):
    def __init__(self, is_freeze:bool = True):
        super(VIT_V1_KHS, self).__init__()
        self.backborn = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        if(is_freeze == True):
            for p in self.backborn.parameters():
                p.requires_grad = False
        self.backborn.heads = nn.Sequential(nn.Linear(768, 18))

    def forward(self, x):
        x = self.backborn(x)
        return x

class VIT_V2_KHS(nn.Module):
    def __init__(self, is_freeze:bool = True):
        super(VIT_V2_KHS, self).__init__()
        self.backborn = models.vit_l_16(weights = models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        if(is_freeze == True):
            for p in self.backborn.parameters():
                p.requires_grad = False
        self.backborn.heads = nn.Sequential(nn.Linear(1024, 18))

    def forward(self, x):
        x = self.backborn(x)
        return x

class VIT_V3_KHS(nn.Module):
    def __init__(self, is_freeze:bool = True):
        super(VIT_V3_KHS, self).__init__()
        self.backborn = models.vit_l_16(weights = models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        if(is_freeze == True):
            for p in self.backborn.parameters():
                p.requires_grad = False
        self.backborn.heads = nn.Sequential(nn.Linear(1024, 18))

    def forward(self, x):
        x = self.backborn(x)
        return x