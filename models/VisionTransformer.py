import torch.nn as nn
import timm

class VIT(nn.Module):
    def __init__(self, num_classes):
        super(VIT,self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes = num_classes) #pretrained=True

    def forward(self, x):
        x = self.model(x)
        return x