import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn, VGG19_BN_Weights
from torchvision.models import resnext50_32x4d, ResNet34_Weights, ResNeXt50_32X4D_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

import timm

import torch.nn.init as init


def initialize_weights(model):
    """
    Xavier uniform 분포로 모든 weight 를 초기화합니다.
    더 많은 weight 초기화 방법은 다음 문서에서 참고해주세요. https://pytorch.org/docs/stable/nn.init.html
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()



class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x


class Custom_Vgg_19bn(nn.Module):
    '''
        Vgg_19bn 
    '''
    def __init__(self, num_classes=18):
        super(Custom_Vgg_19bn, self).__init__()
        
        self.num_classes = num_classes
        self.model = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        output = self.model(x)
        return output
    
        
class Custom_Resnext50_32x4d(nn.Module):
    '''
        Resnext50
    '''
    def __init__(self,num_classes=18):
        super(Custom_Resnext50_32x4d, self).__init__()
        
        self.model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, num_classes),
        )
        
    def forward(self, x):
        output = self.model(x)
        return output
    
class Custom_Resnext50_32x4d_freeze(nn.Module):
    '''
        Resnext50 freeze
    '''
    
    def __init__(self, num_classes=18):
        super(Custom_Resnext50_32x4d_freeze, self).__init__()
        
        self.num_classes = num_classes
        self.model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(in_features=2048, out_features=self.num_classes)
        
        for name,params in self.model.named_parameters():
            if name.startswith('fc'):
                continue
            params.requires_grad = False
            
    def forward(self, x):
        outputs = self.model(x)
        return outputs
    
    
    
    
class Custom_Vit_b_16(nn.Module):
    '''
        Vit_b_16 
    '''
    def __init__(self, num_classes=18):
        super(Custom_Vit_b_16,self).__init__()
        
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads = nn.Linear(768, num_classes)
        initialize_weights(self.model.heads)                    # 가중치 초기화
        
        
        # Dropout 설정
        self.model.conv_proj.dropout = nn.Dropout(0.3)
        self.model.encoder.dropout = nn.Dropout(0.3)
        for i in self.model.encoder.layers:
            i.dropout = nn.Dropout(0.3)
        self.model.encoder.layers.encoder_layer_0.mlp[2] = nn.Dropout(0.3)
        self.model.encoder.layers.encoder_layer_0.mlp[4] = nn.Dropout(0.3)

        self.model.encoder.layers.encoder_layer_1.mlp[2] = nn.Dropout(0.3)
        self.model.encoder.layers.encoder_layer_1.mlp[4] = nn.Dropout(0.3)

        self.model.encoder.layers.encoder_layer_2.mlp[2] = nn.Dropout(0.3)
        self.model.encoder.layers.encoder_layer_2.mlp[4] = nn.Dropout(0.3)

        self.model.encoder.layers.encoder_layer_3.mlp[2] = nn.Dropout(0.3)
        self.model.encoder.layers.encoder_layer_3.mlp[4] = nn.Dropout(0.3)
        
        self.model.encoder.layers.encoder_layer_4.mlp[2] = nn.Dropout(0.3)
        self.model.encoder.layers.encoder_layer_4.mlp[4] = nn.Dropout(0.3)
        
        self.model.encoder.layers.encoder_layer_5.mlp[2] = nn.Dropout(0.3)
        self.model.encoder.layers.encoder_layer_5.mlp[4] = nn.Dropout(0.3)
        
        self.model.encoder.layers.encoder_layer_6.mlp[2] = nn.Dropout(0.3)
        self.model.encoder.layers.encoder_layer_6.mlp[4] = nn.Dropout(0.3)
        
        self.model.encoder.layers.encoder_layer_7.mlp[2] = nn.Dropout(0.3)
        self.model.encoder.layers.encoder_layer_7.mlp[4] = nn.Dropout(0.3)
        
        self.model.encoder.layers.encoder_layer_8.mlp[2] = nn.Dropout(0.3)
        self.model.encoder.layers.encoder_layer_8.mlp[4] = nn.Dropout(0.3)
        
        self.model.encoder.layers.encoder_layer_9.mlp[2] = nn.Dropout(0.3)
        self.model.encoder.layers.encoder_layer_9.mlp[4] = nn.Dropout(0.3)
        
        self.model.encoder.layers.encoder_layer_10.mlp[2] = nn.Dropout(0.3)
        self.model.encoder.layers.encoder_layer_10.mlp[4] = nn.Dropout(0.3)
        
        self.model.encoder.layers.encoder_layer_11.mlp[2] = nn.Dropout(0.3)
        self.model.encoder.layers.encoder_layer_11.mlp[4] = nn.Dropout(0.3)
        
        # 출력층 빼고 freezing
        for name, params in self.model.named_parameters():
            if name.startswith('heads'):
                continue
            params.requires_grad = False
        
        
    def forward(self, x):
        output = self.model(x)
        return output
    
    
class Custom_EfficientNet(nn.Module):
    '''
        EfficientNet_b2
    '''
    def __init__(self, num_classes=18):
        super(Custom_EfficientNet,self).__init__()
        
        self.model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1408, num_classes)
        )
        
    def forward(self, x):
        output = self.model(x)
        return output
    
    
class Simple_vit_b_16(nn.Module):
    def __init__(self,num_classes=18):
        super(Simple_vit_b_16,self).__init__()
        
        self.num_classes = num_classes
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads = nn.Linear(in_features=768, out_features=self.num_classes)
    
    def forward(self, x):
        output = self.model(x)
        return output
    
    
class Timm_vit(nn.Module):
    def __init__(self, num_classes=18):
        super(Timm_vit, self).__init__()
        
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=18)
        
    def forward(self, x):
        outputs = self.model(x)
        return outputs
        
            
            
    
    
    
    


    
    

    


        
        
        
        