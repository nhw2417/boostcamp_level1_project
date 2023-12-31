import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)
    
class WideResNet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.wide_resnet = models.wide_resnet101_2(pretrained=True)
        self.wide_resnet.fc = nn.Linear(self.wide_resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.wide_resnet(x)

class ResNext101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnext = models.resnext101_32x8d(pretrained=True)
        #self.resnext.fc = nn.Linear(self.resnext.fc.in_features, num_classes)
        
        ###
        fc_layers = [nn.Linear(self.resnext.fc.in_features, 512),
                     nn.ReLU(),
                     nn.Linear(512,256),
                     nn.ReLU(),
                     nn.Linear(256,128),
                     nn.ReLU(),
                     nn.Linear(128, num_classes)]
        self.resnext.fc = nn.Sequential(*fc_layers)
        ###

    def forward(self, x):
        return self.resnext(x)

class VitB16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
    
    def forward(self, x):
        return self.vit(x)

class BaseModel(nn.Module):
    """
    기본적인 컨볼루션 신경망 모델
    """

    def __init__(self, num_classes):
        """
        모델의 레이어 초기화

        Args:
            num_classes (int): 출력 레이어의 뉴런 수
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            x (torch.Tensor): num_classes 크기의 출력 텐서
        """
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
