"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torch.nn.functional as F

class VGGBlock(nn.Module):
    """Modular VGG block with configurable number of conv layers and channels.

    C configuration from Simonyan & Zisserman's VGG paper.
    """
    def __init__(self, in_channels, out_channels, num_convs, padding=1):
        super().__init__()
        layers = []
        current_in_channels = in_channels
        for i in range(num_convs):
            is_config_c_tail = (num_convs == 3 and i == 2)
            kernel_size = 1 if is_config_c_tail else 3
            layers.append(nn.Conv2d(current_in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """ResBlock with 3x3 convolutions (He et al., 2016)."""
    def __init__(self, in_channels, out_channels, activation, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = activation
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If spatial size shrinks (stride > 1) or channels change, adjust the shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  
        out = self.activation(out)
        return out


class AlexNet(nn.Module):
    """AlexNet (Krizhevsky et al., 2012) adapted for smaller inputs."""
    def __init__(self, **kwargs):
        super().__init__()

        drop_rate = kwargs.get("drop_rate", 0.5)
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(3072, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 11),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class VGG16(nn.Module):
    """VGG16 in C configuration of Simonyan & Zisserman, (2014) adapted for smaller inputs."""
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        drop_rate = kwargs.get("drop_rate", 0.5)

        self.features = nn.Sequential(
            VGGBlock(in_channels, 64, num_convs=2),
            VGGBlock(64, 128, num_convs=2),
            VGGBlock(128, 256, num_convs=3),
            VGGBlock(256, 512, num_convs=3),
            VGGBlock(512, 512, num_convs=3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNet18(nn.Module):
    """ResNet18 utilizing Full Fine-Tuning and strict ImageNet input normalization."""
    def __init__(self, in_channels=1, num_classes=11, **kwargs):
        super(ResNet18, self).__init__()
        
        self.model = tv_models.resnet18(pretrained=True)
        
        # 1. UNFREEZE ALL LAYERS: Medical X-rays have completely different low-level textures 
        # than ImageNet photos. We MUST let the early edge detectors adapt.
        for param in self.model.parameters():
            param.requires_grad = True
            
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
        
        # 2. IMAGENET NORMALIZATION CONSTANTS
        # Pre-trained weights mathematically expect inputs to be centered around these exact values.
        # register_buffer ensures these tensors move to the GPU automatically.
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # Upsample to native resolution
        if x.size(2) < 224 or x.size(3) < 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
        # Apply strict ImageNet normalization to prevent activation collapse
        x = (x - self.mean) / self.std
            
        return self.model(x)

class VGG16(nn.Module):
    def __init__(self, in_channels=1, num_classes=11, **kwargs):
        super(VGG16, self).__init__()
        
        # The configuration structure for a standard VGG16
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        
        layers = []
        current_channels = in_channels # This tracks the channels dynamically as we go deeper
        
        activation_str = kwargs.get("activation_str", "ReLU")
        drop_rate = kwargs.get("drop_rate", 0.5)
        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # FIX: Use current_channels instead of a hardcoded initial value!
                activation = getattr(nn, activation_str)

                layers += [
                    nn.Conv2d(current_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    activation(inplace=True)
                ]
                current_channels = v # FIX: Move the pointer forward to the new output channel size
                
        self.features = nn.Sequential(*layers)
        
        # Adaptive pooling ensures a stable input size for the linear classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class LeNet(nn.Module):
    """Modern LeNet-5: Restored FC layers for parameter capacity, stabilized with BatchNorm & Dropout."""
    def __init__(self, in_channels=1, num_classes=11, **kwargs):
        super(LeNet, self).__init__()
        
        activation_str = kwargs.get("activation_str", "ReLU")
        self.activation = getattr(nn, activation_str)
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6),
            self.activation(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            self.activation(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        
        # Restoring the 400-feature pipeline with balanced 0.3 Dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            self.activation(inplace=True),
            nn.Dropout(p=0.3), 
            
            nn.Linear(120, 84),
            self.activation(inplace=True),
            nn.Dropout(p=0.3),
            
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x