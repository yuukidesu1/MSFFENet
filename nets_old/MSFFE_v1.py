import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SqueezeExcitationLayer(nn.Module):
    def __init__(self, channel, reduction=None):
        super(SqueezeExcitationLayer, self).__init__()
        if reduction is None:
            reduction = max(1, channel // 16)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction, bias=False),
            Swish(),
            # nn.BatchNorm1d(reduction),
            nn.Linear(reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MSFFE(nn.Module):
    def __init__(self, in_channels):
        super(MSFFE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2)
        self.se = SqueezeExcitationLayer(in_channels * 3)
        self.conv_fusion = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)

        # Concatenate along the channel dimension
        fusion = torch.cat([conv1, conv3, conv5], dim=1)

        # Apply Squeeze-and-Excitation
        fusion = self.se(fusion)

        # Fusion convolution
        fusion = self.conv_fusion(fusion)

        # Batch normalization and ReLU
        fusion = self.bn(fusion)
        fusion = self.relu(fusion)

        # Jump connection (concatenate input and fused feature maps)
        # 这段理论上不需要，跳跃连接被安排在后续工作中，这个模块只需要输出处理好的特征就好
        # output = torch.cat([x, fusion], dim=1)

        return fusion

class MSFFE_Model(nn.Module):
    def __init__(self):
        super(MSFFE_Model, self).__init__()
        self.f1 = MSFFE(64)
        self.f2 = MSFFE(256)
        self.f3 = MSFFE(512)
        self.f4 = MSFFE(1024)
        self.f5 = MSFFE(2048)

    def forward(self, feat1, feat2, feat3, feat4, feat5):
        feat1 = self.f1(feat1)
        feat2 = self.f2(feat2)
        feat3 = self.f3(feat3)
        feat4 = self.f4(feat4)
        feat5 = self.f5(feat5)

        return [feat1, feat2, feat3, feat4, feat5]