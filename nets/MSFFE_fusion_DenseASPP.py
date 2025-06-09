import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation Layer
class SqueezeExcitationLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitationLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# MSFFE Module
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

        return fusion

# DenseASPP Module
class DenseASPP(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=5, dilation_rates=(3, 6, 12, 18, 30)):
        super(DenseASPP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation_rate = dilation_rates[i]
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=dilation_rate, dilation=dilation_rate, bias=False),
                    nn.BatchNorm2d(growth_rate),
                    nn.ReLU(inplace=True)
                )
            )
        self.conv1x1 = nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        x = torch.cat(features, dim=1)
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# MSFFE + DenseASPP Fusion Module with SE Attention
class MSFFE_DenseASPP_Fusion(nn.Module):
    def __init__(self, in_channels):
        super(MSFFE_DenseASPP_Fusion, self).__init__()
        self.msffe = MSFFE(in_channels)
        self.dense_aspp = DenseASPP(in_channels)
        self.se = SqueezeExcitationLayer(in_channels * 2)
        self.conv_fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        msffe_out = self.msffe(x)
        dense_aspp_out = self.dense_aspp(x)

        # Concatenate MSFFE and DenseASPP outputs
        fusion = torch.cat([msffe_out, dense_aspp_out], dim=1)

        # Apply SE module for channel attention
        fusion = self.se(fusion)

        # Fusion convolution, batch normalization, and activation
        fusion = self.conv_fusion(fusion)
        fusion = self.bn(fusion)
        fusion = self.relu(fusion)

        return fusion

# Example usage in a larger model
class MSFFE_DenseASPP_Model(nn.Module):
    def __init__(self):
        super(MSFFE_DenseASPP_Model, self).__init__()
        self.f1 = MSFFE_DenseASPP_Fusion(64)
        self.f2 = MSFFE_DenseASPP_Fusion(256)
        self.f3 = MSFFE_DenseASPP_Fusion(512)
        self.f4 = MSFFE_DenseASPP_Fusion(1024)
        self.f5 = MSFFE_DenseASPP_Fusion(2048)

    def forward(self, feat1, feat2, feat3, feat4, feat5):
        feat1 = self.f1(feat1)
        feat2 = self.f2(feat2)
        feat3 = self.f3(feat3)
        feat4 = self.f4(feat4)
        feat5 = self.f5(feat5)

        return [feat1, feat2, feat3, feat4, feat5]

# Example usage
in_channels = 64
x = torch.rand(1, in_channels, 256, 256)
model = MSFFE_DenseASPP_Fusion(in_channels)
output = model(x)
# print(output.shape)
# print(model)