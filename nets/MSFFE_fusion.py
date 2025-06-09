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

# PPM Module
class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for pool_size in pool_sizes
        ])
        self.bottleneck = nn.Conv2d(in_channels + len(pool_sizes) * out_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x]
        for stage in self.stages:
            pyramids.append(F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True))
        output = torch.cat(pyramids, dim=1)
        output = self.bottleneck(output)
        output = self.bn(output)
        output = self.relu(output)
        return output

# MSFFE + PPM Fusion Module with SE Attention
class MSFFE_PPM_Fusion(nn.Module):
    def __init__(self, in_channels):
        super(MSFFE_PPM_Fusion, self).__init__()
        self.msffe = MSFFE(in_channels)
        self.ppm = PPM(in_channels, in_channels // 4)
        self.se = SqueezeExcitationLayer(in_channels * 2)
        self.conv_fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        msffe_out = self.msffe(x)
        ppm_out = self.ppm(x)

        # Concatenate MSFFE and PPM outputs
        fusion = torch.cat([msffe_out, ppm_out], dim=1)

        # Apply SE module for channel attention
        fusion = self.se(fusion)

        # Fusion convolution, batch normalization, and activation
        fusion = self.conv_fusion(fusion)
        fusion = self.bn(fusion)
        fusion = self.relu(fusion)

        return fusion

# Example usage in a larger model
class MSFFE_PPM_Model(nn.Module):
    def __init__(self):
        super(MSFFE_PPM_Model, self).__init__()
        self.f1 = MSFFE_PPM_Fusion(64)
        self.f2 = MSFFE_PPM_Fusion(256)
        self.f3 = MSFFE_PPM_Fusion(512)
        self.f4 = MSFFE_PPM_Fusion(1024)
        self.f5 = MSFFE_PPM_Fusion(2048)

    def forward(self, feat1, feat2, feat3, feat4, feat5):
        feat1 = self.f1(feat1)
        feat2 = self.f2(feat2)
        feat3 = self.f3(feat3)
        feat4 = self.f4(feat4)
        feat5 = self.f5(feat5)

        return [feat1, feat2, feat3, feat4, feat5]

# Example usage
# in_channels = 64
# x = torch.rand(1, in_channels, 256, 256)
# model = MSFFE_PPM_Fusion(in_channels)
# output = model(x)
# print(output.shape)
