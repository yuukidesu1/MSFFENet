import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[6, 12, 18]):
        super(LightweightASPP, self).__init__()
        self.aspp_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // len(dilation_rates), kernel_size=1, bias=False)
            for _ in dilation_rates
        ])
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(out_channels // len(dilation_rates), out_channels // len(dilation_rates),
                      kernel_size=3, padding=rate, dilation=rate, groups=out_channels // len(dilation_rates), bias=False)
            for rate in dilation_rates
        ])

    def forward(self, x):
        aspp_features = [F.relu(conv(x)) for conv in self.aspp_blocks]
        aspp_features = [dconv(feat) for feat, dconv in zip(aspp_features, self.dilated_convs)]
        return torch.cat(aspp_features, dim=1)

# class LightweightPPM(nn.Module):
#     def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6]):
#         super(LightweightPPM, self).__init__()
#         self.pool_sizes = pool_sizes
#         self.ppm_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.channel_attention = LightweightChannelAttention(out_channels)
#
#
#     def forward(self, x):
#         input_size = x.size()[2:]
#         ppm_features = [F.adaptive_avg_pool2d(x, output_size=size) for size in self.pool_sizes]
#         ppm_features = [F.interpolate(self.ppm_conv(feat), size=input_size, mode='bilinear', align_corners=False)
#                         for feat in ppm_features]
#         return LightweightChannelAttention(torch.cat(ppm_features, dim=1))

class LightweightPPM(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6]):
        super(LightweightPPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.ppm_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // len(pool_sizes), kernel_size=1, bias=False)
            for _ in pool_sizes
        ])
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        input_size = x.size()[2:]
        ppm_features = [F.adaptive_avg_pool2d(x, output_size=size) for size in self.pool_sizes]
        ppm_features = [F.interpolate(conv(feat), size=input_size, mode='bilinear', align_corners=False)
                        for feat, conv in zip(ppm_features, self.ppm_convs)]
        concat_features = torch.cat(ppm_features, dim=1)
        return self.final_conv(concat_features)


class LightweightChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(LightweightChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class LightweightFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightweightFusion, self).__init__()
        self.aspp = LightweightASPP(in_channels, out_channels // 2)
        self.ppm = LightweightPPM(in_channels, out_channels // 2)
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, groups=out_channels, bias=False)
        self.channel_attention = LightweightChannelAttention(out_channels)

    def forward(self, x):
        aspp_out = self.aspp(x)
        ppm_out = self.ppm(x)
        combined = torch.cat([aspp_out, ppm_out], dim=1)
        fused = self.fusion_conv(combined)
        fused = self.channel_attention(fused)
        return fused