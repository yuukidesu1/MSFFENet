import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False)
            for rate in dilation_rates
        ])
        self.pointwise = nn.Conv2d(len(dilation_rates) * out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        aspp_features = [F.relu(conv(x)) for conv in self.aspp_blocks]
        aspp_out = torch.cat(aspp_features, dim=1)
        return self.pointwise(aspp_out)


class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6]):
        super(PPM, self).__init__()
        self.ppm_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            ) for pool_size in pool_sizes
        ])
        self.pointwise = nn.Conv2d(len(pool_sizes) * out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        input_size = x.size()[2:]
        ppm_features = [F.interpolate(block(x), size=input_size, mode='bilinear', align_corners=False) for block in
                        self.ppm_blocks]
        ppm_out = torch.cat(ppm_features, dim=1)
        return self.pointwise(ppm_out)


class ASPP_PPM_Fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP_PPM_Fusion, self).__init__()
        self.aspp = ASPP(in_channels, out_channels)
        self.ppm = PPM(in_channels, out_channels)
        self.final_conv = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        aspp_out = self.aspp(x)
        ppm_out = self.ppm(x)
        combined = torch.cat([aspp_out, ppm_out], dim=1)
        return self.final_conv(combined)