import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16
# from nets.MsTFA import feat0_Block
from nets.MSFFE import MSFFE_Model



class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.channels = channels
        self.ca = self.ChannelAttention(self.channels)
        self.sa = self.SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

    class ChannelAttention(nn.Module):
        def __init__(self, channels):
            super(CBAM.ChannelAttention, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Sequential(
                nn.Conv2d(channels, channels // 16, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(channels // 16, channels, 1, bias=False)
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
            out = avg_out + max_out
            return self.sigmoid(out)

    class SpatialAttention(nn.Module):
        def __init__(self):
            super(CBAM.SpatialAttention, self).__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            out = torch.cat([avg_out, max_out], dim=1)
            out = self.conv(out)
            return self.sigmoid(out)

'''class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bridge, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x'''

'''test block'''
'''
class testblock(nn.Module):
    def __init__(self):
        super(testblock, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=7, stride=3, padding=3, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.bn1 = nn.BatchNorm2d(256)
        self.cbam1 = CBAM(256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.bn1(x)
        x = self.cbam1(x)

        return x
'''

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.cbam1 = CBAM(out_size)  # 在第一个卷积层后添加 CBAM 模块
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.cbam2 = CBAM(out_size)  # 在第二个卷积层后添加 CBAM 模块
        self.bn2 = nn.BatchNorm2d(out_size)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.bn1(outputs)
        outputs = self.cbam1(outputs)  # 应用第一个 CBAM 模块
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.cbam2(outputs)  # 应用第二个 CBAM 模块
        outputs = self.bn2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='resnet50'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
            # bridge_in_channels = 1024
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]

            # bridge_in_channels = 2048
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # bridge
        # self.bridge = Bridge(bridge_in_channels, bridge_in_channels)
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # 1111111111111111
        # self.testblock = testblock()
        # self.feat0block = feat0_Block(input_channels=3)
        # self.f_modle = F_Model()
        self.msffe = MSFFE_Model()




        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # bridge
        # bridge_out = self.bridge(feat5)

        # using MSFFE
        [f1, f2, f3, f4, f5] = self.msffe(feat1, feat2, feat3, feat4, feat5)


        up4 = self.up_concat4(f4, f5)
        up3 = self.up_concat3(f3, up4)
        up2 = self.up_concat2(f2, up3)
        up1 = self.up_concat1(f1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True


'''
count parameters
'''

'''def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = Unet(num_classes=6, pretrained=False, backbone='resnet50')
print(f"Total trainable parameters: {count_parameters(model)}")'''
