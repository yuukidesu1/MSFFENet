import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo


# 3x3卷积层定义
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 卷积层，带有步幅、分组和膨胀"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


# 1x1卷积层定义
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积层"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ResNeXt Bottleneck Block
class ResNeXtBottleneck(nn.Module):
    expansion = 4  # ResNeXt uses a bottleneck structure, so output channels are 4x the base width

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=32, base_width=4, dilation=1):
        super(ResNeXtBottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups  # 调整基宽和组卷积

        # 1x1, 减少维度
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = nn.BatchNorm2d(width)

        # 3x3, 分组卷积
        self.conv2 = conv3x3(width, width, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(width)

        # 1x1, 扩展回到 planes * 4
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ResNeXt50_32x4d 主干
class ResNeXt(nn.Module):
    def __init__(self, block, layers, num_classes=1000, groups=32, width_per_group=4):
        super(ResNeXt, self).__init__()
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group

        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 最大池化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个阶段
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        #
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        #
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.relu(x)

        x = self.maxpool(feat1)
        feat2 = self.layer1(x)

        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)

        return [feat1, feat2, feat3, feat4, feat5]


# 构建 ResNeXt50_32x4d 模型
def resnext50_32x4d(pretrained=False, **kwargs):
    layers = [3, 4, 6, 3]  # ResNeXt50 的层数配置
    model = ResNeXt(ResNeXtBottleneck, layers, groups=32, width_per_group=4)

    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth', model_dir='model_data'), strict=False)

    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

    del model.avgpool
    del model.fc
    return model




# 加载预训练权重的函数
# def load_pretrained_weights(model, weight_path):
#     state_dict = torch.load(weight_path)
#     model.load_state_dict(state_dict)


# 构建模型并加载预训练权重
# model = resnext50_32x4d(num_classes=1000)

# 加载预训练权重（路径需要根据具体位置指定）
# weight_path = 'path_to_pretrained_weights.pth'
# load_pretrained_weights(model, weight_path)

# 打印模型（可选）
# print(model)