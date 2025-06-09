"""import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.version())"""

import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # 使用注释掉的代码进行前向传播（逐层处理）
        # x = self.features(x)                          # 特征提取部分
        # x = self.avgpool(x)                           # 平均池化
        # x = torch.flatten(x, 1)                       # 将特征图展平为一维
        # x = self.classifier(x)                        # 分类器部分

        # 使用切片操作对输入进行逐层处理，分别获取不同层的特征图
        feat1 = self.features[:4](x)  # 前4个卷积层的特征
        feat2 = self.features[4:9](feat1)  # 5个
        feat3 = self.features[9:16](feat2)  # 7个
        feat4 = self.features[16:23](feat3)  # 7个
        feat5 = self.features[23:-1](feat4)  # 最后一层卷积层之前的特征

        # 返回特征图列表
        return [feat1, feat2, feat3, feat4, feat5]


"""features = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 128, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(256, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
)
"""


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 512, 'M']
}

'''
接下来，如果需要利用make_layers替换上面的手动定义的卷积块只需要调用这个函数即可
'''

features = make_layers(cfgs['D'])
# 创建模型实例
model = VGG(features)

# 创建一个随机输入张量（示例输入）
input_tensor = torch.randn(64, 3, 512, 512)  # 8张图像，3个通道，512x512尺寸

# 使用模型进行前向传播
output_features = model(input_tensor)

# 打印每个特征图的形状
for i, feat in enumerate(output_features):
    print(f"Feature {i + 1} shape: {feat.shape}")
