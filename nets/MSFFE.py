import torch
import torch.nn as nn
import torch.nn.functional as F


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

# 这段代码实现了Squeeze-and-Excitation (SE) 模块。SE模块的主要作用是通过学习通道间的相互依赖关系来重新校准通道特征图的权重。

# 具体来说，这个模块做了以下几件事：
# 1. 在初始化函数中，设置了一个自适应平均池化层和一个全连接层序列。
# 2. 在前向传播函数中：
#    a. 首先对输入进行全局平均池化，将每个通道的特征压缩成一个数值。
#    b. 然后通过全连接层序列进行处理，这个序列包含了降维、ReLU激活、升维和Sigmoid激活。
#    c. 最后，将得到的通道权重与原始输入相乘，实现了对每个通道的重新加权。

# 这种机制允许模型学习到不同通道的重要性，从而提高特征的表达能力。


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

# 这段代码实现了一个名为MSFFE（Multi-Scale Feature Fusion and Enhancement）的神经网络模块。以下是该模块的主要功能：

# 1. 多尺度特征提取：
#    - 使用三个不同大小的卷积核（1x1, 3x3, 5x5）对输入特征进行卷积操作，以捕获不同尺度的特征信息。

# 2. 特征融合：
#    - 将三个卷积操作的输出在通道维度上进行拼接，形成一个更宽的特征图。

# 3. 通道注意力机制：
#    - 使用Squeeze-and-Excitation (SE) 模块对拼接后的特征进行重新校准，增强重要通道的权重。

# 4. 特征降维和整合：
#    - 通过1x1卷积将融合后的特征图通道数降回原始输入的通道数。

# 5. 特征规范化和非线性激活：
#    - 使用批归一化（Batch Normalization）和ReLU激活函数对特征进行进一步处理。

# 这个模块的设计目的是通过多尺度特征提取和通道注意力机制来增强特征的表达能力，同时保持输出特征图的空间尺寸与输入相同。
# 它可以用于各种计算机视觉任务中，如目标检测、语义分割等，以提高模型的性能。

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

# 这段代码定义了一个名为MSFFE_Model的神经网络模型。它的主要功能如下：

# 1. 模型结构：
#    - 包含5个MSFFE（Multi-Scale Feature Fusion and Enhancement）模块，每个模块处理不同通道数的特征图。

# 2. 初始化：
#    - 在__init__方法中，初始化5个MSFFE模块，分别处理64、256、512、1024和2048通道的特征图。

# 3. 前向传播：
#    - forward方法接收5个不同尺度的特征图作为输入（feat1到feat5）。
#    - 每个特征图通过对应的MSFFE模块进行处理。
#    - 处理后的特征图被收集到一个列表中并返回。

# 这个模型的设计目的是对多尺度的特征图进行增强和融合处理。
# 它可能是一个更大网络的一部分，用于处理从骨干网络（如ResNet）提取的不同层级的特征。
# 通过这种方式，模型可以在保持原始特征图空间尺寸的同时，增强每个尺度的特征表达能力。



'''# Example usage
in_channels = 64
a = torch.rand(1, in_channels, 256, 256)
model = MSFFE(in_channels)
print(model)

output = model(a)
print(output.shape)'''