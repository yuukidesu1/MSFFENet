import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.utils import model_zoo


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

class GhostModuleV2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode=None, args=None):
        super(GhostModuleV2, self).__init__()
        self.mode = mode
        self.gate_fn = nn.Sigmoid()

        if self.mode in ['original']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        elif self.mode in ['attn']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :]
        elif self.mode in ['attn']:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :] * F.interpolate(self.gate_fn(res), size=(out.shape[-2], out.shape[-1]),
                                                           mode='nearest')


class GhostBottleneckV2(nn.Module):

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., layer_id=None, args=None):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        if layer_id <= 1:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True, mode='original', args=args)
        else:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True, mode='attn', args=args)

            # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        self.ghost2 = GhostModuleV2(mid_chs, out_chs, relu=False, mode='original', args=args)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

class GhostNetV2(nn.Module):
    def __init__(self, cfgs, width=1.0, block=GhostBottleneckV2, args=None):
        super(GhostNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks 构建反向残差块
        stages = []
        layer_id = 0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                if block == GhostBottleneckV2:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                        se_ratio=se_ratio, layer_id=layer_id, args=args))
                input_channel = output_channel
                layer_id += 1
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers (global pool and conv_head removed)

    def forward(self, x):
        features = []

        # Initial stem
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Forward through each stage
        for stage in self.blocks:
            x = stage(x)
            features.append(x)  # Save the output of each stage

        return [features[0], features[2], features[4], features[6], features[-1]]  # Return the list of features

# @register_model
def ghostnetv2(**kwargs):
    cfgs_1 = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        # stage3
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        # stage4
        [[3, 240, 80, 0, 2]],
        [[3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
         ]
    ]

    cfgs_2 = [
        # k, t, c, SE, s
        # stage1
        [[3, 32, 32, 0.25, 1]],  # 增加 t 值，c 值为 32，启用 SE 模块
        # stage2
        [[3, 96, 64, 0.25, 2]],  # 增加 t 值，c 值为 64，启用 SE 模块
        [[3, 144, 64, 0.25, 1]],  # 增加 t 值，c 值为 64，启用 SE 模块
        # stage3
        [[5, 144, 128, 0.25, 2]],  # 增加 t 值，c 值为 128，启用 SE 模块
        [[5, 240, 128, 0.25, 1]],  # 增加 t 值，c 值为 128，启用 SE 模块
        # stage4
        [[3, 480, 256, 0.25, 2]],  # 增加 t 值，c 值为 256，启用 SE 模块
        [[3, 400, 256, 0.25, 1],  # 增加 t 值，c 值为 256，启用 SE 模块
         [3, 368, 256, 0.25, 1],
         [3, 368, 256, 0.25, 1],
         [3, 960, 320, 0.25, 1],  # 输出通道数增至 320
         [3, 1344, 320, 0.25, 1]  # 输出通道数增至 320，启用 SE
         ],
        # stage5
        [[5, 1344, 512, 0.25, 2]],  # 最大化 t 值和 c 值
        [[5, 1920, 512, 0.25, 1],  # 增加 t 值，c 值为 512
         [5, 1920, 512, 0.25, 1],
         [5, 1920, 512, 0.25, 1],
         [5, 1920, 512, 0.25, 1]
         ]
    ]

    # cfgs_16 = [
    #     # k, t, c, SE, s
    #     # stage1
    #     [[3, 16, 16, 0, 1]],  # 16->16, no SE, stride=1
    #     # stage2
    #     [[3, 64, 24, 0, 2]],  # 16->24, stride=2
    #     [[3, 72, 24, 0, 1]],  # 24->24, stride=1
    #     # stage3
    #     [[5, 72, 40, 0.25, 2]],  # 24->40, SE=0.25, stride=2
    #     [[5, 120, 40, 0.25, 1]],  # 40->40, SE=0.25, stride=1
    #     # stage4
    #     [[3, 240, 80, 0, 2]],  # 40->80, stride=2
    #     [[3, 200, 80, 0, 1]],  # 80->80, stride=1
    #     [[3, 184, 80, 0, 1]],  # 80->80, stride=1
    #     [[3, 184, 80, 0, 1]],  # 80->80, stride=1
    #     [[3, 480, 112, 0.25, 1]],  # 80->112, SE=0.25, stride=1
    #     [[3, 672, 112, 0.25, 1]],  # 112->112, SE=0.25, stride=1
    #     # stage5
    #     [[5, 672, 160, 0.25, 2]],  # 112->160, SE=0.25, stride=2
    #     [[5, 960, 160, 0, 1]],  # 160->160, stride=1
    #     [[5, 960, 160, 0.25, 1]],  # 160->160, SE=0.25, stride=1
    #     [[5, 960, 160, 0, 1]],  # 160->160, stride=1
    #     [[5, 960, 160, 0.25, 1]],  # 160->160, SE=0.25, stride=1
    # ]

    cfgs_16 = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 64, 0, 1]],  # 16->16, no SE, stride=1
        # stage2
        [[3, 64, 128, 0, 2]],  # 16->24, stride=2
        [[3, 72, 128, 0, 1]],  # 24->24, stride=1
        # stage3
        [[5, 72, 256, 0.25, 2]],  # 24->40, SE=0.25, stride=2
        [[5, 120, 256, 0.25, 1]],  # 40->40, SE=0.25, stride=1
        # stage4
        [[3, 240, 256, 0, 2]],  # 40->80, stride=2
        [[3, 200, 256, 0, 1]],  # 80->80, stride=1
        [[3, 184, 256, 0, 1]],  # 80->80, stride=1
        [[3, 184, 512, 0, 1]],  # 80->80, stride=1
        [[3, 480, 512, 0.25, 1]],  # 80->112, SE=0.25, stride=1
        [[3, 672, 512, 0.25, 1]],  # 112->112, SE=0.25, stride=1
        # stage5
        [[5, 672, 512, 0.25, 2]],  # 112->160, SE=0.25, stride=2
        [[5, 960, 1024, 0, 1]],  # 160->160, stride=1
        [[5, 960, 1024, 0.25, 1]],  # 160->160, SE=0.25, stride=1
        [[5, 960, 1024, 0, 1]],  # 160->160, stride=1
        [[5, 960, 2048, 0.25, 1]],  # 160->160, SE=0.25, stride=1
    ]

    model = GhostNetV2(width=1.6, cfgs=cfgs_1, block=GhostBottleneckV2)
    """
    加载预训练权重
    """
    if False:
        state_dict = torch.load('model_data/ck_ghostnetv2_16.pth')
        model.load_state_dict(state_dict, strict=False)
        result = model.load_state_dict(state_dict, strict=False)

        # 打印加载的结果
        print('====================================================================================')
        print('加载成功')
        print('缺失的键:', result.missing_keys)  # 打印缺失的参数
        print('意外的键:', result.unexpected_keys)  # 打印未预期的参数
        print('加载的键:', list(state_dict.keys()))  # 打印加载的所有键
        print('模型的键:', list(model.state_dict().keys()))  # 打印模型的所有键

    return model
    # return GhostNetV2(cfgs,
    #                   width=kwargs['width'],
    #                   args=kwargs['args'])
#
# cfgs = [
#         # k, t, c, SE, s
#         [[3, 16, 16, 0, 1]],
#         [[3, 48, 24, 0, 2]],
#         [[3, 72, 24, 0, 1]],
#         [[5, 72, 40, 0.25, 2]],
#         [[5, 120, 40, 0.25, 1]],
#         [[3, 240, 80, 0, 2]],
#         [[3, 200, 80, 0, 1],
#          [3, 184, 80, 0, 1],
#          [3, 184, 80, 0, 1],
#          [3, 480, 112, 0.25, 1],
#          [3, 672, 112, 0.25, 1]
#          ],
#         [[5, 672, 160, 0.25, 2]],
#         [[5, 960, 160, 0, 1],
#          [5, 960, 160, 0.25, 1],
#          [5, 960, 160, 0, 1],
#          [5, 960, 160, 0.25, 1]
#          ]
#     ]
#
# input_tensor = torch.randn(64, 3, 512, 512)
#
# model = ghostnetv2()
# # width=1.0, args=cfgs# 忽略num_classes
# features = model(input_tensor)
# for i, feature in enumerate(features):
#     print(i, feature.shape)
# # features 列表中包含每个stage的特征图
