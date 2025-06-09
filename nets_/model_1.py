import torch
import torch.nn as nn
from ghostnetv2 import ghostnetv2
from torchsummary import summary

cfgs = [
    # k, t, c, SE, s
    [[3, 16, 16, 0, 1]],
    [[3, 48, 24, 0, 2]],
    [[3, 72, 24, 0, 1]],
    [[5, 72, 40, 0.25, 2]],
    [[5, 120, 40, 0.25, 1]],
    [[3, 240, 80, 0, 2]],
    [[3, 200, 80, 0, 1],
     [3, 184, 80, 0, 1],
     [3, 184, 80, 0, 1],
     [3, 480, 112, 0.25, 1],
     [3, 672, 112, 0.25, 1]
     ],
    [[5, 672, 160, 0.25, 2]],
    [[5, 960, 160, 0, 1],
     [5, 960, 160, 0.25, 1],
     [5, 960, 160, 0, 1],
     [5, 960, 160, 0.25, 1]
     ]
]


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ghostnetv2(num_classes=10, width=1.0, dropout=0.2, args=cfgs)



# 打印模型
print('----------------------------- model ---------------------------------------------')
print(model)

# # 打印模型摘要
# print('----------------------------- model summary ---------------------------------------------')
# x = torch.randn(1, 3, 256, 256).to(device)
# summary(model, input_size=(3, 256, 256), device=str(device))
#
# # 将模型输出为ONNX(使用Netron监看)
# dummy_input = torch.randn(1, 3, 256, 256).to(device)
# torch.onnx.export(model, dummy_input, "ghostnetv2.onnx")