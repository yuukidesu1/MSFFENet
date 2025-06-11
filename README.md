# 项目名称（Project Title）
Multi-Scale Attention U-Net for Semantic Segmentation in High-Resolution Satellite Images

一个基于多尺度注意力机制的 U-Net 模型，用于高分辨率遥感图像的语义分割。

# 项目简介（Introduction）
本项目旨在构建一个增强版U-Net网络结构，引入多尺度注意力机制，提升在高分辨率遥感影像中建筑物、道理、农田等地物的分割准确率。项目使用了公开数据集进行训练和测试，支持模型可视化和评估指标输出。

# 项目结构（Project Structure）
```bash
semantic-segmentation-project/
├── data/                   # 数据集目录
│   ├── VOCdevkit           # 数据集的格式应该与VOC数据集格式一致
│       ├── VOC2007         # 也可以自定义格式，但是需要修改训练代码
│           ├── ImageSets
│               ├── Segmentation
│           ├── JPEGImages
│           ├── SegmentationClass
│
├── Postprocess             # 后处理
│
├── Preprocess              # 前处理
│
├── checkpoints/            # 模型保存路径（自己定义）
│
├── train.py                # 训练脚本
├── test.py                 # 测试脚本
├── predict.py              # 预测脚本
│
├── requirements.txt        # Python依赖文件
└── README.md               # 项目说明文档
```

# 快速开始（Quick Start）
## 1. 环境配置
```bash
conda create -n segmentation python=3.9
conda activate segmentation
pip install -r requirements.txt
```
## 2. 数据准备
ISPRS Potsdam/Vaihingen 数据集的处理请参考：https://blog.csdn.net/qq_44961869/article/details/123760704

切好图片后用 voc_annotation.py 处理为VOC格式的数据集，并声称对应的 txt。

## 3. 模型训练
训练前需要对 train.py 进行对应的修改。

例如：
    - epochs
    - loss
    - 优化器
    - backbone
    - ...
```bash
python train.py
```

## 4. 模型推理
```bash
python predict.py
```
