import time
import os
import cv2
import numpy as np
from PIL import Image
import tqdm
import torch
from thop import profile

from DataSetpreProcessing.img_fushion_v1 import output_path
from nets.MSFFE_fusion_DenseASPP import output
# from unet import Unet_ONNX, Unet
from nets.unet_MSFFE_007 import Unet

unet = Unet()
unet.eval()

input_tensor = torch.randn(1, 1, 3, 512, 512)

flops, params = profile(unet, input_tensor, verbose=False)

print("FLOPs: {:.2f} Gb".format(flops / 1e9))
print("Params: {:.2f} Mb".format(params / 1e6))