import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

# 加载标注 JSON 文件
json_path = r"D:\PyCharm\UNet\DataSetpreProcessing\Potsdam_images\images\382.json"  # 替换成实际 JSON 文件路径
with open(json_path, "r") as f:
    data = json.load(f)

# 打开原始图像
image_path = r"D:\PyCharm\UNet\DataSetpreProcessing\Potsdam_images\images\382.jpg"  # 替换成实际图像路径
image = Image.open(image_path)

# 设置绘图
fig, ax = plt.subplots()
ax.imshow(image)

# 绘制轮廓
for shape in data["shapes"]:
    points = np.array(shape["points"])
    polygon = Polygon(points, closed=True, edgecolor="purple", facecolor="brown", alpha=0.5)
    ax.add_patch(polygon)

plt.axis("off")
plt.show()
