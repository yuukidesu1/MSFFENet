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

# 定义不同类别的深色轮廓颜色
color_map = {
    "diai": "darkred",
    "shu": "darkblue",
    "other": "darkgreen",
    "road": "indigo",
    "car": "brown"
    # 根据需要添加更多类别和颜色
}

# 设置绘图
fig, ax = plt.subplots()
ax.imshow(image)

# 绘制轮廓
for shape in data["shapes"]:
    points = np.array(shape["points"])
    label = shape.get("label", "default")  # 获取标签名称，默认为 "default"

    # 根据标签获取颜色，若标签不存在于 color_map 则使用黑色
    edge_color = color_map.get(label, "black")

    # 创建多边形轮廓，无填充颜色
    polygon = Polygon(points, closed=True, edgecolor=edge_color, linewidth=2, fill=False)
    ax.add_patch(polygon)

# 去掉坐标轴
plt.axis("off")
plt.show()
