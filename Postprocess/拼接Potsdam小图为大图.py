import os
from PIL import Image

# 参数设置
input_folder = r'D:\PyCharm\UNet\DataSetpreProcessing\Potsdam_images\images'  # 存放3800张图片的文件夹
output_folder = r'D:\PyCharm\UNet\Postprocess\Potsdam\original_combine'  # 输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 单张小图尺寸（假设所有图像尺寸相同）
img_sample = Image.open(os.path.join(input_folder, '0.jpg'))
img_width, img_height = img_sample.size

# 每张大图由10×10小图组成
rows, cols = 10, 10

# 总的图片数量
total_images = 3800

# 计算需要拼接的大图数量
images_per_large = rows * cols
num_large_images = total_images // images_per_large

for idx in range(num_large_images):
    large_img = Image.new('RGB', (cols * img_width, rows * img_height))

    for i in range(images_per_large):
        img_idx = idx * images_per_large + i
        img_path = os.path.join(input_folder, f'{img_idx}.jpg')

        if os.path.exists(img_path):
            img = Image.open(img_path)
            row = i // cols
            col = i % cols
            position = (col * img_width, row * img_height)
            large_img.paste(img, position)

    output_path = os.path.join(output_folder, f'large_image_{idx}.png')
    large_img.save(output_path)
    print(f'Saved {output_path}')
