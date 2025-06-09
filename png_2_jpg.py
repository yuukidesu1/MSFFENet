from PIL import Image
import os


def convert_png_to_jpg(folder_path):
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):  # 只处理.png文件
            # 获取完整文件路径
            png_path = os.path.join(folder_path, filename)

            # 打开.png图像
            img = Image.open(png_path).convert("RGB")  # 将图像转换为RGB模式以确保兼容.jpg格式

            # 创建输出文件名，将扩展名更改为.jpg
            jpg_path = os.path.join(folder_path, filename[:-4] + ".jpg")

            # 保存为.jpg格式，调整质量以减少文件大小（可选）
            img.save(jpg_path, "JPEG", quality=100)

            print(f"Converted {filename} to .jpg format.")


if __name__ == "__main__":
    folder_path = r"D:\PyCharm\UNet\VOCdevkit\VOC2007\JPEGImages"  # 替换为包含.png文件的文件夹路径
    convert_png_to_jpg(folder_path)
