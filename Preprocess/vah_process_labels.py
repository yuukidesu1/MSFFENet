import os
import numpy as np
from PIL import Image

def process_label(in_path, out_path):
    """
    读取标签图像，转换为8位灰度图后，将每个像素的值减 1，
    使得类别标签从 0 开始（例如原始标签为 1~6，处理后变为 0~5）。
    """
    img = Image.open(in_path)
    gray = img.convert('L')  # 转为8位灰度
    arr = np.array(gray, dtype=np.int16)  # 使用较大数据类型，防止运算溢出
    arr = arr - 1  # 将标签值减1
    # 如存在负值，则将其归零（确保标签值>=0）
    arr = np.clip(arr, 0, 255)
    new_img = Image.fromarray(arr.astype('uint8'))
    new_img.save(out_path)

def main():
    input_folder = os.path.join('cropped', 'gts_for_participants')
    output_folder = 'processed_labels'
    os.makedirs(output_folder, exist_ok=True)

    label_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]
    for label_file in label_files:
        in_path = os.path.join(input_folder, label_file)
        out_path = os.path.join(output_folder, label_file)
        process_label(in_path, out_path)
        print(f"处理完成: {label_file}")

if __name__ == '__main__':
    main()
