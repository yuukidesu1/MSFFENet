import os
import re
from collections import defaultdict
from PIL import Image


def stitch_image(patches):
    """
    根据传入的patch列表 [(left, top, patch), ...] 拼接成一张大图。
    """
    max_right = max(left + patch.width for left, top, patch in patches)
    max_bottom = max(top + patch.height for left, top, patch in patches)
    # 根据需要可以调整模式，示例中假设是RGB图像
    stitched_img = Image.new('RGB', (max_right, max_bottom))
    for left, top, patch in patches:
        stitched_img.paste(patch, (left, top))
    return stitched_img


def main():
    # cropped_folder = os.path.join('cropped', 'top')
    # output_folder = 'stitched'
    cropped_folder = r'D:\PyCharm\UNet\Preprocess\cropped\top_res'
    output_folder = r'E:\issue_postprocess_data\mmsegmentation-main\result\Vaihingen dataset\origin_combine'
    os.makedirs(output_folder, exist_ok=True)

    # 假定文件命名格式为: originalName_top_left.tif
    pattern = re.compile(r'(.+?)_(\d+)_(\d+)\.png', re.IGNORECASE)
    grouped = defaultdict(list)
    for file in os.listdir(cropped_folder):
        if file.lower().endswith('.png'):
            match = pattern.match(file)
            if match:
                original_name = match.group(1)
                top = int(match.group(2))
                left = int(match.group(3))
                patch = Image.open(os.path.join(cropped_folder, file))
                grouped[original_name].append((left, top, patch))
            else:
                print(f"文件名格式不匹配: {file}")

    # 对每个原始图像的裁切块进行拼接
    for original_name, patches in grouped.items():
        stitched = stitch_image(patches)
        out_path = os.path.join(output_folder, original_name + '_stitched.png')
        stitched.save(out_path)
        print(f"已保存拼接图: {out_path}")


if __name__ == '__main__':
    main()
