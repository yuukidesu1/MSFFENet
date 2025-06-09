import os
from PIL import Image


def crop_image(img, crop_size=512):
    """
    对输入图像按照固定尺寸进行裁切。
    仅保留完整裁切区域。
    """
    w, h = img.size
    patches = []
    for top in range(0, h, crop_size):
        for left in range(0, w, crop_size):
            right = left + crop_size
            bottom = top + crop_size
            # 只处理尺寸满足要求的区域
            if right <= w and bottom <= h:
                patch = img.crop((left, top, right, bottom))
                patches.append(((left, top), patch))
    return patches


def main():
    # 原始数据文件夹
    image_folder = r'E:\issue_postprocess_data\mmsegmentation-main\data\Vaihingen_ori\ISPRS_semantic_labeling_Vaihingen\top'
    label_folder = r'E:\issue_postprocess_data\mmsegmentation-main\data\Vaihingen_ori\ISPRS_semantic_labeling_Vaihingen\gts_for_participants'
    # 输出文件夹
    out_img_folder = os.path.join('cropped', 'top')
    out_label_folder = os.path.join('cropped', 'gts_for_participants')

    os.makedirs(out_img_folder, exist_ok=True)
    os.makedirs(out_label_folder, exist_ok=True)

    # 遍历所有.tif文件
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.tif')]
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, image_file)
        if not os.path.exists(label_path):
            print(f"警告：未找到 {image_file} 对应的标签文件！")
            continue

        img = Image.open(image_path)
        label = Image.open(label_path)

        img_patches = crop_image(img)
        label_patches = crop_image(label)

        # 使用文件名附加裁切起始坐标作为patch索引保存裁切块
        for (left, top), patch in img_patches:
            patch_name = f"{os.path.splitext(image_file)[0]}_{top}_{left}.jpg"
            patch.save(os.path.join(out_img_folder, patch_name))
        for (left, top), patch in label_patches:
            patch_name = f"{os.path.splitext(image_file)[0]}_{top}_{left}.png"
            patch.save(os.path.join(out_label_folder, patch_name))
        print(f"完成裁切 {image_file}")


if __name__ == '__main__':
    main()
