import os
from PIL import Image


def crop_vaihingen_no_partial(
        top_dir,
        gts_dir,
        out_img_dir,
        out_gt_dir,
        tile_size=512
):
    """
    将Vaihingen数据集中的原图(.tif)与标签(.tif/.png)按网格裁成 tile_size×tile_size。
    不保留越界的部分（即不足512×512的块忽略），文件名从 0 开始编号。
    """

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_gt_dir, exist_ok=True)

    top_files = [f for f in os.listdir(top_dir) if f.lower().endswith(".tif")]
    top_files.sort()

    tile_index = 0  # 统一编号，从 0 开始

    for top_name in top_files:
        base_name = os.path.splitext(top_name)[0]

        top_path = os.path.join(top_dir, top_name)
        img = Image.open(top_path)

        # 假设标签是相同 base_name + ".tif"
        label_name = base_name + ".tif"
        label_path = os.path.join(gts_dir, label_name)

        if not os.path.exists(label_path):
            print(f"[警告] 标签文件 {label_name} 不存在，跳过。")
            img.close()
            continue

        label_img = Image.open(label_path)

        width, height = img.size

        # 仅遍历到 width-tile_size、height-tile_size
        for top_y in range(0, height - tile_size + 1, tile_size):
            for left_x in range(0, width - tile_size + 1, tile_size):
                # 计算右下角坐标(正好 tile_size 大小)
                bottom_y = top_y + tile_size
                right_x = left_x + tile_size

                # 裁剪原图与标签
                crop_box = (left_x, top_y, right_x, bottom_y)
                tile_img = img.crop(crop_box)
                tile_label = label_img.crop(crop_box)

                # 统一编号，图像 & 标签
                out_img_name = f"{tile_index}.jpg"
                out_label_name = f"{tile_index}.png"

                # 转RGB后存 .jpg
                tile_img = tile_img.convert("RGB")
                tile_img.save(os.path.join(out_img_dir, out_img_name), quality=95)

                # 标签保存为 .png
                tile_label.save(os.path.join(out_gt_dir, out_label_name))

                tile_index += 1  # 递增编号

        img.close()
        label_img.close()
        print(f"[完成] {top_name} 的切片处理。")


if __name__ == "__main__":
    top_folder = r"E:\issue_postprocess_data\mmsegmentation-main\data\Vaihingen_ori\ISPRS_semantic_labeling_Vaihingen\top"
    gts_folder = r"E:\issue_postprocess_data\mmsegmentation-main\data\Vaihingen_ori\ISPRS_semantic_labeling_Vaihingen\gts_for_participants"

    output_images = r"D:\PyCharm\UNet\Preprocess\Vaihingen\images"
    output_masks = r"D:\PyCharm\UNet\Preprocess\Vaihingen\gt"

    crop_vaihingen_no_partial(
        top_dir=top_folder,
        gts_dir=gts_folder,
        out_img_dir=output_images,
        out_gt_dir=output_masks,
        tile_size=512
    )
