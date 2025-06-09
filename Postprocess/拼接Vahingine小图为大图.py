import os
from PIL import Image
from collections import defaultdict


def stitch_tiles_by_area(input_dir, output_dir):
    """
    从形如 areaX_top_left_bottom_right.png 的裁剪图中，
    解析坐标后，按区域名“areaX”进行拼接，并输出至 output_dir。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 用于存储每个 area 的所有 tile 信息
    # 结构: area_dict[area_name] = {
    #   'tiles': [(top, left, bottom, right, filename), ... ],
    #   'max_bottom': ...,
    #   'max_right': ...
    # }
    area_dict = defaultdict(lambda: {
        'tiles': [],
        'max_bottom': 0,
        'max_right': 0
    })

    # 遍历输入文件夹，收集信息
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.png'):
            continue

        # 尝试解析文件名: area1_0_0_512_512.png => ["area1", "0", "0", "512", "512.png"]
        parts = filename.split('_')
        if len(parts) < 5:
            # 若格式不符合，跳过或可自行处理
            continue

        area_name = parts[0]  # "area1"

        try:
            top = int(parts[1])
            left = int(parts[2])
            bottom = int(parts[3])
            # 因为最后一个可能带扩展名，需再 split 一下
            right_str = parts[4]
            right_str = right_str.split('.')[0]  # "512.png" -> "512"
            right = int(right_str)
        except ValueError:
            # 若转换失败，说明文件名格式不符，也可自行处理
            continue

        # 记录
        area_dict[area_name]['tiles'].append((top, left, bottom, right, filename))

        # 更新最大 bottom/right 值，以便后面知道整幅图多大
        if bottom > area_dict[area_name]['max_bottom']:
            area_dict[area_name]['max_bottom'] = bottom
        if right > area_dict[area_name]['max_right']:
            area_dict[area_name]['max_right'] = right

    # 开始对每个 area 进行拼接
    for area_name, info in area_dict.items():
        tiles = info['tiles']
        max_bottom = info['max_bottom']
        max_right = info['max_right']

        # 创建一张空白画布 (高 = max_bottom, 宽 = max_right)
        # 如果是三通道图，就用"RGB"，若是其他模式可灵活调整
        stitched_img = Image.new("RGB", (max_right, max_bottom), (255, 255, 255))

        # 逐块放置
        for (top, left, bottom, right, fn) in tiles:
            tile_path = os.path.join(input_dir, fn)
            with Image.open(tile_path) as tile_img:
                # 注意 paste 的左上角是 (left, top)
                stitched_img.paste(tile_img, (left, top))

        # 保存结果
        out_name = f"{area_name}_stitched.png"
        stitched_img.save(os.path.join(output_dir, out_name))
        print(f"Saved {out_name}")


if __name__ == "__main__":
    input_folder = r"E:\issue_postprocess_data\mmsegmentation-main\result\Vaihingen dataset\deeplabv3plus_r50_vaihingen"  # 小图所在文件夹
    output_folder = r"E:\issue_postprocess_data\mmsegmentation-main\result\Vaihingen dataset\deeplabv3plus_r50_vaihingen_combine"  # 拼接后大图的输出文件夹

    stitch_tiles_by_area(input_folder, output_folder)
