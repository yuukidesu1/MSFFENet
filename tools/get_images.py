import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def download_images(url, save_folder, max_images):
    # 创建保存图片的文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 获取网页内容
    response = requests.get(url)
    if response.status_code != 200:
        print(f"无法访问网页: {url}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找所有图片标签
    img_tags = soup.find_all('img')
    count = 0

    for img_tag in img_tags:
        if count >= max_images:
            break

        # 获取图片URL
        img_url = img_tag.get('src') or img_tag.get('data-src')
        if not img_url:
            continue

        # 将相对URL转换为绝对URL
        img_url = urljoin(url, img_url)

        try:
            # 下载图片
            img_data = requests.get(img_url).content
            img_name = os.path.join(save_folder, f'image_{count + 1}.jpg')

            # 保存图片到文件夹
            with open(img_name, 'wb') as img_file:
                img_file.write(img_data)

            print(f"图片已保存: {img_name}")
            count += 1
        except Exception as e:
            print(f"下载失败: {img_url}, 错误: {e}")

    if count == 0:
        print("未找到任何图片。")
    else:
        print(f"总共下载了 {count} 张图片。")


# 示例使用
url = "https://www.pexels.com"
save_folder = r"D:\PyCharm\UNet\tools\images"
max_images = 10

download_images(url, save_folder, max_images)
