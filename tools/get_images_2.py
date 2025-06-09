import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_google_image_links(search_url, max_links=10):
    """
    从 Google 图片搜索页面获取高分辨率图片链接
    :param search_url: Google 图片搜索结果的 URL
    :param max_links: 最大获取图片数量
    :return: 高分辨率图片链接列表
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        print("无法访问 Google 图片搜索页面。请检查链接或网络连接。")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for img_tag in soup.find_all('img'):
        # 尝试获取高分辨率图片链接
        highres = img_tag.get('data-iurl')  # 高分辨率图片字段
        if highres:
            links.append(highres)
            if len(links) >= max_links:
                break
    return links

def download_images(image_links, save_folder):
    """
    下载图片并保存到指定文件夹
    :param image_links: 图片链接列表
    :param save_folder: 保存图片的文件夹路径
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for idx, img_url in enumerate(image_links):
        try:
            response = requests.get(img_url)
            if response.status_code == 200:
                # 图片保存路径
                img_path = os.path.join(save_folder, f'image_{idx + 1}.jpg')
                with open(img_path, 'wb') as img_file:
                    img_file.write(response.content)
                print(f"图片已保存: {img_path}")
            else:
                print(f"无法下载图片: {img_url}")
        except Exception as e:
            print(f"下载失败: {img_url}, 错误: {e}")

# 示例使用
if __name__ == "__main__":
    # 替换为你的 Google 图片搜索链接
    search_url = "https://www.google.co.jp/search?q=%E5%A3%81%E7%BA%B8&sca_esv=269d92588f55ed80&udm=2&biw=1699&bih=834&sxsrf=ADLYWIJH8-moaRVUZKvgmxEmvSbskSYhZg%3A1733835317128&ei=NTpYZ-PkBu_S1e8P5rKkyQQ&ved=0ahUKEwij07Pxn52KAxVvafUHHWYZKUkQ4dUDCBA&oq=%E5%A3%81%E7%BA%B8&gs_lp=EgNpbWciBuWjgee6uDIGEAAYBxgeMgYQABgHGB4yBRAAGIAEMgsQABiABBixAxiDATIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABEjylx5QAFgAcAF4AJABAJgBAKABAKoBALgBDMgBAJgCAaACBZgDAIgGAZIHATGgBwA&sclient=img"
    save_folder = "downloaded_images"
    max_links = 10  # 要获取的最大图片数量

    print("正在获取图片链接...")
    image_links = get_google_image_links(search_url, max_links)

    if image_links:
        print(f"找到 {len(image_links)} 个图片链接，开始下载...")
        download_images(image_links, save_folder)
    else:
        print("未找到任何图片链接。")
