import os
from PIL import Image

# 指定图片所在的目录
directory = '/data_4T/dlg/需要标定一下'

# 遍历目录中的所有 bmp 文件
for filename in os.listdir(directory):
    if filename.lower().endswith('.bmp'):
        bmp_path = os.path.join(directory, filename)
        jpg_filename = os.path.splitext(filename)[0] + '.jpg'
        jpg_path = os.path.join(directory, jpg_filename)

        try:
            # 打开并转换图像
            with Image.open(bmp_path) as img:
                img = img.convert('RGB')  # 确保为 RGB 格式
                img.save(jpg_path, 'JPEG')  # 保存为 JPEG 格式

            print(f"已转换: {filename} -> {jpg_filename}")
        except Exception as e:
            print(f"转换失败 {filename}: {e}")

print("BMP 转 JPG 完成。")