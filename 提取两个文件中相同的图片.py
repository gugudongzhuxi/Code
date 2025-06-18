import os
import shutil
#    将这些重复命名的图片从 dir1 提取出来，复制到目标目录 output_di
# 定义路径
dir1 = "/data_4T/dlg/ultralytics-main1/runs/detect/predict10"     #复制的来源
dir2 = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/test_cpt"   
output_dir = "/data_4T/dlg/ultralytics-main1/runs/detect/predict10/predict_cpt"

# 支持的图片扩展名
image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 获取 dir2 中所有图片的基本文件名（不带扩展名）
dir2_files = set()
for filename in os.listdir(dir2):
    name, ext = os.path.splitext(filename)
    if ext.lower() in image_extensions:
        dir2_files.add(name.lower())  # 忽略大小写

print(f"在 {dir2} 中找到 {len(dir2_files)} 个图片文件名")

# 遍历 dir1，找出与 dir2 名称相同的图片并复制
copied_count = 0
for filename in os.listdir(dir1):
    name, ext = os.path.splitext(filename)
    if ext.lower() in image_extensions:
        if name.lower() in dir2_files:
            src_path = os.path.join(dir1, filename)
            dst_path = os.path.join(output_dir, filename)
            shutil.copy2(src_path, dst_path)
            print(f"已复制: {filename}")
            copied_count += 1

print(f"\n共复制 {copied_count} 张与 {dir2} 命名重复的图片到 {output_dir}")