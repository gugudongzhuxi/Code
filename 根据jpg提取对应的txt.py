import os
import shutil

# 定义路径
dir_images = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/predict_cp_cpt"
dir_labels = "/data_4T/dlg/runs/detect/predict/labels"
output_dir = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/predict_cp_cpt"

# 支持的图像扩展名
image_extensions = ('.jpg', '.jpeg', '.png', '.webp')

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 获取 dir_images 中所有图片的 base name（不含扩展名）
image_names = set()
for filename in os.listdir(dir_images):
    name, ext = os.path.splitext(filename)
    if ext.lower() in image_extensions:
        image_names.add(name)

print(f"在 {dir_images} 中找到 {len(image_names)} 张图片")

# 遍历 dir_labels，找出与图片同名的 .txt 文件
copied_count = 0
for label_file in os.listdir(dir_labels):
    name, ext = os.path.splitext(label_file)
    if ext.lower() == ".txt" and name in image_names:
        src_path = os.path.join(dir_labels, label_file)
        dst_path = os.path.join(output_dir, label_file)
        shutil.copy2(src_path, dst_path)
        print(f"已复制: {label_file}")
        copied_count += 1

print(f"\n共复制 {copied_count} 个与图片同名的 .txt 文件到 {output_dir}")