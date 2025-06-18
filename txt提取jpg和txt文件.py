import os
import shutil

# 定义原始标签路径、原始图像路径和目标路径
src_dir_labels = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/labels/test"
src_dir_images = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/images/test"
dst_dir = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/tmp1"

# 创建目标目录（如果不存在）
os.makedirs(dst_dir, exist_ok=True)

# 需要匹配的目标类别
# target_classes = {'4', '5'}
target_classes = {'1'}
# 获取所有txt文件名（不带扩展名）
for label_file in os.listdir(src_dir_labels):
    if not label_file.endswith(".txt"):
        continue

    # 构造完整路径
    label_path = os.path.join(src_dir_labels, label_file)
    image_name = os.path.splitext(label_file)[0] + ".jpg"
    image_path = os.path.join(src_dir_images, image_name)

    # 检查是否有目标类别
    has_target_class = False
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0] in target_classes:
                    has_target_class = True
                    break
    except Exception as e:
        print(f"读取文件失败: {label_file}，错误：{e}")
        continue

    # 如果存在目标类别，复制 txt 和 jpg
    if has_target_class:
        # 复制 .txt 文件
        dst_label_path = os.path.join(dst_dir, label_file)
        shutil.copy2(label_path, dst_label_path)

        # 复制 .jpg 图像文件
        if os.path.exists(image_path):
            dst_image_path = os.path.join(dst_dir, image_name)
            shutil.copy2(image_path, dst_image_path)
            print(f"已复制: {label_file}, {image_name}")
        else:
            print(f"警告: 图像文件不存在: {image_name}")