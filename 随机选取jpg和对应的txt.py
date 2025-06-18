import os
import random
import shutil

# 源路径和目标路径
src_img_dir = '/data_4T/dlg/BCCD/BCCD_Dataset/BCCD/JPEGImages'
src_lbl_dir = '/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064/labels/test'

dst_img_dir = '/data_4T/dlg/yolov11_gaijin/datasets/images/test'
dst_lbl_dir = '/data_4T/dlg/yolov11_gaijin/datasets/labels/test'

# 创建目标目录（如果不存在）
os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(dst_lbl_dir, exist_ok=True)

# 获取所有 .jpg 文件名（不带后缀）
all_images = [f for f in os.listdir(src_img_dir) if f.endswith('.jpg')]
all_basenames = [os.path.splitext(f)[0] for f in all_images]

# 随机选取 5000 个样本（不足则全选）
num_to_select = min(700, len(all_basenames))
selected_basenames = random.sample(all_basenames, num_to_select)

# 复制文件
count = 0
for basename in selected_basenames:
    jpg_src = os.path.join(src_img_dir, basename + '.jpg')
    txt_src = os.path.join(src_lbl_dir, basename + '.txt')

    jpg_dst = os.path.join(dst_img_dir, basename + '.jpg')
    txt_dst = os.path.join(dst_lbl_dir, basename + '.txt')

    # 确保 jpg 和 txt 文件都存在再复制
    if os.path.exists(jpg_src) and os.path.exists(txt_src):
        shutil.copy2(jpg_src, jpg_dst)  # 使用 copy2 代替 copy
        shutil.copy2(txt_src, txt_dst)  # 使用 copy2 代替 copy
        count += 1

print(f"✅ 成功复制 {count} 组 jpg + txt 文件，并保留了文件的元数据。")