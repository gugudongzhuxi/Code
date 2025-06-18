import os
import shutil

# 定义基础目录
base_dir = "/data_4T/dlg/ultralytics-main1/datasets_ag"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# 创建目标文件夹（如果不存在）
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# 遍历 base_dir 下的所有文件
for filename in os.listdir(base_dir):
    file_path = os.path.join(base_dir, filename)
    
    # 确保是文件而不是目录
    if os.path.isfile(file_path):
        if filename.lower().endswith(".jpg"):
            shutil.move(file_path, os.path.join(images_dir, filename))
            print(f"Moved image: {filename}")
        elif filename.lower().endswith(".txt"):
            shutil.move(file_path, os.path.join(labels_dir, filename))
            print(f"Moved label: {filename}")

print("✅ 所有文件移动完成！")