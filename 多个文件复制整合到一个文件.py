import os
import shutil
import glob

# 源目录路径
source_dir = "/data_4T/dlg/datasets/datasets_ag/ag_6357_ConcretePumpTruck"

# 目标目录路径
target_dir = "/data_4T/dlg/datasets/datasets_ag"

# 创建目标目录（如果不存在）
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 初始化计数器
xml_count = 0
jpg_count = 0
txt_count = 0

# 复制所有XML文件
for xml_file in glob.glob(os.path.join(source_dir, "**", "*.xml"), recursive=True):
    shutil.copy(xml_file, target_dir)
    xml_count += 1

# 复制所有JPG文件
for jpg_file in glob.glob(os.path.join(source_dir, "**", "*.jpg"), recursive=True):
    shutil.copy(jpg_file, target_dir)
    jpg_count += 1

# 复制所有TXT文件
for txt_file in glob.glob(os.path.join(source_dir, "**", "*.txt"), recursive=True):
    shutil.copy(txt_file, target_dir)
    txt_count += 1

# 打印复制文件数量统计
total_files = xml_count + jpg_count + txt_count
print(f"文件复制完成，统计信息：")
print(f"- 复制了 {xml_count} 个XML文件")
print(f"- 复制了 {jpg_count} 个JPG文件")
print(f"- 复制了 {txt_count} 个TXT文件")
print(f"- 总共复制了 {total_files} 个文件到 {target_dir} 文件夹")