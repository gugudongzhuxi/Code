import os
from datetime import datetime

# 定义两个文件夹路径
dataset_folder = '/data_4T/dlg/datasets/ConcretePumpTruck/12519测试ConcretePumpTruck' #不删除的
tmp_folder = '/data_4T/dlg/datasets/ConcretePumpTruck/ag_训练_filp'   #删除的

# 日志文件路径
log_file_path = os.path.join(dataset_folder, 'deleted_files_log.txt')

# 获取 dataset 文件夹中的所有 jpg 文件名（不含路径和后缀）
dataset_base_names = set()
for f in os.listdir(dataset_folder):
    if f.lower().endswith('.jpg'):
        base_name = os.path.splitext(f)[0]  # 去掉 .jpg
        dataset_base_names.add(base_name)

# 打开日志文件准备追加写入
with open(log_file_path, 'a') as log_file:
    # 遍历 tmp 文件夹中的文件
    for filename in os.listdir(tmp_folder):
        if filename.lower().endswith('.jpg'):
            file_base_name = os.path.splitext(filename)[0]

            # 判断是否是以 "_flip" 结尾，并且 base name 在 dataset 中存在非_flip版本
            if file_base_name.endswith('_flip'):
                original_base_name = file_base_name[:-5]  # 去掉 "_flip"
                if original_base_name in dataset_base_names:
                    file_path = os.path.join(tmp_folder, filename)
                    delete_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_message = f"[{delete_time}] 删除文件: {file_path}\n"

                    # 打印到控制台
                    print(log_message.strip())

                    # 写入日志文件
                    log_file.write(log_message)

                    # 删除图像文件
                    os.remove(file_path)

                    # 删除对应的 .xml 文件（如果有的话）
                    xml_path = os.path.join(tmp_folder, os.path.splitext(filename)[0] + '.xml')
                    if os.path.exists(xml_path):
                        os.remove(xml_path)
                        print(f"同时删除 XML 文件: {xml_path}")
                        log_file.write(f"[{delete_time}] 同时删除 XML 文件: {xml_path}\n")

print("清理完成！删除记录已保存至:", log_file_path)