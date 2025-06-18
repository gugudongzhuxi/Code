import os
from datetime import datetime
# 定义两个文件夹路径
dataset_folder = '/data_4T/dlg/datasets/测试集与训练集/201/201_11_删去CombTruck/txt'  #重复的不需要删去
tmp_folder = '/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/labels/train'  #被删去的路径

# 日志文件路径
log_file_path = os.path.join(dataset_folder, 'deleted_files_log.txt')

# 获取 dataset 文件夹中的所有 jpg 文件名（不含路径）
dataset_files = set(f for f in os.listdir(dataset_folder) if f.lower().endswith('.txt'))

# 打开日志文件准备追加写入
with open(log_file_path, 'a') as log_file:
    # 遍历 tmp 文件夹中的文件
    for filename in os.listdir(tmp_folder):
        if filename.lower().endswith('.txt'):
            if filename in dataset_files:
                file_path = os.path.join(tmp_folder, filename)
                delete_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_message = f"[{delete_time}] 删除文件: {file_path}\n"
                
                # 打印到控制台
                print(log_message.strip())  # 去掉换行符再打印
                
                # 写入日志文件
                log_file.write(log_message)
                
                # 删除文件
                os.remove(file_path)

print("清理完成！删除记录已保存至:")