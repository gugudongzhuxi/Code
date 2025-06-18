import os

# 指定目录路径
directory = '/data_4T/dlg/datasets/ConcretePumpTruck/ag_6357'

# 支持的扩展名
extensions = ('.jpg', '.xml')

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    name, ext = os.path.splitext(filename)

    # 判断是否是 .jpg 或 .xml 文件，并且名字以 '_flip' 结尾
    if ext in extensions and name.endswith('_flip'):
        file_path = os.path.join(directory, filename)
        try:
            os.remove(file_path)
            print(f"已删除: {filename}")
        except Exception as e:
            print(f"无法删除 {filename}: {e}")