import os
import shutil

# 定义路径
image_dir = '/data_4T/dlg/yolov11_gaijin/datasets/images/test'
label_dir = '/data_4T/dlg/yolov11_gaijin/datasets/labels/test'
new_dir = '/data_4T/dlg/yolov11_gaijin/datasets/new'

def move_files_with_class_6():
    # 创建目标目录
    os.makedirs(new_dir, exist_ok=True)
    
    # 遍历标签目录中的所有txt文件
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(label_dir, label_file)
        should_move = False

        # 读取标签文件内容
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0] == '6':
                    should_move = True
                    break

        if should_move:
            # 构造对应的jpg文件路径
            jpg_file = os.path.splitext(label_file)[0] + '.jpg'
            jpg_path = os.path.join(image_dir, jpg_file)

            # 移动文件
            try:
                # 移动图片
                if os.path.exists(jpg_path):
                    shutil.move(jpg_path, os.path.join(new_dir, jpg_file))
                else:
                    print(f"警告：图片文件 {jpg_file} 不存在")
                
                # 移动标签
                shutil.move(label_path, os.path.join(new_dir, label_file))
                print(f"已移动：{label_file} 和 {jpg_file}")

            except Exception as e:
                print(f"移动文件时出错：{str(e)}")

if __name__ == '__main__':
    move_files_with_class_6()