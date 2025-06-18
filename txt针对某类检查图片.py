#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil

def extract_concrete_pump_truck_images(img_dir, label_dir, output_dir, label_name='5', limit=20):
    """
    从数据集中提取指定数量的包含特定标签的图像和对应的标签文件
    
    参数:
        img_dir: 图像文件目录
        label_dir: 标签文件目录
        output_dir: 输出目录
        label_name: 要查找的标签名称，默认为'5'
        limit: 要提取的图像数量限制，默认为20
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
    
    # 创建标签输出目录
    label_output_dir = os.path.join(os.path.dirname(output_dir), 'labels', os.path.basename(output_dir))
    if not os.path.exists(label_output_dir):
        os.makedirs(label_output_dir)
        print(f"已创建标签输出目录: {label_output_dir}")
    
    count = 0
    extracted_files = []
    
    # 遍历图像目录中的所有文件
    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_name = os.path.splitext(img_file)[0]
        label_file = os.path.join(label_dir, img_name + '.txt')
        
        # 检查对应的标签文件是否存在
        if os.path.isfile(label_file):
            with open(label_file, 'r') as f:
                # 查找是否包含指定标签
                has_target_label = False
                for line in f:
                    label = line.strip().split()[0]
                    if label == label_name:
                        has_target_label = True
                        break
                
                # 如果包含目标标签，复制图像和标签文件
                if has_target_label:
                    src_img = os.path.join(img_dir, img_file)
                    dst_img = os.path.join(output_dir, img_file)
                    
                    src_label = label_file
                    dst_label = os.path.join(label_output_dir, img_name + '.txt')
                    
                    shutil.copy(src_img, dst_img)
                    shutil.copy(src_label, dst_label)
                    
                    count += 1
                    extracted_files.append(img_file)
                    print(f"已提取 ({count}/{limit}): {img_file}")
                    
                    # 如果达到指定数量，停止提取
                    if count >= limit:
                        break
        
        # 如果达到指定数量，停止提取
        if count >= limit:
            break
    
    print(f"\n提取完成！共提取了 {count} 张包含标签 '{label_name}' 的图像及其标签文件")
    print(f"图像保存在: {output_dir}")
    print(f"标签保存在: {label_output_dir}")
    
    # 输出提取的文件列表
    if extracted_files:
        print("\n提取的文件列表:")
        for i, file in enumerate(extracted_files, 1):
            print(f"{i}. {file}")
    else:
        print("\n没有找到包含指定标签的图像")

# 使用示例
if __name__ == "__main__":
    img_dir = '/data_4T/dlg/yolov11_gaijin/datasets/images/test'
    label_dir = '/data_4T/dlg/yolov11_gaijin/datasets/labels/test'
    output_dir = '/data_4T/dlg/yolov11_gaijin/datasets/mix'
    
    # 提取20张包含标签为'5'的图像和对应的txt文件
    extract_concrete_pump_truck_images(img_dir, label_dir, output_dir, label_name='6', limit=200)