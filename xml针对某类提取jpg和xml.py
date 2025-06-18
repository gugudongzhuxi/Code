#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import xml.etree.ElementTree as ET

# 源目录路径
src_dir = "/data_4T/dlg/datasets/测试集与训练集/6537/6537删去CombTruck"
# 目标目录路径（用于存放提取的文件）
target_dir = "/data_4T/dlg/datasets/ConcretePumpTruck/6537训练_1"

# 创建目标目录（如果不存在）
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print(f"创建目标目录: {target_dir}")

# 用于统计找到的文件数量
xml_count = 0
jpg_count = 0

# 遍历源目录
for root, _, files in os.walk(src_dir):
    for file in files:
        # 检查XML文件
        if file.endswith('.xml'):
            xml_path = os.path.join(root, file)
            try:
                # 解析XML文件
                tree = ET.parse(xml_path)
                root_elem = tree.getroot()
                
                # 查找name元素
                name_elements = root_elem.findall('.//name')
                
                # 检查是否包含'ConcretePumpTruck'
                has_concrete_pump_truck = any(name_elem.text == 'ConcretePumpTruck' for name_elem in name_elements)
                
                if has_concrete_pump_truck:
                    # 复制XML文件到目标目录
                    shutil.copy2(xml_path, os.path.join(target_dir, file))
                    xml_count += 1
                    
                    # 查找对应的JPG文件
                    jpg_filename = os.path.splitext(file)[0] + '.jpg'
                    jpg_path = os.path.join(root, jpg_filename)
                    
                    # 如果JPG文件存在，也复制它
                    if os.path.exists(jpg_path):
                        shutil.copy2(jpg_path, os.path.join(target_dir, jpg_filename))
                        jpg_count += 1
                    else:
                        print(f"警告: 找不到对应的JPG文件: {jpg_filename}")
            
            except ET.ParseError:
                print(f"错误: 无法解析XML文件: {xml_path}")
            except Exception as e:
                print(f"处理文件时出错 {xml_path}: {str(e)}")

print(f"完成! 共提取了 {xml_count} 个XML文件和 {jpg_count} 个JPG文件到 {target_dir}")