#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import sys

def extract_xml_files(source_dir, target_dir):
    """
    从源目录提取所有XML文件到目标目录
    
    参数:
        source_dir: 源目录，包含XML文件
        target_dir: 目标目录，将提取的XML文件放在这里
    """
    # 确保源目录存在
    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在 - {source_dir}")
        return False
    
    # 创建目标目录（如果不存在）
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"已创建目标目录: {target_dir}")
    
    # 用于计数找到并复制的文件数量
    xml_count = 0
    
    # 遍历源目录及其所有子目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 检查文件是否为XML文件
            if file.lower().endswith('.xml'):
                # 构建完整的源文件路径
                source_file = os.path.join(root, file)
                # 构建目标文件路径
                target_file = os.path.join(target_dir, file)
                
                # 复制文件
                try:
                    shutil.copy2(source_file, target_file)
                    xml_count += 1
                    if xml_count % 100 == 0:  # 每复制100个文件输出一次进度
                        print(f"已复制 {xml_count} 个XML文件...")
                except Exception as e:
                    print(f"复制文件时出错 {source_file}: {str(e)}")
    
    print(f"\n完成! 共复制了 {xml_count} 个XML文件到 {target_dir}")
    return True

def main():
    # 源目录 - 包含要提取的XML文件的目录
    source_dir = "/data_4T/dlg/datasets/6537训练ConcretePumpTruck"
    
    # 目标目录 - 提取的XML文件将被复制到这里
    # 这里使用与源目录同级的一个新目录
    target_dir = "/data_4T/dlg/datasets/6537训练ConcretePumpTruck/xml"
    
    print(f"开始从 {source_dir} 提取XML文件到 {target_dir}...")
    
    # 提取XML文件
    extract_xml_files(source_dir, target_dir)

if __name__ == "__main__":
    main()