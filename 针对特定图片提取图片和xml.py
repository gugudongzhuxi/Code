#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import sys

def extract_files(source_dir, target_dir, jpg_list):
    """
    从指定目录提取特定的jpg文件和对应的xml文件到目标目录
    
    参数:
        source_dir: 源目录，包含jpg和xml文件
        target_dir: 目标目录，将提取的文件放在这里
        jpg_list: 需要提取的jpg文件名列表
    """
    # 创建目标目录（如果不存在）
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"已创建目标目录: {target_dir}")
    
    # 遍历需要提取的jpg文件
    for jpg_file in jpg_list:
        # 构建jpg文件的完整路径
        jpg_path = os.path.join(source_dir, jpg_file)
        
        # 构建对应xml文件的名称和路径
        xml_file = os.path.splitext(jpg_file)[0] + ".xml"
        xml_path = os.path.join(source_dir, xml_file)
        
        # 复制jpg文件
        if os.path.exists(jpg_path):
            shutil.copy2(jpg_path, target_dir)
            print(f"已复制: {jpg_file} -> {target_dir}")
        else:
            print(f"错误: 找不到jpg文件 {jpg_path}")
        
        # 复制xml文件
        if os.path.exists(xml_path):
            shutil.copy2(xml_path, target_dir)
            print(f"已复制: {xml_file} -> {target_dir}")
        else:
            print(f"错误: 找不到xml文件 {xml_path}")

def main():
    # 源目录
    source_dir = "/data_4T/dlg/datasets/测试集与训练集/5622/测试5622_11删去combtruck/xml_img"
    
    # 创建目标目录（当前目录下的extracted_files文件夹）
    target_dir = os.path.join(os.getcwd(), "extracted_files")
    
    # 需要提取的jpg文件列表
    jpg_list = [
        "174036166783320250222-153457-934.jpg",
        "174036166800020250222-153424-177.jpg",
        "174036423592220250221-143509-131.jpg",
        "174090776420320250302-155159-925.jpg",
        "SCSZSL2005STD0044_2_255__1576918926305198176_2024_11_22_15_51_07.jpg"
    ]
    
    # 提取文件
    extract_files(source_dir, target_dir, jpg_list)
    
    print(f"\n文件提取完成! 所有文件都已被复制到: {target_dir}")

if __name__ == "__main__":
    main()