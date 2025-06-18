#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

def check_matching_files(images_dir, labels_dir):
    """
    检查图片目录和标签目录中的文件是否一一对应
    
    参数:
        images_dir: 图片目录路径
        labels_dir: 标签目录路径
    """
    # 检查目录是否存在
    if not os.path.exists(images_dir):
        print(f"错误: 图片目录不存在 - {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"错误: 标签目录不存在 - {labels_dir}")
        return
    
    # 获取所有jpg文件名（不含扩展名）
    jpg_files = set()
    for file in os.listdir(images_dir):
        if file.lower().endswith(('.jpg', '.jpeg')):
            jpg_files.add(os.path.splitext(file)[0])
    
    # 获取所有xml文件名（不含扩展名）
    xml_files = set()
    for file in os.listdir(labels_dir):
        if file.lower().endswith('.xml'):
            xml_files.add(os.path.splitext(file)[0])
    
    # 计算统计数据
    total_jpg = len(jpg_files)
    total_xml = len(xml_files)
    jpg_without_xml = jpg_files - xml_files
    xml_without_jpg = xml_files - jpg_files
    matching_files = jpg_files.intersection(xml_files)
    
    # 打印统计结果
    print(f"\n===== 文件匹配检查报告 =====")
    print(f"图片目录: {images_dir}")
    print(f"标签目录: {labels_dir}")
    print(f"JPG文件总数: {total_jpg}")
    print(f"XML文件总数: {total_xml}")
    print(f"匹配的文件数: {len(matching_files)}")
    print(f"没有对应XML的JPG文件数: {len(jpg_without_xml)}")
    print(f"没有对应JPG的XML文件数: {len(xml_without_jpg)}")
    
    # 打印没有对应XML的JPG文件
    if jpg_without_xml:
        print("\n===== 没有对应XML的JPG文件 =====")
        for i, filename in enumerate(sorted(jpg_without_xml), 1):
            print(f"{i}. {filename}.jpg")
    
    # 打印没有对应JPG的XML文件
    if xml_without_jpg:
        print("\n===== 没有对应JPG的XML文件 =====")
        for i, filename in enumerate(sorted(xml_without_jpg), 1):
            print(f"{i}. {filename}.xml")
    
    # 输出总结
    if not jpg_without_xml and not xml_without_jpg:
        print("\n结论: 所有文件均完美匹配!")
    else:
        print("\n结论: 存在不匹配的文件，请检查上述列表。")

def main():
    # 设置图片和标签目录路径
    images_dir = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622/images/train"
    labels_dir = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622/labels/train"
    
    print(f"正在检查图片和标签文件是否匹配...")
    check_matching_files(images_dir, labels_dir)

if __name__ == "__main__":
    main()