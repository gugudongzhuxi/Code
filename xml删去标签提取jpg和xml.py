import os
import xml.etree.ElementTree as ET
from shutil import copy2
from pathlib import Path

# 原始路径和新路径
src_dir = Path("/data_4T/dlg/datasets/测试集与训练集/201/201原图-xml")
dst_dir = Path("/data_4T/dlg/datasets/测试集与训练集/201/201_删去CombTruck")

# 确保目标目录存在
dst_dir.mkdir(parents=True, exist_ok=True)

def contains_combtruck(xml_path):
    """检查XML是否包含CombTruck标签"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is not None and name_elem.text == 'CombTruck':
                return True  # 包含CombTruck
        return False  # 不包含CombTruck
    except ET.ParseError as e:
        print(f"解析错误: {xml_path} - {str(e)}")
        return True  # 解析失败也当作包含，避免误复制损坏文件

def process_dataset():
    """处理整个数据集"""
    xml_files = list(src_dir.glob('*.xml'))
    total_files = len(xml_files)
    saved_xml = 0
    saved_jpg = 0
    
    print(f"开始处理: 共扫描到 {total_files} 个XML文件")
    
    for i, xml_path in enumerate(xml_files, 1):
        # 获取对应的图片路径
        img_extensions = ['.jpg', '.png', '.jpeg']
        img_path = None
        for ext in img_extensions:
            possible_path = xml_path.with_suffix(ext)
            if possible_path.exists():
                img_path = possible_path
                break
        
        # 判断是否包含 CombTruck
        if contains_combtruck(xml_path):
            # 包含 CombTruck，跳过复制
            continue
        
        # 不包含 CombTruck，才复制 XML 和图片
        try:
            # 复制 XML
            copy2(xml_path, dst_dir / xml_path.name)
            saved_xml += 1
            
            # 复制图片（如果存在）
            if img_path and img_path.exists():
                copy2(img_path, dst_dir / img_path.name)
                saved_jpg += 1
        except Exception as e:
            print(f"文件复制失败: {xml_path} - {str(e)}")
            continue
        
        # 进度显示
        if i % 100 == 0 or i == total_files:
            print(f"\r处理进度: {i}/{total_files} | 已保存: {saved_xml} XML, {saved_jpg} JPG", end="", flush=True)
    
    print(f"\n处理完成！最终保存: {saved_xml} 个XML文件, {saved_jpg} 个图片文件")

if __name__ == "__main__":
    process_dataset()