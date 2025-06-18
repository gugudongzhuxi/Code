import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path

# 标签中英文对照表
LABEL_MAPPING = {
    'Bulldoze': '推土机',
    'DumpTruck': '翻斗车',
    'TowerCrane': '塔吊',
    'Excavator': '挖掘机',
    'PileDriver': '打桩机',
    'ConcretePump': '水泥浆车',
    'CraneTruck': '吊车'
}

def xml_to_json(xml_path, output_dir=None):
    """将XML标注文件转换为JSON格式(带中文标签)"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 解析XML数据
        size = root.find('size')
        
        # 构建JSON数据结构
        json_data = {
            "version": "5.5.0",
            "flags": {},
            "shapes": [],
            "imagePath": root.find('filename').text,
            "imageData": None,
            "imageHeight": int(size.find('height').text),
            "imageWidth": int(size.find('width').text)
        }
        
        # 处理所有object标签
        for obj in root.findall('object'):
            en_label = obj.find('name').text
            zh_label = LABEL_MAPPING.get(en_label, en_label)  # 如果找不到映射则保持原标签
            
            bndbox = obj.find('bndbox')
            
            shape = {
                "label": zh_label,  # 使用中文标签
                "points": [
                    [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text)],
                    [int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }
            json_data["shapes"].append(shape)
        
        # 设置输出路径
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / (xml_path.stem + '.json')
        else:
            output_path = xml_path.with_suffix('.json')
        
        # 保存JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return True
    
    except Exception as e:
        print(f"转换失败 {xml_path}: {str(e)}")
        return False

def batch_convert(xml_dir, output_dir=None):
    """批量转换XML到JSON(带中文标签)"""
    xml_dir = Path(xml_dir)
    if output_dir is None:
        output_dir = xml_dir.parent / (xml_dir.name + '_json_zh')
    
    xml_files = list(xml_dir.glob('*.xml'))
    total = len(xml_files)
    success = 0
    
    print("使用以下标签映射关系:")
    for en, zh in LABEL_MAPPING.items():
        print(f"  {en} -> {zh}")
    
    print(f"\n开始转换: 共找到 {total} 个XML文件")
    
    for i, xml_path in enumerate(xml_files, 1):
        if xml_to_json(xml_path, output_dir):
            success += 1
        
        if i % 10 == 0 or i == total:
            print(f"\r处理进度: {i}/{total} | 成功: {success} 失败: {i-success}", end="", flush=True)
    
    print(f"\n转换完成! 成功转换 {success}/{total} 个文件")
    print(f"中文标签JSON文件已保存到: {output_dir}")

if __name__ == "__main__":
    # 使用示例
    xml_directory = "/data_4T/dlg/datasets/测试集与训练集/测试集/测试集_7"
    output_directory = "/data_4T/dlg/datasets/测试集与训练集/测试集/测试集_7/json_zh"
    
    batch_convert(xml_directory, output_directory)