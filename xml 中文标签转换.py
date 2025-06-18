import os
import xml.etree.ElementTree as ET

# 中文标签到英文标签的映射
label_mapping = {
    "翻斗车": "DumpTruck",
    "吊车臂": "CraneTruckExtend",
    "搅拌车": "MixerTruck",
    "塔吊": "TowerCrane"
}


def modify_xml_files(directory):
    """
    遍历指定目录中的 XML 文件，将 name 标签中的中文标签转换为英文标签。

    :param directory: 包含 XML 文件的目录路径
    """
    total_files = 0  # 统计遍历的文件总数
    modified_files = 0  # 统计修改的文件数量

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):  # 只处理 XML 文件
            file_path = os.path.join(directory, filename)
            total_files += 1

            try:
                # 解析 XML 文件
                tree = ET.parse(file_path)
                root = tree.getroot()

                # 查找所有 <name> 标签
                modified = False
                for name_tag in root.iter("name"):
                    if name_tag.text in label_mapping:
                        print(f"Replacing '{name_tag.text}' with '{label_mapping[name_tag.text]}' in {filename}")
                        name_tag.text = label_mapping[name_tag.text]
                        modified = True

                # 如果有修改，则保存更改后的 XML 文件
                if modified:
                    tree.write(file_path, encoding="utf-8", xml_declaration=True)
                    modified_files += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # 打印统计结果
    print(f"Total XML files processed: {total_files}")
    print(f"Total XML files modified: {modified_files}")


# 示例用法
if __name__ == "__main__":
    # 指定包含 XML 文件的目录路径
    xml_directory = "/industai_data/dlg/工程器械/挂载/datasets/测试集与训练集/背景复杂样本"

    # 调用函数修改 XML 文件
    modify_xml_files(xml_directory)