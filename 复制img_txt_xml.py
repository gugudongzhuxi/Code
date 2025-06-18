import os
import shutil


def copy_files(src_dir, dst_dir, file_type="both"):
    """
    复制指定类型的文件（.jpg、.txt 或 .xml）从源目录到目标目录。

    参数:
        src_dir (str): 源目录路径。
        dst_dir (str): 目标目录路径。
        file_type (str): 要复制的文件类型，可选值为 "jpg", "txt", "xml", "jpg+xml", 或 "both"。
                         默认值为 "both"，表示同时复制 .jpg 和对应的 .txt/.xml 文件。
    """
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)

    # 遍历源目录中的所有文件
    for filename in os.listdir(src_dir):
        # 构建源文件路径
        src_path = os.path.join(src_dir, filename)

        # 根据文件类型进行处理
        if file_type == "jpg" and filename.endswith(".jpg"):
            # 如果只复制 .jpg 文件
            dst_path = os.path.join(dst_dir, filename)
            shutil.copy2(src_path, dst_path)

        elif file_type == "txt" and filename.endswith(".txt"):
            # 如果只复制 .txt 文件
            dst_path = os.path.join(dst_dir, filename)
            shutil.copy2(src_path, dst_path)

        elif file_type == "xml" and filename.endswith(".xml"):
            # 如果只复制 .xml 文件
            dst_path = os.path.join(dst_dir, filename)
            shutil.copy2(src_path, dst_path)

        elif file_type == "jpg+xml":
            # 如果同时复制 .jpg 和 .xml 文件
            if filename.endswith(".jpg"):
                # 复制 .jpg 文件
                dst_path_jpg = os.path.join(dst_dir, filename)
                shutil.copy2(src_path, dst_path_jpg)

                # 获取对应的 .xml 文件名
                xml_filename = os.path.splitext(filename)[0] + ".xml"
                src_path_xml = os.path.join(src_dir, xml_filename)

                # 如果对应的 .xml 文件存在，则复制
                if os.path.exists(src_path_xml):
                    dst_path_xml = os.path.join(dst_dir, xml_filename)
                    shutil.copy2(src_path_xml, dst_path_xml)

        elif file_type == "both":
            # 如果同时复制 .jpg 和对应的 .txt/.xml 文件
            if filename.endswith(".jpg"):
                # 复制 .jpg 文件
                dst_path_jpg = os.path.join(dst_dir, filename)
                shutil.copy2(src_path, dst_path_jpg)

                # 获取对应的 .txt 文件名
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                src_path_txt = os.path.join(src_dir, txt_filename)

                # 如果对应的 .txt 文件存在，则复制
                if os.path.exists(src_path_txt):
                    dst_path_txt = os.path.join(dst_dir, txt_filename)
                    shutil.copy2(src_path_txt, dst_path_txt)

                # 获取对应的 .xml 文件名
                xml_filename = os.path.splitext(filename)[0] + ".xml"
                src_path_xml = os.path.join(src_dir, xml_filename)

                # 如果对应的 .xml 文件存在，则复制
                if os.path.exists(src_path_xml):
                    dst_path_xml = os.path.join(dst_dir, xml_filename)
                    shutil.copy2(src_path_xml, dst_path_xml)

    print(f"复制完成! 文件类型: {file_type}")


# 示例用法
if __name__ == "__main__":
    # 源目录和目标目录路径
    src_dir = "/data_4T/dlg/datasets/测试集与训练集/201/201_11_删去CombTruck/txt_7"
    dst_dir = "/data_4T/zy_GCJX_607/data/datasets_7+6700+5622+6362+6064+new/train_7"

    # 选择要复制的文件类型
    # 可选值: "jpg", "txt", "xml", "jpg+xml", "both"
    file_type = "txt"  # 修改这里以选择不同的文件类型

    # 执行复制操作
    copy_files(src_dir, dst_dir, file_type)