import os
import random
import xml.etree.ElementTree as ET
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class MosaicAugmentation:
    def __init__(self, output_size=640):
        self.output_size = output_size  # 输出图片大小（正方形）

    def get_object_regions(self, img, xml_path):
        """获取所有标注框的有效区域"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        regions = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # 确保裁剪区域包含目标
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            offset = random.randint(-100, 100)
            new_xmin = max(0, center_x - 300 + offset)
            new_ymin = max(0, center_y - 300 + offset)
            new_xmax = min(img.shape[1], center_x + 300 + offset)
            new_ymax = min(img.shape[0], center_y + 300 + offset)

            regions.append((new_xmin, new_ymin, new_xmax, new_ymax))

        return regions

    def smart_crop(self, img, regions):
        """智能裁剪：优先选择包含目标的区域"""
        if not regions:
            # 无标注时返回随机区域
            h, w = img.shape[:2]
            x = random.randint(0, w - 600)
            y = random.randint(0, h - 600)
            return img[y:y + 600, x:x + 600]

        # 从候选区域中随机选择一个
        selected = random.choice(regions)
        xmin, ymin, xmax, ymax = selected
        return img[ymin:ymax, xmin:xmax]

    def generate_mosaic(self, image_paths, xml_paths):
        """生成 Mosaic 图片和新的 XML 文件"""
        mosaic_img = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
        mosaic_root = ET.Element("annotation")
        for i, (img_path, xml_path) in enumerate(zip(image_paths, xml_paths)):
            # 读取图片和标注文件
            img = cv2.imread(img_path)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取裁剪区域
            regions = self.get_object_regions(img, xml_path)
            crop_img = self.smart_crop(img, regions)

            # 自适应缩放
            h, w = crop_img.shape[:2]
            scale = min(self.output_size // 2 / max(h, w), 1.0)
            resized_img = cv2.resize(crop_img, (int(w * scale), int(h * scale)))

            # 计算拼接位置
            x_offset = (i % 2) * (self.output_size // 2)
            y_offset = (i // 2) * (self.output_size // 2)
            dx = random.randint(-50, 50)
            dy = random.randint(-50, 50)
            x1 = max(0, x_offset + dx)
            y1 = max(0, y_offset + dy)
            x2 = min(x1 + resized_img.shape[1], self.output_size)
            y2 = min(y1 + resized_img.shape[0], self.output_size)

            # 执行贴图
            mosaic_img[y1:y2, x1:x2] = resized_img[0:(y2 - y1), 0:(x2 - x1)]

            # 更新标注框坐标
            for obj in root.findall('object'):
                new_obj = deepcopy(obj)
                bbox = new_obj.find('bndbox')

                # 原始坐标转换
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                # 裁剪偏移补偿
                if i < len(regions):
                    region = regions[i]
                    xmin -= region[0]
                    ymin -= region[1]
                    xmax -= region[0]
                    ymax -= region[1]

                # 缩放比例补偿
                xmin *= scale
                ymin *= scale
                xmax *= scale
                ymax *= scale

                # 拼接位置补偿
                xmin += x1
                ymin += y1
                xmax += x1
                ymax += y1

                # 边界保护
                xmin = max(0, min(xmin, self.output_size))
                ymin = max(0, min(ymin, self.output_size))
                xmax = max(0, min(xmax, self.output_size))
                ymax = max(0, min(ymax, self.output_size))

                # 过滤无效标注
                if (xmax - xmin) < 10 or (ymax - ymin) < 10:
                    continue

                # 更新坐标
                bbox.find('xmin').text = str(int(xmin))
                bbox.find('ymin').text = str(int(ymin))
                bbox.find('xmax').text = str(int(xmax))
                bbox.find('ymax').text = str(int(ymax))
                mosaic_root.append(new_obj)

        return mosaic_img, mosaic_root

    def save_results(self, mosaic_img, mosaic_root, output_img_path, output_xml_path):
        """保存 Mosaic 图片和 XML 文件"""
        # 保存图片
        cv2.imwrite(output_img_path, mosaic_img)

        # 保存 XML 文件
        tree = ET.ElementTree(mosaic_root)
        tree.write(output_xml_path)

    def visualize(self, mosaic_img, mosaic_root):
        """可视化生成的 Mosaic 图片和标注框"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB))

        for obj in mosaic_root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.title("Generated Mosaic Image with Bounding Boxes")
        plt.axis("off")
        plt.show()


def find_matching_files(directory):
    """从指定目录中提取匹配的图片和标注文件路径"""
    jpg_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    xml_files = [f for f in os.listdir(directory) if f.endswith('.xml')]

    # 提取文件名（去掉后缀）
    jpg_names = {os.path.splitext(f)[0] for f in jpg_files}
    xml_names = {os.path.splitext(f)[0] for f in xml_files}

    # 找到交集
    common_names = jpg_names & xml_names

    # 构造完整路径
    image_paths = [os.path.join(directory, name + '.jpg') for name in common_names]
    xml_paths = [os.path.join(directory, name + '.xml') for name in common_names]

    return image_paths, xml_paths


# 示例用法
if __name__ == "__main__":
    # 输入数据目录
    data_directory = "/data/dlg/工程器械/挂载/datasets/ConcretePumpTruck/all"

    # 查找匹配的图片和标注文件
    image_paths, xml_paths = find_matching_files(data_directory)

    # 随机选择4组文件
    if len(image_paths) < 4:
        raise ValueError("目录中的有效图片-标注对不足4组，请检查数据！")
    selected_indices = random.sample(range(len(image_paths)), 4)
    selected_image_paths = [image_paths[i] for i in selected_indices]
    selected_xml_paths = [xml_paths[i] for i in selected_indices]

    # 初始化 Mosaic 增强器
    mosaic_aug = MosaicAugmentation(output_size=640)

    # 生成 Mosaic 图片和 XML 文件
    mosaic_img, mosaic_root = mosaic_aug.generate_mosaic(selected_image_paths, selected_xml_paths)

    # 保存结果
    mosaic_aug.save_results(mosaic_img, mosaic_root, "mosaic_output.jpg", "mosaic_output.xml")

    # 可视化结果
    mosaic_aug.visualize(mosaic_img, mosaic_root)