import os
import shutil
import random
import xml.etree.ElementTree as ET

# 指定筛选出的图片和标注文件目录
data_dir = 'selected_data/'

# 指定需要筛选的类别
target_classes = ['CraneTruckFold', 'CraneTruckExtend', 'ConcretePump', 'ConcretePumpTruck']

# 测试集比例
test_ratio = 0.15

# 创建训练集和测试集文件夹
train_dir = 'data/train/'
test_dir = 'data/test/'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 获取所有XML文件名
xml_files = [f for f in os.listdir(data_dir) if f.endswith('.xml')]

# 随机打乱XML文件顺序
random.shuffle(xml_files)

# 初始化计数器
total_count = len(xml_files)
test_count = int(total_count * test_ratio)
current_test_count = 0
current_class_counts = {cls: 0 for cls in target_classes}

# 遍历XML文件
for xml_filename in xml_files:
    # 构建对应的JPG文件路径
    jpg_filename = xml_filename[:-4] + '.jpg'
    jpg_path = os.path.join(data_dir, jpg_filename)
    
    # 检查对应的JPG文件是否存在
    if os.path.exists(jpg_path):
        # 解析XML文件
        tree = ET.parse(os.path.join(data_dir, xml_filename))
        root = tree.getroot()
        
        # 获取图片尺寸
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # 初始化标注信息
        lines = []
        
        # 获取图片包含的类别
        image_classes = set()
        
        # 获取图片包含的类别和标注框
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in target_classes:
                image_classes.add(class_name)
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # 转换为YOLO格式的标注
                x_center = (xmin + xmax) / (2 * width)
                y_center = (ymin + ymax) / (2 * height)
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height
                
                class_index = target_classes.index(class_name)
                lines.append(f"{class_index} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
        
        # 判断图片是否划分到测试集
        if current_test_count < test_count:
            # 如果当前类别计数小于平均值,则优先将其划分到测试集
            avg_count = test_count // len(target_classes)
            if any(current_class_counts[cls] < avg_count for cls in image_classes):
                output_dir = test_dir
                current_test_count += 1
                for cls in image_classes:
                    current_class_counts[cls] += 1
            else:
                output_dir = train_dir
        else:
            output_dir = train_dir
        
        # 复制图片文件到相应的文件夹
        jpg_dest = os.path.join(output_dir, jpg_filename)
        shutil.copy(jpg_path, jpg_dest)
        
        # 将标注信息写入TXT文件
        txt_filename = xml_filename[:-4] + '.txt'
        txt_dest = os.path.join(output_dir, txt_filename)
        with open(txt_dest, 'w') as f:
            f.writelines(lines)

# 统计训练集图片数量
train_images = [f for f in os.listdir(train_dir) if f.endswith('.jpg')]
print(f"训练集图片数量: {len(train_images)}")

# 统计测试集图片数量
test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
print(f"测试集图片数量: {len(test_images)}")

# 统计测试集中每个类别的图片数量
print("\n测试集中每个类别的图片数量:")
for cls in target_classes:
    cls_images = [f for f in test_images if cls in f]
    print(f"{cls}: {len(cls_images)}")