import os
import cv2
import numpy as np
import random
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm



# ===== 修改这里的配置 =====
INPUT_DIR = "/data_4T/dlg/datasets/ConcretePumpTruck/ag_6357"  # 替换为你的输入目录
OUTPUT_DIR = "/data_4T/dlg/datasets/ConcretePumpTruck/ag_6357"  # 替换为你的输出目录
AUGMENT_TYPES = "color"  # 可选: "color", "flip", "rotate", "scale", "noise", "blur", "all" 或组合如 "color,flip,rotate"
# 高级配置
MAX_ROTATION_ANGLE = 30  # 随机旋转的最大角度 (±度数)
SCALE_RANGE = (2, 3)  # 随机缩放范围
TRANSLATE_PERCENT = 0.1  # 随机平移范围 (图像尺寸的百分比)
NOISE_LEVEL = 20         # 高斯噪声水平 (0-255)
BLUR_MAX_SIZE = 3        # 最大模糊核大小


# =========================

def mkdir_if_not_exists(dir_path):
    """如果目录不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def prettify(elem):
    """将XML树转换为格式化的字符串"""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def update_xml_for_flip(xml_content, flip_direction, image_width, image_height):
    """根据翻转方向更新XML内容中的边界框坐标"""
    root = ET.fromstring(xml_content)

    # 更新图像尺寸（如果有变化）
    size = root.find('size')
    if size is not None:
        width = size.find('width')
        height = size.find('height')
        if width is not None and height is not None:
            width.text = str(image_width)
            height.text = str(image_height)

    # 更新每个边界框的坐标
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        if bbox is not None:
            xmin = bbox.find('xmin')
            xmax = bbox.find('xmax')
            ymin = bbox.find('ymin')
            ymax = bbox.find('ymax')

            if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
                xmin_val = int(float(xmin.text))
                xmax_val = int(float(xmax.text))
                ymin_val = int(float(ymin.text))
                ymax_val = int(float(ymax.text))

                if 'h' in flip_direction:  # 水平翻转
                    new_xmin = image_width - xmax_val
                    new_xmax = image_width - xmin_val
                    xmin.text = str(new_xmin)
                    xmax.text = str(new_xmax)
                if 'v' in flip_direction:  # 垂直翻转
                    new_ymin = image_height - ymax_val
                    new_ymax = image_height - ymin_val
                    ymin.text = str(new_ymin)
                    ymax.text = str(new_ymax)

    return prettify(root)


def apply_random_flip(image, xml_content, p_horizontal=1, p_vertical=0.3):
    """应用随机翻转并返回处理后的图像和XML内容"""
    flip_direction = None
    height, width = image.shape[:2]

    # 水平翻转
    if random.random() < p_horizontal:
        image = cv2.flip(image, 1)
        flip_direction = 'h'

    # # 垂直翻转（可以与水平翻转同时应用）
    # if random.random() < p_vertical:
    #     image = cv2.flip(image, 0)
    #     flip_direction = 'v' if flip_direction is None else flip_direction + 'v'

    # 如果有翻转，则更新XML
    if flip_direction is not None:
        xml_content = update_xml_for_flip(xml_content, flip_direction, width, height)

    return image, xml_content


def adjust_color(image, brightness=1.0, contrast=1.0, saturation=1.0, hue=0.0):
    """调整图像的颜色空间"""
    # 转换为HSV空间进行饱和度和色调调整
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)

    # 饱和度调整
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    # 色调调整
    hsv[..., 0] = (hsv[..., 0] + hue * 180) % 180
    hsv = np.clip(hsv, 0, 255)
    hsv = hsv.astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 亮度和对比度调整
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness * 255 * (1 - contrast))

    return image


def apply_random_color_jitter(image):
    """应用随机颜色变换"""
    brightness = random.uniform(0.8, 1.2)  # 亮度调整范围±20%
    contrast = random.uniform(0.8, 1.2)  # 对比度调整范围±20%
    saturation = random.uniform(0.8, 1.2)  # 饱和度调整范围±20%
    hue = random.uniform(-0.1, 0.1)  # 色相调整范围±0.1

    return adjust_color(image, brightness, contrast, saturation, hue)


def add_gaussian_noise(image, level=20):
    """向图像添加高斯噪声"""
    row, col, ch = image.shape
    mean = 0
    sigma = level
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


def apply_random_blur(image):
    """应用随机模糊"""
    kernel_size = random.choice([3, 5, 7])
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return image


def rotate_point(point, center, angle_rad):
    """围绕中心点旋转一个点"""
    x, y = point
    cx, cy = center

    # 转换到以中心点为原点的坐标系
    x_shifted = x - cx
    y_shifted = y - cy

    # 旋转
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    x_rotated = x_shifted * cos_theta - y_shifted * sin_theta
    y_rotated = x_shifted * sin_theta + y_shifted * cos_theta

    # 转换回原始坐标系
    x_final = x_rotated + cx
    y_final = y_rotated + cy

    return (x_final, y_final)


def rotate_image_and_update_xml(image, xml_content, angle):
    """旋转图像并更新XML中的边界框"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # 计算旋转后保证所有内容可见的新图像大小
    max_dim = int(math.sqrt(width ** 2 + height ** 2)) + 10

    # 创建旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # 调整旋转矩阵以移动中心
    rotation_matrix[0, 2] += (max_dim / 2) - center[0]
    rotation_matrix[1, 2] += (max_dim / 2) - center[1]

    # 应用旋转
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (max_dim, max_dim),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

    # 解析XML
    root = ET.fromstring(xml_content)

    # 更新图像尺寸
    size = root.find('size')
    if size is not None:
        width_elem = size.find('width')
        height_elem = size.find('height')
        if width_elem is not None and height_elem is not None:
            width_elem.text = str(max_dim)
            height_elem.text = str(max_dim)

    # 旋转角度转弧度
    angle_rad = math.radians(-angle)  # 负号是因为OpenCV的旋转方向与数学旋转方向相反

    new_center = (max_dim // 2, max_dim // 2)

    # 更新每个边界框的坐标
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        if bbox is not None:
            xmin_elem = bbox.find('xmin')
            xmax_elem = bbox.find('xmax')
            ymin_elem = bbox.find('ymin')
            ymax_elem = bbox.find('ymax')

            if (xmin_elem is not None and xmax_elem is not None and
                    ymin_elem is not None and ymax_elem is not None):

                # 获取原始坐标
                xmin = float(xmin_elem.text)
                xmax = float(xmax_elem.text)
                ymin = float(ymin_elem.text)
                ymax = float(ymax_elem.text)

                # 计算原始边界框的四个角点
                points = [
                    (xmin, ymin),  # 左上
                    (xmax, ymin),  # 右上
                    (xmax, ymax),  # 右下
                    (xmin, ymax)  # 左下
                ]

                # 应用旋转矩阵到每个点
                new_points = []
                for point in points:
                    # 应用旋转矩阵
                    x = point[0] * rotation_matrix[0, 0] + point[1] * rotation_matrix[0, 1] + rotation_matrix[0, 2]
                    y = point[0] * rotation_matrix[1, 0] + point[1] * rotation_matrix[1, 1] + rotation_matrix[1, 2]
                    new_points.append((x, y))

                # 计算新的边界框坐标（旋转后的最小外接矩形）
                xs = [p[0] for p in new_points]
                ys = [p[1] for p in new_points]
                new_xmin = min(xs)
                new_xmax = max(xs)
                new_ymin = min(ys)
                new_ymax = max(ys)

                # 更新XML中的坐标值
                xmin_elem.text = str(int(new_xmin))
                xmax_elem.text = str(int(new_xmax))
                ymin_elem.text = str(int(new_ymin))
                ymax_elem.text = str(int(new_ymax))

    return rotated_image, prettify(root)


def scale_image_and_update_xml(image, xml_content, scale_factor):
    """缩放图像并更新XML中的边界框"""
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # 缩放图像
    scaled_image = cv2.resize(image, (new_width, new_height),
                              interpolation=cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LINEAR)

    # 解析XML
    root = ET.fromstring(xml_content)

    # 更新图像尺寸
    size = root.find('size')
    if size is not None:
        width_elem = size.find('width')
        height_elem = size.find('height')
        if width_elem is not None and height_elem is not None:
            width_elem.text = str(new_width)
            height_elem.text = str(new_height)

    # 更新每个边界框的坐标
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        if bbox is not None:
            xmin_elem = bbox.find('xmin')
            xmax_elem = bbox.find('xmax')
            ymin_elem = bbox.find('ymin')
            ymax_elem = bbox.find('ymax')

            if (xmin_elem is not None and xmax_elem is not None and
                    ymin_elem is not None and ymax_elem is not None):
                # 获取原始坐标
                xmin = float(xmin_elem.text)
                xmax = float(xmax_elem.text)
                ymin = float(ymin_elem.text)
                ymax = float(ymax_elem.text)

                # 更新坐标
                xmin_elem.text = str(int(xmin * scale_factor))
                xmax_elem.text = str(int(xmax * scale_factor))
                ymin_elem.text = str(int(ymin * scale_factor))
                ymax_elem.text = str(int(ymax * scale_factor))

    return scaled_image, prettify(root)


def translate_image_and_update_xml(image, xml_content, tx, ty):
    """平移图像并更新XML中的边界框"""
    height, width = image.shape[:2]

    # 创建平移矩阵
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

    # 应用平移
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))

    # 解析XML
    root = ET.fromstring(xml_content)

    # 更新每个边界框的坐标
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        if bbox is not None:
            xmin_elem = bbox.find('xmin')
            xmax_elem = bbox.find('xmax')
            ymin_elem = bbox.find('ymin')
            ymax_elem = bbox.find('ymax')

            if (xmin_elem is not None and xmax_elem is not None and
                    ymin_elem is not None and ymax_elem is not None):
                # 获取原始坐标
                xmin = float(xmin_elem.text)
                xmax = float(xmax_elem.text)
                ymin = float(ymin_elem.text)
                ymax = float(ymax_elem.text)

                # 更新坐标
                xmin_elem.text = str(int(xmin + tx))
                xmax_elem.text = str(int(xmax + tx))
                ymin_elem.text = str(int(ymin + ty))
                ymax_elem.text = str(int(ymax + ty))

    return translated_image, prettify(root)


def generate_augmented_images(image, xml_content, augment_types):
    """根据选择的增强类型生成增强后的图像和XML"""
    results = []

    # 原始图像和XML（作为基准）
    base_image = image.copy()
    base_xml = xml_content
    height, width = base_image.shape[:2]

    # 应用选定的增强类型
    if 'flip' in augment_types:
        flip_image, flip_xml = apply_random_flip(base_image.copy(), base_xml)
        results.append(('flip', flip_image, flip_xml))

    if 'color' in augment_types:
        color_image = apply_random_color_jitter(base_image.copy())
        results.append(('color', color_image, base_xml))  # XML不变

    if 'rotate' in augment_types:
        # 随机旋转角度
        angle = random.uniform(-MAX_ROTATION_ANGLE, MAX_ROTATION_ANGLE)
        rotate_image, rotate_xml = rotate_image_and_update_xml(base_image.copy(), base_xml, angle)
        results.append(('rotate', rotate_image, rotate_xml))

    if 'scale' in augment_types:
        # 随机缩放因子
        scale_factor = random.uniform(SCALE_RANGE[0], SCALE_RANGE[1])
        scale_image, scale_xml = scale_image_and_update_xml(base_image.copy(), base_xml, scale_factor)
        results.append(('scale', scale_image, scale_xml))

    if 'translate' in augment_types:
        # 随机平移
        tx = int(width * random.uniform(-TRANSLATE_PERCENT, TRANSLATE_PERCENT))
        ty = int(height * random.uniform(-TRANSLATE_PERCENT, TRANSLATE_PERCENT))
        translate_image, translate_xml = translate_image_and_update_xml(base_image.copy(), base_xml, tx, ty)
        results.append(('translate', translate_image, translate_xml))

    if 'noise' in augment_types:
        noise_image = add_gaussian_noise(base_image.copy(), NOISE_LEVEL)
        results.append(('noise', noise_image, base_xml))  # XML不变

    if 'blur' in augment_types:
        blur_image = apply_random_blur(base_image.copy())
        results.append(('blur', blur_image, base_xml))  # XML不变

    if 'both' in augment_types:
        # 先翻转再颜色变换
        both_image, both_xml = apply_random_flip(base_image.copy(), base_xml)
        both_image = apply_random_color_jitter(both_image)
        results.append(('both', both_image, both_xml))

    if 'all' in augment_types:
        # 综合应用多种增强
        all_image = base_image.copy()
        all_xml = base_xml

        # 随机应用几种变换组合
        transformations = [
            ('flip', lambda img, xml: apply_random_flip(img, xml)),
            ('color', lambda img, xml: (apply_random_color_jitter(img), xml)),
            ('rotate', lambda img, xml: rotate_image_and_update_xml(img, xml, random.uniform(-MAX_ROTATION_ANGLE / 2,
                                                                                             MAX_ROTATION_ANGLE / 2))),
            ('scale',
             lambda img, xml: scale_image_and_update_xml(img, xml, random.uniform(SCALE_RANGE[0], SCALE_RANGE[1]))),
            ('noise', lambda img, xml: (add_gaussian_noise(img, NOISE_LEVEL / 2), xml))
        ]

        # 随机选择2-3种变换
        num_transforms = random.randint(2, 3)
        selected_transforms = random.sample(transformations, num_transforms)

        for _, transform_func in selected_transforms:
            all_image, all_xml = transform_func(all_image, all_xml)

        results.append(('all', all_image, all_xml))

    return results


def process_image_and_xml(input_dir, output_dir, img_file, augment_types):
    """处理单个图像及其XML文件"""
    # 读取图像
    img_path = os.path.join(input_dir, img_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"无法读取图像: {img_path}")
        return

    # 读取XML文件
    xml_file = os.path.splitext(img_file)[0] + '.xml'
    xml_path = os.path.join(input_dir, xml_file)
    if not os.path.exists(xml_path):
        print(f"未找到对应的XML文件: {xml_path}")
        return

    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()

    # 生成增强后的图像和XML
    augmented_results = generate_augmented_images(image, xml_content, augment_types)

    # 保存增强后的结果
    for aug_type, aug_image, aug_xml in augmented_results:
        aug_img_file = os.path.splitext(img_file)[0] + f'_{aug_type}.jpg'
        aug_xml_file = os.path.splitext(img_file)[0] + f'_{aug_type}.xml'

        cv2.imwrite(os.path.join(output_dir, aug_img_file), aug_image)
        with open(os.path.join(output_dir, aug_xml_file), 'w', encoding='utf-8') as f:
            f.write(aug_xml)


def parse_augment_types(augment_str):
    """解析增强类型字符串"""
    valid_types = {'color', 'flip', 'rotate', 'scale', 'translate', 'noise', 'blur', 'both', 'all'}
    selected_types = set(augment_str.lower().split(','))

    # 验证输入的有效性
    invalid_types = selected_types - valid_types
    if invalid_types:
        raise ValueError(f"无效的增强类型: {', '.join(invalid_types)}. 请选择: {', '.join(valid_types)}")

    return selected_types


def main():
    """主函数"""
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    augment = AUGMENT_TYPES

    try:
        # 检查输入目录是否存在
        if not os.path.exists(input_dir):
            print(f"错误: 输入目录 '{input_dir}' 不存在")
            return

        # 解析增强类型
        augment_types = parse_augment_types(augment)

        # 创建输出目录
        mkdir_if_not_exists(output_dir)

        # 获取所有JPG文件
        img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg'))]

        if not img_files:
            print(f"警告: 在输入目录 '{input_dir}' 中没有找到JPG图像")
            return

        print(f"共找到 {len(img_files)} 个JPG图像")
        print(f"应用增强类型: {', '.join(augment_types)}")

        # 处理每个图像
        for img_file in tqdm(img_files, desc="处理图像中"):
            try:
                process_image_and_xml(input_dir, output_dir, img_file, augment_types)
            except Exception as e:
                print(f"处理 {img_file} 时出错: {str(e)}")

        print(f"处理完成。结果保存在: {output_dir}")

    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()