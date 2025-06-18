import os
import shutil
from pathlib import Path
import glob

def find_missed_detections(pre_txt_dir, label_txt_dir, pre_image_dir, label_image_dir, output_dir="missed_detections"):
    """
    检测漏检的图片并复制到输出目录
    
    参数:
    pre_txt_dir: 预测标签目录
    label_txt_dir: 真实标签目录  
    pre_image_dir: 预测图片目录
    label_image_dir: 真实标签对应的图片目录
    output_dir: 输出目录
    """
    
    # 创建输出目录
    output_pre_dir = os.path.join(output_dir, "pre_images")
    output_label_dir = os.path.join(output_dir, "label_images")
    os.makedirs(output_pre_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 获取所有真实标签文件
    label_files = glob.glob(os.path.join(label_txt_dir, "*.txt"))
    
    missed_count = 0
    total_with_labels = 0
    
    print(f"开始检测漏检图片...")
    print(f"真实标签目录: {label_txt_dir}")
    print(f"预测标签目录: {pre_txt_dir}")
    print("-" * 50)
    
    for label_file in label_files:
        label_filename = os.path.basename(label_file)
        label_name = os.path.splitext(label_filename)[0]
        
        # 对应的预测标签文件
        pre_file = os.path.join(pre_txt_dir, label_filename)
        
        # 检查真实标签是否有内容（即是否有目标）
        with open(label_file, 'r') as f:
            label_content = f.read().strip()
        
        if not label_content:  # 如果真实标签为空，跳过
            continue
            
        total_with_labels += 1
        
        # 检查预测标签是否存在且有内容
        pred_has_detection = False
        if os.path.exists(pre_file):
            with open(pre_file, 'r') as f:
                pred_content = f.read().strip()
            if pred_content:
                pred_has_detection = True
        
        # 如果真实标签有目标但预测没有检测到，则为漏检
        if not pred_has_detection:
            missed_count += 1
            print(f"漏检图片 {missed_count}: {label_name}")
            
            # 查找对应的图片文件（支持多种格式）
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            
            # 复制预测图片
            pre_image_copied = False
            for ext in image_extensions:
                pre_image_path = os.path.join(pre_image_dir, label_name + ext)
                if os.path.exists(pre_image_path):
                    shutil.copy2(pre_image_path, os.path.join(output_pre_dir, label_name + ext))
                    pre_image_copied = True
                    break
            
            # 复制真实标签对应的图片
            label_image_copied = False
            for ext in image_extensions:
                label_image_path = os.path.join(label_image_dir, label_name + ext)
                if os.path.exists(label_image_path):
                    shutil.copy2(label_image_path, os.path.join(output_label_dir, label_name + ext))
                    label_image_copied = True
                    break
            
            # 复制标签文件用于对比
            shutil.copy2(label_file, os.path.join(output_label_dir, label_filename))
            if os.path.exists(pre_file):
                shutil.copy2(pre_file, os.path.join(output_pre_dir, label_filename))
            else:
                # 创建空的预测标签文件
                with open(os.path.join(output_pre_dir, label_filename), 'w') as f:
                    pass
            
            if not pre_image_copied:
                print(f"  警告: 未找到预测图片 {label_name}")
            if not label_image_copied:
                print(f"  警告: 未找到真实标签图片 {label_name}")
    
    print("-" * 50)
    print(f"检测完成!")
    print(f"有目标的图片总数: {total_with_labels}")
    print(f"漏检图片数量: {missed_count}")
    print(f"漏检率: {missed_count/total_with_labels*100:.2f}%" if total_with_labels > 0 else "漏检率: 0%")
    print(f"漏检图片已保存到: {output_dir}")

def analyze_detection_stats(pre_txt_dir, label_txt_dir):
    """
    分析检测统计信息
    """
    label_files = glob.glob(os.path.join(label_txt_dir, "*.txt"))
    
    stats = {
        'total_images': len(label_files),
        'images_with_labels': 0,
        'images_with_predictions': 0,
        'missed_detections': 0,
        'correct_detections': 0,
        'false_positives': 0
    }
    
    for label_file in label_files:
        label_filename = os.path.basename(label_file)
        pre_file = os.path.join(pre_txt_dir, label_filename)
        
        # 检查真实标签
        with open(label_file, 'r') as f:
            label_content = f.read().strip()
        has_label = bool(label_content)
        
        # 检查预测标签
        has_prediction = False
        if os.path.exists(pre_file):
            with open(pre_file, 'r') as f:
                pred_content = f.read().strip()
            has_prediction = bool(pred_content)
        
        if has_label:
            stats['images_with_labels'] += 1
            if has_prediction:
                stats['correct_detections'] += 1
            else:
                stats['missed_detections'] += 1
        else:
            if has_prediction:
                stats['false_positives'] += 1
        
        if has_prediction:
            stats['images_with_predictions'] += 1
    
    return stats
if __name__ == "__main__":
    # 配置路径
    pre_txt_dir = "/data_4T/dlg/runs/detect/predict16/labels"
    label_txt_dir = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/labels/test"
    pre_image_dir = "/data_4T/dlg/runs/detect/predict16"  # 注意这里应该是images目录
    label_image_dir = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/images/test"
    
    # 首先分析统计信息
    print("正在分析检测统计信息...")
    stats = analyze_detection_stats(pre_txt_dir, label_txt_dir)
    
    print("=" * 60)
    print("检测统计信息:")
    print(f"总图片数: {stats['total_images']}")
    print(f"有真实标签的图片数: {stats['images_with_labels']}")
    print(f"有预测结果的图片数: {stats['images_with_predictions']}")
    print(f"正确检测的图片数: {stats['correct_detections']}")
    print(f"漏检的图片数: {stats['missed_detections']}")
    print(f"误检的图片数: {stats['false_positives']}")
    
    if stats['images_with_labels'] > 0:
        recall = stats['correct_detections'] / stats['images_with_labels'] * 100
        print(f"图像级召回率: {recall:.2f}%")
    
    if stats['images_with_predictions'] > 0:
        precision = stats['correct_detections'] / stats['images_with_predictions'] * 100
        print(f"图像级精确率: {precision:.2f}%")
    
    print("=" * 60)
    
    # 查找并复制漏检图片
    find_missed_detections(pre_txt_dir, label_txt_dir, pre_image_dir, label_image_dir)