import os

# 定义11类别的映射关系
categories_11 = {
    0: "CraneTruckExtend",
    1: "TowerCrane",
    2: "DumpTruck",
    3: "Excavator",
    4: "ConcretePump",
    5: "ConcretePumpTruck",
    6: "MixerTruck",
    7: "PileDriver",
    8: "Bulldozer",
    9: "LadderTruck",
    10: "CraneTruckFold"
}

# 定义7类别的映射关系
categories_7 = {
    0: "Bulldozer",          # 原Bulldozer
    1: "DumpTruck",          # 原DumpTruck
    2: "TowerCrane",         # 原TowerCrane
    3: "CraneTruck",         # 合并CraneTruckExtend、LadderTruck和CraneTruckFold
    4: "Excavator",          # 原Excavator
    5: "PileDriver",         # 原PileDriver
    6: "ConcretePump"        # 合并ConcretePump、ConcretePumpTruck和MixerTruck
}

def check_labels(txt_dir, mode='7'):
    """检查标签文件
    
    Args:
        txt_dir: 标签文件目录
        mode: '7'表示7类别模式, '11'表示11类别模式
    """
    # 选择类别映射
    if mode == '7':
        categories = categories_7
        max_category_id = 6
        print(f"使用7类别模式检查: {txt_dir}")
    else:  # mode == '11'
        categories = categories_11
        max_category_id = 10
        print(f"使用11类别模式检查: {txt_dir}")
    
    # 初始化计数字典
    counts = {cat: 0 for cat in categories.values()}
    label_counts = {str(i): 0 for i in range(max_category_id + 1)}

    # 检查目录是否存在
    if not os.path.exists(txt_dir):
        print(f"错误：标签目录 {txt_dir} 不存在！")
        return
    
    # 统计文件数量
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
    print(f"找到 {len(txt_files)} 个标签文件")
    
    # 初始化不合法标签计数
    invalid_labels = {}
    empty_files = []
    total_objects = 0
    
    # 遍历目录下所有txt文件
    for txt_file in txt_files:
        file_path = os.path.join(txt_dir, txt_file)
        file_empty = True
        
        # 读取txt文件内容
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # 统计每个类别的个数
        for line in lines:
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            file_empty = False
            total_objects += 1
            parts = line.split()
            
            # 检查标签格式
            if len(parts) < 5:  # YOLO格式应该至少有5个值：类别和4个边界框坐标
                print(f"警告：文件 {txt_file} 中发现不完整的行: {line}")
                continue
                
            try:
                category_id = int(parts[0])
                
                # 检查标签ID是否在有效范围内
                if 0 <= category_id <= max_category_id:
                    label_counts[str(category_id)] += 1
                    category_name = categories[category_id]
                    counts[category_name] += 1
                else:
                    # 记录不合法标签
                    if parts[0] not in invalid_labels:
                        invalid_labels[parts[0]] = 0
                    invalid_labels[parts[0]] += 1
                    print(f"警告：文件 {txt_file} 中发现无效的标签ID: {parts[0]}")
            except ValueError:
                print(f"错误：文件 {txt_file} 中标签ID不是整数: {parts[0]}")
        
        # 记录空文件
        if file_empty:
            empty_files.append(txt_file)
    
    # 打印结果
    print(f"\n========== {mode}类别标签检查结果 ==========")
    print(f"总标签文件数: {len(txt_files)}")
    print(f"总目标物体数: {total_objects}")
    print(f"空文件数: {len(empty_files)}")
    
    print("\n按标签ID统计:")
    for label_id, count in sorted(label_counts.items(), key=lambda x: int(x[0])):
        category_name = categories.get(int(label_id), "未知")
        print(f"标签 {label_id} ({category_name}): {count} 个")
    
    print("\n按类别名称统计:")
    for category, count in counts.items():
        print(f"{category}: {count}")
    
    if invalid_labels:
        print("\n发现不合法的标签ID:")
        for label, count in invalid_labels.items():
            print(f"标签 {label}: {count} 个")
    
    if empty_files:
        print("\n空文件列表:")
        for empty_file in empty_files[:10]:  # 只打印前10个
            print(empty_file)
        if len(empty_files) > 10:
            print(f"... 以及其他 {len(empty_files) - 10} 个文件")

    return empty_files  # 返回空文件列表，以便后续处理

# 在这里直接设置路径和模式
# 11类别标签路径
txt_dir_11 = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/labels/train"
# 7类别标签路径
txt_dir_7 = "/data_4T/zy_GCJX_607/data/ag_flip/ag_训练_filp_ConcretePump/7_txt"

# 选择要运行的模式和路径
# 将mode设为'7'检查7类别标签，设为'11'检查11类别标签
mode = '11'  # 默认为11类别模式，可以根据需要改为'7'

# 根据所选模式确定使用的目录
if mode == '7':
    txt_dir = txt_dir_7
else:  # mode == '11'
    txt_dir = txt_dir_11

# 执行检查
empty_files = check_labels(txt_dir, mode)