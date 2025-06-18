import os
import glob

# 定义标签映射关系
label_mapping = {
    '0': '3',  # CraneTruckExtend → 3
    '1': '2',  # TowerCrane → 2
    '2': '1',  # DumpTruck → 1
    '3': '4',  # Excavator → 4
    '4': '6',  # ConcretePump → 6
    '5': '6',  # ConcretePumpTruck → 6
    '6': '6',  # MixerTruck → 6
    '7': '5',  # PileDriver → 5
    '8': '0',  # Bulldoze → 0
    '9': '3',  # LadderTruck → 3
    '10': '3'  # CraneTruckFold → 3
}

# 指定标签文件目录路径
labels_dir = '/data_4T/dlg/datasets/测试集与训练集/201/201_11_删去CombTruck/txt'
# 指定合并后标签文件的保存目录路径
output_dir = '/data_4T/dlg/datasets/测试集与训练集/201/201_11_删去CombTruck/txt_7'

# 如果输出目录不存在，创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取所有txt文件
txt_files = glob.glob(os.path.join(labels_dir, '*.txt'))
print(f"找到{len(txt_files)}个标签文件")

# 处理每个txt文件
for file_path in txt_files:
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:  # 跳过空行
            continue
            
        old_label = parts[0]
        # 如果标签在映射表中，则替换为新标签
        if old_label in label_mapping:
            parts[0] = label_mapping[old_label]
            new_line = ' '.join(parts) + '\n'
            new_lines.append(new_line)
        else:
            print(f"警告：文件 {file_name} 中发现未知标签: {old_label}")
            # 仍然保留原始行
            new_lines.append(line)
    
    # 写入新文件
    with open(output_path, 'w') as f:
        f.writelines(new_lines)

print(f"标签合并完成！合并后的标签文件已保存到: {output_dir}")

# 统计新标签分布
new_label_counts = {str(i): 0 for i in range(7)}  # 初始化标签计数器 (0-6)

for file_path in glob.glob(os.path.join(output_dir, '*.txt')):
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                label = parts[0]
                if label in new_label_counts:
                    new_label_counts[label] += 1

print("\n合并后的标签分布:")
for label, count in new_label_counts.items():
    print(f"标签 {label}: {count} 个")