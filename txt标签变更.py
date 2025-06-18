import os

# 原始目录路径
input_dir = '/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/labels/train_11'
# 输出目录路径（原路径后加 _替换）
output_dir = '/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/labels/train'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 遍历目录下所有 .txt 文件
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                parts = line.strip().split()
                if parts:  # 非空行
                    label = parts[0]
                    if label == '4':
                        parts[0] = '0'  # 替换标签4为0
                    # 写入修改后的行
                    outfile.write(' '.join(parts) + '\n')
        print(f"已处理文件：{filename}")

print("✅ 所有文件处理完成，结果已保存至：", output_dir)