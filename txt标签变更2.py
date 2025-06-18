import os

# 原始目录路径
input_dir = '/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/labels/test'

# 定义替换映射表
replace_map = {
    '5': '4',
    '6': '5',
    '7': '6',
    '8': '7',
    '9': '8',
    '10': '9'
}

# 遍历目录下所有 .txt 文件
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_dir, filename)

        # 读取并处理文件内容
        with open(file_path, 'r', encoding='utf-8') as infile:
            lines = []
            for line in infile:
                parts = line.strip().split()
                if parts:
                    label = parts[0]
                    if label in replace_map:
                        parts[0] = replace_map[label]
                    lines.append(' '.join(parts) + '\n')

        # 直接覆盖写回原文件
        with open(file_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(lines)

        print(f"已处理并覆盖文件：{filename}")

print("✅ 所有文件处理完成，原始文件已被直接覆盖。")