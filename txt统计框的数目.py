import os

# 定义类别信息
categories = {
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

# 初始化计数字典
counts = {cat: 0 for cat in categories.values()}

# 指定txt文件目录
txt_dir = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/labels/test"

# 遍历目录下所有txt文件
for txt_file in os.listdir(txt_dir):
    if txt_file.endswith(".txt"):
        file_path = os.path.join(txt_dir, txt_file)

        # 读取txt文件内容
        with open(file_path, "r") as f:
            lines = f.readlines()

        # 统计每个类别的个数
        for line in lines:
            category_id = int(line.split()[0])
            category_name = categories[category_id]
            counts[category_name] += 1

# 打印结果
for category, count in counts.items():
    print(f"{category}: {count}")