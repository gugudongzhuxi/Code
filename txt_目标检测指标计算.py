# 图指标1：以图片为单位计算算法每一个类别召回和精度，其中图像中有一个被召回计入正确召回
# 框指标：以框为单位计计算算法每一个类别召回和精度

from pathlib import Path

# 统计每个样本图片的结果
# [{"标签1"：{GT,GTR,TP,FP}},{"标签2"：{GT,GTR,TP,FP}}] GT，GT召回，预测正确数，预测错误数
# 文件不存在定义为
def one_sample_result(gt_txt_path,pre_txt_path, conf_threshold=0.5, iou_threshold=0.1):
    # 计算预测框与GT框的IOU的函数
    def box_iou(pre_box, gt_box):

        # 计算交集的左上角和右下角坐标
        # pre_box = pre_box.astype(float)
        # gt_box = gt_box.astype(float)

        pre_cx,pre_cy,pre_w,pre_h = pre_box[:4]
        pre_x1 = pre_cx - pre_w / 2
        pre_y1 = pre_cy - pre_h / 2
        pre_x2 = pre_cx + pre_w / 2
        pre_y2 = pre_cy + pre_h / 2

        gt_cx, gt_cy, gt_w, gt_h = gt_box[:4]
        gt_x1 = gt_cx - gt_w / 2
        gt_y1 = gt_cy - gt_h / 2
        gt_x2 = gt_cx + gt_w / 2
        gt_y2 = gt_cy + gt_h / 2

        # 计算交集的左上角和右下角坐标
        inter_x1 = max(pre_x1, gt_x1)
        inter_y1 = max(pre_y1, gt_y1)
        inter_x2 = min(pre_x2, gt_x2)
        inter_y2 = min(pre_y2, gt_y2)
        # 计算交集的面积
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        # 计算预测框和GT框的面积
        pre_area = (pre_x2 - pre_x1) * (pre_y2 - pre_y1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        # 计算IOU
        iou = inter_area / float(pre_area + gt_area - inter_area)
        return iou

    # 读取预测结果
    pre_data=[]
    if not pre_txt_path is None and Path(pre_txt_path).exists():
        with open(pre_txt_path, 'r') as f:
            pre_data = f.readlines()
    # 读取真实结果
    gt_data=[]
    if not gt_txt_path is None and Path(gt_txt_path).exists():
        with open(gt_txt_path, 'r') as f:
            gt_data = f.readlines()

    # 初始化每个类别的统计结果
    result = {}
    for line in gt_data:
        label = line.split()[0]
        if label not in result:
            result[label] = {"GT": 0, "GTR": 0,"TP": 0, "FP": 0}
        result[label]["GT"] += 1
    for line in pre_data:
        label = line.split()[0]
        if label not in result:
            result[label] = {"GT": 0, "GTR": 0,"TP": 0, "FP": 0}

    # 如果数据中有置信度时，只保留大于conf_threshold的预测框
    pre_data_tmp = []
    for line in pre_data:
        line_list = line.split()
        if len(line_list) > 5:
            conf = float(line_list[5])
            if conf > conf_threshold:
                pre_data_tmp.append(line)
        else:
            pre_data_tmp.append(line)
    pre_data = pre_data_tmp


    # 计算每个类别的TP和FP
    for pre_line in pre_data:
        pre_label = pre_line.split()[0]
        pre_box = [float(x) for x in pre_line.split()[1:]]
        for gt_line in gt_data:
            gt_label = gt_line.split()[0]
            gt_box = [float(x) for x in gt_line.split()[1:]]
            if pre_label == gt_label and box_iou(pre_box, gt_box) > iou_threshold:
                result[pre_label]["TP"] += 1
                break
        else:
            result[pre_label]["FP"] += 1

    # 计算每个类别的GTR
    for gt_line in gt_data:
        gt_label = gt_line.split()[0]
        gt_box = [float(x) for x in gt_line.split()[1:]]
        for pre_line in pre_data:
            pre_label = pre_line.split()[0]
            pre_box = [float(x) for x in pre_line.split()[1:]]
            if pre_label == gt_label and box_iou(pre_box, gt_box) > iou_threshold:
                result[gt_label]["GTR"] += 1
                break
    return  result


# 计算文件夹下所有数据
from pathlib import Path
def all_sample_result(gt_txt_root,pre_txt_root, conf_threshold=0.5,iou_threshold=0.5):
    pre_txt_paths = Path(pre_txt_root).rglob("*.txt")
    gt_txt_paths = Path(gt_txt_root).rglob("*.txt")


    # 预测的txt数据路径字典
    pre_txt_path_dict = {str(path.stem): str(path) for path in pre_txt_paths}
    if "classes" in pre_txt_path_dict.keys():
        # 删除classes.txt
        pre_txt_path_dict.pop("classes")

    # GT的txt数据路径字典
    gt_txt_path_dict = {str(path.stem): str(path) for path in gt_txt_paths}
    if "classes" in gt_txt_path_dict.keys():
        # 删除classes.txt
        gt_txt_path_dict.pop("classes")

    # print(gt_txt_path_dict)
    # print(pre_txt_path_dict)

    # 关键字列表
    keys_set = set(pre_txt_path_dict.keys()) | set(gt_txt_path_dict.keys())
    all_key = list(keys_set)

    # 框指标和图指标
    box_all_result={}
    img_all_result={}
    # 便利所有图片
    for key in all_key:
        gt_txt_path = gt_txt_path_dict.get(key)
        pre_txt_path = pre_txt_path_dict.get(key)

        result = one_sample_result(gt_txt_path, pre_txt_path,conf_threshold, iou_threshold)

        # 框指标统计
        for k,v in result.items():
            box_all_result[k] = box_all_result.get(k, {"GT": 0, "GTR": 0,"TP": 0, "FP": 0})
            for kk, vv in v.items():
                box_all_result[k][kk] += vv

        # 图指标统计
        for k,v in result.items():
            img_all_result[k] = img_all_result.get(k, {"IMG_GT": 0, "IMG_GTR": 0,"IMG_FP": 0})
            box_gt = v.get("GT", 0)
            box_gtr = v.get("GTR", 0)
            box_fp = v.get("FP", 0)

            # 图片GT和召回
            img_all_result[k]["IMG_GT"] += int(box_gt>0)
            img_all_result[k]["IMG_GTR"] += int(box_gtr>0)

            # 图片FP:一个目标框都没被击中，并且有误检
            img_all_result[k]["IMG_FP"] += int(box_gtr==0 and box_fp>0)

    return box_all_result,img_all_result

# 统计与显示
def show_result(gt_txt_root="", pre_txt_root="", img_root="",iou_threshold=0.5):
    gt_txt_root = "/data/800工程机械/挂载/zy_GCJX_607/data/testing/val"
    pre_txt_root = "/data/800工程机械/挂载/zy_GCJX_607/data/testing/s"
    # GT的txt路径
    gt_txt_root = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+12519/labels/test"
    # 预测的txt路径
    pre_txt_root = "/data_4T/dlg/runs/detect/predict23/labels"
    # 图片根目录
    img_root = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+12519/images/test"

    conf_threshold = 0.1 # 置信度阈值
    iou_threshold = 0.1 # IOU阈值

    box_all_result,img_all_result = all_sample_result(gt_txt_root, pre_txt_root, conf_threshold,iou_threshold)

    # ---------------------------------- 框指标 -------------------------------------
    # 统计
    lines=[]
    for k,v in box_all_result.items():
        # GT、召回数、召回率
        box_GT = v.get("GT", 0)
        box_GTR = v.get("GTR", 0)
        box_R = 0 if box_GTR == 0 else float(box_GTR) / box_GT

        # TP、FP、精确率
        box_TP = v.get("TP", 0)
        box_FP = v.get("FP", 0)
        box_P = 0 if  box_TP==0 else float(box_TP) / (box_TP+box_FP)

        line = [k,box_GT,box_GTR,box_TP,box_FP,f"{round(box_R*100, 2)}%",f"{round(box_P*100, 2)}%"]

        lines.append(line)
    
    with open("box_result.cvs","w") as f:
        f.write(f"---------------------- 框指标 ---------------------img_root={img_root} iou_threshold={iou_threshold}\n")
        f.write("标签,框GT,框召回,正确框(TP),错误框(FP),召回率,精确率\n")
        print("----------------------------- 框指标 -----------------------------\n")
        print("标签   框GT    框召回    正确框(TP)  错误框(FP)  召回率  精确率")
        for line in lines:
            print(f"{line[0]:>2}",  # 标签
                  f"{line[1]:>7}",  # 框GT
                  f"{line[2]:>7}",  # 框召回
                  f"{line[3]:>11}",  # 正确框(TP)
                  f"{line[4]:>7}",  # 错误框(FP)
                  f"{line[5]:>11}",  # 召回率
                  f"{line[6]:>9}")  # 精确率

            line=','.join(map(str, line))
            f.write(line+"\n")
    print("------------------------------------------------------------------")

    # ---------------------------------- 图指标 -------------------------------------
    # 图片数据路径字典
    img_paths = list(Path(img_root).rglob("*.png"))
    img_paths += list(Path(img_root).rglob("*.jpg"))
    img_paths += list(Path(img_root).rglob("*.jpeg"))
    # img_path_dict = {str(path.stem): str(path) for path in img_paths}

    # 总的数据量
    img_total = len(img_paths)
    # print("img_all_result:", img_all_result)
    # print("img_total:", img_total)

    # 统计 IMG_GT': 1, 'IMG_GTR': 1, 'IMG_FP': 0
    lines=[]
    for k,v in img_all_result.items():
        # IMG_GT、IMG_GTR、召回率
        img_GT = v.get("IMG_GT", 0)
        img_GTR = v.get("IMG_GTR", 0)
        img_R = 0 if img_GTR == 0 else float(img_GTR) / img_GT

        # 图片TP、IMG_FP、精确率
        img_TP = img_GTR
        img_FP = v.get("IMG_FP", 0)
        img_P = 0 if  img_TP==0 else float(img_TP) / (img_TP+img_FP)

        line = [k,img_GT,img_total-img_GT,img_GTR,img_FP,f"{round(img_R*100, 2)}%",f"{round(img_P*100, 2)}%"]

        lines.append(line)
    with open("img_result.csv","w") as f:
        f.write(f"---------------------- 图指标 ---------------------conf_threshold={conf_threshold} iou_threshold={iou_threshold}\n")
        f.write("标签,GT图,非GT图,图召回,误报图,召回率,精确率\n")
        print(f"----------------------------- 图指标 -----------------------------conf_threshold={conf_threshold} iou_threshold={iou_threshold}")
        print("标签   GT图  非GT图   图召回   误报图  召回率   精确率")
        for line in lines:
            print(f"{line[0]:>2}",  # 标签
                  f"{line[1]:>7}",  # GT图
                  f"{line[2]:>7}",  # 非GT图
                  f"{line[3]:>7}",  # 图召回
                  f"{line[4]:>7}",  # 误报图
                  f"{line[5]:>7}",  # 召回率
                  f"{line[6]:>9}")  # 精确率

            line=','.join(map(str, line))
            f.write(line+"\n")
    print("------------------------------------------------------------------ ")
# show_result()

def show_result2(gt_txt_root="", pre_txt_root="", img_root="",iou_threshold=0.5):
    # gt_txt_root = "/data/800工程机械/挂载/zy_GCJX_607/data/testing/val"
    # pre_txt_root = "/data/800工程机械/挂载/zy_GCJX_607/data/testing/s"
    # GT的txt路径
    gt_txt_root = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/labels/test"
    # 预测的txt路径
    pre_txt_root = "/data_4T/dlg/ultralytics-main1/runs/detect/predict5/labels"
    # 图片根目录
    img_root = "/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/images/test"

    conf_threshold = 0.1# 置信度阈值
    iou_threshold = 0.01# 评估的IOU阈值

    box_all_result,img_all_result = all_sample_result(gt_txt_root, pre_txt_root, conf_threshold,iou_threshold)

    # ---------------------------------- 框指标 -------------------------------------
    # 统计
    box_lines=[]
    for k,v in box_all_result.items():
        # GT、召回数、召回率
        box_GT = v.get("GT", 0)
        box_GTR = v.get("GTR", 0)
        box_R = 0 if box_GTR == 0 else float(box_GTR) / box_GT

        # TP、FP、精确率
        box_TP = v.get("TP", 0)
        box_FP = v.get("FP", 0)
        box_P = 0 if  box_TP==0 else float(box_TP) / (box_TP+box_FP)

        line = [k,box_GT,box_GTR,box_TP,box_FP,f"{round(box_R*100, 2)}%",f"{round(box_P*100, 2)}%"]

        box_lines.append(line)
    # 框指标排序
    box_lines.sort(key=lambda x: int(x[0]))
    # ---------------------------------- 图指标 -------------------------------------
    # 图片数据路径字典
    img_paths = list(Path(img_root).rglob("*.png"))
    img_paths += list(Path(img_root).rglob("*.jpg"))
    img_paths += list(Path(img_root).rglob("*.jpeg"))
    # img_path_dict = {str(path.stem): str(path) for path in img_paths}

    # 总的数据量
    img_total = len(img_paths)
    # print("img_all_result:", img_all_result)
    # print("img_total:", img_total)

    # 统计 IMG_GT': 1, 'IMG_GTR': 1, 'IMG_FP': 0
    img_lines=[]
    for k,v in img_all_result.items():
        # IMG_GT、IMG_GTR、召回率
        img_GT = v.get("IMG_GT", 0)
        img_GTR = v.get("IMG_GTR", 0)
        img_R = 0 if img_GTR == 0 else float(img_GTR) / img_GT

        # 图片TP、IMG_FP、精确率
        img_TP = img_GTR
        img_FP = v.get("IMG_FP", 0)
        img_P = 0 if  img_TP==0 else float(img_TP) / (img_TP+img_FP)

        line = [k,img_GT,img_total-img_GT,img_GTR,img_FP,f"{round(img_R*100, 2)}%",f"{round(img_P*100, 2)}%"]

        img_lines.append(line)

    # 图指标排序
    img_lines.sort(key=lambda x: int(x[0]))

    with open("index_result.csv","w") as f:
        # 框指标
        f.write(f"---------------------- 框指标 ---------------------conf_threshold={conf_threshold} iou_threshold={iou_threshold}\n")
        f.write("标签,框GT,框召回,正确框(TP),错误框(FP),召回率,精确率\n")
        print("----------------------------- 框指标 -----------------------------\n")
        print("标签   框GT    框召回    正确框(TP)  错误框(FP)  召回率  精确率")
        for line in box_lines:
            print(f"{line[0]:>2}",  # 标签
                  f"{line[1]:>7}",  # 框GT
                  f"{line[2]:>7}",  # 框召回
                  f"{line[3]:>11}",  # 正确框(TP)
                  f"{line[4]:>7}",  # 错误框(FP)
                  f"{line[5]:>11}",  # 召回率
                  f"{line[6]:>9}")  # 精确率

            line = ','.join(map(str, line))
            f.write(line + "\n")
        print("------------------------------------------------------------------")

        # 图指标
        f.write(f"---------------------- 图指标 ---------------------conf_threshold={conf_threshold} iou_threshold={iou_threshold}\n")
        f.write("标签,GT图,非GT图,图召回,误报图,召回率,精确率\n")
        print(f"----------------------------- 图指标 -----------------------------conf_threshold={conf_threshold} iou_threshold={iou_threshold}")
        print("标签   GT图  非GT图   图召回   误报图  召回率   精确率")
        for line in img_lines:
            print(f"{line[0]:>2}",  # 标签
                  f"{line[1]:>7}",  # GT图
                  f"{line[2]:>7}",  # 非GT图
                  f"{line[3]:>7}",  # 图召回
                  f"{line[4]:>7}",  # 误报图
                  f"{line[5]:>7}",  # 召回率
                  f"{line[6]:>9}")  # 精确率

            line=','.join(map(str, line))
            f.write(line+"\n")
    print("------------------------------------------------------------------ ")

    return  img_lines, box_lines

show_result2()

