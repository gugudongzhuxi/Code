import os
import glob

# jpg图片文件夹路径
img_dir = '/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/images/train'

# 输出的txt文件路径
output_file = '/data_4T/dlg/ultralytics-main1/datasets_11+6700+5622+6362+6064+13527/train.txt'

# 获取文件夹中所有'.jpg'图片的路径
img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))

# 将图片路径写入txt文件
with open(output_file, 'w') as f:
    for path in img_paths:
        f.write(path + '\n')

print(f"所有jpg图片的路径已经写入到{output_file}")