import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.transforms import Resize, Grayscale, ToPILImage, ToTensor, Compose
from PIL import Image  # 导入 PIL.Image
import numpy as np  # 导入 numpy
import os
import cv2  



def save_image(image, filename):
    """保存图像"""
    if isinstance(image, torch.Tensor):
        image = image.numpy().transpose(1, 2, 0)  # CHW -> HWC
        if image.shape[2] == 1:  # 如果是单通道图像，移除最后一个维度
            image = image[:, :, 0]
    elif isinstance(image, Image.Image):
        image = np.array(image)
    
    # 如果图像是浮点类型，将其缩放到 [0, 255] 并转换为 uint8
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    
    # 确保目录存在
    output_dir = "/data_4T/dlg/code"
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果是灰度图像，需要特别处理以正确保存
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        cv2.imwrite(os.path.join(output_dir, filename), image)
    else:
        cv2.imwrite(os.path.join(output_dir, filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# 定义图像处理流程
transform = Compose([
    ToPILImage(),  # 先转换成 PIL 图像
    Resize((256, 256)),  # 强制大小为 256x256
    Grayscale(),
    ToTensor()  # 然后转换回 Tensor
])

# 加载单张图像
image_path = '/data_4T/dlg/code/对角卷积/954c02ace1b64150a6d87535155aa342.jpg'
image = read_image(image_path)  # 使用 torchvision 读取图像 (C, H, W)
image = image.float() / 255.0   # 归一化到 [0,1]
image = transform(image)        # 应用变换
image = image.unsqueeze(0)      # 添加 batch 维度 -> shape: [1, C, H, W]

print("图像形状：", image.shape)  # torch.Size([1, 1, 256, 256])


class Model3x3(nn.Module):
    def __init__(self):
        super(Model3x3, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(7, 7))

    def forward(self, x):
        return self.conv3x3(x)


class Model3x1and1x3(nn.Module):
    def __init__(self):
        super(Model3x1and1x3, self).__init__()
        self.conv3x1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 7))
        self.conv1x3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(7, 1))

    def forward(self, x):
        x = self.conv3x1(x)
        x = self.conv1x3(x)
        return x


model3x3 = Model3x3()
model3x1and1x3 = Model3x1and1x3()

# 推理
with torch.no_grad():
    outputs3x3 = model3x3(image)
    print("outputs3x3 shape:", outputs3x3.shape)  # torch.Size([1, 1, 250, 250])
    output3x3 = outputs3x3[0, 0].cpu().numpy()
    save_image(output3x3, '对角output3x3.png')

    # outputs3x1and1x3 = model3x1and1x3(image)
    # print("outputs3x1and1x3 shape:", outputs3x1and1x3.shape)
    # output3x1and1x3 = outputs3x1and1x3[0, 0].cpu().numpy()
    # save_image(output3x1and1x3, 'outputs3x1and1x3.png')

    # diff = output3x3 - output3x1and1x3
    # print("diff mean:", diff.mean())
    # print("diff max:", diff.max())
    # save_image(diff, 'diff3.png')