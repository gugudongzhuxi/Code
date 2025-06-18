import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.transforms import Resize, Grayscale, ToPILImage, ToTensor, Compose
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

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
    
    # 确保目录存在 - 自动选择可用目录
    possible_dirs = [
        "/data_4T/dlg/code/对角卷积",
        "./output",  # 当前目录下的output文件夹
        ".",  # 当前目录
        os.path.expanduser("~/Desktop"),  # 桌面
        "/tmp"  # 临时目录
    ]
    
    output_dir = None
    for dir_path in possible_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            # 测试是否可写
            test_file = os.path.join(dir_path, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            output_dir = dir_path
            break
        except:
            continue
    
    if output_dir is None:
        print("警告：找不到可写目录，图像可能无法保存")
        return
    
    file_path = os.path.join(output_dir, filename)
    
    # 如果是灰度图像，需要特别处理以正确保存
    try:
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            cv2.imwrite(file_path, image)
        else:
            cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"已保存: {file_path}")
    except Exception as e:
        print(f"保存图像失败 {filename}: {e}")

# 加载单张图像 - 自动处理路径问题
def load_image():
    # 可能的图像路径列表
    possible_paths = [
        '/data_4T/dlg/code/对角卷积/954c02ace1b64150a6d87535155aa342.jpg',
        # './20250529-102704.jpg',  # 当前目录
        # '../20250529-102704.jpg',  # 上级目录
    ]
    
    # 尝试找到图像文件
    for path in possible_paths:
        if os.path.exists(path):
            print(f"找到图像文件: {path}")
            return read_image(path)
    
    # 如果找不到文件，创建一个测试图像 - 模拟建筑工地场景
    print("未找到指定图像文件，创建建筑工地测试图像...")
    # 创建一个256x256的测试图像，包含建筑工地元素
    test_img = torch.zeros(3, 256, 256)
    
    # 添加建筑工地特征
    for i in range(256):
        for j in range(256):
            # 吊车臂 - 水平方向
            if 80 <= i <= 85 and 50 <= j <= 200:  # 水平吊车臂
                test_img[:, i, j] = 0.9
            if 60 <= i <= 65 and 150 <= j <= 220:  # 另一个水平臂
                test_img[:, i, j] = 0.8
                
            # 塔吊 - 垂直方向  
            elif 50 <= i <= 180 and 120 <= j <= 125:  # 垂直塔身
                test_img[:, i, j] = 0.7
            elif 30 <= i <= 200 and 75 <= j <= 80:   # 另一个垂直结构
                test_img[:, i, j] = 0.6
                
            # 挖掘机臂 - 对角线方向
            elif abs(i - j + 50) < 3 and 100 <= i <= 180 and 50 <= j <= 130:  # 斜臂1
                test_img[:, i, j] = 1.0
            elif abs(i + j - 300) < 3 and 120 <= i <= 180 and 120 <= j <= 180:  # 斜臂2
                test_img[:, i, j] = 0.9
                
            # 混凝土泵管 - 水平延展
            elif 150 <= i <= 155 and 30 <= j <= 180:
                test_img[:, i, j] = 0.5
                
            # 钢筋 - 垂直排列
            elif j % 20 == 0 and 200 <= i <= 240:
                test_img[:, i, j] = 0.4
                
            # 添加一些背景噪声
            else:
                test_img[:, i, j] = 0.1 + 0.1 * torch.rand(1)
    
    return (test_img * 255).byte()

# 定义图像处理流程
transform = Compose([
    ToPILImage(),
    Resize((256, 256)),
    Grayscale(),
    ToTensor()
])

# 加载图像
image = load_image()
image = image.float() / 255.0  # 归一化到 [0,1]
image = transform(image)  # 应用变换
image = image.unsqueeze(0)  # 添加 batch 维度 -> shape: [1, C, H, W]
print("图像形状：", image.shape)

class DirectionalConvModel(nn.Module):
    """
    三方向卷积模型：适用于建筑工地设备检测
    """
    def __init__(self, dim=1):
        super(DirectionalConvModel, self).__init__()
        
        print("初始化三方向卷积模型...")
        
        # 水平方向：适合吊车臂、混凝土泵管等横向延展设备
        self.conv_horizontal = nn.Conv2d(
            dim, dim, 
            kernel_size=(1, 7), 
            padding=(0, 3), 
            padding_mode='reflect'
        )
        
        # 垂直方向：适合塔吊等纵向延展设备
        self.conv_vertical = nn.Conv2d(
            dim, dim, 
            kernel_size=(7, 1), 
            padding=(3, 0), 
            padding_mode='reflect'
        )
        
        # 对角线方向：适合斜置的挖掘机臂等
        self.conv_diagonal = nn.Conv2d(
            dim, dim, 
            kernel_size=5, 
            padding=4, 
            dilation=2, 
            padding_mode='reflect'
        )
        
        # 初始化权重为更有意义的模式
        self._init_directional_weights()
    
    def _init_directional_weights(self):
        """初始化方向性权重"""
        with torch.no_grad():
            # 水平卷积核：强调水平边缘
            h_kernel = torch.tensor([[[[-1, -1, 0, 1, 1, 1, 0]]]], dtype=torch.float32)
            self.conv_horizontal.weight.data = h_kernel
            self.conv_horizontal.bias.data.fill_(0)
            
            # 垂直卷积核：强调垂直边缘
            v_kernel = torch.tensor([[[[-1], [-1], [0], [1], [1], [1], [0]]]], dtype=torch.float32)
            self.conv_vertical.weight.data = v_kernel
            self.conv_vertical.bias.data.fill_(0)
            
            # 对角卷积核：强调对角边缘 (使用膨胀卷积)
            d_kernel = torch.tensor([[[[1, 0, 0, 0, -1],
                                      [0, 1, 0, -1, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, -1, 0, 1, 0],
                                      [-1, 0, 0, 0, 1]]]], dtype=torch.float32)
            self.conv_diagonal.weight.data = d_kernel
            self.conv_diagonal.bias.data.fill_(0)
    
    def forward(self, x):
        # 分别提取三个方向的特征
        horizontal_features = self.conv_horizontal(x)
        vertical_features = self.conv_vertical(x)
        diagonal_features = self.conv_diagonal(x)
        
        return horizontal_features, vertical_features, diagonal_features
    
    def get_kernel_info(self):
        """获取卷积核信息"""
        print("\n" + "="*60)
        print("三方向卷积核详细信息:")
        print("="*60)
        
        print("\n🔄 水平方向卷积核 (1×7) - 检测水平设备:")
        print("用途：吊车臂、混凝土泵管等横向延展设备")
        h_kernel = self.conv_horizontal.weight.data.squeeze().numpy()
        print("形状:", h_kernel.shape)
        print("权重:", h_kernel)
        
        print("\n🔄 垂直方向卷积核 (7×1) - 检测垂直设备:")
        print("用途：塔吊等纵向延展设备")
        v_kernel = self.conv_vertical.weight.data.squeeze().numpy()
        print("形状:", v_kernel.shape)
        print("权重:", v_kernel.reshape(-1, 1))
        
        print("\n🔄 对角线方向卷积核 (5×5, 膨胀=2) - 检测斜向设备:")
        print("用途：斜置的挖掘机臂等")
        d_kernel = self.conv_diagonal.weight.data.squeeze().numpy()
        print("形状:", d_kernel.shape)
        print("权重:")
        print(d_kernel)
        print("注意：膨胀卷积，实际感受野更大")

# 创建模型并运行
print("\n开始三方向特征提取分析...")
model = DirectionalConvModel(dim=1)

# 显示卷积核信息
model.get_kernel_info()

# 推理过程
with torch.no_grad():
    print("\n" + "="*60)
    print("开始特征提取...")
    print("="*60)
    
    # 获取三个方向的特征图
    horizontal_features, vertical_features, diagonal_features = model(image)
    
    print(f"\n水平特征图形状: {horizontal_features.shape}")
    print(f"垂直特征图形状: {vertical_features.shape}")
    print(f"对角特征图形状: {diagonal_features.shape}")
    
    # 保存原始图像
    original = image[0, 0].cpu().numpy()
    save_image(original, '0_original_image.png')
    
    # 保存水平特征图
    h_feature = horizontal_features[0, 0].cpu().numpy()
    save_image(h_feature, '1_horizontal_features.png')
    print("🔄 水平特征图已保存 - 显示横向设备（吊车臂、泵管等）")
    
    # 保存垂直特征图
    v_feature = vertical_features[0, 0].cpu().numpy()
    save_image(v_feature, '2_vertical_features.png')
    print("🔄 垂直特征图已保存 - 显示纵向设备（塔吊等）")
    
    # 保存对角特征图
    d_feature = diagonal_features[0, 0].cpu().numpy()
    save_image(d_feature, '3_diagonal_features.png')
    print("🔄 对角特征图已保存 - 显示斜向设备（挖掘机臂等）")
    
    # 创建组合特征图
    combined = horizontal_features + vertical_features + diagonal_features
    combined_feature = combined[0, 0].cpu().numpy()
    save_image(combined_feature, '4_combined_features.png')
    print("🔄 组合特征图已保存 - 所有方向特征的组合")
    
    # 统计分析
    print("\n" + "="*60)
    print("特征图统计分析:")
    print("="*60)
    
    def analyze_features(features, name):
        feat = features[0, 0].cpu().numpy()
        print(f"\n{name}:")
        print(f"  均值: {feat.mean():.4f}")
        print(f"  标准差: {feat.std():.4f}")
        print(f"  最大值: {feat.max():.4f}")
        print(f"  最小值: {feat.min():.4f}")
        print(f"  激活像素数 (>0.1): {(feat > 0.1).sum()}")
    
    analyze_features(horizontal_features, "🔄 水平特征 (检测横向设备)")
    analyze_features(vertical_features, "🔄 垂直特征 (检测纵向设备)")  
    analyze_features(diagonal_features, "🔄 对角特征 (检测斜向设备)")

print("\n" + "="*60)
print("🎉 三方向特征提取完成！")
print("="*60)
print("生成的文件说明:")
print("📸 0_original_image.png - 原始输入图像")
print("➡️  1_horizontal_features.png - 水平方向特征（吊车臂、泵管）")
print("⬆️  2_vertical_features.png - 垂直方向特征（塔吊）")
print("↗️  3_diagonal_features.png - 对角方向特征（挖掘机臂）")
print("🔀 4_combined_features.png - 组合特征图")
print("\n💡 应用场景:")
print("- 水平特征：检测吊车臂、混凝土泵管等横向延展设备")
print("- 垂直特征：检测塔吊、起重机等纵向延展设备")
print("- 对角特征：检测挖掘机臂、斜坡道等倾斜结构")
print("\n🚀 使用方法：直接运行此代码即可！")

# ============================================================================
# 主程序入口 - 如果直接运行此脚本会执行
# ============================================================================
if __name__ == "__main__":
    print("🏗️ 建筑工地设备方向特征提取程序启动...")
    print("程序将自动展示水平、垂直、对角三种卷积的特征提取效果...")
    # 所有代码都已经在上面执行了，这里只是一个标识
    pass