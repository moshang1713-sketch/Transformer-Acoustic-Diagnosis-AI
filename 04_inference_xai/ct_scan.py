import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# --- 引入 Grad-CAM 神器 ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. 配置 ---
CLASSES = ['normal', 'loose', 'discharge']
MODEL_PATH = '../03_models/transformer_doctor.pth'
# 我们特意挑一张“放电”的图，因为它的特征最明显
target_class_name = 'discharge' 
DATA_DIR_VAL = f'/mnt/d/AI_Data/dataset/val/{target_class_name}'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 准备“病人”图片 ---
# 随机挑一张图
try:
    image_name = np.random.choice(os.listdir(DATA_DIR_VAL))
except FileNotFoundError:
    print(f"❌ 错误：找不到文件夹 {DATA_DIR_VAL}")
    print("请确认你的 D 盘数据是否还在？")
    exit()

image_path = os.path.join(DATA_DIR_VAL, image_name)
print(f"🔍 正在扫描图片: {image_path}")

# 读取原始图片 (用于最后画图展示)
# 注意：OpenCV 读进来是 BGR，要转 RGB
img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# 归一化到 0-1 之间，这是 Grad-CAM 库要求的格式
rgb_img_float = np.float32(img_rgb) / 255.0

# 准备喂给模型的张量 (需要经过严格的标准化)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
input_tensor = normalize(transforms.ToTensor()(Image.fromarray(img_rgb))).unsqueeze(0).to(device)


# --- 3. 唤醒“AI医生” (加载模型) ---
print("🧠 加载模型大脑...")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
# 加载之前保存的权重
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print(f"❌ 错误：找不到模型文件 {MODEL_PATH}")
    print("请确认你之前运行过 train_plot.py 并且生成了 .pth 文件")
    exit()
    
model = model.to(device)
model.eval()

# --- ⭐ 4. 核心：指定要监听的目标层 ⭐ ---
# MobileNetV2 的特征提取器部分叫 'features'。
# 我们通常看最后一层，因为它包含最高级的语义信息。
target_layers = [model.features[-1]]

# --- 5. 启动 CT 扫描 (Grad-CAM) ---
print("📸 正在生成热力图...")
# 初始化 GradCAM
cam = GradCAM(model=model, target_layers=target_layers)

# 我们想看模型为什么觉得它是 'discharge' (放电)
target_category_index = CLASSES.index(target_class_name)
targets = [ClassifierOutputTarget(target_category_index)]

# 生成灰度热力图
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

# --- 6. 合成并保存结果 ---
# 将灰度热力图叠加到原始 RGB 图像上
visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

# 用 matplotlib 画出来对比看
plt.figure(figsize=(10, 5))

# 左边：原图
plt.subplot(1, 2, 1)
plt.imshow(rgb_img_float)
plt.title(f'Original: {image_name}')
plt.axis('off')

# 右边：CT扫描结果
plt.subplot(1, 2, 2)
plt.imshow(visualization)
plt.title('AI Focus (Grad-CAM)')
plt.axis('off')

SAVE_PATH = '../outputs/ct_scan_result.png'
plt.savefig(SAVE_PATH, bbox_inches='tight')
print(f"🎉 扫描完成！结果已保存为: {SAVE_PATH}")
print("请去 Windows 文件夹查看这张极具冲击力的图片！")
