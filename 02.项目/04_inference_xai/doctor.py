import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import random

# --- 1. 配置 ---
# 必须和训练时完全一致
CLASSES = ['normal', 'loose', 'discharge']
MODEL_PATH = '../03_models/transformer_doctor.pth'
# 注意：我们要去 D 盘的 val 文件夹里抓图来测试
DATA_DIR = '/mnt/d/AI_Data/dataset/val' 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 复活模型 (加载大脑) ---
print("🧠 正在唤醒 AI 医生...")
model = models.mobilenet_v2(weights=None) # 不需要下载预训练权重，因为我们要加载自己的
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES)) # 结构要对齐
model.load_state_dict(torch.load(MODEL_PATH)) # 加载存档
model = model.to(device)
model.eval() # 切换到“考试模式” (关闭 Dropout 等)
print("✅ AI 医生已就位！")

# --- 3. 预处理 (必须和训练时一样) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. 随机抓一个“病人”来诊断 ---
# 随机选一个类别
true_label = random.choice(CLASSES)
# 随机选一张图
folder_path = os.path.join(DATA_DIR, true_label)
image_name = random.choice(os.listdir(folder_path))
image_path = os.path.join(folder_path, image_name)

print(f"\n📂 抽取病例: {image_path}")
print(f"👀 真实病因: {true_label}")

# --- 5. AI 诊断 ---
image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0).to(device) # 加一个维度 (Batch Size)

with torch.no_grad():
    outputs = model(image_tensor)
    # 计算概率 (Softmax)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    # 找到概率最大的那个
    confidence, predicted_idx = torch.max(probabilities, 1)
    predicted_label = CLASSES[predicted_idx.item()]

# --- 6. 输出报告 ---
print("-" * 30)
print(f"🤖 AI 诊断结果: {predicted_label.upper()}")
print(f"📊 确信度: {confidence.item()*100:.2f}%")

if predicted_label == true_label:
    print("✅ 诊断正确！")
else:
    print("❌ 误诊了！")
print("-" * 30)
