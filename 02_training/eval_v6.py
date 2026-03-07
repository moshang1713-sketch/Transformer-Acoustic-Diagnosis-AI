import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 环境与模型加载
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['friction', 'imbalance', 'loose', 'normal']

model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(model.last_channel, 4)
# 加载你刚刚跑出来的 99% 巅峰权重
model.load_state_dict(torch.load("../03_models/transformer_doctor_v6_99.pth"))
model = model.to(DEVICE).eval()

# 2. 加载测试集 (或扩增后的验证集)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder("../dataset/train", transform=transform) # 先用训练集看拟合情况
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 3. 获取预测结果
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.to(DEVICE))
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 4. 绘图：硬核科研成果可视化
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted (模型预测)')
plt.ylabel('True (实际类别)')
plt.title(f'Transformer Fault Diagnosis - V6 99% Acc\n(变压器故障诊断混淆矩阵)')
plt.savefig("../outputs/confusion_matrix_v6.png") # 自动保存到 outputs 文件夹
print("✅ 混淆矩阵已生成！请查看 outputs 文件夹。")
print(classification_report(all_labels, all_preds, target_names=CLASSES))