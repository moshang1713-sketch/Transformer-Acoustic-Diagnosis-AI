import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- 硬件与路径配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = "../dataset/train"
MODEL_SAVE_PATH = "../03_models/transformer_doctor_v6_99.pth"

# --- 低内存优化超参数 ---
BATCH_SIZE = 4            # 极低批大小以适配 Chrome 共存环境
ACCUMULATION_STEPS = 4    # 梯度累加步数，等效 Batch Size = 4 * 4 = 16
NUM_WORKERS = 0           # 禁用多进程数据加载，防止内存碎片导致崩溃
LEARNING_RATE = 1e-4      # 学习率 (Learning Rate)
EPOCHS = 50               # 最大轮次

# --- 数据预处理 (Data Augmentation & Normalization) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train_model():
    # 1. 加载扩增后的 8000 张数据集
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS
    )

    # 2. 构建 MobileNetV2 模型
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # 修改分类头为 4 分类 (故障识别)
    model.classifier[1] = nn.Linear(model.last_channel, 4)
    model = model.to(DEVICE)

    # 3. 损失函数与优化器 (针对 99% 准度优化)
    # 使用标签平滑 (Label Smoothing) 防止对扩增数据过拟合
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    # 余弦退火学习率调度器 (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"🚀 启动 V6 巅峰训练 | 设备: {DEVICE} | 目标: 99% Acc")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        optimizer.zero_grad() # 初始化梯度
        
        # 使用 tqdm 可视化训练进度
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # 前向传播 (Forward Pass)
            outputs = model(inputs)
            # 计算 Loss 并除以累加步数进行归一化
            loss = criterion(outputs, labels) / ACCUMULATION_STEPS
            
            # 反向传播 (Backward Pass)
            loss.backward()
            
            # 梯度累加逻辑
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # 统计准度 (Accuracy)
            running_loss += loss.item() * ACCUMULATION_STEPS
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({"Loss": f"{running_loss/(i+1):.4f}", "Acc": f"{100.*correct/total:.2f}%"})

        epoch_acc = correct / total
        scheduler.step()

        # 4. 早停机制 (Early Stopping)：达到保研级准度目标即停止
        if epoch_acc >= 0.995:
            print(f"\n🎯 达成 99.5% 准度目标！当前 Acc: {epoch_acc:.4f}")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ 模型已保存至: {MODEL_SAVE_PATH}")
            break
        
        # 每 5 轮强制保存一次，防止断电
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"../03_models/v6_checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_model()