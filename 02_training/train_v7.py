import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 1. 低内存与抗过拟合配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8            # 适配 16GB RAM
ACCUMULATION_STEPS = 4    # 等效 Batch Size = 32
EPOCHS = 30

# --- 2. 建立 V7 巅峰模型 ---
def get_v7_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # 强制插入 Dropout 防止死记硬背
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5), 
        nn.Linear(model.last_channel, 4)
    )
    return model.to(DEVICE)

model = get_v7_model()

# --- 3. 优化器：AdamW + L2 正则化 ---
# weight_decay=1e-2 也是抗过拟合的利器
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# --- 4. 训练循环 ---
def train():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_set = datasets.ImageFolder("../dataset/train", transform)
    val_set = datasets.ImageFolder("../dataset/val", transform)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"🚀 V7 巅峰对决开始！训练集: {len(train_set)} | 验证集: {len(val_set)}")

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            outputs = model(inputs.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE)) / ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # 验证逻辑：看看这次还会不会“翻车”
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.to(DEVICE))
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels.to(DEVICE)).sum().item()
        
        val_acc = correct / total
        scheduler.step()
        print(f"📈 验证集准度: {val_acc:.4f}")

        if val_acc >= 0.992:
            print("🎯 达成真实 99% 准度！保存 V7 最强心脏...")
            torch.save(model.state_dict(), "../03_models/transformer_doctor_v7_final.pth")
            break

if __name__ == "__main__":
    train()