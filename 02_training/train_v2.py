import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# 1. 路径配置 (确保包含 4 个文件夹)
DATA_DIR = "../dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train") # 包含 friction, imbalance, loose, normal
VAL_DIR = os.path.join(DATA_DIR, "val")

# 2. 增强预处理 (针对电力声谱图优化)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # 增加模型鲁棒性
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 3. 加载数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}
# 显式打印类别映射，确保索引 0-3 对应正确
print(f"类别映射表: {image_datasets['train'].class_to_idx}") 

train_loader = DataLoader(image_datasets['train'], batch_size=16, shuffle=True) # 16G内存建议BS=16
val_loader = DataLoader(image_datasets['val'], batch_size=16, shuffle=False)

# 4. 构建 4 分类模型
def train_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # 修改分类头为 4
    model.classifier[1] = nn.Linear(model.last_channel, 4) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 简单训练循环 (建议训练 15-20 轮以达到 SOTA)
    for epoch in range(15):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/15, Loss: {running_loss/len(train_loader):.4f}")

    # 保存权重
    torch.save(model.state_dict(), "../03_models/transformer_doctor_v2.pth")
    print("高精度版 4 分类模型已保存至 03_models/transformer_doctor_v2.pth")

if __name__ == "__main__":
    train_model()