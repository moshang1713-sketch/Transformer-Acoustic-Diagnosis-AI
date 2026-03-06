import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# --- 配置 ---
data_dir = '/mnt/d/AI_Data/dataset'  # 就在当前目录下生成
classes = ['normal', 'loose', 'imbalance', 'friction']
num_train = 20  # 训练集每类 20 张
num_val = 5     # 验证集每类 5 张

# --- 核心生成函数 ---
def create_dummy_spectrogram(save_path):
    # 生成随机噪声图 (模拟声谱图)
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.axis('off')
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    
    # 随机生成一些斑点和线条，假装是故障特征
    data = np.random.rand(128, 128)
    plt.imshow(data, cmap='magma', aspect='auto')
    plt.savefig(save_path, bbox_inches=None, pad_inches=0)
    plt.close()

# --- 开始干活 ---
print(f"🏭 正在 '{data_dir}' 下重新构建数据集...")

for phase in ['train', 'val']:
    num_items = num_train if phase == 'train' else num_val
    for cls in classes:
        # 创建文件夹: dataset/train/normal ...
        dir_path = os.path.join(data_dir, phase, cls)
        os.makedirs(dir_path, exist_ok=True)
        
        # 生成图片
        for i in range(num_items):
            file_path = os.path.join(dir_path, f"{cls}_{i}.jpg")
            create_dummy_spectrogram(file_path)
            
print("✅ 数据生成完毕！")

