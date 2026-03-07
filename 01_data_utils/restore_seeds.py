import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# --- 1. 配置中心 (对齐你的项目路径) ---
# 确保指向项目根目录下的 dataset 文件夹
DATA_DIR = '../dataset' 
CLASSES = ['friction', 'imbalance', 'loose', 'normal'] # 对齐 V6/V7 类别
SAMPLE_RATE = 44100
DURATION = 3 
NUM_SEEDS = 20 # 每类生成 20 张原始种子图

# --- 2. 物理仿真引擎 (补充了摩擦和不平衡的数学模型) ---
def generate_audio(fault_type):
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    # 基础工频：$50\text{Hz}$ 嗡嗡声
    base_hum = 0.5 * np.sin(2 * np.pi * 50 * t)
    
    if fault_type == 'normal':
        noise = 0.05 * np.random.normal(0, 1, len(t))
        audio = base_hum + noise
        
    elif fault_type == 'loose':
        # 松动：产生 $100\text{Hz}$ 和 $150\text{Hz}$ 的高次谐波
        harmonic = 0.3 * np.sin(2 * np.pi * 100 * t) + 0.1 * np.sin(2 * np.pi * 150 * t)
        audio = base_hum + harmonic + 0.1 * np.random.normal(0, 1, len(t))
        
    elif fault_type == 'friction':
        # 摩擦：模拟原代码中的 discharge (放电/高频脉冲) 逻辑
        noise = 0.1 * np.random.normal(0, 1, len(t))
        num_spikes = 100 # 增加脉冲密度
        spike_indices = np.random.randint(0, len(t), num_spikes)
        spikes = np.zeros_like(t)
        spikes[spike_indices] = 5.0 * np.random.choice([-1, 1], num_spikes)
        audio = base_hum + noise + spikes
        
    elif fault_type == 'imbalance':
        # 不平衡：模拟幅值调制 (Amplitude Modulation, $10\text{Hz}$ 波动)
        mod = 0.3 * np.sin(2 * np.pi * 10 * t)
        audio = (1 + mod) * base_hum + 0.08 * np.random.normal(0, 1, len(t))
        
    return audio

def save_spectrogram(audio, save_path):
    S = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.axis('off')
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(S_dB, sr=SAMPLE_RATE, fmax=8000, cmap='magma')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# --- 3. 执行复活 ---
print(f"🚀 正在重新生成原始种子数据到: {DATA_DIR}...")
for cls in CLASSES:
    # 统一存放在 train 目录下，后续用 V7 脚本自动分配验证集
    dir_path = os.path.join(DATA_DIR, 'train', cls)
    os.makedirs(dir_path, exist_ok=True)
    
    for i in range(NUM_SEEDS):
        audio_data = generate_audio(cls)
        save_name = os.path.join(dir_path, f"{cls}_{i}.jpg")
        save_spectrogram(audio_data, save_name)
        print(f"已恢复: {cls}_{i}.jpg", end='\r')
    print(f"\n✅ {cls} 类别 20 张原图恢复完成！")

print("\n✨ 种子数据已全部找回。现在你可以重新运行 python ../data_remix_v7.py 了！")