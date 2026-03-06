import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io.wavfile import write

# --- 1. 配置中心 ---
# 你的 D 盘基地
DATA_DIR = '/mnt/d/AI_Data/dataset' 
CLASSES = ['normal', 'loose', 'discharge']
SAMPLE_RATE = 44100
DURATION = 3  # 每段声音 3 秒
NUM_TRAIN = 50 # 训练集每类生成 50 张
NUM_VAL = 10   # 验证集每类生成 10 张

# --- 2. 物理仿真引擎：造声音 ---
def generate_audio(fault_type):
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    
    # 基础声音：50Hz 的工频嗡嗡声
    base_hum = 0.5 * np.sin(2 * np.pi * 50 * t)
    
    if fault_type == 'normal':
        # 正常：只有一点点背景白底噪
        noise = 0.05 * np.random.normal(0, 1, len(t))
        audio = base_hum + noise
        
    elif fault_type == 'loose':
        # 松动：出现谐波 (100Hz, 150Hz) 且声音变大
        harmonic1 = 0.3 * np.sin(2 * np.pi * 100 * t) # 2倍频
        harmonic2 = 0.1 * np.sin(2 * np.pi * 150 * t) # 3倍频
        noise = 0.1 * np.random.normal(0, 1, len(t))  # 机械震动带来更多噪音
        audio = base_hum + harmonic1 + harmonic2 + noise
        
    elif fault_type == 'discharge':
        # 放电：加入随机的高频脉冲 (滋滋声)
        noise = 0.1 * np.random.normal(0, 1, len(t))
        # 随机生成一些脉冲
        num_spikes = 50
        spike_indices = np.random.randint(0, len(t), num_spikes)
        spikes = np.zeros_like(t)
        spikes[spike_indices] = 5.0 * np.random.choice([-1, 1], num_spikes) # 瞬间高能
        audio = base_hum + noise + spikes
        
    else:
        audio = base_hum
        
    return audio

# --- 3. 视觉转换引擎：声音转图片 ---
def save_spectrogram(audio, save_path):
    # 计算 Mel 声谱
    S = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # 画图 (无边框)
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.axis('off')
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(S_dB, sr=SAMPLE_RATE, fmax=8000, cmap='magma')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# --- 4. 主程序 ---
print(f"🏭 正在 D 盘 ({DATA_DIR}) 构建变压器故障数据集...")

for phase in ['train', 'val']:
    num_items = NUM_TRAIN if phase == 'train' else NUM_VAL
    for cls in CLASSES:
        # 创建文件夹
        dir_path = os.path.join(DATA_DIR, phase, cls)
        os.makedirs(dir_path, exist_ok=True)
        
        print(f"   正在生成 {phase} - {cls} ({num_items}个)...")
        
        for i in range(num_items):
            # 1. 造声音
            audio_data = generate_audio(cls)
            # 2. 转图片并保存
            save_name = os.path.join(dir_path, f"{cls}_{i}.jpg")
            save_spectrogram(audio_data, save_name)

print("\n✅ 所有数据生成完毕！你可以去 D:\\AI_Data\\dataset 查看成果了。")
