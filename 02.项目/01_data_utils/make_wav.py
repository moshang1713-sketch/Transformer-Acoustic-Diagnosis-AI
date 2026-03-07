import numpy as np
from scipy.io.wavfile import write

# 1. 参数设置
sample_rate = 44100  # 采样率 (每秒44100个点)
duration = 5         # 时长 5秒
frequency = 50       # 50Hz (模拟工频变压器的电流声)

# 2. 生成正弦波 (模拟嗡嗡声)
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
signal = 0.5 * np.sin(2 * np.pi * frequency * t)

# 3. 加一点噪声 (模拟环境杂音)
noise = 0.2 * np.random.normal(0, 1, len(t))
audio_data = signal + noise

# 4. 保存为 .wav 文件
# 必须转成 16-bit 整数格式才能保存为标准wav
scaled = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
write('transformer_hum.wav', sample_rate, scaled)

print("✅ 声音文件已生成：transformer_hum.wav")
