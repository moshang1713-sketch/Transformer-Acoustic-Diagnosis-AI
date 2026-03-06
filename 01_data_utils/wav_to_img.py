import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载声音文件
# y 是音频数据(一串数字), sr 是采样率
filename = 'transformer_hum.wav'
y, sr = librosa.load(filename, duration=3) # 只取前3秒

print(f"正在处理: {filename} ...")

# 2. 核心变换：计算 Mel 声谱 (Mel Spectrogram)
# 这一步是把“时间-振幅”变成“时间-频率-能量”
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# 3. 转成对数刻度 (dB)
# 也就是把能量变成人耳感知的“分贝”，这样图的颜色对比度才正常
S_dB = librosa.power_to_db(S, ref=np.max)

# 4. 画图并保存 (关键！)
# 我们需要一张“纯净”的图，不要坐标轴，不要白边
plt.figure(figsize=(2.24, 2.24), dpi=100) # 224x224 像素 (MobileNet的标准尺寸)
plt.axis('off') # 关掉坐标轴
plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # 铺满画布

# 画出声谱图
librosa.display.specshow(S_dB, sr=sr, cmap='magma') # magma颜色更适合看能量分布

# 保存
output_name = 'spectrogram.jpg'
plt.savefig(output_name, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"🎉 转换成功！图片已保存为: {output_name}")
