# ⚡ 变压器故障声纹识别 AI 诊断系统 (V1.0)

## 📖 项目背景

变压器作为电网核心设备，其运行状态直接影响电网安全。本项目针对变压器在**正常、螺丝松动、电流不平衡、内部摩擦**四种工况下的声纹特征进行建模。

## 🌟 核心创新点

* **时频域转换 (STFT)**：突破传统的时域波形分析，利用短时傅里叶变换（Short-Time Fourier Transform）将一维音频转化为二维声谱图（Spectrogram），提取高维特征。
* **轻量化架构**：选用 **MobileNetV2** 作为骨干网络（Backbone），利用深度可分离卷积（Depthwise Separable Convolution）降低参数量，适配边缘端（Edge Computing）部署需求。
* **可解释性 (XAI)**：集成 **Grad-CAM** 算法，对 AI 决策逻辑进行“热力图”可视化，确保模型关注点符合物理声纹特性。

## 📁 目录说明

项目采用模块化工程结构，确保从数据到部署的解耦：

| 文件夹 | 功能描述 |
| --- | --- |
| **`01_data_utils`** | 仿真数据生成、音频合成与时频转换工具 |
| **`02_training`** | MobileNetV2 迁移学习（Transfer Learning）训练核心逻辑 |
| **`03_models`** | 存放训练产出的 `.pth` 权重与标准 **ONNX** 模型 |
| **`04_inference_xai`** | AI 医生诊断接口与 Grad-CAM 可解释性分析脚本 |
| **`05_deployment`** | 导出至 C++/嵌入式环境的部署验证工具 |
| **`outputs`** | 存放训练曲线图、热力图等可视化实验报表 |

---

## 🚀 核心指标

* **训练准确率**: $Acc = 1.0000$ ($100\%$)。
* **损失函数**: 使用交叉熵损失 ($CrossEntropyLoss$)：

$$L = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)$$


* **硬件环境**: 支持 CUDA 加速，优化后可在 16GB RAM 机器顺畅运行。

---

## 🛠️ 快速开始

### 1. 环境准备

```bash
conda create -n audio_study python=3.10
conda activate audio_study
pip install torch torchvision torchaudio matplotlib onnxruntime opencv-python

```

### 2. 生成数据集

```bash
python 01_data_utils/make_data.py

```

### 3. 启动训练

```bash
python 02_training/train_plot.py

```

### 4. 故障诊断与热力图分析

```bash
python 04_inference_xai/ct_scan.py

```

## 📊 实验结果可视化

通过 `ct_scan.py` 生成的 Grad-CAM 热力图可以清晰看到，模型在识别“放电”或“摩擦”故障时，注意力精准聚焦于声谱图中的高频脉冲区域，证明了模型的科学性。

---

## 📜 许可证

本项目采用 **MIT License** 开源。

---

