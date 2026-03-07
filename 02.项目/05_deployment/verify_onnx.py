import onnxruntime as ort
import numpy as np
import os
from PIL import Image

# --- 1. 配置 ---
ONNX_MODEL_PATH = 'transformer_doctor.onnx'
CLASSES = ['discharge', 'loose', 'normal']

# 模拟 C 语言环境中的预处理参数
# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
std = np.array([0.229, 0.224, 0.225]).astype(np.float32)

# --- 2. 准备一张测试图 ---
# 我们随便找一张验证集的图
val_dir = '/mnt/d/AI_Data/dataset/val'
# 遍历找一张存在的图
test_image_path = None
for root, dirs, files in os.walk(val_dir):
    if files:
        test_image_path = os.path.join(root, files[0])
        break

if not test_image_path:
    print("❌ 没找到图片，请检查路径")
    exit()

print(f"🖼️ 测试图片: {test_image_path}")

# --- 3. 手写预处理 (模拟 C 语言的逻辑) ---
# 在 C 语言里没有 transforms 库，我们需要自己写像素运算
img = Image.open(test_image_path).convert('RGB')
img = img.resize((224, 224)) # 1. 缩放
img_data = np.array(img).astype(np.float32) / 255.0 # 2. 归一化到 0-1

# 3. 标准化 (减均值，除方差)
# 这一步在 C 语言里就是一个 for 循环
img_data = (img_data - mean) / std

# 4. 维度变换 (HWC -> CHW)
# C 语言读取图片通常是 HWC (高,宽,通道)，但 ONNX 模型需要 CHW (通道,高,宽)
# 这是一个非常经典的“坑”，在 C 里需要手动搬运内存
img_data = img_data.transpose(2, 0, 1)

# 5. 增加 Batch 维度 (CHW -> BCHW)
img_data = np.expand_dims(img_data, axis=0)

# --- 4. 启动 ONNX 引擎 ---
print("🚀 启动 ONNX Runtime 引擎...")
session = ort.InferenceSession(ONNX_MODEL_PATH)

# 获取输入输出节点的名称 (对应你在 Netron 里看到的名字)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- 5. 推理 ---
result = session.run([output_name], {input_name: img_data})
scores = result[0][0] # 拿到第一个样本的输出

# --- 6. 结果解析 ---
print("\n📊 推理结果 (Raw Scores):", scores)

# 找最大值的索引 (argmax)
pred_idx = np.argmax(scores)
pred_label = CLASSES[pred_idx]

print(f"✅ 最终诊断: {pred_label.upper()}")
print("-" * 30)
print("💡 重点理解：")
print("在 C 语言中，刚才的 img_data 其实就是一段长长的 float 数组。")
print(f"数组长度 = 1 * 3 * 224 * 224 = {1*3*224*224} 个浮点数。")
print("只要你能在 C 里构造出这串数组，喂给模型，结果就是一样的！")
