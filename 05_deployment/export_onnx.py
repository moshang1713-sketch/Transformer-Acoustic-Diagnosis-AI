import torch
import torch.nn as nn
from torchvision import models
import onnx

# --- 1. 配置 ---
MODEL_PATH = '../03_models/transformer_doctor.pth'
ONNX_PATH = '../03_models/transformer_doctor.onnx'
CLASSES = ['normal', 'loose', 'discharge']
device = torch.device("cpu") # 导出时通常用 CPU 就够了，兼容性更好

# --- 2. 就像之前一样复活模型 ---
print("🧠 正在加载 PyTorch 模型...")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
# 加载权重 (映射到 CPU)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # 必须切换到 eval 模式！否则 Batchnorm 层会出错

# --- 3. 创建一个“假人” (Dummy Input) ---
# ONNX 导出需要追踪数据流，所以我们需要喂给它一个形状正确的随机张量
# 格式: (Batch_Size, Channels, Height, Width) -> (1, 3, 224, 224)
dummy_input = torch.randn(1, 3, 224, 224, device=device)

# --- 4. 执行导出 (最关键的一步) ---
print("🔄 正在转换为 ONNX 格式...")
torch.onnx.export(
    model,                  # 也就是 PyTorch 模型
    dummy_input,            # 假数据
    ONNX_PATH,              # 输出文件名
    verbose=False,          # 是否打印详细日志
    input_names=['input'],  # 给输入层起个名字 (以后在 C++ 里调用要用)
    output_names=['output'],# 给输出层起个名字
    # 动态轴 (可选)：允许 Batch Size 变动，比如一次传 1 张或 10 张图
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# --- 5. 验证导出是否成功 ---
print("✅ 转换完成！正在进行完整性校验...")
onnx_model = onnx.load(ONNX_PATH)
try:
    onnx.checker.check_model(onnx_model)
    print(f"🎉 校验通过！ONNX 模型已保存为: {ONNX_PATH}")
    print("现在，这个模型已经脱离 Python，可以被 C++ 直接调用了！")
except onnx.checker.ValidationError as e:
    print(f"❌ 模型校验失败: {e}")
