import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. 页面配置
st.set_page_config(page_title="变压器 AI 诊断系统", layout="wide")
st.title("⚡ 变压器故障声纹识别 + 热力图扫描系统")
st.sidebar.info("开发者：陈则欣 | 华北电力大学") #

# 2. 加载模型逻辑 (自动适配 3 类输出)
@st.cache_resource
def load_pytorch_model():
    # 初始化 MobileNetV2 结构
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 3) # 强制匹配 3 类
    # 加载你的权重文件
    model.load_state_dict(torch.load("../03_models/transformer_doctor.pth", map_location='cpu'))
    model.eval()
    return model

model = load_pytorch_model()
CLASSES = ['friction', 'imbalance', 'loose'] # 严格对应 3 个类别

# 3. 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 4. Web 交互逻辑
uploaded_file = st.sidebar.file_uploader("上传声谱图 (.jpg)", type=["jpg", "png"])

if uploaded_file is not None:
    # 读取图片
    input_image = Image.open(uploaded_file).convert('RGB')
    input_tensor = preprocess(input_image).unsqueeze(0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(input_image, caption="原始声谱图", use_container_width=True)
    
    if st.button("🚀 启动 AI 深度扫描"):
        # A. 执行诊断推理
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
        
        # B. 生成 Grad-CAM 热力图
        target_layers = [model.features[-1]] # 提取最后一层卷积特征
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(pred_idx)]
        
        # 叠加原始图与热力图
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        rgb_img = np.array(input_image.resize((224, 224))) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        with col2:
            st.image(visualization, caption=f"决策热力图 (识别结果: {CLASSES[pred_idx]})", use_container_width=True)
        
        # C. 概率分布可视化
        st.divider()
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("诊断结论", CLASSES[pred_idx].upper())
        res_col2.metric("置信度", f"{probs[pred_idx]*100:.2f}%")
        
        # 修复越界问题的柱状图
        chart_data = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        st.bar_chart(chart_data)
else:
    st.warning("👈 请在左侧上传待检测图片。")