import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import os
import plotly.express as px
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ================= 1. 页面配置与高级美化 =================
st.set_page_config(page_title="变压器声纹智能诊断系统 V6", layout="wide", page_icon="⚡")

# 注入 CSS 适配深/浅色模式并增强视觉对比
st.markdown("""
    <style>
    @media (prefers-color-scheme: light) {
        :root { --main-bg: #f8fafc; --card-bg: #ffffff; --text: #1f2937; }
    }
    @media (prefers-color-scheme: dark) {
        :root { --main-bg: #1e1e1e; --card-bg: #2d2d2d; --text: #FAFAFA; }
    }
    .stApp { background-color: var(--main-bg); color: var(--text); }
    div[data-testid="stMetric"] {
        background-color: var(--card-bg);
        border: 1px solid #404040;
        border-radius: 15px;
        padding: 20px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    h1, h2, h3 { font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. 核心模型加载 (更新至 V6) =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 关键修改：指向你刚刚训练成功的 99% 权重文件
# 找到 MODEL_PATH 这一行，修改为 v7_final.pth
MODEL_PATH = os.path.join(BASE_DIR, "../03_models/transformer_doctor_v7_final.pth")

@st.cache_resource
def load_assets():
    model = models.mobilenet_v2()
    model.classifier[1] = torch.nn.Linear(model.last_channel, 4) 
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ 未找到 V6 巅峰模型：{MODEL_PATH}，请确认路径。")
        return None, None
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return model, preprocess

model, preprocess = load_assets()

# 对应 data_factory.py 的类别顺序
CLASSES = [
    '摩擦故障 (Friction)',      # Index 0
    '电流不平衡 (Imbalance)',    # Index 1
    '螺丝松动 (Loose)',          # Index 2
    '正常运行 (Normal)'          # Index 3
]

# ================= 3. 侧边栏与主界面 =================
with st.sidebar:
    st.title("⚡ 华北电力大学")
    st.caption("人工智能专业 - 陈则欣")
    st.divider()
    st.success("🎯 模型状态：V6 巅峰版 (99.62% Acc)")
    st.info("硬件环境：16GB RAM / Win11")

st.title("⚡ 变压器故障声纹智能诊断平台 (科研级)")
st.caption("基于 8000 张增强样本训练的深度学习方案")

tab1, tab2 = st.tabs(["🔍 AI 实时听诊", "📂 实验数据详情"])

with tab1:
    uploaded_file = st.file_uploader("上传声谱图 (Spectrogram)", type=["jpg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        col_img, col_cam = st.columns(2)
        
        with col_img:
            st.subheader("🖼️ 原始声谱图")
            st.image(img, use_container_width=True)

        if st.button("🚀 启动 99% 准度深度诊断", use_container_width=True):
            input_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)[0].numpy()
                pred_idx = np.argmax(probs)
            
            # Grad-CAM 可解释性分析 (科研加分项)
            target_layers = [model.features[-1]]
            cam = GradCAM(model=model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx)])[0, :]
            visualization = show_cam_on_image(np.array(img.resize((224, 224)))/255.0, grayscale_cam, use_rgb=True)
            
            with col_cam:
                st.subheader("🕵️ 故障特征定位 (Grad-CAM)")
                st.image(visualization, use_container_width=True)

            # --- 诊断报告核心指标 (修正逻辑) ---
            st.divider()
            res1, res2, res3 = st.columns(3)
            
            # 状态锁定
            res1.metric("预测工况", CLASSES[pred_idx].split(' ')[0])
            # 巅峰置信度展示
            res2.metric("诊断置信度", f"{probs[pred_idx]*100:.2f}%")
            
            # 建议措施纠错：只有索引 3 是正常
            if pred_idx == 3:
                res3.metric("建议措施", "运行稳定", delta="✅ 状态良好", delta_color="normal")
            else:
                res3.metric("建议措施", "立即停机检修", delta="⚠️ 存在隐患", delta_color="inverse")

            # 可视化概率分布
            fig = px.bar(x=CLASSES, y=probs, color=CLASSES, title="全类别识别概率分布")
            fig.update_layout(xaxis_title=None, yaxis_title="概率", showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("### 📈 V6 训练实验报告")
    st.table({
        "指标项目": ["原始样本数", "扩增样本数", "训练耗时", "验证集准度", "模型架构"],
        "数值": ["80 张", "8000 张", "6 Epochs", "99.62%", "MobileNetV2"]
    })