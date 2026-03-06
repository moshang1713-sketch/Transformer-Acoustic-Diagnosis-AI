import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import time
import plotly.express as px  # 用于精美绘图并适配主题
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ================= 1. 页面配置与高级美化 (核心部分) =================
st.set_page_config(page_title="变压器声纹智能诊断系统", layout="wide", page_icon="⚡")

# 注入 CSS 变量与自定义样式，强制适配主题
st.markdown("""
    <style>
    /* =================== 定义 CSS 变量 (最硬核修复) =================== */
    /* 浅色模式下的变量 */
    @media (prefers-color-scheme: light) {
        :root {
            --st-app-background-color: #f8fafc;
            --st-text-color: #1f2937;
            --st-title-color: #1E3A8A; /* 华电蓝 */
            --st-metric-background-color: #ffffff;
            --st-metric-text-color: #1f2937;
            --st-metric-label-color: #6B7280;
            --st-metric-value-color: #1f2937;
            --st-metric-help-color: #A3A3A3;
            --st-metric-icon-color: #1f2937;
            --st-metric-border-color: #e2e8f0;
            --st-metric-shadow-color: rgba(0, 0, 0, 0.08);
            /* 强制 Plotly 文字颜色 */
            --st-plotly-text-color: #31333F !important;
        }
    }

    /* 深色模式下的变量 */
    @media (prefers-color-scheme: dark) {
        :root {
            --st-app-background-color: #1e1e1e; /* 不是纯黑，有层次感 */
            --st-text-color: #FAFAFA;
            --st-title-color: #FAFAFA;
            --st-metric-background-color: #2d2d2d; /* 比背景稍浅 */
            --st-metric-text-color: #FAFAFA;
            --st-metric-label-color: #FAFAFA;
            --st-metric-value-color: #FAFAFA;
            --st-metric-help-color: #FAFAFA;
            --st-metric-icon-color: #FAFAFA;
            --st-metric-border-color: #404040;
            --st-metric-shadow-color: rgba(255, 255, 255, 0.05);
            /* 强制 Plotly 文字颜色 */
            --st-plotly-text-color: #FAFAFA !important;
        }
    }

    /* =================== 应用 CSS 变量 =================== */
    .stApp {
        background-color: var(--st-app-background-color);
        color: var(--st-text-color);
    }

    /* 优化指标卡片 (Metrics) 样式化 */
    div[data-testid="stMetric"] {
        background-color: var(--st-metric-background-color);
        color: var(--st-metric-text-color);
        border: 1px solid var(--st-metric-border-color);
        border-radius: 15px;
        padding: 20px !important;
        box-shadow: 0 4px 12px var(--st-metric-shadow-color);
        margin-bottom: 20px;
    }

    /* 定位 Metrics 内部标签、数值、表情、 delta */
    div[data-testid="stMetricLabel"] {
        color: var(--st-metric-label-color);
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        color: var(--st-metric-value-color);
        font-size: 2.5rem;
    }
    div[data-testid="stMetricHelp"] {
        color: var(--st-metric-help-color);
    }
    /* 自定义图标/表情颜色 */
    div[data-testid="stMetricValue"] span {
        color: var(--st-metric-icon-color);
    }

    /* 标题颜色适配主题 */
    h1, h2, h3 { color: var(--st-title-color) !important; font-family: 'Inter', sans-serif; }

    /* =================== 柱状图适配 (通过 CSS 强制适配文字) =================== */
    /* 定位 Plotly 图表中的文字元素 (坐标轴文字、标题) */
    div.plot-container text {
        fill: var(--st-plotly-text-color) !important;
    }
    /* 定位 Plotly 的 Y 轴刻度标签 */
    div.plot-container div.yaxis text {
        fill: var(--st-plotly-text-color) !important;
    }
    /* 定位 Plotly 的 X 轴刻度标签 */
    div.plot-container div.xaxis text {
        fill: var(--st-plotly-text-color) !important;
    }
    /* 定位 Plotly 的标题 */
    div.plot-container div.plot-title text {
        fill: var(--st-plotly-text-color) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. 后端资产与逻辑加载 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../03_models/transformer_doctor.pth")

@st.cache_resource
def load_assets():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 3)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return model, preprocess

model, preprocess = load_assets()
# 缩短显示名称，强制横向
# 必须与 ls ~/ai_study/机器学习/02.项目/dataset/train 的结果完全一致
CLASSES = [
    '摩擦故障 (Friction)',     # 索引 0
    '电流不平衡 (Imbalance)',   # 索引 1
    '螺丝松动 (Loose)',        # 索引 2
    '正常运行 (Normal)'        # 索引 3
]

# ================= 3. 侧边栏设计 (集成 Logo) =================
# 如果有 image_10.png，使用 st.sidebar.image。如果没有，保留原文字
# ================= 3. 侧边栏设计 (优化 Logo 展示) =================
with st.sidebar:
    # 强制尝试加载本地 logo.png 文件
    LOGO_FILE = os.path.join(BASE_DIR, "logo.png")
    
    if os.path.exists(LOGO_FILE):
        # 使用本地文件，增加底部间距
        st.image(LOGO_FILE, use_container_width=True)
    else:
        # 如果文件不存在，显示文字备份
        st.title("⚡ 华北电力大学")
        st.caption("AI 诊断实验室 (缺失 logo.png)")
        
    st.divider()
    st.markdown(f"**项目作者**：陈则欣") #
    st.markdown(f"**专业方向**：人工智能 (电力特色)") 
    st.success("核心模型：MobileNetV2")
# ================= 4. 主界面布局 =================
st.title("⚡ 变压器故障声纹智能诊断平台")
st.caption("基于 MobileNetV2 + Grad-CAM 的可解释性深度学习方案")

tab1, tab2 = st.tabs(["🔍 AI 听诊室", "📖 技术架构"])

with tab1:
    uploaded_file = st.file_uploader("上传声谱图进行 AI 深度推理诊断", type=["jpg", "png"], help="支持从验证集目录选择图片")

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        
        # 左右对比布局
        col_img, col_cam = st.columns(2)
        with col_img:
            st.subheader("🖼️ 原始声谱图")
            st.image(img, use_container_width=True)

        if st.button("🚀 启动深度扫描诊断", use_container_width=True):
            with st.spinner('正在分析声纹特征并计算梯度图...'):
                # 推理
                input_tensor = preprocess(img).unsqueeze(0)
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)[0].detach().numpy()
                pred_idx = np.argmax(probs)
                
                # 热力图
                target_layers = [model.features[-1]]
                cam = GradCAM(model=model, target_layers=target_layers)
                grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx)])[0, :]
                visualization = show_cam_on_image(np.array(img.resize((224, 224)))/255.0, grayscale_cam, use_rgb=True)
            
            with col_cam:
                st.subheader("🕵️ AI 决策热力图")
                st.image(visualization, use_container_width=True)

            st.divider()
            
            # --- 结果展示区 (指标卡主题适配) ---
            st.subheader("📊 诊断报告核心指标")
            res1, res2, res3 = st.columns(3)
            res1.metric("预测工况", CLASSES[pred_idx].split(' ')[0], delta="已锁定", delta_color="normal")
            res2.metric("置信概率", f"{probs[pred_idx]*100:.2f}%")
            res3.metric("建议措施", "立即检修" if pred_idx != 2 else "状态良好")

            # --- 柱状图适配 (文字强制横向且主题适配) ---
            st.write("各工况识别概率分布：")
            # 核心修改：确保 x 和 y 长度一致
            fig = px.bar(
                x=CLASSES[:len(probs)],  # 关键点：只取与概率值数量相等的标签
                y=probs,
                color=CLASSES[:len(probs)],
                # ... 其他参数保持不变
            )
            fig.update_layout(
                xaxis_title=None,
                yaxis_title="概率 (Probability)",
                showlegend=False,
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                # 强制文字横向排列，且适配字体大小
                xaxis=dict(tickangle=0, tickfont=dict(size=14, family='Microsoft YaHei'))
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("""
    ### 🔬 技术架构
    - **特征提取**：短时傅里叶变换 (STFT)。
    - **核心骨干**：轻量化 MobileNetV2。
    - **可解释性**：Grad-CAM 可视化异常频率关注点。
    - **工程集成**：基于 Streamlit 的 Web 诊断仪表盘。
    """)