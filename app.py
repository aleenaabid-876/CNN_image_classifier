import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import time

# Try importing the interpretation module if available locally
try:
    from interpret import GradCAM, overlay_heatmap
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

# ---------------------------------------------------------
# Page Setup & Midnight Blue + Neon Cyan Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Image Classification",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Color Palette Injection: Midnight Blue + Neon Cyan
st.markdown("""
<style>
    /* 1. Background: Midnight Blue (#0B1120) */
    .stApp {
        background-color: #0B1120;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* 2. Hero Header Container (#1E293B) */
    .hero-container {
        padding: 2.5rem 1rem;
        background-color: #1E293B;
        border-radius: 16px;
        border: 1px solid rgba(6, 182, 212, 0.2); /* Neon Cyan subtle border */
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #F8FAFC;
        margin: 0;
        padding: 0;
        background: linear-gradient(90deg, #3B82F6 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 0.6rem;
    }

    /* 3. Stat Cards: #1E293B with Neon Cyan (#06B6D4) Accents */
    .stat-card {
        background-color: #1E293B;
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 14px;
        padding: 1.25rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: #06B6D4; /* Neon Cyan glow on hover */
    }

    .stat-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #06B6D4; /* Neon Cyan text */
        margin-top: 0.3rem;
    }

    /* 4. Top Prediction Banner */
    .top-prediction {
        background: linear-gradient(135deg, #3B82F6 0%, #06B6D4 100%);
        color: #F8FAFC;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 8px 20px rgba(6, 182, 212, 0.25);
        margin-bottom: 1.25rem;
    }

    /* 5. Custom Neon Cyan Progress Bars */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg, #3B82F6, #06B6D4);
    }

    /* 6. COMPLETE DARK FIX FOR FILE UPLOADER & DROPZONE */
    [data-testid="stFileUploader"] {
        background-color: #1E293B !important;
        border: 2px dashed #06B6D4 !important;
        border-radius: 12px;
        padding: 1rem;
    }

    [data-testid="stFileUploaderDropzone"], 
    [data-testid="stFileUploaderDropzone"] > div,
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #0B1120 !important;
        border-color: rgba(6, 182, 212, 0.4) !important;
    }

    [data-testid="stFileUploaderFile"] {
        background-color: #1E293B !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 8px;
    }

    [data-testid="stFileUploader"] *, 
    [data-testid="stFileUploaderDropzone"] * {
        color: #F8FAFC !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #3B82F6 !important;
        border: none !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
    }

    /* Text colors for secondary elements */
    p, span, label {
        color: #F8FAFC;
    }
    
    .stCaption {
        color: #94A3B8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Class Mapping (ImageNet-1k 1000 Standard Categories)
# ---------------------------------------------------------
weights = ResNet50_Weights.DEFAULT
CLASS_NAMES = weights.meta["categories"]

# ---------------------------------------------------------
# Cache Model Loading (Pre-trained ResNet50)
# ---------------------------------------------------------
@st.cache_resource
def load_pytorch_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    return model

model = load_pytorch_model()

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------

# Title Section
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Image Classification & Object Recognition</h1>
    <p class="hero-subtitle">Deep Learning / Computer Vision / Model Interpretability Engine</p>
</div>
""", unsafe_allow_html=True)

# System Stat Cards
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""
<div class="stat-card">
    <div class="stat-label">Neural Engine</div>
    <div class="stat-value">PyTorch</div>
</div>
""", unsafe_allow_html=True)

with col_b:
    st.markdown("""
<div class="stat-card">
    <div class="stat-label">Dataset Benchmark</div>
    <div class="stat-value">ImageNet-1k</div>
</div>
""", unsafe_allow_html=True)

with col_c:
    st.markdown("""
<div class="stat-card">
    <div class="stat-label">Model Architecture</div>
    <div class="stat-value">ResNet50</div>
</div>
""", unsafe_allow_html=True)

st.write("")
st.write("")

# Image Uploader
uploaded_file = st.file_uploader("🖼️ Select or Drop an Image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("🖼️ Input Media")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🎯 Neural Analysis")
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        latency_ms = (time.time() - start_time) * 1000

        top_idx = torch.argmax(probabilities).item()
        top_conf = probabilities[top_idx].item()
        top_class_name = CLASS_NAMES[top_idx].replace('_', ' ').title()

        st.markdown(f'<div class="top-prediction">{top_class_name} — {top_conf*100:.1f}%</div>', unsafe_allow_html=True)
        st.caption(f"⏱️ Inference Latency: {latency_ms:.1f} ms")

        st.write("#### Confidence Distribution")
        top_5_probs, top_5_indices = torch.topk(probabilities, 5)
        
        for prob, idx in zip(top_5_probs, top_5_indices):
            class_name = CLASS_NAMES[idx.item()].replace('_', ' ').title()
            conf_percent = prob.item() * 100
            
            p_col1, p_col2 = st.columns([2, 5])
            with p_col1:
                st.write(f"**{class_name}**")
            with p_col2:
                st.progress(float(prob.item()))
                st.caption(f"{conf_percent:.1f}% Confidence")

    # Grad-CAM Section
    if GRADCAM_AVAILABLE:
        st.divider()
        st.subheader("🧬 Model Interpretability (Grad-CAM)")
        st.write("Visualizing the convolutional feature maps driving the classification decision:")

        try:
            target_layer = model.layer4[-1]
            cam_extractor = GradCAM(model, target_layer)
            
            heatmap = cam_extractor.generate(input_tensor, class_idx=top_idx)
            
            orig_np = np.array(image.resize((224, 224)))
            cam_overlay = overlay_heatmap(orig_np, heatmap)

            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.image(heatmap, caption="Activation Heatmap", use_container_width=True, clamp=True)
            with g_col2:
                st.image(cam_overlay, caption="Neural Attention Overlay (Grad-CAM)", use_container_width=True)
        
        except Exception as e:
            st.error(f"Visualization error: {e}")

else:
    st.write("")
    st.info("Upload an image above to run the vision recognition model.")