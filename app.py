# app.py (Modern Web Layout - UI Only)
import streamlit as st
from PIL import Image
from classifier import classify_image, ClassificationResult
import cv2
import numpy as np

# ---- Page Config ----
st.set_page_config(
    page_title="Microplastic Classifier",
    layout="wide",
    page_icon="🔬"
)

# ---- Header ----
st.markdown(
    """
    <div style="text-align:center; padding:15px; background-color:#f0f8ff; border-radius:10px;">
        <h2 style="color:#1f77b4;">🔬 Microplastic Morphology Classifier</h2>
        <p style="color:#555;">Upload microplastic images and explore morphology & ecological risk.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---- Upload Section ----
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    # ---- Two Column Layout ----
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Original Image")
        st.image(img, width=400)

    with col2:
        st.subheader("⚙️ Classification Settings")
        fov_um = st.number_input("Field of View (µm)", min_value=500, max_value=10000, value=2000, step=100)

        if st.button("Classify"):
            with st.spinner("Analyzing microplastic..."):
                result: ClassificationResult = classify_image(img, assumed_fov_um=fov_um)

            # =================== Metrics Display ===================
            with col2:
                st.markdown("### 🧪 Microplastic Metrics", unsafe_allow_html=True)

                metrics_html = f"""
                <div style='background-color:#f0f4f8; padding:15px; border-radius:10px;'>
                    <p style='color:#1e40af; font-size:16px; margin:5px 0;'><b>Morphology:</b> {result.morphology}</p>
                    <p style='color:#1e40af; font-size:16px; margin:5px 0;'><b>Confidence:</b> {result.confidence * 100:.1f}%</p>
                    <p style='color:#1e40af; font-size:16px; margin:5px 0;'><b>Feret Diameter:</b> {result.feret_diameter_um} µm</p>
                    <p style='color:#1e40af; font-size:16px; margin:5px 0;'><b>Aspect Ratio:</b> {result.aspect_ratio}</p>
                    <p style='color:#1e40af; font-size:16px; margin:5px 0;'><b>Solidity:</b> {result.solidity}</p>
                    <p style='color:#1e40af; font-size:16px; margin:5px 0;'><b>Contour Area:</b> {result.contour_area}</p>
                    <p style='color:#1e40af; font-size:16px; margin:5px 0;'><b>Risk Score:</b> {result.risk_score}</p>
                    <p style='color:#1e40af; font-size:16px; margin:5px 0;'><b>Risk Level:</b> {result.risk_level}</p>
                </div>
                """
                st.markdown(metrics_html, unsafe_allow_html=True)

    # ---- Visualization Tabs ----
    st.subheader("📊 Visualizations")
    tabs = st.tabs(["Contour Overlay", "Metrics Overview", "Risk Summary"])

    with tabs[0]:
        st.write("Contour overlay will appear here (optional).")
        img_np = np.array(img.convert("RGB"))
        thresh = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        vis_img = img_np.copy()
        if contours:
            cv2.drawContours(vis_img, contours, -1, (255,0,0), 2)
        st.image(vis_img, width=400, caption="Contour Overlay")

    with tabs[1]:
        st.write("Metrics Overview:")
        st.write(f"- Morphology: {result.morphology}")
        st.write(f"- Confidence: {result.confidence*100:.1f}%")
        st.write(f"- Feret Diameter: {result.feret_diameter_um} µm")
        st.write(f"- Aspect Ratio: {result.aspect_ratio}")
        st.write(f"- Solidity: {result.solidity}")
        st.write(f"- Contour Area: {result.contour_area}")

    with tabs[2]:
        st.write("Risk Summary:")
        st.write(f"- Risk Level: {result.risk_level} ({result.risk_score})")
        st.progress(min(result.risk_score, 100)/100)

else:
    st.info("Upload an image to start classification.")