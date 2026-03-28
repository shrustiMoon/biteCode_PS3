# 🌊 Microplastic Morphology Classifier

AI-powered computer vision system for detecting and analyzing microplastic particles from microscope images. This tool classifies particle morphology, estimates particle size, and computes an ecological risk score to support marine ecosystem protection and research.

---

## 🚀 Features

### 🔍 Morphology Classification
- Detects and classifies microplastic particles into four main types:
  - **Fiber** – elongated thread-like particles  
  - **Fragment** – irregular broken pieces  
  - **Film** – thin, flat sheet-like particles  
  - **Pellet / Microbead** – round, smooth particles  
- Confidence score (%) is provided for each classification.

### 📏 Size Estimation
- Calculates **Feret diameter** (longest dimension of the particle) in micrometers (µm).  
- Computes **aspect ratio** and **solidity** for additional morphological insights.  

### ⚠️ Ecological Threat Index (ETI)
- Provides an **ecological risk score (0–100)** for each particle based on:
  - Particle morphology (shape)
  - Particle size  
- Categorized into **risk levels**:
  - 🔴 **Critical**
  - 🟠 **High**
  - 🟡 **Moderate**
  - 🟢 **Low**

### 📊 Visualizations
- Display the uploaded image with detected particle overlays.  
- Shows morphological metrics in a **clean, interactive UI**.  
- Optional plots for morphology distribution and risk trends.

---

## 🧠 System Architecture

The system follows a **stepwise pipeline**:

1. **Image Upload**: Users upload microscope images in PNG/JPG format.  
2. **Preprocessing**: Image is converted to grayscale, blurred, and thresholded for clear particle extraction.  
3. **Contour Detection**: Detects all potential microplastic particles in the image.  
4. **Morphology Classification**: Classifies each particle using geometric features:
   - Aspect ratio
   - Solidity
   - Circularity
   - Extent
5. **Size Estimation**: Computes Feret diameter in microns and converts pixel measurements to real-world units.  
6. **Risk Scoring**: Computes ecological risk score using a trained RandomForest model or a rule-based scoring system.  
7. **Visualization & UI**: Displays image, particle metrics, and risk information in a modern Streamlit interface.

---

## 🛠️ Tech Stack

- **Python** – core programming language  
- **OpenCV** – image processing and contour detection  
- **NumPy / Pandas** – numerical and data operations  
- **Streamlit** – interactive web UI for visualization  
- **Matplotlib** – plots and charts for visual analysis  

---


