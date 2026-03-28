# 🌊 Microplastic Morphology Classifier
AI-powered computer vision system for detecting and analyzing microplastic particles from microscope images. This tool classifies particle morphology, estimates size, and computes an ecological risk score to support marine ecosystem protection.

# 🚀 Features

# 🔍 Morphology Classification
Classifies particles into:
  Fiber
  Fragment
  Film
  Pellet / Microbead
  
# 📏 Size Estimation
Calculates Feret diameter (longest dimension) in micrometers (µm)

# ⚠️ Ecological Threat Index (ETI)
Risk score (0–100) based on:
Particle shape
Particle size

# 🧠 System Architecture
Image → Preprocessing → Contour Detection → Classification → Size Estimation → Risk Score → Visualization → UI

# 🛠️ Tech Stack
1.Python
2.OpenCV
3.NumPy / Pandas
4.Streamlit
5.Matplotlib
