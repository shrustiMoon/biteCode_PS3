import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, Optional

import pickle

# Load the saved model
with open('microplastic_model.pkl', 'rb') as f:
    model = pickle.load(f)

@dataclass
class ClassificationResult:
    morphology: str
    confidence: float
    feret_diameter_um: float
    risk_score: float
    risk_level: str
    aspect_ratio: float
    solidity: float
    contour_area: float
    grad_cam_overlay: Optional[np.ndarray] = None


MORPHOLOGY_RISK_BASE = {
    "Fiber": 45,
    "Fragment": 30,
    "Film": 20,
    "Pellet/Microbead": 15,
}


# ================= PREPROCESS =================
def preprocess_image(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    return thresh


# ================= CONTOUR =================
def get_main_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    img_area = thresh.shape[0] * thresh.shape[1]

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if 200 < area < 0.8 * img_area:
            valid.append(c)

    if not valid:
        return None

    return max(valid, key=cv2.contourArea)


# ================= FEATURES =================
def get_features(contour):
    area = cv2.contourArea(contour)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / max(1, min(w, h))

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

    extent = area / (w * h) if w * h > 0 else 0

    return aspect_ratio, solidity, circularity, extent


# ================= CLASSIFICATION =================
def classify_morphology(contour):
    if contour is None:
        return "Fragment", 0.5, 1.0, 0.5

    aspect_ratio, solidity, circularity, extent = get_features(contour)

    # 🔴 HARD RULES (MOST IMPORTANT)
    if aspect_ratio > 4:
        return "Fiber", 0.95, aspect_ratio, solidity

    if circularity > 0.85 and solidity > 0.9:
        return "Pellet/Microbead", 0.95, aspect_ratio, solidity

    if extent > 0.6 and aspect_ratio < 2:
        return "Film", 0.85, aspect_ratio, solidity

    if solidity < 0.8:
        return "Fragment", 0.85, aspect_ratio, solidity

    # 🟡 FALLBACK (rare cases)
    scores = {
        "Fiber": aspect_ratio / 5,
        "Pellet/Microbead": circularity,
        "Film": extent,
        "Fragment": 1 - solidity,
    }

    best = max(scores, key=scores.get)
    return best, 0.6, aspect_ratio, solidity


# ================= SIZE =================
def compute_feret_diameter(contour):
    if contour is None:
        return 0
    rect = cv2.minAreaRect(contour)
    return max(rect[1])


def pixels_to_microns(px, width, fov=2000):
    return (px / width) * fov if width > 0 else 0


# ================= RISK =================
def compute_ecological_risk(morphology, feret_um):
    base = MORPHOLOGY_RISK_BASE.get(morphology, 20)

    if feret_um < 100:
        size = 55
    elif feret_um < 500:
        size = 35
    elif feret_um < 1000:
        size = 20
    else:
        size = 10

    score = min(base + size, 100)

    if score >= 75:
        level = "🔴 CRITICAL"
    elif score >= 55:
        level = "🟠 HIGH"
    elif score >= 35:
        level = "🟡 MODERATE"
    else:
        level = "🟢 LOW"

    return score, level


# ================= MAIN =================
# Map numeric score to risk level
def map_score_to_level(score: float) -> str:
    """Map numeric risk score to risk level string with emojis."""
    if score >= 75:
        return "🔴 CRITICAL"
    elif score >= 55:
        return "🟠 HIGH"
    elif score >= 35:
        return "🟡 MODERATE"
    else:
        return "🟢 LOW"


def classify_image(pil_image: Image.Image, assumed_fov_um=2000):
    img = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    thresh = preprocess_image(img_bgr)
    contour = get_main_contour(thresh)

    morphology, confidence, aspect_ratio, solidity = classify_morphology(contour)

    feret_px = compute_feret_diameter(contour)
    feret_um = pixels_to_microns(feret_px, img.shape[1], assumed_fov_um)

    # Predict using your trained RandomForest model
    risk_score = model.predict([[feret_um]])[0]
    risk_level = map_score_to_level(risk_score)

    area = cv2.contourArea(contour) if contour is not None else 0

    return ClassificationResult(
        morphology,
        confidence,
        round(feret_um, 2),
        risk_score,
        risk_level,
        round(aspect_ratio, 2),
        round(solidity, 2),
        round(area, 1),
        None
    )