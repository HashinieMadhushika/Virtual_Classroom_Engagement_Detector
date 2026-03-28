from pathlib import Path
import json

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import hog

st.set_page_config(page_title="Virtual Classroom Engagement Detector", page_icon="🎭", layout="centered")
st.title("🎭 Virtual Classroom Engagement Detector")
st.caption("Day 7 UI - SVM / Random Forest inference using HaarCascade + HOG")

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_emotion_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error(
            "Model artifacts not found. Run notebook Day 7 export cell first to create:\n"
            "- artifacts/best_emotion_model.joblib\n"
            "- artifacts/model_metadata.json"
        )
        st.stop()

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def detect_largest_face(image_bgr, face_cascade):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return image_bgr[y : y + h, x : x + w], (x, y, w, h)


def preprocess_face_for_hog(face_bgr, target_size):
    denoised = cv2.fastNlMeansDenoisingColored(face_bgr, None, 10, 10, 7, 21)
    rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, tuple(target_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return resized, features


def get_engagement_status(predicted_emotion, engagement_mapping):
    for status, emotions in engagement_mapping.items():
        if predicted_emotion in emotions:
            return status
    return "Unknown" # Should not happen if mapping is complete


model, metadata = load_artifacts()
labels = metadata["labels"]
target_size = metadata.get("target_size", [64, 64])
engagement_mapping = metadata.get("engagement_mapping", {})

st.write(f"Loaded model: **{metadata.get('best_model_name', 'Unknown')}**")

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    st.error("Failed to load HaarCascade classifier.")
    st.stop()

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    st.subheader("Original Image")
    st.image(image_rgb, use_column_width=True)

    result = detect_largest_face(image_bgr, face_cascade)
    if result is None:
        st.warning("No face detected. Try another image with a clear frontal face.")
        st.stop()

    face_bgr, (x, y, w, h) = result

    boxed = image_rgb.copy()
    cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)
    st.subheader("Detected Face")
    st.image(boxed, use_column_width=True)

    processed_face, hog_features = preprocess_face_for_hog(face_bgr, target_size)

    pred_idx = int(model.predict([hog_features])[0])
    pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)

    st.subheader("Processed Face (Model Input)")
    st.image(processed_face, caption=f"Resized to {target_size[0]}x{target_size[1]}")

    st.success(f"Predicted Emotion: **{pred_label}**")

    # Determine and display engagement status
    if engagement_mapping:
        engagement_status = get_engagement_status(pred_label, engagement_mapping)
        st.info(f"Engagement Status: **{engagement_status}**")
    else:
        st.warning("Engagement mapping not found in metadata. Cannot determine engagement status.")

st.markdown("---")
st.caption("Tip: Run with `streamlit run app.py`")