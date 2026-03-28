from pathlib import Path
import json
import cv2
import joblib
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from skimage.feature import hog

st.set_page_config(page_title="Virtual Classroom Engagement Detector", page_icon="🎭", layout="centered")
st.title("🎭 Virtual Classroom Engagement Detector (Live)")
st.caption("Real-time engagement detection using webcam")

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_emotion_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error("Model artifacts not found. Run training notebook first.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


model, metadata = load_artifacts()
labels = metadata["labels"]
target_size = metadata.get("target_size", [64, 64])
engagement_mapping = metadata.get("engagement_mapping", {})

st.write(f"Loaded model: **{metadata.get('best_model_name', 'Unknown')}**")


# ---------------- FACE DETECTOR ----------------
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)


# ---------------- FUNCTIONS ----------------
def detect_largest_face(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return image_bgr[y:y+h, x:x+w], (x, y, w, h)


def preprocess_face_for_hog(face_bgr):
    denoised = cv2.fastNlMeansDenoisingColored(face_bgr, None, 10, 10, 7, 21)
    rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, tuple(target_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


def get_engagement_status(predicted_emotion):
    for status, emotions in engagement_mapping.items():
        if predicted_emotion in emotions:
            return status
    return "Unknown"


# ---------------- VIDEO PROCESSOR ----------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        result = detect_largest_face(img)

        if result is not None:
            face_bgr, (x, y, w, h) = result

            hog_features = preprocess_face_for_hog(face_bgr)

            pred_idx = int(model.predict([hog_features])[0])
            pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)

            engagement = get_engagement_status(pred_label)

            # Draw rectangle + text
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{engagement}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")


# ---------------- STREAMLIT WEBRTC ----------------
webrtc_streamer(
    key="engagement-detector",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")
st.caption("🎥 Live webcam feed detecting engagement in real-time")