import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import mediapipe as mp
import numpy as np

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load MediaPipe Face Mesh
# ----------------------------
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices (MediaPipe)
LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]

# ----------------------------
# Load models
# ----------------------------
@st.cache_resource
def load_models():
    # Eye model
    eye_model = models.mobilenet_v3_large(weights=None)
    eye_model.classifier[3] = nn.Linear(
        eye_model.classifier[3].in_features, 1
    )
    eye_model.load_state_dict(
        torch.load("models/mobilenetv3_eye_state.pth", map_location=device)
    )
    eye_model.eval().to(device)

    # Mouth model
    mouth_model = models.mobilenet_v3_large(weights=None)
    mouth_model.classifier[3] = nn.Linear(
        mouth_model.classifier[3].in_features, 1
    )
    mouth_model.load_state_dict(
        torch.load("models/mobilenetv3_mouth_state.pth", map_location=device)
    )
    mouth_model.eval().to(device)

    return eye_model, mouth_model

eye_model, mouth_model = load_models()

# ----------------------------
# Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# Eye prediction (MediaPipe)
# ----------------------------
def predict_eye_prob(model, frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return 1.0

    landmarks = result.multi_face_landmarks[0].landmark

    def crop_eye(indices):
        xs = [int(landmarks[i].x * w) for i in indices]
        ys = [int(landmarks[i].y * h) for i in indices]

        x1, x2 = max(min(xs) - 15, 0), min(max(xs) + 15, w)
        y1, y2 = max(min(ys) - 15, 0), min(max(ys) + 15, h)

        return frame[y1:y2, x1:x2]

    left_eye = crop_eye(LEFT_EYE)
    right_eye = crop_eye(RIGHT_EYE)

    probs = []

    for eye in [left_eye, right_eye]:
        if eye.size == 0:
            continue

        img = Image.fromarray(cv2.cvtColor(eye, cv2.COLOR_BGR2RGB))
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(img)).item()
            probs.append(prob)

    if len(probs) == 0:
        return 1.0

    return sum(probs) / len(probs)

# ----------------------------
# Mouth prediction (full face)
# ----------------------------
def predict_mouth_prob(model, frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(img)).item()

    return prob

# ----------------------------
# Fusion logic
# ----------------------------
EYE_THRESHOLD = 0.4
MOUTH_THRESHOLD = 0.5

def fuse_alertness(eye_prob, mouth_prob):
    eye_open = eye_prob > EYE_THRESHOLD
    mouth_yawn = mouth_prob > MOUTH_THRESHOLD

    if eye_open and not mouth_yawn:
        return "ALERT"
    else:
        return "NOT ALERT"

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🚗 Driver Alertness Detection (MediaPipe Eye Cropping)")
st.markdown("**MobileNetV3 + Multi-Cue Fusion (Eye + Mouth)**")

run = st.checkbox("Start Camera")

frame_window = st.image([])
status_text = st.empty()

cap = None

if run:
    cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not accessible")
        break

    eye_prob = predict_eye_prob(eye_model, frame)
    mouth_prob = predict_mouth_prob(mouth_model, frame)

    status = fuse_alertness(eye_prob, mouth_prob)

    frame_window.image(frame, channels="BGR")
    status_text.markdown(
        f"""
        ### 🧠 Status: **{status}**
        - 👁 Eye open probability: `{eye_prob:.2f}`
        - 👄 Yawn probability: `{mouth_prob:.2f}`
        """
    )

if cap:
    cap.release()
