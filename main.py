import cv2
import mediapipe as mp
import argparse
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from utils.feature_extraction import *
from utils.strings import *
from utils.model import ASLClassificationModel
from config import MODEL_NAME, MODEL_CONFIDENCE

# Temporarily ignore warning
import warnings
warnings.filterwarnings("ignore")

# Streamlit UI
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .big-font {
            color: #e76f51 !important;
            font-size: 60px !important;
            border: 0.5rem solid #fcbf49 !important;
            border-radius: 2rem;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Create layout
col1, col2 = st.columns([4, 2])
prediction_placeholder = col2.empty()


# === Custom VideoProcessor class for webcam ===
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.expression_handler = ExpressionHandler()
        self.model = ASLClassificationModel.load_model(f"models/{MODEL_NAME}")

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=MODEL_CONFIDENCE,
            min_tracking_confidence=MODEL_CONFIDENCE
        )

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=MODEL_CONFIDENCE,
            min_tracking_confidence=MODEL_CONFIDENCE
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.message = ""

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MediaPipe processing
        face_results = self.face_mesh.process(image_rgb)
        hand_results = self.hands.process(image_rgb)

        # Feature extraction + predict
        feature = extract_features(self.mp_hands, face_results, hand_results)
        expression = self.model.predict(feature)
        self.expression_handler.receive(expression)
        self.message = self.expression_handler.get_message()

        # Draw face mesh
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )

        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

        return frame.from_ndarray(image, format="bgr24")


# === Stream webcam and display ===
ctx = webrtc_streamer(key="asl-stream", video_processor_factory=ASLProcessor, client_settings={"rtcConfiguration": {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}})

# Display prediction continuously
while True:
    if ctx.video_processor:
        prediction = ctx.video_processor.message
        if prediction:
            prediction_placeholder.markdown(f'''<h2 class="big-font">{prediction}</h2>''', unsafe_allow_html=True)
