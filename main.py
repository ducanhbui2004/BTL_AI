import cv2
import mediapipe as mp
import argparse

from utils.feature_extraction import *
from utils.strings import *
from utils.model import ASLClassificationModel
from config import MODEL_NAME, MODEL_CONFIDENCE

import streamlit as st

# Tạm thời bỏ qua cảnh báo
import warnings
warnings.filterwarnings("ignore")

# Khởi tạo MediaPipe Holistic (bao gồm mặt, tay, tư thế)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

if __name__ == "__main__":
    # Khởi động webcam
    cap = cv2.VideoCapture(0)

    # Tạo đối tượng xử lý biểu cảm/dự đoán
    expression_handler = ExpressionHandler()

    # Khởi tạo giao diện Streamlit
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

    # Tạo hai cột: 1 để hiển thị video, 1 để hiển thị kết quả
    col1, col2 = st.columns([4, 2])

    # Vùng hiển thị webcam ở cột 1
    with col1:
        video_placeholder = st.empty()

    # Vùng hiển thị kết quả dự đoán ở cột 2
    with col2:
        prediction_placeholder = st.empty()

    # Tải mô hình đã huấn luyện
    print("Initialising model ...")
    model = ASLClassificationModel.load_model(f"models/{MODEL_NAME}")

    # Khởi tạo MediaPipe Face Mesh để xử lý khuôn mặt
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=MODEL_CONFIDENCE,
                                      min_tracking_confidence=MODEL_CONFIDENCE)

    # Khởi tạo MediaPipe Hands để nhận diện tay
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=MODEL_CONFIDENCE,
                           min_tracking_confidence=MODEL_CONFIDENCE)

    # Cấu hình vẽ các điểm đặc trưng (landmarks)
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Bắt đầu ứng dụng
    print("Starting application")

    # Vòng lặp xử lý từng khung hình từ webcam
    while cap.isOpened():
        # Kiểm tra xem việc lấy khung có thành công không
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Chuyển hình ảnh sang định dạng RGB để xử lý
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Xử lý ảnh và tim kiếm các điểm đặc trưng khuôn mặt
        face_results = face_mesh.process(image)

        # Xử lý anh và tìm kiếm các điểm đặc trưng tay
        hand_results = hands.process(image)

        # Trích xuất đặc trưng từ khuôn mặt và tay
        feature = extract_features(mp_hands, face_results, hand_results)
        expression = model.predict(feature)
        expression_handler.receive(expression)

        # Vẽ mesh khuôn mặt lên ảnh
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

        # Vẽ mesh tay lên ảnh
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

        # Hiển thị ảnh có vẽ landmarks và kết quả dự đoán lên giao diện Streamlit
        video_placeholder.image(image, channels="RGB", use_column_width=True)
        prediction_placeholder.markdown(f'''<h2 class="python scripts/train.py --model_name=sign_modelbig-font">{expression_handler.get_message()}</h2>''', unsafe_allow_html=True)

        # bấm q để thoát
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Giải phóng webcam và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()