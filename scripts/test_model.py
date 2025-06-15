import cv2
import mediapipe as mp
import argparse

from utils.features import *
from utils.model import ASLClassificationModel

# Tạm thời bỏ qua cảnh báo
import warnings
warnings.filterwarnings("ignore")

# Khởi tạo mô hình Holistic từ MediaPipe (gồm mặt, tay, dáng người)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

if __name__ == "__main__":
    # Lấy tên cử chỉ từ đối số dòng lệnh
    parser = argparse.ArgumentParser("Pose Data Capture")

    # Thêm và phân tích các đối số
    parser.add_argument("--model_path", help="Path of the ASL classification model",
                        type=str, default="models/simple_5_expression_model.pkl")
    parser.add_argument("--confidence", help="Confidence of the model",
                        type=float, default=0.6)
    args = parser.parse_args()

    # Khởi tạo webcam
    cap = cv2.VideoCapture(0)

    # Tải mô hình 
    print("Initialising model ...")
    model = ASLClassificationModel.load_model(args.model_path)

    # Khởi tạo Face Mesh từ MediaPipe (nhận diện khuôn mặt)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=args.confidence,
                                      min_tracking_confidence=args.confidence)

    # Khởi tạo mô-đun phát hiện bàn tay
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=args.confidence,
                           min_tracking_confidence=args.confidence)

    # Khởi tạo thông số vẽ
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Bắt đầu ứng dụng
    print("Starting application")

    # Vòng lặp chính đọc từ webcam
    while cap.isOpened():
        # Kiểm tra xem có đọc được khung hình hay không
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Chuyển đổi ảnh sang RGB để xử lý bằng MediaPipe
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Xử lý ảnh và tim kiếm các điểm đặc trưng khuôn mặt
        face_results = face_mesh.process(image)

        # Xử lý ảnh và tim kiếm các điểm đặc trưng tay
        hand_results = hands.process(image)

        # Trích xuất đặc trưng từ khuôn mặt và bàn tay
        feature = extract_features(mp_hands, face_results, hand_results)
        expression = model.predict(feature)

        # Chuẩn bị văn bản để hiển thị
        text = expression

        # Lấy kích thước khung hình
        height, width, _ = image.shape

        # Thiết lập thông số hiển thị chữ
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        font_color = (255, 255, 255)  # White color
        font_thickness = 2

        # Tính toán kích thước dòng chữ
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        # Tính toán vị trí hiển thị ở góc trên bên phải
        padding = 10
        text_x = width - text_size[0] - padding
        text_y = text_size[1] + padding

        # Vẽ khung đen phía sau chữ để dễ nhìn
        cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5),
                      (width - 5, text_y + 5), (0, 0, 0), -1)

        # Vẽ dòng chữ lên khung hình
        cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

        # Vẽ lưới khuôn mặt (face mesh) nếu phát hiện được khuôn mặt
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

        # Vẽ các điểm nối bàn tay nếu phát hiện được tay
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

        # Chuyển ảnh trở lại sang BGR để hiển thị với OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Hiển thị hình ảnh lên màn hình (lật gương để dễ nhìn)
        cv2.imshow('ASL Classification Model', cv2.flip(image, 1))

        # Nhấn phím 'q' để thoát chương trình
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Giải phóng camera và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()
