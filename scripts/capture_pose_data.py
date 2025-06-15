import cv2
import mediapipe as mp
import argparse
import time
from utils.feature_extraction import *



# Bỏ qua cảnh báo
import warnings
warnings.filterwarnings("ignore")

# Khởi tạo các mô-đun từ MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

if __name__ == "__main__":
    # Nhận tên tư thế từ đối số dòng lệnh
    parser = argparse.ArgumentParser("Pose Data Capture")

    # Thêm các tham số và phân tích chúng
    parser.add_argument("--pose_name", help="Name of the pose to be save in the data folder",
                        type=str, default="test")
    parser.add_argument("--confidence", help="Confidence of the model",
                        type=float, default=0.6)
    parser.add_argument("--duration", help="Duration to capture pose data",
                        type=int, default=60)
    args = parser.parse_args()

    # Khởi động webcam
    cap = cv2.VideoCapture(0)

    # Đếm ngược 5 giây để người dùng sẵn sàng
    print("Get ready!")
    time.sleep(5)
    print("Capturing pose data")

    # Ghi lại thời điểm bắt đầu
    start_time = time.time()

    # Mảng lưu trữ dữ liệu đặc trưng
    pose_data = []

    # IKhởi tạo mô-đun MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=args.confidence,
                                      min_tracking_confidence=args.confidence)

    # IKhởi tạo mô-đun MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=args.confidence,
                           min_tracking_confidence=args.confidence)

    # IKhởi tạo cấu hình vẽ
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Bắt đầu vòng lặp ghi dữ liệu
    while cap.isOpened():
        # CDừng nếu đã đủ thời gian ghi
        if time.time() - start_time >= args.duration:
            print("End capturing")
            break

        # Đọc khung hình từ webcam
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Chuyển ảnh sang định dạng RGB
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Xử lý khuôn mặt
        face_results = face_mesh.process(image)

        # Xử lý bàn tay
        hand_results = hands.process(image)

        # Trích xuất đặc trưng từ khuôn mặt và tay
        feature = extract_features(mp_hands, face_results, hand_results)
        pose_data.append(feature)

        # Vẽ lưới khuôn mặt nếu phát hiện
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

        #  Vẽ bàn tay nếu phát hiện
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

        # Chuyển ảnh trở lại BGR để hiển thị
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Hiển thị hình ảnh đã vẽ (lật ngang như gương)
        cv2.imshow('MediaPipe Face and Hand Detection', cv2.flip(image, 1))

        # P Nhấn phím 'q' để dừng sớm
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # RGiải phóng camera và đóng cửa sổ hiển thị
    cap.release()
    cv2.destroyAllWindows()

    # CChuyển đổi danh sách thành mảng numpy và lưu lại
    pose_data = np.array(pose_data)

    # Save
    np.save(f"data/{args.pose_name}.npy", pose_data)
    print("Save pose data successfully!")
