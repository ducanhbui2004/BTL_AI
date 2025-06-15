import numpy as np
from config import *


def extract_hand_result(mp_hands, hand_results):
    """
    Extract features from hand result
    :param mp_hands from mediapipe
    :param hand_results from media_pipe
    :return: features
    """
    #Nếu không phát hiện được landmark bàn tay nào -> trả về vector toàn số 0
    if hand_results.multi_hand_landmarks is None:
        # Mỗi tay có 21 điểm, mỗi điểm gồm 2 tọa độ (x, y) => 21 x 2
        return np.zeros(FEATURES_PER_HAND * 4)

    # Lấy số lượng tay được phát hiện
    num_hands = len(hand_results.multi_hand_landmarks)
    handedness = hand_results.multi_handedness

    # Xử lý sự thuận tay
    if num_hands == 1:
        # kiểm tra xem có phải là tay phải hay tay trái
        hand_array = extract_single_hand(mp_hands, hand_results.multi_hand_landmarks[0])
        if handedness[0].classification[0].label == "Right":
            return np.hstack((hand_array.flatten(), np.zeros(FEATURES_PER_HAND * 2)))
        else:
            return np.hstack((np.zeros(FEATURES_PER_HAND * 2), hand_array.flatten()))
    else:
        # Nếu có đủ hai tay -> phân biệt trái/phải
        if handedness[0].classification[0].label == "Right":
            left_hand = hand_results.multi_hand_landmarks[0]
            right_hand = hand_results.multi_hand_landmarks[1]
        else:
            left_hand = hand_results.multi_hand_landmarks[1]
            right_hand = hand_results.multi_hand_landmarks[0]

        # Trích xuất đặc trưng từng tay
        left_hand_array = extract_single_hand(mp_hands, left_hand)
        right_hand_array = extract_single_hand(mp_hands, right_hand)

        return np.hstack((left_hand_array, right_hand_array)).flatten()


def extract_single_hand(mp_hands, hand_landmarks):
    # Tạo mảng NumPy 2 chiều để lưu tọa độ các điểm landmark
    landmarks_array = np.zeros((21, 2))

    # Hàm phụ để lấy tọa độ an toàn
    def get_landmark(landmark):
        if landmark is None:
            return np.array([0.0, 0.0])
        return np.array([landmark.x, landmark.y])

    # Lấy lần lượt từng điểm landmark theo thứ tự
    landmarks_array[0] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])
    landmarks_array[1] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC])
    landmarks_array[2] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP])
    landmarks_array[3] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP])
    landmarks_array[4] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP])
    landmarks_array[5] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP])
    landmarks_array[6] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP])
    landmarks_array[7] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP])
    landmarks_array[8] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])
    landmarks_array[9] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP])
    landmarks_array[10] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP])
    landmarks_array[11] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP])
    landmarks_array[12] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP])
    landmarks_array[13] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP])
    landmarks_array[14] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP])
    landmarks_array[15] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP])
    landmarks_array[16] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP])
    landmarks_array[17] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP])
    landmarks_array[18] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP])
    landmarks_array[19] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP])
    landmarks_array[20] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP])

    return landmarks_array


def extract_face_result(face_results):
    """
    Extract features from face result
    :param face_results from mediapipe
    :return: features
    """
    # Lấy khuôn mặt đầu tiên
    single_face = face_results.multi_face_landmarks[0]

    # Lưu toàn bộ điểm landmark khuôn mặt vào mảng 2 chiều
    face_array = np.array([
        [lm.x, lm.y] for lm in single_face.landmark
    ])

    # Trả về trung bình các tọa độ landmark
    return np.mean(face_array, axis=0)


def extract_features(mp_hands, face_results, hand_results):
    """
    Combine the results into one single feature array
    :param mp_hands from mediapipe
    :param face_results from mediapipe
    :param hand_results from mediapipe
    :return: single feature array
    """
    face_features = extract_face_result(face_results)
    hand_features = extract_hand_result(mp_hands, hand_results)
    return np.hstack((face_features, hand_features))