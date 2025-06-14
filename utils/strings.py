
class ExpressionHandler:

    MAPPING = {
        "bình_thường": "Ngồi yên (Normal)",
        "cảm_ơn": "Cảm ơn (Thank you)",
        "xin_chào": "Xin chào (Hello)",
        "yêu": "Yêu (Love)",
        "không": "Không (No)",    
        "có": "Có (Yes)",
        "du lịch": "Du lịch (Travel)",
        "ăn": "Ăn (Eat)",
        "đau": "Đau (hurt)",
        "khỏe": "khỏe (fine)",
        "đi_chơi": "Chơi (‍play)",
        "thích": "Thích (like)",
        "nhiều_hơn": "Nhiều hơn (More)",
        "uống": "Uống (Drink)",
        "wrong": "Sai (Wrong)",
        "Nghĩ": "Nghĩ (Think)",
        "nhìn": "Nhìn (Look)",
        "cắt": "Cắt  (Cut)",
        "tỉnhdậy": "Tỉnh dậy (Wake up)",
        "hoàn thành": "Hoàn thành (Complete)",
        'vẽ': "Vẽ (Draw)",
        "sợ": "Sợ (Scared)",
        "nhận": "Nhận (Receive)",
        "khóc": "Khóc (Cry)",
        "mẹ": "Mẹ (Mother)",
        "bố": "Bố (Father)",
        "tôi": "bạn (you)",
        "bạn": "tôi (I)",
        "đúng": "Đúng (right)",
        "khi nào": "Khi nào (When)",
        "nhỏ": "Nhỏ (small)",
        "0": "Số 0 0️⃣",
        "1": "Số 1 1️⃣",
        "2": "Số 2 2️⃣",
        "3": "Số 3 3️⃣",
        "4": "Số 4 4️⃣",
        "5": "Số 5 5️⃣",  
        "6": "Số 6 6️⃣",
        "7": "Số 7 7️⃣",
        "8": "Số 8 8️⃣",
        "9": "Số 9 9️⃣",
        "kết hôn": "Kết hôn (marry)",
        "ly hôn": "Ly hôn (divorce)",
        "meet": "Gặp gỡ (meet)",
        "nói": "Nói chuyện (talk)",
        "tập trung": "Tập trung (concentrate)",
        "nhà": "Nhà (house)",
        "tức_giận": "Tức giận (angry)",
        "hiểu": "Hiểu (understand)",
        "bảo trọng": "Bảo trọng (take care)",

        
    }
    
    def __init__(self):
        # Save the current message and the time received the current message
        self.current_message = ""

    def receive(self, message):
        self.current_message = message

    def get_message(self):
        return ExpressionHandler.MAPPING[self.current_message]
