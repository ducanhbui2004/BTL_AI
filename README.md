Giới thiệu

Dự án này sử dụng thư viện MediaPipe của Google để thực hiện định vị tư thế qua camera.
Sau đó, chương trình xây dựng một mô hình phân loại đơn giản nhằm nhận diện các động tác ngôn ngữ ký hiệu cơ bản.

Cách sử dụng mã nguồn

Lưu ý: Bạn có thể chạy dự án ngay với dữ liệu và mô hình đã được huấn luyện sẵn. Tuy nhiên, bạn cũng có thể thu thập dữ liệu và huấn luyện mô hình của riêng mình theo các bước sau:

Tải về dự án (clone dự án về máy).

Cài đặt các thư viện cần thiết trong tệp requirements.txt.

Thiết lập đúng đường dẫn thư mục dự án trên máy bạn.

(TÙY CHỌN) Thu thập dữ liệu cho một ký hiệu ngôn ngữ ký hiệu:

python scripts/capture_pose_data.py --pose_name="[TÊN KÝ HIỆU]" --confidence=[ĐỘ TIN CậY (VD: 0.5)]

