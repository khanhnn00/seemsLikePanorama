Link tải file executable: https://drive.google.com/open?id=1xb0d-_L2gF-eWxi_1G27JzkywcVpcsn-

Input - Chọn thư mục chứa các ảnh cần ghép, chương trình tự động lấy tất cả ảnh bên trong thư mục
Output - Chọn thư mục chứa ảnh đầu ra, cần ghi rõ tên ảnh và định dạng đầu ra mong muốn. Ví dụ: result.png, output.jpg ...

Các lựa chọn đầu tiên trong drop down list là default choice

Descriptor - Chọn descriptor
Match confidence - Ngưỡng khoảng cách tối thiểu so khớp hai feature, giá trị từ 0 -> 1
GPU acceleration - Dùng GPU chạy thuật toán - chỉ dùng được cho card NVIDIA
Straightening - Hướng chỉnh thẳng ảnh
Warp surface - bề mặt của ảnh ghép lại: sphere hình cầu, plane mặt phẳng, cylindric trụ
Blending - Phương pháp blend, strength - mức độ blend, giá trị từ 0 -> 100 (%)

Sau khi chọn hết thông số bấm Run để chạy, có thể bị khựng nếu số ảnh input cao (>10 ảnh)