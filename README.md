# Panorama Image Stitching Web App

Web app chạy trên trình duyệt bằng Streamlit để ghép ảnh toàn cảnh.

## Tính năng

- Upload 2 hoặc nhiều ảnh.
- Preview ảnh sau khi upload.
- Có 2 chế độ thứ tự ảnh:
  - **Tự động sắp xếp trái → phải**: dùng pairwise feature matching + RANSAC để ước lượng thứ tự ảnh.
  - **Giữ nguyên thứ tự upload**: dùng khi bạn đã đặt ảnh theo thứ tự thủ công.
- Ghép ảnh bằng pipeline Computer Vision cổ điển:
  - Gaussian Blur
  - SIFT hoặc ORB
  - BFMatcher
  - Lowe Ratio Test
  - RANSAC Homography
  - Perspective Warp
  - Feather Blending
  - Auto Crop
- Hiển thị ảnh trung gian và metrics.

## Cài đặt

```cmd
conda activate panorama
cd /d D:\XLA\panorama_web_app_auto
pip install -r requirements.txt
streamlit run app.py
```

Sau đó mở trình duyệt ở địa chỉ:

```text
http://localhost:8501
```

## Cách upload ảnh

Bạn có thể upload ảnh ngẫu nhiên nếu bật chế độ **Tự động sắp xếp trái → phải**.

Tuy nhiên ảnh vẫn cần thỏa điều kiện:

- Ảnh thật, cùng một cảnh.
- Chụp từ cùng một vị trí, chỉ xoay camera nhẹ.
- Mỗi cặp ảnh liền nhau nên có 30–60% vùng chồng lấn.
- Tránh ảnh quá nhiều trời, nước, tường trơn vì ít đặc trưng.

Nếu auto-order sai, chuyển sang chế độ **Giữ nguyên thứ tự upload** và upload theo thứ tự trái → phải.
