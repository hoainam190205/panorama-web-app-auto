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
  - Canny Edge Detection
  - SIFT hoặc ORB
  - BFMatcher
  - Lowe Ratio Test
  - RANSAC Homography
  - Perspective Warp
  - Feather Blending
  - Smart Auto Crop giảm viền đen
- Hiển thị ảnh trung gian, gồm ảnh resized, grayscale + Gaussian Blur, Canny edge map, keypoints, matches, RANSAC inliers, warp, valid mask, crop mask và panorama trước/sau crop.
- Hiển thị metrics định lượng, gồm thêm phần trăm diện tích đã crop để đánh giá mức giảm viền đen.

## Cài đặt

```cmd
conda activate panorama
cd /d D:\XLA\panorama_web_app_auto
pip install -r requirements.txt
python -m streamlit run app.py
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


## Giảm viền đen sau khi ghép

Bản này đã tối ưu phần crop sau warp để giảm viền đen tốt hơn. Trong thanh bên trái có mục **Giảm viền đen** với 3 chế độ:

- **strict**: tìm hình chữ nhật lớn nhất nằm hoàn toàn trong vùng có ảnh thật. Chế độ này giảm viền đen mạnh nhất, phù hợp để xuất ảnh đẹp nhưng có thể cắt mất nhiều nội dung hơn.
- **soft**: cắt dần các hàng/cột biên có quá nhiều vùng đen. Chế độ này cân bằng giữa giảm viền đen và giữ nội dung.
- **bbox**: chỉ cắt theo bounding box ngoài cùng của vùng có ảnh. Chế độ này giữ nhiều nội dung nhất nhưng có thể còn tam giác đen ở góc.

Gợi ý dùng khi demo: chọn **strict** trước. Nếu ảnh bị cắt quá nhiều thì đổi sang **soft**.
