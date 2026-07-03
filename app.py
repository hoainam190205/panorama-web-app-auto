from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from panorama_core import (
    StitchConfig,
    auto_order_images,
    bgr_to_rgb,
    stitch_many,
    stitch_pair,
)


st.set_page_config(
    page_title="Panorama Image Stitching",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Panorama Image Stitching Web App")
st.caption(
    "Upload ảnh thật có vùng chồng lấn. App chạy SIFT/ORB + Matching + Lowe Ratio + RANSAC + Homography + Warp + Blending để tạo ảnh toàn cảnh."
)

with st.sidebar:
    st.header("Cấu hình thuật toán")
    detector = st.selectbox("Feature detector", ["sift", "orb"], index=0)
    nfeatures = st.slider("Số keypoints tối đa", 500, 8000, 3000, 500)
    lowe_ratio = st.slider("Lowe ratio", 0.50, 0.95, 0.75, 0.05)
    ransac_threshold = st.slider("RANSAC threshold (px)", 1.0, 10.0, 4.0, 0.5)
    max_width = st.slider("Resize max width", 500, 1800, 1100, 100)
    blend = st.selectbox("Blending", ["feather", "none"], index=0)
    max_matches_to_draw = st.slider("Số match hiển thị", 20, 200, 80, 10)

    st.divider()
    st.header("Sắp xếp ảnh")
    order_mode = st.radio(
        "Chế độ thứ tự ảnh",
        ["Tự động sắp xếp trái → phải", "Giữ nguyên thứ tự upload"],
        index=0,
    )
    min_order_inliers = st.slider("Auto-order: inlier tối thiểu", 6, 50, 12, 1)
    min_order_ratio = st.slider("Auto-order: inlier ratio tối thiểu", 0.05, 0.50, 0.15, 0.05)

    st.divider()
    st.markdown("**Gợi ý ảnh upload**")
    st.write("- Có thể upload ngẫu nhiên nếu bật Auto-order.")
    st.write("- Ảnh nên là ảnh thật, cùng một vị trí, xoay camera nhẹ.")
    st.write("- Mỗi cặp ảnh liền nhau nên chồng lấn 30–60%.")
    st.write("- Tránh ảnh toàn trời/nước vì ít đặc trưng.")

cfg = StitchConfig(
    detector=detector,
    nfeatures=nfeatures,
    lowe_ratio=lowe_ratio,
    ransac_threshold=ransac_threshold,
    max_width=max_width,
    blend=blend,
    max_matches_to_draw=max_matches_to_draw,
)

uploaded_files = st.file_uploader(
    "Upload 2 hoặc nhiều ảnh. Có thể upload ngẫu nhiên nếu bật chế độ tự động sắp xếp.",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
)


def read_uploaded_image(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {uploaded_file.name}")
    return img


def image_download_bytes(img_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Không encode được ảnh kết quả.")
    return encoded.tobytes()


def show_uploaded_preview(images: list[np.ndarray], names: list[str]):
    cols = st.columns(min(len(images), 4))
    for i, (img, name) in enumerate(zip(images, names)):
        with cols[i % len(cols)]:
            st.image(bgr_to_rgb(img), caption=name, use_container_width=True)


if uploaded_files:
    st.subheader("1. Ảnh đã upload")
    images: list[np.ndarray] = []
    names: list[str] = []

    for f in uploaded_files:
        try:
            img = read_uploaded_image(f)
            images.append(img)
            names.append(f.name)
        except Exception as e:
            st.error(str(e))

    if images:
        show_uploaded_preview(images, names)

    st.info(
        "Nếu Homography bị méo mạnh, nguyên nhân thường là ảnh không cùng cảnh, overlap quá thấp, hoặc thứ tự ảnh sai. Với nhiều ảnh, bật Auto-order để app tự ước lượng thứ tự trái → phải."
    )

    if len(images) >= 2:
        run = st.button("🚀 Ghép ảnh panorama", type="primary")
        if run:
            try:
                working_images = images
                working_names = names
                order_diag = None

                if order_mode == "Tự động sắp xếp trái → phải" and len(images) >= 2:
                    with st.spinner("Đang tự sắp xếp ảnh bằng pairwise matching + RANSAC..."):
                        order_result = auto_order_images(
                            images,
                            names,
                            cfg,
                            min_inliers=min_order_inliers,
                            min_inlier_ratio=min_order_ratio,
                        )
                    working_images = order_result.ordered_images
                    working_names = order_result.ordered_names
                    order_diag = order_result.diagnostics

                    st.subheader("2. Thứ tự ảnh sau auto-order")
                    order_df = pd.DataFrame(
                        {
                            "Order": list(range(1, len(working_names) + 1)),
                            "Filename": working_names,
                            "Original index": [i + 1 for i in order_result.order_indices],
                        }
                    )
                    st.dataframe(order_df, use_container_width=True)
                    show_uploaded_preview(working_images, working_names)

                    with st.expander("Xem chẩn đoán auto-order"):
                        st.dataframe(pd.DataFrame(order_diag), use_container_width=True)

                else:
                    st.caption("Đang dùng đúng thứ tự upload do bạn chọn.")

                with st.spinner("Đang phát hiện đặc trưng, matching, RANSAC và warp ảnh..."):
                    t0 = time.time()
                    if len(working_images) == 2:
                        result = stitch_pair(working_images[0], working_images[1], cfg)
                    else:
                        result = stitch_many(working_images, cfg)
                    elapsed = time.time() - t0

                st.success(f"Ghép ảnh xong trong {elapsed:.2f} giây")

                st.subheader("3. Kết quả panorama")
                st.image(bgr_to_rgb(result.panorama), caption="Panorama final", use_container_width=True)

                st.download_button(
                    "⬇️ Tải ảnh panorama_final.jpg",
                    data=image_download_bytes(result.panorama),
                    file_name="panorama_final.jpg",
                    mime="image/jpeg",
                )

                st.subheader("4. Metrics định lượng")
                metrics_df = pd.DataFrame([result.metrics]).T.reset_index()
                metrics_df.columns = ["Metric", "Value"]
                st.dataframe(metrics_df, use_container_width=True)

                st.subheader("5. Ảnh trung gian")
                tabs = st.tabs([
                    "Preprocess", "Keypoints", "Matches", "RANSAC Inliers", "Warp", "Raw Panorama"
                ])
                with tabs[0]:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(bgr_to_rgb(result.intermediates["left_resized"]), caption="Left/current resized", use_container_width=True)
                    with c2:
                        st.image(bgr_to_rgb(result.intermediates["right_resized"]), caption="Right/next resized", use_container_width=True)
                with tabs[1]:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(bgr_to_rgb(result.intermediates["keypoints_left"]), caption="Keypoints left/current", use_container_width=True)
                    with c2:
                        st.image(bgr_to_rgb(result.intermediates["keypoints_right"]), caption="Keypoints right/next", use_container_width=True)
                with tabs[2]:
                    st.image(bgr_to_rgb(result.intermediates["good_matches"]), caption="Good matches sau Lowe ratio", use_container_width=True)
                with tabs[3]:
                    st.image(bgr_to_rgb(result.intermediates["inlier_matches"]), caption="Inlier matches sau RANSAC", use_container_width=True)
                with tabs[4]:
                    st.image(bgr_to_rgb(result.intermediates["warped_left"]), caption="Warped current image", use_container_width=True)
                with tabs[5]:
                    st.image(bgr_to_rgb(result.intermediates["panorama_raw"]), caption="Panorama trước auto crop", use_container_width=True)

                st.subheader("6. Tham số đã sử dụng")
                st.code(
                    f"""
Detector: {cfg.detector}
N_FEATURES: {cfg.nfeatures}
Lowe ratio: {cfg.lowe_ratio}
RANSAC threshold: {cfg.ransac_threshold} px
Resize max width: {cfg.max_width} px
Blending: {cfg.blend}
Order mode: {order_mode}
Auto-order min inliers: {min_order_inliers}
Auto-order min inlier ratio: {min_order_ratio}
                    """.strip(),
                    language="text",
                )

            except Exception as e:
                st.error("Không ghép được ảnh.")
                st.exception(e)
                st.warning(
                    "Cách sửa thường dùng: dùng ảnh thật có overlap 30–60%, giảm Lowe ratio xuống 0.65, tăng nfeatures, hoặc đổi SIFT/ORB. Nếu dùng nhiều ảnh và auto-order sai, thử giữ thứ tự upload thủ công."
                )
    else:
        st.warning("Cần upload ít nhất 2 ảnh.")
else:
    st.subheader("Cách sử dụng")
    st.write("1. Upload 2 hoặc nhiều ảnh. Có thể upload ngẫu nhiên khi bật Auto-order.")
    st.write("2. Chỉnh detector, Lowe ratio, RANSAC threshold ở thanh bên trái.")
    st.write("3. Bấm **Ghép ảnh panorama**.")
    st.write("4. Xem ảnh kết quả, thứ tự ảnh, ảnh trung gian và metrics.")

st.divider()
st.markdown(
    """
**Kỹ thuật trong pipeline:** Gaussian Blur, SIFT/ORB Feature Detection, BFMatcher, Lowe Ratio Test, RANSAC Homography, Perspective Warp, Feather Blending, Auto Crop, Pairwise Matching Graph for Auto-order.
"""
)
