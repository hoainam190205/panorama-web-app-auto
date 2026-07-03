from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class StitchConfig:
    detector: str = "sift"          # "sift" or "orb"
    nfeatures: int = 3000
    lowe_ratio: float = 0.75
    ransac_threshold: float = 4.0
    max_width: int = 1200
    blend: str = "feather"          # "none" or "feather"
    max_matches_to_draw: int = 80


@dataclass
class StitchResult:
    panorama: np.ndarray
    metrics: Dict[str, float]
    intermediates: Dict[str, np.ndarray]


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_keep_aspect(img: np.ndarray, max_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img.copy()
    scale = max_width / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def preprocess(img: np.ndarray, max_width: int = 1200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    resized = resize_keep_aspect(img, max_width)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    return resized, gray, blur


def create_detector(detector: str, nfeatures: int):
    detector = detector.lower().strip()
    if detector == "sift":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(nfeatures=nfeatures)
        raise RuntimeError("SIFT is not available. Install opencv-contrib-python or choose ORB.")
    if detector == "orb":
        return cv2.ORB_create(nfeatures=nfeatures)
    raise ValueError("detector must be 'sift' or 'orb'")


def detect_and_compute(gray: np.ndarray, cfg: StitchConfig):
    detector = create_detector(cfg.detector, cfg.nfeatures)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        raise RuntimeError("Không tìm được keypoint/descriptor. Hãy dùng ảnh có nhiều texture hoặc tăng nfeatures.")
    return keypoints, descriptors


def match_descriptors(desc1: np.ndarray, desc2: np.ndarray, cfg: StitchConfig):
    if cfg.detector.lower() == "orb":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    raw_matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < cfg.lowe_ratio * n.distance:
            good_matches.append(m)
    return raw_matches, good_matches


def matched_points(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def reprojection_error(H: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, mask: np.ndarray) -> float:
    if H is None or mask is None or len(pts1) == 0:
        return float("nan")
    mask_bool = mask.ravel().astype(bool)
    if mask_bool.sum() == 0:
        return float("nan")
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T
    proj = H @ pts1_h
    proj = (proj[:2] / proj[2]).T
    err = np.linalg.norm(proj[mask_bool] - pts2[mask_bool], axis=1)
    return float(np.mean(err))


def find_homography_ransac(pts1: np.ndarray, pts2: np.ndarray, cfg: StitchConfig):
    if len(pts1) < 4:
        raise RuntimeError("Cần ít nhất 4 good matches để tính Homography.")
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, cfg.ransac_threshold)
    if H is None or mask is None:
        raise RuntimeError("Không tính được Homography. Ảnh có thể thiếu overlap hoặc match sai quá nhiều.")
    return H, mask


def safe_mask(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return (np.sum(img, axis=2) > 0).astype(np.uint8)
    return (img > 0).astype(np.uint8)


def feather_blend(img1: np.ndarray, mask1: np.ndarray, img2: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    mask1 = mask1.astype(np.uint8)
    mask2 = mask2.astype(np.uint8)
    dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5).astype(np.float32)
    dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5).astype(np.float32)
    denom = dist1 + dist2 + 1e-6
    alpha1 = dist1 / denom
    alpha2 = dist2 / denom

    # Non-overlap should be purely from the existing image.
    only1 = (mask1 == 1) & (mask2 == 0)
    only2 = (mask2 == 1) & (mask1 == 0)
    overlap = (mask1 == 1) & (mask2 == 1)

    out = np.zeros_like(img1, dtype=np.float32)
    out[only1] = img1[only1]
    out[only2] = img2[only2]
    if overlap.any():
        a1 = alpha1[..., None]
        a2 = alpha2[..., None]
        out[overlap] = (img1.astype(np.float32) * a1 + img2.astype(np.float32) * a2)[overlap]
    return np.clip(out, 0, 255).astype(np.uint8)


def auto_crop(img: np.ndarray, margin: int = 0) -> np.ndarray:
    mask = safe_mask(img)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img.shape[1] - 1, x2 + margin)
    y2 = min(img.shape[0] - 1, y2 + margin)
    return img[y1:y2 + 1, x1:x2 + 1].copy()


def warp_left_to_right(left: np.ndarray, right: np.ndarray, H_left_to_right: np.ndarray, blend: str = "feather"):
    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]

    corners_left = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners_right = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners_left = cv2.perspectiveTransform(corners_left, H_left_to_right)
    all_corners = np.concatenate([warped_corners_left, corners_right], axis=0)

    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    tx, ty = -x_min, -y_min
    out_w = int(x_max - x_min)
    out_h = int(y_max - y_min)
    if out_w <= 0 or out_h <= 0 or out_w > 12000 or out_h > 8000:
        raise RuntimeError("Canvas warp bất thường. Homography có thể sai do match/outlier.")

    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    warped_left = cv2.warpPerspective(left, T @ H_left_to_right, (out_w, out_h))

    canvas_right = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas_right[ty:ty + h2, tx:tx + w2] = right

    mask_left = safe_mask(warped_left)
    mask_right = safe_mask(canvas_right)

    if blend == "none":
        pano = warped_left.copy()
        pano[mask_right > 0] = canvas_right[mask_right > 0]
    else:
        pano = feather_blend(warped_left, mask_left, canvas_right, mask_right)

    return pano, warped_left, canvas_right


def stitch_pair(left_bgr: np.ndarray, right_bgr: np.ndarray, cfg: StitchConfig) -> StitchResult:
    left, left_gray, left_blur = preprocess(left_bgr, cfg.max_width)
    right, right_gray, right_blur = preprocess(right_bgr, cfg.max_width)

    kp1, desc1 = detect_and_compute(left_blur, cfg)
    kp2, desc2 = detect_and_compute(right_blur, cfg)

    raw_matches, good_matches = match_descriptors(desc1, desc2, cfg)
    if len(good_matches) < 4:
        raise RuntimeError(
            f"Chỉ có {len(good_matches)} good matches. Cần ảnh có vùng overlap rõ hơn hoặc tăng nfeatures/ratio."
        )

    pts1, pts2 = matched_points(kp1, kp2, good_matches)
    H, inlier_mask = find_homography_ransac(pts1, pts2, cfg)
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / max(len(good_matches), 1)
    mean_err = reprojection_error(H, pts1, pts2, inlier_mask)

    if inlier_count < 10:
        raise RuntimeError(f"Inlier quá ít ({inlier_count}). Homography không đáng tin.")
    if inlier_ratio < 0.15:
        raise RuntimeError(f"Inlier ratio thấp ({inlier_ratio:.2f}). Ảnh có thể không cùng cảnh/overlap kém.")

    panorama_raw, warped_left, canvas_right = warp_left_to_right(left, right, H, cfg.blend)
    panorama = auto_crop(panorama_raw)

    raw_vis = cv2.drawMatchesKnn(
        left, kp1, right, kp2, raw_matches[: cfg.max_matches_to_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    good_vis = cv2.drawMatches(
        left, kp1, right, kp2, good_matches[: cfg.max_matches_to_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    inlier_matches = [m for m, keep in zip(good_matches, inlier_mask.ravel().astype(bool)) if keep]
    inlier_vis = cv2.drawMatches(
        left, kp1, right, kp2, inlier_matches[: cfg.max_matches_to_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    keypoints_left = cv2.drawKeypoints(left, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypoints_right = cv2.drawKeypoints(right, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    metrics = {
        "left_keypoints": float(len(kp1)),
        "right_keypoints": float(len(kp2)),
        "raw_matches": float(len(raw_matches)),
        "good_matches": float(len(good_matches)),
        "inliers": float(inlier_count),
        "inlier_ratio": float(inlier_ratio),
        "mean_reprojection_error": float(mean_err),
        "panorama_width": float(panorama.shape[1]),
        "panorama_height": float(panorama.shape[0]),
    }

    intermediates = {
        "left_resized": left,
        "right_resized": right,
        "left_blur_gray": left_blur,
        "right_blur_gray": right_blur,
        "keypoints_left": keypoints_left,
        "keypoints_right": keypoints_right,
        "raw_matches": raw_vis,
        "good_matches": good_vis,
        "inlier_matches": inlier_vis,
        "warped_left": warped_left,
        "canvas_right": canvas_right,
        "panorama_raw": panorama_raw,
    }
    return StitchResult(panorama=panorama, metrics=metrics, intermediates=intermediates)



@dataclass
class AutoOrderResult:
    ordered_images: List[np.ndarray]
    ordered_names: List[str]
    order_indices: List[int]
    diagnostics: List[Dict[str, float | str]]


def estimate_pair_relation(img_a: np.ndarray, img_b: np.ndarray, cfg: StitchConfig, name_a: str = "A", name_b: str = "B") -> Dict[str, float | str]:
    """Estimate whether image A is left/right of image B using feature matches.

    For a horizontal panorama, if the same physical points appear at smaller x in image B
    than in image A, then B is usually the image to the right of A. Therefore:
      median_dx = median(x_B - x_A)
      median_dx < 0  => A is left of B
      median_dx > 0  => B is left of A
    """
    a, _, a_blur = preprocess(img_a, cfg.max_width)
    b, _, b_blur = preprocess(img_b, cfg.max_width)

    kp_a, desc_a = detect_and_compute(a_blur, cfg)
    kp_b, desc_b = detect_and_compute(b_blur, cfg)
    _, good = match_descriptors(desc_a, desc_b, cfg)
    if len(good) < 4:
        raise RuntimeError("not_enough_good_matches")

    pts_a, pts_b = matched_points(kp_a, kp_b, good)
    H, mask = find_homography_ransac(pts_a, pts_b, cfg)
    mask_bool = mask.ravel().astype(bool)
    inliers = int(mask_bool.sum())
    inlier_ratio = inliers / max(len(good), 1)
    err = reprojection_error(H, pts_a, pts_b, mask)

    if inliers == 0:
        median_dx = 0.0
        median_dy = 0.0
    else:
        diff = pts_b[mask_bool] - pts_a[mask_bool]
        median_dx = float(np.median(diff[:, 0]))
        median_dy = float(np.median(diff[:, 1]))

    # Confidence: more inliers and lower error are better. Use a bounded positive score.
    confidence = float((inliers * max(inlier_ratio, 1e-6)) / (1.0 + (0.0 if np.isnan(err) else err)))

    if median_dx < 0:
        relation = f"{name_a} left of {name_b}"
        left_index = "a"
    else:
        relation = f"{name_b} left of {name_a}"
        left_index = "b"

    return {
        "image_a": name_a,
        "image_b": name_b,
        "good_matches": float(len(good)),
        "inliers": float(inliers),
        "inlier_ratio": float(inlier_ratio),
        "mean_reprojection_error": float(err),
        "median_dx_b_minus_a": float(median_dx),
        "median_dy_b_minus_a": float(median_dy),
        "confidence": float(confidence),
        "relation": relation,
        "left_index": left_index,
    }


def auto_order_images(
    images_bgr: List[np.ndarray],
    names: Optional[List[str]],
    cfg: StitchConfig,
    min_inliers: int = 12,
    min_inlier_ratio: float = 0.15,
    min_abs_dx: float = 10.0,
) -> AutoOrderResult:
    """Automatically order unordered panorama images from left to right.

    The method builds a pairwise match graph. For each pair, SIFT/ORB + Lowe ratio +
    RANSAC estimates reliable correspondences. The median horizontal displacement of
    inlier points decides which image is likely on the left. Images are then sorted by
    an aggregate left-to-right score.

    This works best for horizontal panoramas where the camera rotates left/right and
    adjacent images have overlap. If the graph is weak, the original upload order is used.
    """
    n = len(images_bgr)
    if names is None:
        names = [f"image_{i+1}" for i in range(n)]
    if n < 2:
        return AutoOrderResult(images_bgr, names, list(range(n)), [])

    position_score = np.zeros(n, dtype=np.float64)
    diagnostics: List[Dict[str, float | str]] = []
    accepted_edges = 0

    for i in range(n):
        for j in range(i + 1, n):
            try:
                info = estimate_pair_relation(images_bgr[i], images_bgr[j], cfg, names[i], names[j])
                diagnostics.append(info)

                inliers = int(info["inliers"])
                ratio = float(info["inlier_ratio"])
                dx = float(info["median_dx_b_minus_a"])
                conf = max(float(info["confidence"]), 1.0)
                err = float(info["mean_reprojection_error"])

                # Ignore weak or geometrically ambiguous pairs.
                if inliers < min_inliers or ratio < min_inlier_ratio or abs(dx) < min_abs_dx:
                    continue
                if not np.isnan(err) and err > max(cfg.ransac_threshold * 2.5, 8.0):
                    continue

                accepted_edges += 1
                if dx < 0:
                    # i is left of j: i should receive a smaller position score.
                    position_score[i] -= conf
                    position_score[j] += conf
                else:
                    # j is left of i.
                    position_score[j] -= conf
                    position_score[i] += conf
            except Exception as e:
                diagnostics.append({
                    "image_a": names[i],
                    "image_b": names[j],
                    "error": str(e),
                    "relation": "failed",
                })

    if accepted_edges == 0:
        order = list(range(n))
    else:
        order = sorted(range(n), key=lambda idx: (position_score[idx], names[idx]))

    ordered_images = [images_bgr[i] for i in order]
    ordered_names = [names[i] for i in order]

    # Add score summary rows for UI readability.
    for idx in range(n):
        diagnostics.append({
            "image": names[idx],
            "position_score": float(position_score[idx]),
            "type": "order_score",
        })

    return AutoOrderResult(ordered_images, ordered_names, order, diagnostics)


def stitch_many(images_bgr: List[np.ndarray], cfg: StitchConfig) -> StitchResult:
    if len(images_bgr) < 2:
        raise ValueError("Cần ít nhất 2 ảnh để ghép panorama.")
    current = images_bgr[0]
    all_metrics = []
    last_intermediates = {}
    for idx in range(1, len(images_bgr)):
        result = stitch_pair(current, images_bgr[idx], cfg)
        all_metrics.append({f"pair_{idx}": result.metrics})
        current = result.panorama
        last_intermediates = result.intermediates
    metrics = {"num_images": float(len(images_bgr)), "num_pairwise_steps": float(len(images_bgr) - 1)}
    if all_metrics:
        # expose last pair metrics for display
        metrics.update(result.metrics)
    return StitchResult(panorama=current, metrics=metrics, intermediates=last_intermediates)
