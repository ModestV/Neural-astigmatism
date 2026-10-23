"""ORB/SIFT и последовательный RANSAC для нескольких экземпляров образца."""

from __future__ import annotations

import math
from typing import Any

from algorithms.common import nms
from models import Box, Detection, Settings


def detect(
    target: Any,
    template: Any,
    template_id: str,
    symbol: str,
    settings: Settings,
    method: str,
) -> tuple[list[Detection], dict]:
    """Оценить перенос, поворот и равномерный масштаб по локальным признакам."""
    import cv2
    import numpy as np

    cv2.setRNGSeed(settings.seed)
    if method == "orb":
        extractor = cv2.ORB_create(
            nfeatures=settings.features,
            edgeThreshold=7,
            patchSize=15,
        )
        norm = cv2.NORM_HAMMING
    else:
        extractor = cv2.SIFT_create(nfeatures=settings.features)
        norm = cv2.NORM_L2
    source_kp, source_desc = extractor.detectAndCompute(template, None)
    target_kp, target_desc = extractor.detectAndCompute(target, None)
    info = {
        "template_keypoints": len(source_kp),
        "target_keypoints": len(target_kp),
        "score_definition": "доля RANSAC inliers внутри найденной рамки",
    }
    if source_desc is None or target_desc is None or len(source_kp) < settings.min_inliers:
        return [], {**info, "status": "insufficient_keypoints"}
    # Ищем для каждой точки страницы соответствие в образце, а не наоборот.
    # Так одна точка образца может встречаться в нескольких экземплярах на странице.
    pairs = cv2.BFMatcher(norm).knnMatch(target_desc, source_desc, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < settings.ratio * p[1].distance]
    info["ratio_matches"] = len(good)
    source = np.float32([source_kp[m.trainIdx].pt for m in good])
    destination = np.float32([target_kp[m.queryIdx].pt for m in good])
    remaining = np.arange(len(good))
    found = []
    attempts = 0
    rejected = dict.fromkeys(
        ("unique_points", "spatial_spread", "scale", "angle", "page_bounds", "inlier_ratio"), 0
    )
    height, width = template.shape
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    while len(remaining) >= settings.min_inliers and attempts < settings.max_detections * 3:
        attempts += 1
        src, dst = source[remaining], destination[remaining]
        matrix, mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=settings.ransac_px,
            maxIters=5000,
            confidence=0.995,
            refineIters=10,
        )
        if matrix is None or mask is None:
            break
        inlier_mask = mask.ravel().astype(bool)
        count = int(inlier_mask.sum())
        if count < settings.min_inliers:
            break
        inlier_source = src[inlier_mask]
        # Любая проверенная гипотеза удаляется, чтобы поиск не зацикливался.
        remaining = remaining[~inlier_mask]
        if len(np.unique(inlier_source, axis=0)) < settings.min_inliers:
            rejected["unique_points"] += 1
            continue
        hull_area = cv2.contourArea(cv2.convexHull(inlier_source))
        if hull_area < 0.01 * width * height:
            rejected["spatial_spread"] += 1
            continue
        transformed = cv2.transform(corners[None, :, :], matrix)[0]
        x, y = transformed.min(axis=0)
        right, bottom = transformed.max(axis=0)
        scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
        # Та же конвенция угла, что у getRotationMatrix2D.
        angle = float(math.degrees(math.atan2(matrix[0, 1], matrix[0, 0])) % 360)
        if not any(abs(scale / s - 1) <= settings.scale_tolerance for s in settings.scales):
            rejected["scale"] += 1
            continue
        if not any(
            abs((angle - a + 180) % 360 - 180) <= settings.angle_tolerance for a in settings.angles
        ):
            rejected["angle"] += 1
            continue
        if x < -1 or y < -1 or right > target.shape[1] + 1 or bottom > target.shape[0] + 1:
            rejected["page_bounds"] += 1
            continue
        inside = (dst[:, 0] >= x) & (dst[:, 0] <= right) & (dst[:, 1] >= y) & (dst[:, 1] <= bottom)
        matches_in_box = int(inside.sum())
        score = count / max(count, matches_in_box)
        if score < settings.min_inlier_ratio:
            rejected["inlier_ratio"] += 1
            continue
        found.append(
            Detection(
                symbol=symbol,
                template_id=template_id,
                bbox=Box(
                    x=max(0, float(x)),
                    y=max(0, float(y)),
                    w=float(right - max(0, x)),
                    h=float(bottom - max(0, y)),
                ),
                score=score,
                angle=angle,
                scale=scale,
                inliers=count,
                matches=matches_in_box,
            )
        )
    return nms(found, settings.nms_iou, settings.max_detections), {
        **info,
        "status": "ok" if len(good) >= settings.min_inliers else "insufficient_matches",
        "ransac_attempts": attempts,
        "rejected_hypotheses": rejected,
        "candidate_limit_reached": attempts >= settings.max_detections * 3,
        "model": "estimateAffinePartial2D, similarity, RANSAC",
    }
