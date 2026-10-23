"""Поиск по нормализованной корреляции с перебором преобразований."""

from __future__ import annotations

import math
from typing import Any

from algorithms.common import nms
from models import Box, Detection, Settings


def peak_locations(
    response: Any,
    threshold: float,
    width: int,
    height: int,
    limit: int,
) -> tuple[list[tuple[float, int, int]], bool]:
    """Тот же жадный максимум с подавлением; пересчитываем лишь затронутые плитки."""
    import heapq
    import cv2

    tile = 128
    rows, columns = response.shape
    horizontal = (columns + tile - 1) // tile
    vertical = (rows + tile - 1) // tile
    revisions = [0] * (horizontal * vertical)
    heap = []

    def update(index: int) -> None:
        top, left = (index // horizontal) * tile, (index % horizontal) * tile
        _, score, _, (x, y) = cv2.minMaxLoc(response[top : top + tile, left : left + tile])
        if score >= threshold:
            heapq.heappush(heap, (-score, top + y, left + x, index, revisions[index]))

    for index in range(len(revisions)):
        update(index)
    found = []
    while heap and len(found) < limit:
        negative, y, x, index, revision = heapq.heappop(heap)
        if revision != revisions[index]:
            continue
        found.append((-negative, x, y))
        left, right = max(0, x - width // 2), min(columns, x + width // 2 + 1)
        top, bottom = max(0, y - height // 2), min(rows, y + height // 2 + 1)
        response[top:bottom, left:right] = -1
        for ty in range(top // tile, (bottom - 1) // tile + 1):
            for tx in range(left // tile, (right - 1) // tile + 1):
                changed = ty * horizontal + tx
                revisions[changed] += 1
                update(changed)
    return found, len(found) == limit


def detect(
    target: Any, template: Any, template_id: str, symbol: str, settings: Settings
) -> tuple[list[Detection], dict]:
    """Вернуть рамки в пикселях и диагностические сведения, без порога качества."""
    import cv2
    import numpy as np

    if min(template.shape) < 3 or float(template.std()) < 1:
        return [], {"status": "uninformative_template"}
    # Светлый фон не должен доминировать при сравнении линейных символов.
    source_ink = 255 - target
    template_ink = 255 - template
    found = []
    tried = 0
    truncated = False
    for scale in settings.scales:
        for angle in settings.angles:
            height, width = template.shape
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
            out_w = max(1, math.ceil(width * cosine + height * sine - 1e-8))
            out_h = max(1, math.ceil(height * cosine + width * sine - 1e-8))
            if out_w > target.shape[1] or out_h > target.shape[0]:
                continue
            matrix[:, 2] += (np.array([out_w, out_h]) - [width, height]) / 2
            transformed = cv2.warpAffine(template_ink, matrix, (out_w, out_h))
            if float(np.linalg.norm(transformed)) < 1:
                continue
            response = cv2.matchTemplate(source_ink, transformed, cv2.TM_CCORR_NORMED)
            response = np.nan_to_num(response, nan=-1, posinf=-1, neginf=-1)
            tried += 1
            peaks, hit_limit = peak_locations(
                response, settings.template_threshold, out_w, out_h, settings.max_detections
            )
            truncated = truncated or hit_limit
            for score, x, y in peaks:
                found.append(
                    Detection(
                        symbol=symbol,
                        template_id=template_id,
                        bbox=Box(x=x, y=y, w=out_w, h=out_h),
                        score=float(score),
                        angle=angle,
                        scale=scale,
                    )
                )
    return nms(found, settings.nms_iou, settings.max_detections), {
        "status": "ok" if tried else "no_valid_transform",
        "transforms": tried,
        "candidate_limit_reached": truncated,
        "score_definition": "TM_CCORR_NORMED на инвертированном grayscale",
    }
