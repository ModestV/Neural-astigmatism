"""Операции аугментации: каждая берёт картинку и лист с рамками и возвращает новые (исходные не меняет).

Здесь геометрические операции: меняют и картинку, и координаты рамок.
Рамки: `(x, y, w, h)` в пикселях, начало слева сверху, ось Y вниз.
"""

from __future__ import annotations

from dataclasses import replace

from PIL import Image

from coco_io import Box, Sheet

# Знак изменения атрибута `rotation` при повороте листа против часовой стрелки.
# `rotation` — поворот знака на плане относительно образца в легенде по часовой стрелке (проверено
# глазами на svg_supertest: у розеток с rotation=90 «ножка» смотрит влево, у образца — вниз). Значит,
# при повороте листа против часовой стрелки угол уменьшается.
ROTATION_SIGN_CCW = -1


def _with_boxes(sheet: Sheet, width: int, height: int, boxes: list[Box], step: str) -> Sheet:
    return replace(sheet, width=width, height=height, boxes=boxes, history=[*sheet.history, step])


def _rotate_attr(box: Box, delta: float) -> dict:
    """Копия атрибутов с увеличенным углом; если угла у рамки нет, не выдумываем его."""
    attributes = dict(box.attributes)
    if attributes.get("rotation") is not None:
        attributes["rotation"] = (float(attributes["rotation"]) + delta) % 360
    return attributes


def rotate_90(image: Image.Image, sheet: Sheet, angle: int) -> tuple[Image.Image, Sheet]:
    """Поворачивает лист против часовой стрелки на 90, 180 или 270 градусов."""
    if angle not in (90, 180, 270):
        raise ValueError(f"Поворот только на 90, 180 или 270 градусов, получено {angle}")
    w_img, h_img = sheet.width, sheet.height
    boxes = []
    for box in sheet.boxes:
        x, y, w, h = box.bbox
        if angle == 90:
            bbox = (y, w_img - x - w, h, w)
        elif angle == 180:
            bbox = (w_img - x - w, h_img - y - h, w, h)
        else:
            bbox = (h_img - y - h, x, h, w)
        boxes.append(Box(box.category, bbox, _rotate_attr(box, ROTATION_SIGN_CCW * angle)))
    method = {
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }[angle]
    new_w, new_h = (h_img, w_img) if angle != 180 else (w_img, h_img)
    return image.transpose(method), _with_boxes(sheet, new_w, new_h, boxes, f"rotate_90:{angle}")


def flip(image: Image.Image, sheet: Sheet, axis: str) -> tuple[Image.Image, Sheet]:
    """Зеркальное отражение: `h` — слева направо, `v` — сверху вниз.

    ОСТОРОЖНО: у знаков с направлением зеркало превращает знак в другой, которого нет в легенде
    (см. REVIEW.md). Используется только по явной просьбе (`--mirror`).
    Угол `rotation` пересчитывается: при `h` он становится 180 - угол, при `v` — минус угол.
    Это верно при любом направлении отсчёта угла.
    """
    if axis not in ("h", "v"):
        raise ValueError(f"axis должен быть 'h' или 'v', получено {axis!r}")
    w_img, h_img = sheet.width, sheet.height
    boxes = []
    for box in sheet.boxes:
        x, y, w, h = box.bbox
        attributes = dict(box.attributes)
        angle = attributes.get("rotation")
        if axis == "h":
            bbox = (w_img - x - w, y, w, h)
            new_angle = None if angle is None else (180 - float(angle)) % 360
        else:
            bbox = (x, h_img - y - h, w, h)
            new_angle = None if angle is None else (-float(angle)) % 360
        if new_angle is not None:
            attributes["rotation"] = new_angle
        boxes.append(Box(box.category, bbox, attributes))
    method = Image.Transpose.FLIP_LEFT_RIGHT if axis == "h" else Image.Transpose.FLIP_TOP_BOTTOM
    return image.transpose(method), _with_boxes(sheet, w_img, h_img, boxes, f"flip:{axis}")


def scale(image: Image.Image, sheet: Sheet, factor: float) -> tuple[Image.Image, Sheet]:
    """Меняет размер листа в `factor` раз; рамки умножаются на реальные коэффициенты по осям.

    Размер картинки округляется до целых пикселей, поэтому коэффициенты по X и Y чуть отличаются
    от `factor`. Рамки считаются по ним, чтобы остаться точно на своих местах.
    """
    if factor <= 0:
        raise ValueError("Коэффициент масштаба должен быть положительным")
    new_w = max(1, round(sheet.width * factor))
    new_h = max(1, round(sheet.height * factor))
    sx, sy = new_w / sheet.width, new_h / sheet.height
    boxes = [
        Box(b.category, (b.bbox[0] * sx, b.bbox[1] * sy, b.bbox[2] * sx, b.bbox[3] * sy),
            dict(b.attributes))
        for b in sheet.boxes
    ]
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized, _with_boxes(sheet, new_w, new_h, boxes, f"scale:{factor:g}")
