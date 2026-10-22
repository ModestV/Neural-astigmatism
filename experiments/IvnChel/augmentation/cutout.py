"""Вырезание случайных квадратов (cutout): модель учится узнавать частично закрытый знак.

Что происходит, простыми словами: на картинку ставятся белые квадраты, а потом для каждого `object`
считается, какая доля его рамки закрыта.
- закрыто меньше `max_covered` (по умолчанию треть): рамка остаётся, ей ставится `occluded = True`;
- закрыто больше: рамка убирается, потому что знака на картинке уже не узнать.
Образцы легенды (`legend_item`) никогда не закрываются: квадрат не ставится, если задевает образец.
"""

from __future__ import annotations

import random
from dataclasses import replace

from PIL import Image, ImageDraw

from coco_io import LEGEND_ITEM, OBJECT, Box, Sheet


def _covered_fraction(box: Box, squares: list[tuple[int, int, int, int]]) -> float:
    """Доля площади рамки, закрытая объединением квадратов (квадраты могут перекрываться)."""
    x, y, w, h = box.bbox
    left, top = int(x), int(y)
    right, bottom = int(x + w + 0.999), int(y + h + 0.999)
    mask = Image.new("L", (max(right - left, 1), max(bottom - top, 1)), 0)
    draw = ImageDraw.Draw(mask)
    for sx0, sy0, sx1, sy1 in squares:
        draw.rectangle([sx0 - left, sy0 - top, sx1 - left - 1, sy1 - top - 1], fill=1)
    covered = sum(mask.tobytes())
    return covered / (mask.width * mask.height)


def plan_cutout(
    sheet: Sheet,
    width: int,
    height: int,
    count: int = 5,
    size: tuple[int, int] = (20, 60),
    max_covered: float = 1 / 3,
    on_objects: float = 0.7,
    seed: int = 0,
) -> tuple[list[tuple[int, int, int, int]], Sheet]:
    """Выбирает квадраты `(x0, y0, x1, y1)` в пикселях и пересчитывает рамки; саму картинку не трогает.

    Нужна и растру (`cutout`), и векторному PDF (`pdf_ops`): квадраты рисует тот, кто вызвал.
    Параметры те же, что у `cutout`; `width` и `height` — размер листа в пикселях.
    """
    if count < 0 or not 0 < size[0] <= size[1]:
        raise ValueError("Число квадратов >= 0, размер: 0 < минимум <= максимум")
    if not 0 <= max_covered <= 1 or not 0 <= on_objects <= 1:
        raise ValueError("max_covered и on_objects должны быть от 0 до 1")
    rng = random.Random(seed)
    legends = [b.bbox for b in sheet.boxes if b.category == LEGEND_ITEM]
    objects = [b.bbox for b in sheet.boxes if b.category == OBJECT]
    squares: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        side = rng.randint(*size)
        for _attempt in range(50):
            if objects and rng.random() < on_objects:
                ox, oy, ow, oh = rng.choice(objects)
                cx, cy = ox + rng.uniform(0, ow), oy + rng.uniform(0, oh)
            else:
                cx, cy = rng.uniform(0, width), rng.uniform(0, height)
            x0, y0 = int(cx - side / 2), int(cy - side / 2)
            x0, y0 = min(max(x0, 0), max(width - side, 0)), min(max(y0, 0), max(height - side, 0))
            x1, y1 = min(x0 + side, width), min(y0 + side, height)
            if any(x0 < lx + lw and x1 > lx and y0 < ly + lh and y1 > ly for lx, ly, lw, lh in legends):
                continue
            squares.append((x0, y0, x1, y1))
            break
    boxes = []
    removed = 0
    for box in sheet.boxes:
        attributes = dict(box.attributes)
        if box.category == OBJECT and squares:
            covered = _covered_fraction(box, squares)
            if covered > max_covered:
                removed += 1
                continue
            if covered > 0:
                attributes["occluded"] = True
        boxes.append(Box(box.category, box.bbox, attributes))
    step = f"cutout:{len(squares)},{size[0]}-{size[1]},{seed},removed={removed}"
    return squares, replace(sheet, boxes=boxes, history=[*sheet.history, step])


def cutout(
    image: Image.Image,
    sheet: Sheet,
    count: int = 5,
    size: tuple[int, int] = (20, 60),
    max_covered: float = 1 / 3,
    on_objects: float = 0.7,
    seed: int = 0,
) -> tuple[Image.Image, Sheet]:
    """Ставит до `count` белых квадратов со стороной из `size` px.

    on_objects: вероятность, что квадрат центрируется на случайном `object` (иначе — куда попало),
        иначе на больших листах почти все квадраты попали бы в пустое место.
    На квадрат даётся до 50 попыток найти место, не задевающее образцы; если не нашлось, он пропускается.
    """
    squares, new_sheet = plan_cutout(
        sheet, image.width, image.height, count, size, max_covered, on_objects, seed
    )
    out = image.copy() if image.mode in ("L", "RGB") else image.convert("RGB")
    white = 255 if out.mode == "L" else (255, 255, 255)
    draw = ImageDraw.Draw(out)
    for x0, y0, x1, y1 in squares:
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=white)
    return out, new_sheet
