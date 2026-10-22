"""Операции аугментации, которые меняют только вид картинки: рамки остаются на тех же местах.

Каждая операция берёт картинку и лист с рамками и возвращает новые (исходные не меняет).
Случайные операции принимают `seed`, поэтому результат можно повторить.
Работают только с режимами «L» (серый) и «RGB»; остальные картинки приводятся к RGB.
"""

from __future__ import annotations

import io
import random
from dataclasses import replace

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from coco_io import Box, Sheet


def _prepare(image: Image.Image) -> Image.Image:
    return image if image.mode in ("L", "RGB") else image.convert("RGB")


def _same_boxes(sheet: Sheet, step: str) -> Sheet:
    """Тот же лист с копиями рамок и записью операции в истории."""
    boxes = [Box(b.category, b.bbox, dict(b.attributes)) for b in sheet.boxes]
    return replace(sheet, boxes=boxes, history=[*sheet.history, step])


def brightness_contrast(
    image: Image.Image, sheet: Sheet, brightness: float = 1.0, contrast: float = 1.0
) -> tuple[Image.Image, Sheet]:
    """Яркость и контраст: 1.0 — без изменений, больше — ярче/контрастнее, меньше — темнее/мягче."""
    if brightness <= 0 or contrast <= 0:
        raise ValueError("Яркость и контраст должны быть положительными")
    out = ImageEnhance.Brightness(_prepare(image)).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    return out, _same_boxes(sheet, f"brightness_contrast:{brightness:g},{contrast:g}")


def add_noise(
    image: Image.Image, sheet: Sheet, amplitude: float = 20.0, seed: int = 0
) -> tuple[Image.Image, Sheet]:
    """Зернистый шум (равномерный, до ±amplitude уровней яркости из 255), как у плохого скана."""
    if amplitude < 0:
        raise ValueError("Сила шума не может быть отрицательной")
    image = _prepare(image)
    raw = random.Random(seed).randbytes(image.width * image.height)
    noise = Image.frombytes("L", image.size, raw)
    noise = noise.point([round(128 + (v - 127.5) * amplitude / 127.5) for v in range(256)])
    if image.mode == "RGB":
        noise = noise.convert("RGB")
    out = ImageChops.add(image, noise, scale=1.0, offset=-128)
    return out, _same_boxes(sheet, f"add_noise:{amplitude:g},{seed}")


def blur(image: Image.Image, sheet: Sheet, radius: float = 1.0) -> tuple[Image.Image, Sheet]:
    """Гауссово размытие; radius — в пикселях. Сильное размытие стирает мелкие знаки."""
    if radius < 0:
        raise ValueError("Радиус размытия не может быть отрицательным")
    out = _prepare(image).filter(ImageFilter.GaussianBlur(radius))
    return out, _same_boxes(sheet, f"blur:{radius:g}")


def jpeg_compress(
    image: Image.Image, sheet: Sheet, quality: int = 40
) -> tuple[Image.Image, Sheet]:
    """Сжимает картинку в JPEG и читает обратно, чтобы вокруг линий появились артефакты."""
    if not 1 <= quality <= 95:
        raise ValueError("Качество JPEG должно быть от 1 до 95")
    image = _prepare(image)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    out = Image.open(buffer)
    out.load()
    return out, _same_boxes(sheet, f"jpeg_compress:{quality}")


def line_thickness(image: Image.Image, sheet: Sheet, delta: int = 1) -> tuple[Image.Image, Sheet]:
    """Делает тёмные линии на светлом фоне толще (delta > 0) или тоньше (delta < 0) на |delta| px.

    Линия утолщается примерно на 2 * |delta| пикселей по ширине. Для светлых линий на тёмном фоне
    эффект обратный.
    """
    if delta == 0:
        return _prepare(image).copy(), _same_boxes(sheet, "line_thickness:0")
    size = 2 * abs(delta) + 1
    rank = ImageFilter.MinFilter(size) if delta > 0 else ImageFilter.MaxFilter(size)
    return _prepare(image).filter(rank), _same_boxes(sheet, f"line_thickness:{delta}")


def invert(image: Image.Image, sheet: Sheet) -> tuple[Image.Image, Sheet]:
    """Инверсия цветов: светлые линии на тёмном фоне."""
    return ImageOps.invert(_prepare(image)), _same_boxes(sheet, "invert")


def threshold(image: Image.Image, sheet: Sheet, level: int = 200) -> tuple[Image.Image, Sheet]:
    """Чёрно-белый порог: всё темнее `level` становится чёрным, остальное белым (как ксерокопия)."""
    if not 0 <= level <= 255:
        raise ValueError("Порог должен быть от 0 до 255")
    image = _prepare(image)
    gray = image.convert("L").point([0 if v < level else 255 for v in range(256)])
    out = gray if image.mode == "L" else gray.convert("RGB")
    return out, _same_boxes(sheet, f"threshold:{level}")


def stains(
    image: Image.Image,
    sheet: Sheet,
    count: int = 20,
    radius: tuple[int, int] = (3, 12),
    seed: int = 0,
) -> tuple[Image.Image, Sheet]:
    """Рисует тёмные пятна. Пятна не заходят на рамки, чтобы не стереть и не исказить знаки.

    Для каждого пятна делается до 50 попыток найти место без пересечения с рамками (с запасом
    в 2 px); если не нашлось, пятно пропускается, поэтому на густых листах их может быть меньше.
    """
    if count < 0 or not 0 < radius[0] <= radius[1]:
        raise ValueError("Число пятен >= 0, радиус: 0 < минимум <= максимум")
    rng = random.Random(seed)
    out = _prepare(image).copy()
    draw = ImageDraw.Draw(out)
    margin = 2
    for _ in range(count):
        r = rng.randint(*radius)
        for _attempt in range(50):
            cx = rng.randint(0, out.width - 1)
            cy = rng.randint(0, out.height - 1)
            left, top, right, bottom = cx - r - margin, cy - r - margin, cx + r + margin, cy + r + margin
            if any(
                left < b.bbox[0] + b.bbox[2] and right > b.bbox[0]
                and top < b.bbox[1] + b.bbox[3] and bottom > b.bbox[1]
                for b in sheet.boxes
            ):
                continue
            gray = rng.randint(30, 100)
            color = gray if out.mode == "L" else (gray, gray, gray)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
            break
    return out, _same_boxes(sheet, f"stains:{count},{radius[0]}-{radius[1]},{seed}")
