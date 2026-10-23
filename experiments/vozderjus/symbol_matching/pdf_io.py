"""Чтение PDF, инвентаризация и кэш страниц без изменения исходников."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from models import Box


def pdf_path(root: Path, relative: str) -> Path:
    """Разрешить относительное имя внутри набора, включая кириллицу."""
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Ожидается путь внутри drawings: {relative}")
    if candidate.suffix.lower() != ".pdf" or not candidate.is_file():
        raise ValueError(f"PDF не найден: {candidate}")
    return candidate


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def inspect_pdfs(root: Path) -> list[dict]:
    """Вернуть характеристики страниц и SHA256, не утверждая чистую векторность."""
    import pymupdf

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".pdf")
    if not files:
        raise ValueError(f"В {root} не найдены PDF")
    result = []
    for path in files:
        item = {
            "file": path.relative_to(root).as_posix(),
            "sha256": checksum(path),
            "bytes": path.stat().st_size,
            "pages": [],
        }
        try:
            with pymupdf.open(path) as document:
                if document.needs_pass:
                    raise ValueError("PDF защищён паролем")
                for page in document:
                    paths = len(page.get_drawings())
                    images = len(page.get_images())
                    item["pages"].append(
                        {
                            "page": page.number + 1,
                            "width": page.rect.width,
                            "height": page.rect.height,
                            "rotation": page.rotation,
                            "vector_paths": paths,
                            "image_resources": images,
                            "has_vector_content": paths > 0,
                        }
                    )
        except (ValueError, RuntimeError) as error:
            item["error"] = str(error)
        result.append(item)
    return result


def render_page(
    path: Path,
    page_number: int,
    dpi: int,
    cache_dir: Path,
    max_pixels: int,
) -> tuple[Any, dict]:
    """Растеризовать видимую повёрнутую страницу; ключ включает PDF, DPI и версию."""
    import cv2
    import numpy as np
    import pymupdf

    started = perf_counter()
    digest = checksum(path)
    key = hashlib.sha256(
        f"{digest}:{page_number}:{dpi}:{pymupdf.VersionBind}:gray:v1".encode()
    ).hexdigest()
    cached = cache_dir / f"{key}.png"
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError(f"PDF защищён паролем: {path.name}")
        if not 1 <= page_number <= len(document):
            raise ValueError(f"Нет страницы {page_number} в {path.name}")
        page = document[page_number - 1]
        width, height = page.rect.width, page.rect.height
        zoom = dpi / 72
        pixels = math.ceil(width * zoom) * math.ceil(height * zoom)
        if pixels > max_pixels:
            raise ValueError(
                f"{path.name}: страница требует {pixels:,} пикселей; "
                "уменьшите --dpi или осознанно увеличьте --max-pixels"
            )
        hit = cached.exists()
        image = None
        if hit:
            image = cv2.imdecode(np.fromfile(cached, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            hit = False
            pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
            image = (
                np.frombuffer(pixmap.samples, dtype=np.uint8)
                .reshape(pixmap.height, pixmap.width)
                .copy()
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            ok, data = cv2.imencode(".png", image)
            if not ok:
                raise ValueError("Не удалось сохранить растровую страницу")
            data.tofile(cached)
        # Внешние рамки используют систему отображаемой страницы (page.rect).
        return image, {
            "sha256": digest,
            "page": page_number,
            "dpi": dpi,
            "width": width,
            "height": height,
            "rotation": page.rotation,
            "pixel_width": image.shape[1],
            "pixel_height": image.shape[0],
            "pixels_per_unit": zoom,
            "cache_hit": hit,
            "prepare_seconds": perf_counter() - started,
        }


def validate_box(box: Box, metadata: dict) -> None:
    if box.x + box.w > metadata["width"] + 0.01 or box.y + box.h > metadata["height"] + 0.01:
        raise ValueError(f"Рамка {box.as_list()} выходит за границы страницы")


def crop_template(image: Any, box: Box, metadata: dict) -> Any:
    validate_box(box, metadata)
    zoom = metadata["pixels_per_unit"]
    left, top = math.floor(box.x * zoom), math.floor(box.y * zoom)
    right, bottom = math.ceil((box.x + box.w) * zoom), math.ceil((box.y + box.h) * zoom)
    crop = image[top:bottom, left:right].copy()
    if min(crop.shape) < 3:
        raise ValueError("Образец меньше трёх пикселей: увеличьте рамку или DPI")
    return crop


def save_image(path: Path, image: Any) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Не удалось записать PNG: {path}")
    encoded.tofile(path)
